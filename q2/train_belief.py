"""Train the Q2 belief net on a dataset made by q2/dataset.py.

Outputs:
  /home/sup/data/perchpilot/runs/q2_belief_<tag>/{best.pt, belief.ts, belief.json, train_log.json}
  perchpilot/results/q2_belief_<tag>.json   (C3-style; proportions as Wilson {p, lo, hi})
Run: python q2/train_belief.py --data /home/sup/data/perchpilot/q2data/<ds> --tag <tag> [--drop fnormal] [--device cuda]
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from q2.belief_net import (CHANNEL_GROUPS, BeliefNet, auroc, brier, ece, fit_temperature, loss_fn,  # noqa: E402
                           reliability, wilson)
from sim.contact_phase import SENSOR_CHANNELS, SLIP_CHANNELS  # noqa: E402


def load_split(d, device, max_n=None):
    """Load float16 shards straight into one preallocated tensor on `device` (peak host RAM = one shard)."""
    xs = sorted(Path(d).glob("X_*.npy"))
    shapes = [np.load(x, mmap_mode="r").shape for x in xs]
    n = sum(s[0] for s in shapes)
    if max_n:
        n = min(n, max_n)
    X = torch.empty((n,) + tuple(shapes[0][1:]), dtype=torch.float16, device=device)
    lab = {}
    i = 0
    for x in xs:
        if i >= n:
            break
        a = np.load(x)
        y = np.load(str(x).replace("X_", "Y_").replace(".npy", ".npz"))
        m = min(len(a), n - i)
        X[i:i + m] = torch.from_numpy(a[:m]).to(device)
        for k in y.files:
            lab.setdefault(k, []).append(y[k][:m])
        i += m
        del a
    lab = {k: np.concatenate(v) for k, v in lab.items()}
    return X, lab


@torch.no_grad()
def predict(model, X, bs=1024):
    out = []
    for i in range(0, len(X), bs):
        out.append(model.logits(X[i:i + bs]).float())
    return torch.cat(out)


def metrics(logit, lab, temp, dev):
    y = torch.from_numpy(lab["held"]).bool()
    ys = torch.from_numpy(lab["slip"]).bool()
    p_raw = torch.sigmoid(logit[:, 0]).cpu()
    p = torch.sigmoid(logit[:, 0] / temp).cpu()
    ps = torch.sigmoid(logit[:, 1]).cpu()
    mt = torch.from_numpy(lab["margin_min_g"]).clamp(-2, 2)
    out = {"n": len(y), "held_star": wilson(int(y.sum()), len(y)), "slip_now": wilson(int(ys.sum()), len(y)),
           "auroc_hold": auroc(p, y), "ece15_hold": ece(p, y), "ece15_hold_uncalibrated": ece(p_raw, y),
           "brier_hold": brier(p, y), "acc_hold_at_0.5": wilson(int(((p >= 0.5) == y).sum()), len(y)),
           "auroc_slip": auroc(ps, ys), "brier_slip": brier(ps, ys),
           "margin_mae_g": float((logit[:, 2].cpu() - mt).abs().mean()),
           # shortcut check: how well does the TRUE 'not slipping now' bit alone rank the hold label?
           "auroc_hold_from_true_not_slipping": auroc(1.0 - ys.float(), y),
           "reliability_hold": reliability(p, y), "by_decision_step": {}}
    k, fx = lab["k"], lab["fixed"].astype(bool)
    for kk in np.unique(k[fx]):
        m = torch.from_numpy(fx & (k == kk))
        out["by_decision_step"][f"t{kk * 0.005:.2f}s"] = {
            "n": int(m.sum()), "held_star": wilson(int(y[m].sum()), int(m.sum())),
            "auroc_hold": auroc(p[m], y[m]), "ece15_hold": ece(p[m], y[m]), "brier_hold": brier(p[m], y[m]),
            "auroc_slip": auroc(ps[m], ys[m]), "auroc_hold_from_true_not_slipping": auroc(1.0 - ys[m].float(), y[m])}
    return out


def r4(d):
    if isinstance(d, dict):
        return {k: r4(v) for k, v in d.items()}
    if isinstance(d, list):
        return [r4(v) for v in d]
    if isinstance(d, float):
        return round(d, 4)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--drop", default="", help="channel groups to remove: " + ",".join(CHANNEL_GROUPS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default="held", choices=["held", "strict", "nofall"],
                    help="p_hold target: held = held_star v2 (on the plate 10 s, slip < 2 cm); "
                         "strict = margin >= 0 at every step (v1 label); nofall = no fall within 10 s")
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out_root", default="/home/sup/data/perchpilot/runs")
    ap.add_argument("--results_dir", default=str(ROOT / "results"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    name = f"q2_belief_{args.tag}"
    out = Path(args.out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    data = Path(args.data)
    manifest = (data / "meta.json").read_bytes() if (data / "meta.json").exists() else b"{}"
    ds_meta = json.loads(manifest)
    v4 = ds_meta.get("protocol") == "v4"
    t0 = time.time()
    Xtr, Ltr = load_split(data / "train", dev, args.max_train)
    Xva, Lva = load_split(data / "val", dev)
    evaluation = {"val": (Xva, Lva)}
    if not v4:
        evaluation["test_unseen_materials"] = load_split(data / "test", dev)
    for L in (Ltr, *(labels for _, labels in evaluation.values())):
        L["held_spec"] = L["held"].copy()
        if args.label == "nofall":
            L["held"] = ~L["fell_if_off"].astype(bool)
        elif args.label == "strict":
            L["held"] = L["held_strict"].astype(bool)
    yt = {k: torch.from_numpy(Ltr[k].astype(np.float32)).to(dev) for k in ("held", "slip", "margin_min_g")}
    yv = {k: torch.from_numpy(Lva[k].astype(np.float32)).to(dev) for k in ("held", "slip", "margin_min_g")}
    sub = Xtr[:min(len(Xtr), 5000)].float().reshape(-1, Xtr.shape[-1])
    mean, std = sub.mean(0), sub.std(0)
    del sub
    drop = [g for g in args.drop.split(",") if g]
    model = BeliefNet(mean, std, drop=drop).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(Xtr) + args.bs - 1) // args.bs)
    sched = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
             if steps > 10 else None)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    best, log = {"loss": float("inf"), "epoch": -1}, []
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g).to(dev)
        tl, nb = 0.0, 0
        for i in range(0, len(Xtr), args.bs):
            b = perm[i:i + args.bs]
            loss, _ = loss_fn(model.logits(Xtr[b]), yt["held"][b], yt["slip"][b], yt["margin_min_g"][b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            tl += loss.item()
            nb += 1
        model.eval()
        lv = predict(model, Xva)
        vl, parts = loss_fn(lv, yv["held"], yv["slip"], yv["margin_min_g"])
        va = auroc(torch.sigmoid(lv[:, 0]), yv["held"])
        row = {"epoch": ep, "train_loss": round(tl / nb, 4), "val_loss": round(vl.item(), 4),
               "val_bce_hold": round(parts[0], 4), "val_auroc_hold": round(va, 4), "wall_s": round(time.time() - t0, 1)}
        log.append(row)
        print(json.dumps(row), flush=True)
        if vl.item() < best["loss"]:
            best = {"loss": vl.item(), "epoch": ep}
            torch.save(model.state_dict(), out / "best.pt")
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    model.eval()
    lv = predict(model, Xva)
    T = fit_temperature(lv[:, 0], yv["held"])
    model.temp.fill_(T)
    torch.save(model.state_dict(), out / "best.pt")
    ev = {"temperature": T, "val": metrics(lv, Lva, T, dev)}
    if not v4:
        Xte, Lte = evaluation["test_unseen_materials"]
        ev["test_unseen_materials"] = metrics(predict(model, Xte), Lte, T, dev)
    # TorchScript export (raw window in, [p_hold, p_slip, margin_min_g] out), CPU
    m_cpu = BeliefNet(mean.cpu(), std.cpu(), drop=drop)
    m_cpu.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    m_cpu.eval()
    ts = torch.jit.trace(m_cpu, torch.zeros(1, *Xtr.shape[1:]))
    ts.save(str(out / "belief.ts"))
    with torch.no_grad():
        chk = (ts(Xva[:64].float().cpu()) - m_cpu(Xva[:64].float().cpu())).abs().max().item()
    meta = {"input": f"raw sensor window [B, {Xtr.shape[1]}, {Xtr.shape[2]}], 200 Hz, channels per docs/CONTRACTS.md C5",
            "channels": (SENSOR_CHANNELS + SLIP_CHANNELS)[:Xtr.shape[-1]],
            "output": ["p_hold (temperature-scaled)", "p_slip", "margin_min_g"], "temperature": T,
            "drop": drop, "label": args.label, "dataset": str(data.resolve()),
            "dataset_sha256": hashlib.sha256(manifest).hexdigest(), "protocol": ds_meta.get("protocol", "legacy"),
            "git": git, "seed": args.seed, "ts_check_maxabs": chk}
    (out / "belief.json").write_text(json.dumps(meta, indent=1))
    (out / "train_log.json").write_text(json.dumps(log))
    res = {"name": name, "phase": "P3", "git": git, "seed": args.seed,
           "config": {**vars(args), "dataset_git": ds_meta.get("git"),
                      "dataset_sizes": {s: v.get("windows") for s, v in ds_meta.get("splits", {}).items()
                                        if not v4 or s in ("train", "val")}},
           "eval": r4(ev),
           "train": {"epochs": args.epochs, "best_epoch": best["epoch"], "n_train": len(Xtr), "n_val": len(Xva),
                     "n_test": 0 if v4 else len(Xte), "wall_min": round((time.time() - t0) / 60, 2), "device": str(dev),
                     "run_dir": str(out)}}
    (out / "result.json").write_text(json.dumps(res, indent=1))
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{name}.json").write_text(json.dumps(res, indent=1))
    s = lambda e: {k: e[k] for k in ("auroc_hold", "ece15_hold", "brier_hold", "auroc_slip")}
    print(json.dumps({"name": name, "T": round(T, 3), **{k: r4(s(v)) for k, v in ev.items() if k != "temperature"}}))


if __name__ == "__main__":
    main()

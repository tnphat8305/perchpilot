"""Sensor features for B4d (logistic) and B4e (optical displacement thresholds)."""
import numpy as np
import torch
import torch.nn.functional as F


def slip_distance(w):
    """Norm of the cumulative optical displacement (mm) over a C5 v2 window."""
    return w[:, :, 11:13].sum(1).norm(dim=1)


def features(w, dt, use_slip=None):
    if use_slip is None:
        use_slip = w.shape[-1] == 13
    # Component-wise high-pass energies preserve differences between the body axes.
    hp = torch.diff(w[:, :, :6], dim=1).square().mean(1).log1p()
    fn = w[:, :, 6]
    t = (torch.arange(w.shape[1], device=w.device) - (w.shape[1] - 1) / 2) * dt
    slope = (fn * t).sum(1) / t.square().sum()
    # Double-integrate tangential acceleration after removing a constant gravity/bias
    # estimate. Start at zero relative velocity; this is a sensor proxy, in metres.
    a = w[:, :, :2]
    dp = ((a - a.mean(1, keepdim=True)).cumsum(1).sum(1) * dt**2).norm(dim=1)
    components = [hp, fn.mean(1, keepdim=True), slope[:, None], dp[:, None]]
    if use_slip:
        if w.shape[-1] != 13:
            raise ValueError("B4d with optical slip requires a C5 v2 window")
        components.append(slip_distance(w)[:, None])
    return torch.cat(components, 1)


@torch.no_grad()
def load_features(path, dt, device, use_slip):
    x, y = [], []
    for shard in sorted(path.glob("X_*.npy")):
        w = torch.from_numpy(np.load(shard)).to(device, torch.float32)
        x.append(features(w, dt, use_slip))
        with np.load(shard.with_name(shard.name.replace("X_", "Y_")).with_suffix(".npz")) as labels:
            y.append(torch.from_numpy(labels["held"]).to(device, torch.float32))
    return torch.cat(x), torch.cat(y)


def fit(data, dt, device, use_slip=False):
    """Fit on train and choose L2 penalty by validation log loss."""
    x, y = load_features(data / "train", dt, device, use_slip)
    xv, yv = load_features(data / "val", dt, device, use_slip)
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    x, xv = (x - mean) / scale, (xv - mean) / scale
    net = torch.nn.Linear(x.shape[1], 1, device=device)
    best = None
    for penalty in (0.0001, 0.01, 0.1):
        with torch.no_grad():
            net.weight.zero_()
            net.bias.zero_()
        opt = torch.optim.LBFGS(net.parameters(), max_iter=40, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(net(x).flatten(), y) + penalty * net.weight.square().sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            loss = F.binary_cross_entropy_with_logits(net(xv).flatten(), yv).item()
            if best is None or loss < best["val_log_loss"]:
                best = {"mean": mean.tolist(), "scale": scale.tolist(), "weight": net.weight.flatten().tolist(),
                        "bias": net.bias.item(), "l2": penalty, "val_log_loss": loss, "use_slip": use_slip}
    return best


def scorer(params, device):
    mean, scale, weight = (torch.tensor(params[k], device=device) for k in ("mean", "scale", "weight"))

    def score(w, dt):
        x = features(w, dt, params.get("use_slip", False))
        return torch.sigmoid(((x - mean) / scale) @ weight + params["bias"])

    return score

"""Q2 belief network: 1D-CNN over a 0.4 s sensor window (docs/CONTRACTS.md C5).

Heads: p_hold (held* if motors are cut now), p_slip (slipping now), margin_min regression
(in g units, clipped). Inputs are RAW sensor windows; per-channel normalisation, the
channel-ablation mask and the temperature (calibration of p_hold) are buffers inside the
module, so the exported TorchScript takes raw windows:
    forward(x: Tensor[B, 80, 11 or 13]) -> Tensor[B, 3] = [p_hold, p_slip, margin_min_g]
Also: evaluation metrics (AUROC, ECE, Brier) without sklearn.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNEL_GROUPS = {"imu": [0, 1, 2, 3, 4, 5], "fnormal": [6], "contact": [7],
                  "thrust": [8], "phase": [9, 10], "slip": [11, 12]}
MARGIN_CLIP_G = 2.0


class BeliefNet(nn.Module):
    def __init__(self, mean, std, drop=(), ch=64, win=80, n_ch=None):
        super().__init__()
        n_ch = mean.numel() if n_ch is None else n_ch
        mask = torch.ones(n_ch)
        for g in drop:
            idx = CHANNEL_GROUPS[g]
            if max(idx) >= n_ch:
                raise ValueError(f"cannot drop {g}: input has {n_ch} channels")
            mask[idx] = 0.0
        self.register_buffer("mean", mean.float().clone())
        self.register_buffer("std", std.float().clamp(min=1e-3).clone())
        self.register_buffer("mask", mask)
        self.register_buffer("temp", torch.ones(1))
        self.conv = nn.Sequential(
            nn.Conv1d(n_ch, ch, 5, padding=2), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(ch, ch, 5, stride=2, padding=2), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(ch * (win // 4), 128), nn.ReLU(), nn.Linear(128, 3))

    def logits(self, x):
        z = (x.float() - self.mean) / self.std * self.mask
        return self.head(self.conv(z.transpose(1, 2)))

    def forward(self, x):
        o = self.logits(x)
        return torch.stack([torch.sigmoid(o[:, 0] / self.temp), torch.sigmoid(o[:, 1]), o[:, 2]], -1)


def loss_fn(o, y_hold, y_slip, y_margin, w=(1.0, 0.5, 0.5)):
    lh = F.binary_cross_entropy_with_logits(o[:, 0], y_hold)
    ls = F.binary_cross_entropy_with_logits(o[:, 1], y_slip)
    lm = F.smooth_l1_loss(o[:, 2], y_margin.clamp(-MARGIN_CLIP_G, MARGIN_CLIP_G), beta=0.1)
    return w[0] * lh + w[1] * ls + w[2] * lm, (lh.item(), ls.item(), lm.item())


def fit_temperature(logit, y, iters=200):
    """Temperature scaling (Guo et al. 2017) on validation logits; returns T > 0."""
    logit, y = logit.detach().float().cpu(), y.float().cpu()
    lt = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([lt], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        l = F.binary_cross_entropy_with_logits(logit / lt.exp(), y)
        l.backward()
        return l
    opt.step(closure)
    return float(lt.exp())


# ------------------------------------------------------------------ metrics
def auroc(score, y):
    """Mann-Whitney AUROC with average ranks for ties. score, y: 1-D tensors."""
    score, y = score.double().cpu(), y.bool().cpu()
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None
    order = torch.argsort(score)
    s = score[order]
    ranks = torch.empty_like(s)
    # average ranks over ties
    _, inv, counts = torch.unique_consecutive(s, return_inverse=True, return_counts=True)
    ends = torch.cumsum(counts, 0).double()
    avg = ends - (counts.double() - 1) / 2
    ranks = avg[inv]
    r = torch.empty_like(ranks)
    r[order] = ranks
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def ece(p, y, bins=15):
    """Expected calibration error, equal-width bins on p (probability of the positive class)."""
    p, y = p.double().cpu(), y.double().cpu()
    idx = (p * bins).long().clamp(max=bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.double().mean().item() * abs(p[m].mean().item() - y[m].mean().item())
    return e


def brier(p, y):
    return float(((p.double().cpu() - y.double().cpu()) ** 2).mean())


def reliability(p, y, bins=15):
    p, y = p.double().cpu(), y.double().cpu()
    idx = (p * bins).long().clamp(max=bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append({"bin": [round(b / bins, 4), round((b + 1) / bins, 4)], "n": int(m.sum()),
                        "p_mean": round(p[m].mean().item(), 4), "y_mean": round(y[m].mean().item(), 4)})
    return out


def wilson(k, n, z=1.96, digits=4):
    """Wilson 95% interval (same as scripts/train.py)."""
    if n == 0:
        return {"p": None, "lo": None, "hi": None, "n": 0}
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    lo, hi = max(c - h, 0.0), min(c + h, 1.0)
    if digits is not None:
        p, lo, hi = (round(x, digits) for x in (p, lo, hi))
    return {"p": p, "lo": lo, "hi": hi, "n": n}

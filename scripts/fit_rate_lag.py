"""Fit first-order lag + pure delay of the PX4 rate loop from g2_probe.json doublets."""
import json, sys
import numpy as np
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/home/sup/data/perchpilot/sitl/g2_probe.json"))
a = np.array(d["rate"])            # t, cmd r p y, gyro x y z, thr, z
t = a[:, 0]; cmd = a[:, 1:4]; gyro = a[:, 4:7]
dt = np.median(np.diff(t))
res = {"samples": len(t), "dt_median_s": float(dt), "dt_p95_s": float(np.percentile(np.diff(t), 95)), "z_range": [float(a[:,8].min()), float(a[:,8].max())]}
names = ["roll", "pitch", "yaw"]
for ax in range(3):
    best = None
    for delay in np.arange(0, 0.15, 0.005):
        for tau in np.arange(0.005, 0.5, 0.005):
            # simulate y' = (u(t-delay) - y)/tau on the logged time grid
            u = np.interp(t - delay, t, cmd[:, ax])
            y = np.zeros_like(u); y[0] = gyro[0, ax]
            for k in range(1, len(t)):
                h = t[k] - t[k-1]
                y[k] = y[k-1] + (u[k-1] - y[k-1]) * min(1.0, h / tau)
            # evaluate only around doublets (|cmd| large) +- 0.5 s
            m = np.convolve((np.abs(cmd[:, ax]) > 1.0).astype(float), np.ones(50), "same") > 0
            e = np.sqrt(np.mean((y[m] - gyro[m, ax]) ** 2))
            if best is None or e < best[0]:
                best = (e, delay, tau)
    peak = float(np.abs(gyro[:, ax]).max())
    res[names[ax]] = {"rmse_rad_s": round(float(best[0]), 3), "delay_s": round(float(best[1]), 3), "tau_s": round(float(best[2]), 3), "peak_gyro": round(peak, 2)}
print(json.dumps(res, indent=1))

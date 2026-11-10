"""
plot_steplog.py - "Smoothed Latency and Utilization over Time" from a steplog CSV.

Usage (needs pandas + matplotlib + scipy, i.e. run on the host, not in counsel:t113):
  python3 tools/plot_steplog.py results/raw/steplog/X.csv [--episode 0] [--sigma 3] [--slo 150]
                          [--raw] [--out results/figures/X.png]

Left axis: latency (Gaussian-smoothed) with the SLO as a dashed line.
Right axis: mean component utilization (Gaussian-smoothed), 0..1.
Red ticks along the bottom mark steps where the env counted an SLO breach.
--raw overlays the unsmoothed series faintly so smoothing artefacts are visible.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d

p = argparse.ArgumentParser()
p.add_argument("csv")
p.add_argument("--episode", type=int, default=0)
p.add_argument("--sigma", type=float, default=3)
p.add_argument("--slo", type=float, default=None,
               help="default: inferred as latency / lat_over_slo")
p.add_argument("--util", default="util_mean", choices=["util_mean", "util_max"])
p.add_argument("--raw", action="store_true")
p.add_argument("--lat_cap", type=float, default=None,
               help="clip latency for display (default 3*SLO)")
p.add_argument("--out", default=None)
a = p.parse_args()

df = pd.read_csv(a.csv)
df = df[df["episode"] == a.episode].reset_index(drop=True)
if df.empty:
    raise SystemExit(f"no rows for episode {a.episode}")

slo = a.slo
if slo is None:
    nz = df[df["lat_over_slo"] > 0]
    slo = float((nz["latency"] / nz["lat_over_slo"]).median()) if len(nz) else 150.0

# A component shrunk to 0 CPU/mem gives latency ~ load/1e-7 (~1e9), which would
# flatten everything else; clip for display only and say how many points.
cap = a.lat_cap if a.lat_cap is not None else 3 * slo
lat_raw = df["latency"].to_numpy(float)
n_clip = int((lat_raw > cap).sum())
lat_disp = lat_raw.clip(max=cap)
lat_s = gaussian_filter1d(lat_disp, sigma=a.sigma)
util_s = gaussian_filter1d(df[a.util].to_numpy(float), sigma=a.sigma)

fig, ax1 = plt.subplots(figsize=(10, 6))
lat_label = "Latency" + (f" (clipped at {cap:g}: {n_clip} pts)" if n_clip else "")
ax1.plot(df["step"], lat_s, color="blue", label=lat_label)
if a.raw:
    ax1.plot(df["step"], lat_disp, color="blue", alpha=0.2, lw=0.6)
ax1.axhline(slo, color="red", ls="--", label="SLO Latency")
breaches = df.loc[df["breach"] == 1, "step"]
ax1.plot(breaches, [0] * len(breaches), "|", color="red", ms=8,
         label=f"SLO breach ({len(breaches)})")
ax1.set_xlabel("Time Steps")
ax1.set_ylabel("Latency", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.set_xlim(0, df["step"].max())
ax1.set_ylim(0, max(slo * 1.33, lat_disp.max() * 1.05 if a.raw else lat_s.max() * 1.1))

ax2 = ax1.twinx()
ax2.plot(df["step"], util_s, color="green", label="Utilization")
if a.raw:
    ax2.plot(df["step"], df[a.util], color="green", alpha=0.2, lw=0.6)
ax2.set_ylabel("Utilization", color="green")
ax2.tick_params(axis="y", labelcolor="green")
ax2.set_ylim(0, max(1.06, util_s.max() * 1.05))
ax2.grid(True)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper right")
plt.title("Smoothed Latency and Utilization over Time")
plt.tight_layout()

out = a.out or os.path.join("results", "figures", os.path.splitext(os.path.basename(a.csv))[0]
                            + f"_ep{a.episode}.png")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
plt.savefig(out, dpi=100)
print(out)

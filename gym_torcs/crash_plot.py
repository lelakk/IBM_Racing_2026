"""
crash_plot.py  —  Visualise where the car crashes from telemetry/crashes.csv

Usage:
    python crash_plot.py                  # reads ./telemetry/crashes.csv
    python crash_plot.py --csv my/path.csv
    python crash_plot.py --last 500       # only most-recent 500 crashes
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--csv",  default="./telemetry/crashes.csv")
parser.add_argument("--last", type=int, default=None,
                    help="Plot only the most recent N crashes")
args = parser.parse_args()

csv_path = Path(args.csv)
if not csv_path.exists():
    sys.exit(f"[crash_plot] File not found: {csv_path}\n"
             "Train first — crashes are logged automatically.")

df = pd.read_csv(csv_path)
if df.empty:
    sys.exit("[crash_plot] No crashes logged yet.")

if args.last:
    df = df.tail(args.last)

print(f"Loaded {len(df)} crashes from {csv_path}")


# ── Figure setup ─────────────────────────────────────────────────────────────

BG   = "#0d1117"
FG   = "#e6edf3"
ACC  = "#58a6ff"
WARN = "#f78166"
OK   = "#3fb950"

REASON_COLORS = {
    "off_track":   "#f78166",
    "no_progress": "#d29922",
    "wrong_way":   "#bc8cff",
}

fig = plt.figure(figsize=(16, 10), facecolor=BG)
fig.suptitle("TORCS Crash Telemetry", color=FG, fontsize=16, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                       left=0.07, right=0.97, top=0.92, bottom=0.08)

ax_main  = fig.add_subplot(gs[:, 0:2])   # large: dist_raced vs track_pos
ax_hist  = fig.add_subplot(gs[0, 2])     # dist_raced histogram
ax_speed = fig.add_subplot(gs[1, 2])     # speed at crash histogram


def _style(ax, title):
    ax.set_facecolor("#161b22")
    ax.spines[:].set_color("#30363d")
    ax.tick_params(colors=FG, labelsize=8)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.set_title(title, color=FG, fontsize=10, pad=8)
    ax.grid(color="#21262d", linewidth=0.5)


# ── Main scatter: track position vs distance raced ───────────────────────────

_style(ax_main, "Crash map  (dist_raced vs track_pos)")

# Draw track boundaries
ax_main.axhline( 1.0, color="#30363d", linewidth=1.2, linestyle="--")
ax_main.axhline(-1.0, color="#30363d", linewidth=1.2, linestyle="--")
ax_main.axhline( 0.0, color="#21262d", linewidth=0.7, linestyle=":")

# Shade off-track zones
ax_main.axhspan( 1.0,  2.0, color="#f78166", alpha=0.06)
ax_main.axhspan(-2.0, -1.0, color="#f78166", alpha=0.06)

ax_main.text(df["dist_raced"].max() * 0.98,  1.05, "left edge",
             color="#f78166", fontsize=7, ha="right", va="bottom")
ax_main.text(df["dist_raced"].max() * 0.98, -1.05, "right edge",
             color="#f78166", fontsize=7, ha="right", va="top")

for reason, colour in REASON_COLORS.items():
    sub = df[df["reason"] == reason]
    if sub.empty:
        continue
    ax_main.scatter(
        sub["dist_raced"], sub["track_pos"],
        c=colour, s=22, alpha=0.65, linewidths=0,
        label=f"{reason}  ({len(sub)})",
        zorder=3,
    )

# Training progression: colour by episode (early = dark, late = bright)
if len(df) > 1:
    scatter_all = ax_main.scatter(
        df["dist_raced"], df["track_pos"],
        c=df["episode"], cmap="plasma",
        s=10, alpha=0.25, linewidths=0, zorder=2,
    )
    cbar = fig.colorbar(scatter_all, ax=ax_main, pad=0.01)
    cbar.ax.tick_params(colors=FG, labelsize=7)
    cbar.set_label("episode", color=FG, fontsize=8)

ax_main.set_xlabel("Distance raced (m)", fontsize=9)
ax_main.set_ylabel("Track position  (−1 = right edge, +1 = left edge)", fontsize=9)
ax_main.set_ylim(-1.8, 1.8)
legend = ax_main.legend(fontsize=8, framealpha=0.15, labelcolor=FG,
                         facecolor="#161b22", edgecolor="#30363d")


# ── Histogram: where on track do crashes happen? ─────────────────────────────

_style(ax_hist, "Crash density along track")

bins = min(60, max(10, len(df) // 5))
ax_hist.hist(df["dist_raced"], bins=bins, color=ACC, alpha=0.8, edgecolor="none")
ax_hist.set_xlabel("dist_raced (m)", fontsize=8)
ax_hist.set_ylabel("# crashes", fontsize=8)


# ── Histogram: speed at crash ─────────────────────────────────────────────────

_style(ax_speed, "Speed at crash (km/h)")

speeds = df["speed_x"].abs()   # TORCS speedX can be negative if reversed
ax_speed.hist(speeds, bins=bins, color=WARN, alpha=0.8, edgecolor="none")
ax_speed.set_xlabel("speedX at crash (km/h)", fontsize=8)
ax_speed.set_ylabel("# crashes", fontsize=8)


# ── Summary stats ────────────────────────────────────────────────────────────

reason_counts = df["reason"].value_counts()
stats_lines = [
    f"Total crashes : {len(df)}",
    f"Episodes span : {df['episode'].min()}–{df['episode'].max()}",
    "",
    *[f"  {r:<14}: {n}" for r, n in reason_counts.items()],
    "",
    f"Median dist   : {df['dist_raced'].median():.0f} m",
    f"Median speed  : {speeds.median():.1f} km/h",
    f"Most dangerous: {df['dist_raced'].value_counts(bins=20).idxmax().mid:.0f} m",
]
fig.text(0.01, 0.01, "\n".join(stats_lines),
         color="#8b949e", fontsize=7.5, va="bottom",
         fontfamily="monospace")


# ── Save + show ───────────────────────────────────────────────────────────────

out = Path("./telemetry/crash_map.png")
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
plt.show()
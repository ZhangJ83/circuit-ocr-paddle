#!/usr/bin/env python3
"""Redraw fig_model_comparison.png and fig_ablation_summary.png
with updated metrics: CompF1 + LineAcc + NED. No old JointF1 bars."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "grid.linewidth": 0.4,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})

C_BLUE = "#2166AC"
C_ORANGE = "#FC8D59"
C_RED = "#B2182B"
C_GREEN = "#4DAF4A"
C_GRAY = "#999999"
C_BASE = "#BDBDBD"
C_BEST = "#053061"

UP = r"$\uparrow$"
DOWN = r"$\downarrow$"

# ==============================================================
# Figure 1: Model Comparison - Base vs v1 vs v2
# Left: CompF1 + LineAcc grouped bar | Right: NED horizontal bar
# ==============================================================
models_labels = ["Base\nPaddleOCR-VL", "v1 (exp6)\n+synth text 20%", "v2 (Phase 1)\nsynth KiCad 5k"]
models_short = ["Base", "v1", "v2"]
comp_f1 = [0.000, 0.119, 0.304]
line_acc = [0.000, 0.033, 0.040]
ned = [0.944, 0.946, 0.942]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5),
                                 gridspec_kw={"width_ratios": [1.5, 1]})

# Panel A: Grouped bar - CompF1 + LineAcc
x = np.arange(len(models_labels))
w = 0.30
b1 = ax1.bar(x - w/2, comp_f1, w, color=[C_BASE, C_ORANGE, C_BLUE],
             edgecolor="black", linewidth=0.5, label="CompF1 " + UP)
b2 = ax1.bar(x + w/2, line_acc, w, color=[C_BASE, "#FDD49E", "#92C5DE"],
             edgecolor="black", linewidth=0.5, label="LineAcc " + UP)

for bar, val in zip(b1, comp_f1):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
             f"{val:.3f}", ha="center", fontsize=7.5, fontweight="bold")
for bar, val in zip(b2, line_acc):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
             f"{val:.3f}", ha="center", fontsize=7.5)

# 2.6x annotation
ax1.annotate("2.6x", xy=(2, 0.304), xytext=(2.3, 0.26),
             fontsize=9, fontweight="bold", color=C_RED,
             arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2))

ax1.set_ylabel("Score " + UP)
ax1.set_xticks(x)
ax1.set_xticklabels(models_labels, fontsize=8)
ax1.set_ylim(0, 0.38)
ax1.legend(loc="upper left", framealpha=0.9, edgecolor="gray", fontsize=8)
ax1.grid(axis="y", alpha=0.2)
ax1.set_title("(a)  CompF1 + LineAcc", fontsize=10, fontweight="bold", loc="left")

# Panel B: NED horizontal (lower is better, invert for display)
ned_display = [1.0 - n for n in ned]
ned_colors = [C_BASE, C_ORANGE, C_BLUE]
bars = ax2.barh(range(len(models_short)), ned_display, height=0.5,
                color=ned_colors, edgecolor="black", linewidth=0.5)
for i, (bar, val) in enumerate(zip(bars, ned)):
    ax2.text(bar.get_width() - 0.012, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", ha="right", va="center", fontsize=8, fontweight="bold",
             color="white")
ax2.set_yticks(range(len(models_short)))
ax2.set_yticklabels(models_short, fontsize=8)
ax2.set_xlabel("1.0 - NED  " + UP + " (higher better)")
ax2.set_xlim(0, 0.10)
ax2.grid(axis="x", alpha=0.2)
ax2.set_title("(b)  NED (inverted)", fontsize=10, fontweight="bold", loc="left")

fig.suptitle("Model Comparison: Base vs v1 vs v2", fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout(pad=1.5)
fig.savefig(os.path.join(OUT, "fig_model_comparison.png"),
            facecolor="white", edgecolor="none")
plt.close()
print("Saved: fig_model_comparison.png")

# ==============================================================
# Figure 2: Ablation Summary - 8 experiments, CompF1 bars
# ==============================================================
exp_names = [
    "exp1\nBaseline", "exp2\n512px", "exp3\nLowLR\n+DO",
    "exp4\nUnfreeze\nProj", "exp5\nSynth\n(reg)", "exp6\nSynth\n(base)",
    "exp7\nr=32", "exp8\nSPICE"
]
exp_comp_f1 = [0.037, 0.058, 0.028, 0.092, 0.126, 0.156, 0.084, 0.042]
exp_colors = [
    C_GRAY, C_GRAY, C_GRAY, C_RED, C_ORANGE, C_GREEN, C_GRAY, C_RED
]
exp_hatch = ["", "", "", "//", "", "//", "", "//"]

fig, ax = plt.subplots(figsize=(7.5, 3.5))
x = np.arange(len(exp_names))
bars = ax.bar(x, exp_comp_f1, color=exp_colors, edgecolor="black", linewidth=0.5)

for bar, hatch in zip(bars, exp_hatch):
    if hatch:
        bar.set_hatch(hatch)

for i, (bar, val) in enumerate(zip(bars, exp_comp_f1)):
    y_pos = bar.get_height() + 0.005
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f"{val:.3f}",
            ha="center", fontsize=7.5,
            fontweight="bold" if val >= 0.15 else "normal")

# Collapse labels
for i, val in enumerate(exp_comp_f1):
    if val < 0.05 and i not in [2]:  # exp3 is low lr, not strictly collapse
        if i == 0:
            ax.annotate("collapse", (i, val), textcoords="offset points",
                        xytext=(0, 12), fontsize=6.5, color=C_RED, ha="center",
                        fontstyle="italic")

# Best label
ax.annotate("v1 best", (5, exp_comp_f1[5]), textcoords="offset points",
            xytext=(0, 14), fontsize=8, color=C_GREEN, ha="center", fontweight="bold")

ax.set_ylabel("CompF1 " + UP)
ax.set_xticks(x)
ax.set_xticklabels(exp_names, fontsize=7.5)
ax.set_ylim(0, 0.22)
ax.grid(axis="y", alpha=0.2)

legend_elements = [
    mpatches.Patch(facecolor=C_GREEN, label="Effective (CompF1 > 0.15)"),
    mpatches.Patch(facecolor=C_ORANGE, label="Moderate (0.10-0.15)"),
    mpatches.Patch(facecolor=C_GRAY, label="Collapse / No gain (< 0.10)"),
    mpatches.Patch(facecolor=C_RED, label="Degenerate / Harmful"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=7,
          framealpha=0.9, edgecolor="gray", ncol=2)

ax.set_title("Phase 1-2 Ablation: CompF1 across 8 Experiments",
             fontsize=11, fontweight="bold")
fig.tight_layout(pad=1.0)
fig.savefig(os.path.join(OUT, "fig_ablation_summary.png"),
            facecolor="white", edgecolor="none")
plt.close()
print("Saved: fig_ablation_summary.png")

print("\nDone. Both figures regenerated.")

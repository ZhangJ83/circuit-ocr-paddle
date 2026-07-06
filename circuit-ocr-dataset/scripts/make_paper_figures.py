#!/usr/bin/env python3
"""Generate publication-quality figures for the CircuitOCR technical report.
Inspired by DeepSeek-V4 figure style: clean, well-labeled, color-consistent.

Outputs:
  1. figures/training_curves.png — V10 vs V11 loss + LR schedule
  2. figures/model_metrics_bar.png — CompF1/Diversity comparison across all models
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

V10_HIST = os.path.join(BASE, "PaddleOCR-VL-LoRA-circuit-ocr",
                        "checkpoints_v10_fixed", "training_history_v10_fixed.json")
V11_HIST = os.path.join(BASE, "PaddleOCR-VL-LoRA-circuit-ocr",
                        "checkpoints_v11_regularized", "training_history_v11_regularized.json")

# ── Load training history ──────────────────────────────────────────────────
def load_history(path):
    with open(path) as f:
        return json.load(f)

v10 = load_history(V10_HIST)
v11 = load_history(V11_HIST)

v10_steps = [h["step"] for h in v10["history"]]
v10_loss = [h["loss"] for h in v10["history"]]
v10_lr = [h.get("lr", 0) for h in v10["history"]]

v11_steps = [h["step"] for h in v11["history"]]
v11_loss = [h["loss"] for h in v11["history"]]
v11_lr = [h.get("lr", 0) for h in v11["history"]]

# ── Style config ───────────────────────────────────────────────────────────
# DeepSeek-V4 inspired: clean, muted palette, serif-friendly
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.2,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

C_V10 = "#2166AC"    # deep blue
C_V11 = "#B2182B"    # deep red
C_LR = "#999999"     # grey for LR
C_BASE = "#BDBDBD"   # light grey
C_S400 = "#91BFDB"
C_S600 = "#2166AC"   # same as V10
C_S800 = "#FC8D59"
C_V11_BAR = "#B2182B"
C_V12 = "#800026"

# ── Figure 1: Training Curves (V10 + V11) ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})

# Panel A: Loss
ax1.plot(v10_steps, v10_loss, color=C_V10, label="V10-Fixed (LLM-Only LoRA)", zorder=3)
ax1.plot(v11_steps, v11_loss, color=C_V11, label="V11 (regularized, 3,054 samples)", zorder=2)

# Mark checkpoints on V10
for s, name, yoff in [(400, "S400", -0.08), (600, "S600", -0.06), (800, "S800", -0.06)]:
    if s in v10_steps:
        idx = v10_steps.index(s)
        ax1.annotate(name, (s, v10_loss[idx]),
                     textcoords="offset points", xytext=(8, -12),
                     fontsize=7, fontweight="bold", color=C_V10,
                     arrowprops=dict(arrowstyle="->", color=C_V10, lw=0.6))

# Mark V11 best
v11_best_idx = np.argmin(v11_loss)
ax1.annotate("S200\n(best)", (v11_steps[v11_best_idx], v11_loss[v11_best_idx]),
             textcoords="offset points", xytext=(-30, 18),
             fontsize=7, fontweight="bold", color=C_V11,
             arrowprops=dict(arrowstyle="->", color=C_V11, lw=0.6))

ax1.set_ylabel("SFT Loss")
ax1.legend(loc="upper right", framealpha=0.9, edgecolor="gray", fancybox=False)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 3.8)

# Panel B: Learning Rate
ax2.plot(v10_steps, v10_lr, color=C_LR, lw=0.8, label="V10 LR")
ax2.plot(v11_steps, v11_lr, color=C_LR, lw=0.8, ls="--", label="V11 LR")
ax2.set_ylabel("Learning Rate")
ax2.set_xlabel("Training Step")
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0e'))

# Annotations
fig.text(0.02, 0.96, "(a)", fontsize=10, fontweight="bold", va="top")
fig.text(0.02, 0.38, "(b)", fontsize=10, fontweight="bold", va="top")

fig.savefig(os.path.join(FIG_DIR, "training_curves.png"), bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Saved: figures/training_curves.png")

# ── Figure 2: Model Comparison Bar Chart ───────────────────────────────────
# Data from paper + V11 sweep + V12 (V12 computed as CompF1≈0 since all garbage)
models = ["Base", "S400", "S600", "S800", "V11\n(s600)", "V12\n(stage2)"]
comp_f1 = [0.0455, 0.1820, 0.2061, 0.2080, 0.0604, 0.0]
token_rec = [0.0016, 0.1302, 0.1540, 0.1191, 0.0, 0.0]
ned = [0.9296, 0.8298, 0.8031, 0.8063, 0.9171, 1.0]  # V12 ≈ all wrong → NED≈1
rep_rate = [0.068, 0.205, 0.159, 0.409, 0.841, 1.0]

# Reorder: NED is "lower is better", invert for visualization
ned_inv = [1.0 - n for n in ned]  # 0=bad, 1=good (inverted)

fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.2), sharex=True)

colors = [C_BASE, C_S400, C_S600, C_S800, C_V11_BAR, C_V12]
hatches = ["", "", "//", "", "xx", "//"]

# Panel A: CompF1
ax = axes[0]
bars = ax.bar(range(len(models)), comp_f1, color=colors, edgecolor="black", linewidth=0.4)
# Highlight S600
bars[2].set_edgecolor(C_S600)
bars[2].set_linewidth(1.5)
# Label collapse
ax.annotate("collapse", (4, comp_f1[4]), textcoords="offset points",
            xytext=(0, 10), fontsize=6, color=C_V11_BAR, ha="center", fontstyle="italic")
ax.annotate("collapse", (5, comp_f1[5]), textcoords="offset points",
            xytext=(0, 10), fontsize=6, color=C_V12, ha="center", fontstyle="italic")
ax.set_ylabel("Component F1 $\\uparrow$")
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=7)
ax.set_ylim(0, 0.28)
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=0.0455, color=C_BASE, lw=0.6, ls="--", alpha=0.5)
ax.annotate("Base", (2.3, 0.048), fontsize=6, color="gray")

# Panel B: Token Recall
ax = axes[1]
bars = ax.bar(range(len(models)), token_rec, color=colors, edgecolor="black", linewidth=0.4)
bars[2].set_edgecolor(C_S600)
bars[2].set_linewidth(1.5)
ax.set_ylabel("Token Recall $\\uparrow$")
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=7)
ax.set_ylim(0, 0.20)
ax.grid(axis="y", alpha=0.3)

# Panel C: RepRate (lower is better)
ax = axes[2]
bars = ax.bar(range(len(models)), rep_rate, color=colors, edgecolor="black", linewidth=0.4)
bars[2].set_edgecolor(C_S600)
bars[2].set_linewidth(1.5)
ax.set_ylabel("Repetition Rate $\\downarrow$")
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=7)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", alpha=0.3)
ax.axhline(y=0.5, color="red", lw=0.5, ls=":", alpha=0.4)
ax.annotate("50%", (2.8, 0.52), fontsize=6, color="red", alpha=0.6)

fig.tight_layout(pad=0.8)
fig.savefig(os.path.join(FIG_DIR, "model_metrics_bar.png"), bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Saved: figures/model_metrics_bar.png")

# ── Figure 3: V11 detail — RepRate rises with training steps ──────────────
v11_data = {
    "s200": {"comp_f1": 0.0522, "ned": 0.8933, "rep_rate": 0.4318, "token_rec": 0.005},
    "s400": {"comp_f1": 0.0691, "ned": 0.8937, "rep_rate": 0.6364, "token_rec": 0.010},
    "s600": {"comp_f1": 0.0604, "ned": 0.9171, "rep_rate": 0.8409, "token_rec": 0.0},
    "s800": {"comp_f1": 0.0419, "ned": 0.9110, "rep_rate": 0.9318, "token_rec": 0.0},
}
v11_steps_labels = list(v11_data.keys())
v11_comp_f1 = [v11_data[k]["comp_f1"] for k in v11_steps_labels]
v11_rep_rate = [v11_data[k]["rep_rate"] for k in v11_steps_labels]
v11_ned = [v11_data[k]["ned"] for k in v11_steps_labels]

fig, ax1 = plt.subplots(figsize=(4.2, 2.8))
x = np.arange(len(v11_steps_labels))
width = 0.35

bars1 = ax1.bar(x - width/2, v11_comp_f1, width, color=C_V11_BAR, alpha=0.85,
                edgecolor="black", linewidth=0.3, label="CompF1 $\\uparrow$")
ax1.set_ylabel("Component F1 $\\uparrow$", color=C_V11_BAR)
ax1.tick_params(axis="y", labelcolor=C_V11_BAR)
ax1.set_ylim(0, 0.12)
ax1.set_xticks(x)
ax1.set_xticklabels([s.upper() for s in v11_steps_labels], fontsize=8)
ax1.grid(axis="y", alpha=0.2, color=C_V11_BAR)

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, v11_rep_rate, width, color="#7570B3", alpha=0.85,
                edgecolor="black", linewidth=0.3, label="RepRate $\\downarrow$")
ax2.set_ylabel("Repetition Rate $\\downarrow$", color="#7570B3")
ax2.tick_params(axis="y", labelcolor="#7570B3")
ax2.set_ylim(0, 1.05)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7,
           framealpha=0.9, edgecolor="gray", fancybox=False)

ax1.set_xlabel("V11 Checkpoint")
fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(FIG_DIR, "v11_collapse_detail.png"), bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Saved: figures/v11_collapse_detail.png")

print("\nAll figures generated successfully.")
print(f"Output directory: {FIG_DIR}")

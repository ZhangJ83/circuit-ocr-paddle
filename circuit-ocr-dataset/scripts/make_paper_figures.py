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

# ── Figure 4: Dataset composition donut ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

# Panel A: Full dataset breakdown
labels_a = ["Train\n(real 1,104)", "Train\n(synth 450)", "Val\n(real 171)", "Val\n(synth 50)",
            "Test\n(held-out 82)"]
sizes_a = [1104, 450, 171, 50, 82]
colors_a = ["#2166AC", "#92C5DE", "#4393C3", "#D1E5F0", "#053061"]
explode_a = (0, 0, 0, 0, 0.05)
wedges1, texts1, autotexts1 = axes[0].pie(
    sizes_a, explode=explode_a, labels=labels_a, colors=colors_a,
    autopct="%1.0f%%", startangle=90, pctdistance=0.6, labeldistance=1.12,
    textprops={"fontsize": 6.5})
axes[0].set_title("Full V5 Golden\n(1,857 samples)", fontsize=9, fontweight="bold")

# Panel B: Training set (1,554)
sizes_b = [450, 1104]
labels_b = ["Synthetic V3\n(450, 29%)", "Real KiCad\n(1,104, 71%)"]
colors_b = ["#FC8D59", "#2166AC"]
wedges2, texts2, autotexts2 = axes[1].pie(
    sizes_b, labels=labels_b, colors=colors_b, autopct="%1.0f%%",
    startangle=90, pctdistance=0.55, textprops={"fontsize": 7.5})
axes[1].set_title("Training Set\n(1,554 samples)", fontsize=9, fontweight="bold")

# Panel C: Validation set (221)
sizes_c = [50, 171]
labels_c = ["Synthetic V3\n(50, 23%)", "Real KiCad\n(171, 77%)"]
colors_c = ["#FC8D59", "#2166AC"]
wedges3, texts3, autotexts3 = axes[2].pie(
    sizes_c, labels=labels_c, colors=colors_c, autopct="%1.0f%%",
    startangle=90, pctdistance=0.55, textprops={"fontsize": 7.5})
axes[2].set_title("Validation Set\n(221 samples)", fontsize=9, fontweight="bold")

fig.tight_layout(pad=1.0)
fig.savefig(os.path.join(FIG_DIR, "dataset_donut.png"), bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Saved: figures/dataset_donut.png")

# ── Figure 5: V10 Checkpoint sweep line chart ─────────────────────────────
checkpoints = ["S400", "S600", "S800"]
comp_f1_sweep = [0.1820, 0.2061, 0.2080]
token_rec_sweep = [0.1302, 0.1540, 0.1191]
ned_sweep = [0.8298, 0.8031, 0.8063]        # lower better
rep_rate_sweep = [0.205, 0.159, 0.409]
diversity_sweep = [0.955, 0.909, 0.932]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 4.5), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1], "hspace": 0.06})
x = np.arange(len(checkpoints))
w = 0.18

# Panel A: CompF1 + TokenRec
b1 = ax1.bar(x - w, comp_f1_sweep, w, color="#2166AC", edgecolor="white", lw=0.3,
             label="CompF1 $\\uparrow$")
ax1_twin = ax1.twinx()
b2 = ax1_twin.bar(x, token_rec_sweep, w, color="#FC8D59", edgecolor="white", lw=0.3,
                  label="TokenRec $\\uparrow$")
b3 = ax1_twin.bar(x + w, diversity_sweep, w, color="#4DAF4A", edgecolor="white", lw=0.3,
                  label="Diversity $\\uparrow$")

ax1.set_ylabel("CompF1 $\\uparrow$", fontsize=8, color="#2166AC")
ax1.tick_params(axis="y", labelcolor="#2166AC", labelsize=7)
ax1.set_ylim(0, 0.40)
ax1.grid(axis="y", alpha=0.15, color="#2166AC")
ax1_twin.set_ylabel("TokenRec / Diversity $\\uparrow$", fontsize=8, color="#FC8D59")
ax1_twin.tick_params(axis="y", labelcolor="#FC8D59", labelsize=7)
ax1_twin.set_ylim(0, 1.05)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7,
           framealpha=0.9, edgecolor="gray", fancybox=False, ncol=3)

# Highlight S600 as optimal
for i, (name, val) in enumerate(zip(checkpoints, comp_f1_sweep)):
    if name == "S600":
        ax1.annotate("[BEST]", (i - w, val), textcoords="offset points",
                     xytext=(0, 12), fontsize=7, ha="center", fontweight="bold",
                     color="#2166AC")

# Panel B: NED + RepRate (both lower is better)
ax2.plot(x, ned_sweep, "o-", color="#B2182B", lw=1.5, markersize=7, label="NED $\\downarrow$",
         markerfacecolor="white", markeredgewidth=1.5)
for i, val in enumerate(ned_sweep):
    ax2.annotate(f"{val:.4f}", (i, val), textcoords="offset points",
                 xytext=(0, -14), fontsize=6.5, ha="center", color="#B2182B")
ax2_twin = ax2.twinx()
ax2_twin.bar(x, rep_rate_sweep, 0.4, color="#7570B3", alpha=0.6, edgecolor="white", lw=0.3,
             label="RepRate $\\downarrow$")
for i, val in enumerate(rep_rate_sweep):
    ax2_twin.annotate(f"{val:.1%}", (i, val), textcoords="offset points",
                      xytext=(0, 8), fontsize=6.5, ha="center", color="#7570B3")

ax2.set_ylabel("NED $\\downarrow$", fontsize=8, color="#B2182B")
ax2.tick_params(axis="y", labelcolor="#B2182B", labelsize=7)
ax2.set_ylim(0.78, 0.86)
ax2.grid(axis="y", alpha=0.15)
ax2_twin.set_ylabel("RepRate $\\downarrow$", fontsize=8, color="#7570B3")
ax2_twin.tick_params(axis="y", labelcolor="#7570B3", labelsize=7)
ax2_twin.set_ylim(0, 0.55)

lines3, labels3 = ax2.get_legend_handles_labels()
lines4, labels4 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines3 + lines4, labels3 + labels4, loc="upper left", fontsize=7,
           framealpha=0.9, edgecolor="gray", fancybox=False)

ax2.set_xticks(x)
ax2.set_xticklabels(checkpoints, fontsize=8)
ax2.set_xlabel("Checkpoint", fontsize=8)

fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(FIG_DIR, "checkpoint_sweep.png"), bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Saved: figures/checkpoint_sweep.png")

# ── Figure 6: Phase 2 Topology capacity gap ────────────────────────────────
models_p2 = ["Base", "S400", "S600", "S800"]
comp_f1_p2 = [0.0455, 0.1820, 0.2061, 0.2080]
joint_f1_p2 = [0.0000, 0.0027, 0.0191, 0.0064]
value_acc_p2 = [0.0, 0.006, 0.133, 0.023]

fig, ax = plt.subplots(figsize=(5.2, 3.0))
x = np.arange(len(models_p2))
w = 0.22

ax.bar(x - w, comp_f1_p2, w, color="#2166AC", edgecolor="white", lw=0.3,
       label="comp_f1 (refdes-only)")
ax.bar(x, joint_f1_p2, w, color="#FC8D59", edgecolor="white", lw=0.3,
       label="joint_f1 (refdes, value)")
ax.bar(x + w, value_acc_p2, w, color="#4DAF4A", edgecolor="white", lw=0.3,
       label="value_acc")

# Annotate S600 values
for i, (jf, cf) in enumerate(zip(joint_f1_p2, comp_f1_p2)):
    if models_p2[i] == "S600":
        ax.annotate(f"{cf:.3f}", (i - w, cf), textcoords="offset points",
                    xytext=(0, 6), fontsize=6.5, ha="center", fontweight="bold", color="#2166AC")
        ax.annotate(f"only {jf:.3f}", (i, jf), textcoords="offset points",
                    xytext=(0, 6), fontsize=6.5, ha="center", fontweight="bold", color="#FC8D59")
    elif jf > 0.001:
        ax.annotate(f"{jf:.4f}", (i, jf), textcoords="offset points",
                    xytext=(0, 5), fontsize=5.5, ha="center", color="#FC8D59")

ax.set_ylabel("F1 Score $\\uparrow$")
ax.set_xticks(x)
ax.set_xticklabels(models_p2, fontsize=8)
ax.legend(fontsize=7, framealpha=0.9, edgecolor="gray", fancybox=False, ncol=3)
ax.grid(axis="y", alpha=0.2)
ax.set_ylim(0, 0.30)

# Add gap annotation
ax.annotate("", xy=(2 - w/2, 0.0191), xytext=(2 - w/2, 0.2061),
            arrowprops=dict(arrowstyle="<->", color="red", lw=1.0, ls="--"))
ax.annotate("10.8× gap", xy=(1.85, 0.12), fontsize=7, color="red", fontweight="bold")

fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(FIG_DIR, "topology_gap.png"), bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Saved: figures/topology_gap.png")

# ── Figure 7: Version progression ──────────────────────────────────────────
versions = ["V3\n(2026-06)", "V5\n(2026-06)", "V8\n(2026-06)", "V9\n(2026-07)",
            "V10-Fixed\n(2026-07)", "V11\n(2026-07)", "V12\n(2026-07)"]
ned_hist = [0.8271, 0.9031, 0.8257, 0.7797, 0.8031, 0.9171, 1.0]
diversity_hist = [0.04, 0.90, 0.90, 0.90, 0.909, 0.50, 0.0]
comp_f1_hist = [0.03, 0.05, 0.12, 0.16, 0.2061, 0.0604, 0.0]
v_colors = ["#D1E5F0", "#92C5DE", "#4393C3", "#2166AC", "#053061", "#FC8D59", "#B2182B"]

fig, ax1 = plt.subplots(figsize=(7.0, 3.2))
x = np.arange(len(versions))

# NED line
ax1.plot(x, ned_hist, "s-", color="#B2182B", lw=1.5, markersize=6, label="NED $\\downarrow$",
         markerfacecolor="white", markeredgewidth=1.5)
ax1.set_ylabel("NED $\\downarrow$", color="#B2182B", fontsize=9)
ax1.tick_params(axis="y", labelcolor="#B2182B")
ax1.set_ylim(0.70, 1.05)

# Diversity line
ax1b = ax1.twinx()
ax1b.plot(x, diversity_hist, "D-", color="#4DAF4A", lw=1.5, markersize=6,
          label="Diversity $\\uparrow$", markerfacecolor="white", markeredgewidth=1.5)
ax1b.set_ylabel("Diversity $\\uparrow$", color="#4DAF4A", fontsize=9)
ax1b.tick_params(axis="y", labelcolor="#4DAF4A")
ax1b.set_ylim(-0.05, 1.1)

# Highlight milestones
# V5: diversity breakthrough
ax1b.annotate("diversity\nrestored 90%", (1, 0.90), textcoords="offset points",
              xytext=(-15, -25), fontsize=7, color="#4DAF4A", ha="center",
              arrowprops=dict(arrowstyle="->", color="#4DAF4A", lw=0.8))

# V9: best NED
ax1.annotate("best NED\n0.7797", (3, 0.7797), textcoords="offset points",
             xytext=(5, -30), fontsize=7, color="#B2182B", ha="center",
             arrowprops=dict(arrowstyle="->", color="#B2182B", lw=0.8))

# V10-Fixed: best CompF1
ax1.annotate("CompF1 0.206\n4.5× baseline", (4, 0.8031), textcoords="offset points",
             xytext=(0, 20), fontsize=7, color="#053061", ha="center", fontweight="bold")

# V11 collapse
ax1.annotate("collapse", (5, 0.9171), textcoords="offset points",
             xytext=(0, -25), fontsize=7, color="#FC8D59", ha="center", fontstyle="italic")
ax1.annotate("collapse", (6, 1.0), textcoords="offset points",
             xytext=(0, -25), fontsize=7, color="#B2182B", ha="center", fontstyle="italic")

# Shade V10 as final
ax1.axvspan(3.5, 6.5, alpha=0.04, color="gray")

ax1.set_xticks(x)
ax1.set_xticklabels(versions, fontsize=7)
ax1.grid(axis="y", alpha=0.15)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=7.5,
           framealpha=0.9, edgecolor="gray", fancybox=False)

fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(FIG_DIR, "version_progression.png"), bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Saved: figures/version_progression.png")

print("\nAll figures generated successfully.")
print(f"Output directory: {FIG_DIR}")

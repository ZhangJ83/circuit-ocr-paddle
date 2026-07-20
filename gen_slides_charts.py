"""Generate charts for Beamer presentation."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

OUT = r'g:\mimo_project\circuit_ocr\slides_figures'
os.makedirs(OUT, exist_ok=True)

# Style
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05
})

# ── Chart 1: CompF1 comparison bar chart ──
fig, ax = plt.subplots(figsize=(8, 4.5))
experiments = ['exp1\nBaseline\n384px', 'exp2\nHiRes\n512px', 'exp3\nAnti-Overfit\n384px',
               'exp4\nUnfrozen\n384px', 'exp5\nAnti-Overfit\n+Synth', 'exp6\nBaseline\n+Synth']
comp_f1 = [0.037, 0.058, 0.028, 0.092, 0.126, 0.156]
joint_f1 = [0.005, 0.000, 0.000, 0.005, 0.011, 0.011]
colors = ['#4472C4','#4472C4','#4472C4','#4472C4','#ED7D31','#ED7D31']

x = np.arange(len(experiments))
w = 0.35
bars1 = ax.bar(x - w/2, comp_f1, w, label='Component F1', color=colors, edgecolor='black', linewidth=0.5)
# Add value labels
for bar, val in zip(bars1, comp_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, f'{val:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Component F1 Score')
ax.set_xticks(x)
ax.set_xticklabels(experiments, fontsize=8)
ax.set_ylim(0, 0.22)
ax.axhline(y=0.206, color='red', linestyle='--', linewidth=1, label='V10-Fixed S600 (0.206)')
ax.legend(loc='upper left', fontsize=9)
ax.set_title('Component F1: Phase 1 vs Phase 2 (+Synthetic Data)', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.axvline(x=3.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax.text(3.7, 0.20, '← Phase 1 | Phase 2 →', fontsize=8, color='gray', style='italic')
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'comp_f1_comparison.pdf'))
fig.savefig(os.path.join(OUT, 'comp_f1_comparison.png'))
plt.close()
print("Chart 1 done: comp_f1_comparison")

# ── Chart 2: Joint F1 comparison ──
fig, ax = plt.subplots(figsize=(8, 4.5))
bars2 = ax.bar(x - w/2, joint_f1, w, label='Joint F1 (Refdes+Value)', color=colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars2, joint_f1):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('Joint F1 Score')
ax.set_xticks(x)
ax.set_xticklabels(experiments, fontsize=8)
ax.set_ylim(0, 0.025)
ax.axhline(y=0.019, color='red', linestyle='--', linewidth=1, label='V10-Fixed S600 (0.019)')
ax.legend(loc='upper left', fontsize=9)
ax.set_title('Joint F1 (Refdes + Value pairs): Phase 1 vs Phase 2', fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.axvline(x=3.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'joint_f1_comparison.pdf'))
fig.savefig(os.path.join(OUT, 'joint_f1_comparison.png'))
plt.close()
print("Chart 2 done: joint_f1_comparison")

# ── Chart 3: Training evolution (exp3 validation quality) ──
fig, ax = plt.subplots(figsize=(8, 3.5))
steps = [200, 400, 600, 800]
exp3_quality = [0, 1, 3, 4]  # subjective quality score
exp5_quality = [4, 4, 4, 3]

ax.plot(steps, exp3_quality, 'o-', color='#4472C4', linewidth=2, markersize=8, label='exp3 (Anti-Overfit, no synth)')
ax.plot(steps, exp5_quality, 's-', color='#ED7D31', linewidth=2, markersize=8, label='exp5 (Anti-Overfit + Synthetic Data)')
ax.set_xlabel('Training Steps')
ax.set_ylabel('Validation Quality (0-4)')
ax.set_title('Training Evolution: Validation Inference Quality', fontweight='bold')
ax.set_ylim(-0.5, 5)
ax.set_yticks([0, 1, 2, 3, 4])
ax.set_yticklabels(['0: Collapsed\n(counting)', '1: Fuzzy text', '2: Partial\nwords', '3: Component\nlabels', '4: Accurate\nvalues'])
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'training_evolution.pdf'))
fig.savefig(os.path.join(OUT, 'training_evolution.png'))
plt.close()
print("Chart 3 done: training_evolution")

# ── Chart 4: Synthetic data ablation ──
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# Subplot A: CompF1
ax = axes[0]
cats = ['Baseline\n(lr=2e-5)', 'Anti-Overfit\n(lr=1e-5, do=0.1)']
no_synth = [0.037, 0.028]
with_synth = [0.156, 0.126]
x2 = np.arange(len(cats))
w2 = 0.3
ax.bar(x2 - w2/2, no_synth, w2, label='Original', color='#4472C4', edgecolor='black', linewidth=0.5)
ax.bar(x2 + w2/2, with_synth, w2, label='+Synthetic Data', color='#ED7D31', edgecolor='black', linewidth=0.5)
for i, (v1, v2) in enumerate(zip(no_synth, with_synth)):
    ax.text(i - w2/2, v1 + 0.003, f'{v1:.3f}', ha='center', fontsize=8)
    ax.text(i + w2/2, v2 + 0.003, f'{v2:.3f}', ha='center', fontsize=8)
ax.set_xticks(x2); ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel('Component F1'); ax.set_title('CompF1: Before vs After', fontweight='bold')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

# Subplot B: JointF1
ax = axes[1]
j_no = [0.005, 0.000]
j_with = [0.011, 0.011]
ax.bar(x2 - w2/2, j_no, w2, label='Original', color='#4472C4', edgecolor='black', linewidth=0.5)
ax.bar(x2 + w2/2, j_with, w2, label='+Synthetic Data', color='#ED7D31', edgecolor='black', linewidth=0.5)
for i, (v1, v2) in enumerate(zip(j_no, j_with)):
    if v1 > 0: ax.text(i - w2/2, v1 + 0.0003, f'{v1:.3f}', ha='center', fontsize=8)
    if v2 > 0: ax.text(i + w2/2, v2 + 0.0003, f'{v2:.3f}', ha='center', fontsize=8)
ax.set_xticks(x2); ax.set_xticklabels(cats, fontsize=9)
ax.set_ylabel('Joint F1'); ax.set_title('JointF1: Before vs After', fontweight='bold')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)

fig.suptitle('Synthetic Data Ablation: 300 Text-Image Pairs Mixed at 20%', fontweight='bold', fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'synth_ablation.pdf'))
fig.savefig(os.path.join(OUT, 'synth_ablation.png'))
plt.close()
print("Chart 4 done: synth_ablation")

# ── Chart 5: Pipeline overview ──
fig, ax = plt.subplots(figsize=(8, 3))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')

boxes = [
    (0.5, 3, 2, 2, 'Data\n1200 Circuit\n+ 300 Synth', '#D6E4F0'),
    (2.8, 3, 2, 2, 'PaddleOCR-VL\n0.9B\n+ LoRA r=16', '#D6E4F0'),
    (5.1, 3, 2, 2, 'Training\nManual CE Loss\nGradAccum=4', '#D6E4F0'),
    (7.4, 3, 2, 2, 'Evaluation\nCompF1 + JointF1\n+ RepRate + NED', '#FFF2CC'),
]
for xb, yb, wb, hb, text, color in boxes:
    rect = plt.Rectangle((xb, yb), wb, hb, fill=True, facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(xb + wb/2, yb + hb/2, text, ha='center', va='center', fontsize=9, fontweight='bold')

# Arrows
for xa in [2.5, 4.8, 7.1]:
    ax.annotate('', xy=(xa + 0.3, 4), xytext=(xa, 4), arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

ax.set_title('Training Pipeline Overview', fontweight='bold', fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'pipeline_overview.pdf'))
fig.savefig(os.path.join(OUT, 'pipeline_overview.png'))
plt.close()
print("Chart 5 done: pipeline_overview")

# ── Chart 6: NED comparison ──
fig, ax = plt.subplots(figsize=(8, 3.5))
ned_values = [0.944, 0.941, 0.940, 0.944, 0.941, 0.944]
bars3 = ax.bar(x, ned_values, 0.5, color=colors, edgecolor='black', linewidth=0.5)
for bar, val in zip(bars3, ned_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f'{val:.3f}',
            ha='center', va='bottom', fontsize=8)
ax.set_ylabel('Normalized Edit Distance (↓)')
ax.set_xticks(x)
ax.set_xticklabels(['exp1','exp2','exp3','exp4','exp5','exp6'], fontsize=9)
ax.set_title('NED (Lower is Better)', fontweight='bold')
ax.set_ylim(0.92, 0.96)
ax.grid(axis='y', alpha=0.3)
ax.axvline(x=3.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'ned_comparison.pdf'))
fig.savefig(os.path.join(OUT, 'ned_comparison.png'))
plt.close()
print("Chart 6 done: ned_comparison")

print(f"\nAll charts saved to: {OUT}")

"""Dataset visualization charts — all stats first, then all charts."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json, os, re
from collections import Counter

SRC = r'g:\mimo_project\circuit_ocr'
DST = r'g:\mimo_project\circuit_ocr_dataset_final'
OUT = r'g:\mimo_project\circuit_ocr\slides_figures'
os.makedirs(OUT, exist_ok=True)

CB = {'blue':'#0072B2','orange':'#E69F00','green':'#009E73','red':'#D55E00',
      'purple':'#CC79A7','sky':'#56B4E9'}
SYSU_BLUE = '#003366'; SYSU_RED = '#B41E1E'
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.titlesize':12,
    'figure.dpi':200,'savefig.dpi':200,'savefig.bbox':'tight'})

# ════════════ LOAD & COMPUTE ALL STATS ════════════
with open(os.path.join(SRC, 'output', 'test_clean.jsonl'), encoding='utf-8') as f:
    test = [json.loads(l) for l in f if l.strip()]
n = len(test)
re_comp = re.compile(r'\b((?:LED|[RCDLQUJYF])\d+)\b')

comps_per_sample = []; comp_types_all = Counter(); label_lens = []; line_counts = []
for s in test:
    label = s['messages'][1]['content']
    cs = re_comp.findall(label)
    comps_per_sample.append(len(cs))
    label_lens.append(len(label))
    line_counts.append(len(label.split('\n')))
    for c in cs:
        comp_types_all[re.match(r'[A-Z]+', c).group()] += 1

types_ordered = sorted(comp_types_all.items(), key=lambda x: x[1], reverse=True)
type_names = [t[0] for t in types_ordered]
type_counts = [t[1] for t in types_ordered]
total_comps = sum(type_counts)
type_labels = {'R':'Resistors','C':'Capacitors','D':'Diodes','U':'ICs',
    'J':'Connectors','Q':'Transistors','L':'Inductors','LED':'LEDs','F':'Fuses','Y':'Crystals'}

# Difficulty labels
ocr_diff_labels = []; vis_diff_labels = []
test_diff_path = os.path.join(DST, 'dataset_a', 'test.jsonl')
if os.path.exists(test_diff_path):
    with open(test_diff_path, encoding='utf-8') as f:
        test_diff = [json.loads(l) for l in f if l.strip()]
    if 'difficulty' in test_diff[0]:
        ocr_diff_labels = [s['difficulty']['ocr_based'] for s in test_diff]
        vis_diff_labels = [s['difficulty']['visual'] for s in test_diff]

print("Stats computed. n={}, total_comps={}".format(n, total_comps))

# ════════════ CHART 1: Donut Composition ════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ax = axes[0]
ax.pie([1520,150,150], labels=['Train\n1,520','Test\n150','Val\n150'],
    autopct='%1.0f%%', colors=[CB['blue'],CB['orange'],CB['green']],
    startangle=90, wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
    textprops={'fontsize':11,'fontweight':'bold'})
ax.set_title('Data Split (n=1,820)', fontweight='bold', fontsize=13)

ax = axes[1]
ax.pie([1200,300,20], labels=['KiCad\n1,200','Synth Text\n300','Real Photos\n20'],
    autopct='%1.0f%%', colors=[CB['blue'],CB['purple'],CB['red']],
    startangle=90, wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
    textprops={'fontsize':10,'fontweight':'bold'})
ax.set_title('Training Sources', fontweight='bold', fontsize=13)
fig.suptitle('Dataset A Composition', fontweight='bold', fontsize=15, y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'dataset_composition.pdf'))
fig.savefig(os.path.join(OUT, 'dataset_composition.png'))
plt.close()
print("1/8: Composition donuts")

# ════════════ CHART 2: Component Type Bars ════════════
fig, ax = plt.subplots(figsize=(9, 5))
colors_bar = [CB['blue'],CB['orange'],CB['green'],CB['red'],CB['purple'],
              CB['sky'],'#F0E442',CB['orange'],'#999','#666']
bars = ax.barh(range(len(type_names)), type_counts,
    color=[colors_bar[i%10] for i in range(len(type_names))],
    edgecolor='black', linewidth=0.5, height=0.7)
for bar, cnt in zip(bars, type_counts):
    ax.text(bar.get_width()+20, bar.get_y()+bar.get_height()/2,
        f'{cnt:,} ({cnt/total_comps*100:.1f}%)', va='center', fontsize=10, fontweight='bold')
ax.set_yticks(range(len(type_names)))
ax.set_yticklabels([f'{n} ({type_labels.get(n,n)})' for n in type_names], fontsize=10)
ax.set_xlim(0, max(type_counts)*1.25)
ax.set_xlabel('OCR Instances (Test Set)', fontsize=11)
ax.set_title(f'Component Type Distribution (n={total_comps:,})', fontweight='bold', fontsize=13)
ax.invert_yaxis(); ax.grid(axis='x', alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'component_types.pdf'))
fig.savefig(os.path.join(OUT, 'component_types.png'))
plt.close()
print("2/8: Component types")

# ════════════ CHART 3: OCR Distribution Histogram ════════════
fig, ax = plt.subplots(figsize=(9, 4.5))
bins = np.arange(0, max(comps_per_sample)+10, 10)
ax.hist(comps_per_sample, bins=bins, color=CB['blue'], edgecolor='white', alpha=0.8)
ax.axvspan(0,15,alpha=0.08,color='green',label='Easy (<15)')
ax.axvspan(15,35,alpha=0.08,color='orange',label='Medium (15-35)')
ax.axvspan(35,max(comps_per_sample)+5,alpha=0.08,color='red',label='Hard (>35)')
mean_v = np.mean(comps_per_sample)
ax.axvline(mean_v, color=SYSU_RED, linestyle='--', linewidth=2, label=f'Mean: {mean_v:.1f}')
ax.set_xlabel('OCR Instances per Sample'); ax.set_ylabel('Samples')
ax.set_title('OCR Instance Distribution (Test Set, n=150)', fontweight='bold', fontsize=13)
ax.legend(fontsize=9, framealpha=0.9); ax.grid(axis='y', alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'ocr_distribution.pdf'))
fig.savefig(os.path.join(OUT, 'ocr_distribution.png'))
plt.close()
print("3/8: OCR distribution")

# ════════════ CHART 4: Radar Scores ════════════
fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
cats = ['1.1 Data\nScale', '1.2 Annotation\nAccuracy', '1.3 Data\nDiversity', '1.4 Difficulty\nRationality']
scores = [4.0, 4.5, 2.5, 4.5]
N = len(cats)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
scores_plot = scores + [scores[0]]
ax.fill(angles, scores_plot, alpha=0.15, color=CB['blue'])
ax.plot(angles, scores_plot, 'o-', linewidth=2.5, color=CB['blue'], markersize=10,
    markerfacecolor=CB['blue'], markeredgecolor='white', markeredgewidth=2)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(cats, fontsize=11, fontweight='bold')
ax.set_ylim(0, 5.5); ax.set_yticks([1,2,3,4,5])
ax.set_title('Dataset A: 4-Dimension Evaluation (15.5/20)', fontweight='bold', fontsize=14, pad=25)
ax.grid(True, alpha=0.3)
for angle, score in zip(angles[:-1], scores):
    ax.annotate(f'{score}/5', xy=(angle, score), xytext=(angle, score+0.6),
        fontsize=12, fontweight='bold', ha='center', color=CB['blue'])
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'radar_scores.pdf'))
fig.savefig(os.path.join(OUT, 'radar_scores.png'))
plt.close()
print("4/8: Radar scores")

# ════════════ CHART 5: Difficulty Heatmap ════════════
if ocr_diff_labels and vis_diff_labels:
    cross = Counter()
    for o, v in zip(ocr_diff_labels, vis_diff_labels):
        cross[(o, v)] += 1
    levels = ['easy', 'medium', 'hard']
    matrix = np.zeros((3, 3))
    for i, o in enumerate(levels):
        for j, v in enumerate(levels):
            matrix[i, j] = cross.get((o, v), 0)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{matrix[i,j]:.0f}\n({matrix[i,j]/n*100:.0f}%)',
                ha='center', va='center', fontsize=13, fontweight='bold',
                color='white' if matrix[i,j]>n*0.15 else 'black')
    ax.set_xticks(range(3)); ax.set_xticklabels(['Easy','Medium','Hard'], fontsize=12)
    ax.set_yticks(range(3)); ax.set_yticklabels(['Easy','Medium','Hard'], fontsize=12)
    ax.set_xlabel('Visual Difficulty', fontsize=12, fontweight='bold')
    ax.set_ylabel('OCR-Based Difficulty', fontsize=12, fontweight='bold')
    ax.set_title('Difficulty Cross-Tabulation (n=150)', fontweight='bold', fontsize=13)
    plt.colorbar(im, ax=ax, label='Samples', shrink=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'difficulty_heatmap.pdf'))
    fig.savefig(os.path.join(OUT, 'difficulty_heatmap.png'))
    plt.close()
    print("5/8: Difficulty heatmap")

# ════════════ CHART 6: Label Length ════════════
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(label_lens, bins=30, color=CB['blue'], edgecolor='white', alpha=0.8)
ax.axvline(np.mean(label_lens), color=SYSU_RED, linestyle='--', linewidth=2,
    label=f'Mean: {np.mean(label_lens):.0f}')
ax.axvline(np.median(label_lens), color=SYSU_BLUE, linestyle=':', linewidth=2,
    label=f'Median: {np.median(label_lens):.0f}')
ax.set_xlabel('Label Length (characters)'); ax.set_ylabel('Samples')
ax.set_title('Label Length Distribution', fontweight='bold', fontsize=13)
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'label_length_dist.pdf'))
fig.savefig(os.path.join(OUT, 'label_length_dist.png'))
plt.close()
print("6/8: Label length")

# ════════════ CHART 7: Stacked Component Bar ════════════
fig, ax = plt.subplots(figsize=(10, 3.5))
bar_colors_v = plt.cm.viridis(np.linspace(0.15, 0.9, len(type_names)))
for i in range(len(type_names)):
    ax.barh(0, type_counts[i], left=sum(type_counts[:i]), height=0.6,
        color=bar_colors_v[i], edgecolor='white', linewidth=1.5,
        label=f'{type_names[i]} ({type_counts[i]/total_comps*100:.1f}%)')
ax.set_yticks([])
ax.set_xlabel('OCR Instances'); ax.set_title(f'Component Types (n={total_comps:,})', fontweight='bold', fontsize=13)
ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.3), ncol=5, fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'component_stacked.pdf'))
fig.savefig(os.path.join(OUT, 'component_stacked.png'))
plt.close()
print("7/8: Component stacked")

# ════════════ CHART 8: Lines per sample ════════════
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(line_counts, bins=25, color=CB['green'], edgecolor='white', alpha=0.8)
ax.axvline(np.mean(line_counts), color=SYSU_RED, linestyle='--', linewidth=2,
    label=f'Mean: {np.mean(line_counts):.0f} lines')
ax.set_xlabel('Lines per Sample'); ax.set_ylabel('Samples')
ax.set_title('Structural Complexity: Lines per Sample', fontweight='bold', fontsize=13)
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.2)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'line_count_dist.pdf'))
fig.savefig(os.path.join(OUT, 'line_count_dist.png'))
plt.close()
print("8/8: Line count")
print(f"\nALL DONE -> {OUT}")

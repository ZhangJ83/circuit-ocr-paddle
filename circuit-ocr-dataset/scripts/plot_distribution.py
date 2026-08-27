#!/usr/bin/env python3
import matplotlib.pyplot as plt
from pathlib import Path

# Setup clean design style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# 1. Dataset Source Distribution (Pie Chart)
sources = ['Open Schematics', 'Synthetic', 'Masala-CHAI']
counts = [2690, 580, 207]
colors_pie = ['#3f88c5', '#f19a3e', '#84b082']

wedges, texts, autotexts = ax1.pie(
    counts, 
    labels=sources, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=colors_pie, 
    textprops=dict(color="black", fontsize=11, weight="bold"),
    pctdistance=0.75,
    wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)  # Donut chart style
)

# Customize text properties inside the pie chart
for autotext in autotexts:
    autotext.set_fontsize(10)
    autotext.set_color('black')

ax1.set_title("Dataset Source Distribution", fontsize=14, weight='bold', pad=20)

# 2. Split Distribution (Bar Chart)
splits = ['Train Set', 'Val Set', 'Test Set']
split_counts = [2433, 521, 523]
colors_bar = ['#5b5f97', '#ffc09f', '#ffee93']

bars = ax2.bar(splits, split_counts, color=colors_bar, edgecolor='grey', linewidth=1, width=0.5)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width()/2.0, 
        height + 50, 
        f'{height}\n({height/sum(split_counts)*100:.1f}%)', 
        ha='center', 
        va='bottom', 
        fontsize=11, 
        weight='bold'
    )

ax2.set_title("Train / Val / Test Split", fontsize=14, weight='bold', pad=20)
ax2.set_ylabel("Number of Samples", fontsize=12)
ax2.set_ylim(0, 2800)
ax2.tick_params(axis='both', labelsize=11)

# Overall Title and layout adjustments
plt.suptitle("Circuit OCR Dataset Distribution & Splitting", fontsize=16, weight='bold', y=0.98)
plt.tight_layout()

# Ensure output directory exists
output_path = Path("docs/dataset_distribution.png")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Save the figure
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Chart successfully generated and saved to {output_path}")

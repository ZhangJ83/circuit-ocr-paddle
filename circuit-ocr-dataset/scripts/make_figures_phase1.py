import json, os, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DATASET_DIR = r'G:\mimo_project\circuit_ocr\circuit-ocr-dataset'
OUT_DIR = f'{DATASET_DIR}/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Figure 1: Phase 1 Multi-Metric Bar Chart
# ============================================================
def make_phase1_metrics_chart():
    """Bar chart comparing Base, S400, S600, S800 across all 8 metrics."""
    print('Generating Phase 1 multi-metric bar chart...')

    models = ['Base', 'S400', 'S600', 'S800']
    colors = ['#d4d4d4', '#87CEEB', '#1f77b4', '#ff7f0e']

    # Data: [CompF1, CompPrec, CompRec, TokenRec, NED, RepRate, Diversity]
    # Note: ExactMatch is 0 for all, skip it. NED lower is better.
    metrics_data = {
        'CompF1':      [0.0455, 0.1820, 0.2061, 0.2080],
        'CompPrec':    [0.0455, 0.1862, 0.2024, 0.2862],
        'CompRec':     [0.0455, 0.2501, 0.3114, 0.1996],
        'TokenRec':    [0.0016, 0.1302, 0.1540, 0.1191],
        'NED (↓)':     [0.9296, 0.8298, 0.8031, 0.8063],
        'RepRate':     [0.068,  0.205,  0.159,  0.409],
        'Diversity':   [0.909,  0.955,  0.909,  0.932],
    }

    # Special: NED and RepRate are "lower is better"
    lower_better = {'NED (↓)': True, 'RepRate': True}

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    metric_names = list(metrics_data.keys())

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        values = metrics_data[metric]
        is_lower = lower_better.get(metric, False)

        bars = ax.bar(models, values, color=colors, edgecolor='white', linewidth=0.8)

        # Highlight S600
        bars[2].set_edgecolor('#003366')
        bars[2].set_linewidth(2.5)

        # Value labels on bars
        for bar, val in zip(bars, values):
            if val < 0.01:
                label = f'{val:.4f}'
            elif val < 1.0:
                label = f'{val:.3f}'
            else:
                label = f'{val:.1f}'
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    label, ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=7)

        if is_lower:
            # Add a note
            ax.text(0.5, -0.18, '(lower is better)', transform=ax.transAxes,
                    ha='center', fontsize=7, color='#888888', style='italic')

        # S600 star marker
        ax.annotate('★', xy=(2, ax.get_ylim()[1]*0.92), fontsize=14, ha='center', color='#003366')

    # Hide the 8th subplot (empty)
    axes[7].set_visible(False)

    fig.suptitle('Phase 1 Multi-Metric Benchmark: Base vs S400 vs S600 vs S800 (easy50-pure, 44 samples)',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.text(0.5, -0.02, '★ = Best Checkpoint (S600)  |  ExactMatch = 0% for all models (not shown)',
             ha='center', fontsize=9, color='#555555')

    plt.tight_layout()
    out = f'{OUT_DIR}/phase1_metrics_chart.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {out}')
    return out


# ============================================================
# Figure 2: Updated Model Comparison with S600 predictions
# ============================================================
def load_font(size=16):
    for fp in ['C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simsun.ttc',
               'C:/Windows/Fonts/simhei.ttf','C:/Windows/Fonts/arial.ttf']:
        if os.path.exists(fp):
            try: return ImageFont.truetype(fp, size)
            except: pass
    return ImageFont.load_default()

def draw_text_box(draw, text, x, y, max_w, font, color):
    lines = []
    for line in text.split('\n'):
        if not line.strip(): lines.append(' '); continue
        cpl = max(1, int(max_w / (font.size * 0.55)))
        for k in range(0, len(line), cpl):
            lines.append(line[k:k+cpl])
    for j, line in enumerate(lines[:28]):
        draw.text((x, y + j * (font.size + 2)), line, fill=color, font=font)

def make_model_comparison_v6():
    """Generate model comparison: Original | GT | Base Model | V10-Fixed S600"""
    print('Generating V10-Fixed model comparison...')

    # Load base and S600 results
    base_file = f'{DATASET_DIR}/results_base_easy50.jsonl'
    s600_file = f'{DATASET_DIR}/scripts/results_v3_lora_s600_easy50.json'

    base_data = {}
    with open(base_file, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                base_data[d['images'][0]] = d

    with open(s600_file, encoding='utf-8') as f:
        s600_results = json.load(f)

    # Build S600 prediction lookup
    s600_preds = {}
    for r in s600_results.get('results', s600_results.get('details', [])):
        img = r.get('image', r.get('images', [''])[0])
        pred = r.get('prediction', '')
        s600_preds[img] = pred

    # Showcase samples
    common = [
        './data/test/benjiaomodular_sot2dip.png',
        './data/test/Pari55051_cat-pcb.png',
        './data/test/rh1tech_echo.png'
    ]

    rows, cols = len(common), 4
    cell_w, cell_h = 240, 350
    canvas = Image.new('RGB', (cols*cell_w, rows*cell_h+35), (255,255,255))
    font = load_font(12)
    font_s = load_font(8.5)
    draw = ImageDraw.Draw(canvas)

    headers = ['Original Image', 'Ground Truth', 'Base Model', 'V10-Fixed S600']
    for j, h in enumerate(headers):
        draw.text((j*cell_w+10, 8), h, fill=(0,0,0), font=font)

    for i, img_key in enumerate(common):
        y_base = 35 + i*cell_h
        img_rel = img_key.lstrip('./')
        img_path = f'{DATASET_DIR}/{img_rel}'
        if not os.path.exists(img_path):
            alt = img_path.replace('.png','.jpg').replace('.JPG','.jpg')
            if os.path.exists(alt): img_path = alt

        try:
            img = Image.open(img_path).convert('RGB')
            img.thumbnail((cell_w-20, cell_h-40), Image.LANCZOS)
            canvas.paste(img, (10, y_base+10))
        except Exception as e:
            draw.text((10, y_base+20), f'ERR:{e}', fill=(255,0,0), font=font_s)

        # GT
        gt = base_data.get(img_key, {}).get('label',
             base_data.get(img_key, {}).get('messages', [{},{}])[1].get('content', 'N/A'))

        # Base prediction
        base_pred = base_data.get(img_key, {}).get('prediction', 'N/A')

        # S600 prediction
        s600_pred = s600_preds.get(img_key, s600_preds.get(img_rel, 'N/A'))

        draw_text_box(draw, str(gt)[:150], cell_w+10, y_base+10, cell_w-20, font_s, (0,100,0))
        draw_text_box(draw, str(base_pred)[:150], 2*cell_w+10, y_base+10, cell_w-20, font_s, (180,0,0))
        draw_text_box(draw, str(s600_pred)[:150], 3*cell_w+10, y_base+10, cell_w-20, font_s, (0,0,180))
        draw.line([(0, y_base+cell_h-1), (cols*cell_w, y_base+cell_h-1)], fill=(200,200,200))

    out = f'{OUT_DIR}/model_comparison_v6.png'
    canvas.save(out, quality=95)
    print(f'  Saved: {out}')
    return out


if __name__ == '__main__':
    make_phase1_metrics_chart()
    make_model_comparison_v6()
    print('Done!')

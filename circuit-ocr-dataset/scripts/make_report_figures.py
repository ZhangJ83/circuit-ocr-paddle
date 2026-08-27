"""Generate two key figures for the technical report:
1. dataset_overview.png - 3x3 grid showing all data sources
2. model_comparison.png - 3 examples: original, GT, base output, S600 output
"""

import json, os
from PIL import Image, ImageDraw, ImageFont
import textwrap

DATASET_DIR = r'G:\mimo_project\circuit_ocr\circuit-ocr-dataset'
OUT_DIR = f'{DATASET_DIR}/figures'
os.makedirs(OUT_DIR, exist_ok=True)

def load_font(size=12):
    for fp in ['C:/Windows/Fonts/arial.ttf','C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simsun.ttc']:
        if os.path.exists(fp):
            try: return ImageFont.truetype(fp, size)
            except: pass
    return ImageFont.load_default()

def resolve_image(rel_path):
    """Resolve relative image path to absolute."""
    # rel_path may be like './data/test/11.jpg' or 'data/train/analog_med_0011.png'
    rel_path = rel_path.lstrip('./')
    full = os.path.join(DATASET_DIR, rel_path)
    if os.path.exists(full):
        return full
    # Try alternate extensions
    for ext in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']:
        alt = os.path.splitext(full)[0] + ext
        if os.path.exists(alt):
            return alt
    return None

def load_image_safe(path, max_size):
    """Load and resize image safely."""
    try:
        img = Image.open(path).convert('RGB')
        img.thumbnail(max_size, Image.LANCZOS)
        return img
    except Exception as e:
        print(f"  WARN: Cannot load {path}: {e}")
        return None


# ===========================================================================
# FIGURE 1: Dataset Overview (3x3 grid)
# ===========================================================================
print("=" * 60)
print("FIGURE 1: Dataset Overview")
print("=" * 60)

# Load training data
with open(f'{DATASET_DIR}/ocr_vl_sft-train-v9-pure.jsonl', encoding='utf-8') as f:
    train_samples = [json.loads(l) for l in f if l.strip()]

# Load test data (easy50-pure, 44 samples)
with open(f'{DATASET_DIR}/ocr_vl_sft-test-easy50-pure.jsonl', encoding='utf-8') as f:
    test_samples = [json.loads(l) for l in f if l.strip()]

# Separate training by source
synth_samples = []
real_samples = []
for s in train_samples:
    img = s['images'][0].lower()
    if 'synthetic' in img:
        synth_samples.append(s)
    else:
        real_samples.append(s)

print(f'Training: {len(synth_samples)} synthetic + {len(real_samples)} real')
print(f'Test: {len(test_samples)} samples')

# Pick 3 from each
selected_synth = synth_samples[:3]
selected_real = real_samples[:3]
selected_test = test_samples[:3]

# Build 3x3 grid
cols, rows = 3, 3
cell_w, cell_h = 280, 300
margin = 15

canvas_w = cols * cell_w + margin * 2
canvas_h = rows * cell_h + margin * 2 + 30  # extra for row labels
canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)
font_hdr = load_font(14)
font_label = load_font(10)
font_gt = load_font(8)

# Row labels
row_labels = ['Synthetic V3 (Training)', 'train-real KiCad (Training)', 'Test (easy50-pure)']

all_selected = [
    (selected_synth, row_labels[0], (0, 51, 153)),
    (selected_real, row_labels[1], (0, 100, 0)),
    (selected_test, row_labels[2], (180, 0, 0)),
]

for row_idx, (samples, label, color) in enumerate(all_selected):
    # Row header
    draw.text((margin, margin + row_idx * cell_h), label, fill=color, font=font_hdr)

    for col_idx, sample in enumerate(samples):
        x0 = margin + col_idx * cell_w
        y0 = margin + 20 + row_idx * cell_h

        # Load image
        img_rel = sample['images'][0]
        img_path = resolve_image(img_rel)

        if img_path:
            img = load_image_safe(img_path, (cell_w - 20, cell_h - 80))
            if img:
                img_x = x0 + (cell_w - img.width) // 2
                img_y = y0 + 5
                canvas.paste(img, (img_x, img_y))
                img_bottom = img_y + img.height
            else:
                draw.text((x0 + 5, y0 + 5), 'Image not found', fill=(255, 0, 0), font=font_label)
                img_bottom = y0 + 25
        else:
            draw.text((x0 + 5, y0 + 5), f'Missing: {img_rel}', fill=(255, 0, 0), font=font_label)
            img_bottom = y0 + 25

        # GT label
        gt_text = sample['messages'][1]['content']
        gt_lines = gt_text.split('\n')[:8]
        gt_display = '\n'.join(gt_lines)
        if len(gt_text.split('\n')) > 8:
            gt_display += '\n...'

        gt_y = max(img_bottom + 4, y0 + cell_h - 70)
        for j, line in enumerate(gt_display.split('\n')[:9]):
            draw.text((x0 + 3, gt_y + j * 10), line, fill=(80, 80, 80), font=font_gt)

out_path = f'{OUT_DIR}/dataset_overview.png'
canvas.save(out_path, quality=95)
print(f'Saved: {out_path}')


# ===========================================================================
# FIGURE 2: Model Comparison (3 examples: Image | GT | Base | S600)
# ===========================================================================
print("\n" + "=" * 60)
print("FIGURE 2: Model Comparison")
print("=" * 60)

# Load base model results
with open(f'{DATASET_DIR}/results_base_easy50.jsonl', encoding='utf-8') as f:
    base_results = [json.loads(l) for l in f if l.strip()]

# Load S600 results (v9_final = S600 checkpoint)
with open(f'{DATASET_DIR}/results_v9_final_easy50_pure.jsonl', encoding='utf-8') as f:
    s600_results = [json.loads(l) for l in f if l.strip()]

print(f'Base results: {len(base_results)} entries')
print(f'S600 results: {len(s600_results)} entries')

# Build lookup by image path
base_by_img = {}
for r in base_results:
    img = r['images'][0] if isinstance(r['images'], list) else r['images']
    base_by_img[img] = r

s600_by_img = {}
for r in s600_results:
    img = r['images'][0] if isinstance(r['images'], list) else r['images']
    s600_by_img[img] = r

# Find test samples that exist in BOTH base and S600 results
common_imgs = []
for s in test_samples:
    img = s['images'][0]
    if img in base_by_img and img in s600_by_img:
        common_imgs.append(s)

print(f'Common test samples (in both base and S600): {len(common_imgs)}')

# Pick 3 diverse examples: pick ones where S600 is better than base but not perfect
def ned(pred, gt):
    """Simple NED calculation."""
    if not pred or not gt:
        return 1.0
    if len(gt) == 0:
        return 1.0 if len(pred) > 0 else 0.0
    # Levenshtein distance
    m, n = len(pred), len(gt)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if pred[i-1] == gt[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[m][n] / max(m, n)

# Score each common sample
scored = []
for s in common_imgs:
    img = s['images'][0]
    gt = s['messages'][1]['content']
    base_pred = base_by_img[img]['prediction']
    s600_pred = s600_by_img[img]['prediction']
    base_ned = ned(base_pred, gt)
    s600_ned = ned(s600_pred, gt)
    improvement = base_ned - s600_ned
    scored.append((improvement, s, base_pred, s600_pred, base_ned, s600_ned))

# Sort by improvement (largest improvement first) and pick 3
scored.sort(key=lambda x: x[0], reverse=True)
selected = scored[:3]

print(f'\nSelected 3 examples for comparison:')
for i, (imp, s, base_pred, s600_pred, base_ned, s600_ned) in enumerate(selected):
    img = s['images'][0]
    gt = s['messages'][1]['content'][:60].replace('\n', ' | ')
    print(f'  {i+1}. {img}')
    print(f'     GT: {gt}...')
    print(f'     Base NED={base_ned:.4f}, S600 NED={s600_ned:.4f}, Δ={imp:.4f}')

# Build comparison figure
cols = 4  # Image | GT | Base Output | S600 Output
rows = len(selected)
cell_w, cell_h = 220, 280
margin = 15
header_h = 25

canvas_w = cols * cell_w + margin * 2
canvas_h = rows * cell_h + margin * 2 + header_h
canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

# Column headers
col_headers = ['Original Image', 'Ground Truth', 'Base Model Output', 'S600 Output']
for j, hdr in enumerate(col_headers):
    x = margin + j * cell_w + 5
    draw.text((x, margin), hdr, fill=(0, 0, 0), font=font_hdr)

for row_idx, (imp, s, base_pred, s600_pred, base_ned, s600_ned) in enumerate(selected):
    y0 = margin + header_h + row_idx * cell_h

    # Column 1: Original Image
    img_rel = s['images'][0]
    img_path = resolve_image(img_rel)
    if img_path:
        img = load_image_safe(img_path, (cell_w - 20, cell_h - 20))
        if img:
            img_x = margin + (cell_w - img.width) // 2
            img_y = y0 + (cell_h - img.height) // 2
            canvas.paste(img, (img_x, img_y))

    # Column 2: GT
    gt_text = s['messages'][1]['content']
    gt_lines = gt_text.split('\n')[:15]
    for j, line in enumerate(gt_lines):
        draw.text((margin + cell_w + 5, y0 + 8 + j * 12), line, fill=(0, 100, 0), font=font_gt)

    # Column 3: Base Model Output (red)
    base_lines = base_pred.split('\n')[:15]
    for j, line in enumerate(base_lines):
        text = line[:30]
        draw.text((margin + 2*cell_w + 5, y0 + 8 + j * 12), text, fill=(180, 0, 0), font=font_gt)
    # Show NED
    draw.text((margin + 2*cell_w + 5, y0 + cell_h - 15), f'NED={base_ned:.4f}', fill=(180, 0, 0), font=font_label)

    # Column 4: S600 Output (blue)
    s600_lines = s600_pred.split('\n')[:15]
    for j, line in enumerate(s600_lines):
        text = line[:30]
        draw.text((margin + 3*cell_w + 5, y0 + 8 + j * 12), text, fill=(0, 0, 180), font=font_gt)
    # Show NED
    draw.text((margin + 3*cell_w + 5, y0 + cell_h - 15), f'NED={s600_ned:.4f}', fill=(0, 0, 180), font=font_label)

    # Horizontal separator
    if row_idx < rows - 1:
        draw.line([(margin, y0 + cell_h), (canvas_w - margin, y0 + cell_h)], fill=(200, 200, 200), width=1)

out_path = f'{OUT_DIR}/model_comparison.png'
canvas.save(out_path, quality=95)
print(f'\nSaved: {out_path}')
print("\nDone! Both figures generated.")

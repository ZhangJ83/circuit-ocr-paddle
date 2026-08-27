import json, os
from PIL import Image, ImageDraw, ImageFont

DATASET_DIR = r'G:\mimo_project\circuit_ocr\circuit-ocr-dataset'
OUT_DIR = f'{DATASET_DIR}/figures'
os.makedirs(OUT_DIR, exist_ok=True)

def load_font(size=14):
    for fp in ['C:/Windows/Fonts/arial.ttf','C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simsun.ttc']:
        if os.path.exists(fp):
            try: return ImageFont.truetype(fp, size)
            except: pass
    return ImageFont.load_default()

# Load V5 Golden training data
with open(f'{DATASET_DIR}/ocr_vl_sft-train-v9-pure.jsonl', encoding='utf-8') as f:
    samples = [json.loads(l) for l in f if l.strip()]

# Separate by source
synth_samples = []
real_samples = []
for s in samples:
    img = s['images'][0].lower()
    if any(w in img for w in ['synthetic', 'synth', 'simple', 'complex', 'medium', 'analog', 'digital', 'power', 'mixed']):
        synth_samples.append(s)
    else:
        real_samples.append(s)

print(f'Synthetic V3: {len(synth_samples)}, train-real: {len(real_samples)}')

# Pick 3 synthetic + 3 real samples
selected = synth_samples[:3] + real_samples[:3]
categories = ['Synthetic V3'] * 3 + ['train-real (KiCad)'] * 3

cell_w, cell_h = 320, 340
cols = 3
rows = 2
header_h = 36

canvas = Image.new('RGB', (cols * cell_w, rows * cell_h + header_h), (255, 255, 255))
font_title = load_font(14)
font_label = load_font(12)
font_gt = load_font(9)
draw = ImageDraw.Draw(canvas)

# Column headers
col_headers = ['Synthetic V3', 'Synthetic V3', 'Synthetic V3',
               'train-real (KiCad)', 'train-real (KiCad)', 'train-real (KiCad)']
for j, hdr in enumerate(col_headers):
    col = j % cols
    row_idx = j // cols
    x = col * cell_w + 10
    y = row_idx * cell_h + 8
    draw.text((x, y), hdr, fill=(0, 51, 153), font=font_title)

for i, sample in enumerate(selected):
    col = i % cols
    row_idx = i // cols
    x0 = col * cell_w
    y0 = row_idx * cell_h + header_h

    # Load and paste image
    img_rel = sample['images'][0].lstrip('./')
    img_path = os.path.join(DATASET_DIR, img_rel)
    if not os.path.exists(img_path):
        alt = img_path.replace('.png', '.PNG').replace('.jpg', '.JPG')
        if os.path.exists(alt):
            img_path = alt

    try:
        img = Image.open(img_path).convert('RGB')
        # Scale to fit cell
        max_w, max_h = cell_w - 20, cell_h - 80
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        img_x = x0 + (cell_w - img.width) // 2
        img_y = y0 + 5
        canvas.paste(img, (img_x, img_y))
        img_bottom = img_y + img.height
    except Exception as e:
        draw.text((x0 + 10, y0 + 20), f'Image error: {e}', fill=(255, 0, 0), font=font_label)
        img_bottom = y0 + 40

    # Draw GT label below image
    gt_text = sample['messages'][1]['content']
    gt_lines = gt_text.split('\n')[:6]  # Show first 6 lines
    gt_display = '\n'.join(gt_lines)
    if len(gt_text.split('\n')) > 6:
        gt_display += '\n...'

    gt_y = max(img_bottom + 6, y0 + cell_h - 85)
    draw.text((x0 + 5, gt_y - 12), 'GT:', fill=(0, 120, 0), font=font_label)
    for j, line in enumerate(gt_display.split('\n')[:7]):
        draw.text((x0 + 35, gt_y + j * 12), line, fill=(0, 80, 0), font=font_gt)

    # Separator lines
    if col < cols - 1:
        draw.line([(x0 + cell_w, y0), (x0 + cell_w, y0 + cell_h)], fill=(220, 220, 220), width=1)

# Row separator
draw.line([(0, cell_h + header_h), (cols * cell_w, cell_h + header_h)], fill=(180, 180, 180), width=2)

out_path = f'{OUT_DIR}/dataset_samples.png'
canvas.save(out_path, quality=95)
print(f'Saved: {out_path}')

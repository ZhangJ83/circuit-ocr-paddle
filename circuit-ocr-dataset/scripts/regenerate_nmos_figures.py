import os
from PIL import Image, ImageDraw, ImageFont

DATASET_DIR = r'G:\mimo_project\circuit_ocr\circuit-ocr-dataset'
OUT_DIR = f'{DATASET_DIR}/figures'
os.makedirs(OUT_DIR, exist_ok=True)

def load_font(size=12):
    for fp in ['C:/Windows/Fonts/arial.ttf','C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simsun.ttc']:
        if os.path.exists(fp):
            try: return ImageFont.truetype(fp, size)
            except: pass
    return ImageFont.load_default()

def create_fig(out_filename, title, prediction_text, label_text, title_color=(180, 0, 0)):
    img_path = f'{DATASET_DIR}/data/test/benjiaomodular_sot2dip.png'
    if not os.path.exists(img_path):
        img_path = img_path.replace('.png', '.PNG')
        if not os.path.exists(img_path):
            print(f"Error: {img_path} not found!")
            return

    img = Image.open(img_path).convert('RGB')
    img.thumbnail((320, 320), Image.LANCZOS)

    w, h = 680, 340
    canvas = Image.new('RGB', (w, h), (255, 255, 255))
    canvas.paste(img, (10, (h - img.height)//2))

    draw = ImageDraw.Draw(canvas)
    draw.line([(340, 0), (340, h)], fill=(200, 200, 200), width=2)

    font_bold = load_font(12)
    font_regular = load_font(10)

    # Ground Truth (green)
    y = 10
    draw.text((360, y), "Ground Truth:", fill=(0, 100, 0), font=font_bold)
    y += 18
    for line in label_text.split('\n'):
        draw.text((360, y), line, fill=(0, 100, 0), font=font_regular)
        y += 13

    # Prediction (title color)
    y += 12
    draw.text((360, y), title, fill=title_color, font=font_bold)
    y += 18
    pred_lines = prediction_text.split('\n')
    for line in pred_lines[:12]:
        draw.text((360, y), line, fill=title_color, font=font_regular)
        y += 13
    if len(pred_lines) > 12:
        draw.text((360, y), "... (truncated)", fill=title_color, font=font_regular)

    out_path = f'{OUT_DIR}/{out_filename}'
    canvas.save(out_path, quality=95)
    print(f"Generated: {out_path}")

# Ground truth for benjiaomodular_sot2dip.png
GT = (
    "J2\nConn_01x06_Male\n"
    "J3\nConn_01x03_Male\n"
    "J1\nConn_01x03_Male"
)

# 1. Base model failure (REAL data from results_base_easy50.jsonl)
create_fig(
    'v5_NMOS_Circuit_1_sch.png',
    'Base Model Prediction:',
    'Parameter | Value\nParameter_a | 1.0\nParameter_b | 1.0\nParameter_c | 1.0',
    GT
)

# 2. Old model collapse (REAL data from broken V2 eval - results_v3_s600_easy50.jsonl)
create_fig(
    'v5_NMOS_Circuit_2_old.png',
    'Collapsed Model (V4) Prediction:',
    '1\nJ22\n\n1\nJ22\n\n1\nJ22\n\n1\nJ22\n\n1\nJ22\n\n1\nJ22',
    GT
)

# 3. V10-Fixed S600 prediction (REAL data from eval_benchmark_v3.py)
create_fig(
    'v5_NMOS_Circuit_3_v5.png',
    'V10-Fixed S600 Prediction:',
    'GND\nR1\n10k\nC2\n10nF\nJ1\nConn_01x04_Pin',
    GT,
    title_color=(0, 0, 180)
)

print("Regenerated all 3 NMOS figures with real evaluation data!")

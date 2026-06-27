
import json, os, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DATASET_DIR = r'G:\mimo_project\circuit_ocr\circuit-ocr-dataset'
OUT_DIR = f'{DATASET_DIR}/figures'
os.makedirs(OUT_DIR, exist_ok=True)

def load_font(size=16):
    for fp in ['C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simsun.ttc','C:/Windows/Fonts/simhei.ttf','C:/Windows/Fonts/arial.ttf']:
        if os.path.exists(fp):
            try: return ImageFont.truetype(fp, size)
            except: pass
    return ImageFont.load_default()

def draw_text_box(draw, text, x, y, max_w, font, color):
    lines = []
    for line in text.split(chr(10)):
        if not line.strip(): lines.append(' '); continue
        cpl = max(1, int(max_w / (font.size * 0.55)))
        for k in range(0, len(line), cpl):
            lines.append(line[k:k+cpl])
    for j, line in enumerate(lines[:12]):
        draw.text((x, y + j * (font.size + 2)), line, fill=color, font=font)

def make_dataset_figure():
    print('Generating dataset visualization...')
    train_path = f'{DATASET_DIR}/ocr_vl_sft-train.jsonl'
    with open(train_path, encoding='utf-8') as f:
        train_data = [json.loads(l) for l in f if l.strip()]
    cats = {'Open Schematics':[], 'Masala-CHAI':[], 'Synthetic':[]}
    for s in train_data:
        img = s['images'][0].lower()
        if 'open_schematics' in img or 'openschematics' in img: cats['Open Schematics'].append(s)
        elif 'masala' in img or 'chai' in img: cats['Masala-CHAI'].append(s)
        elif any(w in img for w in ['synthetic','synth','simple','complex','medium']): cats['Synthetic'].append(s)
    test_path = f'{DATASET_DIR}/ocr_vl_sft-test-easy50.jsonl'
    with open(test_path, encoding='utf-8') as f:
        test_data = [json.loads(l) for l in f if l.strip()][:2]
    rows = []
    for cat in ['Open Schematics','Masala-CHAI','Synthetic']:
        sub = cats[cat][:4] if len(cats[cat])>=4 else cats[cat]
        for s in sub[:2]: rows.append((cat, s))
    for s in test_data: rows.append(('Test Set', s))
    cell_w, cell_h = 300, 300
    cols = 4
    grid_h = ((len(rows)+cols-1)//cols) * cell_h
    canvas = Image.new('RGB', (cols*cell_w, grid_h), (255,255,255))
    font = load_font(12)
    draw = ImageDraw.Draw(canvas)
    for i, (cat, sample) in enumerate(rows):
        img_rel = sample['images'][0].lstrip('./')
        img_path = f'{DATASET_DIR}/{img_rel}'
        if not os.path.exists(img_path): continue
        try:
            img = Image.open(img_path).convert('RGB')
            img.thumbnail((cell_w-20, cell_h-40), Image.LANCZOS)
            x = (i%cols)*cell_w + (cell_w-img.width)//2
            y = (i//cols)*cell_h + 10
            canvas.paste(img, (x,y))
            draw.text((i%cols*cell_w+5, y+img.height+5), cat, fill=(80,80,80), font=font)
        except: pass
    out = f'{OUT_DIR}/dataset_samples.png'
    canvas.save(out, quality=95)
    print(f'  Saved: {out}')

def make_model_comparison():
    print('Generating model comparison...')
    base_file = f'{DATASET_DIR}/results_base_easy50.jsonl'
    r16_file = f'{DATASET_DIR}/results_easy50_r16e3.jsonl'
    for fp in [base_file, r16_file]:
        if not os.path.exists(fp): print(f'  MISSING: {fp}'); return
    with open(base_file, encoding='utf-8') as f:
        base = {json.loads(l)['images'][0]:json.loads(l) for l in f if l.strip()}
    with open(r16_file, encoding='utf-8') as f:
        r16 = {json.loads(l)['images'][0]:json.loads(l) for l in f if l.strip()}
    common = sorted(set(base.keys()) & set(r16.keys()))[:6]
    print(f'  Common samples: {len(common)}')
    rows, cols = len(common), 4
    cell_w, cell_h = 240, 280
    canvas = Image.new('RGB', (cols*cell_w, rows*cell_h+30), (255,255,255))
    font = load_font(12)
    font_s = load_font(10)
    draw = ImageDraw.Draw(canvas)
    for j, h in enumerate(['Original Image','Ground Truth','Base Model','r16 LoRA']):
        draw.text((j*cell_w+10,5), h, fill=(0,0,0), font=font)
    for i, img_key in enumerate(common):
        y_base = 30 + i*cell_h
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
            draw.text((10,y_base+20), f'ERR:{e}', fill=(255,0,0), font=font_s)
        draw_text_box(draw, base[img_key].get('label','')[:80], cell_w+10, y_base+10, cell_w-20, font_s, (0,100,0))
        draw_text_box(draw, base[img_key].get('prediction','')[:80], 2*cell_w+10, y_base+10, cell_w-20, font_s, (180,0,0))
        draw_text_box(draw, r16[img_key].get('prediction','')[:80], 3*cell_w+10, y_base+10, cell_w-20, font_s, (0,0,180))
        draw.line([(0,y_base+cell_h-1),(cols*cell_w,y_base+cell_h-1)], fill=(200,200,200))
    out = f'{OUT_DIR}/model_comparison.png'
    canvas.save(out, quality=95)
    print(f'  Saved: {out}')

if __name__ == '__main__':
    make_dataset_figure()
    make_model_comparison()
    print('Done!')

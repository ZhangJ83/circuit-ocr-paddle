"""
Generate print PDFs. Test: 1/page img2pdf lossless. Train: 2/page via PIL.
All images embedded at near-native resolution.

Output:
  - output/print_easy50/easy50_print_v3.pdf    (44 pages)
  - output/print_train/train_print_v3.pdf      (50 pages)
"""
import json, io, os, random, tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).parent.parent
DATASET_DIR = PROJECT_DIR / "circuit-ocr-dataset"
OUT_DIR_TEST = PROJECT_DIR / "output" / "print_easy50"
OUT_DIR_TRAIN = PROJECT_DIR / "output" / "print_train"
OUT_DIR_TEST.mkdir(parents=True, exist_ok=True)
OUT_DIR_TRAIN.mkdir(parents=True, exist_ok=True)

# Use img2pdf if available (lossless), else fall back to PIL
try:
    import img2pdf
    HAS_IMG2PDF = True
except ImportError:
    HAS_IMG2PDF = False
    print("img2pdf not available, using PIL PDF (still good quality)")


def get_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        return ImageFont.load_default()


def add_label(img, label_text, font_size=36):
    w, h = img.size
    draw = ImageDraw.Draw(img)
    font = get_font(font_size)
    bbox = draw.textbbox((0, 0), label_text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bar_y = h - th - 20
    draw.rectangle([(0, bar_y - 5), (w, h)], fill="white")
    draw.text(((w - tw) // 2, bar_y), label_text, fill="black", font=font)
    return img


def save_pdf(pages, pdf_path):
    """Save pages as PDF. Uses img2pdf for lossless if available."""
    if HAS_IMG2PDF:
        # Save pages to temp PNGs, then combine
        png_bytes = []
        for page in pages:
            buf = io.BytesIO()
            page.save(buf, format="PNG", compress_level=1)
            png_bytes.append(buf.getvalue())

        with open(pdf_path, "wb") as f:
            f.write(img2pdf.convert(png_bytes))
    else:
        pages[0].save(pdf_path, save_all=True, append_images=pages[1:],
                       resolution=300, quality=95)


# ============================================================
# TEST SET: 1 image per page, at native resolution
# ============================================================
print("=== Test Set (easy50-pure) ===")
test_jsonl = DATASET_DIR / "ocr_vl_sft-test-easy50-pure.jsonl"

samples = []
with open(test_jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

pages = []
mapping = {}

for i, sample in enumerate(samples):
    photo_id = f"{i+1:03d}"
    img_rel = sample["images"][0]
    img_path = DATASET_DIR / img_rel.lstrip("./")

    if not img_path.exists():
        print(f"  MISSING: {img_path}")
        continue

    img = Image.open(img_path).convert("RGB")
    fname = os.path.basename(img_rel)

    gt = ""
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            gt = msg.get("content", "")
            break

    mapping[photo_id] = {
        "original_image": fname, "gt": gt,
        "gt_lines": len(gt.splitlines()) if gt else 0,
    }

    label = f"Photo #{photo_id}  |  {fname}"
    fs = max(18, min(img.size) // 40)
    img = add_label(img, label, font_size=fs)
    pages.append(img)
    print(f"  [{photo_id}] {fname} ({img.size[0]}x{img.size[1]})")

pdf_path = OUT_DIR_TEST / "easy50_print_v3.pdf"
save_pdf(pages, pdf_path)

with open(OUT_DIR_TEST / "photo_mapping.json", 'w', encoding='utf-8') as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)
print(f"  PDF: {pdf_path} ({len(pages)} pages)\n")

# ============================================================
# TRAIN SET: 2 images per page, high-resolution canvas
# ============================================================
print("=== Train Set (100 samples) ===")
train_jsonl = DATASET_DIR / "ocr_vl_sft-train-v9-pure.jsonl"

samples = []
with open(train_jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

# Stratified selection
sample_info = []
for s in samples:
    gt = ""
    for msg in s.get("messages", []):
        if msg.get("role") == "assistant":
            gt = msg.get("content", "")
            break
    sample_info.append((s, len(gt.splitlines()) if gt else 0))

import random
random.seed(42)
simple = [s for s, n in sample_info if n <= 10]
medium = [s for s, n in sample_info if 10 < n <= 20]
complex_ = [s for s, n in sample_info if n > 20]

total = len(sample_info)
n_simple = max(20, int(100 * len(simple) / total))
n_medium = max(20, int(100 * len(medium) / total))
n_complex = 100 - n_simple - n_medium

selected = (
    random.sample(simple, min(n_simple, len(simple))) +
    random.sample(medium, min(n_medium, len(medium))) +
    random.sample(complex_, min(n_complex, len(complex_)))
)
random.shuffle(selected)

# Layout: each image gets at least 1600px width
CANVAS_W = 2000
CANVAS_H_PER_IMG = 1600
MARGIN = 50
LABEL_SPACE = 60
GAP = 30

mapping_train = {}
pages = []
i = 0
page_num = 1

while i < len(selected):
    batch = selected[i:i+2]
    n = len(batch)
    canvas_h = MARGIN * 2 + CANVAS_H_PER_IMG * n + LABEL_SPACE * n + GAP * (n - 1)
    canvas = Image.new("RGB", (CANVAS_W, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = get_font(24)

    usable_w = CANVAS_W - 2 * MARGIN

    y = MARGIN
    for j, sample in enumerate(batch):
        photo_id = f"{i + j + 1:03d}"
        img_rel = sample["images"][0]
        img_path = DATASET_DIR / img_rel.lstrip("./")

        if not img_path.exists():
            print(f"  MISSING: {img_path}")
            continue

        fname = os.path.basename(img_rel)
        gt = ""
        for msg in sample.get("messages", []):
            if msg.get("role") == "assistant":
                gt = msg.get("content", "")
                break

        mapping_train[photo_id] = {
            "original_image": fname, "gt": gt,
            "gt_lines": len(gt.splitlines()) if gt else 0,
        }

        img = Image.open(img_path).convert("RGB")
        iw, ih = img.size
        scale = min(usable_w / iw, CANVAS_H_PER_IMG / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        img = img.resize((nw, nh), Image.LANCZOS)

        ix = MARGIN + (usable_w - nw) // 2
        canvas.paste(img, (ix, y))

        label = f"#{photo_id}  {fname}"
        draw.text((MARGIN, y + nh + 6), label, fill="black", font=font)

        y = y + nh + LABEL_SPACE + GAP

    pages.append(canvas)
    ids = [f"{i+j+1:03d}" for j in range(n)]
    print(f"  Page {page_num}: {', '.join(ids)}")
    i += n
    page_num += 1

pdf_path2 = OUT_DIR_TRAIN / "train_print_v3.pdf"
save_pdf(pages, pdf_path2)

with open(OUT_DIR_TRAIN / "photo_mapping_train.json", 'w', encoding='utf-8') as f:
    json.dump(mapping_train, f, ensure_ascii=False, indent=2)
print(f"  PDF: {pdf_path2} ({len(pages)} pages, {len(mapping_train)} images)")

print(f"\nDone!")

"""
Generate a print-ready PDF from easy50-pure test images, with numbered pages
so photographed images can be automatically matched to ground truth annotations.

Usage: python scripts/generate_print_pdf.py
Output:
  - output/print_easy50/easy50_print.pdf       # Print-ready PDF
  - output/print_easy50/photo_mapping.json      # photo_num -> GT mapping
"""
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Paths
DATASET_DIR = Path(__file__).parent.parent / "circuit-ocr-dataset"
TEST_JSONL = DATASET_DIR / "ocr_vl_sft-test-easy50-pure.jsonl"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "print_easy50"

# A4 at 300 DPI (pixels)
A4_W, A4_H = 2480, 3508
IMAGE_MAX_W, IMAGE_MAX_H = 2300, 3100  # Leave margin for label

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load samples
samples = []
with open(TEST_JSONL, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            samples.append(json.loads(line))

print(f"Loaded {len(samples)} samples")

# Build mapping and pages
mapping = {}
pages = []

for i, sample in enumerate(samples):
    photo_id = f"{i+1:03d}"  # 001, 002, ..., 044

    # Extract image path and GT
    img_rel = sample["images"][0]  # e.g. "./data/test/foo.png"
    img_path = DATASET_DIR / img_rel.lstrip("./")

    # Extract GT from assistant message
    gt = ""
    for msg in sample.get("messages", []):
        if msg.get("role") == "assistant":
            gt = msg.get("content", "")
            break

    # Build mapping
    mapping[photo_id] = {
        "original_image": os.path.basename(img_rel),
        "gt": gt,
        "gt_lines": len(gt.splitlines()) if gt else 0,
    }

    # Load image
    try:
        img = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        print(f"  WARNING: Image not found: {img_path}")
        continue

    # Resize to fit page while keeping aspect ratio
    img_w, img_h = img.size
    scale = min(IMAGE_MAX_W / img_w, IMAGE_MAX_H / img_h)
    new_w, new_h = int(img_w * scale), int(img_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    # Create A4 canvas
    canvas = Image.new("RGB", (A4_W, A4_H), "white")
    draw = ImageDraw.Draw(canvas)

    # Paste image centered (shifted up slightly for label at bottom)
    img_x = (A4_W - new_w) // 2
    img_y = (A4_H - new_h - 120) // 2  # 120px for bottom label
    canvas.paste(img, (img_x, img_y))

    # Draw photo ID at bottom
    label = f"Photo #{photo_id}  |  {os.path.basename(img_rel)}"
    try:
        font = ImageFont.truetype("arial.ttf", 42)
    except (OSError, IOError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((A4_W - tw) // 2, A4_H - 100), label, fill="black", font=font)

    # Draw separator line
    draw.line([(80, A4_H - 70), (A4_W - 80, A4_H - 70)], fill="gray", width=2)

    pages.append(canvas)
    print(f"  [{photo_id}] {os.path.basename(img_rel)} ({gt.splitlines()[0] if gt else '?'} ...)")

# Save PDF
pdf_path = OUTPUT_DIR / "easy50_print_v2.pdf"
pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=300)
print(f"\nPDF saved: {pdf_path} ({len(pages)} pages)")

# Save mapping
mapping_path = OUTPUT_DIR / "photo_mapping.json"
with open(mapping_path, 'w', encoding='utf-8') as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)
print(f"Mapping saved: {mapping_path}")

print(f"""
========================================
  Instructions:
  1. Print {pdf_path}
  2. Photograph each page with your phone
  3. Name photos: 001.jpg, 002.jpg, ..., {len(pages):03d}.jpg
  4. Give the photos back to me
  5. I'll use photo_mapping.json to auto-match GT
========================================
""")

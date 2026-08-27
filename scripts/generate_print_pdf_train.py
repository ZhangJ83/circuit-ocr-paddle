"""
Generate print-ready PDF from training images for real photo collection.
Selects 100 diverse samples, packs 2-3 per A4 page, with numbered mapping.

Usage: python scripts/generate_print_pdf_train.py [--num 100]
Output:
  - output/print_train/train_print.pdf
  - output/print_train/photo_mapping_train.json
"""
import json
import os
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).parent.parent
DATASET_DIR = PROJECT_DIR / "circuit-ocr-dataset"
TRAIN_JSONL = DATASET_DIR / "ocr_vl_sft-train-v9-pure.jsonl"
OUTPUT_DIR = PROJECT_DIR / "output" / "print_train"

# A4 at 300 DPI
A4_W, A4_H = 2480, 3508
MARGIN = 80
LABEL_H = 60
NUM_SAMPLES = 100

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def layout_page(images_with_ids, page_num):
    """
    Layout 2-3 images on one A4 page with ID labels.
    Returns PIL Image of the page.
    """
    n = len(images_with_ids)
    canvas = Image.new("RGB", (A4_W, A4_H), "white")
    draw = ImageDraw.Draw(canvas)

    # Page header
    try:
        font_h = ImageFont.truetype("arial.ttf", 36)
        font_l = ImageFont.truetype("arial.ttf", 28)
    except (OSError, IOError):
        font_h = ImageFont.load_default()
        font_l = ImageFont.load_default()

    header = f"Train Page {page_num}"
    draw.text((MARGIN, 20), header, fill="black", font=font_h)

    # Calculate layout
    usable_w = A4_W - 2 * MARGIN
    usable_h = A4_H - 2 * MARGIN - LABEL_H * n - 80  # Reserve for labels

    # Always 2 per page
    cell_h = usable_h // 2

    y_offset = MARGIN + 60
    for idx, (img, photo_id, fname) in enumerate(images_with_ids):
        # Resize image to fit cell
        img_w, img_h = img.size
        scale = min(usable_w / img_w, cell_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        img_rs = img.resize((new_w, new_h), Image.LANCZOS)

        # Center in cell
        img_x = MARGIN + (usable_w - new_w) // 2
        img_y = y_offset
        canvas.paste(img_rs, (img_x, img_y))

        # Label below image
        label = f"#{photo_id}  {fname}"
        draw.text((MARGIN, img_y + new_h + 8), label, fill="black", font=font_l)

        y_offset = img_y + new_h + LABEL_H + 20

    # Page number at bottom
    draw.text((A4_W // 2 - 50, A4_H - 60), f"- {page_num} -", fill="gray", font=font_l)

    return canvas


def main():
    random.seed(42)

    # Load training data
    samples = []
    with open(TRAIN_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} training samples")

    # Select diverse subset: balance by image complexity (GT line count)
    sample_info = []
    for s in samples:
        gt = ""
        for msg in s.get("messages", []):
            if msg.get("role") == "assistant":
                gt = msg.get("content", "")
                break
        n_lines = len(gt.splitlines()) if gt else 0
        sample_info.append((s, n_lines))

    # Stratified sampling: ensure mix of simple/medium/complex
    simple = [s for s, n in sample_info if n <= 10]
    medium = [s for s, n in sample_info if 10 < n <= 20]
    complex_ = [s for s, n in sample_info if n > 20]
    print(f"  Simple (<=10 lines): {len(simple)}")
    print(f"  Medium (11-20 lines): {len(medium)}")
    print(f"  Complex (>20 lines): {len(complex_)}")

    # Proportional allocation
    total = len(sample_info)
    n_simple = max(20, int(NUM_SAMPLES * len(simple) / total))
    n_medium = max(20, int(NUM_SAMPLES * len(medium) / total))
    n_complex = NUM_SAMPLES - n_simple - n_medium

    selected = (
        random.sample(simple, min(n_simple, len(simple))) +
        random.sample(medium, min(n_medium, len(medium))) +
        random.sample(complex_, min(n_complex, len(complex_)))
    )
    random.shuffle(selected)
    print(f"Selected {len(selected)} samples ({n_simple}s + {n_medium}m + {n_complex}c)")

    # Build pages (2-3 images per page)
    all_images = []
    mapping = {}
    photo_id = 1

    for sample in selected:
        img_rel = sample["images"][0]
        img_path = DATASET_DIR / img_rel.lstrip("./")

        if not img_path.exists():
            print(f"  SKIP: {img_path} not found")
            continue

        img = Image.open(img_path).convert("RGB")
        pid = f"{photo_id:03d}"
        fname = os.path.basename(img_rel)

        # GT
        gt = ""
        for msg in sample.get("messages", []):
            if msg.get("role") == "assistant":
                gt = msg.get("content", "")
                break

        all_images.append((img, pid, fname, gt))
        mapping[pid] = {
            "original_image": fname,
            "gt": gt,
            "gt_lines": len(gt.splitlines()) if gt else 0,
        }
        photo_id += 1

    # Layout into pages: 2 per page
    pages = []
    i = 0
    page_num = 1
    while i < len(all_images):
        batch = all_images[i:i + 2]
        page_data = [(img, pid, fname) for img, pid, fname, _gt in batch]
        page = layout_page(page_data, page_num)
        pages.append(page)

        # Log
        ids = [pid for _, pid, _ in page_data]
        print(f"  Page {page_num}: {', '.join(ids)}")
        i += len(batch)
        page_num += 1

    # Save PDF
    pdf_path = OUTPUT_DIR / "train_print_v2.pdf"
    pages[0].save(pdf_path, save_all=True, append_images=pages[1:], resolution=300)
    print(f"\nPDF saved: {pdf_path} ({len(pages)} pages, {photo_id - 1} images)")

    # Save mapping
    mapping_path = OUTPUT_DIR / "photo_mapping_train.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Mapping saved: {mapping_path}")

    print(f"""
========================================
  Instructions:
  1. Print {pdf_path}
  2. Photograph each image (@2-3 per page, crop if needed)
  3. Name photos: 001.jpg, 002.jpg, ..., {(photo_id-1):03d}.jpg
  4. Put in output/print_train/photos/
  5. Run: python scripts/build_photo_testset.py --train --photo_dir output/print_train/photos
========================================
""")


if __name__ == "__main__":
    main()

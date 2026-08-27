#!/usr/bin/env python3
"""Pre-convert all images in a JSONL dataset to JPEG (max 768px).
Creates a new JSONL with updated image paths.
"""
import json, sys
from pathlib import Path
from PIL import Image

DATA_IN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ocr_vl_sft-test-easy50.jsonl")
DATA_OUT = DATA_IN.with_stem(DATA_IN.stem + "-jpeg")
JPEG_DIR = Path("data/test_jpeg")
JPEG_DIR.mkdir(exist_ok=True, parents=True)

samples = []
with open(DATA_IN, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

print(f"Converting {len(samples)} images to JPEG (max 768px)...")
converted = 0
for i, s in enumerate(samples):
    img_rel = s["images"][0]
    img_path = DATA_IN.parent / img_rel
    if not img_path.exists():
        img_path = DATA_IN.parent / Path(img_rel).name

    jpeg_name = Path(img_rel).stem + ".jpg"
    jpeg_path = JPEG_DIR / jpeg_name

    if jpeg_path.exists():
        converted += 1
        s["images"] = [str(jpeg_path.relative_to(DATA_IN.parent))]
        continue

    try:
        img = Image.open(img_path).convert("RGB")
        # Force all images to 384x384 to avoid Paddle C++ variable-size bugs
        img = img.resize((384, 384), Image.LANCZOS)
        img.save(jpeg_path, "JPEG", quality=95)
        img.close()
        s["images"] = [str(jpeg_path.relative_to(DATA_IN.parent))]
        converted += 1
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)}...")
    except Exception as e:
        print(f"  [{i+1}] FAIL {img_path.name}: {e}")

# Write new JSONL
with open(DATA_OUT, "w", encoding="utf-8") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Done: {converted}/{len(samples)} converted")
print(f"Output: {DATA_OUT}")
print(f"JPEG dir: {JPEG_DIR}")

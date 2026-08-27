"""
Process unused kicad_sch files and filter clean ones for review.
"""
import json, sys, os, re
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
SRC_DIR = PROJECT_DIR / 'circuit-ocr-dataset/data/open_schematics_v2/kicad_sch'
IMG_DIR = PROJECT_DIR / 'circuit-ocr-dataset/data/open_schematics_v2/images'
REVIEW_IMG = PROJECT_DIR / 'output/review_1000/images'
REVIEW_ANNO = PROJECT_DIR / 'output/review_1000/annotations'

sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
from generate_gt_from_kicad import parse

# Load existing mapping
with open(PROJECT_DIR / 'output/review_1000/mapping.json', 'r') as f:
    mapping = json.load(f)
mapped_names = set(m['original_name'] for m in mapping)

# Find unused source files
all_src = {f.stem: f for f in SRC_DIR.glob('*.kicad_sch')}
unused = {k: v for k, v in all_src.items() if k not in mapped_names}
print(f"Unused source files: {len(unused)}")

# Find max existing ID
existing_ids = [int(m['id']) for m in mapping]
next_id = max(existing_ids) + 1
print(f"Next ID starts at: {next_id:04d}")

good, bad = [], []
for name, sch_path in sorted(unused.items()):
    try:
        gt, blocks = parse(str(sch_path))
    except Exception as e:
        bad.append((name, f"parse error: {e}"))
        continue

    # Check for ? in refs (unannotated components)
    if re.search(r'(?<!\w)\?\n', gt) or re.search(r'\n\?', gt):
        bad.append((name, "? ref"))
        continue

    # Check lines count (too few = probably empty/broken)
    lines = [l for l in gt.splitlines() if l.strip()]
    if len(lines) < 5:
        bad.append((name, f"too few lines: {len(lines)}"))
        continue

    good.append((name, gt, lines))

print(f"Good: {len(good)}, Bad: {len(bad)}")
for name, reason in bad:
    print(f"  BAD {name}: {reason}")

# Write new review samples
added = 0
for name, gt, lines in good:
    uid = f"{next_id:04d}"

    # Write annotation
    with open(REVIEW_ANNO / f"{uid}.txt", 'w', encoding='utf-8') as f:
        f.write(gt)

    # Copy image
    import shutil
    img_src = IMG_DIR / f"{name}.png"
    if img_src.exists():
        shutil.copy2(img_src, REVIEW_IMG / f"{uid}.png")
    else:
        print(f"  WARNING: image not found for {name}")
        continue

    mapping.append({"id": uid, "original_name": name})
    next_id += 1
    added += 1

# Save updated mapping
with open(PROJECT_DIR / 'output/review_1000/mapping.json', 'w') as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)

print(f"\nAdded {added} new samples")
print(f"Total review samples: {len(mapping)}")

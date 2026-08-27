"""
Build a test JSONL from photographed images by matching filenames to GT.
The user names photos as 001.jpg, 002.jpg, ... and this script auto-matches.

Usage: python scripts/build_photo_testset.py --photo_dir <dir> [--output <jsonl>]
Output: A JSONL test file ready for eval_benchmark_v3.py
"""
import argparse
import json
import os
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
MAPPING_FILE = PROJECT_DIR / "output" / "print_easy50" / "photo_mapping.json"
DEFAULT_PHOTO_DIR = PROJECT_DIR / "output" / "print_easy50" / "photos"
DEFAULT_OUTPUT = PROJECT_DIR / "output" / "print_easy50" / "ocr_vl_sft-test-easy50-photo.jsonl"


def main():
    parser = argparse.ArgumentParser(description="Build test JSONL from photographed images")
    parser.add_argument("--photo_dir", default=str(DEFAULT_PHOTO_DIR),
                        help="Directory containing 001.jpg, 002.jpg, ...")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output JSONL path")
    args = parser.parse_args()

    # Load mapping
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    print(f"Loaded mapping: {len(mapping)} samples")

    # Scan photos
    photo_dir = Path(args.photo_dir)
    if not photo_dir.exists():
        print(f"ERROR: Photo directory not found: {photo_dir}")
        print(f"  Create it and place your photos there, named 001.jpg ... 044.jpg")
        return

    matched = 0
    missing = []
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy photos to a managed directory (relative to JSONL)
    managed_photo_dir = output_dir / "photo_images"
    managed_photo_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as out_f:
        for photo_id in sorted(mapping.keys()):
            entry = mapping[photo_id]

            # Find photo file (try common extensions)
            found = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                candidate = photo_dir / f"{photo_id}{ext}"
                if candidate.exists():
                    found = candidate
                    break

            if found is None:
                missing.append(photo_id)
                print(f"  [{photo_id}] MISSING - no file found for {photo_id}.*")
                continue

            # Copy to managed dir with consistent naming
            dest_name = f"{photo_id}.jpg"
            dest_path = managed_photo_dir / dest_name
            shutil.copy2(found, dest_path)

            # Use relative path from output dir
            rel_img_path = f"./photo_images/{dest_name}"

            # Build JSONL entry (same format as original test set)
            sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": rel_img_path},
                            {"type": "text", "text": "Please identify all the components shown and their values in this circuit diagram picture. List it in the order of appearance with one component per line, using the format \"component_label\\nvalue\"."}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": entry["gt"]
                    }
                ],
                "images": [rel_img_path],
                "_meta": {
                    "photo_id": photo_id,
                    "original_image": entry["original_image"],
                    "source": "printed_photographed",
                }
            }
            out_f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            matched += 1
            print(f"  [{photo_id}] ✓ {entry['original_image']} ({entry['gt_lines']} GT lines)")

    print(f"\nMatched: {matched}/{len(mapping)}")
    if missing:
        print(f"Missing: {', '.join(missing)}")
    print(f"Output: {args.output}")
    print(f"Photos copied to: {managed_photo_dir}")
    print(f"\nReady for evaluation:")
    print(f"  python scripts/eval_benchmark_v3.py --test_jsonl {args.output}")


if __name__ == "__main__":
    main()

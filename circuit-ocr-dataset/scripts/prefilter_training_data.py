#!/usr/bin/env python3
"""
Pre-filter training data to fit max_seq_len constraint.

PaddleOCR-VL's max_seq_len covers both image tokens (~800-1500) and text tokens.
With max_seq_len=2048:
  - Image tokens: ~800 for 384x384 patches + positional
  - Text budget:  ~1248 tokens ≈ ~960 chars (English technical text)
  - Safe threshold: label > 1000 chars → likely truncated

This script:
1. Reads the unified training data
2. Filters out samples where label exceeds the safe threshold
3. Writes a filtered copy (original is untouched)
4. Reports statistics

Usage:
    python scripts/prefilter_training_data.py [--max_chars 1000]
"""

import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Filter training data by label length")
    parser.add_argument("--input", type=str,
                        default="ocr_vl_sft-train.jsonl",
                        help="Input JSONL file (relative to script dir parent)")
    parser.add_argument("--output", type=str,
                        default="ocr_vl_sft-train-filtered.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--max_chars", type=int, default=1000,
                        help="Maximum label character count (default: 1000)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only report stats, don't write output")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, args.input)
    output_path = os.path.join(base_dir, args.output)

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    kept = []
    dropped = []
    total = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            sample = json.loads(line)
            label = sample["messages"][1]["content"]
            label_len = len(label)
            if label_len <= args.max_chars:
                kept.append(sample)
            else:
                dropped.append((label_len, sample["images"][0]))

    # Statistics
    print("=" * 50)
    print("  Training Data Pre-Filtering Report")
    print("=" * 50)
    print(f"  Input:           {args.input}")
    print(f"  Max label chars: {args.max_chars}")
    print(f"  Total samples:   {total}")
    print(f"  Kept:            {len(kept)} ({len(kept)/total*100:.1f}%)")
    print(f"  Dropped:         {len(dropped)} ({len(dropped)/total*100:.1f}%)")
    print("-" * 50)

    if dropped:
        # Show distribution of dropped label lengths
        dropped_lens = sorted(d[0] for d in dropped)
        print(f"  Dropped label lengths:")
        print(f"    Min:    {min(dropped_lens)}")
        print(f"    Median: {dropped_lens[len(dropped_lens)//2]}")
        print(f"    Max:    {max(dropped_lens)}")
        print(f"  Top 5 longest dropped:")
        for label_len, img_path in sorted(dropped, reverse=True)[:5]:
            print(f"    {label_len:5d} chars — {img_path}")

    print("=" * 50)

    if args.dry_run:
        print("[DRY RUN] No output written.")
        return

    # Write filtered output
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in kept:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Filtered data saved to: {output_path}")
    print(f"\nTo use for training, update YAML config:")
    print(f'  train_dataset_path: ./{args.output}')


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Dataset Build Script
====================
Build the complete circuit schematic OCR dataset.

Usage:
    python scripts/build_dataset.py --project-dir .
    python scripts/build_dataset.py --skip-scraping --synthetic-count 500
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.dataset_builder import DatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="Build circuit OCR dataset")
    parser.add_argument("--project-dir", default=".", help="Project root")
    parser.add_argument("--skip-scraping", action="store_true",
                       help="Skip GitHub scraping (use existing data)")
    parser.add_argument("--skip-synthetic", action="store_true",
                       help="Skip synthetic data generation")
    parser.add_argument("--skip-degradation", action="store_true",
                       help="Skip degradation augmentation")
    parser.add_argument("--max-repos", type=int, default=200)
    parser.add_argument("--synthetic-count", type=int, default=300)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--github-token", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = {
        "dpi": args.dpi,
        "max_repos": args.max_repos,
        "synthetic_count": args.synthetic_count,
        "github_token": args.github_token,
    }

    builder = DatasetBuilder(project_dir=args.project_dir, config=config)
    stats = builder.build(
        skip_scraping=args.skip_scraping,
        skip_synthetic=args.skip_synthetic,
        skip_degradation=args.skip_degradation,
    )

    print(f"\n{'='*60}")
    print(f"Dataset Built Successfully!")
    print(f"{'='*60}")
    print(f"  Total images:      {stats.total_images}")
    print(f"  Total annotations: {stats.total_annotations}")
    print(f"  Train:             {stats.train_images}")
    print(f"  Val:               {stats.val_images}")
    print(f"  Test:              {stats.test_images}")
    print(f"  Synthetic:         {stats.synthetic_images}")
    print(f"  Degraded:          {stats.degraded_images}")
    print(f"  Unique chars:      {stats.unique_chars}")


if __name__ == "__main__":
    main()

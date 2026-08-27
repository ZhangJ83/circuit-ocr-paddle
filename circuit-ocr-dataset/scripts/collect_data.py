#!/usr/bin/env python3
"""
Data Collection Script
======================
Scrape GitHub for open-source KiCad projects and collect schematic files.

Usage:
    python scripts/collect_data.py --max-repos 200 --github-token YOUR_TOKEN
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.github_scraper import GitHubKiCadScraper


def main():
    parser = argparse.ArgumentParser(description="Collect KiCad projects from GitHub")
    parser.add_argument("--output-dir", default="data/raw", help="Output directory")
    parser.add_argument("--max-repos", type=int, default=200, help="Max repos to clone")
    parser.add_argument("--github-token", default=None, help="GitHub token")
    parser.add_argument("--min-file-size", type=int, default=1000,
                       help="Min .kicad_sch file size in bytes")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    scraper = GitHubKiCadScraper(
        output_dir=args.output_dir,
        github_token=args.github_token,
        max_repos=args.max_repos,
        min_file_size=args.min_file_size,
    )

    projects = scraper.run()
    total_files = sum(len(p.sch_files) for p in projects)

    print(f"\nCollection complete!")
    print(f"  Projects: {len(projects)}")
    print(f"  Schematic files: {total_files}")


if __name__ == "__main__":
    main()

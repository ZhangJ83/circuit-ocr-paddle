#!/usr/bin/env python3
"""
Dataset Integrity Verification Script
=====================================
Checks the generated SFT JSONL files to ensure format correctness
and verify that all referenced image paths exist.
"""

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("verify_integrity")

def verify_dataset():
    splits = ["train", "val", "test"]
    base_dir = Path(".")
    all_ok = True

    logger.info("Starting dataset integrity verification...")

    for split in splits:
        jsonl_path = base_dir / f"ocr_vl_sft-{split}.jsonl"
        if not jsonl_path.exists():
            logger.error(f"SFT file {jsonl_path} does not exist!")
            all_ok = False
            continue

        logger.info(f"Checking {jsonl_path}...")
        line_count = 0
        missing_images = 0
        invalid_format = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line_count += 1
                try:
                    data = json.loads(line)
                    # Check messages format
                    if "messages" not in data or "images" not in data:
                        logger.error(f"Line {i+1}: Missing 'messages' or 'images' keys.")
                        invalid_format += 1
                        continue
                    
                    messages = data["messages"]
                    if len(messages) != 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
                        logger.error(f"Line {i+1}: Invalid messages role structure.")
                        invalid_format += 1
                    
                    # Check images path
                    images = data["images"]
                    for img_rel_path in images:
                        img_path = base_dir / img_rel_path
                        if not img_path.exists():
                            logger.error(f"Line {i+1}: Referenced image {img_path} not found.")
                            missing_images += 1

                except Exception as e:
                    logger.error(f"Line {i+1}: Failed to parse JSON: {e}")
                    invalid_format += 1

        logger.info(f"Split '{split}' Summary:")
        logger.info(f"  Total samples: {line_count}")
        logger.info(f"  Invalid format: {invalid_format}")
        logger.info(f"  Missing images: {missing_images}")

        if invalid_format > 0 or missing_images > 0:
            all_ok = False

    if all_ok:
        logger.info("SUCCESS: All splits are 100% valid with all referenced images existing.")
        return True
    else:
        logger.error("FAILURE: Integrity issues found in dataset.")
        return False

if __name__ == "__main__":
    verify_dataset()

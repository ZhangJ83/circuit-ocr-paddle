#!/usr/bin/env python3
"""
Dataset Split and Export Script
===============================
This script shuffles and splits the cleaned dataset into train (70%),
val (15%), and test (15%) splits, copying files and generating SFT jsonl
files using relative paths.

Output files (in repository root):
- ocr_vl_sft-train.jsonl
- ocr_vl_sft-val.jsonl
- ocr_vl_sft-test.jsonl
"""

import os
import json
import random
import shutil
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("split_and_export")

def split_and_export_dataset(data_dir: str, train_ratio=0.70, val_ratio=0.15):
    base_path = Path(data_dir)
    cleaned_dir = base_path / "cleaned"

    if not cleaned_dir.exists():
        logger.error(f"Cleaned directory {cleaned_dir} does not exist. Run clean_and_verify.py first.")
        return

    # Find all cleaned json files
    cleaned_json_files = list(cleaned_dir.glob("*.json"))
    logger.info(f"Found {len(cleaned_json_files)} cleaned schematic samples.")

    if len(cleaned_json_files) == 0:
        logger.warning("No clean samples found, skipping split.")
        return

    # Shuffle
    random.seed(42) # Fixed seed for reproducible splits
    random.shuffle(cleaned_json_files)

    total_samples = len(cleaned_json_files)
    n_train = int(total_samples * train_ratio)
    n_val = int(total_samples * val_ratio)

    splits = {
        "train": cleaned_json_files[:n_train],
        "val": cleaned_json_files[n_train:n_train + n_val],
        "test": cleaned_json_files[n_train + n_val:]
    }

    prompt = "OCR:"

    for split_name, json_files in splits.items():
        split_dir = base_path / split_name
        if split_dir.exists():
            logger.info(f"Clearing old split directory: {split_dir}")
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)
        
        jsonl_lines = []
        logger.info(f"Processing split '{split_name}' with {len(json_files)} samples...")

        for ann_file in json_files:
            try:
                with open(ann_file, "r", encoding="utf-8") as f:
                    ann_data = json.load(f)

                # Find associated image
                img_path_field = ann_data.get("image_path", "")
                if img_path_field:
                    img_name = Path(img_path_field).name
                else:
                    img_name = ann_file.with_suffix(".png").name
                
                src_img_path = cleaned_dir / img_name
                
                if not src_img_path.exists():
                    # Check other fallbacks if not found
                    for ext in [".png", ".jpg", ".jpeg"]:
                        fallback_img_name = Path(img_name).with_suffix(ext).name
                        if (cleaned_dir / fallback_img_name).exists():
                            img_name = fallback_img_name
                            src_img_path = cleaned_dir / img_name
                            break
                    
                    if not src_img_path.exists():
                        logger.warning(f"Image not found for {ann_file.name} (checked {img_name}), skipping.")
                        continue

                # Copy files to split folder
                dst_img_path = split_dir / img_name
                dst_ann_path = split_dir / ann_file.name
                
                shutil.copy2(src_img_path, dst_img_path)
                shutil.copy2(ann_file, dst_ann_path)

                # Format annotation texts
                annotations = ann_data.get("annotations", [])
                text_list = [ann["text"] for ann in annotations if "text" in ann]
                text_target = "\n".join(text_list)

                # Construct PaddleFormers SFT messages item with relative paths
                relative_img_path = f"./data/{split_name}/{img_name}"
                sft_item = {
                    "messages": [
                        {"role": "user", "content": f"<image>{prompt}"},
                        {"role": "assistant", "content": text_target}
                    ],
                    "images": [relative_img_path]
                }
                jsonl_lines.append(json.dumps(sft_item, ensure_ascii=False))

            except Exception as e:
                logger.error(f"Failed to process split item {ann_file.name}: {e}")

        # Write to SFT JSONL file in root dir
        jsonl_output_path = Path(f"./ocr_vl_sft-{split_name}.jsonl")
        with open(jsonl_output_path, "w", encoding="utf-8") as out_f:
            out_f.write("\n".join(jsonl_lines) + "\n")

        logger.info(f"Successfully exported {len(jsonl_lines)} SFT samples to {jsonl_output_path}")

    logger.info("Splitting and exporting completed successfully.")

if __name__ == "__main__":
    split_and_export_dataset("./data")

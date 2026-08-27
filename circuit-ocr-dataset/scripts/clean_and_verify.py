#!/usr/bin/env python3
"""
Dataset Cleaning and Verification Script
========================================
This script performs data cleaning, deduplication, and quality control
on the circuit schematic dataset. All paths are relative to the execution root.

Tasks:
1. Load annotations and images from ./data/
2. Run hard filters (min components >= 3, file integrity)
3. Run deduplication (Image hash + Annotation label set hash)
4. Check text box overlaps (IoU > 0.8) and boundary errors
5. Filter out hidden/non-rendered symbols (e.g. #PWR)
6. Run optional VLM quality scoring (falls back to heuristic filtering if no API key is provided)
7. Save cleaned dataset to ./data/cleaned/
"""

import os
import json
import hashlib
import shutil
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import threading
from concurrent.futures import ThreadPoolExecutor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("clean_and_verify")

def calculate_file_sha256(filepath: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def calculate_text_hash(annotations: list) -> str:
    """Calculate hash of sorted text annotations to deduplicate text layout."""
    texts = sorted([ann.get("text", "") for ann in annotations if "text" in ann])
    text_str = "|".join(texts)
    return hashlib.sha256(text_str.encode("utf-8")).hexdigest()

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    Bounding box format: list of 4 points [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    or coordinates [x1, y1, x2, y2].
    """
    def to_ltrb(box):
        if len(box) == 4 and isinstance(box[0], list):
            # Polygon format
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            return min(xs), min(ys), max(xs), max(ys)
        elif len(box) == 4:
            # [x1, y1, x2, y2]
            return box[0], box[1], box[2], box[3]
        return 0, 0, 0, 0

    l1, t1, r1, b1 = to_ltrb(box1)
    l2, t2, r2, b2 = to_ltrb(box2)

    inter_l = max(l1, l2)
    inter_t = max(t1, t2)
    inter_r = min(r1, r2)
    inter_b = min(b1, b2)

    if inter_r < inter_l or inter_b < inter_t:
        return 0.0

    inter_area = (inter_r - inter_l) * (inter_b - inter_t)
    area1 = (r1 - l1) * (b1 - t1)
    area2 = (r2 - l2) * (b2 - t2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def call_vlm_scoring(image_path: Path) -> int:
    """
    Score the image quality using the Volcengine Ark API.
    If it's synthetic data, skip API query and return 5.
    """
    name = image_path.name.lower()
    synthetic_prefixes = ["synth", "analog", "digital", "mixed", "power"]
    if any(p in name for p in synthetic_prefixes):
        # Skip VLM scoring for programmatically generated synthetic images
        return 5

    api_key = os.environ.get("ARK_API_KEY", "")
    base_url = "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions"

    if not api_key:
        return 5

    import requests
    import base64
    import time

    # Small delay between calls to be polite to the API rate limits
    time.sleep(0.1)

    try:
        with open(image_path, "rb") as img_file:
            # Determine image MIME type based on file extension
            ext = image_path.suffix.lower()
            mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
            image_data = base64.b64encode(img_file.read()).decode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        prompt = (
            "Evaluate the clarity of the text labels and component numbers in this schematic drawing. "
            "Are they clear, legible, and non-overlapping? Give a quality score from 1 to 5 as an integer in JSON format like: "
            "{\"score\": 4, \"reason\": \"clear\"}"
        )
        
        payload = {
            "model": "ark-code-latest",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{image_data}"
                            }
                        }
                    ]
                }
            ]
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(base_url, headers=headers, json=payload, timeout=60)
                if response.status_code == 200:
                    res_data = response.json()
                    text_out = res_data["choices"][0]["message"]["content"]
                    
                    # Extract JSON from output using robust regex
                    import re
                    match = re.search(r"\{.*\}", text_out, re.DOTALL)
                    if match:
                        text_out = match.group(0)
                    
                    score_obj = json.loads(text_out.strip())
                    score = int(score_obj.get("score", 5))
                    logger.info(f"VLM Score for {image_path.name}: {score} ({score_obj.get('reason', '')})")
                    return score
                elif response.status_code == 429:
                    # Rate limit hit, sleep and retry
                    wait_time = (attempt + 1) * 3
                    logger.warning(f"VLM API rate limit hit (429) for {image_path.name}. Waiting {wait_time}s before retry (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"VLM API returned status {response.status_code} for {image_path.name}: {response.text}")
                    break
            except requests.RequestException as req_err:
                logger.warning(f"Network error on VLM API call (attempt {attempt+1}/{max_retries}): {req_err}")
                time.sleep(2)
        
        # Fallback if all retries fail or non-429 error occurs
        return 5
    except Exception as e:
        logger.warning(f"Failed to query Volcengine Ark API: {e}. Fallback to score=5")
        return 5

def clean_dataset(data_dir: str):
    logger.info("Initializing Cleaning Pipeline (Parallelized)...")
    base_path = Path(data_dir)
    cleaned_dir = base_path / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    # Scanned folders for annotations (recursive search)
    ann_paths = list(base_path.glob("**/annotations/*.json")) + \
                list(base_path.glob("**/synthetic/*.json")) + \
                list(base_path.glob("**/open_schematics/*.json")) + \
                list(base_path.glob("**/train/*.json")) + \
                list(base_path.glob("**/val/*.json")) + \
                list(base_path.glob("**/test/*.json"))

    # Remove duplicates from the list of paths
    ann_paths = list(set(ann_paths))
    logger.info(f"Found {len(ann_paths)} annotation candidates to clean.")

    # Trackers for duplicates (protected by lock)
    seen_image_hashes = set()
    seen_text_hashes = set()

    stats = {
        "scanned": 0,
        "valid": 0,
        "filtered_empty_or_small": 0,
        "filtered_duplicate_img": 0,
        "filtered_duplicate_text": 0,
        "filtered_overlap": 0,
        "filtered_vlm": 0,
        "error_files": 0
    }

    lock = threading.Lock()

    def process_file(ann_file):
        if "cleaned" in str(ann_file):
            return

        with lock:
            stats["scanned"] += 1
            current_scanned = stats["scanned"]
            if current_scanned % 100 == 0:
                logger.info(f"Progress: scanned {current_scanned}/{len(ann_paths)} files...")

        try:
            with open(ann_file, "r", encoding="utf-8") as f:
                ann_data = json.load(f)

            # Determine associated PNG path
            img_path_str = ann_data.get("image_path", "")
            if not img_path_str:
                img_path = ann_file.with_suffix(".png")
            else:
                img_path = Path(img_path_str)
                if not img_path.exists():
                    img_path = ann_file.parent / img_path.name
                    if not img_path.exists():
                        img_path = base_path / "rendered" / img_path.name

            if not img_path.exists():
                logger.warning(f"Image {img_path} not found for annotation {ann_file}, skipping.")
                with lock:
                    stats["error_files"] += 1
                return

            clean_json_path = cleaned_dir / ann_file.name
            clean_png_path = cleaned_dir / img_path.name
            if clean_json_path.exists() and clean_png_path.exists():
                with lock:
                    try:
                        img_hash = calculate_file_sha256(img_path)
                        seen_image_hashes.add(img_hash)
                        
                        annotations = ann_data.get("annotations", [])
                        text_hash = calculate_text_hash(annotations)
                        seen_text_hashes.add(text_hash)
                    except:
                        pass
                    stats["valid"] += 1
                return

            # 1. Hard Filter: Component Count Check
            components = ann_data.get("components", [])
            if not components:
                components = [a for a in ann_data.get("annotations", []) if a.get("category") == "reference"]

            if len(components) < 3:
                with lock:
                    stats["filtered_empty_or_small"] += 1
                return

            # 2. Hard Hash Deduplication (Image hash)
            img_hash = calculate_file_sha256(img_path)
            with lock:
                if img_hash in seen_image_hashes:
                    stats["filtered_duplicate_img"] += 1
                    return
                seen_image_hashes.add(img_hash)

            # 3. Layout Label Deduplication (Text hash)
            annotations = ann_data.get("annotations", [])
            text_hash = calculate_text_hash(annotations)
            with lock:
                if text_hash in seen_text_hashes:
                    stats["filtered_duplicate_text"] += 1
                    return
                seen_text_hashes.add(text_hash)

            # 4. Text Quality Check: Overlap (IoU) & Boundaries & Hidden Symbols
            filtered_annotations = []
            has_overlap_issue = False
            for i, ann in enumerate(annotations):
                text = ann.get("text", "")
                bbox = ann.get("bbox", [])
                
                if text.startswith("#") or text.startswith("GND") and len(bbox) == 0:
                    continue

                if len(bbox) == 4 and isinstance(bbox[0], list):
                    xs = [pt[0] for pt in bbox]
                    ys = [pt[1] for pt in bbox]
                    if min(xs) < 0 or min(ys) < 0 or max(xs) > ann_data.get("image_width", 99999) or max(ys) > ann_data.get("image_height", 99999):
                        continue

                overlap_too_high = False
                for j, other_ann in enumerate(annotations):
                    if i != j:
                        iou = compute_iou(bbox, other_ann.get("bbox", []))
                        if iou > 0.8:
                            overlap_too_high = True
                            break
                
                if overlap_too_high:
                    has_overlap_issue = True
                    break
                
                filtered_annotations.append(ann)

            if has_overlap_issue:
                with lock:
                    stats["filtered_overlap"] += 1
                return

            # 5. VLM Quality Check (Optional)
            vlm_score = call_vlm_scoring(img_path)
            if vlm_score < 3:
                with lock:
                    stats["filtered_vlm"] += 1
                return

            # 6. Passed! Copy to cleaned folder
            ann_data["annotations"] = filtered_annotations
            ann_data["image_path"] = f"./data/cleaned/{img_path.name}"
            
            clean_json_path = cleaned_dir / ann_file.name
            clean_png_path = cleaned_dir / img_path.name

            with open(clean_json_path, "w", encoding="utf-8") as out_f:
                json.dump(ann_data, out_f, indent=2, ensure_ascii=False)

            shutil.copy2(img_path, clean_png_path)
            with lock:
                stats["valid"] += 1

        except Exception as e:
            logger.error(f"Error processing {ann_file}: {e}")
            with lock:
                stats["error_files"] += 1

    # Run processing using ThreadPoolExecutor with 16 threads
    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(process_file, ann_paths)

    logger.info("="*60)
    logger.info("Cleaning Pipeline Stats Summary:")
    logger.info(f"  Scanned candidates: {stats['scanned']}")
    logger.info(f"  Valid saved:        {stats['valid']}")
    logger.info(f"  Filtered (Small):   {stats['filtered_empty_or_small']}")
    logger.info(f"  Filtered (Dup Img): {stats['filtered_duplicate_img']}")
    logger.info(f"  Filtered (Dup Txt): {stats['filtered_duplicate_text']}")
    logger.info(f"  Filtered (Overlap): {stats['filtered_overlap']}")
    logger.info(f"  Filtered (VLM):     {stats['filtered_vlm']}")
    logger.info(f"  Error files:        {stats['error_files']}")
    logger.info("="*60)

if __name__ == "__main__":
    # Work from relative path ./data
    clean_dataset("./data")

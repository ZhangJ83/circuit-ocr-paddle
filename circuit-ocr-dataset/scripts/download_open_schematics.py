#!/usr/bin/env python3
"""
Open Schematics Downloader & Preprocessor (Direct Parquet Version)
==================================================================
This script downloads parquet shards of the bshada/open-schematics dataset directly
via Hugging Face mirror, parses them using pandas/pyarrow, renders/saves the images,
and parses their KiCad metadata into standard annotations with estimated pixel-level bounding boxes.

All paths are relative to the project root.
"""

import os
import json
import logging
import io
import requests
from pathlib import Path
import pandas as pd
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("download_open_schematics")

def estimate_text_bbox(x_mm, y_mm, text, scale):
    """
    Estimate the pixel-level bounding box of text from KiCad mm coordinates.
    Default KiCad font dimensions:
      - Height: 1.27 mm
      - Char width: ~0.65 mm per character
    """
    text_len = len(str(text))
    w_mm = max(1.0, text_len * 0.65)
    h_mm = 1.27

    x1 = x_mm * scale
    y1 = (y_mm - h_mm) * scale
    x2 = (x_mm + w_mm) * scale
    y2 = y_mm * scale

    return [
        [int(x1), int(y1)],
        [int(x2), int(y1)],
        [int(x2), int(y2)],
        [int(x1), int(y2)]
    ]

def download_shard(shard_idx: int, data_dir: Path) -> Path:
    """Download a single parquet shard directly from Hugging Face mirror."""
    filename = f"train-{shard_idx:05d}-of-00078.parquet"
    url = f"https://hf-mirror.com/datasets/bshada/open-schematics/resolve/main/data/{filename}"
    local_path = data_dir / filename

    if local_path.exists():
        logger.info(f"Parquet shard {filename} already exists, skipping download.")
        return local_path

    logger.info(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Write to local file chunk-by-chunk to save RAM
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"Successfully downloaded {filename} ({local_path.stat().st_size / (1024*1024):.2f} MB)")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download shard {filename}: {e}")
        if local_path.exists():
            local_path.unlink()
        raise e

def process_parquet_file(file_path: Path, output_dir: Path, max_samples_per_shard: int = 1500):
    logger.info(f"Reading parquet file: {file_path}")
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded {len(df)} rows from parquet.")

    count = 0
    for idx, row in df.iterrows():
        if count >= max_samples_per_shard:
            break

        name = row.get("name")
        if not name:
            name = f"open_sch_{file_path.stem}_{idx:05d}"
        
        name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")

        img_data = row.get("image")
        metadata_str = row.get("json")

        if img_data is None or metadata_str is None:
            continue

        try:
            # Extract image bytes
            image_bytes = None
            if isinstance(img_data, bytes):
                image_bytes = img_data
            elif isinstance(img_data, dict) and "bytes" in img_data:
                image_bytes = img_data["bytes"]
            
            if not image_bytes:
                continue

            # Load image
            img = Image.open(io.BytesIO(image_bytes))

            # Parse metadata
            if isinstance(metadata_str, str):
                metadata = json.loads(metadata_str)
            else:
                metadata = metadata_str

            # Save PNG image
            img_filename = f"{name}.png"
            img_save_path = output_dir / img_filename
            img.save(img_save_path, format="PNG")

            # Scale calculation (default to A4 width 297mm if paperWidth not specified or invalid)
            paper_w_mm = float(metadata.get("paperWidth", 297.0))
            if paper_w_mm <= 0:
                paper_w_mm = 297.0
            scale = img.width / paper_w_mm

            # Map symbols & annotations
            annotations = []
            components = []

            symbols = metadata.get("schematicSymbols", [])
            for sym in symbols:
                ref = ""
                val = ""
                sym_type = sym.get("entryName", "Unknown")
                
                properties = sym.get("properties", [])
                for prop in properties:
                    key = prop.get("key", "")
                    value = prop.get("value", "")
                    pos = prop.get("position", {})

                    if key == "Reference":
                        ref = value
                    elif key == "Value":
                        val = value

                    if pos and "x" in pos and "y" in pos:
                        x = float(pos["x"])
                        y = float(pos["y"])
                        bbox = estimate_text_bbox(x, y, value, scale)
                        annotations.append({
                            "text": str(value),
                            "bbox": bbox,
                            "category": key.lower(),
                            "component_ref": ref
                        })

                if ref:
                    components.append({
                        "ref": ref,
                        "value": val,
                        "type": sym_type
                    })

            # Map global labels
            labels = metadata.get("globalLabels", [])
            for lbl in labels:
                text = lbl.get("text", "")
                pos = lbl.get("position", {})
                if text and pos and "x" in pos and "y" in pos:
                    x = float(pos["x"])
                    y = float(pos["y"])
                    bbox = estimate_text_bbox(x, y, text, scale)
                    annotations.append({
                        "text": str(text),
                        "bbox": bbox,
                        "category": "net_label"
                    })

            # Save JSON annotation
            ann_filename = f"{name}.json"
            ann_save_path = output_dir / ann_filename

            ann_data = {
                "image_path": f"./data/raw/open_schematics/{img_filename}",
                "image_width": img.width,
                "image_height": img.height,
                "annotations": annotations,
                "components": components
            }

            with open(ann_save_path, "w", encoding="utf-8") as f:
                json.dump(ann_data, f, indent=2, ensure_ascii=False)

            count += 1
            if count % 100 == 0:
                logger.info(f"Processed {count} samples from {file_path.name}...")

        except Exception as e:
            logger.warning(f"Failed to process sample {name} in {file_path.name}: {e}")
            continue

    logger.info(f"Finished processing {file_path.name}: extracted {count} samples.")
    return count

def download_and_preprocess(data_dir: str, num_shards: int = 3, max_samples_per_shard: int = 1500):
    base_path = Path(data_dir)
    raw_dir = base_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    open_schematics_dir = raw_dir / "open_schematics"
    open_schematics_dir.mkdir(parents=True, exist_ok=True)

    total_extracted = 0
    for shard_idx in range(num_shards):
        try:
            parquet_path = download_shard(shard_idx, raw_dir)
            extracted = process_parquet_file(parquet_path, open_schematics_dir, max_samples_per_shard)
            total_extracted += extracted
            
            # Clean up the large parquet file to save disk space
            if parquet_path.exists():
                parquet_path.unlink()
                logger.info(f"Cleaned up temporary parquet file: {parquet_path.name}")
                
        except Exception as e:
            logger.error(f"Failed to process shard {shard_idx}: {e}")
            continue

    logger.info(f"Open Schematics pipeline complete. Total extracted: {total_extracted} samples.")

if __name__ == "__main__":
    # Download and extract the first 3 shards (yielding up to 4,500 samples)
    download_and_preprocess("./data", num_shards=3, max_samples_per_shard=1500)

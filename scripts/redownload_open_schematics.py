"""
Re-download bshada/open-schematics from HuggingFace.
Correct column names: schematic_image, schematic_json, schematic (kicad_sch source!)
Batch download: 10 parquets per run.
"""
import json, io, os, logging, sys, time
from pathlib import Path
from PIL import Image
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download")

PROJECT_DIR = Path(__file__).parent.parent
DATASET_DIR = PROJECT_DIR / "circuit-ocr-dataset"
OUT_DIR = DATASET_DIR / "data" / "open_schematics_v2"
IMG_DIR = OUT_DIR / "images"
SRC_DIR = OUT_DIR / "kicad_sch"  # Save .kicad_sch source files
PARQUET_DIR = OUT_DIR / "parquet"
for d in [IMG_DIR, SRC_DIR, PARQUET_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MAX_SAMPLES = 2000
BATCH_SIZE = 10
BATCH_START = 0

BASE_URLS = [
    "https://huggingface.co/datasets/bshada/open-schematics/resolve/main/data",
    "https://hf-mirror.com/datasets/bshada/open-schematics/resolve/main/data",
]

BATCH_END = BATCH_START + BATCH_SIZE
logger.info(f"=== Batch: shards {BATCH_START} to {BATCH_END-1} ===")

session = requests.Session()
total = 0
all_samples = []
start_time = time.time()

for shard_idx in range(BATCH_START, BATCH_END):
    if total >= MAX_SAMPLES:
        break

    fname = f"train-{shard_idx:05d}.parquet"
    local_path = PARQUET_DIR / fname

    # Delete corrupted 0-byte files from previous attempts
    if local_path.exists() and local_path.stat().st_size == 0:
        local_path.unlink()

    if not local_path.exists():
        downloaded = False
        for base_url in BASE_URLS:
            url = f"{base_url}/{fname}"
            logger.info(f"[{shard_idx}] Downloading {fname} from {base_url.split('/')[2]}...")
            try:
                resp = session.get(url, stream=True, timeout=120)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                total_size = int(resp.headers.get("content-length", 0))
                with open(local_path, "wb") as f:
                    dl = 0
                    last_report = 0
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                        dl += len(chunk)
                        if total_size > 0 and dl - last_report >= total_size // 10:
                            pct = dl * 100 // total_size
                            print(f"\r  {dl//1024//1024}MB/{total_size//1024//1024}MB ({pct}%)", end="")
                            sys.stdout.flush()
                            last_report = dl
                print(f"\r  {total_size//1024//1024}MB done.     ")
                sys.stdout.flush()
                downloaded = True
                break
            except Exception as e:
                logger.warning(f"  {base_url.split('/')[2]}: {e}")
                continue

        if not downloaded:
            logger.error(f"  FAILED to download {fname}")
            break
    else:
        sz = local_path.stat().st_size / 1024 / 1024
        logger.info(f"[{shard_idx}] {fname} exists ({sz:.1f}MB), processing...")

    # Process
    try:
        df = pd.read_parquet(local_path)
    except Exception as e:
        logger.error(f"  Corrupted parquet: {e}, re-downloading...")
        local_path.unlink()
        continue

    shard_count = 0
    for idx, row in df.iterrows():
        if total >= MAX_SAMPLES:
            break

        raw_name = str(row.get("name", f"sch_{shard_idx}_{idx}"))
        name = raw_name.replace(" ", "_").replace("/", "_").replace("\\", "_")

        img_data = row.get("schematic_image")  # dict with 'bytes'
        json_str = row.get("schematic_json")   # JSON string
        sch_content = row.get("schematic")     # raw .kicad_sch S-expression

        if img_data is None or json_str is None:
            continue

        try:
            # Extract image
            if isinstance(img_data, dict) and "bytes" in img_data:
                image_bytes = img_data["bytes"]
            else:
                continue
            img = Image.open(io.BytesIO(image_bytes))

            # Parse JSON metadata
            if isinstance(json_str, str):
                metadata = json.loads(json_str)
            else:
                metadata = json_str

            # Save PNG image
            img_filename = f"{name}.png"
            img.save(IMG_DIR / img_filename, format="PNG")

            # Save .kicad_sch source file
            if sch_content:
                sch_filename = f"{name}.kicad_sch"
                with open(SRC_DIR / sch_filename, "w", encoding="utf-8") as f:
                    f.write(sch_content)

            # --- Extract correct ref/value pairs ---
            components = []
            symbols = metadata.get("schematicSymbols", [])
            for sym in symbols:
                ref = ""
                val = ""
                for prop in sym.get("properties", []):
                    k = prop.get("key", "")
                    v = prop.get("value", "")
                    if k == "Reference":
                        ref = v
                    elif k == "Value":
                        val = v
                if ref:
                    components.append((ref, val))

            # Global labels (power nets, signal names)
            labels = []
            for lbl in metadata.get("globalLabels", []):
                t = lbl.get("text", "")
                if t:
                    labels.append(t)

            # Build clean GT text
            gt_lines = []
            for ref, val in components:
                gt_lines.append(ref)
                gt_lines.append(val if val else "")
            for lbl in labels:
                gt_lines.append(lbl)

            sample = {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "image", "image": f"./images/{img_filename}"},
                        {"type": "text", "text": "Please identify all the components shown and their values in this circuit diagram."},
                    ]},
                    {"role": "assistant", "content": "\n".join(gt_lines)},
                ],
                "images": [f"./images/{img_filename}"],
            }
            all_samples.append(sample)
            total += 1
            shard_count += 1

        except Exception:
            continue

    elapsed = time.time() - start_time
    rate = total / elapsed * 60 if elapsed > 0 else 0
    eta_str = f"ETA ~{(MAX_SAMPLES-total)/rate:.0f}min" if rate > 0 and total < MAX_SAMPLES else ""
    logger.info(f"  [{shard_idx}] +{shard_count} samples | Total: {total}/{MAX_SAMPLES} | {rate:.0f}/min {eta_str}")

# Save JSONL (append mode to accumulate across batches)
jsonl_path = OUT_DIR / "train_clean.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as f:
    for s in all_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

# Save progress
progress_path = OUT_DIR / "download_progress.json"
with open(progress_path, "w") as f:
    json.dump({"last_batch_end": BATCH_END, "total_samples": total}, f)

total_time = time.time() - start_time
logger.info(f"=== BATCH DONE: {total} samples in {total_time/60:.1f}min ===")
logger.info(f"  JSONL: {jsonl_path}")
logger.info(f"  Images: {IMG_DIR} ({len(list(IMG_DIR.glob('*.png')))} PNGs)")
logger.info(f"  Source: {SRC_DIR} ({len(list(SRC_DIR.glob('*.kicad_sch')))} .kicad_sch)")
logger.info(f"  To continue: change BATCH_START to {BATCH_END}")

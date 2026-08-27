"""
Batch GT generator for all .kicad_sch files + continue downloading parquet shards.
"""
import json, os, sys, time, io
from pathlib import Path
import pandas as pd
from PIL import Image

PROJECT_DIR = Path(__file__).parent.parent
DATASET_DIR = PROJECT_DIR / "circuit-ocr-dataset"
SRC_DIR = DATASET_DIR / "data" / "open_schematics_v2" / "numbered" / "source"
IMG_DIR = DATASET_DIR / "data" / "open_schematics_v2" / "numbered" / "images"
GT_OUT_DIR = PROJECT_DIR / "output" / "gt_clean"
PARQUET_DIR = DATASET_DIR / "data" / "open_schematics_v2" / "parquet"
IMG_OUT = DATASET_DIR / "data" / "open_schematics_v2" / "images"
SRC_OUT = DATASET_DIR / "data" / "open_schematics_v2" / "kicad_sch"
GT_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Import our GT generator
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
from generate_gt_from_kicad import parse_kicad_sch_full as gen_gt

TARGET_SAMPLES = 1000
BATCH_SIZE = 10  # parquet shards per batch


def batch_generate_existing():
    """Generate GT for all existing .kicad_sch files."""
    src_files = sorted(SRC_DIR.glob("*.kicad_sch"))
    all_samples = []

    print(f"Processing {len(src_files)} existing .kicad_sch files...")
    start = time.time()

    for i, sch_path in enumerate(src_files):
        try:
            gt, groups, standalone, blocks = gen_gt(str(sch_path))
            all_samples.append({
                'gt': gt,
                'source': sch_path.name,
                'lines': len([l for l in gt.splitlines() if l.strip()]),
            })
        except Exception as e:
            print(f"  ERROR [{sch_path.name}]: {e}")
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed * 60
            print(f"  {i+1}/{len(src_files)} ({rate:.0f}/min)")

    return all_samples


def download_batch(start_shard):
    """Download a batch of 10 parquet shards."""
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HOME"] = "C:/Users/zzz/.cache/huggingface"
    from huggingface_hub import hf_hub_download, list_repo_files
    import requests

    repo_id = "bshada/open-schematics"

    downloaded = 0
    new_samples = 0

    for shard_idx in range(start_shard, start_shard + BATCH_SIZE):
        fname = f"train-{shard_idx:05d}.parquet"
        local_path = PARQUET_DIR / fname

        # Download if not exists
        if not local_path.exists() or local_path.stat().st_size == 0:
            url = f"https://hf-mirror.com/datasets/{repo_id}/resolve/main/data/{fname}"
            print(f"  [{shard_idx}] Downloading {fname}...")
            try:
                resp = requests.get(url, stream=True, timeout=120)
                if resp.status_code == 404:
                    # Try main HF
                    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/data/{fname}"
                    resp = requests.get(url, stream=True, timeout=120)
                    if resp.status_code == 404:
                        print(f"    Not found, stopping shard download.")
                        break
                resp.raise_for_status()
                total_size = int(resp.headers.get("content-length", 0))
                with open(local_path, "wb") as f:
                    dl = 0
                    for chunk in resp.iter_content(65536):
                        f.write(chunk)
                        dl += len(chunk)
                        if total_size > 0 and dl % (total_size // 5 + 1) < 65536:
                            pct = dl * 100 // total_size
                            print(f"\r    {dl//1024//1024}MB/{total_size//1024//1024}MB ({pct}%)", end="", flush=True)
                print(f"\r    {total_size//1024//1024}MB done.    ")
            except Exception as e:
                print(f"    Failed: {e}")
                break
        else:
            print(f"  [{shard_idx}] {fname} already exists ({local_path.stat().st_size//1024//1024}MB)")

        # Process parquet
        try:
            df = pd.read_parquet(local_path)
        except:
            print(f"    Corrupted, skipping")
            continue

        shard_count = 0
        for idx, row in df.iterrows():
            img_data = row.get("schematic_image")
            json_str = row.get("schematic_json")
            sch_content = row.get("schematic")
            raw_name = str(row.get("name", f"sch_{shard_idx}_{idx}"))
            name = raw_name.replace(" ", "_").replace("/", "_").replace("\\", "_")

            if img_data is None or json_str is None:
                continue

            try:
                # Save image
                if isinstance(img_data, dict) and "bytes" in img_data:
                    img_bytes = img_data["bytes"]
                else:
                    continue

                img = Image.open(io.BytesIO(img_bytes))
                img_filename = f"{name}.png"
                img.save(IMG_OUT / img_filename, format="PNG")

                # Save .kicad_sch source
                if sch_content:
                    sch_filename = f"{name}.kicad_sch"
                    with open(SRC_OUT / sch_filename, "w", encoding="utf-8") as f:
                        f.write(sch_content)

                new_samples += 1
                shard_count += 1

            except Exception:
                continue

        print(f"    +{shard_count} samples, total new: {new_samples}")

    return new_samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--start-shard", type=int, default=10)
    args = parser.parse_args()

    total = 0

    if not args.download_only:
        # Generate GT for existing files
        existing = batch_generate_existing()
        total = len(existing)

        # Save batched JSONL
        jsonl_path = GT_OUT_DIR / "train_gt_batch.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for s in existing:
                f.write(json.dumps({"gt": s['gt'], "source": s['source']}, ensure_ascii=False) + "\n")

        # Save individual files
        for s in existing:
            out_path = GT_OUT_DIR / s['source'].replace('.kicad_sch', '.txt')
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(s['gt'])

        print(f"\nGenerated GT for {total} samples")
        print(f"  JSONL: {jsonl_path}")
        print(f"  Individual: {GT_OUT_DIR}/")

    if not args.generate_only:
        # Continue downloading
        shard = args.start_shard
        while total < TARGET_SAMPLES:
            print(f"\n=== Downloading shards {shard}-{shard+BATCH_SIZE-1} (have {total}/{TARGET_SAMPLES}) ===")
            new = download_batch(shard)
            if new == 0:
                print("No more samples available.")
                break
            total += new
            shard += BATCH_SIZE

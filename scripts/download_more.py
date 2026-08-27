"""Continue downloading parquet shards and extract samples."""
import os, io, time, json, sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "C:/Users/zzz/.cache/huggingface"
import requests, pandas as pd
from PIL import Image
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent / "circuit-ocr-dataset"
PARQUET_DIR = DATASET_DIR / "data/open_schematics_v2/parquet"
IMG_OUT = DATASET_DIR / "data/open_schematics_v2/images"
SRC_OUT = DATASET_DIR / "data/open_schematics_v2/kicad_sch"
for d in [PARQUET_DIR, IMG_OUT, SRC_OUT]:
    d.mkdir(parents=True, exist_ok=True)

TARGET = 1000
START_SHARD = 10
REPO_ID = "bshada/open-schematics"

existing = len(list(IMG_OUT.glob("*.png")))
print(f"Existing samples: {existing}")
total = existing

for shard in range(START_SHARD, 999):
    if total >= TARGET:
        break

    fname = f"train-{shard:05d}.parquet"
    local = PARQUET_DIR / fname

    if not local.exists() or local.stat().st_size == 0:
        urls = [
            f"https://hf-mirror.com/datasets/{REPO_ID}/resolve/main/data/{fname}",
            f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/data/{fname}",
        ]
        downloaded = False
        for url in urls:
            try:
                print(f"[{shard}] Downloading {fname}...", flush=True)
                resp = requests.get(url, stream=True, timeout=120)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                sz = int(resp.headers.get("content-length", 0))
                with open(local, "wb") as f:
                    dl = 0
                    for chunk in resp.iter_content(65536):
                        f.write(chunk)
                        dl += len(chunk)
                        if sz > 0:
                            pct = dl * 100 // sz
                            print(f"\r  {dl//1024//1024}MB/{sz//1024//1024}MB ({pct}%)", end="", flush=True)
                if sz > 0:
                    print(f"\r  {sz//1024//1024}MB done.    ")
                downloaded = True
                break
            except Exception as e:
                continue
        if not downloaded:
            print(f"  All URLs failed, stopping download.")
            break
    else:
        sz = local.stat().st_size // 1024 // 1024
        print(f"[{shard}] {fname} exists ({sz}MB)")

    # Process
    try:
        df = pd.read_parquet(local)
    except Exception:
        print(f"  Corrupted, skipping")
        continue

    added = 0
    for idx, row in df.iterrows():
        if total >= TARGET:
            break
        img_data = row.get("schematic_image")
        json_str = row.get("schematic_json")
        sch_content = row.get("schematic")
        raw_name = str(row.get("name", f"sch_{shard}_{idx}"))
        name = raw_name.replace(" ", "_").replace("/", "_").replace(chr(92), "_")

        if img_data is None or json_str is None:
            continue
        try:
            if isinstance(img_data, dict) and "bytes" in img_data:
                img_bytes = img_data["bytes"]
            else:
                continue
            img = Image.open(io.BytesIO(img_bytes))
            img.save(IMG_OUT / f"{name}.png", format="PNG")
            if sch_content:
                with open(SRC_OUT / f"{name}.kicad_sch", "w", encoding="utf-8") as f:
                    f.write(sch_content)
            added += 1
            total += 1
        except Exception:
            continue

    print(f"  +{added} = {total}/{TARGET}")

print(f"Done. Total: {total} samples")
print(f"  Images: {len(list(IMG_OUT.glob('*.png')))}")
print(f"  Sources: {len(list(SRC_OUT.glob('*.kicad_sch')))}")

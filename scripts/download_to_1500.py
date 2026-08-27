"""
Download additional parquet shards and extract clean samples to reach target.
"""
import os, io, time, json, sys, re
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "C:/Users/zzz/.cache/huggingface"
import requests, pandas as pd
from PIL import Image
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
PARQUET_DIR = PROJECT_DIR / "circuit-ocr-dataset/data/open_schematics_v2/parquet"
IMG_DIR = PROJECT_DIR / "circuit-ocr-dataset/data/open_schematics_v2/images"
SRC_DIR = PROJECT_DIR / "circuit-ocr-dataset/data/open_schematics_v2/kicad_sch"
REVIEW_IMG = PROJECT_DIR / "output/review_1000/images"
REVIEW_ANNO = PROJECT_DIR / "output/review_1000/annotations"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
from generate_gt_from_kicad import parse

REPO_ID = "bshada/open-schematics"
TARGET = 1500

# Load current state
with open(PROJECT_DIR / 'output/review_1000/mapping.json', 'r') as f:
    mapping = json.load(f)
existing_names = set(m['original_name'] for m in mapping)
max_id = max(int(m['id']) for m in mapping)
current = len(mapping)
print(f"Current: {current}, Target: {TARGET}, Need: {TARGET - current}")

# Find max shard number
max_shard = 0
for f in PARQUET_DIR.glob("train-*.parquet"):
    try:
        n = int(f.stem.split("-")[1])
        max_shard = max(max_shard, n)
    except: pass
print(f"Max shard: {max_shard}")

next_id = max_id + 1
start_shard = max_shard + 1
added = 0

for shard in range(start_shard, 999):
    if current + added >= TARGET:
        break

    fname = f"train-{shard:05d}.parquet"
    local = PARQUET_DIR / fname

    # Download
    if not local.exists():
        urls = [
            f"https://hf-mirror.com/datasets/{REPO_ID}/resolve/main/data/{fname}",
            f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/data/{fname}",
        ]
        ok = False
        for url in urls:
            try:
                print(f"[shard {shard}] Downloading {fname}...", flush=True)
                resp = requests.get(url, stream=True, timeout=180)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                total_size = int(resp.headers.get("content-length", 0))
                dl_size = 0
                last_pct = -1
                with open(local, "wb") as f:
                    for chunk in resp.iter_content(65536):
                        f.write(chunk)
                        dl_size += len(chunk)
                        if total_size > 0:
                            pct = dl_size * 100 // total_size
                            if pct // 10 != last_pct // 10:
                                print(f"\r  {dl_size//1024//1024}MB/{total_size//1024//1024}MB ({pct}%)", end="", flush=True)
                                last_pct = pct
                if total_size > 0:
                    print(f"\r  {total_size//1024//1024}MB done.    ", flush=True)
                ok = True
                break
            except Exception as e:
                continue
        if not ok:
            print(f"  No more shards available (stopped at {shard-1})")
            break

    # Process
    try:
        df = pd.read_parquet(local)
    except Exception as e:
        print(f"  Corrupted parquet, skipping: {e}")
        continue

    batch_added = 0
    for idx, row in df.iterrows():
        if current + added + batch_added >= TARGET:
            break

        img_data = row.get("schematic_image")
        sch_content = row.get("schematic")
        raw_name = str(row.get("name", f"sch_{shard}_{idx}"))
        name = raw_name.replace(" ", "_").replace("/", "_").replace(chr(92), "_")

        # Skip if already processed
        if name in existing_names:
            continue

        if img_data is None or sch_content is None:
            continue

        try:
            # Extract image
            if isinstance(img_data, dict) and "bytes" in img_data:
                img_bytes = img_data["bytes"]
            else:
                continue

            # Save image
            img_path = IMG_DIR / f"{name}.png"
            if not img_path.exists():
                img = Image.open(io.BytesIO(img_bytes))
                img.save(img_path, format="PNG")

            # Save source
            sch_path = SRC_DIR / f"{name}.kicad_sch"
            if not sch_path.exists():
                with open(sch_path, "w", encoding="utf-8") as f:
                    f.write(sch_content)

            # Generate GT
            gt, blocks = parse(str(sch_path))

            # Filter: skip User paper
            if '\nSize: User\n' in gt:
                continue

            # Filter: skip ? refs
            if re.search(r'(?<!\w)\?\n', gt) or re.search(r'\n\?', gt):
                continue

            # Filter: too few lines
            gt_lines = [l for l in gt.splitlines() if l.strip()]
            if len(gt_lines) < 5:
                continue

            # Add to review
            uid = f"{next_id:04d}"
            with open(REVIEW_ANNO / f"{uid}.txt", 'w', encoding='utf-8') as f:
                f.write(gt)
            import shutil
            shutil.copy2(img_path, REVIEW_IMG / f"{uid}.png")

            mapping.append({"id": uid, "original_name": name})
            existing_names.add(name)
            next_id += 1
            batch_added += 1

        except Exception as e:
            continue

    added += batch_added
    print(f"  shard {shard}: +{batch_added} (total {current + added}/{TARGET})")
    df = None  # free memory

# Save mapping
with open(PROJECT_DIR / 'output/review_1000/mapping.json', 'w') as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)

print(f"\nDone. Final: {len(mapping)} samples")
print(f"  Annotations: {len(list(REVIEW_ANNO.glob('*.txt')))}")
print(f"  Images: {len(list(REVIEW_IMG.glob('*.png')))}")

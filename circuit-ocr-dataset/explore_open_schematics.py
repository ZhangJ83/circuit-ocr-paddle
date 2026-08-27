"""
Explore Open Schematics dataset via HF mirror.
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
import json
from PIL import Image
import io

print("=" * 60)
print("Downloading samples from bshada/open-schematics...")
print("=" * 60)

# Load just the first parquet shard in streaming mode
ds = load_dataset(
    "bshada/open-schematics",
    split="train",
    streaming=True,
)

# Collect 20 samples with different characteristics
samples = []
for s in ds:
    samples.append(s)
    if len(samples) >= 20:
        break

print(f"\nDownloaded {len(samples)} samples")
print(f"Available keys: {sorted(samples[0].keys())}")

# Analyze each field
print("\n" + "=" * 60)
print("FIELD ANALYSIS")
print("=" * 60)

for key in sorted(samples[0].keys()):
    non_none = sum(1 for s in samples if s.get(key) is not None)
    types = set()
    sizes = []
    for s in samples:
        v = s.get(key)
        if v is not None:
            types.add(type(v).__name__)
            if isinstance(v, (str, bytes)):
                sizes.append(len(v))
            elif isinstance(v, dict):
                sizes.append(len(json.dumps(v)))

    avg_size = sum(sizes) // len(sizes) if sizes else 0
    print(f"  {key:25s}  non-null: {non_none}/{len(samples)}  "
          f"types: {types}  avg_size: {avg_size} bytes")

# Deep dive into key fields
print("\n" + "=" * 60)
print("DEEP DIVE: image field")
print("=" * 60)
for i, s in enumerate(samples):
    img = s.get("image")
    if img is not None:
        if hasattr(img, 'size'):
            print(f"  [{i}] {s.get('name','?')}: {img.size} mode={getattr(img,'mode','?')}")
        elif isinstance(img, dict):
            print(f"  [{i}] {s.get('name','?')}: dict with keys={list(img.keys())}")
        elif isinstance(img, bytes):
            try:
                pil_img = Image.open(io.BytesIO(img))
                print(f"  [{i}] {s.get('name','?')}: {pil_img.size} mode={pil_img.mode} (from bytes)")
            except:
                print(f"  [{i}] {s.get('name','?')}: {len(img)} bytes (not a valid image)")
        else:
            print(f"  [{i}] {s.get('name','?')}: {type(img).__name__} len={len(str(img))}")

print("\n" + "=" * 60)
print("DEEP DIVE: components_used")
print("=" * 60)
for i, s in enumerate(samples):
    comps = s.get("components_used")
    if comps is not None:
        preview = str(comps)[:200]
        n_comps = len(comps) if isinstance(comps, (list, dict)) else "?"
        print(f"  [{i}] {n_comps} components: {preview}")
    else:
        print(f"  [{i}] None")

print("\n" + "=" * 60)
print("DEEP DIVE: json / yaml fields")
print("=" * 60)
for i, s in enumerate(samples):
    jdata = s.get("json")
    ydata = s.get("yaml")
    if jdata:
        if isinstance(jdata, dict):
            print(f"  [{i}] json keys: {list(jdata.keys())[:10]}")
        else:
            print(f"  [{i}] json: {str(jdata)[:150]}")
    if ydata:
        print(f"  [{i}] yaml: {str(ydata)[:150]}")

print("\n" + "=" * 60)
print("DEEP DIVE: schematic (raw file)")
print("=" * 60)
for i, s in enumerate(samples):
    sch = s.get("schematic")
    if sch:
        preview = str(sch)[:200]
        print(f"  [{i}] {len(str(sch))} bytes: {preview}")

print("\n" + "=" * 60)
print("CATEGORY / TYPE DISTRIBUTION")
print("=" * 60)
names = [str(s.get('name','?')) for s in samples]
types = [str(s.get('type','?')) for s in samples]
for n, t in zip(names, types):
    print(f"  {n:40s}  type={t}")

# Save one full sample as JSON for reference
print("\n" + "=" * 60)
print("Saving one full sample...")
print("=" * 60)
s0 = samples[0]
clean = {}
for k, v in s0.items():
    if hasattr(v, 'size'):
        clean[k] = f"<PIL.Image {v.size} {v.mode}>"
    elif isinstance(v, bytes):
        clean[k] = f"<bytes len={len(v)}>"
    elif isinstance(v, str) and len(v) > 500:
        clean[k] = v[:500] + "..."
    else:
        clean[k] = v

with open('open_schematics_sample.json', 'w', encoding='utf-8') as f:
    json.dump(clean, f, indent=2, ensure_ascii=False)
print("Saved to open_schematics_sample.json")

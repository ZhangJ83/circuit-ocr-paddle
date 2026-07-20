"""Quick status check on cloud."""
import os, json
from PIL import Image

# Check image sizes
sizes = []
for l in open("output/train_clean.jsonl"):
    d = json.loads(l)
    img = Image.open(d["images"][0])
    sizes.append(max(img.size))

sizes.sort()
n = len(sizes)
print(f"IMAGE SIZES (longest side): N={n}")
print(f"  min={sizes[0]}, p25={sizes[n//4]}, p50={sizes[n//2]}, p75={sizes[3*n//4]}, max={sizes[-1]}")
print(f"  count >384: {sum(1 for s in sizes if s>384)}/{n} ({sum(1 for s in sizes if s>384)/n*100:.0f}%)")
print(f"  count >512: {sum(1 for s in sizes if s>512)}/{n} ({sum(1 for s in sizes if s>512)/n*100:.0f}%)")
print(f"  count >768: {sum(1 for s in sizes if s>768)}/{n} ({sum(1 for s in sizes if s>768)/n*100:.0f}%)")

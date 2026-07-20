"""Rebuild train/val/test JSONL from all 1500 review samples."""
import json, random, os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
ANNO_DIR = PROJECT_DIR / 'output/review_1000/annotations'
IMG_DIR = PROJECT_DIR / 'output/review_1000/images'
OUTPUT_DIR = PROJECT_DIR / 'output'

# Cloud-absolute image paths (used for training on cloud instance)
CLOUD_IMG_BASE = '/root/circuit_ocr/output/review_1000/images'

TRAIN_N = 1200
VAL_N = 150
TEST_N = 150

# Collect all samples
samples = []
for f in sorted(ANNO_DIR.glob('*.txt')):
    uid = f.stem
    img_path = IMG_DIR / f'{uid}.png'
    if not img_path.exists():
        continue
    gt_text = f.read_text(encoding='utf-8').strip()
    if not gt_text:
        continue
    samples.append((uid, gt_text))

print(f'Total samples: {len(samples)}')

# Shuffle deterministically
random.seed(42)
random.shuffle(samples)

# Split
train = samples[:TRAIN_N]
val = samples[TRAIN_N:TRAIN_N + VAL_N]
test = samples[TRAIN_N + VAL_N:TRAIN_N + VAL_N + TEST_N]

print(f'Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')

def build_jsonl(data, name, cloud_base):
    """Build JSONL file in the training format."""
    records = []
    for uid, gt_text in data:
        cloud_img = f'{cloud_base}/{uid}.png'
        record = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": cloud_img},
                        {"type": "text", "text": "<image>OCR:"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": gt_text
                }
            ],
            "images": [cloud_img]
        }
        records.append(record)

    out_path = OUTPUT_DIR / f'{name}_clean.jsonl'
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{name}: {len(records)} -> {out_path}')
    return out_path

build_jsonl(train, 'train', CLOUD_IMG_BASE)
build_jsonl(val, 'val', CLOUD_IMG_BASE)
build_jsonl(test, 'test', CLOUD_IMG_BASE)

# Print a sample for verification
print('\n--- Sample train record ---')
with open(OUTPUT_DIR / 'train_clean.jsonl', 'r', encoding='utf-8') as f:
    r = json.loads(f.readline())
    content = r['messages'][1]['content']
    lines = content.split('\n')
    print(f'Image: {r["images"][0]}')
    print(f'Lines: {len(lines)}')
    print('First 8:')
    for l in lines[:8]:
        print(f'  |{l}|')
    print('Last 5:')
    for l in lines[-5:]:
        print(f'  |{l}|')

"""Fix image paths in JSONL files for cloud instance."""
import json, re, os

for fname in ['output/train_clean.jsonl', 'output/val_clean.jsonl', 'output/test_clean.jsonl']:
    samples = []
    with open(fname, 'r') as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                old = s['images'][0]
                m = re.search(r'(\d{4}\.png)$', old)
                if m:
                    s['images'][0] = '/root/circuit_ocr/output/review_1000/images/' + m.group(1)
                samples.append(s)
    with open(fname, 'w') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f'{fname}: {len(samples)} samples')

with open('output/train_clean.jsonl') as f:
    s = json.loads(f.readline())
    print(f'First: {s["images"][0]}')
    print(f'Exists: {os.path.exists(s["images"][0])}')

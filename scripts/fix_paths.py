"""Fix image paths for cloud instance."""
import json, os, re

for f in ['/root/circuit_ocr/output/train_clean.jsonl',
          '/root/circuit_ocr/output/val_clean.jsonl',
          '/root/circuit_ocr/output/test_clean.jsonl']:
    samples = []
    with open(f, "r") as fh:
        for line in fh:
            if line.strip():
                s = json.loads(line)
                old_img = s["images"][0]
                m = re.search(r"(\d{4}\.png)$", old_img)
                if m:
                    s["images"][0] = "/root/circuit_ocr/output/review_1000/images/" + m.group(1)
                samples.append(s)
    with open(f, "w") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"{f}: {len(samples)} samples")

# Verify
with open("/root/circuit_ocr/output/train_clean.jsonl", "r") as f:
    s = json.loads(f.readline())
    print(f"First image: {s['images'][0]}")
    print(f"Exists: {os.path.exists(s['images'][0])}")

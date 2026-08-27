import json
train = [json.loads(l) for l in open('g:/mimo_project/circuit_ocr/circuit-ocr-dataset/ocr_vl_sft-train.jsonl', encoding='utf-8') if l.strip()]
lens = [len(s["messages"][1]["content"]) for s in train]
for threshold in [100, 150, 200, 250, 300, 500, 1000]:
    count = sum(1 for l in lens if l <= threshold)
    print(f"label <= {threshold:5d} chars: {count:5d}/{len(train)} ({100*count/len(train):.1f}%)")
print(f"Total: {len(train)}")
print(f"Min label: {min(lens)}, Max: {max(lens)}, Median: {sorted(lens)[len(lens)//2]}")

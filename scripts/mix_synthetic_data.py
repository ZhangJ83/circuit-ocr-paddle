"""Mix synthetic + real + synth-text data for training.
Maintains anti-collapse synth text ratio at ~5% (sufficient for large dataset).
"""
import json, random, os, sys

SYNTH_CIRCUITS = 'g:/mimo_project/circuit_ocr/output/synthetic_circuits/train_synthetic.jsonl'
REAL_CIRCUITS = 'g:/mimo_project/circuit_ocr/output/train_v10fmt_synth.jsonl'
OUTPUT = 'g:/mimo_project/circuit_ocr/output/train_synthetic_mix.jsonl'
TARGET_SYNTH_TEXT_RATIO = 0.05  # 5% — sufficient for 6500+ samples

def main():
    # Load synthetic circuits
    with open(SYNTH_CIRCUITS, encoding='utf-8') as f:
        synth = [json.loads(l) for l in f if l.strip()]
    print(f"Synthetic circuits: {len(synth)}")

    # Load real data, split into circuits vs synth text
    with open(REAL_CIRCUITS, encoding='utf-8') as f:
        real_all = [json.loads(l) for l in f if l.strip()]

    real_circuits = [s for s in real_all if 'synth_text_images' not in s['images'][0]]
    synth_text = [s for s in real_all if 'synth_text_images' in s['images'][0]]
    print(f"Real circuits: {len(real_circuits)}, Synth text: {len(synth_text)}")

    # Mix: synthetic circuits + real circuits
    all_circuits = synth + real_circuits
    random.shuffle(all_circuits)

    # Calculate synth text needed for target ratio
    total = len(all_circuits) + len(synth_text)
    current_ratio = len(synth_text) / total
    print(f"Current synth text ratio: {current_ratio:.1%} ({len(synth_text)}/{total})")

    if current_ratio < TARGET_SYNTH_TEXT_RATIO:
        # Duplicate synth text samples to reach target ratio
        needed = int(len(all_circuits) * TARGET_SYNTH_TEXT_RATIO / (1 - TARGET_SYNTH_TEXT_RATIO))
        extra = needed - len(synth_text)
        if extra > 0:
            print(f"Duplicating {extra} synth text samples to reach {TARGET_SYNTH_TEXT_RATIO:.0%} ratio")
            extra_text = []
            while len(extra_text) < extra:
                for s in synth_text:
                    if len(extra_text) >= extra:
                        break
                    # Slightly modify path for uniqueness
                    dup = json.loads(json.dumps(s))
                    dup['images'] = [dup['images'][0]]  # Same image, different entry
                    extra_text.append(dup)
            synth_text.extend(extra_text)
            print(f"Synth text now: {len(synth_text)}")

    # Combine
    all_data = all_circuits + synth_text
    random.shuffle(all_data)

    total = len(all_data)
    synth_ratio = len(synth_text) / total
    print(f"Final: {total} samples ({len(all_circuits)} circuits + {len(synth_text)} synth text)")
    print(f"Synth text ratio: {synth_ratio:.1%}")

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for s in all_data:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f"Saved: {OUTPUT}")

if __name__ == '__main__':
    main()

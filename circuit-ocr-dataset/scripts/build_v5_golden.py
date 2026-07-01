"""
Build V5 Golden Dataset:
  - train-real (1,357): Clean KiCad, vertical format, zero noise
  - synthetic-v3 (500):   Programmatic, draw.text() GT, 100% aligned
  - masala-chai (700):    SPICE-derived, cleaned, quality-scored
  Total: ~2,557 samples
  Split: 90/10 train/val (test uses existing easy50/100/200/full523)
"""
import json, random, sys, re
from pathlib import Path
from collections import Counter

DATASET_DIR = Path(__file__).parent.parent
random.seed(42)

MASALA_TARGET = 700
TRAIN_RATIO = 0.90

# ── Masala cleaning (same as build_balanced_dataset.py) ──
SPICE_BAD_TOKENS = {'*', ';', 'node', '.model', '.subckt', '.ends', '.tran', '.ac', '.dc', '.op'}
SPICE_BAD_PATTERNS = [
    'VALUE=', '{', '}', '<', '>', '[value]', 'W=L', 'W/L',
    'g_m', 'g_mb', 'r_pi', 'r_o', 'v(','vgs', 'vsg',
    'replace [', 'Example ', '; ', ' * ',
]

def clean_masala_label(label):
    lines = label.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(tok in stripped.split() for tok in [';', '*']):
            for sep in [';', '*']:
                idx = stripped.find(sep)
                if idx >= 0:
                    stripped = stripped[:idx].strip()
            if not stripped:
                continue
        if '{' in stripped or '}' in stripped:
            stripped = re.sub(r'\{[^}]*\}', '', stripped).strip()
        stripped = re.sub(r'VALUE\s*=\s*\S+', '', stripped).strip()
        parts = stripped.split()
        if not parts:
            continue
        ref = parts[0]
        is_comp = bool(re.match(r'^[RDLQCUMVXYIJF]\d+$', ref))
        if is_comp and len(parts) > 1:
            other = []
            for p in parts[1:]:
                if len(p) == 1 and p.isalpha():
                    continue
                if re.match(r'^[A-Za-z_]+=', p):
                    continue
                if p.replace('.','').replace('-','').isdigit() and len(p) <= 4:
                    continue
                if p == 'DC' and ref[0] in 'VI':
                    continue
                # Skip SPICE net names (net1, net23, etc.)
                if re.match(r'^net\d+$', p, re.IGNORECASE):
                    continue
                # Skip node-like tokens (vdd, vss, vcc, gnd repeated as net names)
                if re.match(r'^[Vv](dd|ss|cc|ee|bb|in|out|bias|b)\d*$', p) and len(parts) > 3:
                    continue
                other.append(p)
            stripped = ' '.join([ref] + other)
        if stripped:
            cleaned_lines.append(stripped)
    return '\n'.join(cleaned_lines)

def is_clean(label):
    tokens = set(label.split())
    for bt in SPICE_BAD_TOKENS:
        if bt in tokens:
            return False
    for pat in SPICE_BAD_PATTERNS:
        if pat in label:
            return False
    return len(label) >= 10

def score_masala(label):
    lines = [l for l in label.split('\n') if l.strip()]
    n = len(lines)
    line_score = n * 0.3 if n < 3 else (n if n <= 15 else 15 + (n-15)*0.2)
    char_score = min(len(set(label)) / 3, 20)
    prefixes = set()
    for line in lines:
        for p in line.split():
            if len(p) >= 2 and p[0] in 'RDLQCUMVXYIJF':
                prefixes.add(p[0])
    div_score = len(prefixes) * 2
    length_penalty = -5 if len(label) < 20 else (0 if len(label) < 40 else min(len(label)/20, 5))
    return line_score + char_score + div_score + length_penalty

# ── Load ──
print('=' * 60)
print('BUILDING V5 GOLDEN DATASET')
print('=' * 60)

print('\n[1] Loading train-real...')
with open(DATASET_DIR / 'ocr_vl_sft-train-real.jsonl', 'r', encoding='utf-8') as f:
    train_real = [json.loads(line) for line in f if line.strip()]
print(f'  {len(train_real)} samples')

print('\n[2] Loading synthetic-v3...')
with open(DATASET_DIR / 'ocr_vl_sft-synthetic-v3.jsonl', 'r', encoding='utf-8') as f:
    synth = [json.loads(line) for line in f if line.strip()]
print(f'  {len(synth)} samples')

print(f'\n[3] Processing Masala-CHAI (target: {MASALA_TARGET})...')
with open(DATASET_DIR / 'ocr_vl_sft-masala.jsonl', 'r', encoding='utf-8') as f:
    masala_raw = [json.loads(line) for line in f if line.strip()]

# Clean
masala_clean = []
for e in masala_raw:
    cleaned = clean_masala_label(e['messages'][1]['content'])
    if is_clean(cleaned):
        new_e = dict(e)
        new_e['messages'] = [dict(m) for m in e['messages']]
        new_e['messages'][1]['content'] = cleaned
        masala_clean.append(new_e)
print(f'  Cleaned: {len(masala_clean)} (removed {len(masala_raw) - len(masala_clean)})')

# Score and select
scored = [(score_masala(e['messages'][1]['content']), e) for e in masala_clean]
scored.sort(key=lambda x: -x[0])

top_n = int(MASALA_TARGET * 0.6)
bottom_n = MASALA_TARGET - top_n
masala_selected = [e for _, e in scored[:top_n]]

# Stratified sampling from remainder
remainder = scored[top_n:]
buckets = {}
for s, e in remainder:
    lines = len([l for l in e['messages'][1]['content'].split('\n') if l.strip()])
    bucket = min(lines // 2, 10)
    buckets.setdefault(bucket, []).append(e)

total_rem = sum(len(v) for v in buckets.values())
for bucket, entries in buckets.items():
    n_sample = max(1, int(bottom_n * len(entries) / total_rem))
    masala_selected.extend(random.sample(entries, min(n_sample, len(entries))))

if len(masala_selected) > MASALA_TARGET:
    masala_selected = masala_selected[:MASALA_TARGET]
print(f'  Selected: {len(masala_selected)}')

# ── Merge ──
print(f'\n[4] Merging...')
all_data = train_real + synth + masala_selected
random.shuffle(all_data)
print(f'  KiCad real:    {len(train_real)}')
print(f'  KiCad synth:   {len(synth)}')
print(f'  Masala clean:  {len(masala_selected)}')
print(f'  TOTAL:         {len(all_data)}')

# ── Split ──
print(f'\n[5] Splitting train/val (test uses existing benchmark)...')
n_train = int(len(all_data) * TRAIN_RATIO)
v5_train = all_data[:n_train]
v5_val = all_data[n_train:]

random.shuffle(v5_train)
random.shuffle(v5_val)

print(f'  Train: {len(v5_train)}')
print(f'  Val:   {len(v5_val)}')

# ── Validate ──
print(f'\n[6] Quality validation...')

def validate(entries, name):
    intra_dup = 0
    multi_word_lines = 0
    total_lines = 0
    noisy_samples = 0

    for e in entries:
        label = e['messages'][1]['content']
        lines = label.split('\n')
        has_intra = False
        for line in lines:
            if not line.strip(): continue
            total_lines += 1
            words = line.split()
            if len(words) > 1:
                multi_word_lines += 1
            for w1, w2 in zip(words[:-1], words[1:]):
                if w1 == w2 and len(w1) > 1:
                    intra_dup += 1
                    has_intra = True
        if has_intra:
            noisy_samples += 1

    print(f'\n  {name} ({len(entries)} samples):')
    print(f'    Single-word lines: {total_lines - multi_word_lines}/{total_lines} ({100*(total_lines-multi_word_lines)/total_lines:.1f}%)')
    print(f'    Intra-line dups:   {intra_dup}')
    print(f'    Noisy samples:     {noisy_samples} ({100*noisy_samples/len(entries):.1f}%)')

    # Source breakdown
    sources = Counter()
    for e in entries:
        img = e['images'][0]
        if 'synthetic_v3' in img:
            sources['synthetic'] += 1
        elif 'masala' in img:
            sources['masala'] += 1
        else:
            sources['kicad_real'] += 1
    print(f'    Sources: {dict(sources)}')

validate(v5_train, 'V5-train')
validate(v5_val, 'V5-val')

# Compare with test set
print('\n  --- Test set reference ---')
for test_name in ['ocr_vl_sft-test-easy100.jsonl']:
    with open(DATASET_DIR / test_name, 'r', encoding='utf-8') as f:
        test = [json.loads(line) for line in f if line.strip()]
    validate(test, f'Test ({test_name})')

# ── Save ──
print(f'\n[7] Saving...')
def save_jsonl(entries, path):
    with open(path, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

save_jsonl(v5_train, DATASET_DIR / 'ocr_vl_sft-train-v5-golden.jsonl')
save_jsonl(v5_val, DATASET_DIR / 'ocr_vl_sft-val-v5-golden.jsonl')

print(f'  Train: {DATASET_DIR / "ocr_vl_sft-train-v5-golden.jsonl"}')
print(f'  Val:   {DATASET_DIR / "ocr_vl_sft-val-v5-golden.jsonl"}')

# ── Show samples ──
print(f'\n[8] Sample outputs:')
for name, entries in [('train', v5_train), ('val', v5_val)]:
    for source_key, source_name in [('kicad_real', 'KiCad'), ('synthetic', 'Synth'), ('masala', 'Masala')]:
        for e in entries:
            img = e['images'][0]
            if source_key == 'kicad_real' and 'synthetic' not in img and 'masala' not in img:
                pass
            elif source_key == 'synthetic' and 'synthetic' in img:
                pass
            elif source_key == 'masala' and 'masala' in img:
                pass
            else:
                continue
            label = e['messages'][1]['content']
            print(f'\n  --- {name}/{source_name}: {img} ---')
            for line in label.split('\n')[:5]:
                print(f'    | {line}')
            break

print(f'\n{"="*60}')
print('V5 GOLDEN DATASET DONE!')
print(f'Total: {len(v5_train)} train + {len(v5_val)} val = {len(v5_train) + len(v5_val)}')
print(f'Test: use existing ocr_vl_sft-test-easy*.jsonl / ocr_vl_sft-test.jsonl')
print(f'{"="*60}')

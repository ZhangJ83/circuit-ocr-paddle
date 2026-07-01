"""
Build balanced dataset:
  1. Clean Masala-CHAI: remove SPICE artifacts (comments, formulas, templates)
  2. Score + select ~700 best Masala-CHAI samples
  3. Clean rescraped/openschematics: filter noise
  4. Train/val/test split (80/10/10)
  5. Output analysis
"""
import json, random, os, sys
from pathlib import Path
from collections import Counter

DATASET_DIR = Path(__file__).parent.parent
random.seed(42)

# ── Config ──
MASALA_TARGET = 700        # Target Masala-CHAI count
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# SPICE artifacts to reject
SPICE_BAD_TOKENS = {'*', ';', 'node', '.model', '.subckt', '.ends', '.tran', '.ac', '.dc', '.op'}
SPICE_BAD_PATTERNS = [
    'VALUE=', '{', '}', '<', '>', '[value]', 'W=L', 'W/L',
    'g_m', 'g_mb', 'r_pi', 'r_o', 'v(','vgs', 'vsg',
    'replace [', 'Example ', '; ', ' * ',
]

def load_jsonl(path):
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries

def save_jsonl(entries, path):
    with open(path, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

def clean_masala_label(label):
    """Clean SPICE artifacts from a Masala label line by line.
    Returns (cleaned_label, was_modified)"""
    lines = label.split('\n')
    cleaned_lines = []
    modified = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect SPICE comment/artifact lines
        if any(tok in stripped.split() for tok in [';', '*']):
            modified = True
            # Try to salvage: strip comment part
            for sep in [';', '*']:
                idx = stripped.find(sep)
                if idx >= 0:
                    stripped = stripped[:idx].strip()
            if not stripped:
                continue

        # Remove SPICE formula patterns
        import re
        # Remove {expression} patterns
        if '{' in stripped or '}' in stripped:
            stripped = re.sub(r'\{[^}]*\}', '', stripped).strip()
            modified = True
        # Remove VALUE=... patterns
        stripped = re.sub(r'VALUE\s*=\s*\S+', '', stripped).strip()
        modified = True

        # Clean MOSFET/IC lines: M1 D G S NMOS L=40n W=300u → M1 NMOS
        # Pattern: ref + net_names... + type + SPICE_params
        parts = stripped.split()
        if not parts:
            continue

        ref = parts[0]
        # Check if first token is a component reference (M1, Q2, R3, etc.)
        is_component_line = bool(re.match(r'^[RDLQCUMVXYIJF]\d+$', ref))

        if is_component_line and len(parts) > 1:
            # Find the type token (NMOS, PMOS, NPN, PNP, etc.)
            comp_types = []
            other_parts = []
            for p in parts[1:]:
                # Skip single-letter net names
                if len(p) == 1 and p.isalpha():
                    modified = True
                    continue
                # Skip L=..., W=..., etc.
                if re.match(r'^[A-Za-z_]+=', p):
                    modified = True
                    continue
                # Skip numeric net names
                if p.replace('.','').replace('-','').isdigit() and len(p) <= 4:
                    modified = True
                    continue
                # Skip "DC" SPICE keyword (but keep if it's the only descriptor)
                if p == 'DC':
                    # Check context: if preceded by a source ref (V1, I1, etc.), DC is a keyword
                    if ref[0] in 'VI':
                        modified = True
                        continue
                other_parts.append(p)

            stripped = ' '.join([ref] + other_parts)

        if stripped:
            cleaned_lines.append(stripped)

    return '\n'.join(cleaned_lines), modified

def is_clean_masala(label):
    """Check if label is free of SPICE artifacts (label is already cleaned)"""
    tokens = set(label.split())
    for bt in SPICE_BAD_TOKENS:
        if bt in tokens:
            return False
    for pattern in SPICE_BAD_PATTERNS:
        if pattern in label:
            return False
    # Reject if too short after cleaning
    if len(label) < 10:
        return False
    return True

def score_masala(label):
    """Quality score for a Masala-CHAI sample. Higher = better."""
    lines = [l.strip() for l in label.split('\n') if l.strip()]
    n_lines = len(lines)

    # Line count: prefer 5-15 lines (rich but not overly complex)
    if n_lines < 3:
        line_score = n_lines * 0.3
    elif n_lines <= 15:
        line_score = n_lines
    else:
        line_score = 15 + (n_lines - 15) * 0.2

    # Diversity: count unique character types
    unique_chars = len(set(label))
    char_score = min(unique_chars / 3, 20)

    # Component diversity: count unique component prefixes
    prefixes = set()
    for line in lines:
        parts = line.split()
        for p in parts:
            if len(p) >= 1 and p[0] in 'RDLQCUMVXYIJF' and len(p) >= 2:
                prefixes.add(p[0])
    div_score = len(prefixes) * 2

    # Penalize very short labels
    total_chars = len(label)
    if total_chars < 20:
        length_penalty = -5
    elif total_chars < 40:
        length_penalty = 0
    else:
        length_penalty = min(total_chars / 20, 5)

    return line_score + char_score + div_score + length_penalty

def clean_rescraped_label(label):
    """
    Clean rescraped/openschematics labels.
    Remove lines that are >80% bare numbers (pin number lines).
    """
    lines = label.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if not parts:
            continue
        # Count purely numeric tokens
        num_count = sum(1 for p in parts if p.replace('.','').replace('-','').replace('+','').isdigit())
        if len(parts) > 0 and num_count / len(parts) > 0.8:
            continue  # Skip pin-number-only lines
        cleaned.append(stripped)
    return '\n'.join(cleaned)

def analyze_entries(entries, name):
    """Print statistics for a set of entries"""
    line_counts = []
    char_counts = []
    for e in entries:
        label = e['messages'][1]['content']
        lines = [l for l in label.split('\n') if l.strip()]
        line_counts.append(len(lines))
        char_counts.append(len(label))

    print(f'  {name}: {len(entries)} samples')
    print(f'    Lines: min={min(line_counts)}, max={max(line_counts)}, '
          f'avg={sum(line_counts)/len(line_counts):.1f}, med={sorted(line_counts)[len(line_counts)//2]}')
    print(f'    Chars: min={min(char_counts)}, max={max(char_counts)}, '
          f'avg={sum(char_counts)/len(char_counts):.0f}, med={sorted(char_counts)[len(char_counts)//2]}')

# ── Main ──
print('=' * 60)
print('BUILDING BALANCED DATASET')
print('=' * 60)

# 1. Load all data
print('\n[1] Loading datasets...')
synthetic = load_jsonl(DATASET_DIR / 'ocr_vl_sft-synthetic-v3.jsonl')
rescraped = load_jsonl(DATASET_DIR / 'ocr_vl_sft-rescraped.jsonl')
openschematics = load_jsonl(DATASET_DIR / 'ocr_vl_sft-openschematics.jsonl')
masala_raw = load_jsonl(DATASET_DIR / 'ocr_vl_sft-masala.jsonl')

print(f'  Synthetic:      {len(synthetic)}')
print(f'  Rescraped:      {len(rescraped)}')
print(f'  OpenSchematics: {len(openschematics)}')
print(f'  Masala (raw):   {len(masala_raw)}')

# 2. Filter Masala-CHAI
print(f'\n[2] Filtering Masala-CHAI (target: {MASALA_TARGET})...')

# First pass: clean all labels, remove bad ones
masala_cleaned_labels = []
for e in masala_raw:
    cleaned, modified = clean_masala_label(e['messages'][1]['content'])
    if is_clean_masala(cleaned):
        new_e = dict(e)
        new_e['messages'] = [dict(m) for m in e['messages']]
        new_e['messages'][1]['content'] = cleaned
        masala_cleaned_labels.append(new_e)

print(f'  After cleaning + filtering: {len(masala_cleaned_labels)} (removed {len(masala_raw) - len(masala_cleaned_labels)})')

# Score and sort
scored = [(score_masala(e['messages'][1]['content']), e) for e in masala_cleaned_labels]
scored.sort(key=lambda x: -x[0])

# Stratified selection: ensure diversity by scoring buckets
# Take top 60% by score, then sample remaining 40% from lower scores for diversity
top_n = int(MASALA_TARGET * 0.6)
bottom_n = MASALA_TARGET - top_n

masala_selected = [e for _, e in scored[:top_n]]

# From remainder, do stratified sampling by line count buckets
remainder = scored[top_n:]
# Bucket by line count
buckets = {}
for score, e in remainder:
    lines = len([l for l in e['messages'][1]['content'].split('\n') if l.strip()])
    bucket = min(lines // 2, 10)  # 0-1→0, 2-3→1, 4-5→2, ..., 20+→10
    if bucket not in buckets:
        buckets[bucket] = []
    buckets[bucket].append(e)

# Sample proportionally from each bucket
total_remainder = sum(len(v) for v in buckets.values())
for bucket, entries in buckets.items():
    n_sample = max(1, int(bottom_n * len(entries) / total_remainder))
    sampled = random.sample(entries, min(n_sample, len(entries)))
    masala_selected.extend(sampled)

# Trim to exactly target
if len(masala_selected) > MASALA_TARGET:
    masala_selected = masala_selected[:MASALA_TARGET]

print(f'  Selected: {len(masala_selected)}')

# 3. Clean rescraped & openschematics
print(f'\n[3] Cleaning KiCad data...')
for name, entries in [('rescraped', rescraped), ('openschematics', openschematics)]:
    cleaned_count = 0
    for e in entries:
        old_label = e['messages'][1]['content']
        new_label = clean_rescraped_label(old_label)
        if old_label != new_label:
            cleaned_count += 1
        e['messages'][1]['content'] = new_label
    print(f'  {name}: {cleaned_count}/{len(entries)} samples had pin-number lines removed')

# 4. Combine KiCad (synthetic + rescraped + openschematics)
kicad_all = synthetic + rescraped + openschematics
print(f'\n[4] KiCad combined: {len(kicad_all)} ({len(synthetic)} synth + {len(rescraped)} rescraped + {len(openschematics)} openschematics)')

# 5. Split into train/val/test
print(f'\n[5] Splitting...')

# Shuffle each subset independently
random.shuffle(kicad_all)
random.shuffle(masala_selected)

# Split KiCad
n_kicad = len(kicad_all)
n_k_train = int(n_kicad * TRAIN_RATIO)
n_k_val = int(n_kicad * VAL_RATIO)

kicad_train = kicad_all[:n_k_train]
kicad_val = kicad_all[n_k_train:n_k_train + n_k_val]
kicad_test = kicad_all[n_k_train + n_k_val:]

# Split Masala
n_masala = len(masala_selected)
n_m_train = int(n_masala * TRAIN_RATIO)
n_m_val = int(n_masala * VAL_RATIO)

masala_train = masala_selected[:n_m_train]
masala_val = masala_selected[n_m_train:n_m_train + n_m_val]
masala_test = masala_selected[n_m_train + n_m_val:]

# Merge
train = kicad_train + masala_train
val = kicad_val + masala_val
test = kicad_test + masala_test

random.shuffle(train)
random.shuffle(val)
random.shuffle(test)

print(f'  Train: {len(train)} ({len(kicad_train)} KiCad + {len(masala_train)} Masala)')
print(f'  Val:   {len(val)} ({len(kicad_val)} KiCad + {len(masala_val)} Masala)')
print(f'  Test:  {len(test)} ({len(kicad_test)} KiCad + {len(masala_test)} Masala)')
print(f'  Total: {len(train) + len(val) + len(test)}')

# 6. Detailed analysis
print(f'\n[6] Analysis:')
for name, entries in [('train', train), ('val', val), ('test', test)]:
    analyze_entries(entries, name)

    # Count by source
    sources = Counter()
    for e in entries:
        img = e['images'][0]
        if 'synthetic_v3' in img:
            sources['synthetic'] += 1
        elif 'rescraped' in img:
            sources['rescraped'] += 1
        elif 'openschematics' in img:
            sources['openschematics'] += 1
        elif 'masala' in img:
            sources['masala'] += 1
    print(f'    Sources: {dict(sources)}')

# 7. Save
print(f'\n[7] Saving...')
save_jsonl(train, DATASET_DIR / 'ocr_vl_sft-train-v4.jsonl')
save_jsonl(val, DATASET_DIR / 'ocr_vl_sft-val-v4.jsonl')
save_jsonl(test, DATASET_DIR / 'ocr_vl_sft-test-v4.jsonl')

print(f'\n  Train: {DATASET_DIR / "ocr_vl_sft-train-v4.jsonl"}')
print(f'  Val:   {DATASET_DIR / "ocr_vl_sft-val-v4.jsonl"}')
print(f'  Test:  {DATASET_DIR / "ocr_vl_sft-test-v4.jsonl"}')

# 8. Show samples from each split
print(f'\n[8] Sample outputs:')
for name, entries in [('train', train), ('val', val), ('test', test)]:
    e = entries[0]
    label = e['messages'][1]['content']
    img = e['images'][0]
    print(f'\n  --- {name}: {img} ---')
    print(f'  Label ({len(label)} chars):')
    for line in label.split('\n')[:6]:
        print(f'    | {line}')
    if len(label.split('\n')) > 6:
        print(f'    ... ({len(label.split(chr(10)))} total lines)')

print(f'\n{"="*60}')
print('DONE!')

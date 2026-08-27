"""Pick a stratified sample of 50 schematics for Gemini cross-validation."""
import json, random
from pathlib import Path
import shutil

PROJECT_DIR = Path(__file__).parent.parent
ANNO_DIR = PROJECT_DIR / 'output/review_1000/annotations'
IMG_DIR = PROJECT_DIR / 'output/review_1000/images'
OUT_DIR = PROJECT_DIR / 'output/gemini_check'
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / 'images').mkdir(exist_ok=True)

random.seed(123)

# Collect all samples with metadata
all_samples = []
for f in ANNO_DIR.glob('*.txt'):
    content = f.read_text(encoding='utf-8').strip()
    lines = [l for l in content.split('\n') if l.strip()]
    n = len(lines)
    tier = 'simple' if n < 50 else ('medium' if n < 200 else 'complex')
    paper = 'A4'
    for l in lines:
        if l.startswith('Size: '): paper = l.replace('Size: ', ''); break
    has_overline = any('̅' in l for l in lines)
    all_samples.append({'id': f.stem, 'lines': n, 'tier': tier, 'paper': paper,
                        'overline': has_overline, 'gt': content})

# Stratified sample
chosen = []

# 5 simple
simple = [s for s in all_samples if s['tier'] == 'simple']
chosen.extend(random.sample(simple, min(5, len(simple))))

# 15 medium
medium = [s for s in all_samples if s['tier'] == 'medium']
chosen.extend(random.sample(medium, min(15, len(medium))))

# 20 complex
complex_s = [s for s in all_samples if s['tier'] == 'complex']
chosen.extend(random.sample(complex_s, min(20, len(complex_s))))

# Ensure non-A4 papers are represented
non_a4 = [s for s in all_samples if s['paper'] not in ('A4',) and s not in chosen]
extra_paper = random.sample(non_a4, min(8, len(non_a4)))
chosen.extend(extra_paper)

# Ensure overline samples
ov = [s for s in all_samples if s['overline'] and s not in chosen]
extra_ov = random.sample(ov, min(4, len(ov)))
chosen.extend(extra_ov)

# Fill to 50
remaining = [s for s in all_samples if s not in chosen]
needed = 50 - len(chosen)
if needed > 0:
    chosen.extend(random.sample(remaining, min(needed, len(remaining))))

chosen = chosen[:50]
print(f'Selected {len(chosen)} samples')

# Copy images and write GT files
manifest = []
for i, s in enumerate(chosen):
    uid = s['id']
    shutil.copy2(IMG_DIR / f'{uid}.png', OUT_DIR / 'images' / f'{i+1:02d}.png')
    with open(OUT_DIR / f'{i+1:02d}.txt', 'w', encoding='utf-8') as f:
        f.write(f'# Image: {i+1:02d}.png\n')
        f.write(f'# ID: {uid}\n')
        f.write(f'# Lines: {s["lines"]}\n')
        f.write(f'# Paper: {s["paper"]}\n')
        f.write(f'# Tier: {s["tier"]}\n\n')
        f.write(s['gt'])
    manifest.append({'num': i+1, 'id': uid, 'lines': s['lines'],
                     'paper': s['paper'], 'tier': s['tier']})

with open(OUT_DIR / 'manifest.json', 'w') as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print(f'\nDistribution:')
from collections import Counter
for tier, cnt in Counter(s['tier'] for s in manifest).most_common():
    print(f'  {tier}: {cnt}')
for paper, cnt in Counter(s['paper'] for s in manifest).most_common():
    print(f'  {paper}: {cnt}')
print(f'  overline: {sum(1 for s in chosen if s["overline"])}')
print(f'\nFiles ready: {OUT_DIR}')

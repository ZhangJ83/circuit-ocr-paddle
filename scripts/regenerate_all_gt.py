"""Regenerate all 1500 GTs with the fixed parser."""
import json, sys, os, time
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR / 'scripts'))
from generate_gt_from_kicad import parse

ANNO_DIR = PROJECT_DIR / 'output/review_1000/annotations'
SRC_DIR = PROJECT_DIR / 'circuit-ocr-dataset/data/open_schematics_v2/kicad_sch'

with open(PROJECT_DIR / 'output/review_1000/mapping.json', 'r') as f:
    mapping = json.load(f)

print(f'Regenerating {len(mapping)} GTs...')
ok, fail = 0, 0
t0 = time.time()

for i, m in enumerate(mapping):
    uid = m['id']
    orig_name = m['original_name']
    sch_path = SRC_DIR / f'{orig_name}.kicad_sch'

    if not sch_path.exists():
        fail += 1
        continue

    try:
        gt, blocks = parse(str(sch_path))
        with open(ANNO_DIR / f'{uid}.txt', 'w', encoding='utf-8') as f:
            f.write(gt)
        ok += 1
    except Exception as e:
        fail += 1
        if fail <= 5:
            print(f'  FAIL {uid} {orig_name}: {e}')

    if (i + 1) % 200 == 0:
        elapsed = time.time() - t0
        print(f'  {i+1}/{len(mapping)} ({elapsed:.1f}s)', flush=True)

elapsed = time.time() - t0
print(f'Done: {ok} OK, {fail} FAIL in {elapsed:.1f}s')

# Quick stats
import re
slash_count = 0
empty_title = 0
for f in ANNO_DIR.glob('*.txt'):
    content = f.read_text(encoding='utf-8')
    if '{slash}' in content:
        slash_count += 1
    if '\nTitle:\n' in content:
        empty_title += 1
print(f'Remaining {{slash}}: {slash_count}')
print(f'Empty Title:: {empty_title}')

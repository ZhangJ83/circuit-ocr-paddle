#!/usr/bin/env python3
"""Sync english.tex with V5 Golden dataset updates."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open(r'G:\mimo_project\circuit_ocr\arxiv_template\english.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# ── 1. Update Abstract ──
old_abs = r"""We construct a multi-source hybrid dataset containing 1,357 real-world open-source project schematics and 500 high-quality synthetic renderings, totaling 1,857 training samples after rigorous multi-round deduplication and quality screening."""
new_abs = r"""We construct a multi-source balanced dataset containing 1,857 KiCad schematics (500 programmatic synthetic renderings + 1,357 real-world open-source project schematics) and 698 Masala-CHAI textbook circuit diagrams, totaling 2,555 samples after SPICE artifact cleaning and quality-scored stratified sampling, split 90/10 into training (2,299) and validation (256) sets, with the test set reusing the existing 523-sample benchmark."""

if old_abs in content:
    content = content.replace(old_abs, new_abs)
    print('OK: Abstract updated')
else:
    print('MISS: Abstract old text')

# ── 2. Update Introduction ──
old_intro = "comprising 24,717 training samples and a 4-tier difficulty evaluation system."
new_intro = "comprising 2,555 high-quality samples (KiCad 1,857 + Masala-CHAI 698), split 90/10 for training/validation with test reuse of the existing 523-sample benchmark, and a 4-tier difficulty evaluation system."
if old_intro in content:
    content = content.replace(old_intro, new_intro)
    print('OK: Introduction updated')
else:
    print('MISS: Introduction old text')

# ── 3. Update Section 3 (Dataset Construction) ──
sec3_start = content.find(r'\section{Dataset Construction}')
sec4_start = content.find(r'\section{Model Fine-tuning and Performance Improvement Roadmap}')

# Read the new English Section 3 from a separate file
with open(r'G:\mimo_project\circuit_ocr\arxiv_template\english_sec3_v5.tex', 'r', encoding='utf-8') as f:
    new_sec3 = f.read()

if sec3_start >= 0 and sec4_start > sec3_start:
    content = content[:sec3_start] + new_sec3 + content[sec4_start:]
    print(f'OK: Section 3 replaced ({len(new_sec3)} chars)')
else:
    print(f'ERROR: Section boundaries not found (sec3={sec3_start}, sec4={sec4_start})')

# ── 4. Update summary/conclusion references ──
reps = [
    ('approximately 24,717 independent samples after three rounds of deduplication (text hash, topology hash, visual perceptual hash) and large-model quality scoring',
     '2,555 independent samples (KiCad 1,857 + Masala-CHAI 698) after SPICE artifact cleaning and stratified sampling, split 90/10 for training/validation'),
    ('24,717 training samples + 4-tier evaluation system + 250-sample degraded evaluation set, providing a standardized evaluation foundation for the field',
     '2,555 high-quality samples (90/10 split) + 4-tier difficulty evaluation system + 250-sample degraded evaluation set, providing a standardized evaluation foundation'),
]

for old, new in reps:
    if old in content:
        content = content.replace(old, new)
        print(f'OK: {old[:50]}...')
    else:
        print(f'MISS: {old[:50]}...')

with open(r'G:\mimo_project\circuit_ocr\arxiv_template\english.tex', 'w', encoding='utf-8') as f:
    f.write(content)
print('\nDone!')

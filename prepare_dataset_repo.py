"""Prepare circuit_ocr_dataset_final with two datasets."""
import json, os, shutil, re

SRC = r'g:\mimo_project\circuit_ocr'
DST = r'g:\mimo_project\circuit_ocr_dataset_final'

# ═══════════════════════════════════════════════════════
# DATASET A: Strict Annotations + Mixed (FRONT)
# ═══════════════════════════════════════════════════════
print("=" * 60)
print("DATASET A: Strict Annotation + Mixed Training Data")
print("=" * 60)

os.makedirs(os.path.join(DST, 'dataset_a', 'images'), exist_ok=True)

# Copy images and fix paths
def fix_paths_and_copy(entries, src_base, dst_img_dir, prefix=''):
    """Fix image paths to relative and copy images."""
    new_entries = []
    copied, skipped = 0, 0
    for entry in entries:
        img_path = entry['images'][0]
        # Fix Linux path
        img_path = img_path.replace('/root/circuit_ocr/', SRC + '/')
        img_path = img_path.replace('\\', '/')

        if not os.path.exists(img_path):
            # Try alternative path
            basename = os.path.basename(img_path)
            alt = os.path.join(SRC, 'output', 'review_1000', 'images', basename)
            if os.path.exists(alt):
                img_path = alt
            else:
                skipped += 1
                continue

        # Copy with unique name
        basename = os.path.basename(img_path)
        dst_name = f'{prefix}{basename}' if prefix else basename
        dst_path = os.path.join(dst_img_dir, dst_name)
        if not os.path.exists(dst_path):
            shutil.copy2(img_path, dst_path)

        # Update entry with relative path
        entry['images'] = [f'images/{dst_name}']
        new_entries.append(entry)
        copied += 1

    return new_entries, copied, skipped

# --- Training data ---
train_files = {
    'train.jsonl': os.path.join(SRC, 'output', 'train_v10fmt_synth_real.jsonl'),
}

for out_name, src_path in train_files.items():
    with open(src_path, encoding='utf-8') as f:
        entries = [json.loads(l) for l in f if l.strip()]
    fixed, copied, skipped = fix_paths_and_copy(entries, SRC,
        os.path.join(DST, 'dataset_a', 'images'), 'train_')
    out_path = os.path.join(DST, 'dataset_a', out_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        for e in fixed:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    print(f'  {out_name}: {len(fixed)} entries ({copied} imgs copied, {skipped} skipped)')

# --- Test data ---
with open(os.path.join(SRC, 'output', 'test_clean.jsonl'), encoding='utf-8') as f:
    test_entries = [json.loads(l) for l in f if l.strip()]
fixed_test, tc, ts = fix_paths_and_copy(test_entries, SRC,
    os.path.join(DST, 'dataset_a', 'images'), 'test_')
with open(os.path.join(DST, 'dataset_a', 'test.jsonl'), 'w', encoding='utf-8') as f:
    for e in fixed_test:
        f.write(json.dumps(e, ensure_ascii=False) + '\n')
print(f'  test.jsonl: {len(fixed_test)} entries ({tc} imgs copied)')

# --- Validation data ---
with open(os.path.join(SRC, 'output', 'val_clean.jsonl'), encoding='utf-8') as f:
    val_entries = [json.loads(l) for l in f if l.strip()]
fixed_val, vc, vs = fix_paths_and_copy(val_entries, SRC,
    os.path.join(DST, 'dataset_a', 'images'), 'val_')
with open(os.path.join(DST, 'dataset_a', 'val.jsonl'), 'w', encoding='utf-8') as f:
    for e in fixed_val:
        f.write(json.dumps(e, ensure_ascii=False) + '\n')
print(f'  val.jsonl: {len(fixed_val)} entries ({vc} imgs copied)')

# Stats for Dataset A
total_train = len(fixed)
train_comps = sum(len(re.findall(r'\b((?:LED|[RCDLQUJYF])\d+)\b', e['messages'][1]['content'])) for e in fixed)
train_with_photo = sum(1 for e in fixed if e.get('meta', {}).get('source') == 'real_camera_photo')
train_with_synth = sum(1 for e in fixed if 'synth_text_images' in e['images'][0])

print(f'\n  Dataset A Summary:')
print(f'    Train: {total_train} ({train_with_synth} synth text + {train_with_photo} real photo)')
print(f'    Test:  {len(fixed_test)}')
print(f'    Val:   {len(fixed_val)}')
print(f'    Total OCR instances (train): {train_comps}')

# ═══════════════════════════════════════════════════════
# DATASET B: Original Heavy Annotation (BACK)
# ═══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DATASET B: Original Heavy Annotation (7-round GT fix)")
print("=" * 60)

# The V9 data is already in the repo root. Move to dataset_b/
os.makedirs(os.path.join(DST, 'dataset_b', 'images'), exist_ok=True)

for old_file in ['ocr_vl_sft-train-v9-pure.jsonl', 'ocr_vl_sft-val-v9-pure.jsonl',
                 'ocr_vl_sft-synthetic-v3.jsonl']:
    src = os.path.join(DST, old_file)
    if os.path.exists(src):
        dst = os.path.join(DST, 'dataset_b', old_file)
        shutil.move(src, dst)
        print(f'  Moved: {old_file} -> dataset_b/')

# Stats for Dataset B
for fname in ['ocr_vl_sft-train-v9-pure.jsonl', 'ocr_vl_sft-val-v9-pure.jsonl',
              'ocr_vl_sft-synthetic-v3.jsonl']:
    fpath = os.path.join(DST, 'dataset_b', fname)
    if os.path.exists(fpath):
        with open(fpath, encoding='utf-8') as f:
            entries = [json.loads(l) for l in f if l.strip()]
        comps = sum(len(re.findall(r'\b((?:LED|[RCDLQUJYF])\d+)\b', e['messages'][1]['content'])) for e in entries)
        print(f'  {fname}: {len(entries)} entries, ~{comps} OCR instances')

# Image count
img_dir = os.path.join(DST, 'dataset_a', 'images')
imgs = [f for f in os.listdir(img_dir) if f.endswith(('.png','.jpg','.jpeg'))]
total_mb = sum(os.path.getsize(os.path.join(img_dir, f)) for f in imgs) / 1024 / 1024
print(f'\n  Images: {len(imgs)} files, {total_mb:.1f} MB')

print("\nDONE - Ready to commit!")

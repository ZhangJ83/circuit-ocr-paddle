"""Match real-camera photos to GT labels and integrate into training data."""
import json, os, re, shutil
from PIL import Image

base_dir = r'g:\mimo_project\circuit_ocr'

# Load original dataset to find GT labels
orig_map = {}
for f in ['output/test_clean.jsonl', 'output/train_clean.jsonl']:
    with open(os.path.join(base_dir, f), encoding='utf-8') as fh:
        for line in fh:
            if not line.strip(): continue
            s = json.loads(line)
            fname = os.path.basename(s['images'][0])
            orig_map[fname] = s['messages'][1]['content']

# PNG -> source schematic mapping (from selection script)
png_to_source = {
    '01.png': '0686.png', '02.png': '0719.png', '03.png': '1172.png',
    '04.png': '0325.png', '05.png': '1387.png', '06.png': '1316.png',
    '07.png': '0898.png', '08.png': '1074.png', '09.png': '0017.png',
    '10.png': '1395.png', '11.png': '1133.png', '12.png': '0099.png',
    '13.png': '0567.png', '14.png': '1019.png', '15.png': '1065.png',
    '16.png': '1349.png', '17.png': '0160.png', '18.png': '0418.png',
    '19.png': '1312.png', '20.png': '1371.png',
}

photo_dir = os.path.join(base_dir, 'output', 'real_photos')
os.makedirs(photo_dir, exist_ok=True)
src_dir = os.path.join(base_dir, 'real_photo_templates', 'selected_circuits')

new_entries = []
for i in range(1, 21):
    png_name = f'{i:02d}.png'
    jpg_name = f'-{i}.jpg'
    jpg_path = os.path.join(src_dir, jpg_name)

    if not os.path.exists(jpg_path):
        print(f'  MISSING: {jpg_name}')
        continue

    source_schem = png_to_source.get(png_name)
    label = orig_map.get(source_schem)
    if label is None:
        print(f'  NO GT: {png_name} -> {source_schem}')
        continue

    dst_name = f'real_{i:02d}.jpg'
    dst_path = os.path.join(photo_dir, dst_name)
    shutil.copy2(jpg_path, dst_path)

    img = Image.open(dst_path)
    w, h = img.size
    img.close()

    entry = {
        'messages': [
            {'role': 'user', 'content': '<image>OCR:'},
            {'role': 'assistant', 'content': label}
        ],
        'images': [dst_path.replace('\\', '/')],
        'meta': {'source': 'real_camera_photo', 'original_schematic': source_schem}
    }
    new_entries.append(entry)

    comps = len(re.findall(r'\b((?:LED|[RCDLQUJYF])\d+)\b', label))
    print(f'  {dst_name}: {w}x{h}, {comps} components <- {source_schem}')

print(f'\nSuccess: {len(new_entries)} real-camera photos')

# Save standalone
real_jsonl = os.path.join(base_dir, 'output', 'real_photos.jsonl')
with open(real_jsonl, 'w', encoding='utf-8') as f:
    for e in new_entries:
        f.write(json.dumps(e, ensure_ascii=False) + '\n')
print(f'Saved: {real_jsonl}')

# Merge into training data
train_path = os.path.join(base_dir, 'output', 'train_v10fmt_synth.jsonl')
with open(train_path, encoding='utf-8') as f:
    train_entries = [json.loads(l) for l in f if l.strip()]

merged = train_entries + new_entries
merged_path = os.path.join(base_dir, 'output', 'train_v10fmt_synth_real.jsonl')
with open(merged_path, 'w', encoding='utf-8') as f:
    for e in merged:
        f.write(json.dumps(e, ensure_ascii=False) + '\n')

print(f'Merged: {merged_path}')
print(f'  {len(train_entries)} existing + {len(new_entries)} real-camera = {len(merged)} total')
print('\nDONE')

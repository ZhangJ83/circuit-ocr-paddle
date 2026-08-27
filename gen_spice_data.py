"""Generate SPICE-format training data from existing labels."""
import json, os, re, random

D = r'g:/mimo_project/circuit_ocr'
SRC = os.path.join(D, 'output', 'train_v10fmt_synth.jsonl')
OUT = os.path.join(D, 'output', 'train_spice.jsonl')

with open(SRC, encoding='utf-8') as f:
    data = [json.loads(l) for l in f if l.strip()]

# Parse existing labels and convert to SPICE format
def to_spice(label):
    """Convert circuit label to SPICE-like format."""
    lines = label.split('\n')
    spice_lines = []
    node_counter = [1]  # mutable counter

    re_comp = re.compile(r'^((?:LED|[RCDLQUJYF])\d+)\s+(.+)$')

    for line in lines:
        line = line.strip()
        if not line: continue

        m = re_comp.match(line)
        if m:
            refdes = m.group(1)
            value = m.group(2).strip()
            # Clean up value
            value = re.sub(r'\s*±\d+%\s*', '', value)  # remove tolerance
            value = re.sub(r'\s+\d+V\s*', '', value)    # remove voltage rating
            value = re.sub(r'\s+(Ceramic|Electrolytic|MLCC|Tantalum|SMD|TH)\s*', '', value, flags=re.I)
            value = value.strip().rstrip(',').rstrip(';')
            if not value: continue

            comp_type = re.match(r'[A-Z]+', refdes).group()

            if comp_type in ('R', 'C', 'L', 'D', 'Q', 'LED', 'F'):
                # Two-terminal: <refdes> N<A> N<B> <value>
                n1 = node_counter[0]; n2 = node_counter[0] + 1
                node_counter[0] += 2
                spice_lines.append(f'{refdes} N{n1} N{n2} {value}')
            elif comp_type in ('U', 'J', 'Y'):
                # Multi-terminal: <refdes> N<A> ... N<Z> <name>
                n_terms = random.randint(3, 8)
                nodes = ' '.join(f'N{node_counter[0] + i}' for i in range(n_terms))
                node_counter[0] += n_terms
                spice_lines.append(f'{refdes} {nodes} {value}')
        else:
            # Non-component line (net label, comment) — keep as comment
            if not any(c in line for c in ('+', '-', '=', ':', '(', ')')):
                spice_lines.append(f'* {line}')  # SPICE comment

    return '\n'.join(spice_lines) if spice_lines else label

# Convert all circuit data (skip synthetic text images)
spice_entries = []
converted = 0; skipped = 0
for s in data:
    if 'synth_text_images' in s['images'][0]:
        skipped += 1
        continue  # skip pure text images

    label = s['messages'][1]['content']
    spice_label = to_spice(label)

    if spice_label != label:
        converted += 1
        entry = {
            'messages': [
                {'role': 'user', 'content': '<image>OCR:'},
                {'role': 'assistant', 'content': spice_label}
            ],
            'images': s['images']
        }
        spice_entries.append(entry)

random.shuffle(spice_entries)

# Save — mix with some original data for stability
with open(OUT, 'w', encoding='utf-8') as f:
    for e in spice_entries:
        f.write(json.dumps(e, ensure_ascii=False) + '\n')

print(f'SPICE entries: {len(spice_entries)} (converted {converted}, skipped {skipped} synth)')
print(f'Example:')
print(spice_entries[0]['messages'][1]['content'][:300])
print(f'\nSaved: {OUT}')

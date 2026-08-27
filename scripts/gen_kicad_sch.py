"""Generate KiCad .kicad_sch schematic files programmatically.
These are S-expression text files that kicad-cli can render to PNG with true KiCad quality.

Format reference (KiCad 8.0):
  (kicad_sch (version 20230121) (generator eeschema)
    (paper "A4")
    (lib_symbols ...)
    (symbol (lib_id "Device:R") (at x y rot) (property "Reference" "R1") ...)
    (wire (pts (xy x1 y1) (xy x2 y2)))
    (text "VCC" (at x y rot))
    (junction (at x y))
  )
"""
import os, sys, json, time, random, math, argparse
from pathlib import Path
from collections import defaultdict

# ============================================================
# KiCad symbol library references (standard Device library)
# ============================================================
SYMBOL_LIB = {
    # (lib_id, pin_count, width_mm, height_mm)
    'R':   ('Device:R', 2, 7.62, 2.54),
    'C':   ('Device:C', 2, 5.08, 2.54),
    'CP':  ('Device:CP', 2, 5.08, 2.54),  # Polarized capacitor
    'L':   ('Device:L', 2, 7.62, 2.54),
    'D':   ('Device:D', 2, 5.08, 2.54),
    'LED': ('Device:LED', 2, 5.08, 2.54),
    'ZD':  ('Device:D_Zener', 2, 5.08, 2.54),
    'Q_NPN': ('Device:Q_NPN_BCE', 3, 7.62, 7.62),
    'Q_PNP': ('Device:Q_PNP_BCE', 3, 7.62, 7.62),
    'Q_NMOS': ('Device:Q_NMOS_GDS', 3, 7.62, 7.62),
    'F':   ('Device:Fuse', 2, 7.62, 2.54),
    'RV':  ('Device:R_Potentiometer', 3, 7.62, 5.08),
    'J':   ('Connector:Conn_01x04', 4, 5.08, 10.16),
    'J_SMALL': ('Connector:Conn_01x02', 2, 5.08, 5.08),
    'SW':  ('Switch:SW_SPST', 2, 5.08, 5.08),
    'Y':   ('Device:Crystal', 2, 7.62, 5.08),
    'U':   ('Amplifier_Operational:LM358', 8, 10.16, 15.24),
    'IC':  ('4xxx:4049', 16, 15.24, 25.4),  # generic 16-pin IC
    'BAT': ('Device:Battery_Cell', 2, 5.08, 5.08),
    'SPK': ('Device:Speaker', 2, 7.62, 7.62),
    'MIC': ('Device:Microphone', 2, 5.08, 5.08),
    'ANT': ('Device:Antenna', 1, 5.08, 5.08),
}

# Simplified: we'll use KiCad's standard "Device" library for most components
# The symbols below use ONLY guaranteed-available KiCad library parts
# All from the "Device" library which ships with every KiCad installation

DEVICE_SYMBOLS = {
    'R':   'Device:R',
    'C':   'Device:C',
    'CP':  'Device:CP',
    'L':   'Device:L',
    'D':   'Device:D',
    'LED': 'Device:LED',
    'ZD':  'Device:D_Zener',
    'Q_NPN': 'Device:Q_NPN_BCE',
    'Q_PNP': 'Device:Q_PNP_BCE',
    'Q_NMOS': 'Transistor_FET:2N7002',
    'F':   'Device:Fuse',
    'RV':  'Device:R_Potentiometer',
    'J':   'Connector:Conn_01x04',
    'J2':  'Connector:Conn_01x02',
    'SW':  'Switch:SW_SPST',
    'Y':   'Device:Crystal',
    'BAT': 'Device:Battery_Cell',
    'SPK': 'Device:Speaker',
    'MIC': 'Device:Microphone',
    'ANT': 'Device:Antenna',
}


VALUE_POOLS = {
    'R':   ['10', '22', '47', '100', '220', '470', '1k', '2.2k', '4.7k', '10k', '22k', '47k', '100k', '220k', '470k', '1M'],
    'C':   ['10pF', '22pF', '47pF', '100pF', '220pF', '1nF', '10nF', '22nF', '100nF', '220nF', '470nF'],
    'CP':  ['1uF', '2.2uF', '4.7uF', '10uF', '22uF', '47uF', '100uF', '220uF', '470uF'],
    'L':   ['1uH', '10uH', '22uH', '47uH', '100uH', '220uH', '470uH', '1mH'],
    'D':   ['1N4148', '1N4007', '1N5819', 'BAT54S', 'SS14'],
    'LED': ['Red', 'Green', 'Blue', 'Yellow', 'White'],
    'ZD':  ['3.3V', '3.6V', '5.1V', '6.2V', '12V', 'BZX84C3V3'],
    'Q_NPN': ['2N3904', 'BC547', 'S8050', '2N2222'],
    'Q_PNP': ['2N3906', 'BC557', 'S8550'],
    'Q_NMOS': ['2N7002', 'BSS138'],
    'F':   ['500mA', '1A', '2A', '3A', '5A'],
    'RV':  ['10k', '50k', '100k', '500k'],
    'J':   ['CONN_4P', 'HEADER_4P'],
    'J2':  ['CONN_2P', 'HEADER_2P'],
    'SW':  ['SW'],
    'Y':   ['8MHz', '12MHz', '16MHz', '25MHz', '32.768kHz'],
    'BAT': ['3.3V', '5V', '3.7V'],
    'SPK': ['8Ω', '32Ω'],
    'MIC': ['MIC'],
    'ANT': ['2.4GHz'],
}

REFDES_PREFIX = {
    'R': 'R', 'C': 'C', 'CP': 'C', 'L': 'L',
    'D': 'D', 'LED': 'LED', 'ZD': 'D',
    'Q_NPN': 'Q', 'Q_PNP': 'Q', 'Q_NMOS': 'Q',
    'F': 'F', 'RV': 'RV', 'J': 'J', 'J2': 'J',
    'SW': 'SW', 'Y': 'Y', 'BAT': 'BAT',
    'SPK': 'SPK', 'MIC': 'MIC', 'ANT': 'ANT',
}


def s_expr(*parts):
    """Build an S-expression string."""
    def _fmt(v):
        if isinstance(v, bool):
            return 'yes' if v else 'no'
        if isinstance(v, (int, float)):
            return str(v)
        return str(v)

    result = '('
    for p in parts:
        if isinstance(p, (list, tuple)):
            result += ' ' + ' '.join(_fmt(x) for x in p)
        else:
            result += ' ' + _fmt(p)
    result += ')'
    return result


def s_symbol(lib_id, at_xy, ref, value, unit=1, in_bom=True, on_board=True):
    """Generate a KiCad symbol S-expression."""
    x, y, rot = at_xy[0], at_xy[1], at_xy[2] if len(at_xy) > 2 else 0
    props = [
        s_expr('property', 'Reference', ref, s_expr('at', x, y, 0), s_expr('effects', s_expr('font', s_expr('size', 1.27, 1.27)), 'hide')),
        s_expr('property', 'Value', value, s_expr('at', x, y - 2.54, 0), s_expr('effects', s_expr('font', s_expr('size', 1.27, 1.27)))),
        s_expr('property', 'Footprint', '', s_expr('at', x, y + 2.54, 0), s_expr('effects', s_expr('font', s_expr('size', 1.27, 1.27)), 'hide')),
        s_expr('property', 'Datasheet', '', s_expr('at', x, y + 5.08, 0), s_expr('effects', s_expr('font', s_expr('size', 1.27, 1.27)), 'hide')),
    ]
    pin_names = ''
    return s_expr('symbol', s_expr('lib_id', lib_id),
                   s_expr('at', x, y, rot),
                   s_expr('unit', unit),
                   s_expr('in_bom', 'yes' if in_bom else 'no'),
                   s_expr('on_board', 'yes' if on_board else 'no'),
                   s_expr('uuid', f'{random.randint(0, 0xFFFFFFFF):08x}-{random.randint(0, 0xFFFF):04x}-{random.randint(0, 0xFFFF):04x}-{random.randint(0, 0xFFFF):04x}-{random.randint(0, 0xFFFF):012x}'),
                   *props,
                   s_expr('pin_numbers', pin_names))


def s_wire(x1, y1, x2, y2):
    """Generate a wire S-expression."""
    return s_expr('wire', s_expr('pts', s_expr('xy', x1, y1), s_expr('xy', x2, y2)),
                   s_expr('stroke', s_expr('width', 0), s_expr('type', 'default'), s_expr('color', 0, 0, 0, 0)),
                   s_expr('uuid', f'{random.randint(0, 0xFFFFFFFF):08x}-{random.randint(0, 0xFFFF):04x}-{random.randint(0, 0xFFFF):04x}-{random.randint(0, 0xFFFF):04x}-{random.randint(0, 0xFFFFFFFFFFFF):012x}'))


def s_junction(x, y):
    """Generate a junction dot."""
    return s_expr('junction', s_expr('at', x, y),
                   s_expr('diameter', 0.91),
                   s_expr('color', 0, 0, 0, 0),
                   s_expr('uuid', f'{random.randint(0, 0xFFFFFFFF):08x}-{random.randint(0, 0xFFFF):04x}-{random.randint(0, 0xFFFF):04x}-{random.randint(0, 0xFFFF):04x}-{random.randint(0, 0xFFFFFFFFFFFF):012x}'))


def s_text(text, x, y, size=1.27):
    """Generate a text label."""
    return s_expr('text', f'"{text}"',
                   s_expr('at', x, y, 0),
                   s_expr('effects', s_expr('font', s_expr('size', size, size)),
                          s_expr('justify', 'left', 'bottom')),
                   s_expr('uuid', f'{random.randint(0, 0xFFFFFFFF):08x}-{random.randint(0, 0xFFFF):04x}-{random.randint(0, 0xFFFF):04x}-{random.randint(0, 0xFFFF):04x}-{random.randint(0, 0xFFFFFFFFFFFF):012x}'))


def s_label(text, x, y, size=1.27):
    """Generate a net label."""
    return s_expr('label', f'"{text}"',
                   s_expr('at', x, y, 0),
                   s_expr('effects', s_expr('font', s_expr('size', size, size))),
                   s_expr('uuid', f'{random.randint(0, 0xFFFFFFFF):08x}-{random.randint(0, 0xFFFF):04x}-{random.randint(0, 0xFFFF):04x}-{random.randint(0, 0xFFFF):04x}-{random.randint(0, 0xFFFFFFFFFFFF):012x}'))


def generate_schematic(rng, n_components=None, sheet_w=190, sheet_h=140):
    """Generate a random KiCad schematic.

    Returns: (schematic_text, gt_label_dict)
      gt_label_dict: {refdes: value} for GT extraction
    """
    if n_components is None:
        n_components = rng.randint(8, 35)

    # Component type distribution (weighted toward passives)
    type_choices = ['R'] * 5 + ['C'] * 4 + ['CP'] * 2 + ['L'] * 2 + \
                   ['D'] * 2 + ['LED'] * 1 + ['ZD'] * 1 + \
                   ['Q_NPN'] * 1 + ['F'] * 1 + ['J'] * 2 + ['Y'] * 1 + ['RV'] * 1
    chosen = [rng.choice(type_choices) for _ in range(n_components)]

    # Build components with refdes
    counters = {}
    comps = []
    for ctype in chosen:
        counters[ctype] = counters.get(ctype, 0) + 1
        prefix = REFDES_PREFIX.get(ctype, 'U')
        ref = f'{prefix}{counters[ctype]}'
        value_pool = VALUE_POOLS.get(ctype, ['?'])
        value = rng.choice(value_pool)
        comps.append({'type': ctype, 'ref': ref, 'value': value})

    # Place on grid
    cols = max(2, int(math.sqrt(n_components * sheet_w / sheet_h)))
    rows = (n_components + cols - 1) // cols
    cell_w = (sheet_w - 20) / cols
    cell_h = (sheet_h - 30) / rows
    margin = 15

    positions = {}
    for i, comp in enumerate(comps):
        col = i % cols
        row = i // cols
        x = margin + col * cell_w + cell_w / 2 + rng.uniform(-cell_w * 0.15, cell_w * 0.15)
        y = margin + 10 + row * cell_h + cell_h / 2 + rng.uniform(-cell_h * 0.15, cell_h * 0.15)
        # Snap to 2.54mm grid
        x = round(x / 2.54) * 2.54
        y = round(y / 2.54) * 2.54
        positions[comp['ref']] = (x, y)
        comp['pos'] = (x, y)

    # Generate symbol S-expressions
    symbol_sexprs = []
    for comp in comps:
        lib_id = DEVICE_SYMBOLS.get(comp['type'], 'Device:R')
        x, y = comp['pos']
        sexpr = s_symbol(lib_id, (x, y, 0), comp['ref'], comp['value'])
        symbol_sexprs.append(sexpr)

    # Wire generation: connect random nearby components
    wire_sexprs = []
    junction_pts = set()

    # Sort components top-to-bottom, group by column-ish
    comp_refs = list(positions.keys())
    rng.shuffle(comp_refs)

    # Connect consecutive pairs to form chains
    for i in range(len(comp_refs) - 1):
        if rng.random() < 0.6:  # 60% chance of connection
            ref1, ref2 = comp_refs[i], comp_refs[i + 1]
            x1, y1 = positions[ref1]
            x2, y2 = positions[ref2]

            # Use Manhattan routing
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            # Choose horizontal-first or vertical-first
            if abs(x1 - x2) > abs(y1 - y2):
                # Horizontal first
                bx = x2
                wire_sexprs.append(s_wire(x1, y1, bx, y1))
                wire_sexprs.append(s_wire(bx, y1, bx, y2))
                wire_sexprs.append(s_wire(bx, y2, x2, y2))
                junction_pts.add((bx, y1))
                junction_pts.add((bx, y2))
            else:
                # Vertical first
                by = y2
                wire_sexprs.append(s_wire(x1, y1, x1, by))
                wire_sexprs.append(s_wire(x1, by, x2, by))
                wire_sexprs.append(s_wire(x2, by, x2, y2))
                junction_pts.add((x1, by))
                junction_pts.add((x2, by))

    # VCC rail at top
    vcc_y = 5
    vcc_comps = [c for c in comps if c['type'] in ('R', 'C', 'U', 'J', 'D', 'LED', 'Y', 'F', 'L')]
    if len(vcc_comps) >= 2:
        vcc_sample = rng.sample(vcc_comps, min(3, len(vcc_comps)))
        vcc_xs = [positions[c['ref']][0] for c in vcc_sample]
        min_x, max_x = min(vcc_xs) - 5, max(vcc_xs) + 5
        wire_sexprs.append(s_wire(min_x, vcc_y, max_x, vcc_y))
        for c in vcc_sample:
            cx, cy = positions[c['ref']]
            wire_sexprs.append(s_wire(cx, vcc_y, cx, cy))
            junction_pts.add((cx, vcc_y))

    # GND rail at bottom
    gnd_y = sheet_h - 5
    gnd_comps = [c for c in comps if c['type'] in ('R', 'C', 'Q_NPN', 'Q_PNP', 'Q_NMOS', 'D', 'LED', 'ZD', 'F', 'L')]
    if len(gnd_comps) >= 2:
        gnd_sample = rng.sample(gnd_comps, min(3, len(gnd_comps)))
        gnd_xs = [positions[c['ref']][0] for c in gnd_sample]
        min_x, max_x = min(gnd_xs) - 5, max(gnd_xs) + 5
        wire_sexprs.append(s_wire(min_x, gnd_y, max_x, gnd_y))
        for c in gnd_sample:
            cx, cy = positions[c['ref']]
            wire_sexprs.append(s_wire(cx, gnd_y, cx, cy))
            junction_pts.add((cx, gnd_y))

    # Junction sexprs
    junction_sexprs = [s_junction(x, y) for x, y in junction_pts]

    # Text labels for VCC and GND
    vcc_label = s_text('VCC', min_x + 2, vcc_y + 3, 1.5)
    gnd_label = s_text('GND', min_x + 2, gnd_y - 3, 1.5)

    # Assemble full .kicad_sch content
    schematic_lines = [
        '(kicad_sch (version 20230121) (generator "eeschema")',
        '',
        f'  (paper "A4")',
        '',
        '  (lib_symbols)',
        '',
    ]

    for sexpr in symbol_sexprs:
        schematic_lines.append(f'  {sexpr}')
        schematic_lines.append('')

    for wexpr in wire_sexprs:
        schematic_lines.append(f'  {wexpr}')

    for jexpr in junction_sexprs:
        schematic_lines.append(f'  {jexpr}')

    schematic_lines.append(f'  {vcc_label}')
    schematic_lines.append(f'  {gnd_label}')

    # Sheet boundary
    schematic_lines.append(f'  (sheet_border (at 5 5) (size {sheet_w - 10} {sheet_h - 10}))')

    schematic_lines.append(')')

    schematic_text = '\n'.join(schematic_lines)

    # Build GT labels (spatial order: top-to-bottom, left-to-right)
    gt_items = []
    for comp in comps:
        x, y = comp['pos']
        gt_items.append((y, x, 'comp', comp['ref'], comp['value']))
    gt_items.append((vcc_y + 3, min_x + 2, 'label', 'VCC', ''))
    gt_items.append((gnd_y - 3, min_x + 2, 'label', 'GND', ''))

    gt_items.sort(key=lambda t: (t[0], t[1]))
    gt_lines = []
    for _, _, kind, label, value in gt_items:
        if kind == 'comp':
            gt_lines.append(label)
            gt_lines.append(value)
        else:
            gt_lines.append(label)

    return schematic_text, '\n'.join(gt_lines)


def batch_generate(out_dir, n_samples, start_idx=0):
    """Generate batch of .kicad_sch files with GT."""
    os.makedirs(out_dir, exist_ok=True)
    sch_dir = os.path.join(out_dir, 'schematic_files')
    os.makedirs(sch_dir, exist_ok=True)

    entries = []
    t0 = time.time()
    log_every = max(1, n_samples // 20)

    for i in range(start_idx, start_idx + n_samples):
        seed = i * 137 + 42
        rng = random.Random(seed)

        try:
            sch_text, gt_text = generate_schematic(rng)
        except Exception as e:
            print(f"[GEN] ERR {i}: {e}", flush=True)
            continue

        # Save .kicad_sch file
        sch_name = f'synth_{i:06d}.kicad_sch'
        sch_path = os.path.join(sch_dir, sch_name)
        with open(sch_path, 'w', encoding='utf-8') as f:
            f.write(sch_text)

        # Build JSONL entry (PNG path will be filled after rendering)
        png_name = f'synth_{i:06d}.png'
        png_path = os.path.join(out_dir, 'images', png_name)

        entry = {
            'images': [png_path.replace('\\', '/')],
            'sch_file': sch_path.replace('\\', '/'),
            'messages': [
                {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'OCR:'}]},
                {'role': 'assistant', 'content': gt_text},
            ]
        }
        entries.append(entry)

        if (i - start_idx + 1) % log_every == 0:
            done = i - start_idx + 1
            et = (time.time() - t0) / 60
            rate = done / max(et, 0.01)
            print(f"[GEN] {done}/{n_samples} .kicad_sch files ({rate:.0f}/min) ETA={((n_samples - done) / max(rate, 0.01)):.0f}m", flush=True)

    tt = (time.time() - t0) / 60
    print(f"\n[GEN] DONE: {n_samples} .kicad_sch files in {tt:.0f}m ({n_samples / tt:.0f}/min)", flush=True)

    # Save JSONL (without image paths — they'll be fixed after rendering)
    jl_path = os.path.join(out_dir, 'train_synthetic_kicad.jsonl')
    with open(jl_path, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    print(f"[GEN] JSONL: {jl_path}", flush=True)
    return entries, sch_dir


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_samples', type=int, default=5000)
    ap.add_argument('--output_dir', default='g:/mimo_project/circuit_ocr/output/synthetic_kicad')
    ap.add_argument('--test', action='store_true')
    args = ap.parse_args()

    if args.test:
        args.n_samples = 5
        args.output_dir += '_test'

    print(f"[GEN] Generating {args.n_samples} KiCad schematic files...", flush=True)
    entries, sch_dir = batch_generate(args.output_dir, args.n_samples)
    print(f"[GEN] Schematic files: {sch_dir}", flush=True)

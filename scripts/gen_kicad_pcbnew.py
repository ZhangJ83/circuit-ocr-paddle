"""Generate KiCad schematics using the pcbnew Python API.
Uses KiCad 10.0's own Python interpreter for guaranteed compatibility.
Creates proper .kicad_sch files that kicad-cli can render to SVG/PDF.

Usage:
  E:/080000software/Kicad/bin/python.exe gen_kicad_pcbnew.py --count 5000
"""
import os, sys, json, time, random, math, argparse
import tempfile, shutil

import pcbnew

# ============================================================
# KiCad symbol library IDs (from standard Device library)
# ============================================================
# These are guaranteed available in every KiCad installation
SYMBOL_IDS = {
    'R':    'Device:R',
    'C':    'Device:C',
    'CP':   'Device:CP',
    'L':    'Device:L',
    'D':    'Device:D',
    'LED':  'Device:LED',
    'ZD':   'Device:D_Zener',
    'Q_NPN': 'Device:Q_NPN_BCE',
    'Q_PNP': 'Device:Q_PNP_BCE',
    'F':    'Device:Fuse',
    'RV':   'Device:R_Potentiometer',
    'J':    'Connector:Conn_01x04_Male',
    'J2':   'Connector:Conn_01x02_Male',
    'SW':   'Switch:SW_SPST',
    'Y':    'Device:Crystal',
    'BAT':  'Device:Battery_Cell',
}

VALUE_POOLS = {
    'R':    ['10', '22', '47', '100', '220', '470', '1k', '2.2k', '4.7k', '10k', '22k', '47k', '100k', '220k', '470k', '1M'],
    'C':    ['10pF', '22pF', '47pF', '100pF', '220pF', '1nF', '10nF', '22nF', '100nF', '220nF', '470nF'],
    'CP':   ['1uF', '2.2uF', '4.7uF', '10uF', '22uF', '47uF', '100uF', '220uF'],
    'L':    ['10uH', '22uH', '47uH', '100uH', '220uH', '470uH', '1mH'],
    'D':    ['1N4148', '1N4007', '1N5819', 'BAT54S', 'SS14'],
    'LED':  ['Red', 'Green', 'Blue', 'Yellow', 'White'],
    'ZD':   ['3.3V', '3.6V', '5.1V', '6.2V', '12V', 'BZX84C3V3'],
    'Q_NPN': ['2N3904', 'BC547', 'S8050', '2N2222'],
    'Q_PNP': ['2N3906', 'BC557', 'S8550'],
    'F':    ['500mA', '1A', '2A', '3A', '5A'],
    'RV':   ['10k', '50k', '100k', '500k'],
    'J':    ['CONN_4P', 'HEADER_4P'],
    'J2':   ['CONN_2P', 'HEADER_2P'],
    'SW':   ['SW'],
    'Y':    ['8MHz', '12MHz', '16MHz', '25MHz', '32.768kHz'],
    'BAT':  ['3.3V', '5V', '3.7V'],
}

PREFIXES = {
    'R': 'R', 'C': 'C', 'CP': 'C', 'L': 'L',
    'D': 'D', 'LED': 'LED', 'ZD': 'D',
    'Q_NPN': 'Q', 'Q_PNP': 'Q', 'F': 'F',
    'RV': 'RV', 'J': 'J', 'J2': 'J', 'SW': 'SW',
    'Y': 'Y', 'BAT': 'BAT',
}


def mm_to_nm(mm):
    """Convert millimeters to nanometers (KiCad's internal unit)."""
    return int(mm * 1_000_000)


def generate_one_schematic(rng, sch_path, idx):
    """Generate one KiCad schematic using pcbnew API."""
    n_comp = rng.randint(8, 35)

    # Component type distribution (weighted toward passives)
    type_choices = ['R'] * 6 + ['C'] * 5 + ['CP'] * 2 + ['L'] * 2 + \
                   ['D'] * 2 + ['LED'] * 1 + ['ZD'] * 1 + \
                   ['Q_NPN'] * 1 + ['F'] * 1 + ['J'] * 2 + ['Y'] * 1 + ['RV'] * 1
    chosen = [rng.choice(type_choices) for _ in range(n_comp)]

    # Build component list with refdes
    counters = {}
    comps = []
    for ctype in chosen:
        counters[ctype] = counters.get(ctype, 0) + 1
        prefix = PREFIXES.get(ctype, 'U')
        ref = f'{prefix}{counters[ctype]}'
        value_pool = VALUE_POOLS.get(ctype, ['?'])
        value = rng.choice(value_pool)
        comps.append({'type': ctype, 'ref': ref, 'value': value})

    # Layout on grid
    cols = max(3, int(math.sqrt(n_comp * 1.5)))
    rows = max(2, (n_comp + cols - 1) // cols)
    cell_w_nm = mm_to_nm(20)
    cell_h_nm = mm_to_nm(18)
    margin_nm = mm_to_nm(12)

    # Create schematic using pcbnew API
    # New schematic: create sheet and screen
    sheet = pcbnew.SCH_SHEET()
    screen = pcbnew.SCH_SCREEN(sheet)
    screen.SetFileName(sch_path)

    # Sheet size
    page_info = screen.GetPageSettings()
    page_info.SetWidthMils(8000)   # ~A4 width in mils
    page_info.SetHeightMils(6000)  # ~A4 height in mils

    # Collect GT items for label generation
    gt_items = []

    # Place components
    symbols = []
    for i, comp in enumerate(comps):
        col, row = i % cols, i // cols
        # Position with jitter
        jx = rng.randint(-2, 2)
        jy = rng.randint(-2, 2)
        x_nm = margin_nm + col * cell_w_nm + cell_w_nm // 2 + mm_to_nm(jx)
        y_nm = margin_nm + row * cell_h_nm + cell_h_nm // 2 + mm_to_nm(jy)

        try:
            lib_id = pcbnew.LIB_ID(SYMBOL_IDS.get(comp['type'], 'Device:R'))
            symbol = pcbnew.SCH_SYMBOL()
            symbol.SetLibId(lib_id)
            symbol.SetPosition(pcbnew.VECTOR2I(x_nm, y_nm))

            # Set reference and value
            symbol.GetField(pcbnew.REFERENCE_FIELD).SetText(comp['ref'])
            symbol.GetField(pcbnew.VALUE_FIELD).SetText(comp['value'])

            screen.Append(symbol)
            symbols.append(symbol)
            gt_items.append((y_nm, x_nm, 'comp', comp['ref'], comp['value']))
        except Exception as e:
            # Skip if symbol not found in library
            pass

    if len(symbols) < 3:
        return None  # Too few symbols

    # Add wires — connect adjacent components in a chain
    # For simplicity, connect consecutive symbols with Manhattan routing
    connected = set()
    for i in range(len(symbols) - 1):
        if rng.random() < 0.5:  # 50% chance of connection
            try:
                s1, s2 = symbols[i], symbols[i + 1]
                p1 = s1.GetPosition()
                p2 = s2.GetPosition()

                # Get pin positions (approximate: use symbol edges)
                x1, y1 = int(p1.x), int(p1.y)
                x2, y2 = int(p2.x), int(p2.y)

                # Manhattan routing
                wire1 = pcbnew.SCH_LINE()
                wire1.SetStartPoint(pcbnew.VECTOR2I(x1, y1))
                wire1.SetEndPoint(pcbnew.VECTOR2I(x2, y1))
                screen.Append(wire1)

                wire2 = pcbnew.SCH_LINE()
                wire2.SetStartPoint(pcbnew.VECTOR2I(x2, y1))
                wire2.SetEndPoint(pcbnew.VECTOR2I(x2, y2))
                screen.Append(wire2)

                connected.add(i)
            except Exception:
                pass

    # Add VCC and GND text labels
    x_mid = margin_nm + cols * cell_w_nm // 2
    vcc_text = pcbnew.SCH_TEXT(pcbnew.VECTOR2I(margin_nm, margin_nm - mm_to_nm(5)))
    vcc_text.SetText('VCC')
    vcc_text.SetTextSize(pcbnew.VECTOR2I(mm_to_nm(1.5), mm_to_nm(1.5)))
    screen.Append(vcc_text)
    gt_items.append((margin_nm - mm_to_nm(5), margin_nm, 'label', 'VCC', ''))

    gnd_text = pcbnew.SCH_TEXT(pcbnew.VECTOR2I(margin_nm, margin_nm + rows * cell_h_nm))
    gnd_text.SetText('GND')
    gnd_text.SetTextSize(pcbnew.VECTOR2I(mm_to_nm(1.5), mm_to_nm(1.5)))
    screen.Append(gnd_text)
    gt_items.append((margin_nm + rows * cell_h_nm, margin_nm, 'label', 'GND', ''))

    # Save
    screen.Save(sch_path)

    # Build GT text
    gt_items.sort(key=lambda t: (t[0], t[1]))
    gt_lines = []
    for _, _, kind, label, value in gt_items:
        if kind == 'comp':
            gt_lines.append(label)
            gt_lines.append(value)
        else:
            gt_lines.append(label)

    return '\n'.join(gt_lines)


def batch_generate(out_dir, n_samples):
    """Generate batch of schematics."""
    os.makedirs(out_dir, exist_ok=True)
    sch_dir = os.path.join(out_dir, 'schematics')
    os.makedirs(sch_dir, exist_ok=True)

    # Set up KiCad project for symbol library access
    prj_dir = os.path.join(out_dir, '.kicad_pro')
    os.makedirs(prj_dir, exist_ok=True)

    entries = []
    t0 = time.time()
    log_every = max(1, n_samples // 20)

    for i in range(n_samples):
        seed = i * 137 + 42
        rng = random.Random(seed)
        sch_name = f'synth_{i:06d}.kicad_sch'
        sch_path = os.path.join(sch_dir, sch_name)

        try:
            gt_text = generate_one_schematic(rng, sch_path, i)
        except Exception as e:
            print(f"[GEN] ERR {i}: {e}", flush=True)
            continue

        if gt_text is None:
            continue

        png_name = f'synth_{i:06d}.png'
        png_path = os.path.join(out_dir, 'images', png_name)
        os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)

        entry = {
            'images': [png_path.replace('\\', '/')],
            'sch_file': sch_path.replace('\\', '/'),
            'messages': [
                {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'OCR:'}]},
                {'role': 'assistant', 'content': gt_text},
            ]
        }
        entries.append(entry)

        if (i + 1) % log_every == 0:
            done = i + 1
            et = (time.time() - t0) / 60
            rate = done / max(et, 0.01)
            print(f"[GEN] {done}/{n_samples} ({rate:.0f}/min) ETA={((n_samples - done) / max(rate, 0.01)):.0f}m", flush=True)

    tt = (time.time() - t0) / 60
    print(f"\n[GEN] DONE: {len(entries)} schematics in {tt:.0f}m ({len(entries) / tt:.0f}/min)", flush=True)

    # Save JSONL
    jl_path = os.path.join(out_dir, 'train_synthetic_kicad.jsonl')
    with open(jl_path, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    return entries, sch_dir


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', type=int, default=5000)
    ap.add_argument('--out', default='g:/mimo_project/circuit_ocr/output/synthetic_kicad_pcbnew')
    ap.add_argument('--test', action='store_true')
    args = ap.parse_args()

    if args.test:
        args.count = 5
        args.out += '_test'

    print(f"[GEN] Generating {args.count} KiCad schematics via pcbnew...", flush=True)
    entries, sch_dir = batch_generate(args.out, args.count)
    print(f"[GEN] Output: {sch_dir}", flush=True)

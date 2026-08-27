"""Generate KiCad .kicad_sch files with correct S-expression format.
Uses the circuit_ocr custom symbol library (embedded in lib_symbols).
Format verified against working KiCad 10 files.
"""
import os, sys, json, time, random, math, argparse, uuid
from pathlib import Path
from collections import defaultdict

# Custom symbol library definitions (extracted from working .kicad_sch files)
# These define the circuit_ocr custom symbols that are self-contained
LIB_SYMBOLS = """  (lib_symbols
    (symbol "circuit_ocr:Resistor"
      (pin_names (offset 1.016))
      (pin passive line (at 0 -2.54 0) (length 2.54) (name "1" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 0.00 0) (length 2.54) (name "2" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
    )
    (symbol "circuit_ocr:Capacitor"
      (pin_names (offset 1.016))
      (pin passive line (at 0 -2.54 0) (length 2.54) (name "1" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 0.00 0) (length 2.54) (name "2" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
    )
    (symbol "circuit_ocr:Capacitor_Polarized"
      (pin_names (offset 1.016))
      (pin passive line (at 0 -2.54 0) (length 2.54) (name "1" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 0.00 0) (length 2.54) (name "2" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
    )
    (symbol "circuit_ocr:Inductor"
      (pin_names (offset 1.016))
      (pin passive line (at 0 -2.54 0) (length 2.54) (name "1" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 0.00 0) (length 2.54) (name "2" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
    )
    (symbol "circuit_ocr:Diode"
      (pin_names (offset 1.016))
      (pin passive line (at 0 -2.54 0) (length 2.54) (name "A" (effects (font (size 1.27 1.27)))) (number "A" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 0.00 0) (length 2.54) (name "K" (effects (font (size 1.27 1.27)))) (number "K" (effects (font (size 1.27 1.27)))))
    )
    (symbol "circuit_ocr:LED"
      (pin_names (offset 1.016))
      (pin passive line (at 0 -2.54 0) (length 2.54) (name "A" (effects (font (size 1.27 1.27)))) (number "A" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 0.00 0) (length 2.54) (name "K" (effects (font (size 1.27 1.27)))) (number "K" (effects (font (size 1.27 1.27)))))
    )
    (symbol "circuit_ocr:Zener"
      (pin_names (offset 1.016))
      (pin passive line (at 0 -2.54 0) (length 2.54) (name "A" (effects (font (size 1.27 1.27)))) (number "A" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 0.00 0) (length 2.54) (name "K" (effects (font (size 1.27 1.27)))) (number "K" (effects (font (size 1.27 1.27)))))
    )
    (symbol "circuit_ocr:BJT_NPN"
      (pin_names (offset 1.016))
      (pin passive line (at 0 -5.08 0) (length 2.54) (name "C" (effects (font (size 1.27 1.27)))) (number "C" (effects (font (size 1.27 1.27)))))
      (pin passive line (at -5.08 0 0) (length 2.54) (name "B" (effects (font (size 1.27 1.27)))) (number "B" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 5.08 0) (length 2.54) (name "E" (effects (font (size 1.27 1.27)))) (number "E" (effects (font (size 1.27 1.27)))))
    )
    (symbol "circuit_ocr:BJT_PNP"
      (pin_names (offset 1.016))
      (pin passive line (at 0 -5.08 0) (length 2.54) (name "C" (effects (font (size 1.27 1.27)))) (number "C" (effects (font (size 1.27 1.27)))))
      (pin passive line (at -5.08 0 0) (length 2.54) (name "B" (effects (font (size 1.27 1.27)))) (number "B" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 5.08 0) (length 2.54) (name "E" (effects (font (size 1.27 1.27)))) (number "E" (effects (font (size 1.27 1.27)))))
    )
    (symbol "circuit_ocr:MOSFET_N"
      (pin_names (offset 1.016))
      (pin passive line (at 0 -5.08 0) (length 2.54) (name "D" (effects (font (size 1.27 1.27)))) (number "D" (effects (font (size 1.27 1.27)))))
      (pin passive line (at -5.08 0 0) (length 2.54) (name "G" (effects (font (size 1.27 1.27)))) (number "G" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 5.08 0) (length 2.54) (name "S" (effects (font (size 1.27 1.27)))) (number "S" (effects (font (size 1.27 1.27)))))
    )
    (symbol "circuit_ocr:OpAmp"
      (pin_names (offset 1.016))
      (pin passive line (at 0 -6.35 0) (length 2.54) (name "V+" (effects (font (size 1.27 1.27)))) (number "V+" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 -3.81 0) (length 2.54) (name "V-" (effects (font (size 1.27 1.27)))) (number "V-" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 -1.27 0) (length 2.54) (name "VCC" (effects (font (size 1.27 1.27)))) (number "VCC" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 1.27 0) (length 2.54) (name "VEE" (effects (font (size 1.27 1.27)))) (number "VEE" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 3.81 0) (length 2.54) (name "OUT" (effects (font (size 1.27 1.27)))) (number "OUT" (effects (font (size 1.27 1.27)))))
    )
    (symbol "circuit_ocr:IC"
      (pin_names (offset 1.016))
      (pin passive line (at -7.62 -7.62 0) (length 2.54) (name "1" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
      (pin passive line (at -7.62 -5.08 0) (length 2.54) (name "2" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
      (pin passive line (at -7.62 -2.54 0) (length 2.54) (name "3" (effects (font (size 1.27 1.27)))) (number "3" (effects (font (size 1.27 1.27)))))
      (pin passive line (at -7.62  0.00 0) (length 2.54) (name "4" (effects (font (size 1.27 1.27)))) (number "4" (effects (font (size 1.27 1.27)))))
      (pin passive line (at -7.62  2.54 0) (length 2.54) (name "5" (effects (font (size 1.27 1.27)))) (number "5" (effects (font (size 1.27 1.27)))))
      (pin passive line (at -7.62  5.08 0) (length 2.54) (name "6" (effects (font (size 1.27 1.27)))) (number "6" (effects (font (size 1.27 1.27)))))
      (pin passive line (at -7.62  7.62 0) (length 2.54) (name "7" (effects (font (size 1.27 1.27)))) (number "7" (effects (font (size 1.27 1.27)))))
      (pin passive line (at  7.62  7.62 0) (length 2.54) (name "8" (effects (font (size 1.27 1.27)))) (number "8" (effects (font (size 1.27 1.27)))))
      (pin passive line (at  7.62  5.08 0) (length 2.54) (name "9" (effects (font (size 1.27 1.27)))) (number "9" (effects (font (size 1.27 1.27)))))
      (pin passive line (at  7.62  2.54 0) (length 2.54) (name "10" (effects (font (size 1.27 1.27)))) (number "10" (effects (font (size 1.27 1.27)))))
      (pin passive line (at  7.62  0.00 0) (length 2.54) (name "11" (effects (font (size 1.27 1.27)))) (number "11" (effects (font (size 1.27 1.27)))))
      (pin passive line (at  7.62 -2.54 0) (length 2.54) (name "12" (effects (font (size 1.27 1.27)))) (number "12" (effects (font (size 1.27 1.27)))))
      (pin passive line (at  7.62 -5.08 0) (length 2.54) (name "13" (effects (font (size 1.27 1.27)))) (number "13" (effects (font (size 1.27 1.27)))))
      (pin passive line (at  7.62 -7.62 0) (length 2.54) (name "14" (effects (font (size 1.27 1.27)))) (number "14" (effects (font (size 1.27 1.27)))))
    )
    (symbol "circuit_ocr:Connector"
      (pin_names (offset 1.016))
      (pin passive line (at 0 -5.08 0) (length 2.54) (name "1" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 -2.54 0) (length 2.54) (name "2" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0  0.00 0) (length 2.54) (name "3" (effects (font (size 1.27 1.27)))) (number "3" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0  2.54 0) (length 2.54) (name "4" (effects (font (size 1.27 1.27)))) (number "4" (effects (font (size 1.27 1.27)))))
    )
    (symbol "circuit_ocr:Connector_2P"
      (pin_names (offset 1.016))
      (pin passive line (at 0 -2.54 0) (length 2.54) (name "1" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 0.00 0) (length 2.54) (name "2" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
    )
    (symbol "circuit_ocr:Crystal"
      (pin_names (offset 1.016))
      (pin passive line (at 0 -2.54 0) (length 2.54) (name "1" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 0.00 0) (length 2.54) (name "2" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
    )
    (symbol "circuit_ocr:Fuse"
      (pin_names (offset 1.016))
      (pin passive line (at 0 -2.54 0) (length 2.54) (name "1" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 0.00 0) (length 2.54) (name "2" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
    )
    (symbol "circuit_ocr:Switch"
      (pin_names (offset 1.016))
      (pin passive line (at 0 -2.54 0) (length 2.54) (name "1" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 0.00 0) (length 2.54) (name "2" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
    )
    (symbol "circuit_ocr:Battery"
      (pin_names (offset 1.016))
      (pin passive line (at 0 -2.54 0) (length 2.54) (name "+" (effects (font (size 1.27 1.27)))) (number "+" (effects (font (size 1.27 1.27)))))
      (pin passive line (at 0 0.00 0) (length 2.54) (name "-" (effects (font (size 1.27 1.27)))) (number "-" (effects (font (size 1.27 1.27)))))
    )
  )"""

# Symbol library IDs (all from circuit_ocr custom library)
SYM_IDS = {
    'R':     'circuit_ocr:Resistor',
    'C':     'circuit_ocr:Capacitor',
    'CP':    'circuit_ocr:Capacitor_Polarized',
    'L':     'circuit_ocr:Inductor',
    'D':     'circuit_ocr:Diode',
    'LED':   'circuit_ocr:LED',
    'ZD':    'circuit_ocr:Zener',
    'Q_NPN': 'circuit_ocr:BJT_NPN',
    'Q_PNP': 'circuit_ocr:BJT_PNP',
    'Q_NMOS':'circuit_OCR:MOSFET_N',
    'U':     'circuit_ocr:OpAmp',
    'IC':    'circuit_ocr:IC',
    'J':     'circuit_ocr:Connector',
    'J2':    'circuit_ocr:Connector_2P',
    'Y':     'circuit_ocr:Crystal',
    'F':     'circuit_ocr:Fuse',
    'SW':    'circuit_ocr:Switch',
    'BAT':   'circuit_ocr:Battery',
}

VALUE_POOLS = {
    'R':  ['10', '22', '47', '100', '220', '470', '1k', '2.2k', '4.7k', '10k', '22k', '47k', '100k', '220k', '470k', '1M'],
    'C':  ['10pF', '22pF', '47pF', '100pF', '220pF', '1nF', '10nF', '22nF', '100nF', '220nF', '470nF'],
    'CP': ['1uF', '2.2uF', '4.7uF', '10uF', '22uF', '47uF', '100uF', '220uF'],
    'L':  ['10uH', '22uH', '47uH', '100uH', '220uH', '470uH', '1mH'],
    'D':  ['1N4148', '1N4007', '1N5819', 'BAT54S', 'SS14'],
    'LED':['Red', 'Green', 'Blue', 'Yellow', 'White'],
    'ZD': ['3.3V', '3.6V', '5.1V', '6.2V', '12V', 'BZX84C3V3'],
    'Q_NPN': ['2N3904', 'BC547', 'S8050', '2N2222'],
    'Q_PNP': ['2N3906', 'BC557', 'S8550'],
    'Q_NMOS':['2N7002', 'BSS138'],
    'U':  ['TL072', 'LM358', 'LMV321', 'MCP6001', 'OPA333'],
    'IC': ['STM32F103', 'ATmega328P', 'ESP32', 'ATtiny85', 'RP2040', 'CH340G', 'MAX232'],
    'J':  ['CONN_4P', 'HEADER_4P', 'JST_4P'],
    'J2': ['CONN_2P', 'HEADER_2P'],
    'Y':  ['8MHz', '12MHz', '16MHz', '25MHz', '32.768kHz'],
    'F':  ['500mA', '1A', '2A', '3A', '5A'],
    'SW': ['SW'],
    'BAT':['3.3V', '5V', '3.7V'],
}

PREFIXES = {
    'R': 'R', 'C': 'C', 'CP': 'C', 'L': 'L',
    'D': 'D', 'LED': 'LED', 'ZD': 'D',
    'Q_NPN': 'Q', 'Q_PNP': 'Q', 'Q_NMOS': 'Q',
    'U': 'U', 'IC': 'U',
    'J': 'J', 'J2': 'J',
    'Y': 'Y', 'F': 'F', 'SW': 'SW', 'BAT': 'BAT',
}


def make_uuid(seed_str):
    """Generate a deterministic KiCad-style UUID."""
    h = abs(hash(seed_str)) % (10**15)
    return f"{h:015d}"


def s_symbol(lib_id, at_xy, ref, value, uuid_str):
    """Generate a correctly formatted symbol S-expression."""
    x, y, rot = at_xy[0], at_xy[1], at_xy[2] if len(at_xy) > 2 else 0
    return f"""  (symbol (lib_id "{lib_id}")
    (at {x:.2f} {y:.2f} {rot})
    (unit 1)
    (uuid "{uuid_str}")
    (property "Reference" "{ref}"
      (at {x:.2f} {y - 3.0:.2f} 0)
      (effects (font (size 1.27 1.27)))
    )
    (property "Value" "{value}"
      (at {x:.2f} {y + 3.0:.2f} 0)
      (effects (font (size 1.27 1.27)))
    )
  )"""


def s_wire(x1, y1, x2, y2, uuid_str):
    """Generate a wire S-expression."""
    return f"""  (wire
    (pts (xy {x1:.2f} {y1:.2f}) (xy {x2:.2f} {y2:.2f}))
    (stroke (width 0) (type default) (color 0 0 0 0))
    (uuid "{uuid_str}")
  )"""


def s_junction(x, y, uuid_str):
    """Generate a junction dot."""
    return f"""  (junction
    (at {x:.2f} {y:.2f})
    (diameter 0.91)
    (color 0 0 0 0)
    (uuid "{uuid_str}")
  )"""


def s_text(text, x, y, uuid_str):
    """Generate a text label."""
    return f"""  (text "{text}"
    (at {x:.2f} {y:.2f} 0)
    (effects (font (size 1.27 1.27)) (justify left bottom))
    (uuid "{uuid_str}")
  )"""


def s_label(text, x, y, uuid_str):
    """Generate a net label."""
    return f"""  (label "{text}"
    (at {x:.2f} {y:.2f} 0)
    (effects (font (size 1.27 1.27)))
    (uuid "{uuid_str}")
  )"""


def generate_one(rng, idx):
    """Generate one schematic."""
    n_comp = rng.randint(10, 40)

    # Component types (weighted)
    types = ['R'] * 6 + ['C'] * 5 + ['CP'] * 2 + ['L'] * 2 + \
            ['D'] * 2 + ['LED'] * 1 + ['ZD'] * 1 + \
            ['Q_NPN'] * 1 + ['F'] * 1 + ['J'] * 2 + ['Y'] * 1 + ['U'] * 1
    chosen = [rng.choice(types) for _ in range(n_comp)]

    # Build components
    counters = {}
    comps = []
    for ctype in chosen:
        counters[ctype] = counters.get(ctype, 0) + 1
        prefix = PREFIXES.get(ctype, 'U')
        ref = f'{prefix}{counters[ctype]}'
        pool = VALUE_POOLS.get(ctype, ['?'])
        value = rng.choice(pool)
        comps.append({'type': ctype, 'ref': ref, 'value': value})

    # Grid layout
    cols = max(3, int(math.sqrt(n_comp * 1.5)))
    rows = max(2, (n_comp + cols - 1) // cols)
    cell_w = 15.24  # mm
    cell_h = 12.7   # mm
    margin = 10.16  # mm

    # Place components
    positions = {}
    for i, comp in enumerate(comps):
        col, row = i % cols, i // cols
        jx = rng.uniform(-2, 2)
        jy = rng.uniform(-2, 2)
        x = margin + col * cell_w + cell_w / 2 + jx
        y = margin + row * cell_h + cell_h / 2 + jy
        # Snap to 1.27mm grid
        x_s = round(x / 1.27) * 1.27
        y_s = round(y / 1.27) * 1.27
        positions[comp['ref']] = (x_s, y_s)

    # Generate symbol S-expressions
    sym_lines = []
    for i, comp in enumerate(comps):
        lib_id = SYM_IDS.get(comp['type'], 'circuit_ocr:Resistor')
        x, y = positions[comp['ref']]
        uid = make_uuid(f"{idx}-sym-{i}")
        sym_lines.append(s_symbol(lib_id, (x, y, 0), comp['ref'], comp['value'], uid))

    # Generate wires (connect random nearby pairs)
    wire_lines = []
    junc_lines = []
    junc_set = set()
    comp_refs = list(positions.keys())
    rng.shuffle(comp_refs)
    wire_count = 0

    for i in range(len(comp_refs) - 1):
        if rng.random() < 0.4:  # 40% chance
            r1, r2 = comp_refs[i], comp_refs[i + 1]
            x1_pt, y1_pt = positions[r1]
            x2_pt, y2_pt = positions[r2]

            # Manhattan routing with junction
            mx = (x1_pt + x2_pt) / 2
            my = (y1_pt + y2_pt) / 2

            if abs(x1_pt - x2_pt) > abs(y1_pt - y2_pt):
                bx, by = x2_pt, y1_pt
            else:
                bx, by = x1_pt, y2_pt

            uid = make_uuid(f"{idx}-w{wire_count}")
            wire_lines.append(s_wire(x1_pt, y1_pt, bx, by, uid + "a"))
            wire_count += 1
            uid = make_uuid(f"{idx}-w{wire_count}")
            wire_lines.append(s_wire(bx, by, x2_pt, y2_pt, uid + "b"))
            wire_count += 1

            junc_pt = (round(bx, 2), round(by, 2))
            if junc_pt not in junc_set:
                junc_set.add(junc_pt)
                uid = make_uuid(f"{idx}-j{len(junc_set)}")
                junc_lines.append(s_junction(bx, by, uid))

    # VCC and GND text labels
    text_lines = []
    vcc_x, vcc_y = margin, margin - 5.08
    gnd_x, gnd_y = margin, margin + rows * cell_h + 2.54

    text_lines.append(s_text("VCC", vcc_x, vcc_y, make_uuid(f"{idx}-vcc")))
    text_lines.append(s_text("GND", gnd_x, gnd_y, make_uuid(f"{idx}-gnd")))

    # Assemble full file
    sheet_w = margin * 2 + cols * cell_w
    sheet_h = margin * 2 + rows * cell_h + 5

    schematic = f"""(kicad_sch (version 20230121) (generator "circuit_ocr_synth")
  (uuid "{make_uuid(f'{idx}-main')}")
  (paper "A")

{LIB_SYMBOLS}

"""
    schematic += '\n'.join(sym_lines) + '\n'
    schematic += '\n'.join(wire_lines) + '\n'
    schematic += '\n'.join(junc_lines) + '\n'
    schematic += '\n'.join(text_lines) + '\n'

    # Sheet border
    schematic += f"""  (rectangle
    (start 5 5)
    (end {sheet_w - 5:.2f} {sheet_h - 5:.2f})
    (stroke (width 0) (type default) (color 0 0 0 0))
    (fill (type none))
  )
)"""

    # Build GT labels
    gt_items = []
    for comp in comps:
        x, y = positions[comp['ref']]
        gt_items.append((y, x, 'comp', comp['ref'], comp['value']))
    gt_items.append((vcc_y, vcc_x, 'label', 'VCC', ''))
    gt_items.append((gnd_y, gnd_x, 'label', 'GND', ''))

    gt_items.sort(key=lambda t: (t[0], t[1]))
    gt_lines = []
    for _, _, kind, label, value in gt_items:
        if kind == 'comp':
            gt_lines.append(label)
            gt_lines.append(value)
        else:
            gt_lines.append(label)

    return schematic, '\n'.join(gt_lines), sheet_w, sheet_h


def batch_generate(out_dir, n_samples):
    os.makedirs(out_dir, exist_ok=True)
    sch_dir = os.path.join(out_dir, 'schematics')
    os.makedirs(sch_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)

    entries = []
    t0 = time.time()
    log_every = max(1, n_samples // 20)

    for i in range(n_samples):
        seed = i * 137 + 42
        rng = random.Random(seed)

        try:
            sch_text, gt_text, sw, sh = generate_one(rng, i)
        except Exception as e:
            print(f"[GEN] ERR {i}: {e}", flush=True)
            continue

        sch_name = f'synth_{i:06d}.kicad_sch'
        sch_path = os.path.join(sch_dir, sch_name)
        with open(sch_path, 'w', encoding='utf-8') as f:
            f.write(sch_text)

        png_name = f'synth_{i:06d}.png'
        png_path = os.path.join(img_dir, png_name)

        entry = {
            'images': [png_path.replace('\\', '/')],
            'sch_file': sch_path.replace('\\', '/'),
            'sheet_size': (sw, sh),
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
    print(f"\n[GEN] Done: {len(entries)} schematics in {tt:.0f}m", flush=True)

    jl_path = os.path.join(out_dir, 'train_synthetic_kicad.jsonl')
    with open(jl_path, 'w', encoding='utf-8') as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    print(f"[GEN] JSONL: {jl_path}", flush=True)
    return entries, sch_dir


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', type=int, default=5000)
    ap.add_argument('--out', default='g:/mimo_project/circuit_ocr/output/synthetic_kicad_v3')
    ap.add_argument('--test', action='store_true')
    args = ap.parse_args()

    if args.test:
        args.count = 5; args.out += '_test'

    print(f"[GEN] {args.count} schematics (KiCad 10 format)...", flush=True)
    entries, sch_dir = batch_generate(args.out, args.count)
    print(f"[GEN] Done: {sch_dir}", flush=True)

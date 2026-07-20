"""
Block-aware coordinate sort. Each item assigned to innermost dashed rectangle.
Sort: block (by position) → y (tolerance) → x.
Component+pins GROUPED together. Title block at end.
"""
import json, sys, os, math, re
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATASET_DIR = PROJECT_DIR / "circuit-ocr-dataset"
SRC_DIR = DATASET_DIR / "data" / "open_schematics_v2" / "numbered" / "source"
OUT_DIR = PROJECT_DIR / "output" / "gt_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

Y_TOLERANCE = 2.0  # mm — tighter row grouping


def parse_sexp(text, start=0):
    while start < len(text) and text[start] in ' \t\r\n': start += 1
    if start >= len(text): return None, start
    if text[start] == '"':
        end = start + 1
        while end < len(text):
            if text[end] == '\\': end += 2
            elif text[end] == '"': return text[start+1:end], end + 1
            else: end += 1
        return text[start+1:], end
    if text[start] == '(':
        children = []
        pos = start + 1
        while pos < len(text):
            while pos < len(text) and text[pos] in ' \t\r\n': pos += 1
            if pos >= len(text): break
            if text[pos] == ')': pos += 1; break
            val, pos = parse_sexp(text, pos)
            if val is not None: children.append(val)
        return children, pos
    end = start
    while end < len(text) and text[end] not in ' \t\r\n()': end += 1
    return text[start:end], end


def sc(node, tag=None):
    if not isinstance(node, list): return []
    return [c for c in node[1:] if isinstance(c, list) and len(c) > 0 and (tag is None or c[0] == tag)]


def sf(node, idx=1):
    try: return float(node[idx])
    except: return 0.0


def has_hide(node):
    """Check if node contains a 'hide' marker (can be atom or (hide yes))."""
    if not isinstance(node, list): return False
    for child in node:
        if child == 'hide': return True
        if isinstance(child, list) and len(child) > 0 and child[0] == 'hide': return True
    return False

def get_prop(props, key):
    for p in props:
        if not isinstance(p, list) or len(p) < 3: continue
        if p[0] != 'property' or p[1] != key: continue
        v = clean_text(p[2]) if isinstance(p[2], str) else ''
        x, y = 0.0, 0.0
        at = sc(p, 'at')
        if at and len(at[0]) >= 3: x, y = sf(at[0], 1), sf(at[0], 2)
        hidden = False
        effects = sc(p, 'effects')
        if effects and has_hide(effects[0]): hidden = True
        return v, x, y, hidden
    return '', 0.0, 0.0, False


KICAD_ESCAPES = {
    '{slash}': '/', '{backslash}': '\\', '{lt}': '<', '{gt}': '>',
    '{amp}': '&', '{dblquote}': '"', '{verticalbar}': '|',
    '{tilde}': '~', '{caret}': '^', '{colon}': ':',
}


def clean_text(text):
    """Convert KiCad escape sequences to readable characters.
    \\n in KiCad text represents an actual line break in multiline text."""
    for old, new in KICAD_ESCAPES.items():
        text = text.replace(old, new)
    # KiCad multiline text uses literal \n for line breaks
    text = text.replace('\\n', '\n')
    # Strip trailing newlines (KiCad end-of-text markers, not real empty lines)
    text = text.rstrip('\n')
    return text


def cal(text):
    """Convert KiCad ~{TEXT} overline notation to Unicode overline (U+0305).
    _ is literal underscore and stays as-is.
    Also applies clean_text() for escape sequences."""
    COMBINING_OVERLINE = '̅'
    text = re.sub(r'~\{([^}]+)\}', lambda m: ''.join(c + COMBINING_OVERLINE for c in m.group(1)), text)
    text = clean_text(text)
    return text


def rotate(x, y, deg):
    if deg == 0: return x, y
    r = math.radians(deg)
    return x * math.cos(r) - y * math.sin(r), x * math.sin(r) + y * math.cos(r)


def get_blocks(root, sch_path):
    """Extract dashed rectangles from shapes section using kiutils (geometry only, no text)."""
    try:
        from kiutils.schematic import Schematic
        sch = Schematic().from_file(sch_path)
    except:
        return []
    blocks = []
    for sh in sch.shapes:
        if 'Rectangle' not in type(sh).__name__: continue
        stroke = getattr(sh, 'stroke', None)
        if not stroke or getattr(stroke, 'type', '') != 'dash': continue
        x1 = min(sh.start.X, sh.end.X); y1 = min(sh.start.Y, sh.end.Y)
        x2 = max(sh.start.X, sh.end.X); y2 = max(sh.start.Y, sh.end.Y)
        blocks.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'area': (x2-x1)*(y2-y1)})
    blocks.sort(key=lambda b: (round(b['y1']/10)*10, b['x1']))
    return blocks


def innermost_block(x, y, blocks):
    containing = [(b['area'], bi) for bi, b in enumerate(blocks)
                  if b['x1'] <= x <= b['x2'] and b['y1'] <= y <= b['y2']]
    containing.sort(key=lambda c: c[0])
    return containing[0][1] if containing else None


def parse(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    root, _ = parse_sexp(text, 0)
    blocks = get_blocks(root, str(path))

    groups = []      # [{y, x, block, lines}]
    standalone = []  # [{y, x, block, text}]
    title_lines = []

    # === Title block (extract from source) ===
    tb = sc(root, 'title_block')
    title_val, date_val, rev_val, company_val = '', '', '1', ''
    if tb:
        for field in tb[0]:
            if isinstance(field, list) and len(field) >= 2 and isinstance(field[1], str):
                if field[0] == 'title': title_val = field[1]
                elif field[0] == 'date': date_val = field[1]
                elif field[0] == 'rev': rev_val = field[1]
                elif field[0] == 'company': company_val = field[1]

    # Sheet info
    sheet_path = '/'
    for si in sc(root, 'sheet_instances'):
        for child in si:
            if isinstance(child, list) and child[0] == 'path' and len(child) >= 2 and isinstance(child[1], str):
                sheet_path = child[1]

    # Generator version — extract version string like "20240108" or "9.0.6"
    gen_ver = ''
    for child in root:
        if isinstance(child, list) and len(child) >= 2 and isinstance(child[1], str):
            if child[0] == 'generator': gen_ver = child[1]
            elif child[0] == 'generator_version': gen_ver = child[1]
    # Build version string: if gen_ver looks like date (8 digits), show as-is,
    # otherwise prefix with "eeschema "
    if gen_ver:
        if re.match(r'^\d{8}$', gen_ver):
            kicad_ver = f'KiCad E.D.A. (build {gen_ver})'
        elif gen_ver == 'eeschema':
            kicad_ver = 'KiCad E.D.A. eeschema'
        else:
            kicad_ver = f'KiCad E.D.A. {gen_ver}'
    else:
        kicad_ver = 'KiCad E.D.A.'

    paper_size = 'A4'
    paper = sc(root, 'paper')
    if paper and len(paper[0]) >= 2 and isinstance(paper[0][1], str):
        paper_size = paper[0][1]

    # File name from parquet metadata (matches rendered image)
    orig_name = os.path.basename(path) if path else ''
    orig_stem = orig_name.replace('.kicad_sch', '')
    file_name = orig_name
    try:
        file_map_path = DATASET_DIR / 'data/open_schematics_v2/file_map.json'
        with open(file_map_path, 'r', encoding='utf-8') as fm:
            file_map = json.load(fm)
        orig_png = file_map.get(orig_stem, '')
        if orig_png:
            file_name = orig_png.replace('.png', '.kicad_sch')
    except: pass

    # Title block — always all fields, empty if not in source
    title_lines.append(company_val if company_val else '')
    title_lines.append(f'Sheet: {sheet_path}')
    title_lines.append(f'File: {file_name}')
    title_lines.append(f'Title: {clean_text(title_val)}' if title_val else 'Title:')
    title_lines.append(f'Size: {paper_size}')
    title_lines.append(f'Date: {date_val}' if date_val else 'Date:')
    title_lines.append(f'Rev: {rev_val}' if rev_val and rev_val != '1' else 'Rev:')
    title_lines.append(kicad_ver)
    title_lines.append(f'Id: 1/1')

    # === lib_symbols: pin templates ===
    lib_nodes = sc(root, 'lib_symbols')
    lib_pins, lib_hidden, lib_texts = {}, {}, {}
    if lib_nodes:
        for sym in lib_nodes[0][1:]:
            if not isinstance(sym, list) or sym[0] != 'symbol': continue
            lid = ''
            for c in sym[1:]:
                if isinstance(c, str) and not c.startswith('('): lid = c; break
            if not lid: continue
            nh = any(has_hide(c) for c in sym if isinstance(c, list) and len(c) > 0 and c[0] == 'pin_names')
            sh = any(has_hide(c) for c in sym if isinstance(c, list) and len(c) > 0 and c[0] == 'pin_numbers')
            lib_hidden[lid] = (nh, sh)
            pins = []
            sym_texts = []  # (text, rel_x, rel_y) — text drawn inside symbol graphics
            def fp(node):
                if isinstance(node, list) and len(node) > 0 and node[0] == 'pin':
                    num, name, px, py = '', '', 0.0, 0.0
                    for c in node:
                        if isinstance(c, list) and len(c) > 0:
                            if c[0] == 'name' and len(c) >= 2: name = c[1] if isinstance(c[1], str) else ''
                            elif c[0] == 'number' and len(c) >= 2: num = c[1] if isinstance(c[1], str) else ''
                            elif c[0] == 'at' and len(c) >= 3: px, py = sf(c, 1), sf(c, 2)
                    pins.append((num, name, px, py))
                elif isinstance(node, list):
                    for c in node: fp(c)
            fp(sym)
            # Also find text elements in sub-symbols
            def ft(node):
                if isinstance(node, list) and len(node) > 0 and node[0] == 'text':
                    txt = ''
                    px, py = 0.0, 0.0
                    for c in node[1:]:
                        if isinstance(c, str) and not c.startswith('('): txt = c; break
                    at = sc(node, 'at')
                    if at and len(at[0]) >= 3: px, py = sf(at[0], 1), sf(at[0], 2)
                    if txt and txt != '~':
                        sym_texts.append((clean_text(txt), px, py))
                elif isinstance(node, list):
                    for c in node: ft(c)
            ft(sym)
            if pins or sym_texts:
                lib_pins[lid] = pins
                if sym_texts: lib_texts[lid] = sym_texts

    # === Placed symbols (group by reference for multi-unit components) ===
    lib_start = next((i for i, c in enumerate(root) if isinstance(c, list) and len(c) > 0 and c[0] == 'lib_symbols'), -1)
    ref_data = {}  # ref -> {val, val_hidden, ref_x, ref_y, ref_hidden, lib_ids: [(lid, sx, sy, sr)]}

    for child in root[lib_start+1:]:
        if not isinstance(child, list) or child[0] != 'symbol': continue
        lid, sx, sy, sr = '', 0.0, 0.0, 0.0
        props = []
        for sub in child[1:]:
            if not isinstance(sub, list) or len(sub) == 0: continue
            if sub[0] == 'lib_id' and len(sub) >= 2: lid = sub[1] if isinstance(sub[1], str) else ''
            elif sub[0] == 'at' and len(sub) >= 4: sx, sy, sr = sf(sub, 1), sf(sub, 2), sf(sub, 3)
            elif sub[0] == 'property': props.append(sub)
        if not lid: continue
        ref, rx, ry, rh = get_prop(props, 'Reference')
        val, vx, vy, vh = get_prop(props, 'Value')
        if not ref: continue

        if ref.startswith('#PWR') or ref.startswith('#FLG') or ref.startswith('#LOGO'):
            if val and not vh:
                standalone.append({'y': vy, 'x': vx, 'block': innermost_block(vx, vy, blocks), 'text': val})
            continue

        if ref not in ref_data:
            ref_data[ref] = {'val': val, 'val_hidden': vh, 'ref_x': rx, 'ref_y': ry, 'ref_hidden': rh, 'units': []}
        else:
            # Merge: keep non-hidden val if current one is hidden
            if vh and ref_data[ref]['val_hidden'] and val:
                ref_data[ref]['val'] = val
                ref_data[ref]['val_hidden'] = vh
            elif val and not vh:
                ref_data[ref]['val'] = val
                ref_data[ref]['val_hidden'] = False
        ref_data[ref]['units'].append((lid, sx, sy, sr))

    # Build groups from merged ref data
    for ref, rd in ref_data.items():
        lines = []
        if not rd['ref_hidden']:
            lines.append(ref)
            lines.append(rd['val'] if (rd['val'] and not rd['val_hidden']) else '')
        # Collect all pins and texts from all units
        all_pins = []
        all_texts = set()
        for lid, sx, sy, sr in rd['units']:
            if lid in lib_texts:
                for txt, trx, try_ in lib_texts[lid]:
                    all_texts.add(txt)
            if lid in lib_pins:
                snh, ssh = lib_hidden.get(lid, (False, False))
                for pn, pname, prx, pry in lib_pins[lid]:
                    ax, ay = rotate(prx, pry, sr); ax += sx; ay += sy
                    sn = pn and pn != '~' and not ssh
                    sm = pname and pname != '~' and not snh
                    if sn or sm:
                        nm = cal(pname) if pname else ''
                        if sn and sm: all_pins.append((pn, f'{pn} {nm}'))
                        elif sn: all_pins.append((pn, pn))
                        elif sm: all_pins.append((pn, nm))
        # Sort pins by number
        def pin_key(p):
            try: return (0, int(p[0]))
            except: return (1, p[0])
        all_pins.sort(key=pin_key)
        # Deduplicate pins (same number from different units)
        seen_pins = set()
        for _, pl in all_pins:
            if pl not in seen_pins:
                seen_pins.add(pl)
                lines.append(pl)
        for t in sorted(all_texts):
            lines.append(t)
        blk = innermost_block(rd['ref_x'], rd['ref_y'], blocks)
        groups.append({'y': rd['ref_y'], 'x': rd['ref_x'], 'block': blk, 'lines': lines})

    # === Labels (convert _ overline) ===
    for tag in ['label', 'global_label', 'hierarchical_label']:
        for node in sc(root, tag):
            txt = ''
            for c in node[1:]:
                if isinstance(c, str) and not c.startswith('('): txt = c; break
            if not txt or txt == '~': continue
            txt = cal(txt)
            ax = sc(node, 'at')
            x = sf(ax[0], 1) if (ax and len(ax[0]) >= 3) else 0.0
            y = sf(ax[0], 2) if (ax and len(ax[0]) >= 3) else 0.0
            standalone.append({'y': y, 'x': x, 'block': innermost_block(x, y, blocks), 'text': txt})

    # === Texts (no _ conversion, free text) ===
    for node in sc(root, 'text'):
        txt = ''
        for c in node[1:]:
            if isinstance(c, str) and not c.startswith('('): txt = c; break
        if not txt or txt == '~': continue
        txt = clean_text(txt)
        ax = sc(node, 'at')
        x = sf(ax[0], 1) if (ax and len(ax[0]) >= 3) else 0.0
        y = sf(ax[0], 2) if (ax and len(ax[0]) >= 3) else 0.0
        standalone.append({'y': y, 'x': x, 'block': innermost_block(x, y, blocks), 'text': txt})

    # === Sort: block (by position order) → y → x ===
    # Map block index to sort order (sorted by block position)
    blk_order = {i: i for i in range(len(blocks))}

    def sort_key(item):
        b = item.get('block')
        if b is None:
            return (999, 0, round(item['y']/Y_TOLERANCE), item['x'])
        else:
            return (blk_order.get(b, 999), 1, round(item['y']/Y_TOLERANCE), item['x'])

    groups.sort(key=sort_key)
    standalone.sort(key=sort_key)

    # Interleave groups and standalone by sort key
    all_items = []
    for g in groups: all_items.append((sort_key(g), 'G', g))
    for s in standalone: all_items.append((sort_key(s), 'S', s))
    all_items.sort(key=lambda a: a[0])

    # Build output
    lines = []
    for _, kind, item in all_items:
        if kind == 'G':
            for l in item['lines']:
                if l.strip(): lines.append(l)
        else:
            if item['text'].strip(): lines.append(item['text'])

    # Title block
    for tl in title_lines:
        if tl.strip(): lines.append(tl)

    # Border grid markers (from paper size)
    border = generate_border_markers(paper_size)
    for bm in border:
        if bm.strip(): lines.append(bm)

    return '\n'.join(lines), blocks


def generate_border_markers(paper_size='A4'):
    """Sheet border reference grid based on KiCad standard template.
    Returns empty list for unknown/custom paper sizes."""
    # KiCad standard template: (columns, rows) for each paper size
    # A4 landscape: 6 cols 1-6, 4 rows A-D
    # A3 landscape: 8 cols 1-8, 5 rows A-E
    # etc.
    GRID = {
        'A0': (20, 10), 'A1': (16, 8), 'A2': (10, 6), 'A3': (8, 5),
        'A4': (6, 4), 'A5': (4, 3),
        'USLetter': (7, 4), 'USLegal': (8, 4), 'USLedger': (10, 6),
        'B': (8, 5), 'C': (10, 6), 'A': (6, 4),  # ISO B/C, and 'A' likely A4
    }
    entry = GRID.get(paper_size)
    if entry is None:
        # Unknown paper size (e.g. "User") — skip border markers
        return []
    n_cols, n_rows = entry
    markers = []
    # Top numbers
    markers.extend([str(i) for i in range(1, n_cols + 1)])
    # Bottom numbers
    markers.extend([str(i) for i in range(1, n_cols + 1)])
    # Right letters
    markers.extend([chr(65 + i) for i in range(n_rows)])
    # Left letters
    markers.extend([chr(65 + i) for i in range(n_rows)])
    return markers


if __name__ == '__main__':
    sch_path = SRC_DIR / '001_TiebeDeclercq_Uart-programmer.kicad_sch'
    gt, blocks = parse(str(sch_path))

    out_path = OUT_DIR / '001_TiebeDeclercq_Uart-programmer.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(gt)

    lines = [l for l in gt.splitlines() if l.strip()]
    print(f'Blocks: {len(blocks)}, Lines: {len(lines)}')
    for i, b in enumerate(blocks):
        print(f"  B{i}: ({b['x1']:.0f},{b['y1']:.0f})-({b['x2']:.0f},{b['y2']:.0f})")
    print(f'Saved: {out_path}\n')
    for i, line in enumerate(lines):
        print(f'  {i+1:4d}: {line}')

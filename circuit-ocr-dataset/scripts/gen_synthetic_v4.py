"""
Synthetic circuit schematic generator v4.

Fixes the v3 root causes of template memorization / hallucination:
  1. Real random netlist wiring (pin-to-pin nets + VCC/GND rails), not the
     v3 fake "6 topologies" that all drew the same positional lines.
  2. Per-type reference designators (R1,R2,R3 / C1,C2 / U1,U2), not the v3
     global positional counter (R1,C2,U3,R4...) that leaked position into the label.
  3. Parametric value pools (large realistic ranges) so value-reading is open
     OCR, not 12-way classification from a tiny fixed list.
  4. Layout randomization (jitter, font-size, image-size, grid toggle).
  5. VCC/GND power rails with net labels (the test set has GND/VBUS/+3.3V;
     v3 had none -> distribution mismatch).

GT format is unchanged from v3: every drawn text (refdes, value, net label)
recorded with its (y,x) and emitted in spatial reading order, newline-joined.
Drop-in compatible with the training jsonl format.

Usage:
    python gen_synthetic_v4.py --count 5 --seed 0 --out data/synthetic_v4
    python gen_synthetic_v4.py --count 2000 --seed 42
"""
import os
import json
import random
import argparse
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DATASET_DIR = Path(__file__).parent.parent
DPI = 150
MM_TO_PX = DPI / 25.4


# ---- Component catalog (symbol geometry + value generators) -----------------
def _resistor(rng):
    mantissa = rng.choice([10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82])
    mult = rng.choice([1, 10, 100, 1000, 10000])
    ohm = mantissa * mult
    if ohm >= 1_000_000:
        return f"{ohm / 1_000_000:g}M"
    if ohm >= 1000:
        return f"{ohm / 1000:g}k"
    return str(ohm)


def _capacitor(rng):
    mantissa = rng.choice([10, 15, 22, 33, 47, 68, 100, 150, 220])
    unit = rng.choice(["pF", "nF", "uF"])
    # keep uF values small (large caps are rare in small schematics)
    if unit == "uF":
        mantissa = rng.choice([1, 2.2, 4.7, 10, 22, 47, 100])
        return f"{mantissa:g}uF"
    return f"{mantissa}{unit}"


def _inductor(rng):
    return rng.choice(["1uH", "2.2uH", "10uH", "22uH", "100uH", "1mH", "10mH", "4.7mH"])


COMPONENTS = {
    "R":   {"prefix": "R",   "pins": 2, "value": _resistor,   "w": 4, "h": 2},
    "C":   {"prefix": "C",   "pins": 2, "value": _capacitor,  "w": 4, "h": 2},
    "L":   {"prefix": "L",   "pins": 2, "value": _inductor,   "w": 4, "h": 2},
    "D":   {"prefix": "D",   "pins": 2, "value": lambda r: r.choice(
                ["1N4148", "1N4007", "BAT54", "BZX84", "1N5819", "SS34"]), "w": 3, "h": 3},
    "LED": {"prefix": "LED", "pins": 2, "value": lambda r: r.choice(
                ["Red", "Green", "Blue", "White", "Yellow"]), "w": 3, "h": 3},
    "Q":   {"prefix": "Q",   "pins": 3, "value": lambda r: r.choice(
                ["2N2222", "BC547", "2N7002", "AO3400", "IRF540", "S8050"]), "w": 4, "h": 4},
    "U":   {"prefix": "U",   "pins": 8, "value": lambda r: r.choice(
                ["STM32F103", "ESP32", "ATmega328", "RP2040", "NE555", "LM358",
                 "SN74HC00", "MCP3008", "TL072", "AMS1117"]), "w": 10, "h": 16},
    "J":   {"prefix": "J",   "pins": 4, "value": lambda r: r.choice(
                ["CONN-2", "CONN-4", "CONN-8", "HEADER-3", "USB-C", "DC-JACK", "JST-2"]), "w": 6, "h": 3},
    "Y":   {"prefix": "Y",   "pins": 2, "value": lambda r: r.choice(
                ["8MHz", "16MHz", "32.768kHz", "25MHz", "12MHz"]), "w": 3, "h": 3},
    "F":   {"prefix": "F",   "pins": 2, "value": lambda r: r.choice(
                ["1A", "2A", "5A", "500mA", "3A", "250mA"]), "w": 3, "h": 2},
}


def draw_component_symbol(draw, cx, cy, comp_type, value, ref, font, label_font):
    """Draw symbol + ref/value text.

    Returns (pins, (ref_y, ref_text, value_y, value_text)) so the caller can
    keep each component's ref+value adjacent in the GT label."""

    comp = COMPONENTS[comp_type]
    w_px = int(comp["w"] * MM_TO_PX)
    h_px = int(comp["h"] * MM_TO_PX)
    pins = []

    if comp_type in ("R", "L", "F"):
        n = 4
        xs = [cx - w_px // 2 + i * w_px // n for i in range(n + 1)]
        ys = [cy + (h_px // 3 if i % 2 == 0 else -h_px // 3) for i in range(n + 1)]
        for i in range(len(xs) - 1):
            draw.line([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])], fill="#000000", width=2)
        draw.line([(cx - w_px // 2 - 5, cy), (cx - w_px // 2, cy)], fill="#000000", width=2)
        draw.line([(cx + w_px // 2, cy), (cx + w_px // 2 + 5, cy)], fill="#000000", width=2)
        pins = [(cx - w_px // 2 - 5, cy), (cx + w_px // 2 + 5, cy)]
    elif comp_type == "C":
        draw.line([(cx - w_px // 2 - 5, cy), (cx - w_px // 3, cy)], fill="#000000", width=2)
        draw.line([(cx - w_px // 3, cy - h_px // 2), (cx - w_px // 3, cy + h_px // 2)], fill="#000000", width=2)
        draw.line([(cx + w_px // 3, cy - h_px // 2), (cx + w_px // 3, cy + h_px // 2)], fill="#000000", width=2)
        draw.line([(cx + w_px // 3, cy), (cx + w_px // 2 + 5, cy)], fill="#000000", width=2)
        pins = [(cx - w_px // 2 - 5, cy), (cx + w_px // 2 + 5, cy)]
    elif comp_type in ("D", "LED"):
        draw.line([(cx - w_px // 2 - 5, cy), (cx + w_px // 2 + 5, cy)], fill="#000000", width=2)
        draw.polygon([(cx - w_px // 3, cy - h_px // 2), (cx - w_px // 3, cy + h_px // 2), (cx + w_px // 3, cy)],
                     outline="#000000", fill=None, width=2)
        draw.line([(cx + w_px // 3, cy - h_px // 2), (cx + w_px // 3, cy + h_px // 2)], fill="#000000", width=2)
        if comp_type == "LED":
            a = 3
            draw.line([(cx + w_px // 3 + 3, cy - h_px // 2 - 2), (cx + w_px // 3 + 3 + a, cy - h_px // 2 - 2 - a)],
                      fill="#0000CC", width=1)
            draw.line([(cx + w_px // 3 + 3, cy - h_px // 2 - 2), (cx + w_px // 3 + 3 - a, cy - h_px // 2 - 2 - a)],
                      fill="#0000CC", width=1)
        pins = [(cx - w_px // 2 - 5, cy), (cx + w_px // 2 + 5, cy)]
    elif comp_type == "Q":
        r = max(w_px, h_px) // 2
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline="#000000", width=2)
        draw.line([(cx - r, cy), (cx - r - 8, cy)], fill="#000000", width=2)
        draw.line([(cx - 2, cy + r), (cx - 2, cy + r + 8)], fill="#000000", width=2)
        draw.line([(cx + 2, cy - r), (cx + 2, cy - r - 8)], fill="#000000", width=2)
        pins = [(cx - r - 8, cy), (cx + 2, cy - r - 8), (cx - 2, cy + r + 8)]  # gate/base, drain/col, source emit
    elif comp_type == "U":
        draw.rectangle([(cx - w_px // 2, cy - h_px // 2), (cx + w_px // 2, cy + h_px // 2)],
                       outline="#000000", fill="#F8F8F8", width=2)
        for i in range(4):
            py = cy - h_px // 3 + i * h_px // 4
            draw.line([(cx - w_px // 2 - 8, py), (cx - w_px // 2, py)], fill="#000000", width=2)
            draw.line([(cx + w_px // 2, py), (cx + w_px // 2 + 8, py)], fill="#000000", width=2)
            pins.append((cx - w_px // 2 - 8, py))
        for i in range(4):
            py = cy - h_px // 3 + i * h_px // 4
            pins.append((cx + w_px // 2 + 8, py))
    elif comp_type == "J":
        draw.rectangle([(cx - w_px // 2, cy - h_px // 2), (cx + w_px // 2, cy + h_px // 2)],
                       outline="#0000CC", fill="#F0F0FF", width=2)
        for i in range(4):
            py = cy - h_px // 3 + i * h_px // 4
            draw.line([(cx - w_px // 2 - 8, py), (cx - w_px // 2, py)], fill="#0000CC", width=2)
            pins.append((cx - w_px // 2 - 8, py))
    elif comp_type == "Y":
        draw.rectangle([(cx - w_px // 2, cy - h_px // 2), (cx + w_px // 2, cy + h_px // 2)],
                       outline="#CC6600", fill="#FFF8F0", width=2)
        draw.line([(cx - w_px // 2 - 5, cy), (cx - w_px // 2, cy)], fill="#000000", width=2)
        draw.line([(cx + w_px // 2, cy), (cx + w_px // 2 + 5, cy)], fill="#000000", width=2)
        pins = [(cx - w_px // 2 - 5, cy), (cx + w_px // 2 + 5, cy)]

    # Reference (above) + value (below) drawn on image; GT pairing returned to caller
    rx, ry = cx - w_px // 2, cy - h_px // 2 - 18
    draw.text((rx, ry), ref, fill="#0000CC", font=font)
    vx, vy = cx - w_px // 2, cy + h_px // 2 + 3
    draw.text((vx, vy), value, fill="#CC0000", font=label_font)
    return pins, (ry, ref, vy, value)  # (ref_y, ref_text, value_y, value_text) as a grouped block
    rx, ry = cx - w_px // 2, cy - h_px // 2 - 18
    draw.text((rx, ry), ref, fill="#0000CC", font=font)
    vx, vy = cx - w_px // 2, cy + h_px // 2 + 3
    draw.text((vx, vy), value, fill="#CC0000", font=label_font)
    return pins


def _junction(draw, x, y):
    r = 2
    draw.ellipse([(x - r, y - r), (x + r, y + r)], fill="#000000")


def _route_l(draw, p1, p2):
    """L-shaped orthogonal route p1->p2 (horizontal first, then vertical)."""
    draw.line([p1, (p2[0], p1[1])], fill="#000000", width=2)
    draw.line([(p2[0], p1[1]), p2], fill="#000000", width=2)


def generate_one(idx, rng):
    n_comps = rng.randint(5, 30)
    # choose component types, weighted toward passives (realistic)
    type_choices = list(COMPONENTS.keys())
    weights = [5, 4, 1, 2, 2, 2, 2, 2, 1, 1]
    chosen_types = [rng.choices(type_choices, weights=weights, k=1)[0] for _ in range(n_comps)]

    # per-type refdes counters
    counters = {}
    comps = []
    for ctype in chosen_types:
        c = COMPONENTS[ctype]
        counters[ctype] = counters.get(ctype, 0) + 1
        ref = f"{c['prefix']}{counters[ctype]}"
        comps.append({"type": ctype, "value": c["value"](rng), "ref": ref, "pins": c["pins"]})

    cols = max(2, int(n_comps ** 0.5))
    rows = (n_comps + cols - 1) // cols
    cell_w = rng.choice([110, 120, 130])
    cell_h = rng.choice([85, 90, 95])
    margin = rng.choice([35, 40, 45])
    width = margin * 2 + cols * cell_w
    height = margin * 2 + rows * cell_h

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    if rng.random() < 0.5:  # optional grid
        gs = int(2.54 * MM_TO_PX)
        for x in range(0, width, gs):
            draw.line([(x, 0), (x, height)], fill="#F0F0F0", width=1)
        for y in range(0, height, gs):
            draw.line([(0, y), (width, y)], fill="#F0F0F0", width=1)

    fs = rng.choice([18, 19, 20, 21])
    try:
        font = ImageFont.truetype("arial.ttf", fs)
        label_font = ImageFont.truetype("arial.ttf", fs - 2)
    except (OSError, IOError):
        font = label_font = ImageFont.load_default()

    # place components + collect pins + GT items (ref/value kept adjacent as a pair)
    all_pins = []  # (x, y, comp_ref, pin_idx)
    items = []     # (sort_y, sort_x, kind, payload): kind 'C'=(ref,value) 'L'=text
    for i, comp in enumerate(comps):
        col = i % cols
        row = i // cols
        cx = margin + col * cell_w + cell_w // 2 + rng.randint(-8, 8)
        cy = margin + row * cell_h + cell_h // 2 + rng.randint(-6, 6)
        pins, (ref_y, ref_text, value_y, value_text) = draw_component_symbol(
            draw, cx, cy, comp["type"], comp["value"], comp["ref"], font, label_font)
        for pi, (px, py) in enumerate(pins):
            all_pins.append((px, py, comp["ref"], pi))
        items.append((ref_y, cx, "C", (ref_text, value_text)))

    # ---- build a random netlist --------------------------------------------
    vcc_y = 20
    gnd_y = height - 14
    rng.shuffle(all_pins)
    nets = []  # list of (label_or_None, [pins])
    vcc_pins, gnd_pins, local = [], [], []
    for pin in all_pins:
        r = rng.random()
        if r < 0.18:
            vcc_pins.append(pin)
        elif r < 0.40:
            gnd_pins.append(pin)
        else:
            local.append(pin)
    if vcc_pins:
        nets.append(("VCC", vcc_pins))
    if gnd_pins:
        nets.append(("GND", gnd_pins))
    # group local pins into nets of size 2 (mostly), some 3
    i = 0
    while i < len(local):
        size = 3 if (rng.random() < 0.15 and i + 2 < len(local)) else 2
        nets.append((None, local[i:i + size]))
        i += size

    # ---- draw wires --------------------------------------------------------
    for label, pins in nets:
        if label == "VCC":
            xs = [p[0] for p in pins]
            draw.line([(min(xs) - 6, vcc_y), (max(xs) + 6, vcc_y)], fill="#000000", width=2)
            for (px, py, _, _) in pins:
                draw.line([(px, py), (px, vcc_y)], fill="#000000", width=2)
                _junction(draw, px, vcc_y)
            draw.text((min(xs) - 6, vcc_y - 16), "VCC", fill="#0000CC", font=font)
            items.append((vcc_y - 16, min(xs) - 6, "L", "VCC"))
        elif label == "GND":
            xs = [p[0] for p in pins]
            draw.line([(min(xs) - 6, gnd_y), (max(xs) + 6, gnd_y)], fill="#000000", width=2)
            for (px, py, _, _) in pins:
                draw.line([(px, py), (px, gnd_y)], fill="#000000", width=2)
                _junction(draw, px, gnd_y)
            draw.text((min(xs) - 6, gnd_y + 2), "GND", fill="#0000CC", font=font)
            items.append((gnd_y + 2, min(xs) - 6, "L", "GND"))
        else:
            # local net: route to centroid meeting point
            mx = sum(p[0] for p in pins) // len(pins)
            my = sum(p[1] for p in pins) // len(pins)
            for (px, py, _, _) in pins:
                draw.line([(px, py), (px, my)], fill="#000000", width=2)
                draw.line([(px, my), (mx, my)], fill="#000000", width=2)
            _junction(draw, mx, my)

    draw.rectangle([(5, 5), (width - 6, height - 6)], outline="#AAAAAA", width=1)

    # GT = spatial reading order; each component emits ref\nvalue (adjacent pair),
    # net labels interleave by position. Matches test-set GT format.
    items.sort(key=lambda t: (t[0], t[1]))
    lines = []
    for _, _, kind, payload in items:
        if kind == "C":
            lines.append(payload[0])   # ref
            lines.append(payload[1])   # value
        else:
            lines.append(payload)      # net label
    label = "\n".join(lines)

    return img, {
        "images": [],  # filled by caller with relative path
        "messages": [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "OCR:"}]},
            {"role": "assistant", "content": label},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="g:/mimo_project/circuit_ocr/output/synthetic_circuits_v4")
    args = ap.parse_args()

    out_dir = Path(args.out) if os.path.isabs(args.out) else DATASET_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    entries = []
    t0 = time.time()
    for i in range(args.count):
        try:
            img, convo = generate_one(i, rng)
            img_name = f"synth_v4_{i + 1:06d}.png"
            img.save(str(img_dir / img_name))
            convo["images"] = [str(img_dir / img_name).replace('\\', '/')]
            entries.append(convo)
        except Exception as e:
            print(f"  [{i}] Error: {e}")
            continue
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{args.count}] {time.time() - t0:.0f}s")

    jsonl_path = out_dir / f"train_synthetic_v4.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"\nDone: {len(entries)} images in {time.time() - t0:.0f}s")
    print(f"Images -> {img_dir}")
    print(f"JSONL  -> {jsonl_path}")
    if entries:
        print("\nSample GT:\n" + entries[0]["messages"][1]["content"][:300])


if __name__ == "__main__":
    main()

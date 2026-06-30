"""
Generate synthetic circuit images with GT = actual rendered text (spatial order).
Records every draw.text() call → GT is exactly what's on the image.
"""
import os, sys, json, random, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DATASET_DIR = Path(__file__).parent.parent
OUTPUT_DIR = DATASET_DIR / "data" / "synthetic_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150
MM_TO_PX = DPI / 25.4

COMPONENTS = {
    "R":  {"prefix":"R",  "name":"Resistor",   "pins":2, "values":["100","220","330","470","1k","2.2k","4.7k","10k","22k","47k","100k","1M"], "w":4, "h":2},
    "C":  {"prefix":"C",  "name":"Capacitor",   "pins":2, "values":["10pF","22pF","100pF","1nF","10nF","100nF","1uF","10uF","100uF"], "w":4, "h":2},
    "L":  {"prefix":"L",  "name":"Inductor",    "pins":2, "values":["1uH","10uH","100uH","1mH","10mH"], "w":4, "h":2},
    "D":  {"prefix":"D",  "name":"Diode",       "pins":2, "values":["1N4148","1N4007","BAT54","BZX84"], "w":3, "h":3},
    "LED":{"prefix":"LED","name":"LED",          "pins":2, "values":["Red","Green","Blue","White"], "w":3, "h":3},
    "Q":  {"prefix":"Q",  "name":"Transistor",   "pins":3, "values":["2N2222","BC547","2N7002","AO3400"], "w":4, "h":4},
    "U":  {"prefix":"U",  "name":"IC",           "pins":8, "values":["STM32F103","ESP32","ATmega328","RP2040","NE555","LM358","SN74HC00","MCP3008"], "w":10, "h":16},
    "J":  {"prefix":"J",  "name":"Connector",    "pins":4, "values":["CONN-2","CONN-4","CONN-8","HEADER-3","USB-C","DC-JACK"], "w":6, "h":3},
    "Y":  {"prefix":"Y",  "name":"Crystal",      "pins":2, "values":["8MHz","16MHz","32.768kHz","25MHz"], "w":3, "h":3},
    "F":  {"prefix":"F",  "name":"Fuse",         "pins":2, "values":["1A","2A","5A","500mA"], "w":3, "h":2},
}

TOPOLOGIES = ["series_chain", "parallel_bank", "mixed_network", "ic_centric", "power_supply", "filter_stage"]

def draw_component_symbol(draw, cx, cy, comp_type, value, ref, font, label_font, text_log):
    comp = COMPONENTS[comp_type]
    w_px = int(comp["w"] * MM_TO_PX)
    h_px = int(comp["h"] * MM_TO_PX)

    if comp_type == "R":
        n = 4
        xs = [cx - w_px//2 + i * w_px//n for i in range(n+1)]
        ys = [cy + (h_px//3 if i % 2 == 0 else -h_px//3) for i in range(n+1)]
        for i in range(len(xs)-1):
            draw.line([(xs[i], ys[i]), (xs[i+1], ys[i+1])], fill="#000000", width=2)
        draw.line([(cx - w_px//2 - 5, cy), (cx - w_px//2, cy)], fill="#000000", width=2)
        draw.line([(cx + w_px//2, cy), (cx + w_px//2 + 5, cy)], fill="#000000", width=2)
    elif comp_type == "C":
        draw.line([(cx - w_px//2 - 5, cy), (cx - w_px//3, cy)], fill="#000000", width=2)
        draw.line([(cx - w_px//3, cy - h_px//2), (cx - w_px//3, cy + h_px//2)], fill="#000000", width=2)
        draw.line([(cx + w_px//3, cy - h_px//2), (cx + w_px//3, cy + h_px//2)], fill="#000000", width=2)
        draw.line([(cx + w_px//3, cy), (cx + w_px//2 + 5, cy)], fill="#000000", width=2)
    elif comp_type == "L":
        r = h_px // 3
        for i in range(4):
            arc_cx = cx - w_px//4 + i * w_px//4
            draw.arc([(arc_cx - r, cy - r), (arc_cx + r, cy + r)], 0, 180, fill="#000000", width=2)
        draw.line([(cx - w_px//2 - 5, cy), (cx - w_px//2, cy)], fill="#000000", width=2)
        draw.line([(cx + w_px//2, cy), (cx + w_px//2 + 5, cy)], fill="#000000", width=2)
    elif comp_type in ("D","LED"):
        draw.line([(cx - w_px//2 - 5, cy), (cx + w_px//2 + 5, cy)], fill="#000000", width=2)
        draw.polygon([(cx - w_px//3, cy - h_px//2), (cx - w_px//3, cy + h_px//2), (cx + w_px//3, cy)], outline="#000000", fill=None, width=2)
        draw.line([(cx + w_px//3, cy - h_px//2), (cx + w_px//3, cy + h_px//2)], fill="#000000", width=2)
        if comp_type == "LED":
            arr_size = 3
            draw.line([(cx + w_px//3 + 3, cy - h_px//2 - 2), (cx + w_px//3 + 3 + arr_size, cy - h_px//2 - 2 - arr_size)], fill="#0000CC", width=1)
            draw.line([(cx + w_px//3 + 3, cy - h_px//2 - 2), (cx + w_px//3 + 3 - arr_size, cy - h_px//2 - 2 - arr_size)], fill="#0000CC", width=1)
    elif comp_type == "Q":
        r = max(w_px, h_px) // 2
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline="#000000", width=2)
        draw.line([(cx - r, cy), (cx - r - 8, cy)], fill="#000000", width=2)
        draw.line([(cx - 2, cy + r), (cx - 2, cy + r + 8)], fill="#000000", width=2)
        draw.line([(cx + 2, cy - r), (cx + 2, cy - r - 8)], fill="#000000", width=2)
    elif comp_type == "U":
        draw.rectangle([(cx - w_px//2, cy - h_px//2), (cx + w_px//2, cy + h_px//2)], outline="#000000", fill="#F8F8F8", width=2)
        for i in range(4):
            pin_y = cy - h_px//3 + i * h_px//4
            draw.line([(cx - w_px//2 - 8, pin_y), (cx - w_px//2, pin_y)], fill="#000000", width=2)
            draw.line([(cx + w_px//2, pin_y), (cx + w_px//2 + 8, pin_y)], fill="#000000", width=2)
    elif comp_type == "J":
        draw.rectangle([(cx - w_px//2, cy - h_px//2), (cx + w_px//2, cy + h_px//2)], outline="#0000CC", fill="#F0F0FF", width=2)
        for i in range(4):
            pin_y = cy - h_px//3 + i * h_px//4
            draw.line([(cx - w_px//2 - 8, pin_y), (cx - w_px//2, pin_y)], fill="#0000CC", width=2)
    elif comp_type == "Y":
        draw.rectangle([(cx - w_px//2, cy - h_px//2), (cx + w_px//2, cy + h_px//2)], outline="#CC6600", fill="#FFF8F0", width=2)
        draw.line([(cx - w_px//2 - 5, cy), (cx - w_px//2, cy)], fill="#000000", width=2)
        draw.line([(cx + w_px//2, cy), (cx + w_px//2 + 5, cy)], fill="#000000", width=2)
    elif comp_type == "F":
        draw.rectangle([(cx - w_px//2, cy - h_px//4), (cx + w_px//2, cy + h_px//4)], outline="#000000", fill="white", width=2)
        draw.line([(cx - w_px//2 - 5, cy), (cx - w_px//2, cy)], fill="#000000", width=2)
        draw.line([(cx + w_px//2, cy), (cx + w_px//2 + 5, cy)], fill="#000000", width=2)

    # Reference (above symbol) — record position
    rx, ry = cx - w_px//2, cy - h_px//2 - 18
    draw.text((rx, ry), ref, fill="#0000CC", font=font)
    text_log.append((ry, rx, ref))

    # Value (below symbol) — record position
    vx, vy = cx - w_px//2, cy + h_px//2 + 3
    draw.text((vx, vy), value, fill="#CC0000", font=label_font)
    text_log.append((vy, vx, value))

    return f"{ref} {value}"


def generate_one(idx):
    text_log = []  # (y, x, text) tuples for all drawn text

    topo = random.choice(TOPOLOGIES)
    num_components = random.randint(5, 30)

    comps = []
    for i in range(num_components):
        ctype = random.choice(list(COMPONENTS.keys()))
        c = COMPONENTS[ctype]
        value = random.choice(c["values"])
        ref = f"{c['prefix']}{i+1}"
        comps.append({"type": ctype, "value": value, "ref": ref})

    cols = max(2, int(num_components ** 0.5))
    rows = (num_components + cols - 1) // cols
    cell_w = 120
    cell_h = 90
    margin = 40
    width = margin * 2 + cols * cell_w
    height = margin * 2 + rows * cell_h

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Grid
    grid_spacing = int(2.54 * MM_TO_PX)
    for x in range(0, width, grid_spacing):
        draw.line([(x, 0), (x, height)], fill="#F0F0F0", width=1)
    for y in range(0, height, grid_spacing):
        draw.line([(0, y), (width, y)], fill="#F0F0F0", width=1)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
        label_font = ImageFont.truetype("arial.ttf", 18)
        title_font = ImageFont.truetype("arial.ttf", 22)
    except (OSError, IOError):
        font = label_font = title_font = ImageFont.load_default()

    # Components — only refs and values go to GT
    for i, comp in enumerate(comps):
        col = i % cols
        row = i // cols
        cx = margin + col * cell_w + cell_w // 2
        cy = margin + row * cell_h + cell_h // 2

        draw_component_symbol(draw, cx, cy, comp["type"], comp["value"],
                             comp["ref"], font, label_font, text_log)

        if topo == "series_chain" and col > 0:
            prev_cx = margin + (col-1) * cell_w + cell_w // 2
            draw.line([(prev_cx + 30, cy), (cx - 30, cy)], fill="#000000", width=2)
        elif topo == "parallel_bank" and col > 0 and row > 0:
            prev_cy = margin + (row-1) * cell_h + cell_h // 2
            draw.line([(cx, prev_cy + 30), (cx, cy - 30)], fill="#000000", width=2)
        elif col > 0 or row > 0:
            if col > 0:
                prev_cx = margin + (col-1) * cell_w + cell_w // 2
                draw.line([(prev_cx + 25, cy), (cx - 25, cy)], fill="#000000", width=1)

    # Border
    draw.rectangle([(5, 5), (width-6, height-6)], outline="#AAAAAA", width=1)

    # Sort all text by y, then x = spatial reading order
    text_log.sort(key=lambda t: (t[0], t[1]))
    label_lines = []
    for (y, x, txt) in text_log:
        label_lines.append(txt)

    label = "\n".join(label_lines)

    # Save image
    img_name = f"synth_v3_{idx+1:04d}.png"
    img_path = OUTPUT_DIR / img_name
    img.save(str(img_path))

    conversation = {
        "images": [f"data/synthetic_v3/{img_name}"],
        "messages": [
            {"role": "user", "content": "<image>\nOCR:"},
            {"role": "assistant", "content": label},
        ],
    }
    return conversation


def main():
    COUNT = 500
    print(f"Generating {COUNT} synthetic schematics with spatial-order GT...")
    t0 = time.time()

    jsonl_path = DATASET_DIR / "ocr_vl_sft-synthetic-v3.jsonl"
    entries = []

    for i in range(COUNT):
        try:
            entry = generate_one(i)
            entries.append(entry)
        except Exception as e:
            print(f"  [{i}] Error: {e}")
            continue
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{COUNT}] {elapsed:.0f}s")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    total_time = time.time() - t0
    print(f"\nDone! {len(entries)} images in {total_time:.0f}s")
    print(f"Images: {OUTPUT_DIR}")
    print(f"JSONL:  {jsonl_path} ({jsonl_path.stat().st_size / 1024:.0f} KB)")

    # Show a sample
    if entries:
        e = entries[0]
        print(f"\nSample label ({len(e['messages'][1]['content'])} chars):")
        print(e['messages'][1]['content'][:300])

if __name__ == "__main__":
    main()

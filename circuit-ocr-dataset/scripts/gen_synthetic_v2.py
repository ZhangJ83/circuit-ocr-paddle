"""
Generate high-quality synthetic circuit images and JSONL annotations.
Uses improved DPI=150 and font sizes from renderer.py fix.
Outputs images to data/synthetic_v2/ and creates ocr_vl_sft-synthetic-v2.jsonl
"""
import os, sys, json, random, time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

DATASET_DIR = Path(__file__).parent.parent
OUTPUT_DIR = DATASET_DIR / "data" / "synthetic_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150
MM_TO_PX = DPI / 25.4  # ~5.906 px/mm

# Component library
COMPONENTS = {
    "R": {"name": "Resistor", "pins": 2, "values": ["100","220","330","470","1k","2.2k","4.7k","10k","22k","47k","100k","1M"], "w": 4, "h": 2},
    "C": {"name": "Capacitor", "pins": 2, "values": ["10pF","22pF","100pF","1nF","10nF","100nF","1uF","10uF","100uF"], "w": 4, "h": 2},
    "L": {"name": "Inductor", "pins": 2, "values": ["1uH","10uH","100uH","1mH","10mH"], "w": 4, "h": 2},
    "D": {"name": "Diode", "pins": 2, "values": ["1N4148","1N4007","BAT54","BZX84"], "w": 3, "h": 3},
    "LED": {"name": "LED", "pins": 2, "values": ["Red","Green","Blue","White"], "w": 3, "h": 3},
    "Q": {"name": "Transistor", "pins": 3, "values": ["2N2222","BC547","2N7002","AO3400"], "w": 4, "h": 4},
    "U": {"name": "IC", "pins": 8, "values": ["STM32F103","ESP32","ATmega328","RP2040","NE555","LM358","SN74HC00","MCP3008"], "w": 10, "h": 16},
    "J": {"name": "Connector", "pins": 4, "values": ["CONN-2","CONN-4","CONN-8","HEADER-3","USB-C","DC-JACK"], "w": 6, "h": 3},
    "Y": {"name": "Crystal", "pins": 2, "values": ["8MHz","16MHz","32.768kHz","25MHz"], "w": 3, "h": 3},
    "F": {"name": "Fuse", "pins": 2, "values": ["1A","2A","5A","500mA"], "w": 3, "h": 2},
}

TOPOLOGIES = ["series_chain", "parallel_bank", "mixed_network", "ic_centric", "power_supply", "filter_stage"]

def draw_component_symbol(draw, cx, cy, comp_type, value, ref, font, label_font):
    """Draw a component symbol at (cx, cy) and return its netlist line."""
    comp = COMPONENTS[comp_type]
    w_px = int(comp["w"] * MM_TO_PX)
    h_px = int(comp["h"] * MM_TO_PX)

    if comp_type in ("R",):
        # Zigzag resistor
        n = 4
        xs = [cx - w_px//2 + i * w_px//n for i in range(n+1)]
        ys = [cy + (h_px//3 if i % 2 == 0 else -h_px//3) for i in range(n+1)]
        points = [(xs[i], ys[i]) for i in range(len(xs))]
        for i in range(len(points)-1):
            draw.line([points[i], points[i+1]], fill="#000000", width=2)
        # Pin lines
        draw.line([(cx - w_px//2 - 5, cy), (cx - w_px//2, cy)], fill="#000000", width=2)
        draw.line([(cx + w_px//2, cy), (cx + w_px//2 + 5, cy)], fill="#000000", width=2)
    elif comp_type in ("C",):
        draw.line([(cx - w_px//2 - 5, cy), (cx - w_px//3, cy)], fill="#000000", width=2)
        draw.line([(cx - w_px//3, cy - h_px//2), (cx - w_px//3, cy + h_px//2)], fill="#000000", width=2)
        draw.line([(cx + w_px//3, cy - h_px//2), (cx + w_px//3, cy + h_px//2)], fill="#000000", width=2)
        draw.line([(cx + w_px//3, cy), (cx + w_px//2 + 5, cy)], fill="#000000", width=2)
    elif comp_type in ("L",):
        r = h_px // 3
        for i in range(4):
            arc_cx = cx - w_px//4 + i * w_px//4
            draw.arc([(arc_cx - r, cy - r), (arc_cx + r, cy + r)], 0, 180, fill="#000000", width=2)
        draw.line([(cx - w_px//2 - 5, cy), (cx - w_px//2, cy)], fill="#000000", width=2)
        draw.line([(cx + w_px//2, cy), (cx + w_px//2 + 5, cy)], fill="#000000", width=2)
    elif comp_type in ("D",):
        draw.line([(cx - w_px//2 - 5, cy), (cx + w_px//2 + 5, cy)], fill="#000000", width=2)
        draw.polygon([(cx - w_px//3, cy - h_px//2), (cx - w_px//3, cy + h_px//2), (cx + w_px//3, cy)], outline="#000000", fill=None, width=2)
        draw.line([(cx + w_px//3, cy - h_px//2), (cx + w_px//3, cy + h_px//2)], fill="#000000", width=2)
    elif comp_type in ("LED",):
        draw.line([(cx - w_px//2 - 5, cy), (cx + w_px//2 + 5, cy)], fill="#000000", width=2)
        draw.polygon([(cx - w_px//3, cy - h_px//2), (cx - w_px//3, cy + h_px//2), (cx + w_px//3, cy)], outline="#000000", fill=None, width=2)
        draw.line([(cx + w_px//3, cy - h_px//2), (cx + w_px//3, cy + h_px//2)], fill="#000000", width=2)
        # arrows for LED
        arr_size = 3
        draw.line([(cx + w_px//3 + 3, cy - h_px//2 - 2), (cx + w_px//3 + 3 + arr_size, cy - h_px//2 - 2 - arr_size)], fill="#0000CC", width=1)
        draw.line([(cx + w_px//3 + 3, cy - h_px//2 - 2), (cx + w_px//3 + 3 - arr_size, cy - h_px//2 - 2 - arr_size)], fill="#0000CC", width=1)
    elif comp_type in ("Q",):
        # Transistor: circle with 3 pins
        r = max(w_px, h_px) // 2
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline="#000000", width=2)
        # E, B, C pins
        draw.line([(cx - r, cy), (cx - r - 8, cy)], fill="#000000", width=2)  # B
        draw.line([(cx - 2, cy + r), (cx - 2, cy + r + 8)], fill="#000000", width=2)  # E
        draw.line([(cx + 2, cy - r), (cx + 2, cy - r - 8)], fill="#000000", width=2)  # C
    elif comp_type in ("U",):
        # IC: rectangle with pins on both sides
        draw.rectangle([(cx - w_px//2, cy - h_px//2), (cx + w_px//2, cy + h_px//2)], outline="#000000", fill="#F8F8F8", width=2)
        for i in range(4):
            pin_y = cy - h_px//3 + i * h_px//4
            draw.line([(cx - w_px//2 - 8, pin_y), (cx - w_px//2, pin_y)], fill="#000000", width=2)
            draw.line([(cx + w_px//2, pin_y), (cx + w_px//2 + 8, pin_y)], fill="#000000", width=2)
    elif comp_type in ("J",):
        draw.rectangle([(cx - w_px//2, cy - h_px//2), (cx + w_px//2, cy + h_px//2)], outline="#0000CC", fill="#F0F0FF", width=2)
        for i in range(4):
            pin_y = cy - h_px//3 + i * h_px//4
            draw.line([(cx - w_px//2 - 8, pin_y), (cx - w_px//2, pin_y)], fill="#0000CC", width=2)
    elif comp_type in ("Y",):
        draw.rectangle([(cx - w_px//2, cy - h_px//2), (cx + w_px//2, cy + h_px//2)], outline="#CC6600", fill="#FFF8F0", width=2)
        draw.line([(cx - w_px//2 - 5, cy), (cx - w_px//2, cy)], fill="#000000", width=2)
        draw.line([(cx + w_px//2, cy), (cx + w_px//2 + 5, cy)], fill="#000000", width=2)
    elif comp_type in ("F",):
        draw.rectangle([(cx - w_px//2, cy - h_px//4), (cx + w_px//2, cy + h_px//4)], outline="#000000", fill="white", width=2)
        draw.line([(cx - w_px//2 - 5, cy), (cx - w_px//2, cy)], fill="#000000", width=2)
        draw.line([(cx + w_px//2, cy), (cx + w_px//2 + 5, cy)], fill="#000000", width=2)

    # Reference text (above)
    draw.text((cx - w_px//2, cy - h_px//2 - 18), ref, fill="#0000CC", font=font)
    # Value text (below)
    draw.text((cx - w_px//2, cy + h_px//2 + 3), value, fill="#CC0000", font=label_font)

    return f"{ref} {value}"

def generate_synthetic_schematic(idx):
    """Generate one synthetic circuit schematic and return JSONL entry."""
    # Pick topology
    topo = random.choice(TOPOLOGIES)
    num_components = random.randint(5, 30)

    # Select components
    comps = []
    for i in range(num_components):
        ctype = random.choice(list(COMPONENTS.keys()))
        c = COMPONENTS[ctype]
        value = random.choice(c["values"])
        ref = f"{ctype}{i+1}"
        comps.append({"type": ctype, "value": value, "ref": ref})

    # Calculate canvas size based on component count
    cols = max(2, int(num_components ** 0.5))
    rows = (num_components + cols - 1) // cols
    cell_w = 80
    cell_h = 70
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

    # Fonts
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        label_font = ImageFont.truetype("arial.ttf", 18)
        title_font = ImageFont.truetype("arial.ttf", 22)
    except (OSError, IOError):
        font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    # Title
    title = f"Schematic {idx+1} - {topo.replace('_',' ').title()} ({num_components} components)"
    draw.text((margin, 10), title, fill="#000000", font=title_font)

    # Draw components
    netlist_lines = []
    for i, comp in enumerate(comps):
        col = i % cols
        row = i // cols
        cx = margin + col * cell_w + cell_w // 2
        cy = margin + row * cell_h + cell_h // 2

        nl = draw_component_symbol(draw, cx, cy, comp["type"], comp["value"],
                                   comp["ref"], font, label_font)
        netlist_lines.append(nl)

        # Draw connecting wires based on topology
        if topo == "series_chain" and col > 0:
            prev_cx = margin + (col-1) * cell_w + cell_w // 2
            draw.line([(prev_cx + 30, cy), (cx - 30, cy)], fill="#000000", width=2)
        elif topo == "parallel_bank" and col > 0 and row > 0:
            # Vertical bus
            prev_cy = margin + (row-1) * cell_h + cell_h // 2
            draw.line([(cx, prev_cy + 30), (cx, cy - 30)], fill="#000000", width=2)
        elif col > 0 or row > 0:
            # Mixed: connect adjacent
            if col > 0:
                prev_cx = margin + (col-1) * cell_w + cell_w // 2
                draw.line([(prev_cx + 25, cy), (cx - 25, cy)], fill="#000000", width=1)

    # Random labels (voltages, signal names)
    label_texts = []
    for _ in range(random.randint(2, 8)):
        lx = random.randint(margin, width - margin)
        ly = random.randint(25, height - 10)
        labels = [f"VDD={random.choice(['3.3V','5V','1.8V','12V'])}",
                  f"GND", f"VCC", f"VIN", f"VOUT",
                  f"CLK", f"RST", f"EN", f"TX", f"RX",
                  f"SCL", f"SDA", f"MISO", f"MOSI", f"SCK",
                  f"AIN{random.randint(0,3)}", f"GPIO{random.randint(0,15)}"]
        lt = random.choice(labels)
        draw.text((lx, ly), lt, fill="#666666", font=label_font)
        label_texts.append(lt)

    # Draw border
    draw.rectangle([(5, 5), (width-6, height-6)], outline="#AAAAAA", width=1)

    # Save image
    img_name = f"synth_v2_{idx+1:04d}.png"
    img_path = OUTPUT_DIR / img_name
    img.save(str(img_path))

    # Build netlist string
    netlist = "\n".join(netlist_lines)
    if label_texts:
        netlist += "\n" + " ".join(label_texts)

    # Build conversation format
    relative_img = f"data/synthetic_v2/{img_name}"
    conversation = {
        "images": [relative_img],
        "messages": [
            {"role": "user", "content": "<image>\nPlease output the netlist of the circuit schematic."},
            {"role": "assistant", "content": netlist},
        ],
    }

    return conversation

def main():
    COUNT = 500
    print(f"Generating {COUNT} synthetic schematics at DPI=150...")
    t0 = time.time()

    jsonl_path = DATASET_DIR / "ocr_vl_sft-synthetic-v2.jsonl"
    entries = []

    for i in range(COUNT):
        try:
            entry = generate_synthetic_schematic(i)
            entries.append(entry)
        except Exception as e:
            print(f"  [{i}] Error: {e}")
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{COUNT}] {elapsed:.0f}s elapsed")

    # Write JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    total_time = time.time() - t0
    print(f"\nDone! Generated {len(entries)} synthetic images in {total_time:.0f}s")
    print(f"Images: {OUTPUT_DIR}")
    print(f"JSONL: {jsonl_path}")
    print(f"File size: {jsonl_path.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()

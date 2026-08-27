"""A1: Expand training data to 3000+ samples."""
import os, json, random, re
from PIL import Image, ImageDraw, ImageFont

D = r'g:/mimo_project/circuit_ocr'
SRC_JSONL = os.path.join(D, 'output', 'train_v10fmt_synth.jsonl')
SYNTH_DIR = os.path.join(D, 'output', 'synth_text_images')
OUT_JSONL = os.path.join(D, 'output', 'train_3k.jsonl')
os.makedirs(SYNTH_DIR, exist_ok=True)

def log(m): print(f"[A1] {m}", flush=True)

# Load existing data
with open(SRC_JSONL, encoding='utf-8') as f:
    existing = [json.loads(l) for l in f if l.strip()]
log(f"Existing: {len(existing)} samples")

# Split existing
circuit_samples = [s for s in existing if 'synth_text_images' not in s['images'][0]]
synth_samples = [s for s in existing if 'synth_text_images' in s['images'][0]]
log(f"  Circuit: {len(circuit_samples)}, Synth: {len(synth_samples)}")

# ── 1. Expand synthetic text from 300 to 800 ──
font_path = None
for fp in ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf", "C:/Windows/Fonts/arial.ttf"]:
    if os.path.exists(fp): font_path = fp; break

new_templates = [
    # Mixed signal
    lambda: "Mixed Signal Section\n" + "\n".join([f"{random.choice(['ADC','DAC','OPAMP','COMP'])}{i}  {random.choice(['INA','OUT','REF','VREF','CLK'])}  {random.choice(['IN+','IN-','OUT','VDD','GND'])}" for i in range(1, random.randint(6, 14))]),
    # Power tree
    lambda: "Power Tree\n" + "\n".join([f"{v}  {random.choice(['1A','500mA','2A','100mA','3A'])}  {random.choice(['LDO','DC-DC','Buck','Boost'])}" for v in ['VIN 12V','VDD 5V','VDD 3.3V','VDD 1.8V','VDD 1.2V','VCORE','VIO','VANA']]),
    # Logic gate pins
    lambda: "Logic Gates\n" + "\n".join([f"U{i}  {random.choice(['74HC00','74HC04','74HC08','74HC32','74HC74','74HC86'])}  {random.choice(['NAND','NOT','AND','OR','FF','XOR'])}" for i in range(1, random.randint(5, 12))]),
    # Test point table
    lambda: "Test Points\n" + "\n".join([f"TP{i:02d}  {random.choice(['GND','VCC','CLK','RST','TX','RX','SDA','SCL'])}  {random.choice(['pad 1mm','via 0.5mm','hook','loop'])}  {random.choice(['bottom','top','both'])}" for i in range(1, random.randint(8, 20))]),
    # PCB silkscreen
    lambda: "PCB Silkscreen\n" + "\n".join([f"{random.choice(['PWR','RST','BOOT','PROG','DEBUG','USER','TX','RX'])}  ->  {random.choice(['J1','J2','J3','SW1','LED1','TP1'])}" for _ in range(random.randint(6, 16))]),
]

all_labels = [s['messages'][1]['content'] for s in circuit_samples]
new_synth = []
for idx in range(500):  # generate 500 new ones
    W, H = random.choice([(800,600),(1024,768),(1200,800),(800,1000),(1000,700)])
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    if random.random() < 0.5:
        text = random.choice(new_templates)()
    else:
        text = random.choice(all_labels)[:500]
    try:
        fs = random.choice([20,24,28,32])
        fn = ImageFont.truetype(font_path, fs)
    except:
        fn = ImageFont.load_default()
    y = random.randint(25,80)
    for line in text.split("\n"):
        if y > H-50: break
        draw.text((random.randint(30,60), y), line, fill="black", font=fn)
        y += fs + random.randint(4,12)
    img_path = os.path.join(SYNTH_DIR, f"s{idx+300:04d}.png")
    img.save(img_path, "PNG")
    entry = {"messages":[{"role":"user","content":"<image>OCR:"},{"role":"assistant","content":text}],"images":[img_path.replace("\\","/")]}
    new_synth.append(entry)
    if (idx+1)%100==0: log(f"  Synth {idx+1}/500")

log(f"New synth: {len(new_synth)}")

# ── 2. Augment existing circuits ──
augmented = []
for i, s in enumerate(circuit_samples[:400]):  # augment 400 samples
    try:
        img_path = s['images'][0]
        img = Image.open(img_path).convert("RGB")
        # Rotation ±5°
        angle = random.uniform(-5, 5)
        img_rot = img.rotate(angle, expand=False, fillcolor='white')
        # Brightness ±10%
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(img_rot)
        img_bright = enhancer.enhance(random.uniform(0.9, 1.1))
        # Save
        aug_path = os.path.join(SYNTH_DIR, f'aug_{i:04d}.png')
        img_bright.save(aug_path, "PNG")
        entry = {"messages": s['messages'], "images": [aug_path.replace("\\","/")]}
        augmented.append(entry)
    except Exception as e:
        pass
    if (i+1)%100==0: log(f"  Augment {i+1}/400")
log(f"Augmented: {len(augmented)}")

# ── 3. Cross-page split ──
split_samples = []
for i, s in enumerate(circuit_samples[:300]):
    label = s['messages'][1]['content']
    lines = label.split('\n')
    if len(lines) > 15:
        mid = len(lines) // 2
        # First half
        s1 = dict(s)
        s1['messages'] = [s['messages'][0], {'role':'assistant','content':'\n'.join(lines[:mid])}]
        split_samples.append(s1)
        # Second half
        s2 = dict(s)
        s2['messages'] = [s['messages'][0], {'role':'assistant','content':'\n'.join(lines[mid:])}]
        split_samples.append(s2)
    if (i+1)%100==0: log(f"  Split {i+1}/300")
log(f"Split: {len(split_samples)}")

# ── Combine ──
all_synth = synth_samples + new_synth  # 300 + 500 = 800
total = circuit_samples + all_synth + augmented + split_samples
random.shuffle(total)
log(f"Total: {len(total)} (circuit:{len(circuit_samples)} synth:{len(all_synth)} aug:{len(augmented)} split:{len(split_samples)})")

with open(OUT_JSONL, 'w', encoding='utf-8') as f:
    for e in total:
        f.write(json.dumps(e, ensure_ascii=False) + '\n')
log(f"Saved: {OUT_JSONL}")
log("A1 DONE")

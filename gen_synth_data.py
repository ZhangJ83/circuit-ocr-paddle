"""Generate 300 synthetic text images and mix with training data."""
import os, sys, json, time, random, re
from PIL import Image, ImageDraw, ImageFont

DATASET_DIR = r"g:/mimo_project/circuit_ocr"
SYNTH_DIR = os.path.join(DATASET_DIR, "output", "synth_text_images")
SYNTH_JSONL = os.path.join(DATASET_DIR, "output", "train_v10fmt_synth.jsonl")

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

os.makedirs(SYNTH_DIR, exist_ok=True)

# Find font
font_path = None
for fp in ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf",
           "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"]:
    if os.path.exists(fp): font_path = fp; break
log(f"Font: {font_path}")

# Extract labels from training data
train_path = os.path.join(DATASET_DIR, "output", "train_v10fmt.jsonl")
all_labels = []
with open(train_path, encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        d = json.loads(line)
        all_labels.append(d["messages"][1]["content"])
log(f"Loaded {len(all_labels)} real labels")

# Templates
templates = [
    lambda: "\n".join([f"R{i}  {random.choice(['10k','2.2k','4.7k','100','1k','47k','330','100k','1M'])}Ω  ±{random.choice(['1','5','10'])}%" for i in range(1, random.randint(8, 20))]),
    lambda: "\n".join([f"C{i}  {random.choice(['100nF','10μF','22μF','0.1μF','1μF','100μF','47μF','4.7μF'])}  {random.choice(['50V','25V','16V','10V','100V'])}  {random.choice(['Ceramic','Electrolytic','MLCC','Tantalum'])}" for i in range(1, random.randint(8, 16))]),
    lambda: "\n".join([f"U{i}  {random.choice(['ESP32','STM32F103','ATmega328P','CH340C','AMS1117','INA219','MPU6050','MAX485','BME280'])}" for i in range(1, random.randint(5, 12))]),
    lambda: "Pin Assignments\n" + "\n".join([f"  {i:2d}  {random.choice(['VCC','GND','TX','RX','SCL','SDA','GPIO','ADC','PWM','RESET','EN','INT']):10s}  {random.choice(['Input','Output','Bidirectional','Power','Analog'])}" for i in range(1, random.randint(8, 16))]),
    lambda: "Net Labels\n" + "\n".join([f"{random.choice(['VCC_5V','VDD_3V3','GND','VBUS','VSYS','VBAT','VIN','VOUT','AGND','PGND','+12V','-12V','+5V','+3.3V'])}" for _ in range(random.randint(8, 20))]),
    lambda: f"J{random.randint(1,4)}: {random.choice(['2.54mm Pin Header','JST-XH','FFC Cable','USB-C','RJ45'])}\n" + "\n".join([f"  {i:2d}  {random.choice(['VCC','GND','SDA','SCL','TX','RX','D+','D-','VBUS','CC1','CC2','SBU1','SBU2','MISO','MOSI','SCK','CS'])}" for i in range(1, random.randint(6, 20))]),
    lambda: "Ref    Value        Package    Qty\n" + "\n".join([f"{random.choice(['R','C','U','J','D','L','Q'])}{i:02d}  {random.choice(['10kΩ','100nF','ESP32','1N4148','SS34','10μH','2N7002']):12s}  {random.choice(['0805','SOT-23','QFN-32','SOD-123','TH','SMD']):10s}  {random.randint(1,10)}" for i in range(1, random.randint(5, 15))]),
]

entries = []
for idx in range(300):
    W, H_img = random.choice([(800, 600), (1024, 768), (1200, 800), (800, 1000), (1000, 700)])
    img = Image.new("RGB", (W, H_img), "white")
    draw = ImageDraw.Draw(img)

    if random.random() < 0.6:
        text = random.choice(templates)()
    else:
        text = random.choice(all_labels)
        if len(text) > 500: text = text[:500]

    try:
        font_size = random.choice([20, 24, 28, 32])
        fn = ImageFont.truetype(font_path, font_size)
    except:
        fn = ImageFont.load_default()

    y = random.randint(25, 80)
    for line in text.split("\n"):
        if y > H_img - 50: break
        x = random.randint(30, 60)
        draw.text((x, y), line, fill="black", font=fn)
        y += font_size + random.randint(4, 12)

    img_path = os.path.join(SYNTH_DIR, f"s{idx:04d}.png")
    img.save(img_path, "PNG")

    entry = {
        "messages": [
            {"role": "user", "content": "<image>OCR:"},
            {"role": "assistant", "content": text}
        ],
        "images": [img_path.replace("\\", "/")]
    }
    entries.append(entry)

    if (idx + 1) % 50 == 0:
        log(f"  {idx + 1}/300 images done")

log(f"Generated {len(entries)} synthetic entries")

# Mix with original
with open(train_path, encoding="utf-8") as f:
    orig_entries = [json.loads(l) for l in f if l.strip()]
mixed = orig_entries + entries
random.shuffle(mixed)

with open(SYNTH_JSONL, "w", encoding="utf-8") as f:
    for e in mixed:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

log(f"Saved: {SYNTH_JSONL} ({len(mixed)} entries = {len(orig_entries)} orig + {len(entries)} synth)")
log(f"Synthetic ratio: {len(entries)/len(mixed)*100:.1f}%")
log("DONE")

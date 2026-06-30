"""
Re-scrape .kicad_sch from GitHub. Render with PIL fallback + extract text from parser.
Parser and renderer share the same data source → GT 100% matches image.
20-minute time limit.
"""
import os, sys, json, time, requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_pipeline.kicad_parser import KiCadParser
from src.data_pipeline.renderer import SchematicRenderer

DATASET_DIR = Path(__file__).parent.parent
OUTPUT_DIR = DATASET_DIR / "data" / "rescraped"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PNG_DIR = OUTPUT_DIR / "png"
PNG_DIR.mkdir(parents=True, exist_ok=True)

MAX_TIME = 20 * 60
t0 = time.time()

def time_up():
    return time.time() - t0 > MAX_TIME

def log(msg):
    print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)

# ── Step 1: Search + Download .kicad_sch from GitHub ──
log("Searching GitHub...")

QUERIES = [
    "topic:kicad",
    "kicad schematic PCB",
    "kicad esp32",
    "kicad arduino",
    "kicad stm32",
]

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
headers = {"Accept": "application/vnd.github.v3+json"}
if GITHUB_TOKEN:
    headers["Authorization"] = f"token {GITHUB_TOKEN}"
found_repos = set()

for query in QUERIES:
    if time_up(): break
    try:
        url = f"https://api.github.com/search/repositories?q={query}&sort=updated&per_page=30"
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            for item in resp.json().get("items", []):
                found_repos.add(item["full_name"])
        elif resp.status_code == 403:
            log(f"  Rate limited, waiting...")
            time.sleep(30)
    except Exception as e:
        log(f"  Search error: {e}")

log(f"Found {len(found_repos)} repos")

# ── Step 2: Download .kicad_sch files ──
log("Downloading .kicad_sch files...")
downloaded = 0

for repo_name in list(found_repos):
    if time_up(): break
    try:
        for branch in ["main", "master"]:
            url = f"https://api.github.com/repos/{repo_name}/git/trees/{branch}?recursive=1"
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200: continue
            tree = resp.json()
            sch_paths = [item["path"] for item in tree.get("tree", [])
                         if item["path"].endswith(".kicad_sch") and item.get("size", 0) > 500]
            for sch_path in sch_paths[:2]:
                if time_up(): break
                try:
                    raw_url = f"https://raw.githubusercontent.com/{repo_name}/{branch}/{sch_path}"
                    resp = requests.get(raw_url, headers=headers, timeout=15)
                    if resp.status_code != 200: continue
                    safe_name = repo_name.replace("/", "_") + "_" + sch_path.replace("/", "_")
                    (OUTPUT_DIR / safe_name).write_bytes(resp.content)
                    downloaded += 1
                    if downloaded % 10 == 0:
                        log(f"  Downloaded {downloaded}...")
                except: continue
            break  # if we got the tree, no need to try other branch
    except: continue

log(f"Downloaded {downloaded} .kicad_sch files")

# ── Step 3: Parse + Render + Extract GT ──
log("Parsing, rendering, extracting GT...")

parser = KiCadParser()
renderer = SchematicRenderer(output_dir=str(PNG_DIR), dpi=150)

entries = []
parsed_count = 0
rendered_count = 0

for sch_file in OUTPUT_DIR.glob("*.kicad_sch"):
    if time_up(): break
    try:
        # Parse
        data = parser.parse(str(sch_file))
        if len(data.components) < 2:
            continue
        parsed_count += 1

        # Collect ALL text with positions from the parsed data
        texts = []  # (y, x, text)

        # Component references and values
        for comp in data.components:
            if comp.reference:
                texts.append((comp.position.y, comp.position.x, comp.reference))
            if comp.value:
                # Values are drawn below references
                texts.append((comp.position.y + 5, comp.position.x, comp.value))
            # Pin names for ICs
            for pin in comp.pins:
                if pin.absolute_pos and pin.pin_name and pin.pin_name != "~":
                    texts.append((pin.absolute_pos.y, pin.absolute_pos.x, pin.pin_name))

        # Net labels
        for label in data.labels:
            if label.name:
                texts.append((label.position.y, label.position.x, label.name))

        # Text annotations
        for text in data.texts:
            if text.text:
                texts.append((text.position.y, text.position.x, text.text))

        if len(texts) < 3:
            continue

        # Render to PNG using PIL fallback
        png_path = renderer._render_with_pil(sch_file, str(PNG_DIR / (sch_file.stem + ".png")))
        if not png_path:
            continue
        rendered_count += 1

        # Sort by y, then x
        texts.sort(key=lambda t: (t[0], t[1]))

        # Group lines (same y within 5px)
        lines = []
        current_line = []
        current_y = None
        for (y, x, txt) in texts:
            if current_y is None or abs(y - current_y) > 5:
                if current_line:
                    current_line.sort(key=lambda t: t[1])
                    lines.append(" ".join(t[2] for t in current_line))
                current_line = [(y, x, txt)]
                current_y = y
            else:
                current_line.append((y, x, txt))
        if current_line:
            current_line.sort(key=lambda t: t[1])
            lines.append(" ".join(t[2] for t in current_line))

        label = "\n".join(lines)
        rel_png = str(Path(png_path).relative_to(DATASET_DIR))

        entries.append({
            "images": [rel_png],
            "messages": [
                {"role": "user", "content": "<image>\nOCR:"},
                {"role": "assistant", "content": label},
            ],
        })

        if len(entries) % 25 == 0:
            log(f"  Processed {len(entries)} schematics...")

    except Exception as e:
        continue

# ── Save ──
jsonl_path = DATASET_DIR / "ocr_vl_sft-rescraped.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as f:
    for e in entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")

total = time.time() - t0
log(f"\n{'='*50}")
log(f"RESCRAPE DONE in {total:.0f}s")
log(f"  Downloaded: {downloaded} .kicad_sch")
log(f"  Parsed:     {parsed_count}")
log(f"  Rendered:   {rendered_count}")
log(f"  Dataset:    {len(entries)} samples")
log(f"  JSONL:      {jsonl_path}")
log(f"  Images:     {PNG_DIR}")

# Show sample
if entries:
    print(f"\nSample ({len(entries[0]['messages'][1]['content'])} chars):")
    print(entries[0]['messages'][1]['content'][:300])

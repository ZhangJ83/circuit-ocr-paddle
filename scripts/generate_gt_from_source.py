"""
Generate clean GT from bshada/open-schematics parquet data.
Rules:
1. Parse schematic_json (already has structured data)
2. Filter #PWRxx internal refs (keep only the Value for power symbols)
3. Filter hidden properties
4. Sort by visual y-coordinate, then x-coordinate
5. Output ref-value pairs + net labels in reading order
"""
import json, sys
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent
DATASET_DIR = PROJECT_DIR / "circuit-ocr-dataset"
PARQUET_DIR = DATASET_DIR / "data" / "open_schematics_v2" / "parquet"

def is_hidden(prop):
    """Check if a property has (hide yes) in its effects."""
    effects = prop.get("effects", {})
    if isinstance(effects, dict):
        return effects.get("hide") == "yes" or str(effects).find("(hide yes)") > -1
    if isinstance(effects, str):
        return "(hide yes)" in effects
    return False

def generate_gt(schematic_json_str):
    """Generate clean GT from schematic_json metadata."""
    metadata = json.loads(schematic_json_str) if isinstance(schematic_json_str, str) else schematic_json_str

    items = []  # (y, x, text_lines)

    symbols = metadata.get("schematicSymbols", [])
    for sym in symbols:
        lib = sym.get("libraryNickname", "")
        is_power = (lib == "power")

        props = sym.get("properties", [])
        ref = ""
        val = ""
        ref_pos = None
        val_pos = None

        for p in props:
            key = p.get("key", "")
            value = p.get("value", "").strip()
            pos = p.get("position", {})
            px = float(pos.get("x", 0))
            py = float(pos.get("y", 0))

            if is_hidden(p):
                continue

            if key == "Reference":
                ref = value
                ref_pos = (px, py)
            elif key == "Value":
                val = value
                val_pos = (px, py)

        if is_power:
            # Power symbol: only output the Value (e.g., VCC, GND, +3.3V)
            # The Reference (#PWRxx) is internal and invisible
            if val and val_pos:
                items.append((val_pos[1], val_pos[0], [val]))
        else:
            # Normal component: output Reference then Value
            if ref and ref_pos:
                lines = [ref]
                if val:
                    lines.append(val)
                else:
                    lines.append("")
                items.append((ref_pos[1], ref_pos[0], lines))

    # Global labels (net names on wires)
    for lbl in metadata.get("globalLabels", []):
        text = lbl.get("text", "").strip()
        pos = lbl.get("position", {})
        px = float(pos.get("x", 0))
        py = float(pos.get("y", 0))
        if text:
            items.append((py, px, [text]))

    # Sort: y first (top to bottom), then x (left to right)
    items.sort(key=lambda t: (round(t[0], 1), round(t[1], 1)))

    # Flatten to text
    all_lines = []
    for _, _, lines in items:
        all_lines.extend(lines)

    return "\n".join(all_lines)


if __name__ == "__main__":
    # Process first parquet file, first sample
    df = pd.read_parquet(PARQUET_DIR / "train-00000.parquet")
    row = df.iloc[0]

    name = str(row.get("name", "unknown"))
    gj = row.get("schematic_json")

    print(f"Sample: {name}")
    print(f"schematic_json type: {type(gj)}")

    gt = generate_gt(gj)

    # Save to file
    out_path = PROJECT_DIR / "output" / "gt_sample_001.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(gt)
    print(f"\nSaved: {out_path} ({len(gt.splitlines())} lines)")

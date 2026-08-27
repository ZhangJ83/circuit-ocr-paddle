#!/usr/bin/env python3
import json
from pathlib import Path

def analyze_dataset():
    splits = ["train", "val", "test"]
    base_dir = Path(".")

    # Stats container
    stats = {
        "train": {"total": 0, "sources": {"Synthetic": 0, "Masala-CHAI": 0, "Open Schematics": 0}, "chars": []},
        "val": {"total": 0, "sources": {"Synthetic": 0, "Masala-CHAI": 0, "Open Schematics": 0}, "chars": []},
        "test": {"total": 0, "sources": {"Synthetic": 0, "Masala-CHAI": 0, "Open Schematics": 0}, "chars": []}
    }

    # Helper to detect source
    def get_source(img_path):
        name = Path(img_path).name.lower()
        if any(p in name for p in ["mixed", "analog", "digital", "power", "synth"]):
            return "Synthetic"
        # Masala-Chai typically has pure numbers or jssc_ prefix
        # e.g., 117.jpg, 225.jpg, jssc_12.jpg
        name_no_ext = Path(img_path).stem.lower()
        if name_no_ext.isdigit() or name_no_ext.startswith("jssc_"):
            return "Masala-CHAI"
        return "Open Schematics"

    for split in splits:
        jsonl_path = base_dir / f"ocr_vl_sft-{split}.jsonl"
        if not jsonl_path.exists():
            continue

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                stats[split]["total"] += 1
                
                # Analyze image source
                img_path = data["images"][0]
                source = get_source(img_path)
                stats[split]["sources"][source] += 1
                
                # Analyze text length
                assistant_content = data["messages"][1]["content"]
                stats[split]["chars"].append(len(assistant_content))

    # Print markdown table
    print("| Split | Source Dataset | Count | Avg Chars | Min Chars | Max Chars | Total Chars |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    total_samples = 0
    total_chars_all = 0
    source_totals = {"Synthetic": 0, "Masala-CHAI": 0, "Open Schematics": 0}

    for split in splits:
        s = stats[split]
        if s["total"] == 0:
            continue
        
        split_chars = s["chars"]
        avg_char = sum(split_chars) / len(split_chars) if split_chars else 0
        min_char = min(split_chars) if split_chars else 0
        max_char = max(split_chars) if split_chars else 0
        sum_char = sum(split_chars) if split_chars else 0
        
        total_samples += s["total"]
        total_chars_all += sum_char
        
        for k in source_totals:
            source_totals[k] += s["sources"][k]

        print(f"| **{split.capitalize()}** | All (Combined) | {s['total']} | {avg_char:.1f} | {min_char} | {max_char} | {sum_char:,} |")
        for k, count in s["sources"].items():
            print(f"| | └─ {k} | {count} | - | - | - | - |")

    print(f"| **Total** | **All Combined** | **{total_samples}** | **{total_chars_all/total_samples:.1f}** | - | - | **{total_chars_all:,}** |")
    print("\nSource Breakdown:")
    for k, count in source_totals.items():
        print(f"- {k}: {count} ({count/total_samples*100:.1f}%)")

if __name__ == "__main__":
    analyze_dataset()

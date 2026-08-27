#!/usr/bin/env python3
"""Parallel PaddleOCR-VL benchmark runner for WSL CPU mode.
Spawns N concurrent subprocesses, each processing one sample.
"""
import subprocess, json, sys, time, os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT = Path(__file__).parent / "eval_benchmark_wsl.py"
DATA_DIR = SCRIPT.parent.parent  # circuit-ocr-dataset
PYTHON = "/usr/bin/python3"

TIER = sys.argv[1] if len(sys.argv) > 1 else "easy50"
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4  # CPU cores to use

# Map tier names
TIER_MAP = {
    "easy50": "ocr_vl_sft-test-easy50-jpeg.jsonl",
    "easy100": "ocr_vl_sft-test-easy100-jpeg.jsonl",
    "easy200": "ocr_vl_sft-test-easy200-jpeg.jsonl",
    "full523": "ocr_vl_sft-test-jpeg.jsonl",
}
DATA_FILE = TIER_MAP.get(TIER, f"ocr_vl_sft-test-{TIER}-jpeg.jsonl")
DATA = DATA_DIR / DATA_FILE
OUTPUT = DATA_DIR / f"results_paddleocr-vl_{TIER}.jsonl"

# Load all samples
samples = []
with open(DATA, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

TOTAL = len(samples)
print(f"=== PaddleOCR-VL {TIER}: {TOTAL} samples, {WORKERS} workers ===")

# Remove old output
if OUTPUT.exists():
    OUTPUT.unlink()

def process_sample(idx, sample):
    """Process one sample: write temp JSONL, run eval_benchmark_wsl.py, return result."""
    tmp_data = DATA_DIR / f"_tmp_sample_{idx}.jsonl"
    tmp_out = DATA_DIR / f"_tmp_result_{idx}.jsonl"

    try:
        with open(tmp_data, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        t0 = time.time()
        result = subprocess.run(
            [PYTHON, str(SCRIPT),
             "--data_path", str(tmp_data.name),
             "--output_path", str(tmp_out.name),
             "--max_length", "200"],
            cwd=str(DATA_DIR),
            capture_output=True, text=True,
            timeout=600,
        )
        elapsed = time.time() - t0

        if tmp_out.exists():
            with open(tmp_out, "r", encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
            if lines:
                d = json.loads(lines[0])
                pred = d.get("prediction", "")
                img_name = Path(sample["images"][0]).name
                return (idx, True, elapsed, pred, img_name, d)
        return (idx, False, elapsed, "", "", None)

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return (idx, False, elapsed, "TIMEOUT", "", None)
    except Exception as e:
        return (idx, False, 0, str(e), "", None)
    finally:
        for f in [tmp_data, tmp_out]:
            if f.exists():
                f.unlink()


start_all = time.time()
ok = 0
fail = 0

# Use ThreadPoolExecutor since subprocesses are already in separate processes
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(process_sample, i, s): i for i, s in enumerate(samples)}

    for future in as_completed(futures):
        idx, success, elapsed, msg, img_name, result = future.result()
        if success:
            ok += 1
            # Write to persistent output
            with open(OUTPUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"[{ok+fail}/{TOTAL}] OK {img_name} {elapsed:.0f}s pred_len={len(msg)}")
        else:
            fail += 1
            print(f"[{ok+fail}/{TOTAL}] FAIL idx={idx} {elapsed:.0f}s: {msg}")

total_time = time.time() - start_all
print(f"\n=== {TIER} Done: {ok} OK, {fail} failed in {total_time:.0f}s ===")

# Compute Avg NED
if OUTPUT.exists() and ok > 0:
    from Levenshtein import distance
    predictions = []
    references = []
    with open(OUTPUT, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                predictions.append(d.get("prediction", ""))
                references.append(d.get("label", d["messages"][1]["content"]))
    if predictions:
        total_ned = sum(distance(p, r) / max(len(p), len(r), 1) for p, r in zip(predictions, references))
        avg_ned = total_ned / len(predictions)
        print(f"Avg. NED: {avg_ned:.4f} (on {len(predictions)} samples)")
    print(f"Results: {OUTPUT}")

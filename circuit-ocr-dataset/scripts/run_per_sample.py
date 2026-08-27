#!/usr/bin/env python3
"""One-process-per-sample PaddleOCR-VL benchmark runner.
Each sample gets a fresh process to avoid Paddle GPU crash (exit 127).
Uses --resume to track progress across crashes.
"""
import subprocess, sys, os, json, time
from pathlib import Path

PYTHON = r"E:\080000software\080900_Miniconda\miniconda3\Library\envs\gpu-pytorch\python.exe"
SCRIPT = Path(__file__).parent / "eval_benchmark.py"
DATA_DIR = SCRIPT.parent.parent  # circuit-ocr-dataset

TIER = sys.argv[1] if len(sys.argv) > 1 else "easy50"
DATA = DATA_DIR / f"ocr_vl_sft-test-{TIER}.jsonl"
OUTPUT = DATA_DIR / f"results_paddleocr-vl_{TIER}.jsonl"

# Load all samples
samples = []
with open(DATA, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

TOTAL = len(samples)
print(f"=== PaddleOCR-VL {TIER}: {TOTAL} samples ===")

# Check already processed
already = set()
if OUTPUT.exists():
    with open(OUTPUT, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                if "images" in d:
                    already.add(tuple(d["images"]))
print(f"Already processed: {len(already)}")

success = 0
failed = 0
start_time = time.time()

for i, sample in enumerate(samples):
    img_key = tuple(sample["images"])
    if img_key in already:
        continue

    idx = i + 1
    img_name = Path(sample["images"][0]).name
    sample_start = time.time()
    print(f"[{idx}/{TOTAL}] {img_name}...", end=" ", flush=True)

    cmd = [
        PYTHON, str(SCRIPT),
        "--model_type", "paddleocr-vl",
        "--model_name_or_path", "PaddlePaddle/PaddleOCR-VL",
        "--data_path", str(DATA),
        "--output_path", str(OUTPUT),
        "--max_length", "1024",
        "--resume",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(DATA_DIR),
            capture_output=True,
            text=True,
            timeout=180,  # 3 min timeout per sample
        )
        elapsed = time.time() - sample_start

        # Check if a new result was written
        new_already = set()
        if OUTPUT.exists():
            with open(OUTPUT, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        if "images" in d:
                            new_already.add(tuple(d["images"]))

        new_count = len(new_already)
        if img_key in new_already:
            success += 1
            already.add(img_key)
            # Find the prediction
            with open(OUTPUT, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        if tuple(d["images"]) == img_key:
                            pred = d.get("prediction", "")
                            print(f"OK {elapsed:.1f}s pred_len={len(pred)} [{success+len([a for a in already if a not in new_already])}/{TOTAL}]")
                            break
        else:
            failed += 1
            print(f"FAIL {elapsed:.1f}s rc={result.returncode} (no output)")
            # Show any error output
            stderr = result.stderr.strip()
            if stderr:
                print(f"  stderr: {stderr[:200]}")

    except subprocess.TimeoutExpired:
        elapsed = time.time() - sample_start
        failed += 1
        print(f"TIMEOUT {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - sample_start
        failed += 1
        print(f"ERROR {elapsed:.1f}s: {e}")

    # Brief pause
    time.sleep(0.5)

total_time = time.time() - start_time
processed = len(already)
print(f"\n=== Done: {processed}/{TOTAL} processed, {failed} failed, {total_time:.0f}s total ===")
print(f"Results: {OUTPUT}")

#!/usr/bin/env python3
"""Run PaddleOCR-VL benchmark: one fresh process per sample.
Each sample gets its own Python process to avoid Paddle 2.6.2 Windows
KV-cache stack overflow bug (STATUS_STACK_BUFFER_OVERRUN).

Usage: python run_benchmark.py <tier>
  tier: easy50, easy100, easy200, full523 (default: easy50)
"""
import subprocess, json, sys, time
from pathlib import Path

PYTHON = r"E:\080000software\080900_Miniconda\miniconda3\Library\envs\gpu-pytorch\python.exe"
SCRIPT = Path(__file__).parent / "eval_benchmark.py"
DATA_DIR = SCRIPT.parent.parent

TIER = sys.argv[1] if len(sys.argv) > 1 else "easy50"
# Map tier names to data files
TIER_MAP = {
    "easy50": "ocr_vl_sft-test-easy50-jpeg.jsonl",
    "easy100": "ocr_vl_sft-test-easy100-jpeg.jsonl",
    "easy200": "ocr_vl_sft-test-easy200-jpeg.jsonl",
    "full523": "ocr_vl_sft-test-jpeg.jsonl",
}
DATA_FILE = TIER_MAP.get(TIER, f"ocr_vl_sft-test-{TIER}-jpeg.jsonl")
DATA = DATA_DIR / DATA_FILE
OUTPUT = DATA_DIR / f"results_paddleocr-vl_{TIER}.jsonl"
TMP_DATA = DATA_DIR / "_tmp_one_sample.jsonl"

# Load all samples
samples = []
with open(DATA, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))

TOTAL = len(samples)
print(f"=== PaddleOCR-VL {TIER}: {TOTAL} samples (JPEG) ===")
print(f"Data: {DATA}")
print(f"Output: {OUTPUT}")

# Remove old output
if OUTPUT.exists():
    OUTPUT.unlink()

ok = 0
fail = 0
start_all = time.time()

for i, sample in enumerate(samples):
    img_name = Path(sample["images"][0]).name
    t0 = time.time()

    # Write single sample to temp file
    with open(TMP_DATA, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Remove output so each run is independent
    if OUTPUT.exists():
        OUTPUT.unlink()

    print(f"[{i+1}/{TOTAL}] {img_name}...", end=" ", flush=True)

    try:
        proc = subprocess.run(
            [PYTHON, str(SCRIPT),
             "--model_type", "paddleocr-vl",
             "--model_name_or_path", "PaddlePaddle/PaddleOCR-VL",
             "--data_path", str(TMP_DATA),
             "--output_path", str(OUTPUT),
             "--max_length", "1024"],
            cwd=str(DATA_DIR),
            capture_output=True, text=True,
            timeout=300,
        )
        elapsed = time.time() - t0

        if OUTPUT.exists():
            lines = open(OUTPUT, "r", encoding="utf-8").readlines()
            if len(lines) > 0:
                d = json.loads(lines[0])
                pred = d.get("prediction", "")
                ok += 1
                print(f"OK {elapsed:.1f}s pred_len={len(pred)}")
                # Append to persistent results
                with open(str(OUTPUT).replace(".jsonl", "_all.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
                continue

        fail += 1
        rc = proc.returncode
        # Map common exit codes
        rc_map = {3221226505: "STACK_OVERFLOW", 3221225477: "ACCESS_VIOLATION"}
        rc_str = rc_map.get(rc, str(rc))
        print(f"FAIL rc={rc}({rc_str}) {elapsed:.1f}s")

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        fail += 1
        print(f"TIMEOUT {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t0
        fail += 1
        print(f"ERROR {elapsed:.1f}s: {e}")

    time.sleep(0.5)

total_time = time.time() - start_all
processed = ok + fail
print(f"\n=== {TIER} Done: {ok} OK, {fail} failed of {TOTAL} in {total_time:.0f}s ===")

# Compute Avg NED if we have results
results_file = Path(str(OUTPUT).replace(".jsonl", "_all.jsonl"))
if results_file.exists() and ok > 0:
    from Levenshtein import distance
    predictions = []
    references = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                predictions.append(d.get("prediction", ""))
                references.append(d.get("label", d["messages"][1]["content"]))
    if predictions:
        total_ned = sum(distance(p, r) / max(len(p), len(r), 1) for p, r in zip(predictions, references))
        avg_ned = total_ned / len(predictions)
        print(f"Avg. NED: {avg_ned:.4f} (on {len(predictions)} samples)")
    print(f"Results: {results_file}")

# Clean up
if TMP_DATA.exists():
    TMP_DATA.unlink()

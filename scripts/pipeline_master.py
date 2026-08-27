"""Master pipeline: render → mix → train → eval. All automated."""
import os, sys, json, time, subprocess

PROJECT = r"g:/mimo_project/circuit_ocr"
SYNTH_DIR = os.path.join(PROJECT, "output", "synthetic_kicad_v3")
REAL_DATA = os.path.join(PROJECT, "output", "train_v10fmt_synth.jsonl")
MIXED_DATA = os.path.join(PROJECT, "output", "train_synthetic_mix.jsonl")
TEST_DATA = os.path.join(PROJECT, "output", "test_clean.jsonl")
VAL_DATA = os.path.join(PROJECT, "output", "val_clean.jsonl")
TRAIN_SCRIPT = os.path.join(PROJECT, "scripts", "train_synthetic_mix.py")
RENDER_SCRIPT = os.path.join(PROJECT, "scripts", "render_kicad_batch.py")
EVAL_SCRIPT = os.path.join(PROJECT, "circuit-ocr-dataset", "scripts", "eval_benchmark_v3.py")

def log(msg):
    print(f"[PIPE-{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def step_render():
    """Step 1: Render .kicad_sch → SVG → PNG."""
    log("Step 1: Rendering schematics...")
    # Check if already done
    if check_rendered():
        log("  Already rendered, skipping")
        return True

    cmd = [
        "python", RENDER_SCRIPT,
        "--sch_dir", os.path.join(SYNTH_DIR, "schematics"),
        "--out_dir", SYNTH_DIR,
        "--jsonl", os.path.join(SYNTH_DIR, "train_synthetic_kicad.jsonl"),
    ]
    result = subprocess.run(cmd, cwd=PROJECT)
    return result.returncode == 0

def check_rendered(min_pct=0.95):
    """Check if enough PNGs have been rendered."""
    img_dir = os.path.join(SYNTH_DIR, "images")
    sch_dir = os.path.join(SYNTH_DIR, "schematics")
    if not os.path.exists(img_dir):
        return False
    n_png = len([f for f in os.listdir(img_dir) if f.endswith('.png')])
    n_sch = len([f for f in os.listdir(sch_dir) if f.endswith('.kicad_sch')])
    if n_sch == 0:
        return False
    pct = n_png / n_sch
    log(f"  Rendered: {n_png}/{n_sch} ({pct:.0%})")
    return pct >= min_pct

def step_mix():
    """Step 2: Mix synthetic + real + synth-text data."""
    log("Step 2: Mixing data...")
    if os.path.exists(MIXED_DATA):
        with open(MIXED_DATA, encoding='utf-8') as f:
            n = sum(1 for _ in f)
        log(f"  Mixed data already exists ({n} samples), skipping")
        return True

    # Load synthetic KiCad data
    synth_jl = os.path.join(SYNTH_DIR, "train_synthetic_kicad.jsonl")
    with open(synth_jl, encoding='utf-8') as f:
        synth_entries = []
        for line in f:
            e = json.loads(line)
            # Only include if PNG exists
            img_path = e['images'][0]
            if os.path.exists(img_path):
                synth_entries.append(e)

    log(f"  Synthetic (with PNG): {len(synth_entries)}")

    # Load real data
    with open(REAL_DATA, encoding='utf-8') as f:
        real_all = [json.loads(l) for l in f if l.strip()]

    real_circuits = [s for s in real_all if 'synth_text_images' not in s['images'][0]]
    synth_text = [s for s in real_all if 'synth_text_images' in s['images'][0]]
    log(f"  Real circuits: {len(real_circuits)}, Synth text: {len(synth_text)}")

    # Mix: all circuits + anti-collapse synth text
    all_circuits = synth_entries + real_circuits
    import random; random.shuffle(all_circuits)

    # Ensure ~5% synth text ratio
    target_ratio = 0.05
    needed_text = int(len(all_circuits) * target_ratio / (1 - target_ratio))
    if len(synth_text) < needed_text:
        extra = needed_text - len(synth_text)
        log(f"  Duplicating {extra} synth text entries")
        while len(synth_text) < needed_text:
            for s in synth_text[:extra]:
                synth_text.append(json.loads(json.dumps(s)))
                if len(synth_text) >= needed_text:
                    break

    all_data = all_circuits + synth_text[:needed_text]
    random.shuffle(all_data)

    with open(MIXED_DATA, 'w', encoding='utf-8') as f:
        for e in all_data:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')

    log(f"  Mixed: {len(all_data)} samples ({len(all_circuits)} circuits + {min(len(synth_text), needed_text)} synth text)")
    return True

def step_train():
    """Step 3: Train on mixed data."""
    log("Step 3: Training...")
    cmd = [
        "python", TRAIN_SCRIPT,
        "--train_data", MIXED_DATA,
        "--val_data", VAL_DATA,
        "--test_data", TEST_DATA,
        "--epochs", "2",
        "--lr", "2e-5",
        "--output_dir", os.path.join(PROJECT, "checkpoints", "synth_kicad_mix"),
        "--n_val", "10",
        "--n_test", "30",
    ]
    result = subprocess.run(cmd, cwd=PROJECT)
    return result.returncode == 0

def main():
    log("=== MASTER PIPELINE START ===")
    t0 = time.time()

    # Step 1: Render (skip if done)
    if not check_rendered():
        step_render()

    # Step 2: Mix data
    step_mix()

    # Step 3: Train
    step_train()

    tt = (time.time() - t0) / 60
    log(f"=== PIPELINE DONE in {tt:.0f}min ===")


if __name__ == '__main__':
    main()

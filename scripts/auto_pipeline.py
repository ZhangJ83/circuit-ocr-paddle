"""Autonomous 6-hour pipeline — file-based output, no pipe buffering."""
import os, sys, json, time, subprocess, glob
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(PROJECT_DIR, "scripts", "train.py")
PYTHON = "/root/miniconda3/bin/python"
STATE_FILE = os.path.join(PROJECT_DIR, "pipeline_state.json")
LOG_FILE = os.path.join(PROJECT_DIR, "pipeline_log.txt")

PLAN = [
    {"name": "baseline",       "max_dim": 384, "rank": 16, "alpha": 32, "epochs": 3, "lr": 2e-5},
    {"name": "hires",          "max_dim": 512, "rank": 16, "alpha": 32, "epochs": 3, "lr": 2e-5},
    {"name": "large_rank",     "max_dim": 384, "rank": 32, "alpha": 64, "epochs": 3, "lr": 2e-5},
    {"name": "hires_large",    "max_dim": 512, "rank": 32, "alpha": 64, "epochs": 3, "lr": 2e-5},
    {"name": "proj_ablation",  "max_dim": 384, "rank": 16, "alpha": 32, "epochs": 3, "lr": 2e-5, "freeze_projector": 0},
]

USABLE_JF1 = 0.05
TIME_LIMIT = 360 * 60  # seconds


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def kill_zombies():
    """Kill any leftover train.py processes from previous crashes."""
    os.system('pkill -9 -f "train.py" 2>/dev/null; sleep 1')


def parse_metrics_from_log(log_path):
    """Parse a train log for BEST lines and return (best_jf1, best_ned)."""
    best_jf1, best_ned = -1.0, 1.0
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if 'BEST:' in line and 'joint_f1=' in line:
                    parts = line.split()
                    for p in parts:
                        if p.startswith('joint_f1='):
                            try: best_jf1 = float(p.split('=')[1])
                            except: pass
                        if p.startswith('NED='):
                            try: best_ned = float(p.split('=')[1])
                            except: pass
    except FileNotFoundError:
        pass
    return best_jf1, best_ned


def run_experiment(cfg):
    """Run one experiment. Returns (best_jf1, best_ned, ckpt_dir) or (None, None, None)."""
    log(f"=== {cfg['name']} === dim={cfg['max_dim']} r={cfg['rank']} epochs={cfg['epochs']}")

    train_log = os.path.join(PROJECT_DIR, f"train_{cfg['name']}.log")
    cmd = [PYTHON, "-u", TRAIN_SCRIPT,
           "--name", cfg["name"],
           "--max_dim", str(cfg["max_dim"]),
           "--rank", str(cfg["rank"]),
           "--alpha", str(cfg["alpha"]),
           "--epochs", str(cfg["epochs"]),
           "--lr", str(cfg["lr"]),
           "--freeze_projector", str(cfg.get("freeze_projector", 1)),
           "--output_dir", os.path.join(PROJECT_DIR, "checkpoints"),
           "--checkpoint_steps", str(cfg.get("checkpoint_steps", 400))]
    log(f"  CMD: {' '.join(cmd)}")
    log(f"  LOG: {train_log}")

    kill_zombies()

    try:
        # KEY FIX: redirect to FILE not pipe — no buffering deadlock
        with open(train_log, 'w') as log_f:
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            log(f"  PID={proc.pid}")

            # Monitor log file in real time
            last_size = 0
            deadline = time.time() + 7500  # 2h5min timeout
            last_output_time = time.time()

            while proc.poll() is None:
                time.sleep(5)
                if time.time() > deadline:
                    log(f"  TIMEOUT — killing PID={proc.pid}")
                    proc.kill()
                    proc.wait()
                    return None, None, None

                # Check for new output
                try:
                    cur_size = os.path.getsize(train_log)
                    if cur_size > last_size:
                        # New output appeared — read and relay last few lines
                        with open(train_log, 'r') as f:
                            f.seek(last_size)
                            new_lines = f.read()
                            # Print last meaningful line
                            for line in new_lines.strip().split('\n'):
                                line = line.strip()
                                if line and ('loss=' in line or 'BEST' in line or 'Done' in line
                                             or 'Checkpoint' in line or 'jf1=' in line
                                             or 'COLLAPSE' in line or 'Epoch' in line):
                                    log(f"  [{cfg['name']}] {line}")
                        last_size = cur_size
                        last_output_time = time.time()
                    elif time.time() - last_output_time > 1800:
                        # 30 minutes with NO output — stuck
                        log(f"  STUCK (30min no output) — killing PID={proc.pid}")
                        proc.kill()
                        proc.wait()
                        return None, None, None
                except Exception:
                    pass

            ret = proc.wait()
            log(f"  Exit={ret}")

        # Parse output for best metrics
        best_jf1, best_ned = parse_metrics_from_log(train_log)

        ckpt_dir = os.path.join(PROJECT_DIR, "checkpoints", cfg["name"])
        if ret != 0:
            log(f"  CRASH (exit={ret})")
            kill_zombies()
            if best_jf1 < 0:
                return None, None, None

        if best_jf1 < 0:
            log(f"  Done: no BEST line found — CRASHED")
            return None, None, None

        log(f"  Done: joint_f1={best_jf1:.4f} NED={best_ned:.4f}")
        return best_jf1, best_ned, ckpt_dir

    except Exception as e:
        log(f"  FAIL: {e}")
        kill_zombies()
        return None, None, None


def main():
    log("=== PIPELINE START ===")
    t0 = time.time()

    # Load or init state (for resume)
    state = {"experiments": {}, "best_name": None, "best_jf1": -1.0, "best_ned": 1.0, "best_ckpt": None}
    if os.path.exists(STATE_FILE):
        try:
            state = json.load(open(STATE_FILE))
            log(f"Resuming pipeline — {len(state.get('experiments', {}))} experiments done")
        except:
            pass

    # Phase 1: quick sweep (3 epochs each)
    log("Phase 1: Quick sweep")
    for cfg in PLAN:
        if cfg["name"] in state.get("experiments", {}):
            prev = state["experiments"][cfg["name"]]
            if prev.get("joint_f1") is not None and prev["joint_f1"] >= 0:
                log(f"  SKIP {cfg['name']} (already done: jf1={prev['joint_f1']})")
                continue

        remaining = TIME_LIMIT - (time.time() - t0)
        if remaining < 1200:
            log(f"  Only {remaining/60:.0f}min left, skipping rest")
            break

        jf1, ned, ckpt = run_experiment(cfg)
        state["experiments"][cfg["name"]] = {"joint_f1": jf1, "ned": ned, "ckpt": ckpt}
        if jf1 is not None and jf1 > state["best_jf1"]:
            state["best_name"] = cfg["name"]
            state["best_jf1"] = jf1
            state["best_ned"] = ned
            state["best_ckpt"] = ckpt
            log(f"  *** NEW BEST: {cfg['name']} joint_f1={jf1:.4f} ***")
        # Save state after each experiment
        json.dump(state, open(STATE_FILE, 'w'), ensure_ascii=False, indent=2)

    log(f"Phase 1 done. Best: {state['best_name']} joint_f1={state['best_jf1']:.4f}")

    # Phase 2: deep train best config
    remaining = TIME_LIMIT - (time.time() - t0)
    if remaining > 1800:
        if state["best_jf1"] > USABLE_JF1:
            deep_name = state["best_name"]
            deep_cfg = next((c for c in PLAN if c["name"] == deep_name), PLAN[0])
        else:
            log("No usable model yet, trying hires_large deep")
            deep_cfg = PLAN[3]  # hires_large

        deep_cfg = {**deep_cfg, "epochs": 10, "name": f"deep_{deep_cfg['name']}"}
        if deep_cfg["name"] not in state.get("experiments", {}):
            jf1, ned, ckpt = run_experiment(deep_cfg)
            state["experiments"][deep_cfg["name"]] = {"joint_f1": jf1, "ned": ned, "ckpt": ckpt}
            if jf1 is not None and jf1 > state["best_jf1"]:
                state["best_name"] = deep_cfg["name"]
                state["best_jf1"] = jf1
                state["best_ned"] = ned
                state["best_ckpt"] = ckpt
            json.dump(state, open(STATE_FILE, 'w'), ensure_ascii=False, indent=2)

    # Phase 3: final push
    remaining = TIME_LIMIT - (time.time() - t0)
    if remaining > 3600:
        log(f"Phase 3: Final push ({remaining/60:.0f}min left)")
        final_cfg = {
            "name": "final",
            "max_dim": 512,
            "rank": 32, "alpha": 64,
            "epochs": min(20, max(10, int(remaining / 180))),
            "lr": 1e-5,
        }
        if "final" not in state.get("experiments", {}):
            jf1, ned, ckpt = run_experiment(final_cfg)
            state["experiments"]["final"] = {"joint_f1": jf1, "ned": ned, "ckpt": ckpt}
            if jf1 is not None and jf1 > state["best_jf1"]:
                state["best_name"] = "final"
                state["best_jf1"] = jf1
                state["best_ckpt"] = ckpt
            json.dump(state, open(STATE_FILE, 'w'), ensure_ascii=False, indent=2)

    # Report
    total = (time.time() - t0) / 60
    log(f"=== PIPELINE DONE: {total:.0f}min ===")
    log(f"Best: {state['best_name']} joint_f1={state['best_jf1']:.4f} NED={state['best_ned']:.4f}")
    log(f"Checkpoint: {state['best_ckpt']}")
    json.dump(state, open(STATE_FILE, 'w'), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

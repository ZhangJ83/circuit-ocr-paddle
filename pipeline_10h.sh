#!/bin/bash
# 10-Hour Pipeline: A2(5微调) -> B(20 RL) -> C(SPICE)
# Survives failures, logs everything, never stops

PYTHON="E:/080000software/080900_Miniconda/miniconda3/envs/pyqpanda-quantum/python.exe"
DIR="g:/mimo_project/circuit_ocr"
LOGDIR="$DIR/experiment_logs"
mkdir -p "$LOGDIR"

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOGDIR/pipeline_10h.log"; }

# ═══════════════════════════════════════════
# PHASE A2: 5 micro-tune variants
# ═══════════════════════════════════════════
log "=== PHASE A2: 5 Fine-Tuning Variants ==="

run_exp() {
    local name=$1 lr=$2 alpha=$3 epochs=$4 warmup=$5
    log "START $name (lr=$lr alpha=$alpha epochs=$epochs warmup=$warmup)"
    $PYTHON -u "$DIR/circuit-ocr-dataset/scripts/train_one.py" \
        --name "$name" --lr "$lr" --alpha "$alpha" --epochs "$epochs" --warmup "$warmup" \
        --data "$DIR/output/train_3k.jsonl" --output "$DIR/checkpoints/$name" \
        > "$LOGDIR/${name}.log" 2>&1
    local rc=$?
    log "DONE $name (exit=$rc)"
    return $rc
}

run_exp "exp10a_baseline" 2e-5 32 3 100
run_exp "exp10b_highLR"   3e-5 32 3 100
run_exp "exp10c_bigAlpha" 2e-5 64 3 100
run_exp "exp10d_4epochs"  2e-5 32 4 100
run_exp "exp10e_longWarm" 2e-5 32 3 200

log "=== A2 DONE: Evaluating best ==="

# Find best checkpoint
$PYTHON -u "$DIR/eval_best.py" > "$LOGDIR/a2_best.log" 2>&1
BEST_CKPT=$(grep "BEST:" "$LOGDIR/a2_best.log" | tail -1 | awk '{print $2}')
BEST_JF1=$(grep "BEST:" "$LOGDIR/a2_best.log" | tail -1 | awk '{print $4}')
log "A2 Best: $BEST_CKPT (JointF1=$BEST_JF1)"

# ═══════════════════════════════════════════
# PHASE B: 20 RL Variants
# ═══════════════════════════════════════════
log "=== PHASE B: 20 RL Post-Training Variants ==="

RL_DIR="$DIR/checkpoints_rl"
mkdir -p "$RL_DIR"

for i in $(seq -w 1 20); do
    TEMP="0.8"
    WEIGHT="standard"
    EXTRA=""
    case $i in
        01) TEMP="0.8"; WEIGHT="standard" ;;
        02) TEMP="1.0"; WEIGHT="standard" ;;
        03) TEMP="0.6"; WEIGHT="standard" ;;
        04) TEMP="0.8"; WEIGHT="cf1_high" ;;
        05) TEMP="0.8"; WEIGHT="jf1_high" ;;
        06) TEMP="0.8"; WEIGHT="diversity" ;;
        07) TEMP="0.8"; WEIGHT="anticollapse" ;;
        08) TEMP="0.8"; WEIGHT="best4" ;;
        09) TEMP="0.8"; WEIGHT="best8" ;;
        10) TEMP="0.8"; WEIGHT="contrastive" ;;
        11) TEMP="0.8"; WEIGHT="iterative" ;;
        12) TEMP="1.0"; WEIGHT="iterative_balanced" ;;
        13) TEMP="0.8"; WEIGHT="curriculum" ;;
        14) TEMP="1.0"; WEIGHT="curriculum_strong" ;;
        15) TEMP="0.8"; WEIGHT="mixed50" ;;
        16) TEMP="0.8"; WEIGHT="mixed70" ;;
        17) TEMP="0.8"; WEIGHT="len_bonus" ;;
        18) TEMP="0.8"; WEIGHT="anticollapse_strong" ;;
        19) TEMP="0.8"; WEIGHT="perline" ;;
        20) TEMP="0.8"; WEIGHT="ensemble3" ;;
    esac
    log "START RL-$i ($WEIGHT temp=$TEMP)"
    $PYTHON -u "$DIR/circuit-ocr-dataset/scripts/rl_train.py" \
        --ckpt "$BEST_CKPT" --temp "$TEMP" --weight "$WEIGHT" \
        --output "$RL_DIR/rl_$i" --data "$DIR/output/train_3k.jsonl" \
        > "$LOGDIR/rl_${i}.log" 2>&1
    log "DONE RL-$i (exit=$?)"
done

log "=== B DONE: Evaluating best RL ==="

$PYTHON -u "$DIR/eval_rl_best.py" > "$LOGDIR/rl_best.log" 2>&1
RL_BEST=$(grep "BEST:" "$LOGDIR/rl_best.log" | tail -1 | awk '{print $2}')
RL_JF1=$(grep "BEST:" "$LOGDIR/rl_best.log" | tail -1 | awk '{print $4}')
log "RL Best: $RL_BEST (JointF1=$RL_JF1)"

# ═══════════════════════════════════════════
# PHASE C: SPICE (only if JointF1 > 0.05)
# ═══════════════════════════════════════════
SPICE_OK=$(echo "$RL_JF1 > 0.05" | bc -l 2>/dev/null || echo 0)
if [ "$SPICE_OK" = "1" ]; then
    log "=== PHASE C: SPICE Post-Training (JointF1=$RL_JF1 > 0.05) ==="
    $PYTHON -u "$DIR/circuit-ocr-dataset/scripts/train_spice.py" \
        --ckpt "$RL_BEST" --output "$DIR/checkpoints/spice_final" \
        > "$LOGDIR/spice.log" 2>&1
    log "DONE SPICE (exit=$?)"
else
    log "=== SPICE SKIPPED: JointF1=$RL_JF1 < 0.05 ==="
    log "Need more data/training to reach threshold"
fi

log "=== PIPELINE COMPLETE ==="
log "Final model: $RL_BEST"

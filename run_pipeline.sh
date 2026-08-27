#!/bin/bash
# 4-Experiment Pipeline — survives disconnection, auto-continues
PYTHON="E:/080000software/080900_Miniconda/miniconda3/envs/pyqpanda-quantum/python.exe"
SCRIPT="g:/mimo_project/circuit_ocr/circuit-ocr-dataset/scripts/train_robust.py"
LOGDIR="g:/mimo_project/circuit_ocr/experiment_logs"
mkdir -p "$LOGDIR"

run_exp() {
    local name=$1; shift
    echo "[$(date +%H:%M:%S)] START $name" | tee -a "$LOGDIR/pipeline.log"
    $PYTHON "$SCRIPT" "$@" > "$LOGDIR/${name}.log" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] DONE $name (exit=$rc)" | tee -a "$LOGDIR/pipeline.log"
    return $rc
}

# Kill any stale python
pkill -9 -f train_robust 2>/dev/null
sleep 2

run_exp exp1_baseline    --name exp1_baseline    --max_dim 384 --lr 2e-5 --epochs 2 --dropout 0.05
run_exp exp2_hires       --name exp2_hires       --max_dim 512 --lr 2e-5 --epochs 2 --dropout 0.05
run_exp exp3_regularized --name exp3_regularized --max_dim 384 --lr 1e-5 --epochs 3 --dropout 0.10
run_exp exp4_unfrozen    --name exp4_unfrozen    --max_dim 384 --lr 2e-5 --epochs 2 --freeze_projector 0

echo "[$(date +%H:%M:%S)] ALL DONE" | tee -a "$LOGDIR/pipeline.log"

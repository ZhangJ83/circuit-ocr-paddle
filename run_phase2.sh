#!/bin/bash
PYTHON="E:/080000software/080900_Miniconda/miniconda3/envs/pyqpanda-quantum/python.exe"
SCRIPT_DIR="g:/mimo_project/circuit_ocr/circuit-ocr-dataset/scripts"
LOGDIR="g:/mimo_project/circuit_ocr/experiment_logs"
mkdir -p "$LOGDIR"

echo "[$(date +%H:%M:%S)] PHASE2 START" | tee "$LOGDIR/phase2.log"

for exp in exp5 exp6; do
    echo "[$(date +%H:%M:%S)] START $exp" | tee -a "$LOGDIR/phase2.log"
    $PYTHON "$SCRIPT_DIR/${exp}.py" > "$LOGDIR/${exp}.log" 2>&1
    RC=$?
    echo "[$(date +%H:%M:%S)] DONE $exp (exit=$RC)" | tee -a "$LOGDIR/phase2.log"
done

echo "[$(date +%H:%M:%S)] PHASE2 ALL DONE" | tee -a "$LOGDIR/phase2.log"

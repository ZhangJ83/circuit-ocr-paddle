#!/bin/bash
PYTHON="E:/080000software/080900_Miniconda/miniconda3/envs/pyqpanda-quantum/python.exe"
SCRIPT="g:/mimo_project/circuit_ocr/circuit-ocr-dataset/scripts/eval_benchmark.py"
DATA_DIR="g:/mimo_project/circuit_ocr/circuit-ocr-dataset"
MODEL="F:/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27"
DATA=$1
OUT=$2
TOTAL=$3

echo "=== TIER: $DATA -> $OUT ($TOTAL samples) ==="
for i in $(seq 1 200); do
  count=$("$PYTHON" -c "
import json
try:
    with open('$DATA_DIR/$OUT', encoding='utf-8') as f:
        data = [json.loads(l) for l in f if l.strip()]
    print(len([d for d in data if d.get('prediction','') != '']))
except: print(0)
" 2>/dev/null)
  echo "[Loop $i] $count/$TOTAL done"
  if [ "$count" -ge "$TOTAL" ]; then echo "TIER COMPLETE!"; break; fi
  "$PYTHON" "$SCRIPT" --model_type paddleocr-vl --model_name_or_path "$MODEL" --data_path "$DATA_DIR/$DATA" --output_path "$DATA_DIR/$OUT" --max_length 512 --resume 2>&1 | grep -E "\[.*/.*\]|NED|Report|FAIL|Error" | tail -5
  sleep 3
done
echo "DONE with $OUT"

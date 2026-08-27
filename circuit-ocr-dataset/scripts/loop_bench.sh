#!/bin/bash
# Run eval_benchmark.py one sample at a time to avoid Paddle memory leak crashes
PYTHON="E:/080000software/080900_Miniconda/miniconda3/envs/pyqpanda-quantum/python.exe"
SCRIPT_DIR="g:/mimo_project/circuit_ocr/circuit-ocr-dataset/scripts"
DATA_DIR="g:/mimo_project/circuit_ocr/circuit-ocr-dataset"
MODEL="F:/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27"

TIERS=(
  "ocr_vl_sft-test-easy50.jsonl:results_paddleocr-vl_easy50.jsonl:50"
  "ocr_vl_sft-test-easy100.jsonl:results_paddleocr-vl_easy100.jsonl:100"
  "ocr_vl_sft-test-easy200.jsonl:results_paddleocr-vl_easy200.jsonl:200"
  "ocr_vl_sft-test.jsonl:results_paddleocr-vl_full523.jsonl:523"
)

for tier in "${TIERS[@]}"; do
  IFS=':' read -r data output total <<< "$tier"
  echo "=== TIER: $data -> $output ($total samples) ==="
  
  for i in $(seq 1 20); do
    # Count current results
    count=$("$PYTHON" -c "
import json
try:
    with open('$DATA_DIR/$output', encoding='utf-8') as f:
        data = [json.loads(l) for l in f if l.strip()]
    print(len([d for d in data if d.get('prediction','') != '']))
except: print(0)
" 2>/dev/null)
    
    echo "  Loop $i: $count/$total done"
    if [ "$count" -ge "$total" ]; then
      echo "  TIER COMPLETE!"
      break
    fi
    
    "$PYTHON" "$SCRIPT_DIR/eval_benchmark.py" \
      --model_type paddleocr-vl \
      --model_name_or_path "$MODEL" \
      --data_path "$DATA_DIR/$data" \
      --output_path "$DATA_DIR/$output" \
      --max_length 512 \
      --resume \
      2>&1 | tail -3
    
    sleep 5
  done
done
echo "ALL DONE!"

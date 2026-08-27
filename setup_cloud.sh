#!/bin/bash
# Cloud instance setup for V15 training
# Run once after uploading the project

echo "=== Checking environment ==="
python -c "import paddle; print(f'Paddle: {paddle.__version__}')"
python -c "import paddleformers; print('PaddleFormers: OK')"
python -c "from PIL import Image; print('Pillow: OK')"

echo ""
echo "=== Verifying data ==="
python -c "
import json
for f in ['output/train_clean.jsonl', 'output/val_clean.jsonl', 'output/test_clean.jsonl']:
    with open(f) as fh:
        n = sum(1 for l in fh if l.strip())
    print(f'{f}: {n} samples')
"

echo ""
echo "=== Training ==="
# Estimated time: 40-60 min on RTX 4090
python scripts/train_v15_clean.py

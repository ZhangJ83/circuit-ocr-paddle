#!/bin/bash
# LoRA Training Launch Script — avoids inline quoting issues
set -e

# Minimal clean PATH
export PATH="/home/zzz/.local/bin:/usr/local/bin:/usr/bin:/bin"
export HF_HOME=/mnt/f/hf_cache/hub
export PADDLE_HOME=/mnt/f/paddle_cache
export HF_HUB_CACHE=/mnt/f/hf_cache/hub
export KMP_DUPLICATE_LIB_OK=TRUE
export LD_LIBRARY_PATH=/home/zzz/miniconda3/lib:/usr/lib/wsl/lib
export CUDA_VISIBLE_DEVICES=0

cd /mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset

echo "=== LoRA Training Start: $(date) ==="
echo "GPU: $(python3 -c 'import paddle; paddle.set_device("gpu"); print(paddle.device.cuda.get_device_name(0))')"
echo ""

paddleformers-cli train \
    configs/paddleocr-vl_lora_8gb.yaml \
    model_name_or_path=PaddlePaddle/PaddleOCR-VL \
    train_dataset_path=./ocr_vl_sft-train.jsonl \
    eval_dataset_path=./ocr_vl_sft-test.jsonl \
    pre_alloc_memory=6.0

echo ""
echo "=== LoRA Training End: $(date) ==="

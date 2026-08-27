#!/bin/bash
# ============================================================================
# Docker Run Command: PaddleOCR-VL LoRA Training on Circuit OCR
# ============================================================================
# Image:  registry.baidubce.com/paddlepaddle/paddle:2.6.2-gpu-cuda12.0-cudnn8.9-trt8.6
# Script: scripts/train_lora_final.py (must exist at project root)
#
# Image contents (based on PaddlePaddle standard GPU image):
#   Pre-installed:  Python 3.10, PaddlePaddle 2.6.2 GPU, CUDA 12.0,
#                   cuDNN 8.9, TensorRT 8.6, numpy
#   NOT installed:  paddleformers, PaddleOCR, pillow, tqdm, safetensors
#                   -> These will be pip-installed at container start
#
# Usage:
#   1. Pull image first (one-time):
#        docker pull registry.baidubce.com/paddlepaddle/paddle:2.6.2-gpu-cuda12.0-cudnn8.9-trt8.6
#
#   2. Run this script from WSL:
#        cd /mnt/g/mimo_project/circuit_ocr && bash run_training.sh
#
#   Or from Windows PowerShell/CMD:
#        wsl -d Ubuntu -u root sh -c 'cd /mnt/g/mimo_project/circuit_ocr && bash run_training.sh'
# ============================================================================

set -e

# ---- Configuration ----------------------------------------------------------
DOCKER_IMAGE="registry.baidubce.com/paddlepaddle/paddle:2.6.2-gpu-cuda12.0-cudnn8.9-trt8.6"
PROJECT_DIR="/mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset"
HF_CACHE_DIR="/mnt/f/hf_cache/hub"
PADDLE_CACHE_DIR="/mnt/f/paddle_cache"

# Pip mirror (change to default PyPI if not in China)
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

# ---- Pre-flight Checks ------------------------------------------------------
echo "============================================================================="
echo " PaddleOCR-VL LoRA Training — Docker Container"
echo "============================================================================="
echo ""
echo "Image:      ${DOCKER_IMAGE}"
echo "Project:    ${PROJECT_DIR}"
echo "HF cache:   ${HF_CACHE_DIR}"
echo "Paddle:     ${PADDLE_CACHE_DIR}"
echo ""

# Verify Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: docker not found. Ensure Docker is installed with WSL2 backend."
    exit 1
fi
echo "[OK] Docker found: $(docker --version)"

# Verify project directory exists
if [ ! -d "${PROJECT_DIR}" ]; then
    echo "ERROR: Project directory not found: ${PROJECT_DIR}"
    echo "  Mount check: ls /mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset/"
    exit 1
fi
echo "[OK] Project directory exists"

# Verify training script exists
if [ ! -f "${PROJECT_DIR}/scripts/train_lora_final.py" ]; then
    echo "ERROR: Training script not found: ${PROJECT_DIR}/scripts/train_lora_final.py"
    echo "  Available scripts:"
    ls "${PROJECT_DIR}/scripts/train_lora"*.py 2>/dev/null || echo "  (no train_lora*.py scripts found)"
    exit 1
fi
echo "[OK] Training script found: scripts/train_lora_final.py"

# Verify HF cache exists
if [ ! -d "${HF_CACHE_DIR}" ]; then
    echo "ERROR: HF cache directory not found: ${HF_CACHE_DIR}"
    echo "  Create it with: mkdir -p ${HF_CACHE_DIR}"
    exit 1
fi
echo "[OK] HF cache directory exists"

# Ensure paddle cache directory exists
mkdir -p "${PADDLE_CACHE_DIR}"
echo "[OK] Paddle cache directory ready"

# Quick GPU check via Docker (will fail if image not pulled yet — non-fatal)
echo ""
echo "--- GPU Check ---"
docker run --rm --gpus all "${DOCKER_IMAGE}" nvidia-smi 2>&1 || {
    echo ""
    echo "NOTE: GPU check failed (likely image not pulled yet)."
    echo "  Pull the image first: docker pull ${DOCKER_IMAGE}"
    echo "  Then re-run this script."
    exit 1
}

# ---- Run Training -----------------------------------------------------------
echo ""
echo "============================================================================="
echo " Starting Training Container"
echo "============================================================================="
echo ""
echo "Mounts:"
echo "  ${PROJECT_DIR}  ->  /workspace"
echo "  ${HF_CACHE_DIR} ->  /mnt/f/hf_cache/hub"
echo "  ${PADDLE_CACHE_DIR} -> /mnt/f/paddle_cache"
echo ""
echo "Environment:"
echo "  HF_HOME=/mnt/f/hf_cache/hub"
echo "  PADDLE_HOME=/mnt/f/paddle_cache"
echo "  HF_HUB_CACHE=/mnt/f/hf_cache/hub"
echo "  HF_HUB_OFFLINE=1"
echo "  CUDA_VISIBLE_DEVICES=0"
echo "  KMP_DUPLICATE_LIB_OK=TRUE"
echo "  FLAGS_allocator_strategy=auto_growth"
echo ""

docker run --rm -it \
    --gpus all \
    --shm-size=8g \
    --ipc=host \
    -v "${PROJECT_DIR}:/workspace" \
    -v "${HF_CACHE_DIR}:/mnt/f/hf_cache/hub" \
    -v "${PADDLE_CACHE_DIR}:/mnt/f/paddle_cache" \
    -e HF_HOME=/mnt/f/hf_cache/hub \
    -e PADDLE_HOME=/mnt/f/paddle_cache \
    -e HF_HUB_CACHE=/mnt/f/hf_cache/hub \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e KMP_DUPLICATE_LIB_OK=TRUE \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e FLAGS_allocator_strategy=auto_growth \
    -w /workspace \
    "${DOCKER_IMAGE}" \
    bash -c "
        set -e
        echo '=== Installing Missing Dependencies ==='
        echo ''
        pip install --no-cache-dir \
            paddleformers \
            paddleocr \
            pillow \
            tqdm \
            safetensors \
            -i ${PIP_INDEX} \
            && echo '' \
            && echo '=== Dependencies Installed Successfully ===' \
            && echo '' \
            && python -c \"
import paddle
print(f'PaddlePaddle: {paddle.__version__}')
paddle.set_device('gpu')
print(f'GPU: {paddle.device.cuda.get_device_name(0)}')
print(f'VRAM: {paddle.device.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
import paddleformers
print(f'PaddleFormers: {paddleformers.__version__}')
\" \
        && echo '' \
        && echo '=== Running Training Script ===' \
        && echo '' \
        && python scripts/train_lora_final.py
    "

EXIT_CODE=$?

echo ""
echo "============================================================================="
echo " Training container exited with code: ${EXIT_CODE}"
echo "============================================================================="
exit ${EXIT_CODE}

# PaddleOCR-VL LoRA Fine-Tuning Plan — RTX 4060 8GB

## 关键前提：训练为什么比推理快 30 倍

| 模式 | 每样本耗时 | 原理 |
|------|-----------|------|
| **推理**（benchmark） | ~25s | 自回归解码：200 tokens = 200 次串行 forward |
| **训练**（teacher forcing） | ~2.5s | 整段标签一次并行 forward + backward |

推理是串行循环，训练是一次性矩阵运算。2.5s ≈ 0.8s forward + 1.7s backward。

## 1 epoch vs 2 epoch 效果分析

核心问题：**优化器步数**。`batch_size=1 × grad_accum=64` → 每 64 个样本才更新一次权重。

| 指标 | 1 epoch | 2 epoch | 官方配置 |
|------|---------|---------|----------|
| 优化器步数 | **35 步** | **70 步** | 数百步 |
| 每样本被训练 | 1 次 | 2 次 | 2 次 |
| warmup（0.01 × 总步数） | 0.35 步 → **无效** | 0.7 步 → **勉强** | 有意义 |
| cosine LR 从 5e-4→5e-5 | 直线下降，无衰减空间 | 有 70 步余量 | 充分 |

### 35 步（1 epoch）的问题

模型要在 35 次权重更新里学会：
1. 输出格式：REFDES → 类型 → 引脚逐行列出
2. 2250 种不同的元件名和参数值
3. 电路图视觉特征到文本的映射

第 1 项可能勉强够，第 2、3 项（2250 张图每张只过一遍）几乎不可能学好。warmup_ratio=0.01 意味着只有 0.35 步 warmup，实际上 LR 从 0 跳到 5e-4 只用了一步，cosine 衰减紧接着开始，完全没有稳定训练的阶段。

### 70 步（2 epoch）的优势

- 每样本见两次：第一次学格式，第二次巩固细节
- warmup 仍然短但至少有意义（~1 步 vs ~0 步）
- cosine 衰减有 70 步空间，LR 从 5e-4 逐步降到 5e-5
- 匹配官方 "每样本训练 2 次" 的设计

### 为什么不用 3 epoch？

LoRA rank=8 只有 0.02M 可训练参数 vs 959M 冻结参数。容量有限，3 epoch 容易过拟合到训练集的具体元件值，泛化变差。

## 显存预算

| 组件 | 显存 |
|------|------|
| 基础模型权重（冻结） | ~1.83 GB |
| LoRA 权重（rank=8） | ~0.02 GB |
| 优化器状态（AdamW，仅 LoRA） | ~0.06 GB |
| 激活值（recompute=full） | ~0.50 GB |
| 视觉编码器 | ~0.80 GB |
| PaddlePaddle + CUDA | ~1.00 GB |
| 安全余量 | ~1.50 GB |
| **合计** | **~5.7 GB ✅** |

## 时间预算（2 epoch）

| 阶段 | 耗时 |
|------|------|
| 微步训练（fw+bw, 1 样本） | ~2.5s |
| 一次参数更新（×64 累积） | ~160s (2.7 min) |
| 训练（70 步 × 2.7 min） | **~3.1 小时** |
| Eval（50 样本 × 2 次） | ~0.7 小时 |
| Checkpoint | ~0.1 小时 |
| **总计** | **~3.9 小时** |

## 启动

```bash
cd /mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset
bash scripts/train_lora.sh
```

## 训练后：合并 + 跑分验证

```bash
# 合并 LoRA
paddleformers-cli export configs/paddleocr-vl_lora_export.yaml \
    model_name_or_path=PaddlePaddle/PaddleOCR-VL \
    output_dir=./PaddleOCR-VL-LoRA-circuit-ocr

# 跑 easy50 验证
python scripts/eval_benchmark_gpu.py \
    --data_path ocr_vl_sft-test-easy50.jsonl \
    --output_path results_lora_easy50.jsonl --max_length 1024 --resume
```

## 预期效果

参考官方孟加拉语 OCR（base 0.8214 → LoRA 0.0064）：
- Base (easy200): Avg. NED = 0.9079
- LoRA (2 epoch): Avg. NED ≈ **0.01-0.05**

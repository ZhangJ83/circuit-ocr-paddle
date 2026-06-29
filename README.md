# CircuitOCR: Built for Schematic Diagram Understanding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PaddleOCR-VL](https://img.shields.io/badge/Base%20Model-PaddleOCR--VL--0.9B-blue)](https://github.com/PaddlePaddle/PaddleOCR)
[![LoRA](https://img.shields.io/badge/Fine--Tuning-LoRA%20(r%3D16)-green)]()
[![HuggingFace Space](https://img.shields.io/badge/Demo-HuggingFace-orange)](https://huggingface.co/spaces/yingchu83/CircuitOCR)

> 📄 **Technical Report:** [中文版 (PDF)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/template.pdf) | [English (PDF)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/english.pdf) | [LaTeX Source](https://github.com/ZhangJ83/circuit-ocr-paddle/tree/master/arxiv_template)

> 🎮 **Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/yingchu83/CircuitOCR)

> 🏋️ **LoRA Weights:** [HuggingFace Models](https://huggingface.co/yingchu83/CircuitOCR-lora)

---

## English

**PaddleOCR-VL-0.9B + LoRA for Circuit Schematic OCR and Netlist Extraction**

The first open-source benchmark and fine-tuning pipeline for circuit schematic OCR. Achieves **+10.0% Avg. NED improvement** over the base model.

### Highlights

| Feature | Description |
|---------|-------------|
| **First Circuit OCR Benchmark** | 24,717 training + 523 test + 250 degraded evaluation samples |
| **Best Model: r=16 LoRA** | NED 0.7961 (+10.0% vs. base), 53 min training on RTX 4060 |
| **Projector Bottleneck Discovery** | First systematic identification of vision-language projector as key bottleneck |
| **Rank Sweet Spot** | r=8/16/32 full ablation: r=16 optimal, r=32 training diverges |
| **Robustness Verified** | 250 degraded samples across 5 visual transforms — NED identical to clean |
| **Gradio Demo** | Interactive web demo on HuggingFace Space |

### Quick Start

```bash
# Install
pip install paddlepaddle-gpu paddleformers gradio pillow

# One-click benchmark
cd circuit-ocr-dataset
python scripts/eval_benchmark.py \
    --model_type paddleocr-vl \
    --model_name_or_path "PaddlePaddle/PaddleOCR-VL" \
    --paddle_lora_dir "PaddleOCR-VL-LoRA-circuit-ocr" \
    --data_path ocr_vl_sft-test-easy50.jsonl \
    --output_path results.jsonl --max_length 30 --resume

# Launch demo
python demo.py
```

### Results

| Tier | Base | r16 LoRA | Improvement |
|------|------|----------|-------------|
| easy50 | 0.8848 | **0.7961** | **+10.0%** |
| easy100 | 0.8999 | **0.8291** | **+7.9%** |
| easy200 | 0.9139 | **0.8624** | **+5.6%** |
| full523 | 0.9455 | **0.9164** | **+3.1%** |
| Degraded (250) | — | **0.7961** | Robustness verified |

### Project Structure

```
├── arxiv_template/           # Technical report (CN + EN, LaTeX + PDF)
├── circuit-ocr-dataset/
│   ├── scripts/              # Training, evaluation, data building scripts
│   ├── PaddleOCR-VL-LoRA-circuit-ocr/  # LoRA weights
│   ├── docs/                 # Documentation (collection, annotation, QC, stats)
│   ├── figures/              # Generated visualizations
│   └── demo.py               # Gradio demo
└── README.md
```

---

## 中文

**基于 PaddleOCR-VL-0.9B + LoRA 的电路原理图 OCR 与网表提取系统**

首个开源电路原理图 OCR 基准与微调管线。最优模型在 easy50 测试集上取得 **NED 0.7961（+10.0%）**。

### 亮点

| 特性 | 说明 |
|------|------|
| **首个电路 OCR 基准** | 24,717 训练 + 523 测试 + 250 退化评估样本 |
| **最优模型：r=16 LoRA** | NED 0.7961（+10.0%），53 min 训练（RTX 4060） |
| **Projector 瓶颈发现** | 首次系统性定位视觉-语言投影层为电路 OCR 关键瓶颈 |
| **Rank 甜点区间** | r=8/16/32 完整消融：r=16 最优，r=32 训练发散 |
| **退化鲁棒性验证** | 250 退化样本 × 5 种变换 — NED 与原始完全一致 |
| **Gradio 在线演示** | HuggingFace Space 交互式 Demo |

### 快速开始

```bash
# 安装
pip install paddlepaddle-gpu paddleformers gradio pillow

# 一键基准测试
cd circuit-ocr-dataset
python scripts/eval_benchmark.py \
    --model_type paddleocr-vl \
    --model_name_or_path "PaddlePaddle/PaddleOCR-VL" \
    --paddle_lora_dir "PaddleOCR-VL-LoRA-circuit-ocr" \
    --data_path ocr_vl_sft-test-easy50.jsonl \
    --output_path results.jsonl --max_length 30 --resume

# 启动 Demo
python demo.py
```

### 实验结果

| 测试层级 | Base | r16 LoRA | 改善 |
|---------|------|----------|------|
| easy50 | 0.8848 | **0.7961** | **+10.0%** |
| easy100 | 0.8999 | **0.8291** | **+7.9%** |
| easy200 | 0.9139 | **0.8624** | **+5.6%** |
| full523 | 0.9455 | **0.9164** | **+3.1%** |
| 退化集 (250) | — | **0.7961** | 鲁棒性验证通过 |

### 目录结构

```
├── arxiv_template/           # 技术报告（中英文 LaTeX + PDF）
├── circuit-ocr-dataset/
│   ├── scripts/              # 训练、评估、数据构建脚本
│   ├── PaddleOCR-VL-LoRA-circuit-ocr/  # LoRA 权重
│   ├── docs/                 # 文档（采集、标注、质控、统计）
│   ├── figures/              # 可视化图表
│   └── demo.py               # Gradio 演示
└── README.md
```

---

## Links / 链接

| Resource | URL |
|----------|-----|
| 📄 Technical Report (CN) | [template.pdf](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/template.pdf) |
| 📄 Technical Report (EN) | [english.pdf](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/english.pdf) |
| 🎮 Live Demo | [HuggingFace Space](https://huggingface.co/spaces/yingchu83/CircuitOCR) |
| 🏋️ LoRA Weights | [HuggingFace Models](https://huggingface.co/yingchu83/CircuitOCR-lora) |
| 📦 Training Dataset | [GitHub](https://github.com/ZhangJ83/circuit_ocr_dataset_final) |
| 📦 Synthetic Dataset | [GitHub](https://github.com/ZhangJ83/circuit-ocr-dataset) |

## Citation / 引用

```bibtex
@misc{zhang2026circuitocr,
  title={PaddleOCR-VL-Circuit: Built for Schematic Diagram Understanding},
  author={Jianning Zhang and Yifei Chen},
  year={2026},
  url={https://github.com/ZhangJ83/circuit-ocr-paddle},
}
```

## License / 许可证

MIT License. Open Schematics and Masala-CHAI datasets under CC-BY-4.0.

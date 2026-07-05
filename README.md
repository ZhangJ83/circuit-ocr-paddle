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

The first open-source benchmark and fine-tuning pipeline for circuit schematic OCR. Phase 1 evaluation achieves **Component F1 0.2061 (4.5× improvement)** and **NED 0.8031 (13.6% relative error reduction)** over the base model.

### Phase 1 Benchmark (V10-Fixed, easy50-pure, 44 samples)

> Evaluated with `eval_benchmark_v3.py` using LoRAModel wrapper + `p.set_value()` (fixes Paddle 3.1.0 `set_state_dict` → None bug)

| Model | ExactMatch | CompF1 | CompPrec | CompRec | TokenRec | NED ↓ | RepRate | Diversity |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Base (PaddleOCR-VL-0.9B) | 0% | 0.0455 | 0.0455 | 0.0455 | 0.0016 | 0.9296 | 6.8% | 90.9% |
| S400 (LoRA step 400) | 0% | 0.1820 | 0.1862 | 0.2501 | 0.1302 | 0.8298 | 20.5% | 95.5% |
| **S600 (LoRA step 600)** ★ | 0% | **0.2061** | 0.2024 | **0.3114** | **0.1540** | **0.8031** | 15.9% | 90.9% |
| S800 (LoRA step 800) | 0% | 0.2080 | 0.2862 | 0.1996 | 0.1191 | 0.8063 | 40.9% | 93.2% |

> **Note:** V11 (regularized training, evaluated on the same easy50-pure split, 44 samples) performed worse than baseline in most metrics: CompF1=0.0604, NED=0.9171, RepRate=84.1%, Diversity=50%. The regularized training approach with synthetic data was counterproductive.

### Phase 2 Topology Metrics (V10-Fixed S600, easy50-pure)

> Full 4-split results (easy50/100/200/full523) are available in the [technical report Appendix B](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/english.pdf).

| Metric | Value | Note |
|:---|---:|:---|
| joint_f1 (refdes + value) | 0.019 | Only ~2% of components have both refdes and value correct |
| value_acc | 0.13 | 87% of values are hallucinated |
| ExactMatch | 0% | No model configuration produces a fully correct netlist |

### Key Findings

| Metric | Improvement | Detail |
|--------|:-----------:|--------|
| Component F1 | **4.5×** | 0.0455 → 0.2061 |
| Token Recall | **96×** | 0.0016 → 0.1540 |
| NED | **13.6%** relative error reduction | 0.9296 → 0.8031 |
| Best Checkpoint | **S600** | S800 overfits (repetition 40.9%) |
| Diversity | **90.9%** | No modality collapse |

### Research Contributions

Despite ExactMatch=0%, the results represent genuine progress for a 0.9B-scale model fine-tuned on consumer hardware:

| Context | Detail |
|:---|---|
| **Model scale** | 0.9B parameters, 5.7M trainable (0.63%) — at this scale, ExactMatch=0% is expected for open-vocabulary structured output |
| **Data budget** | 1,554 training samples — a realistic constraint for niche domains without large labeled datasets |
| **Training cost** | 43 minutes on a consumer RTX 4060 (8GB VRAM) — any individual developer can reproduce |
| **Core achievement** | CompF1 4.5× (0.0455→0.2061), TokenRec 96× (0.0016→0.1540), diversity maintained at 90.9% |
| **Accessibility** | Runs on consumer hardware, no data center needed — democratizes circuit OCR research |

### Exploration Process (V1 → V10)

| Phase | Version | Key Discovery |
|------|---------|---------------|
| V1–V4 | Full LoRA | **Modality collapse**: Projector LoRA destroys pre-trained alignment → 4% diversity |
| V5 | LLM-Only LoRA (r=8) | Freeze Projector → diversity recovers to 90%, proves architecture direction |
| V6–E6 | Controlled experiments | 6 systematic experiments isolating variables (blank image, resolution, epochs, projector layers, LoRA rank, freeze strategy) → identified Projector LoRA as sole root cause of modality collapse |
| V8-Fixed | Wide LoRA (r=16) | 3 training pitfalls discovered and documented for the community: (1) causal token double-shift affects ALL PaddleOCR-VL fine-tuning, (2) BPE boundary merging affects ALL sequence-generation fine-tuning, (3) set_state_dict→None is a Paddle 3.1.0 API compatibility issue. Three additional training infrastructure bugs (LoRA weight precision loss, tokenizer special token offset, gradient accumulation/LR decoupling) are documented in the [dataset README](circuit-ocr-dataset/README.md). |
| V9-Pure | Final training | 1,554 samples, 3 epochs, easy100 NED 0.7797 |
| **V10-Fixed** | **Phase 1 eval** | Multi-metric evaluation: CompF1, TokenRec, NED, RepRate, Diversity |

### V11 & V12 Progress

| Version | Approach | Status | Key Result |
|---------|----------|--------|------------|
| V11 (Phase 2) | Regularized: dropout=0.1, label_smoothing=0.05, data augmentation, 3,054 samples | Completed | Mode collapse — RepRate monotonically increased to 84.1%. Synthetic data visual distribution mismatch confirmed. |
| V12 (Phase 3) | Two-stage: LLM LoRA warmup (V10 S600) → Vision LoRA r=4, 448px resolution | Completed — collapsed | Vision LoRA retraining destroyed the LLM's text generation capability. All 50 predictions are garbage: numeric strings ("100000..."), repeated "+333...", empty strings, or repeated "VCC"/"GND". Two-stage approach confirmed harmful. |

### Previous Benchmark (V9-Pure)

| Tier | Base NED | V9-Pure NED | Improvement |
|------|----------|-------------|-------------|
| easy50-pure | 0.9424 | **0.7869** | **-16.5%** |
| easy100-pure | 0.9390 | **0.7797** | **-17.0%** |

### Quick Start

```bash
# Install
pip install paddlepaddle-gpu paddleformers gradio pillow

# One-click benchmark (Phase 1)
cd circuit-ocr-dataset/scripts
python eval_benchmark_v3.py \
    --data_path ../ocr_vl_sft-test-easy50-pure.jsonl \
    --lora_checkpoint ../PaddleOCR-VL-LoRA-circuit-ocr/checkpoints_v10_fixed/lora_s600.pdparams

# Launch demo
python demo.py
```

### Project Structure

```
├── arxiv_template/           # Technical report (CN + EN, LaTeX + PDF)
├── circuit-ocr-dataset/
│   ├── scripts/              # Training, evaluation, data building scripts
│   │   ├── eval_benchmark_v3.py        # Phase 1 fixed eval (LoRAModel wrapper)
│   │   ├── train_llm_v10_fixed.py      # V10 training script
│   │   └── diagnose_lora_merge.py      # LoRA merge diagnostic
│   ├── PaddleOCR-VL-LoRA-circuit-ocr/  # LoRA weights (checkpoints_v10_fixed/)
│   ├── docs/                 # Documentation
│   ├── figures/              # Generated visualizations
│   └── demo.py               # Gradio demo
└── README.md
```

---

## 中文

**基于 PaddleOCR-VL-0.9B + LoRA 的电路原理图 OCR 与网表提取系统**

首个开源电路原理图 OCR 基准与微调管线。阶段一评估最优模型 S600 取得 **Component F1 0.2061（4.5× 提升）**、**NED 0.8031（13.6% 相对误差降低）**。

### 阶段一基准测试（V10-Fixed, easy50-pure, 44样本）

> 使用 `eval_benchmark_v3.py`（LoRAModel wrapper + `p.set_value()`，修复了 Paddle 3.1.0 `set_state_dict` 返回 None 的 Bug）

| 模型 | ExactMatch | CompF1 | CompPrec | CompRec | TokenRec | NED ↓ | RepRate | Diversity |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Base (PaddleOCR-VL-0.9B) | 0% | 0.0455 | 0.0455 | 0.0455 | 0.0016 | 0.9296 | 6.8% | 90.9% |
| S400 (LoRA step 400) | 0% | 0.1820 | 0.1862 | 0.2501 | 0.1302 | 0.8298 | 20.5% | 95.5% |
| **S600 (LoRA step 600)** ★ | 0% | **0.2061** | 0.2024 | **0.3114** | **0.1540** | **0.8031** | 15.9% | 90.9% |
| S800 (LoRA step 800) | 0% | 0.2080 | 0.2862 | 0.1996 | 0.1191 | 0.8063 | 40.9% | 93.2% |

> **注：** V11（正则化训练，同一 easy50-pure 测试集，44 样本）在大多数指标上劣于基线：CompF1=0.0604，NED=0.9171，RepRate=84.1%，Diversity=50%。正则化+合成数据的方案适得其反。

### 阶段二拓扑指标（V10-Fixed S600, easy50-pure）

> 完整的四组测试结果（easy50/100/200/full523）详见[技术报告附录 B](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/template.pdf)。

| 指标 | 数值 | 说明 |
|:---|---:|:---|
| joint_f1（标识符+数值） | 0.019 | 仅约 2% 的元件标识符和数值同时正确 |
| value_acc | 0.13 | 87% 的数值被幻觉生成 |
| ExactMatch | 0% | 所有模型配置均无法输出完全正确的网表 |

### 关键发现

| 指标 | 提升幅度 | 详情 |
|------|:---:|------|
| Component F1 | **4.5×** | 0.0455 → 0.2061 |
| Token Recall | **96×** | 0.0016 → 0.1540 |
| NED | **13.6%** 相对误差降低 | 0.9296 → 0.8031 |
| 最佳 Checkpoint | **S600** | S800 过拟合（重复率 40.9%） |
| 多样性 | **90.9%** | 无模态坍塌 |

### 研究贡献

尽管 ExactMatch=0%，在 0.9B 规模模型 + 消费级 GPU 的条件下，以下成果代表了真实进展：

| 背景 | 详情 |
|:---|---|
| **模型规模** | 0.9B 参数，5.7M 可训练（0.63%）——在此规模下，开放词表结构化输出的 ExactMatch=0% 是预期结果 |
| **数据预算** | 1,554 训练样本——契合小众领域缺乏大规模标注数据的现实约束 |
| **训练成本** | RTX 4060（8GB 显存）上仅需 43 分钟——任何个人开发者均可复现 |
| **核心成就** | CompF1 4.5×（0.0455→0.2061），TokenRec 96×（0.0016→0.1540），多样性保持 90.9% |
| **可及性** | 消费级硬件即可运行，无需数据中心——降低电路 OCR 研究的门槛 |

### 探索历程（V1 → V10）

| 阶段 | 版本 | 关键发现 |
|------|------|---------|
| V1–V4 | 全量 LoRA | **模态坍塌**：Projector LoRA 破坏预训练对齐 → 多样性仅 4% |
| V5 | LLM-Only LoRA (r=8) | 冻结 Projector → 多样性恢复至 90%，验证架构方向 |
| V6–E6 | 受控实验 | 6 组系统实验，逐一隔离变量（空白图、分辨率、epoch、Projector 层、LoRA rank、冻结策略）→ 锁定 Projector LoRA 为模态坍塌的唯一根因 |
| V8-Fixed | Wide LoRA (r=16) | 三大训练陷阱的发现与社区文档化：(1) causal token 双重偏移影响所有 PaddleOCR-VL 微调，(2) BPE 边界合并影响所有序列生成微调，(3) set_state_dict→None 是 Paddle 3.1.0 API 兼容性问题。另外三个训练基础设施 Bug（LoRA 权重精度丢失、分词器特殊 token 偏移、梯度累积/学习率解耦）记录于[数据集 README](circuit-ocr-dataset/README.md)。 |
| V9-Pure | 最终训练 | 1,554 样本，3 epoch，easy100 NED 0.7797 |
| **V10-Fixed** | **阶段一评估** | 多指标评估体系：CompF1、TokenRec、NED、RepRate、Diversity |

### V11 与 V12 进展

| 版本 | 方案 | 状态 | 关键结果 |
|------|------|------|---------|
| V11（阶段二） | 正则化训练：dropout=0.1, label_smoothing=0.05, 数据增强, 3,054 样本 | 已完成 | 模态坍塌——RepRate 单调上升至 84.1%。确认合成数据视觉分布不匹配。 |
| V12（阶段三） | 两阶段训练：LLM LoRA 预热（V10 S600）→ Vision LoRA r=4, 448px 分辨率 | 已完成 — 崩溃 | Vision LoRA 重新训练完全破坏了 LLM 的文本生成能力。全部 50 个预测均为垃圾输出：纯数字串（"100000..."）、重复 "+333..."、空字符串、或重复 "VCC"/"GND"。两阶段方案被证实有害。 |

### 先前基准（V9-Pure）

| 测试层级 | Base NED | V9-Pure NED | 改善 |
|---------|----------|-------------|------|
| easy50-pure | 0.9424 | **0.7869** | **-16.5%** |
| easy100-pure | 0.9390 | **0.7797** | **-17.0%** |

### 快速开始

```bash
# 安装
pip install paddlepaddle-gpu paddleformers gradio pillow

# 阶段一基准测试
cd circuit-ocr-dataset/scripts
python eval_benchmark_v3.py \
    --data_path ../ocr_vl_sft-test-easy50-pure.jsonl \
    --lora_checkpoint ../PaddleOCR-VL-LoRA-circuit-ocr/checkpoints_v10_fixed/lora_s600.pdparams

# 启动 Demo
python demo.py
```

### 目录结构

```
├── arxiv_template/           # 技术报告（中英文 LaTeX + PDF）
├── circuit-ocr-dataset/
│   ├── scripts/              # 训练、评估、数据构建脚本
│   │   ├── eval_benchmark_v3.py        # 阶段一修复版评估（LoRAModel wrapper）
│   │   ├── train_llm_v10_fixed.py      # V10 训练脚本
│   │   └── diagnose_lora_merge.py      # LoRA merge 诊断
│   ├── PaddleOCR-VL-LoRA-circuit-ocr/  # LoRA 权重（checkpoints_v10_fixed/）
│   ├── docs/                 # 文档
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

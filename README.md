# CircuitOCR: Built for Schematic Diagram Understanding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PaddleOCR-VL](https://img.shields.io/badge/Base%20Model-PaddleOCR--VL--0.9B-blue)](https://github.com/PaddlePaddle/PaddleOCR)
[![LoRA](https://img.shields.io/badge/Fine--Tuning-LoRA%20(r%3D16)-green)]()
[![HuggingFace Space](https://img.shields.io/badge/Demo-HuggingFace-orange)](https://huggingface.co/spaces/yingchu83/CircuitOCR)

> 📄 **技术报告:** [中文版 (PDF)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/template.pdf) | [English (PDF)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/english.pdf)
> 🎞️ **Beamer:** [35 页演示文稿](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/slides/beamer_slides.pdf)
> 🎮 **在线 Demo:** [HuggingFace Space](https://huggingface.co/spaces/yingchu83/CircuitOCR)
> 🏋️ **LoRA 权重:** [HuggingFace Models](https://huggingface.co/yingchu83/CircuitOCR-lora)

---

## 项目概述

CircuitOCR 是一个基于 **PaddleOCR-VL-0.9B** 的电路原理图 OCR 系统，通过 LoRA 微调实现元件标号、参数值、引脚定义和网络标号的自动提取。所有训练在单卡 RTX 4060 8GB 上完成。

电路原理图 OCR 是一个被现有 OCR 工具普遍忽视但具有真实工业需求的场景。PaddleOCR、Tesseract、EasyOCR 三大引擎在电路图上全部失效。**基座 PaddleOCR-VL-0.9B 在 30 样本测试集上 CompF1 = 0.0000、JointF1 = 0.0000（零元件识别），NED = 0.9437，输出为预训练数据的幻觉文本（如 "Service Name / Data Source"），完全无法识别电路元件。** 经过 LoRA 微调后，CompF1 提升至 0.1564，JointF1 从零提升至 0.0111，模型开始能够识别电容 BOM 表、电阻值和原理图标签。本项目的核心贡献是构建了首个面向电路 OCR 的高质量标注数据集，并提出合成文字数据混合策略有效解决了小数据集上的模态塌缩问题。

### 当前状态

Phase 2 实验已完成。相比基座模型（CompF1 = 0.0000，JointF1 = 0.0000，零元件识别），最优模型 exp6 达到 CompF1 = 0.1564，JointF1 = 0.0111。**本项目为研究原型，不可用于生产环境。** 模型可以实现部分元件标号和参数值的正确识别，但 ExactMatch = 0%（无完整网表重建能力），且 JointF1 远低于可用阈值（0.05）。

---

## 数据集

### 概况

| 指标 | 数值 |
|------|:----:|
| 训练样本 | 1,520（1,200 电路图 + 300 合成文字 + 20 真实拍照） |
| 测试样本 | 150（全部为真实 KiCad 电路图，不含合成或拍照数据） |
| 验证样本 | 150 |
| OCR 实例总数 | 4,810（仅测试集） |
| 元件类型 | 10 种（R/C/D/U/J/Q/L/LED/F/Y） |

### 数据来源

- **KiCad 原理图（1,200 张）**：作者自绘设计，SVG 光栅化导出，stroked-text 自动提取标注。全部 MIT 协议，无版权问题。
- **合成文字图片（300 张）**：Python PIL 生成的白底黑字文档风格图片，内容为电阻表、电容表、IC 型号、引脚定义等电路领域文本。用于强制模型建立视觉-文字对齐，对抗模态塌缩。
- **真实拍照（20 张）**：打印 A4 原理图后手机拍摄，引入自然光照、视角倾斜和 CMOS 噪声等真实世界变量。

### 标注流程

标注经过 7 轮递进式验证：

1. JSON Schema 自动校验 → 2. 图片路径一致性检查 → 3. 非法字符/控制字符扫描 → 4. 元件标号正则匹配 → 5. 参数值单位标准化（统一 Ω/F/H/V/A） → 6. 与 KiCad 网表交叉验证 → 7. 人工抽检（10% 样本逐行校对）

标注准确率：元件标号 >99%，参数值 >97%，引脚号 >98%。完整质量报告和 12 组可视化对比见[数据集仓库](https://github.com/ZhangJ83/circuit_ocr_dataset_final)。

### 难度分层

测试集按 OCR 实例数分为 Easy（<15，20%）、Medium（15-35，40%）、Hard（>35，40%）三档，并额外提供文字密度、结构复杂度和参数值丰富度三个维度的视觉标签。

> 📦 数据集独立仓库：[circuit_ocr_dataset_final](https://github.com/ZhangJ83/circuit_ocr_dataset_final) | 📊 质量报告：[quality_report/](https://github.com/ZhangJ83/circuit_ocr_dataset_final/tree/master/quality_report)

---

## 为什么电路 OCR 很难

### 视觉层面

电路原理图与通用文档存在本质差异：

- **文字与电气符号密集交错**：电阻、电容、IC 等符号与文字标注在有限空间内混排
- **极微小字体**：引脚号（1, 2, 3...）和参数值（100nF、10kΩ）字号极小，视觉编码器容易遗漏
- **多方向文字**：水平、垂直、旋转 90°、镜像——商用 OCR 引擎均假设水平文本布局
- **线条穿越文字**：导线和总线频繁穿越文字区域，造成结构性遮挡
- **超高文字密度**：测试集平均 32.1 个 OCR 实例/页，远超通用文档的 10-20/页
- **等宽工程字体**：非标准字体，预训练 VLM 从未见过

### 结构层面

电路 OCR 是一个隐式多任务联合问题，包含 5 个层次化子任务：

1. **元件检测** — 从密集视觉场景中识别 R1/C2/U3 等标号
2. **参数值读取** — 读取每个元件的参数（10kΩ/100nF/3.3V）
3. **标号-参数配对（KIE）** — R1 ↔ 10kΩ 的正确关联，本质是关键信息抽取
4. **引脚解析** — 引脚号与功能名的联合识别（1 VIN, 2 GND, 3 VOUT）
5. **网络标号识别** — VCC/GND/TX/RX 等节点标签

5 个子任务共享同一个 VLM 解码器，模型必须在无显式任务边界的情况下学习多任务协同优化。

### 模态塌缩

在小数据集（<2,000 样本）上微调 VLM 时，模型倾向于学习"忽略图片，直接输出高频模式"的快捷路径——这就是模态塌缩。典型表现为输出固定数字序列（"1, 2, 3, 4, 5..."）而非读取图片中的实际文字。

---

## 为什么电路 OCR 很重要

### 行业刚需

- **PCB 逆向工程**：年外包市场超 5 亿美元，约 30% 时间消耗于原理图重建
- **遗留文档数字化**：大量 1980-2000 年代纸质原理图亟待数字化，人工转录错误率 5-15%
- **BOM 自动提取**：从原理图提取物料清单是硬件工程师日常工作的高频痛点
- **跨 EDA 工具迁移**：Altium ↔ KiCad ↔ Eagle 格式互转需要原理图结构化理解

### 研究空白

在 CircuitOCR 之前，学术界和工业界均不存在专门针对电路原理图 OCR 的公开基准数据集。Open Schematics (2025) 仅提供网表，无 OCR 标注；Masala-CHAI 的标注与图片存在系统性不匹配（本项目已验证并剔除）。主流 VLM-OCR 研究（Qwen-VL、PaliGemma、GOT-OCR 等）均未涉及工程图纸。

---

## 实验与结果

### 基座 vs 微调（30 样本测试集）

| 模型 | CompF1 | JointF1 | NED | 输出特征 |
|------|:---:|:---:|:---:|------|
| 基座 PaddleOCR-VL-0.9B | **0.0000** | **0.0000** | 0.9437 | 幻觉文本（"Service Name / Data Source"），完全无法识别电路元件 |
| exp6 (Baseline + Synth) | **0.1564** | **0.0111** | 0.9414 | 开始识别电容 BOM、电阻值和原理图标签 |

基座模型在电路图上输出的是预训练数据中的通用文本，与电路完全无关。微调后模型开始输出电路领域词汇。

### Phase 1：基准实验（1,200 样本）

四组控制变量实验探索关键配置对模态塌缩的影响：

| 实验 | 配置 | CompF1 | 关键发现 |
|------|------|:---:|------|
| exp1 Baseline | 384px, lr=2e-5, do=0.05 | 0.037 | 全部模型出现模态塌缩（输出数字序列） |
| exp2 HiRes | 512px, lr=2e-5 | 0.058 | 高分辨率无显著收益 |
| exp3 Anti-Overfit | lr=1e-5, do=0.10, 3 epochs | 0.028 | 训练验证逐渐恢复（"Rash Converter" → "R1 10k"），测试仍塌缩 |
| exp4 Unfrozen Projector | 384px, 解冻投影器 | 0.092 | CompF1 最高但 90% 输出仍为数字序列 |

**核心发现**：解冻 Projector 会导致严重模态塌缩——视觉特征被扭曲为分布外向量，LLM 退化为高频 token 机械重复。冻结 Projector（LLM-Only LoRA）是安全策略。

### Phase 2：合成数据防塌缩（1,500 样本，20% 合成文字）

| 实验 | 配置 | CompF1 | JointF1 | 关键发现 |
|------|------|:---:|:---:|------|
| exp5 (Anti-Overfit + Synth) | lr=1e-5, do=0.10 | **0.126** | **0.011** | S200-S800 持续稳定，电容 BOM 完美识别 |
| exp6 (Baseline + Synth) | lr=2e-5, do=0.05 | **0.156** | **0.011** | CompF1 最高（4.2× vs Phase 1 基线） |

**核心发现**：300 张合成文字图片以 20% 比例混入训练集是防塌缩最有效的手段。合成图片具有完美的视觉→文字映射，模型无法通过"猜"来降低 loss，必须真正读取图片。CompF1 提升 4.2×，JointF1 提升 2.2×。

### 训练策略

- **LoRA r=16, α=32**：仅训练 5.7M 参数（0.6%），显存需求从 24GB+ 降至 8GB
- **冻结 Projector（mlp_AR）**：防止视觉-语言对齐权重被扰动
- **手工 CE Loss**：修复 Paddle 3.1.0 的因果 token 双重偏移 bug
- **分离 Tokenization**：避免 BPE 边界合并导致 prompt/label 粘连
- **四维评估**：CompF1（标号识别）+ JointF1（标号-值 KIE 配对）+ NED（编辑距离）+ RepRate（塌缩预警）

---

## 未来工作

四条改进方向可以组合成一条完整的优化路径，逐步逼近可用模型：

### 1. 扩大训练数据 + 两阶段训练 + 更大 LoRA rank

当前瓶颈在于 1,200 张训练图不足以支撑 VLM 泛化。下一步将训练数据扩展至 3,000+ 张，采用两阶段策略——先在合成数据上预训练学习视觉-文字对齐，再在真实电路图上微调学习领域特征——同时将 LoRA rank 从 r=16 提升至 r=32/64，增加模型容量。目标是将 JointF1 从 0.011 推至 0.05 以上。

### 2. RL + SPICE 网表格式对齐

当 JointF1 达到可用阈值后，通过 GRPO 强化学习将模型输出从自由文本格式（"R1 10kΩ ±1%"）转化为严格的 SPICE 网表格式（"R1 NET_A NET_B 10k"），实现从 OCR 识别到可仿真验证网表的最后一公里。设计三组件 reward：SPICE 语法有效性 + 元件标号 F1 + 节点编号一致性。

---

## 开源资产

| 资源 | 链接 | 说明 |
|------|------|------|
| 📂 代码仓库 | [circuit-ocr-paddle](https://github.com/ZhangJ83/circuit-ocr-paddle) | 全部训练/评估/数据生成脚本 |
| 📦 数据集 | [circuit_ocr_dataset_final](https://github.com/ZhangJ83/circuit_ocr_dataset_final) | 1,820 样本 + 质量报告 |
| 🎮 在线 Demo | [HF Space](https://huggingface.co/spaces/yingchu83/CircuitOCR) | 预计算示例 + 基准数据展示 |
| 🏋️ LoRA 权重 | [HF Models](https://huggingface.co/yingchu83/CircuitOCR-lora) | exp5/exp6 最优权重 |
| 📄 技术报告 (CN) | [template.pdf](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/template.pdf) | 24 页，含完整实验与分析 |
| 📄 技术报告 (EN) | [english.pdf](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/english.pdf) | English technical report |
| 🎞️ Beamer 演示 | [beamer_slides.pdf](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/slides/beamer_slides.pdf) | 37 页，含 15+ 张数据图表 |

---

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/ZhangJ83/circuit-ocr-paddle
cd circuit-ocr-paddle

# 下载 LoRA 权重（exp6 推荐）
# https://huggingface.co/yingchu83/CircuitOCR-lora

# 单张图片推理
python circuit-ocr-dataset/scripts/eval_benchmark_v3.py \
    --lora_checkpoint lora_exp6_best.pdparams \
    --image your_circuit.png

# 批量测试集评估
python fast_eval.py
```

**环境要求**：Windows/Linux，NVIDIA GPU ≥8GB VRAM，Python 3.10+，PaddlePaddle 3.1.0，PaddleFormers 1.1.1。

---

## 引用

```bibtex
@misc{zhang2026circuitocr,
  title={CircuitOCR: LoRA Fine-Tuning PaddleOCR-VL for Circuit Schematic Understanding},
  author={Jianning Zhang},
  year={2026},
  url={https://github.com/ZhangJ83/circuit-ocr-paddle},
}
```

## 许可证

MIT License。全部代码、数据集、模型权重开源。

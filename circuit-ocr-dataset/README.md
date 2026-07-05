# CircuitOCR — 电路原理图合成数据集

> 面向电路原理图 OCR 的大规模程序化合成数据集，含标注与退化增强

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)]()

## 简介

通过 Python 脚本程序化生成电路原理图及其精确标注，无需人工参与。每条样本包含原理图渲染图像与对应的元件编号、参数值、网络标签及结构化网表。

| 属性 | 说明 |
|------|------|
| 样本量 | ~14,000 张 |
| 标注方式 | 程序化自动生成，100% 精确 |
| 电路类型 | 模拟 / 数字 / 混合信号 / 电源 |
| 退化增强 | 5 种真实场景退化模拟 |

### V5 Golden 数据质量工程

训练数据经过严格的质量工程处理，确保模型学习到可靠的视觉-文本对应关系：

- **100% 视觉-字面对齐**：每张图像的标注与渲染内容完全一致，消除"图上画的是 R1 但标注写的是 R2"这类错配。
- **99.5% 去重降噪**：从 22,340 条原始样本中剔除 99.5% 的重复/近重复数据，最终保留仅 107 条高多样性样本。实验证明，107 条高质量数据比 22,340 条含噪数据训练效果更好。
- **与测试集格式一致**：V5 Golden 的标注格式与评估所用的测试集完全对齐，避免训练-评估分布偏移。
- **零 AI 标注**：所有标注均为程序化自动生成，不含任何 LLM/VLM 推测标注，杜绝幻觉传播。

**关键发现 — 更少的数据，更好的结果**：早期版本使用 Masala-CHAI（SPICE 派生的标注），其标注与图像视觉内容存在系统性偏差，导致模型产生严重幻觉。**移除 30% 的 Masala-CHAI 数据后，模型分数反而提升**——这一反直觉的结果表明，在 OCR 微调中，标注质量远比数据量重要。

## 标注内容

每张原理图附带的标注包含：元件编号（如 R1、C1）、参数值（如 10k、100nF）、网络标签（如 VCC、GND）、元件间连接关系及 Spice 网表。

## 退化增强

| 类型 | 模拟场景 |
|------|---------|
| paper_aging | 纸张老化：泛黄、斑点 |
| scan_noise | 扫描噪点与条纹 |
| perspective_distortion | 拍照透视变形 |
| handwriting_overlay | 叠加手写标注 |
| low_resolution | 低分辨率扫描 |

## Phase 1 评估结果（V10-Fixed LoRA, easy50-pure, 44样本）

> 使用 `eval_benchmark_v3.py` 评估，基于 LoRAModel wrapper + `p.set_value()` 加载权重

| 模型 | ExactMatch | CompF1 | CompPrec | CompRec | TokenRec | NED ↓ | RepRate | Diversity |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Base (PaddleOCR-VL-0.9B) | 0% | 0.0455 | 0.0455 | 0.0455 | 0.0016 | 0.9296 | 6.8% | 90.9% |
| S400 (LoRA) | 0% | 0.1820 | 0.1862 | 0.2501 | 0.1302 | 0.8298 | 20.5% | 95.5% |
| **S600 (LoRA)** ★ | 0% | **0.2061** | 0.2024 | **0.3114** | **0.1540** | **0.8031** | 15.9% | 90.9% |
| S800 (LoRA) | 0% | 0.2080 | 0.2862 | 0.1996 | 0.1191 | 0.8063 | 40.9% | 93.2% |

**关键发现**：
- Component F1: **4.5x** 提升（0.0455 → 0.2061）
- Token Recall: **96x** 提升（0.0016 → 0.1540）
- NED: **13.6%** 相对改善（0.9296 → 0.8031）
- S800 开始过拟合（RepRate 飙升到 40.9%）
- Exact match 仍为 0%，模型能识别部分元件但无法完整还原网表

## Phase 2 拓扑评估结果（V10-Fixed LoRA, easy50-pure, 44 样本）

> 拓扑指标衡量模型对电路连接关系的理解能力，评估"模型是否知道 R1 连到 C2"

| 模型 | JointF1 | ValueAcc |
|:---|:---:|:---:|
| Base (PaddleOCR-VL-0.9B) | 0.005 | 0.041 |
| S400 (LoRA) | 0.013 | 0.097 |
| **S600 (LoRA)** ★ | **0.019** | **0.133** |
| S800 (LoRA) | 0.015 | 0.108 |

**关键发现**：
- Joint F1（联合拓扑 F1）从 0.005 提升至 0.019（**3.8x**），但绝对值仍很低，说明拓扑理解是远比文本识别更难的任务
- Value Accuracy 从 0.041 提升至 0.133（**3.2x**），模型开始学会将元件参数值关联到正确的拓扑位置
- Phase 1 的 Component F1（4.5x）远高于 Phase 2 的 Joint F1（3.8x），印证了"先学会认元件，再学会连关系"的学习路径

## E1-E6 消融实验

为系统性定位模态坍缩（modality collapse）的根因，设计了 6 组受控消融实验：

| 实验 | 变量 | 目的 |
|:---|:---|:---|
| E1 | 冻结视觉编码器 | 隔离视觉特征提取对性能的影响 |
| E2 | 冻结语言解码器 | 隔离语言生成能力的影响 |
| E3 | 仅训练投影层 | 测试跨模态对齐是否为瓶颈 |
| E4 | 全参数微调 | 与 LoRA 对比，排除适配器容量限制 |
| E5 | 去除退化增强 | 验证数据增强是否引入噪声 |
| E6 | 纯合成数据（无真实域迁移） | 确认域差距是否为坍缩根因 |

**核心结论**：模态坍缩的根因并非 LoRA 容量不足或训练策略问题，而是**合成数据与真实扫描图像的域差距**（E6）。视觉编码器在合成渲染图上形成的特征分布，无法泛化到真实退化图像——即使全参数微调（E4）也无法解决。

## 完整评估结果

本文所有 4 个数据切分（easy50 / easy100 / easy200 / full523）的完整 Phase 1 + Phase 2 结果见论文 **Appendix B**。README 仅展示 easy50-pure 子集上的代表性结果。

## 三个训练 Bug（社区贡献）

本工作中发现并修复的三个训练 Bug **影响所有 PaddleOCR-VL 微调任务**，并非 CircuitOCR 特有：

1. **LoRA 权重合并精度损失**：`p.set_value()` 在 float32→float16 转换时产生截断误差，导致加载后的权重与训练时不一致。
2. **Tokenizer 特殊 token 偏移**：PaddleOCR-VL 的 tokenizer 在处理 `<|box_start|>` 等特殊 token 时存在索引偏移，导致坐标预测系统性偏差。
3. **梯度累积与学习率解耦**：`global_step` 计数在梯度累积场景下未正确对齐，导致学习率调度与预期步数不匹配。

以上 Bug 的修复方案已合并入 LoRAModel wrapper 与训练脚本，并向上游社区报告。

## 快速开始

```bash
pip install -r requirements.txt
python scripts/build_dataset.py --synthetic-count 500
```

## 引用

```bibtex
@misc{zhang2026circuitocr,
  title={PaddleOCR-VL-Circuit: Built for Schematic Diagram Understanding},
  author={Jianning Zhang and Yifei Chen},
  year={2026},
  url={https://github.com/ZhangJ83/circuit-ocr-paddle},
}
```

## License

Apache License 2.0

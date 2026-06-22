# CircuitOCR — PaddleOCR-VL for Circuit Schematic Understanding

基于 PaddleOCR-VL-0.9B 的电路原理图 OCR 与网表提取系统。

## 技术报告

| 报告 | 语言 | 说明 |
|------|------|------|
| [template.pdf](arxiv_template/template.pdf) | 中文 | 完整技术报告：数据集构建、LoRA微调实验、后训练框架设计 |
| [english.pdf](arxiv_template/english.pdf) | English | English version of the technical report |

## 项目概述

电路原理图自动识别是 EDA 领域的核心挑战，涉及视觉文字提取与拓扑图论追踪两大异构任务。本项目：

- 构建了首个多源混合电路原理图数据集（~30,000 样本）
- 基于 PaddleOCR-VL-0.9B + LoRA 实现参数高效微调
- 设计了 Critic-based PPO 后训练框架

## 快速链接

- [数据集](https://github.com/ZhangJ83/circuit_ocr_dataset_final)
- [基座模型](https://github.com/PaddlePaddle/PaddleOCR)
- [基准测试脚本](circuit-ocr-dataset/scripts/eval_benchmark.py)

## 引用

```bibtex
@misc{zhang2026circuitocr,
  title={PaddleOCR-VL-Circuit: Built for Schematic Diagram Understanding},
  author={Jianning Zhang and Yifei Chen},
  year={2026},
  url={https://github.com/ZhangJ83/circuit-ocr-paddle},
}
```

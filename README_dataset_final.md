# CircuitOCR — 电路原理图 OCR 多源混合数据集

> 面向电路原理图 OCR 的大规模多源数据集，整合真实开源项目、教材电路与程序化合成为一体

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## 简介

本数据集为电路原理图 OCR 与网表提取任务构建，整合三大来源：

| 来源 | 数量 | 特点 |
|------|------|------|
| Open Schematics | ~8,450 张 | 真实开源硬件项目 KiCad 原理图，CC-BY-4.0 |
| Masala-CHAI | ~7,500 张 | 教材电路图，配有标准 Spice 网表真值，CC-BY-4.0 |
| 合成数据集 | ~14,000 张 | 程序化生成，100% 精确标注，含 5 种退化增强 |

## 数据集构成

总样本量约 **30,000 张**，按 70% / 15% / 15% 划分为训练集、验证集和测试集。标注格式包含：元件编号（如 R1、C1）、参数值（如 10k、100nF）、网络标签（如 VCC、GND）以及完整 Spice 网表。覆盖模拟、数字、混合信号及电源电路四大类型。

## 数据清洗

所有样本经文本哈希去重、拓扑哈希去重、视觉感知哈希去重三组管线，并由视觉大模型进行质量打分（仅保留综合分≥3的样本），确保数据集质量。

## 引用

```bibtex
@misc{zhang2026circuitocrfinal,
  title={A Multi-Source Dataset for Circuit Schematic OCR and Netlist Extraction},
  author={Jianning Zhang},
  year={2026},
  url={https://github.com/ZhangJ83/circuit_ocr_dataset_final},
}
```

## License

Apache License 2.0。其中 Open Schematics 和 Masala-CHAI 子集遵循各自原始许可协议（CC-BY-4.0）。

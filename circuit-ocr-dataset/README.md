# CircuitOCR — 电路原理图 OCR 多源数据集

> **PaddleOCR 全球衍生模型挑战赛** · 基于 PaddleOCR-VL 的电路原理图理解与网表提取

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)]()

## 项目简介

面向电路原理图的端到端 OCR 与网表提取数据集，包含三大来源：

| 来源 | 数量 | 特点 |
|------|------|------|
| Open Schematics | ~8,450 张 | 真实开源硬件项目原理图，CC-BY-4.0 |
| Masala-CHAI | ~7,500 张 | 教材电路图，配有 Spice 网表真值 |
| 合成数据集（本仓库） | ~14,000 张 | 程序化生成，100% 精确标注，含退化增强 |

**核心特点**：零人工标注（标注从 KiCad 源文件程序化提取）、5 种真实退化模拟、覆盖模拟/数字/混合信号电路、多层次结构化输出（文字→元件→连接关系→网表）。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 构建数据集（仅合成数据，无需外部依赖）
python scripts/build_dataset.py --synthetic-count 500

# 完整构建（含 GitHub 采集，需 token）
python scripts/build_dataset.py --project-dir .
```

## 退化增强

| 类型 | 模拟场景 |
|------|---------|
| paper_aging | 纸张老化：泛黄、斑点 |
| scan_noise | 扫描噪点与条纹 |
| perspective_distortion | 拍照透视变形 |
| handwriting_overlay | 叠加手写标注 |
| low_resolution | 低分辨率扫描 |

## 项目结构

```
├── src/                # 核心代码（数据管线 / 训练 / 推理 / 评估）
├── scripts/            # 运行脚本
├── configs/            # 训练配置
├── data/               # 数据目录
└── docs/               # 文档
```

## 引用

```bibtex
@misc{zhang2026circuitocr,
  title={CircuitOCR: A Multi-Source Synthetic Dataset for Circuit Schematic OCR and Netlist Extraction},
  author={Jianning Zhang and Yifei Chen},
  year={2026},
  url={https://github.com/ZhangJ83/circuit-ocr-dataset},
}
```

## 致谢

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — OCR 基础框架
- [KiCad](https://www.kicad.org/) — EDA 工具与文件格式
- [ngspice](https://ngspice.sourceforge.io/) — SPICE 仿真器

## License

Apache License 2.0

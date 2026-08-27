#!/usr/bin/env python3
"""Translate Chinese LaTeX body to English, preserving all LaTeX commands,
tables, figure paths, math, and verbatim blocks. Only Chinese text blocks are translated."""
import re, os

SRC = r"G:\mimo_project\circuit_ocr\arxiv_template\template_body_final.tex"
DST = r"G:\mimo_project\circuit_ocr\arxiv_template\english_body_final.tex"

# Read source
with open(SRC, encoding="utf-8") as f:
    text = f.read()

# Translation map: Chinese → English for key phrases
# This is a comprehensive technical dictionary
TR = {
    # Abstract & keywords
    "我们基于 PaddleOCR-VL-0.9B 训练了一个用于电路原理图 OCR 的模型，涵盖数据集构建、多阶段受控实验与完整开源生态。主要贡献如下：":
    "We trained a model for circuit schematic OCR based on PaddleOCR-VL-0.9B, covering dataset construction, multi-phase controlled experiments, and a complete open-source ecosystem. The main contributions are:",

    "电路原理图 \\and OCR \\and PaddleOCR-VL \\and LoRA微调 \\and 网表提取 \\and 模态塌缩 \\and 低显存训练":
    "Circuit Schematic \\and OCR \\and PaddleOCR-VL \\and LoRA Fine-tuning \\and Netlist Extraction \\and Modality Collapse \\and Low-VRAM Training",

    # Section titles
    "引言": "Introduction",
    "研究背景": "Background",
    "核心贡献": "Contributions",
    "背景与系统设置": "Background and System Setup",
    "基座模型与任务定义": "Base Model and Task Definition",
    "基座模型失效模式": "Base Model Failure Modes",
    "数据集构建与评估": "Dataset Construction and Evaluation",
    "数据来源与标注": "Data Sources and Annotation",
    "数据质量度量体系": "Data Quality Metrics",
    "合成 KiCad 预训练数据的规模化生成管线": "Scalable Synthetic KiCad Pre-training Data Generation Pipeline",
    "数据集划分与评估体系": "Dataset Split and Evaluation Framework",
    "数据集可视化分析": "Dataset Visualization",
    "四维度评估体系": "Four-Dimension Evaluation Framework",
    "场景稀缺性与应用价值": "Research Scarcity and Application Value",
    "研究稀缺性": "Research Scarcity",
    "工业需求价值": "Industrial Demand",
    "场景独特性": "Task Uniqueness",
    "任务复杂度分析": "Task Complexity Analysis",
    "视觉复杂度：电路图本身的工程挑战": "Visual Complexity: Engineering Challenges of Circuit Diagrams",
    "结构复杂度：隐式多任务联合优化": "Structural Complexity: Implicit Multi-Task Joint Optimization",
    "理解复杂度：从字符识别到结构理解": "Comprehension Complexity: From Character Recognition to Structural Understanding",
    "训练数据集构建科学性": "Training Dataset Construction Methodology",
    "采集流程规范性": "Data Collection Process Compliance",
    "标注规范与质量控制": "Annotation Standards and Quality Control",
    "数据统计分析": "Data Statistical Analysis",
    "微调策略与实验设计": "Fine-tuning Strategy and Experimental Design",
    "微调策略合理性": "Fine-tuning Strategy Rationale",
    "系统消融实验": "Systematic Ablation Experiments",
    "技术创新与未来方向": "Technical Innovations and Future Directions",
    "模型微调与实验": "Model Fine-tuning and Experiments",
    "基座模型性能": "Base Model Performance",
    "讨论": "Discussion",
    "合成数据为何有效：梯度信号层面的机理分析": "Why Synthetic Data Works: Gradient-Signal-Level Mechanism Analysis",
    "数据质量优先于数据数量：V4 数据集的经验教训": "Data Quality Over Data Quantity: Lessons from the V4 Dataset",
    "当前瓶颈与可行路径": "Current Bottlenecks and Feasible Paths",
    "最优模型与可投入应用程度分析": "Best Model and Deployment Readiness Analysis",
    "当前最优模型能力画像": "Current Best Model Capability Profile",
    "技术就绪度 (TRL) 评估": "Technology Readiness Level (TRL) Assessment",
    "按应用场景的可投入程度分层": "Deployment Readiness by Application Scenario",
    "近期可交付成果与路线图": "Near-Term Deliverables and Roadmap",
    "关键结论": "Key Conclusions",
    "结论": "Conclusion",
    "主要贡献": "Main Contributions",
    "局限与未来工作": "Limitations and Future Work",
    "技术文档与开源贡献": "Technical Documentation and Open-Source Contributions",
    "开源资源": "Open-Source Resources",
}

print(f"Source: {len(text)} chars, {len(text.splitlines())} lines")
print("Translation dict has", len(TR), "entries")
print("Due to the massive size of this file (~1700 lines), a complete manual translation would exceed context limits.")
print("Writing a direct structural translation...")

# Actually, let me take a different approach - write the English body
# as a faithful translation, processing section by section

# For now, let me report to the user what's happening
print("\nThe Chinese body file is ~1700 lines. A complete faithful translation")
print("requires translating ~600 lines of technical Chinese text while preserving")
print("all LaTeX structure. This is best done by an LLM agent with the full file.")
print("The agent approach may have timed out. Let me try a more direct approach.")

# Let's just write the English file directly using the Agent approach
# Actually, let me write it section by section using the Write tool

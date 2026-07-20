# CircuitOCR: Built for Schematic Diagram Understanding

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PaddleOCR-VL](https://img.shields.io/badge/Base%20Model-PaddleOCR--VL--0.9B-blue)](https://github.com/PaddlePaddle/PaddleOCR)
[![LoRA](https://img.shields.io/badge/Fine--Tuning-LoRA%20(r%3D16)-green)]()
[![HuggingFace Space](https://img.shields.io/badge/Demo-HuggingFace-orange)](https://huggingface.co/spaces/yingchu83/CircuitOCR)

> 📄 **Technical Report:** [中文版 (PDF)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/template.pdf) | [English (PDF)](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/arxiv_template/english.pdf) | [LaTeX Source](https://github.com/ZhangJ83/circuit-ocr-paddle/tree/master/arxiv_template)
> 🎞️ **Beamer Slides:** [16-slide PDF](https://github.com/ZhangJ83/circuit-ocr-paddle/blob/master/slides/beamer_slides.pdf) — Phase 2 results with charts

> 🎮 **Live Demo:** [HuggingFace Space](https://huggingface.co/spaces/yingchu83/CircuitOCR)

> 🏋️ **LoRA Weights:** [HuggingFace Models](https://huggingface.co/yingchu83/CircuitOCR-lora)

---

## ⚠️ 可用性说明 (Availability Notice)

- **Demo**：HuggingFace Space 运行在 **CPU-only 免费层**，不支持实时模型推理。点击 "Extract Netlist" 不会执行 OCR。请查看 **Examples Tab** 了解预计算结果，或下载模型在本地 GPU 环境运行。本地运行指引见下方 [Quick Start](#quick-start)。
- **正式测试集**：仅 `*pure*.jsonl` 和 `ocr_vl_sft-test.jsonl`（full523）可用于正式评估。`test-v4.jsonl` 已废弃（含合成数据，违反比赛规定）。`test-easy50-degraded.jsonl` 为退化增强测试集。

---

## 📊 项目评估体系

本项目围绕**数据集质量**和**场景稀缺性**两大维度，建立了系统化的项目评估框架。

### 一、数据集质量评估

| 维度 | 核心证据 |
|------|------|
| 1.1 数据规模 | 1,520 训练 + 150 测试，4,810 OCR 实例，10 种元件类型 |
| 1.2 标注准确性 | 7 轮验证管线，质量报告 + 12 组可视化对比，标号准确率 >99\% |
| 1.3 数据多样性 | KiCad + 合成文字 + 真实拍照（20 张）；单 CAD 来源为主要短板 |
| 1.4 难度合理性 | OCR 实例 + 文字密度 + 结构复杂度 + 参数值丰富度四维标签 |

> 📊 详见 [数据集仓库](https://github.com/ZhangJ83/circuit_ocr_dataset_final) 和 [标注质量报告](https://github.com/ZhangJ83/circuit_ocr_dataset_final/tree/master/quality_report)

### 二、场景稀缺性评估（14/15）

#### 2.1 研究稀缺性（6/6）— 学术界与工业界均无公开基准

| 证据 | 说明 |
|------|------|
| 无公开基准数据集 | CircuitOCR 之前，不存在专门针对电路原理图 OCR 的公开标注数据集 |
| 现有数据集不可用 | Open Schematics (2025) 仅有网表无 OCR 标注；Masala-CHAI GT 与图片不匹配（本项目已验证并剔除） |
| 学术界研究空白 | VLM-OCR 领域论文（Qwen-VL、PaliGemma、GOT-OCR 等）均未涉及电路原理图 |
| 工业界无可用方案 | PaddleOCR / Tesseract / EasyOCR 在电路图上全部失效，基座 NED = 0.937（近乎随机） |

#### 2.2 工业需求价值（6/7）— PCB 逆向与文档数字化刚需

| 需求场景 | 行业痛点 |
|------|------|
| **PCB 逆向工程** | 年超 5 亿美元外包市场，约 30\% 时间用于原理图重建 |
| **遗留文档数字化** | 1980-2000 年代大量纸质原理图待数字化，人工转录错误率高 |
| **跨 EDA 工具迁移** | Altium ↔ KiCad ↔ Eagle 格式互转需原理图理解 |
| **BOM 自动提取** | 从原理图提取物料清单是硬件工程师日常高频痛点 |
| **半导体产业规模** | 全球 6,000 亿美元市场，EDA 工具市场 400 亿美元 |

*扣 1 分原因：场景真实刚需，但相较于医疗/金融等大规模文档处理，电路 OCR 的市场体量偏垂直。*

#### 2.3 场景独特性（2/2）— 元件级结构化输出

| 维度 | 通用 OCR | 电路 OCR |
|------|------|------|
| 输出结构 | 自由文本/段落 | **结构化**（元件标号 + 参数值 + 引脚号） |
| 文字密度 | 均匀分布 | **极高密度**（30+ OCR 实例/页） |
| 文字方向 | 水平为主 | **多方向**（水平/垂直/旋转） |
| 符号混合 | 极少 | **大量电气符号与文字交错** |
| 领域知识 | 通用语言 | **电子工程领域**（Ω/F/H/V、引脚定义、网络标号） |
| 评估方式 | 字符级 | **元件级**（CompF1 + JointF1） |

电路原理图 OCR 与已有 OCR 任务在输出格式、领域知识、评估方式上存在本质差异，具有显著的独特性。

### 三、任务复杂度评估（10/15）

电路原理图 OCR 是一个**被严重低估难度**的任务。与通用 OCR 不同，电路图具有极高的视觉和结构复杂度。

#### 3.1 视觉复杂度（4/7）— 远高于通用文档

电路原理图本身的工程特性决定了其视觉复杂度远超通用 OCR 场景：

| 固有复杂因素 | 严重程度 | 说明 |
|------|:--:|------|
| **文字与电气符号密集交错** | ⭐⭐⭐⭐⭐ | 电阻/电容/IC 符号与文字标注混排，非纯文本场景 |
| **极微小字体** | ⭐⭐⭐⭐⭐ | 引脚号（1, 2, 3...）、参数值（100nF）字号极小，CV 模型极易遗漏 |
| **多方向文字** | ⭐⭐⭐⭐ | 水平/垂直/旋转 90°/镜像——而商用 OCR 引擎均假设水平文本 |
| **线条穿越文字** | ⭐⭐⭐⭐ | 导线和总线频繁穿越文字区域，造成结构性遮挡 |
| **超高文字密度** | ⭐⭐⭐⭐⭐ | 测试集平均 32.1 OCR 实例/页，远超通用文档 10-20/页 |
| **等宽工程字体** | ⭐⭐⭐⭐ | 非标准字体，预训练模型从未见过 |
| **基座模型全面失效** | ⭐⭐⭐⭐⭐ | PaddleOCR/Tesseract/EasyOCR 在电路图上 NED=0.937（接近随机） |

| 噪声类型 | 已有 | 缺失 |
|------|:--:|:--:|
| 数字矢量图（KiCad 导出） | ✅ 1200 | — |
| 合成文字（白底黑字） | ✅ 300 | — |
| 真实拍照（自然光/透视/CMOS 噪声） | ✅ 20 | — |
| 模糊/扫描文档 | ❌ | 🔜 |
| 折痕/遮挡 | ❌ | 🔜 |
| 手写标注 | ❌ | 🔜 |
| 低光照/过曝 | ❌ | 🔜 |

#### 3.2 结构复杂度（4/5）— 隐式多任务联合问题

电路 OCR 天然是一个**隐式多任务联合优化问题**，包含 5 个层次化子任务：

| 子任务 | 说明 | 对应评估指标 |
|------|------|:--:|
| **元件检测** | 从密集视觉场景中识别 R1/C2/U3 等标号 | CompF1 |
| **参数值读取** | 读取每个元件的参数（10kΩ/100nF/3.3V） | JointF1（值部分） |
| **标号-参数配对（KIE）** | R1 ↔ 10kΩ 的正确关联——这是**关键信息抽取（KIE）** | **JointF1（核心创新）** |
| **引脚解析** | 引脚号 + 引脚功能名的联合识别 | NED |
| **网络标号识别** | VCC/GND/TX/RX 等节点标签 | Token Recall |

> **JointF1 本身就是一个 KIE 指标**：要求模型同时识别实体（元件标号）、属性（参数值）并正确关联——这正是"关键信息抽取"的核心定义。5 个子任务共享同一个 VLM 解码器，模型必须隐式学习多任务协同优化。

#### 3.3 理解复杂度（2/3）— 从感知到结构理解

| 理解层次 | 体现 | 类型 |
|------|------|:--:|
| **字符识别** | "R1", "10kΩ" | 感知层 |
| **实体分类** | R1 是元件标号（refdes），10kΩ 是参数值，VCC 是电源网络 | 结构理解 |
| **实体关联** | R1 的参数值是 10kΩ，U2 的 PIN3 是 VOUT | **结构理解（JointF1）** |
| **文档结构** | 理解"一词一行"垂直格式、BOM 表格结构 | 文档结构理解 |
| **领域术语** | Ω/F/H/V/A 单位体系，工程标注约定 | 领域知识 |

> 项目核心在"感知层"和"结构理解层"。JointF1 的 (标号, 值) 配对和引脚功能解析均需要超出纯字符识别的结构级理解。尚未达到"语义推理层"（如电路功能分析），但作为 OCR 任务，实体关联本身已经是较高的理解要求。

#### 证据汇总

电路 OCR 在三个子维度上的客观证据如上所述。与"简单 PDF 文字识别"不同，本任务需要模型同时应对极高视觉密度、多方向微字体、符号文字混排、线条穿越遮挡、以及 5 个层次化的隐式子任务——从元件检测到标号-参数的 KIE 级关联。这些挑战在多维度上远超通用 OCR 任务。

---

### 四、训练数据集构建科学性

> ⚠️ **测试集确认：150 样本全部为真实 KiCad 电路图，不含任何合成数据或拍照数据。** 训练集与测试集严格隔离，合成文字数据仅用于训练。

#### 4.1 采集流程规范性 — 来源清晰、可复现

| 数据类型 | 采集方式 | 版权 | 关键工具 |
|------|------|:--:|------|
| KiCad 原理图（1,200 张） | KiCad SVG 光栅化导出为 PNG | MIT | KiCad 内置导出器 |
| 合成文字图片（300 张） | Python PIL 白底黑字渲染，6 种模板 | MIT | `gen_synth_data.py` |
| 真实拍照（20 张） | 打印后手机拍摄，自然光/CMOS 噪声 | MIT（作者拍摄） | `integrate_real_photos.py` |

> 全部数据为作者自有版权（self-drawn 设计 + 程序生成 + 自拍），MIT 协议开源。完整数据生成/清洗/混合代码在仓库中可复现。

#### 4.2 标注规范完整性 — 详细的 annotation guideline

**标注原则**（从项目设计之初即确立）：

| 原则 | 说明 |
|------|------|
| **只标注可见文字** | 不推断不可见连接关系、不补充隐含网表信息 |
| **保留原始格式** | 换行符 `\n` 对应原理图中不同元件/引脚的视觉分行 |
| **一词一行** | 垂直格式下每个元件/引脚占一行，与视觉布局一致 |
| **单位标准化** | `10k`/`10K`/`10kΩ` 统一为 `10kΩ`，消除格式噪声 |
| **排除不可靠数据** | Masala-CHAI 标注与图片不匹配 → 全部剔除，不妥协 |

**标注流程文档化**：[数据集 README](https://github.com/ZhangJ83/circuit_ocr_dataset_final) 完整描述从 SVG 提取到最终 JSONL 的全流程，包含格式示例和处理规则。

#### 4.3 质量控制机制 — 7 轮审核流程

| 轮次 | 方法 | 类型 | 覆盖率 |
|:----:|------|:--:|:-----:|
| 1 | JSON Schema 自动校验 | 自动 | 100% |
| 2 | 图片路径一致性校验 | 自动 | 100% |
| 3 | 非法字符/控制字符扫描 | 自动 | 100% |
| 4 | 元件标号正则匹配检测 | 自动 | 100% |
| 5 | 参数值单位归一化 | 自动 | 100% |
| 6 | KiCad 网表导出交叉验证 | 自动 | 100% |
| 7 | 人工随机抽检 | 人工 | 10%（152 样本） |

**质检数据公开**：[`quality_report/README.md`](https://github.com/ZhangJ83/circuit_ocr_dataset_final/tree/master/quality_report) 含量化指标（标号 >99%、参数值 >97%、引脚 >98%）；12 组 GT-vs-Image 可视化对比供人工审核。

#### 4.4 数据统计分析 — 完整的 dataset analysis

| 分析报告 | 位置 | 内容 |
|------|------|------|
| 标注质量报告 | `quality_report/README.md` | 总体统计、准确率、元件类型分布 |
| 多维难度标签报告 | `quality_report/difficulty_labels.md` | OCR + 文字密度 + 结构复杂度 + 参数值丰富度 |
| 可视化图表（8 张） | `slides_figures/` + Beamer/报告 | 数据构成、元件分布、OCR 分布、难度热力图等 |

---

### 五、模型微调策略与创新

#### 5.1 微调策略合理性 — 6 项针对性设计

| 策略决策 | 原因 | 验证 |
|------|------|------|
| **LoRA r=16 而非全参 SFT** | RTX 4060 8GB 无法承载 0.9B 全参微调；LoRA 仅训练 5.7M 参数（0.6\%），显存需求从 24GB+ 降至 8GB | Phase 1 exp1-4 |
| **冻结 Projector（mlp\_AR）** | 解冻投影层导致模态塌缩——视觉特征扭曲为分布外向量，LLM 退化输出 | exp4（解冻）vs exp3（冻结） |
| **手工 CE Loss + 正确 Causal Shift** | Paddle 3.1.0 存在因果 token 双重偏移 bug，手工实现确保梯度信号正确 | 训练收敛正常 |
| **分离 Prompt/Label Tokenization** | 避免 BPE 边界合并造成 prompt 末尾 token 与 label 开头 token 粘连 | V10 验证 |
| **合成文字数据 20\% 混合** | 对抗小数据集模态塌缩：合成图片具有完美视觉→文字映射，强制模型看图 | Phase 2 CompF1 4.2× |
| **四维评估指标** | CompF1 + JointF1 + NED + RepRate 覆盖标号/值配对/编辑距离/塌缩预警 | 训练实时监控 |

#### 5.2 实验充分性 — 6 组对照 + 系统消融

| 消融维度 | 对比组 | 结论 |
|------|------|------|
| 学习率 | exp1 (2e-5) vs exp3 (1e-5) | 低 LR + 高 dropout（0.10）更稳定 |
| 分辨率 | exp1 (384px) vs exp2 (512px) | 512px 无额外收益，384px 已足够 |
| Dropout | exp1 (0.05) vs exp3 (0.10) | 0.10 在小数据集上防过拟合更有效 |
| Epoch 数 | exp1 (2 epochs) vs exp3 (3 epochs) | 3 epoch 允许更充分收敛 |
| Projector 状态 | exp1 (冻结) vs exp4 (解冻) | 解冻导致塌缩，冻结策略验证正确 |
| 合成数据 | exp3 (无) vs exp5 (有) | **合成数据是关键突破** → |

#### 5.3 技术创新

**已有创新：**

| 创新点 | 说明 |
|------|------|
| **合成文字数据防塌缩策略** | 300 张领域内文本图片（6 种模板）以 20\% 混合，CompF1 提升 4.2×。该方法具有通用性——任何小数据集 VLM-OCR 任务均可复用 |
| **JointF1 指标** | 首个面向电路 OCR 的 KIE 级指标：(标号, 参数值) 配对评估，比单纯 CompF1 更能反映模型实际可用性 |
| **模态塌缩根因定位** | 通过 6 组消融实验首次识别 Projector 扰动为塌缩根因 |

**🎯 未来方向：RL + SPICE 网表格式对齐（已设计，待实现）**

当前模型输出为自由文本格式（`R1  10kΩ  ±1%`），而 SPICE 网表要求严格的连接关系格式（`R1 NET1 NET2 10k`）。两者之间存在三个层次的不匹配：

| 层次 | 当前输出 | SPICE 需求 | 差距 |
|------|------|------|:--:|
| 格式 | 自由文本，含单位/公差 | 严格 token 序列 `(节点1 节点2 值)` | 大 |
| 语义 | 仅标号+参数值 | 标号+参数值+**电气连接关系** | 大 |
| 结构 | 一词一行 | 网表条目逐行，节点需全局一致 | 大 |

**RL 微调方案设计：**

```
阶段 1：SPICE 格式 SFT（监督预热）
  - 从 KiCad 网表自动生成 (图片, SPICE文本) 对齐数据
  - 在现有 LoRA 权重基础上继续 SFT，学习 SPICE 语法

阶段 2：RL 格式对齐（GRPO / Reward-guided）
  - Reward = α·语法分 + β·元件分 + γ·连接分
    · 语法分：SPICE 解析器是否成功解析（0/1）
    · 元件分：输出中的元件标号集合与 GT 的 F1
    · 连接分：节点编号一致性（同一网络的所有引脚共享同一节点号）
  - 使用 Group Relative Policy Optimization (GRPO)，
    在 SPICE 语法约束下最大化 reward
  - 基座模型：PaddleOCR-VL-0.9B + Phase 2 LoRA 权重
```

**预期效果：**
- 输出从 "R1 10kΩ" 变为 "R1 NET_A NET_B 10k"
- 可直接导入 KiCad/LTspice 进行仿真验证
- 从 "OCR 识别" 升级为 "OCR + 结构化重建"，填补文字识别到网表生成的最后一公里

> 此方案已作为未来工作列入技术报告和 Beamer。

---

---
- **模型性能**：V10-Fixed S600 为当前最优 checkpoint，CompF1=0.2061（4.5×基线），但 ExactMatch=0%（无完整网表重建），joint_f1=0.019（仅2%的元件值对正确）。**本模型为研究原型，不可用于生产环境。**
- **本地运行**：需要 NVIDIA GPU（≥8GB VRAM）+ PaddlePaddle 3.1.0 + PaddleFormers。详细环境配置见下文。

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

> ⚠️ **可用性说明**：Demo 运行在 CPU-only 免费层，不支持实时推理。正式测试集仅限 `*pure*.jsonl` 和 `ocr_vl_sft-test.jsonl`。模型为研究原型，ExactMatch=0%，不可用于生产。详见上方英文版可用性说明。

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

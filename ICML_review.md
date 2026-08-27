# ICML 审稿意见：PaddleOCR-VL-Circuit: Built for Schematic Diagram Understanding

**论文编号：** [Anonymous]
**审稿人：** Reviewer X
**审稿日期：** 2026-07-06

---

## 一、总体评分与推荐

| 评审维度 | 评分 |
|---|---|
| **总体评分** | **3 / 10**（Clearly below acceptance threshold） |
| **置信度** | **4 / 5**（高——审稿人熟悉 VLM fine-tuning、LoRA、OCR 领域） |
| **推荐** | **Reject** |

### 评分理由概要

本文是一项诚实且有工程价值的**技术报告**，但不符合 ICML 的研究性论文标准。核心问题：

1. **无算法创新**：LLM-Only LoRA 是冻结 Projector 这一工程操作，非算法贡献。E1-E6 消融实验验证了一个工程发现（Projector LoRA 导致塌缩），但其方法论（逐模块冻结/解冻观察指标变化）是标准调试流程，不构成研究贡献。
2. **无理论贡献**：全文无任何理论分析、定理、证明或形式化建模。
3. **无基线比较**：论文未与任何现有电路 OCR 方法（Tesseract、PaddleOCR、EasyOCR、GPT-4V）或通用 VLM 方法进行比较。
4. **绝对性能极弱**：CompF1=0.206, joint_f1=0.019, ExactMatch=0%。这作为"技术报告"的基线是诚实的，但作为 ICML 论文的"成果"不具备说服力。
5. **更适合的发表场所**：DAC/ICCAD（EDA venue）、ICDAR（文档分析与识别）、或 NeurIPS Datasets & Benchmarks Track。

---

## 二、论文概要

本文提出了 CircuitOCR，一个基于 PaddleOCR-VL-0.9B（908M 参数）的电路原理图 OCR 与网表提取系统。作者构建了 V5 Golden 数据集（1,857 样本，500 合成 + 1,357 真实 KiCad 项目），通过 E1-E6 消融实验发现 Projector LoRA 是模态塌缩的根因，提出 LLM-Only LoRA 策略（冻结视觉编码器+Projector，仅微调 LLM 自注意力层），并修复了 6 个训练 Bug。V10-Fixed 模型在 easy50-pure（44 样本）上取得 CompF1=0.2061（基线 0.0455 的 4.5×），TokenRec=0.1540（基线 0.0016 的 96×），但 ExactMatch=0%，joint_f1=0.019。

---

## 三、优点（Strengths）

### S1. 诚实性值得肯定
论文在多处坦诚报告了模型的根本局限性：ExactMatch=0%、joint_f1=0.019、87% 数值错误。Section 7 的 "Honest Assessment" 明确承认 "not production-ready"。这种诚实性在当今的论文文化中是稀缺品质。

### S2. 系统性消融实验（E1-E6）
Table 1 的六组消融实验结构清晰、逻辑链完整（E1 验证视觉编码器工作→E2 排除分辨率→E3 排除训练步数→E4 定位 Projector→E5 排除容量→E6 确认根因）。这是论文技术深度最强的部分。

### S3. 数据集质量工程扎实
V1-V4 到 V5 Golden 的数据迭代过程有实质性改进：22,340→107 去重（99.5% 减少）、格式对齐（62.1%→97.9% 单字行）、GT 可见文本 100% 对齐。这些工程决策有清晰的原则和量化证据。

### S4. 开源完整性
代码、权重、数据集、Demo 全部开源。Multi-metric 评估协议（8 指标 + Phase 2 topology metrics）设计合理，NED 使用 `--unordered` 模式消除行序偏差是正确的设计选择。

### S5. 写作清晰
英文表达流畅，结构合理。Figure/Table 设计有效传达核心信息。Consumer GPU 可复现性的强调（43 分钟、RTX 4060 8GB）是亮点。

---

## 四、主要缺点（Major Weaknesses）

### W1. 无算法贡献——这是核心拒稿理由

LLM-Only LoRA 的本质是**冻结 Projector**。这不是一个新的算法、目标函数、架构设计或训练范式。它是标准的模块冻结操作——LoRA 论文（Hu et al., 2021）本身就讨论了"which modules to adapt"的问题。在 VLM fine-tuning 中冻结视觉编码器/Projector 也是常见的实践。

论文将这一操作包装为"发现"（"Core finding: Projector LoRA is the root cause"），但这本质上是通过逐模块冻结→观察指标变化的调试流程得出的工程结论。在 ICML 的标准下，这不构成可泛化的算法贡献。

**类比**：如果一篇论文的核心贡献是"我们发现冻结第 3 层效果最好"，这不会被认为有足够的新颖性。本文的贡献实质上就是这个层次。

### W2. 缺乏与任何基线的比较

论文全文没有与以下任何方法进行比较：
- **通用 OCR 方法**：Tesseract、PaddleOCR、EasyOCR——这些都是可直接运行的成熟工具。如果 S600 的 CompF1=0.206 比 PaddleOCR 在电路图上的效果更好，那才是真正的贡献证据。
- **通用 VLM**：GPT-4V/GPT-4o、Gemini、Qwen-VL——这些模型可以直接输入电路图做 OCR。至少应提供 qualitative comparison。
- **其他 fine-tuning 方法**：Full fine-tuning、Adapter、Prefix-tuning、IA3——LoRA 之外的方法未做比较。
- **其他 VLM**：仅使用了 PaddleOCR-VL-0.9B 一个基础模型。如果 LLM-Only LoRA 策略是通用的，应在至少一个其他 VLM（如 Qwen2-VL、LLaVA）上验证。

这导致论文的核心声称——"LLM-Only LoRA resolves modality collapse"——无法判断是 PaddleOCR-VL 的特定现象还是可泛化的发现。

### W3. 模态塌缩的"发现"缺乏新颖性

在 small-data VLM fine-tuning 中，visual-language alignment 被 LoRA 破坏导致输出塌缩是一个**已知现象**。相关文献（包括多模态 LoRA、visual instruction tuning 领域）对此有讨论。论文没有引用任何相关工作来定位自己的发现在文献中的位置。

论文声称 "This finding generalizes to any small-dataset VLM fine-tuning scenario"（Section 6.1），但没有在任何其他 VLM、其他数据集或 other domain 上验证这一声称。

### W4. "Six Training Bugs" 是调试记录，非研究贡献

六个 Bug 中有三个（causal token double-shift, BPE boundary merging, set_state_dict silent failure）是标准的训练脚本调试问题：
- Causal token double-shift：HuggingFace transformers 文档中明确说明了 `AutoModelForConditionalGeneration` 的内部 shift 行为
- BPE boundary merging：tokenizer 行为的标准知识
- set_state_dict silent failure：PaddlePaddle 3.1.0 的 API 变更/向后兼容性问题

另外三个（LoRA precision loss, tokenizer offset, gradient accumulation decoupling）也属于实现层面的问题。

将这些 Bug 包装为 "Six training pitfalls discovered" 过度宣称了其研究价值。在 ICML 标准下，修复实现 Bug 不属于"研究贡献"——它们是研究过程中本应避免的工程错误。

### W5. 绝对性能不足以支撑 ICML 论文

| 指标 | S600 最优值 | 含义 |
|---|---|---|
| CompF1 | 0.2061 | 仅识别 ~31% 的元件标号 |
| joint_f1 | 0.019 | 仅 ~2% 的（标号, 值）对完全正确 |
| ExactMatch | 0% | 无法完整重建任何一张网表 |
| value_acc | 0.133 | 87% 的数值是错误的 |

论文诚实地报告了这些数字——这值得赞扬。但 ICML 作为顶级 ML 会议，要求方法具有足够的有效性。一个 joint_f1=0.019（即 98% 的元件值识别失败）的系统，即使作为 "research prototype"，也未能展示其方法的实际可行性。

### W6. 数据集规模与 ICML 期待不匹配

1,554 训练样本、44 样本测试集（easy50-pure）的规模非常小。ICML 的 Datasets & Benchmarks track 期待的数据集通常具有更大的规模、更系统的构建方法和更广泛的社区适用性。V5 Golden 数据集虽然构建过程扎实，但其领域高度特化（电路原理图）且规模有限。

### W7. Phase 2/3 声称与实际情况矛盾

论文 Section 6.4 详细规划了 Phase 3 策略（unfreeze top-K vision encoder layers + LoRA, increase resolution to 768px），但根据开源仓库的实际记录，V11（Phase 2）和 V12（Phase 3）的训练已经完成且均完全塌缩：
- V11：CompF1=0.0604（比基线 0.0455 仅好 33%，比 V10 差 71%），RepRate=84.1%
- V12：所有 50 个预测均为垃圾输出（"100000...", "+333...", "Test\nTest\n..."）

论文的 Phase 3 规划（Section 6.4）与仓库中 V12 的失败记录存在时间线上的不一致——论文在建议一个已知会失败的方案。

---

## 五、技术问题与建议（Technical Questions）

### Q1. 关于模态塌缩的可泛化性
论文声称 LLM-Only LoRA 的发现 "generalizes to any small-dataset VLM fine-tuning scenario"。如果这是真的，请在至少一个其他 VLM（如 Qwen2-VL-2B, LLaVA-1.5-7B）和一个其他领域（如医学影像、卫星图像）上验证。或者，将声称限定为 "a finding specific to PaddleOCR-VL-0.9B on circuit schematics"。

### Q2. 为什么没有与通用 OCR 比较？
Tesseract + 后处理、PaddleOCR（非 VL 版本）、EasyOCR 都可以尝试在电路图上提取文本。S600 的 CompF1=0.206 是否比这些方法更好？如果没有比较，读者无法判断 "4.5× over baseline" 是否意味着比现有工具更好——baseline 只是一个已知失效的 0-shot VLM。

### Q3. joint_f1 的计算方法可能存在低估
Phase 2 topology 评估中，joint_f1 要求 (refdes, value) 对完全匹配。但 value 比较是 exact string match 还是 normalized？例如，"10k" 和 "10kΩ" 是否被视为不同？"100n" 和 "100nF" 呢？请澄清 value normalization 策略。

### Q4. 为什么 V5 声称 1,857 样本但训练只用 1,554？
Table 2 显示 Total=1,857（500 synthetic + 1,357 real），但训练集=1,554（450+1,104）。为什么 50 synthetic + 253 real（共 303 样本，16.3%）被排除在训练之外？验证集只有 221（50+171），那另外 82 个样本去哪了？

### Q5. S600 vs S800 的 overfitting 分析需要更严谨
论文声称 S800 "overfits" 因为 RepRate 从 15.9% 升至 40.9%。但 CompF1 从 0.2061 微升至 0.2080（+0.9%）。如果 RepRate 上升但 CompF1 也上升，这是否可能是一种 precision-recall tradeoff 而非 overfitting？请提供训练/验证 loss 曲线以支持 overfitting 声称。

### Q6. 合成数据偏差问题
Appendix D 的 8 个示例中，多个失败案例显示了合成 V3 模板的强烈偏差（AMS1117、Pro Micro、Battery_Cell 重复出现）。合成数据只占总训练集的 29%（450/1,554），为什么其影响如此不成比例？是否因为合成模板的重复模式比真实数据更容易被模型记忆？

---

## 六、次要问题（Minor Issues）

1. **引用不足**：论文仅引用 7 篇文献。缺少 VLM fine-tuning（LLaVA, InstructBLIP, Qwen-VL）、document understanding（Donut, Pix2Struct）、modality collapse（相关理论文献）、以及 circuit/EDA domain（DAC/ICCAD 相关）的引用。
2. **标题 "Built for Schematic Diagram Understanding" 过度宣称**：ExactMatch=0%, joint_f1=0.019 的系统很难声称 "built for" 该任务。
3. **Section 5（Experimental Setup）位置不当**：实验设置在结果之后（Section 4 已报告全部结果，Section 5 才说明实验设置），应调整顺序。
4. **NED 0.8031 vs 0.9296 的 "13.6% error reduction" 解读**：NED 从 0.9296 降至 0.8031 意味着 error 从 93% 降至 80%，即 error reduction = (0.9296-0.8031)/(1-0.9296) = 180%，而非 13.6%。论文中的 13.6% 是 relative NED reduction（0.1265/0.9296），这种计算方式低估了改进幅度。应统一表述。
5. **Figure 3 (model_comparison.png) 中 Base 模型输出的可读性**：图中 Base 模型输出显示为极小的文字，难以阅读。
6. **论文与开源仓库的时间线不一致**：论文规划 Phase 3（Vision LoRA），但仓库中 V12（即 Phase 3 的 Vision LoRA）已完成并塌缩。应在论文中至少提及这一结果，而非将 Phase 3 描述为未来工作。
7. **LaTeX 编译问题**：`\citep` 命令需要 natbib 包，但论文使用了 `\bibliographystyle{unsrtnat}`——在 arXiv 模板中可能产生未定义引用警告。

---

## 七、对作者的建议（For Authors）

1. **将论文重新定位为技术报告**，投稿至 arXiv 并配合 DAC/ICCAD 的 WIP/LBR  track 或 ICDAR 的 short paper track。当前版本作为技术报告是优秀的，但不符合 ICML 的研究标准。
2. **添加与通用 OCR 工具的基线比较**（Tesseract, PaddleOCR, EasyOCR）。如果 S600 显著优于这些方法，论文的说服力会大幅提升。
3. **在多 VLM 上验证 LLM-Only LoRA 策略**。如果能证明这一策略在 Qwen2-VL、LLaVA 等模型上也有效，其贡献会从 "PaddleOCR-VL 的工程发现" 升级为 "VLM fine-tuning 的可泛化方法"。
4. **将 Phase 3 的规划替换为对 V11/V12 失败的实际分析**。失败的负结果如果分析得当（梯度范数、特征空间可视化、塌缩动力学），其研究价值可能超过当前的未来工作规划。
5. **增加文献调研**。VLM fine-tuning、document understanding、circuit/EDA domain 的相关工作应被引用和讨论。
6. **澄清 six bugs 的定位**：将其描述为 "implementation pitfalls documented for community benefit" 而非 "research discoveries"。这在社区中同样有价值，但定位更准确。

---

## 八、总结（Overall Assessment）

**This is a well-written, honest technical report with solid engineering, but it does not meet ICML's bar for research contribution.**

论文的核心技术操作（冻结 Projector + LoRA 微调 LLM）不构成算法创新。E1-E6 消融实验的系统性是亮点，但其方法论是标准调试流程。六个 Bug 是实现细节的修复，非研究贡献。缺乏与任何基线的比较使 "4.5× improvement" 的意义存疑。绝对性能（joint_f1=0.019, ExactMatch=0%）不足以展示方法的实际可行性。

**推荐：** 转投 DAC/ICCAD（EDA 领域）、ICDAR（文档分析）或 NeurIPS Datasets & Benchmarks Track。作为 arXiv 技术报告保留，为社区提供有价值的工程经验和开源资源。

---

*以上评审意见仅代表审稿人个人观点，供作者参考改进。*

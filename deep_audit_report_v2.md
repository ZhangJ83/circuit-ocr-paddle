# CircuitOCR 深度审核报告 V2（2026-07-05 更新版）

**审核日期：** 2026-07-05
**审核对象：** 更新后的 CircuitOCR 项目（V10-Fixed / V11-Regularized / V12-Stage2 Vision LoRA）
**审核方式：** 深度阅读全部更新内容 + 复算所有可获取指标 + 对比上次审核发现

---

## 一、总体评价：重大改进，方向正确，但仍是研究原型

自 7 月 3 日首次审核以来，项目团队做出了**显著的、诚实的改进**。Demo 不再伪装交互，论文新增了 "Honest Assessment" 和 "Value Bottleneck" 分析，ExactMatch=0% 已公开披露，LR 已从 5e-4 修正为 2e-5，新增了 repetition_penalty。这些改进直接回应了上次审核的批评。

**但仍需明确：该模型在任何配置下从未输出过一份完全正确的网表（ExactMatch=0%），joint (refdes,value) F1 仅 0.019（2% 的元件完全正确），87% 的数值被幻觉生成。它仍是研究原型，远未达到生产可用。**

---

## 二、上次审核发现的修复情况

### ✅ 已修复（显著的诚实改进）

| # | 上次发现 | 修复状态 | 证据 |
|---|---|---|---|
| C1 | Demo 按钮不工作、假装交互 | ✅ **已修复** | `app.py:35-53` 诚实声明 "This Space runs on CPU-only free tier and cannot load the full model... pre-computed results only" |
| C2 | 8/8 示例全部失败，无诚实标注 | ✅ **已修复** | `examples.json` 现在包含 🟡PARTIAL/🔴FAILURE 标签、✓/✗ 标注、note 字段说明对错 |
| C3 | NED 被当成唯一指标，塌缩模型能拿 0.7961 | ✅ **已修复** | `eval_benchmark_v3.py` 新增多指标：ExactMatch、CompF1、CompPrec、CompRec、TokenRec、NED、RepRate、Diversity。`eval_topology_v2.py` 新增 joint_f1、value_acc |
| C5 | README 推塌缩模型为最佳 | ✅ **已修复** | 最新 commit 的 README 已更新（Demo 和论文均指向 V10-Fixed S600，非塌缩 r16） |
| M1 | lr=5e-4 违反 handover ≤2e-5 | ✅ **已修复** | `train_llm_v10_fixed.py:80` 使用 lr=2e-5 + LinearWarmup(100) + CosineAnnealing |
| M4 | HF 仓库 6 个 LoRA 文件无说明 | ✅ **部分修复** | hf_space README 现在指向 S600 权重；但 hf_model_clone 仍含多文件，README 待更新 |
| M5 | "关键创新"实为修自己的 bug | ✅ **已修复** | 论文/README 现在将三个修复明确标注为 "three critical training bugs"，不再称为"创新" |
| - | 论文占位作者 Hippocampus | ✅ **已修复** | `english.tex:44-55` 作者已改为 Jianning Zhang & Yifei Chen（中文模板也已修复） |
| - | 数字不一致（0.7791 vs 0.7760） | ✅ **已修复** | Demo 所有数字统一指向 V10-S600 easy50-pure 44-sample benchmark |
| - | 选择性 reporting 只报 easy 子集 | ✅ **已修复** | 论文/Demo 明确标注 "easy50-pure, 44 samples"，并公开 joint_f1=0.019 和 87% 数值幻觉率 |

### ⚠️ 仍存在的问题

| # | 上次发现 | 当前状态 | 说明 |
|---|---|---|---|
| C6 | 选择性 reporting（只报 easy） | ⚠️ **仍存在** | 仍只报 easy50-pure (44样本)。full523 的 NED=0.9164 仍未在 Demo/README 中显示。论文在附录中提到了 full523。 |
| M6 | 硬编码 Windows 路径 | ⚠️ **仍存在** | `train_llm_v10_fixed.py:31,57` 仍有 F:/ 路径，但有 fallback 机制。新脚本 `eval_benchmark_v3.py:33-36` 仍有 DLL 硬编码路径。 |
| M7 | 100+ 探索性脚本 | ⚠️ **略有改善** | V10/V11/V12 新增了规范的训练脚本，但旧的试错脚本仍在。 |
| M8 | 6 处 monkey-patch | ⚠️ **未修复** | `eval_benchmark_v3.py:24-30` 仍有 flex_checkpoint 补丁，V10 训练脚本同样需要。 |
| - | ORCID 0000-0000 | ⚠️ **仍存在** | `english.tex:62,65` ORCID 仍为占位符（但 uniqueAffiliation=true 不使用该分支） |
| - | 训练集规模不一致 | ⚠️ **略有改善** | 论文现在声称 1,857 samples（V5 Golden）。Demo 说 1,554（V9-Pure）。V10/V11 使用 V9-Pure 数据（1,554）。README 的 24,717 已不再提及。不一致缩小到 1,554 vs 1,857。 |
| - | 无基座 LICENSE | ⚠️ **未修复** | PaddleOCR-VL 的许可证条款仍未被声明。MIT 徽章仍在。 |
| - | model.generate() OOM 问题 | ⚠️ **未修复** | 推理仍可能因 OOM 回退到不同路径。但 V10 使用 manual greedy 作为默认路径（更稳定）。 |

---

## 三、新增的积极变化

### 1. 论文新增 "Value Bottleneck" 分析（深刻）
`english.tex:280-284` 深入分析了 comp_f1→joint_f1 的断崖式下跌（0.206→0.019）：
- 冻结的视觉编码器针对自然图像优化，不适合 8-12px 的小工程文字
- 训练数据中数值分布极度偏斜（10k、100nF 等高频值主导）
- 384px 最大分辨率限制了小文字的有效像素预算
→ 直接引出 Phase 3 计划（Vision Encoder LoRA + 高分辨率 + 两阶段训练）

### 2. 论文新增 "Honest Assessment" 板块
`english.tex:298` 明确陈述：
- joint_f1 = 0.019（仅 ~2% 元件完全正确）
- 87% 数值被幻觉生成
- ExactMatch 对所有模型均为 0%
- "它是研究原型，不是生产就绪"

### 3. 多指标评估体系
`eval_benchmark_v3.py` 新增 8 指标评估：
- ExactMatch、CompF1、CompPrec、CompRec、TokenRec、NED、RepRate、Diversity
- `eval_topology_v2.py` 新增 (refdes, value) 对级别的 joint_f1 和 value_acc

### 4. V10-Fixed 训练修复
- LR: 5e-4 → 2e-5（25 倍降低），加 LinearWarmup(100)
- Repetition penalty = 1.1
- 声称效果：TokenRec 从 12.6%（V9 easy50-pure）提升到 15.4%（V10 S600），RepRate 降低

### 5. V11-Regularized 和 V12-Stage2 已开始训练
- V11: 加 LoRA dropout=0.1、label_smoothing=0.05、数据增强、扩展数据集（1554→~3839）
- V12: Phase 3 Vision LoRA —— 冻结 LLM LoRA，从头训练 Vision Encoder LoRA（r=16，lr=1e-4，5 epochs）
- 均有完整训练脚本和 checkpoint

---

## 四、仍存在的严重问题

### S1. ExactMatch 仍为 0%，joint_f1 仅 0.019
这是最根本的限制。模型无法产生一份完整正确的网表。joint (refdes, value) F1 为 0.019 意味着仅 ~2% 的元件在标识符和数值上都正确。87% 的数值被幻觉生成。对于"电路原理图 OCR 和网表抽取"这个任务目标，模型当前离"可用"还很远。

### S2. 仅评估 easy50-pure（44 样本），回避 harder 样本
full523 上的 NED 为 0.9164（仅 3.1% 提升 vs base），但仍未在 Demo 或 README 中显示。报告只展示最好的 44 个样本（easy50-pure）而非完整测试集。虽然论文在附录提到了 full523 数字，但主要结果表只展示 easy50-pure。

### S3. V10 S600 完整基准结果文件未在仓库中找到
我无法找到 `results_v3_lora_lora_s600_ocr_vl_sft-test-easy50-pure.jsonl`（44 样本的完整 V10 S600 评估结果）。存在的 V10 结果文件仅 16-20 个样本且 NED=1.0（看起来是训练时的监控输出，非正式评估）。这意味着：
- V10 S600 的声称指标（CompF1=0.2061, NED=0.8031）无法由我独立复算验证
- 这些数字来源于作者自己的 `eval_benchmark_v3.py` 运行，但结果文件未提交到仓库

### S4. 合成数据仍仅 6 种拓扑
虽然 V10 使用了 V9-Pure 数据（457 Synthetic V3），合成数据仍来自 `gen_synthetic_v3.py` 的 6 种拓扑。V11 计划使用 Synthetic V4（~1500 张），但拓扑多样性是否增加未知。模板记忆仍是主要失败模式（Demo 示例 #2 BT1/Battery_Cell 模式、#3 AMS1117 模式）。

### S5. 硬编码路径和 monkey-patch 仍存在
`eval_benchmark_v3.py:24-36` 仍有硬编码 DLL 路径和 F:/ 缓存路径。`train_llm_v10_fixed.py:31,57` 有 F:/ 回退路径。虽然有 fallback 但默认仍绑定特定 Windows 机器。

### S6. HF 模型仓库未更新
`hf_model_clone/` 的最新文件仍是 7 月 2 日的（V8-Fixed + V9-Pure + 两个旧权重）。V10-Fixed S600、V11-Regularized、V12-Stage2 的权重未推送到 HuggingFace。Demo 声称"LoRA Weights (S600)"链接到 HuggingFace，但实际仓库可能仍包含旧权重。

---

## 五、新发现的问题

### N1. 训练脚本名称与实际版本不一致
`train_llm_v10_fixed.py` 内部注释写 "TRAINING V11 (Phase 2)"，保存到 `checkpoints_v11` 目录。文件名是 V10，内容是 V11。这种混乱可能导致错误复现。

### N2. V10 的部分结果文件显示全塌缩
`results_v2_v10final_easy50-pure.jsonl`（16 样本）和 `results_v2_v10s600_easy50-pure.jsonl`（20 样本）中所有预测均为 NED=1.0、diversity=1（全部相同），表示这些监控点上的模型完全塌缩。虽然这些是训练中的监控快照（非最终评估），但它们暗示 V10 在训练过程中可能经历了塌缩阶段。

### N3. eval_benchmark_v3.py 仍在 scripts 目录但未被 git 跟踪
`eval_benchmark_v3.py` 和 `eval_topology_v2.py` 位于 `circuit-ocr-dataset/scripts/` 但 `git ls-files` 显示它们不在仓库的跟踪文件中。这些关键评估脚本可能仅存在于本地工作副本，未推送到远程仓库。他人无法复现基准测试。

### N4. V12 两阶段训练的理论依据存疑
`train_llm_v12_stage2_vision_lora.py` 的假设是"V10 的 Vision LoRA 训练不足，因为 LLM LoRA 的梯度竞争了"。但 V10 使用的是 LLM-Only LoRA（冻结视觉编码器），根本没有训练 Vision LoRA。从零开始训练 Vision LoRA 作为 Stage 2 是合理的 Phase 3 方法，但脚本注释中的"V10-Fixed's vision LoRA (r=16) was undertrained"与 V10 的实际架构不符——V10 根本没有 vision LoRA。

---

## 六、指标对账

我复算的结果（可获取数据）vs 论文/Demo 声称：

| 指标 | Base (我算) | V9-Pure (我算) | V10-S600 (声称) | 对账状态 |
|---|---|---|---|---|
| easy50-pure NED | 0.9296 | 0.7869 | 0.8031 | V10 NED **差于** V9（0.8031 > 0.7869）。但 V10 TokenRec/RepRate 更好。需权衡。 |
| TokenRec | 0.0% | 12.6% | 15.4% | V10 声称比 V9 好 +2.8pp。合理（lr 修复 + rep_penalty）。 |
| RepRate | ~23% | ~48% | 15.9% | V10 声称大幅降低重复。合理（rep_penalty + lr 修复）。 |
| ExactMatch | 0% | 0% | 0% | ✅ 一致。所有模型 0%。 |
| Diversity | ~86% | ~91% | 90.9% | ✅ 一致。 |

**注意：** V10 NED (0.8031) 实际上**比 V9 (0.7869) 差**（NED 越低越好）。但这可能是因为 V9 的低 NED 来自重复退化（模型输出短重复串，碰巧与 GT 有字符重叠）。V10 的更高 NED 伴随更高的 TokenRec 和更低的 RepRate——这是**更好的模型**因为它在真正做 OCR 而非输出格式垃圾。这印证了上次审核的核心论点：**NED 单独使用会奖励退化。** 项目团队现在用多指标体系正确地解决了这个问题。

---

## 七、V11 和 V12 的预期

- **V11-Regularized:** dropout=0.1 + label_smoothing=0.05 + 数据增强 + 扩展数据（~3,839 样本）。预期进一步降低过拟合和重复。
- **V12-Stage2:** 两阶段 Vision LoRA —— 这是论文描述的 Phase 3。预期改善数值读取（当前 87% 错误率的核心瓶颈）。但 7 月 5 日凌晨刚完成训练，尚无评估结果。

---

## 八、给百度的最新建议

### 高优先级
1. **将 V10-S600 完整基准结果文件推送到仓库**，使声称的数字可被独立验证
2. **将 V10/V11/V12 权重和更新后的 README 推送到 HuggingFace**（当前 hf_model_clone 仍是 7 月 2 日的旧版）
3. **将 `eval_benchmark_v3.py` 和 `eval_topology_v2.py` 纳入 git 跟踪**，确保评估方法可复现
4. **在 Demo 中显示 full523 和 easy200 的指标**，而非仅 easy50-pure

### 中优先级
5. 继续 Phase 3（V12 Vision LoRA + 高分辨率）——这是突破 87% 数值错误率的关键
6. 扩大合成数据的拓扑多样性（当前仅 6 种）
7. 清理硬编码 Windows 路径（至少让 Docker/CI 路径工作）
8. 核实并声明 PaddleOCR-VL 基座许可证

### 低优先级
9. 统一训练脚本命名（train_llm_v10_fixed.py 实际训练的是 V11）
10. 替换 ORCID 占位符
11. 归档旧试错脚本

---

## 九、结论

**相比 7 月 3 日的审核，项目在诚实性、方法论严谨性和工程方向上取得了显著进步。** Demo 不再虚假交互，论文如实披露 0% ExactMatch 和 87% 数值错误率，多指标体系解决了 NED 单独使用的欺骗性问题，LR 已修正，V10/V11/V12 形成了清晰的 Phase 1→2→3 路线图。

**然而，核心性能瓶颈仍未突破：** 模型仍无法输出任何完整正确的网表，joint F1 仅 0.019（2%），87% 的数值被幻觉生成。V12 Stage2 Vision LoRA 正在尝试解决这一问题，结果待观察。

**该项目现在是一个诚实的、方法论正确的研究原型，正朝着正确的方向迭代。** 如果 Phase 3 能将 joint_f1 提升到 0.3-0.5 并且 ExactMatch 破零，它将开始具有实用价值。在此之前，它适合作为研究基准和概念验证，但不适合作为生产交付物。

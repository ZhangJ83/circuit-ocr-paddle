# CircuitOCR 深度审核报告 V4（2026-07-06 第四轮更新版）

**审核日期：** 2026-07-06
**审核对象：** Commit `5859b78` "Fix V3 audit findings"
**审核方式：** 逐文件 diff 阅读 + 交叉验证文档一致性

---

## 一、总体评价：V3 审核发现被快速修复，诚实性达历史最高，但新 Bug 描述矛盾出现

Commit `5859b78` 直接回应了 V3 审核的全部 4 个关键发现。修复速度快、态度诚恳。**V11/V12 的塌缩现在被全面诚实披露。** "三个 Bug"被扩展为"六个 Bug"并分为两类。107 vs 1,554 的矛盾被澄清。

**但 app.py 中新增的"数据集 README 的三个 Bug"描述与实际数据集 README 内容完全不同——这是一个新的矛盾。**

---

## 二、V3 审核问题的修复情况

### ✅ 已修复

| # | V3 发现 | 状态 | 证据 |
|---|---|---|---|
| S1 | V11 RepRate 93.2% vs 84.1% 不一致 | ✅ **已修复** | README 统一改为 84.1%（中英文） |
| S2 | V12 塌缩未披露 | ✅ **已修复** | README V12 状态 "Training (Stage 2)" → **"Completed — collapsed"**，详细描述了塌缩现象 |
| S3 | "三个 Bug" 两套列表矛盾 | ✅ **已修复** | 统一为 "6 项训练基础设施发现"，分为 A 类（序列生成通用，3 项）和 B 类（PaddleOCR-VL 特定，3 项） |
| S4 | 107 vs 1,554 矛盾 | ✅ **已修复** | Dataset README 明确：22,340→107 是合成池去重结果（占合成 V3 子集 457 条的 23.4%），1,554 = 1,097 KiCad + 457 合成 = 总训练集 |
| — | 论文 validation total 算错（171） | ✅ **已修复** | 50 + 171 = 221，已更正 |

### 新增的诚实披露

| 位置 | 内容 |
|---|---|
| README Phase 1 表格下方 | 新增注释：V11 CompF1=0.0604, NED=0.9171, RepRate=84.1% — "正则化+合成数据的方案适得其反" |
| README V11/V12 表格 | V12 状态改为 "Completed — collapsed"，详细描述："Vision LoRA 重新训练完全破坏了 LLM 的文本生成能力。全部 50 个预测均为垃圾输出" |
| Demo Benchmark tab | 新增 V11 和 V12 塌缩到 Limitations 板块 |
| Demo About tab | Phase 2/3 状态从 "🔄 In Progress" / "📋 Planned" 改为 "❌ Completed — collapsed"，新增 "Lessons Learned" 行 |
| Dataset README | 新增 "后续实验：V11 与 V12（失败记录）" 表格和教训总结 |
| Dataset README | "三个训练 Bug" → "训练基础设施发现（共 6 项）"，分为 A/B 两类 |

---

## 三、仍未修复的问题

| # | 前几轮发现 | 状态 |
|---|---|---|
| 1 | V12 训练脚本注释 "V10-Fixed's vision LoRA (r=16) was undertrained" | ⚠️ **仍未修复**（第 4, 140, 502 行）。V10 没有 vision LoRA。 |
| 2 | eval_benchmark_v3.py 等 6 个关键脚本未 git 跟踪 | ⚠️ **仍未修复** |
| 3 | 硬编码 Windows 路径 | ⚠️ **仍未修复** |
| 4 | HF 模型仓库未更新 | ⚠️ 未验证但大概率仍为旧版 |

---

## 四、本轮新发现的问题

### 🔴 N1. App.py 描述了第三套"数据集 README 的三个 Bug"（与实际不符）

这是本轮最重要的新发现。`hf_space/app.py` 的 Benchmark tab 中新增了以下文字：

> **Six training pitfalls discovered** ... plus 3 in dataset README — **eval/test data leakage in Easy100/Easy200/Easy50 splits, synthetic V3 template contamination, missing refdes prefix normalization**

但 `circuit-ocr-dataset/README.md` 中实际列出的 B 类（PaddleOCR-VL 特定）Bug 是：

> 4. **LoRA 权重合并精度损失**：`p.set_value()` 在 float32→float16 转换时产生截断误差
> 5. **Tokenizer 特殊 token 偏移**：`<|box_start|>` 等特殊 token 的索引偏移
> 6. **梯度累积与学习率解耦**：`global_step` 计数未正确对齐

**这两组描述完全不同。** App.py 说的"数据泄露、模板污染、refdes 前缀未归一化"与 Dataset README 的"权重精度损失、tokenizer 偏移、梯度累积解耦"毫无重叠。

这意味着现在存在**三套不同的 Bug 列表**：
1. 主 README：causal token double-shift / BPE boundary merging / set_state_dict→None
2. Dataset README：LoRA 权重精度损失 / Tokenizer 特殊 token 偏移 / 梯度累积与学习率解耦
3. App.py 声称的 Dataset README：数据泄露 / 模板污染 / refdes 前缀未归一化

虽然主 README 和 Dataset README 之间已经统一（6 项 = 3+3），但 App.py 引用的 Dataset README Bug 列表仍然与实际不符。**看起来 App.py 的这段文字是独立编写的，没有与 Dataset README 的实际内容对账。**

另外，"eval/test data leakage in Easy100/Easy200/Easy50 splits" 这个声称如果属实，是一个严重问题——意味着训练数据和测试数据有重叠。但这个 Bug 在 Dataset README 中完全没有被提及。如果确实存在数据泄露，需要在 Dataset README 中记录；如果不存在，App.py 就是在引用一个不存在的发现。

### 🟡 N2. "Six pitfalls" 的命名和分类值得商榷

- A 类（序列生成通用问题）中的 "causal token double-shift" 和 "BPE boundary merging" 确实是训练基础设施 Bug
- 但 "set_state_dict→None" 是 PaddlePaddle 3.1.0 的 API 变更，不是"pitfall discovered"，而是 API 兼容性问题
- B 类（PaddleOCR-VL 特定）中的 "梯度累积与学习率解耦" 是标准深度学习实践问题，不具有 PaddleOCR-VL 特异性

### 🟡 N3. V12 塌缩解释"destructive gradient interference"缺乏实证

README 和 Demo 将 V12 塌缩归因为 "destructive gradient interference with the LLM head" 和 "视觉特征空间的扰动破坏了 V10 已学到的跨模态对齐"。这些是合理的假设，但没有提供任何梯度分析、特征空间可视化或其他实证支持。Dataset README 的"教训"部分说"Stage2 训练需要真实域数据支持"，但 V12 使用的是与 V10 相同的数据（V9-Pure），数据域不是变量。

### 🟡 N4. "数据泄露"声称需要核实

App.py 提到 "eval/test data leakage in Easy100/Easy200/Easy50 splits" 是一个新出现的严重声称。如果确实存在数据泄露，V10 的所有基准测试结果（CompF1=0.2061 等）都需要重新评估。但 Dataset README 完全没有提到数据泄露。这可能是：(a) 真实但未被记录的问题，(b) App.py 写错了，(c) 指的是不同的数据切分问题。

---

## 五、文档一致性矩阵（V4）

| 要素 | 主 README | Dataset README | App.py | 论文 |
|---|---|---|---|---|
| V10 CompF1 | 0.2061 ✅ | 0.2061 ✅ | 0.2061 ✅ | 0.2061 ✅ |
| V11 RepRate | 84.1% ✅ | 84.1% ✅ | 84.1% ✅ | — |
| V12 状态 | Collapsed ✅ | Collapsed ✅ | Collapsed ✅ | — |
| Bug 总数 | 6 ✅ | 6 ✅ | 6 ✅ | 3（未更新） |
| B 类 Bug 内容 | 引用 Dataset README ✅ | LoRA/Tok/LR ✅ | **数据泄露/模板/refdes ❌** | — |
| 训练样本数 | 1,554 ✅ | 1,554 ✅ | 1,554 ✅ | 1,554 ✅ |
| 合成池 107 | — | ✅（已澄清） | — | — |

---

## 六、与前几轮审核的对比

| 维度 | V1 (7/3) | V2 (7/5) | V3 (7/6 AM) | V4 (7/6 PM) |
|---|---:|---:|---:|---|
| Demo 诚实性 | ❌ | ✅ | ✅ | ✅ |
| 论文诚实性 | ❌ | ✅ | ✅ | ✅ |
| V11 塌缩披露 | N/A | N/A | ⚠️ 部分 | ✅ 全面 |
| V12 塌缩披露 | N/A | N/A | ❌ 未披露 | ✅ 全面 |
| Bug 列表一致性 | N/A | N/A | ❌ 矛盾 | ⚠️ 基本统一但 App.py 仍有误 |
| 样本数一致性 | ⚠️ | ⚠️ | ❌ 矛盾 | ✅ 已澄清 |
| 失败实验文档化 | N/A | N/A | ❌ | ✅ |

---

## 七、给百度的最新建议

### 紧急（P0）
1. **修正 App.py 中关于 Dataset README Bug 的错误描述**。当前 App.py 说 Dataset README 记录了"数据泄露、模板污染、refdes 前缀未归一化"，但实际 Dataset README 记录的是"LoRA 权重精度损失、Tokenizer 偏移、梯度累积解耦"。要么修改 App.py，要么修改 Dataset README，必须一致。
2. **核实"数据泄露"声称**。如果 Easy100/Easy200/Easy50 确实存在训练-测试数据泄露，需要在 Dataset README 中记录，并重新评估 V10 基准结果。如果不存在，从 App.py 中删除此声称。

### 高优先级（P1）
3. **修正 V12 训练脚本注释**（第 4-5 行："V10-Fixed's vision LoRA"——V10 没有 vision LoRA）
4. **将 6 个关键脚本纳入 git 跟踪**（eval_benchmark_v3.py, eval_topology_v2.py, train_llm_v10_fixed.py, train_llm_v11_regularized.py, train_llm_v12_stage2_vision_lora.py, eval_batch_v12_stage2.py）
5. **论文更新 Bug 数量**（当前仍只说 3 个）

### 中优先级（P2）
6. V12 塌缩分析应更实证（梯度范数、特征空间可视化等），而非纯假设性解释
7. 清理硬编码路径
8. 推送最新权重到 HuggingFace

---

## 八、结论

**第四轮审核确认：项目在 commit `5859b78` 中快速修复了 V3 审核的全部 4 个关键发现。** V11 和 V12 的塌缩现在被全面、诚实地披露在所有渠道（README、Demo、Dataset README）。"三个 Bug"被正确地扩展为"六个 Bug"并分为两类。107 vs 1,554 的样本数矛盾被澄清。项目在诚实性上达到了历史最高水平。

**但 App.py 引入了一个新的矛盾：** 它声称 Dataset README 记录了"数据泄露、模板污染、refdes 前缀未归一化"三个 Bug，而实际 Dataset README 记录的是"LoRA 权重精度损失、Tokenizer 偏移、梯度累积解耦"。这个不一致如果被外部评审发现，会再次损害项目可信度。

**此外，"数据泄露"是一个严重的声称——如果属实，V10 的所有基准测试结果需要重新评估。** 建议项目团队优先核实此事。

**项目现状：** V10 S600 仍是唯一可用的 checkpoint（CompF1=0.206, 4.5×, ExactMatch=0%）。V11（正则化）和 V12（Vision LoRA）均塌缩并被诚实记录为失败。Phase 4 需要架构性突破，而不仅仅是增量改进。

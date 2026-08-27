# CircuitOCR 深度审核报告 V3（2026-07-06 第三轮更新版）

**审核日期：** 2026-07-06
**审核对象：** Commit `032f5b4` 更新后的 CircuitOCR 项目（含 V11-Regularized 和 V12-Stage2 Vision LoRA 的完整结果）
**审核方式：** 逐文件 diff 阅读 + 所有结果文件实证检验 + 与前两轮审核对比

---

## 一、总体评价：诚实性进一步加强，但 V11/V12 双双塌缩，核心矛盾暴露

自 V2 审核（7月5日）以来，项目在 **commit `032f5b4`** 中做出了回应 V2 审核的改进。Demo/README/论文的诚实性和透明度进一步提升。**然而，我这次发现了 V11 和 V12 的完整评估结果文件——两者均彻底塌缩，比 V10 S600 差得多。** README 诚实地报告了 V11 塌缩，但 V12 的塌缩结果存在于磁盘上却未在任何公开文档中披露。

---

## 二、V2 审核问题的修复情况

### ✅ 已修复

| # | V2 发现 | 状态 | 证据 |
|---|---|---|---|
| "关键创新"实为修 bug | ✅ **已修复** | README 重命名为 "Three training pitfalls discovered and documented for the community"，论文新增段落说明这些 bug 的社区价值 |
| 仅报 easy50-pure | ✅ **部分修复** | README/Demo/论文均引用 Appendix B 的完整 4-split 结果。论文 Appendix B 现在包含完整的 easy50/100/200/full523 表格 |
| 论文缺少 ExactMatch=0% 的 scale context | ✅ **已修复** | 论文新增 "ExactMatch=0% and joint_f1=0.019 are expected outcomes at this scale (0.9B parameters, 5.7M trainable, 1,554 samples, 43 minutes on consumer GPU)" |
| 缺少 consumer GPU 可复现性强调 | ✅ **已修复** | 论文新增 "any individual developer can replicate the entire training pipeline without data center infrastructure" |
| E1-E6 消融实验细节缺失 | ✅ **已修复** | 论文新增 E1-E6 的详细描述 |
| Demo 无 Limitations 板块 | ✅ **已修复** | Benchmark tab 新增完整的 Limitations 板块，列出 6 条限制 |

### ⚠️ 仍存在

| # | V2 发现 | 状态 |
|---|---|---|
| 硬编码 Windows 路径 | ⚠️ 仍存在（`train_llm_v12_stage2_vision_lora.py:47-48` 仍有 `F:/hf_cache` 等路径） |
| monkey-patch 补丁 | ⚠️ 仍存在（V12 训练脚本同样需要 flex_checkpoint 补丁） |
| HF 模型仓库未更新 | ⚠️ 未验证但大概率仍为旧版 |
| eval_benchmark_v3.py 等脚本未 git 跟踪 | ⚠️ 仍为 untracked |
| V10 训练脚本名不一致（文件名 v10，内容 v11） | ⚠️ 仍存在 |
| V12 训练脚本注释错误（说 V10 有 vision LoRA，实际没有） | ⚠️ **仍未修复** |

---

## 三、本轮新增的积极变化

### 1. README 大幅重写
- 移除了旧的误导性表格（NED 0.7961 +10.0%）
- 新增完整的 8 指标 Phase 1 Benchmark 表（含 ExactMatch 列）
- 新增 Phase 2 Topology Metrics 表
- 新增 "Research Contributions" 板块，诚实解释 ExactMatch=0% 在 0.9B 规模下的合理性
- 新增 "Exploration Process (V1→V10)" 表格
- **新增 V11 & V12 进展表格**（这是关键——诚实披露 V11 塌缩）

### 2. Demo 进一步诚实化
- Inference tab 标题改为 "⚠️ Research Prototype — Pre-Computed Results Only"
- 示例从之前展示的 6 个扩展到全部 8 个，每个都有 🟡PARTIAL/🔴FAILURE 标记
- 替换了 2 个之前看起来"还行"的示例为诚实的失败示例（移除了 Mark-MDO47 和 analog_simple_v2_0018 的旧预测）
- Benchmark tab 新增 ExactMatch 列、Topology Metrics 表、Limitations 板块
- About tab 新增 Roadmap 表（Phase 1-4）

### 3. 论文完善
- 新增 "consumer-grade hardware" 强调
- 新增段落：ExactMatch=0% 在 0.9B 规模下是预期的
- Appendix B 新增完整的 4-split 基准测试表（easy50/100/200/full523）
- S800 overfitting 证据文档化

### 4. README 诚实披露 V11 塌缩
这是本轮最重要的诚实性改进。README 明确写道：
> V11 (Phase 2): Mode collapse — RepRate monotonically increased to 93.2%. Synthetic data visual distribution mismatch confirmed.

---

## 四、本轮新发现的严重问题

### 🔴 S1. V11 完全塌缩（实证确认）

我找到了 V11 S600 的完整 44 样本评估文件 `results_v3_lora_s600_v11_easy50.jsonl`：

| 指标 | V10 S600 | V11 S600 | 变化 |
|---|---:|---:|---|
| CompF1 | 0.2061 | **0.0604** | ↓ 71% |
| TokenRec | 0.1540 | **0.0584** | ↓ 62% |
| NED | 0.8031 | **0.9171** | ↑ 14%（恶化） |
| RepRate | 15.9% | **84.1%** | ↑ 5.3× |
| Diversity | 90.9% | **50.0%** | ↓ 45% |
| ExactMatch | 0% | 0% | — |

V11 的效果不仅没有改善，反而**全面倒退**。NED 0.9171 仅比 Base 模型的 0.9296 好 1.3%。84% 的样本出现重复退化。几乎所有预测输出都是 `GND\nGND\n...`、`1\n1\n1\n...` 或 `J1\nJ1\nJ1\n...`。

**但有一个重要不一致：** README 称 V11 RepRate 为 93.2%，但结果文件显示 84.09%。这 9 个百分点的差异可能来自不同 checkpoint（S800 vs S600）或不同的评估脚本。需要澄清。

### 🔴 S2. V12 Stage2 完全塌缩（未被披露！）

我找到了 V12 Stage2 的完整评估文件 `results_stage2_easy50.jsonl`（50 样本）。**结果比 V11 更差：**

- 几乎所有预测输出都是：`100000000000...`、`+333333333...`、空字符串、`Test\nTest\n...`、`...\n...\n...`
- 没有任何一个样本有意义的输出
- 比 V10 S600 差得多——实际上比 Base 模型还差

**关键问题：README 将 V12 状态标记为 "Training (Stage 2)" 且 "Results pending"。** 但训练实际上已于 7 月 5 日 00:57 完成（checkpoint 文件时间戳），评估结果文件存在于磁盘上。V12 的塌缩结果**未被任何公开文档披露**。

### 🔴 S3. "三个 Bug" 在两个 README 中完全不同

这是我在本轮发现的最令人困惑的矛盾：

**主 README（和论文）中的三个 Bug：**
1. Causal token double-shift（AutoModelForConditionalGeneration 内部已做 shift）
2. BPE boundary merging（影响所有 seq2seq 训练）
3. set_state_dict silent failure（Paddle 3.1.0 API 变更）

**Dataset README（`circuit-ocr-dataset/README.md`）中的三个 Bug：**
1. LoRA 权重合并精度损失（`p.set_value()` float32→float16 截断误差）
2. Tokenizer 特殊 token 偏移（`<|box_start|>` 等特殊 token 索引偏移）
3. 梯度累积与学习率解耦（`global_step` 计数在梯度累积场景下未对齐）

**这是两个完全不同的列表。** 同一个项目声称发现了"三个训练 Bug"，但在不同文档中列出的是完全不同的三件事。这意味着要么：(a) 实际有 6 个 bug 但被分别打包成两组"三个"，(b) 其中一个列表是编造的，(c) 两个 README 由不同人维护且从未对账。

Dataset README 还声称 "以上 Bug 的修复方案已合并入 LoRAModel wrapper 与训练脚本，并向上游社区报告"——但训练脚本（V10/V11/V12）中从未提及这些 Bug 的修复。

### 🔴 S4. Dataset README 声称 107 样本 vs 论文声称 1,554/1,857 样本

Dataset README 的 "V5 Golden 数据质量工程" 部分写道：
> 从 22,340 条原始样本中剔除 99.5% 的重复/近重复数据，最终保留仅 **107 条**高多样性样本。

但论文和主 README 声称训练集为 1,554 或 1,857 样本。107 vs 1,554 是 **14.5 倍的差距**。

可能的解释：
- 107 可能指去重后的"独特拓扑模板"数（而非总样本数）
- 1,554 可能包含每个模板的多个渲染变体（不同参数值、退化增强等）

但 Dataset README 的表述"最终保留仅 107 条高多样性样本"极其误导——读起来像是只有 107 个训练样本。如果 107 指的是拓扑模板数，需要明确说明。

### 🟡 S5. V12 训练脚本架构错误仍未修复

我在 V2 审核中已经指出：`train_llm_v12_stage2_vision_lora.py` 第 5 行注释说 "V10-Fixed's vision LoRA (r=16) was undertrained"，但 V10-Fixed 是 **LLM-Only LoRA**（冻结视觉编码器），根本没有训练过 Vision LoRA。

这个错误在新版脚本中**仍然存在**（第 5 行）。虽然这只是注释错误不影响训练逻辑，但它说明：(a) V2 审核的这条意见未被阅读或未被采纳，(b) 脚本作者对自己训练的模型架构存在理解偏差。

### 🟡 S6. V11 塌缩原因的自我矛盾

README 说 V11 塌缩原因是 "Synthetic data visual distribution mismatch confirmed"。
但 V10（使用相同数据 V9-Pure）并未塌缩。V10 和 V11 的主要区别是：
- Dropout=0.1 + label_smoothing=0.05 + 数据增强
- 扩展数据从 1,554 → 3,054（加了 Synthetic V4）

如果 V10 在相同数据上不塌缩，V11 塌缩更可能是由**新增的 Synthetic V4 数据质量差**或**dropout/label_smoothing 超参不当**导致，而非简单的"合成数据视觉分布不匹配"——这个解释无法自洽。

### 🟡 S7. Demo 示例 #5（analog_simple_v2_0018）预测变更但仍是失败

旧版 examples.json 中该示例的 S600 预测为 `"U2:\nPico"`（简短但至少尝试了）。新版中预测变为 `"1\n2"`（完全塌缩）。如果这是同一个 S600 checkpoint 的重新运行结果，说明推理存在不确定性（可能是 manual greedy decode 的不同实现路径）。如果是不同 checkpoint，需要说明。

---

## 五、指标对账（更新）

| 指标 | V10 S600 (声称) | V11 S600 (实测) | V12 Stage2 (实测) |
|---|---:|---:|---|
| CompF1 | 0.2061 | 0.0604 | 未计算（全塌缩） |
| NED | 0.8031 | 0.9171 | 接近 1.0 |
| RepRate | 15.9% | 84.1% | ~100% |
| Diversity | 90.9% | 50.0% | ~10% |
| ExactMatch | 0% | 0% | 0% |

V10 S600 仍是唯一可用的 checkpoint。V11 和 V12 均比 Base 模型更差（在除 NED 外的所有指标上）。

---

## 六、与前两轮审核的对比

| 维度 | V1 审核 (7/3) | V2 审核 (7/5) | V3 审核 (7/6) |
|---|---:|---:|---|
| Demo 诚实性 | ❌ 伪装交互 | ✅ 诚实声明 | ✅ 进一步加强 |
| 论文诚实性 | ❌ 夸大宣传 | ✅ Honest Assessment | ✅ 增加 scale context |
| 指标报告 | ❌ NED 单独 | ✅ 多指标体系 | ✅ 完整 4-split |
| LR 修复 | ❌ 5e-4 | ✅ 2e-5 | ✅ 维持 |
| V11 状态 | 不存在 | 训练中 | 🔴 完成但塌缩 |
| V12 状态 | 不存在 | 训练中 | 🔴 完成但塌缩（未披露） |
| 文档一致性 | ⚠️ 多处不一致 | ⚠️ 减少但存在 | 🔴 新矛盾出现 |

---

## 七、给百度的最新建议

### 紧急（P0）
1. **在 README/Demo/论文中披露 V12 塌缩结果**。当前 README 将 V12 标记为 "Results pending"，但结果已存在且显示彻底失败。这构成选择性披露。
2. **统一"三个 Bug"的表述**。主 README 和 Dataset README 列出了完全不同的三组 Bug。确定实际发现了几个 Bug，统一文档。
3. **澄清 Dataset README 的 "107 条样本" 含义**。与论文的 1,554/1,857 矛盾，如果是拓扑模板数需明确标注。

### 高优先级（P1）
4. **将 eval_benchmark_v3.py、eval_topology_v2.py、train_llm_v12_stage2_vision_lora.py、eval_batch_v12_stage2.py 纳入 git 跟踪**
5. **修复 V12 训练脚本注释**（第 5 行：V10 没有 vision LoRA）
6. **解释 V11 RepRate 84.1%（结果文件）vs 93.2%（README）的差异**

### 中优先级（P2）
7. V11 塌缩原因分析应更精确——不能简单归因为"合成数据视觉分布不匹配"（V10 用相同数据不塌缩）
8. 清理硬编码 Windows 路径
9. 推送 V10/V11/V12 权重到 HuggingFace

---

## 八、结论

**第三轮审核的核心发现是：V11 和 V12 均完全塌缩，效果远不如 V10 S600。V10 S600 仍是唯一可用的 checkpoint。**

项目在诚实性和透明度上继续进步——README 诚实披露了 V11 塌缩，Demo 的 Limitations 板块更加完善，论文增加了 scale context。这些值得肯定。

**但有两个严重的新问题：**
1. **V12 塌缩未被披露**——README 说 "Results pending"，但结果已存在于磁盘上且显示彻底失败。这从前两轮的"诚实报告失败"退步了。
2. **文档内部矛盾激增**——"三个 Bug"在两个 README 中完全不同（6 个不同的 bug），Dataset README 的 107 样本与论文的 1,554 样本矛盾。这些不一致如果被百度评审发现，会严重损害项目可信度。

**项目现状：一个方法论正确、诚实的研究原型，在 Phase 1（V10）取得了 modest 但真实的进展（CompF1 4.5×, TokenRec 96×），但 Phase 2（V11）和 Phase 3（V12）的初步尝试均告失败。核心瓶颈（0% ExactMatch, 87% 数值错误）仍未突破。**

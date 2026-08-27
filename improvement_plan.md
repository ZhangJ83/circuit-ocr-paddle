# CircuitOCR 改进方案：从 0% 到可用

## 一、核实结果

审计报告 6 条致命指控全部核实：

| # | 指控 | 核实 | 实测 |
|---|------|:--:|------|
| C1 | Demo 不调用模型 | ✅ | `app.py:46` 返回静态字符串 |
| C2 | Demo 8/8 示例全错 | ✅ | 拒识/退化/单token，无一正确 |
| C3 | NED 指标失效 | ✅ | V8 easy100: 0% exact-match, 41% 重复退化, NED=0.7772 |
| C4 | 推理不可复现 | ✅ | generate() vs 手动贪心路径不同 |
| C5 | 最佳模型自相矛盾 | ✅ | README推r16塌缩模型, handover判不可用 |
| C6 | 选择性reporting | ✅ | full523仅+3.1%, 只报easy50的+10% |

**核心数据**：V8 easy100 — exact-match 0/100, token召回 12%, 41%样本重复退化≥4行, avg NED 0.7772

---

## 二、根因分析

### 根因 1（最致命）：lr=5e-4 导致 90% 退化

V5 用 lr=2e-5 → 90% 多样性、无退化。V8/V9 用 lr=5e-4（25倍）→ 41-90% 退化。

**机制**：高 lr 在 LoRA 低秩空间产生过大梯度步，迫使模型在训练早期就锁定到少数高频 token 组合（`BT6\nBT7\nBT6\nBT7`、`net11\nnet11\nVSS\nVSS`），后续训练无法逃脱这些局部最优。loss 从 2.71 降到 0.30 只是在这些退化 token 上的过拟合。

**修复**：lr 降回 2e-5，加 linear warmup（前 5% 步从 2e-6 → 2e-5），加 repetition_penalty=1.1 在推理端兜底。

### 根因 2：合成数据只有 6 种拓扑模板

`gen_synthetic_v3.py` 仅 6 种拓扑 × 10 类元件 × 固定取值空间。模型实际在记忆模板而非学习 OCR。V8/V9 频繁输出 `U1\nAMS1111\nU2\nAMS1111` 就是 ic_centric 拓扑的固化记忆。

**修复**：(a) 大幅扩增合成数据多样性——随机连线而非固定拓扑，增加元件位置/角度/字体随机化；(b) 大幅增加真实 KiCad 数据比例（当前仅 1,357/2,555=53%）。

### 根因 3：0.9B 基座模型对电路 OCR 容量不足

PaddleOCR-VL-0.9B 的视觉编码器为文档场景优化（发票、表单），电路符号（Ω/μ、IEEE 标准符号、丝印）是其训练分布外数据。0.3B LLM 在网表结构化输出上也容量有限。

**修复**：(a) 短期：用更大的 PaddleOCR-VL 变体（如有 2B/4B 版本）；(b) 若只能 0.9B：提高 max_dim 到 768px，增加 LoRA rank 到 32 覆盖更多视觉适配。

### 根因 4：评估指标与训练目标脱节

NED 奖励"格式像"而非"内容对"。塌缩模型 NED=0.7961，最佳模型 NED=0.7760，差距仅 0.02。

**修复**：以 exact-match 为主指标、元件 F1 为补充、NED 降为参考。训练时用 token 级别的 F.cross_entropy 本身没问题，但需要监控 exact-match 而非只看 loss。

### 根因 5：推理路径不确定

`eval_benchmark.py` 先试 `model.generate()`，OOM 就回退手动贪心。两条路径输出不同，基准数不可复现。

**修复**：统一使用 `model.generate()` + 固定 `generation_config`（max_new_tokens=256, do_sample=False, repetition_penalty=1.1）。若 8GB 不够，用 bfloat16 + 梯度检查点 + max_dim=384（而非 768）。

---

## 三、改进路线图

### Phase 1（本周，1-2天）：修复退化 + 重建基线

**目标**：消除 41% 重复退化，exact-match > 0%

1. **修复 lr**：从 5e-4 降至 2e-5，加 linear warmup 5% steps
2. **加 repetition_penalty=1.1**：推理时抑制重复
3. **统一推理路径**：固定 `model.generate(do_sample=False, max_new_tokens=256)`
4. **统一评估**：以 full523 的 exact-match + 元件 F1 + NED 三指标评估
5. **用 V5 Golden (2,299 samples) 训练 3 epoch**

**预期**：exact-match 从 0/523 → 5-10/523（简单图），重复退化从 41% → <20%

### Phase 2（2周）：扩数据 + 提质量

**目标**：exact-match > 10%，token 召回 > 30%

1. **扩增合成数据**：从 6 种固定拓扑 → 随机连线拓扑，元件 3-50 个，字体/角度随机化，生成 2,000 张
2. **增加真实 KiCad 数据**：从 GitHub topic:kicad 再爬 1,000-2,000 张，用 kicad-cli SVG 渲染 + stroked-text 提取 GT
3. **筛选 Masala-CHAI**：从 698 张中筛掉设计者偏移严重的，留 400 张高质量
4. **数据集**：目标 ~4,000 张 (2,000 synth + 1,600 real + 400 masala)
5. **训练 5 epoch, lr=2e-5, r=16, max_dim=384**

**预期**：exact-match 10-20/523, token 召回 30-50%, 重复退化 <10%

### Phase 3（1个月）：模型能力提升

**目标**：exact-match > 30%，可用于辅助标注

1. **尝试更大基座**：PaddleOCR-VL-2B 或其他 Paddle 生态 VLM
2. **两阶段训练**：Stage 1 纯文字 OCR（冻结视觉、训 LLM），Stage 2 拓扑理解（解冻视觉编码器部分层）
3. **课程学习**：先从简单合成图训练 → 逐步加入真实复杂图
4. **元件级监督**：在 loss 中给元件标号 token 更高权重

### Phase 4（2个月+）：实用化

**目标**：exact-match > 50%，可部署

1. **SPICE 仿真验证**：将输出网表送入 Ngspice，用仿真结果（语法错误/收敛失败/DC 工作点）作为奖励信号
2. **人机协同流程**：模型输出 → 人工修正 → 修正数据回流训练
3. **多模型集成**：用 2-3 个不同 LoRA 变体投票产生最终输出

---

## 四、立即执行项（Phase 1 具体步骤）

### Step 1: 修复训练脚本

```python
# train_llm_v10_fixed.py 关键改动：
# 1. lr: 5e-4 → 2e-5
base_lr = 2e-5  # 而非 5e-4
lr_scheduler = paddle.optimizer.lr.LinearWarmup(
    learning_rate=paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=base_lr, T_max=total_steps - 100, eta_min=2e-6),
    warmup_steps=100, start_lr=2e-6, end_lr=base_lr)

# 2. 训练集：ocr_vl_sft-train-v5-golden.jsonl (2,299 samples)
# 3. max_dim=384, r=16, alpha=32, 3 epochs
# 4. batch_size=1, grad_accum=4
```

### Step 2: 修复推理

```python
# eval_benchmark.py 改动：
# 统一使用 model.generate()，不再回退手动贪心
# 固定 generation_config:
gen_config = {
    'max_new_tokens': 256,
    'do_sample': False,
    'repetition_penalty': 1.1,
    'eos_token_id': tokenizer.eos_token_id,
    'pad_token_id': tokenizer.pad_token_id,
}
```

### Step 3: 修复评估

```python
# 新增指标：
# 1. exact_match_rate = (pred == gt).mean()
# 2. component_f1: 提取 pred 和 gt 中所有匹配 R\d+/C\d+/U\d+ 的 token，计算 F1
# 3. token_recall: pred 中出现在 gt 中的 token 比例
# 4. repetition_rate: 连续≥4相同行 的样本比例
```

### Step 4: 训练 + 评估

```bash
python scripts/train_llm_v10_fixed.py
python scripts/eval_benchmark.py --checkpoint lora_best --test full523 --metrics all
```

---

## 五、风险与注意事项

1. **0.9B 容量上限**：即使所有修复到位，0.9B 模型可能永远达不到 >50% exact-match。需要设置合理预期——目标应是"可用于辅助标注"（>30% exact-match），而非"替代人工"。

2. **数据瓶颈**：合成数据多样性有限，真实 KiCad 数据标注成本高（需 kicad-cli SVG 渲染 + stroked-text 提取）。Masala-CHAI 的 SPICE→OCR 转换存在设计者偏移，不可完全信任。

3. **Paddle 生态限制**：Paddle 3.x 环境不稳定（6 monkey-patch），升级或迁移可能引入新 bug。建议锁定当前可用版本。

4. **评估诚实性**：NED 不应再作为主指标。所有报告必须以 exact-match 为首要指标。0% 必须如实披露。

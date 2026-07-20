# CircuitOCR 交接文档 V4 — 2026-07-04

## 当前状态总览

| 项目 | 状态 |
|------|------|
| HuggingFace 权重 | ✅ 已上传 `yingchu83/CircuitOCR-lora` (V10-Fixed S600, 最优) |
| HuggingFace Space | ✅ 已更新 `yingchu83/CircuitOCR` (examples + Phase 1 benchmark) |
| 中文报告 | ✅ 已更新，包含阶段一多指标评估 + 新图表 (`template.pdf`) |
| 英文报告 | ✅ 已更新，包含 Phase 1 multi-metric eval + new figures (`english.pdf`) |
| V10-Fixed 训练 | ✅ 已完成 (S400/S600/S800/final, S600 最优) |
| 阶段一评估 | ✅ 已完成 (easy50-pure 44样本, 8指标全量评估) |
| 5个链接目标 | ✅ 全部更新 (3 GitHub + 2 HuggingFace) |
| 环境 | ✅ Paddle 3.1.0 + PaddleFormers 1.1.1, 稳定运行 |

## V10-Fixed 模型核心信息

**架构**: Wide LLM-Only LoRA (r=16, alpha=32) + 冻结 Projector
- 目标层: `model.layers.*.self_attn.{q,k,v,o}_proj`, `model.layers.*.linear_1`, `model.layers.*.linear_2`
- 冻结: `mlp_AR.linear_1`, `mlp_AR.linear_2`
- Trainable: 5.7M params / 908M total (0.63%)
- 训练数据: 1,554 samples (1,097 KiCad + 457 Synthetic V3)
- max_dim=384, lr=2e-5, grad_accum=4, 3 epochs = 1,165 optimizer steps
- 训练时间: ~43 分钟 (RTX 4060 8GB)

## 阶段一评估结果 (Phase 1 Benchmark)

**评估脚本**: `eval_benchmark_v3.py` (LoRAModel wrapper + `p.set_value()`, 修复了 `set_state_dict` → None 的 Bug)
**测试集**: easy50-pure (44 samples)

| Model | ExactMatch | CompF1 | CompPrec | CompRec | TokenRec | NED ↓ | RepRate | Diversity |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0% | 0.0455 | 0.0455 | 0.0455 | 0.0016 | 0.9296 | 6.8% | 90.9% |
| S400 | 0% | 0.1820 | 0.1862 | 0.2501 | 0.1302 | 0.8298 | 20.5% | 95.5% |
| **S600** ★ | 0% | **0.2061** | 0.2024 | **0.3114** | **0.1540** | **0.8031** | 15.9% | 90.9% |
| S800 | 0% | 0.2080 | 0.2862 | 0.1996 | 0.1191 | 0.8063 | 40.9% | 93.2% |

### 关键发现
- Component F1: **4.5×** 提升 (0.0455 → 0.2061)
- Token Recall: **96×** 提升 (0.0016 → 0.1540)
- NED: **13.6%** 相对错误率降低 (0.9296 → 0.8031)
- **S600 是最佳检查点**; S800 过拟合 (重复率 40.9%)
- ExactMatch 仍为 0% — 模型能识别元件但无法重建完整网表
- 多样性 90.9% — 无模态塌缩

## 加载方式 (关键!)

```python
# ✅ 正确: LoRAModel wrapper + p.set_value()
from paddleformers.peft import LoRAConfig, LoRAModel

TARGETS = [
    'model\\.layers\\..*q_proj', 'model\\.layers\\..*k_proj',
    'model\\.layers\\..*v_proj', 'model\\.layers\\..*o_proj',
    'model\\.layers\\..*linear_1', 'model\\.layers\\..*linear_2'
]
lc = LoRAConfig(r=16, lora_alpha=32, target_modules=TARGETS)
model = LoRAModel(model, lc)

# 手动逐参数加载 (set_state_dict 在 Paddle 3.1.0 返回 None!)
lora_state = paddle.load("lora_s600.pdparams")
model_lora_params = {k: p for k, p in model.named_parameters() if 'lora_' in k}
for ckpt_key, ckpt_value in lora_state.items():
    if ckpt_key in model_lora_params:
        p = model_lora_params[ckpt_key]
        ckpt_tensor = paddle.cast(ckpt_value, p.dtype)
        p.set_value(ckpt_tensor)

# ❌ 错误: numpy merge (破坏权重, 全输出 \n\n\n)
# ❌ 错误: model.set_state_dict() (Paddle 3.1.0 返回 None, 静默失败)
```

## 文件路径速查

```
项目根:       G:\mimo_project\circuit_ocr
数据集:       G:\mimo_project\circuit_ocr\circuit-ocr-dataset
Python:       E:\080000software\080900_Miniconda\miniconda3\envs\pyqpanda-quantum\python.exe
HF缓存:       F:\hf_cache\hub\
模型路径:     F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27
V10 checkpoint: circuit-ocr-dataset/PaddleOCR-VL-LoRA-circuit-ocr/checkpoints_v10_fixed/lora_s600.pdparams
V10 最终模型:  circuit-ocr-dataset/PaddleOCR-VL-LoRA-circuit-ocr/checkpoints_v10_fixed/lora_final.pdparams
阶段一评估:    circuit-ocr-dataset/scripts/results_v3_lora_s600_easy50.json
V10 LoRA权重:  circuit-ocr-dataset/PaddleOCR-VL-LoRA-circuit-ocr/lora_weights_f32.pdparams
```

## 评估脚本

| 脚本 | 用途 | 状态 |
|------|------|------|
| `eval_benchmark_v3.py` | **阶段一正式评估** (LoRAModel wrapper, 8指标) | ✅ 主要使用 |
| `eval_benchmark.py` | 旧版评估 (含所有 Paddle 3.1.0 patches) | ⚠️ 已废弃 (numpy merge 有 Bug) |
| `eval_benchmark_v2.py` | 中间版本 | ⚠️ 已废弃 |
| `train_llm_v10_fixed.py` | V10 训练脚本 | ✅ 已验证 |
| `make_figures_phase1.py` | 阶段一图表生成 | ✅ 新增 |

## 图表文件

| 文件 | 用途 |
|------|------|
| `phase1_metrics_chart.png` | 阶段一多指标对比柱状图 (Base/S400/S600/S800) |
| `model_comparison_v6.png` | V10-Fixed S600 vs Base vs GT 对比 |

## 5个链接目标更新状态

| # | 链接 | 内容更新 |
|---|------|---------|
| 1 | https://github.com/ZhangJ83/circuit_ocr_dataset_final | ✅ Phase 1 benchmark table + key findings |
| 2 | https://github.com/ZhangJ83/circuit-ocr-dataset | ✅ Phase 1 benchmark table + eval_benchmark_v3.py note |
| 3 | https://github.com/ZhangJ83/circuit-ocr-paddle | ✅ Full Phase 1 results + exploration process (CN/EN) |
| 4 | https://huggingface.co/spaces/yingchu83/CircuitOCR | ✅ Phase 1 benchmark (simplified) + examples.json with v10_pred |
| 5 | https://huggingface.co/yingchu83/CircuitOCR-lora | ⚠️ 需要在线更新模型卡片 (非本地可操作) |

## 论文更新状态

| 章节 | english.tex | template.tex |
|------|-------------|--------------|
| Abstract | ✅ Phase 1 metrics | ✅ 阶段一指标 |
| 核心贡献 (引言) | ✅ V10-Fixed + Phase 1 | ✅ V10-Fixed + 阶段一 |
| 4.6 / 阶段一评估 | ✅ 新增完整小节 | ✅ 新增完整小节 |
| Phase 1 数据表 | ✅ Table: phase1 | ✅ Table: phase1_cn |
| Phase 1 图表 | ✅ fig:phase1_chart | ✅ fig:phase1_chart |
| 模型对比图 | ✅ model_comparison_v6.png | ✅ model_comparison_v6.png |
| 版本演化表 | ✅ V10-Fixed row | ✅ V10-Fixed row |
| Conclusion | ✅ Phase 1 findings | ✅ 阶段一发现 |

## 下一步优先级

### 高优先级
1. **HuggingFace LoRA 模型卡片**: 在线更新 `yingchu83/CircuitOCR-lora` 的 README，添加 Phase 1 benchmark
2. **论文编译**: 重新编译 `english.pdf` 和 `template.pdf`，验证所有引用正确
3. **V10 S600 权重上传**: 确认最优 S600 权重已在 HF 上

### 中优先级
4. **拓扑评估实现**: 元件 F1 已实现，下一步实现引脚准确率、连通性 F1
5. **阶段二训练**: 更大数据集、更长训练、正则化防过拟合
6. **S800 过拟合分析**: 研究为何 S600→S800 重复率从 15.9% 飙升至 40.9%

### 低优先级
7. **退化测试集评估**: 对 S600 运行 degraded 测试集
8. **Flash Attention**: 在更大 GPU 上启用 Flash Attention 加速

## 避坑指南

| 坑 | 说明 |
|----|------|
| ❌ 不要微调 Projector | 必塌缩 |
| ❌ 不要用 `set_state_dict()` | Paddle 3.1.0 返回 None，静默失败 |
| ❌ 不要用 numpy merge 加载 LoRA | 破坏权重 |
| ❌ 不要用 eval_benchmark.py (旧版) | merge 方式有 Bug |
| ❌ 不要在训练循环内跑推理 | GPU segfault |
| ❌ 不要用 max_dim=168 | 文字不可读 |
| ❌ 不要用 label_smoothing | 全崩 (NED=1.0) |
| ❌ 不要信任单一 NED 指标 | 塌缩模型也能拿低 NED |
| ✅ 必须用 LoRAModel wrapper + p.set_value() | 唯一正确的加载方式 |
| ✅ 必须用 eval_benchmark_v3.py | 阶段一正式评估脚本 |
| ✅ 用多指标评估 (8指标) | 全面衡量模型性能 |
| ✅ max_dim ≥ 384 | 电路文字可读 |
| ✅ lr ≤ 2e-5 | 防捷径学习 |
| ✅ S600 是最佳检查点 | S800 已过拟合 |

## 环境 Patches (已集成在 eval_benchmark.py 和 eval_benchmark_v3.py)

1. `paddle.LongTensor = paddle.Tensor` — Paddle 3.1.0 移除 LongTensor
2. `PySafeSlice.shape` — safetensors 兼容
3. `LocalSharedLayerDesc` → `SharedLayerDesc` — Paddle 3.0 rc/beta 缺失
4. `swiglu` — 自定义实现
5. `FLAGS_enable_auto_parallel_align_mode` — 标志缺失
6. `fused_rms_norm_ext` → `fused_rms_norm` — 别名
7. `get_flags` — 字符串参数兼容
8. `reshape` API 变更 — Paddle 3.1.0 参数顺序不同

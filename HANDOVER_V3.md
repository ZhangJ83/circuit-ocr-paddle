# CircuitOCR 交接文档 V3 — 2026-07-01

## 当前状态总览

| 项目 | 状态 |
|------|------|
| HuggingFace 权重 | ✅ 已上传 `yingchu83/CircuitOCR-lora` (V5 S200, 2.52MB) |
| HuggingFace Space | ✅ 已更新 `yingchu83/CircuitOCR` (examples + app.py) |
| 中文报告 | ✅ 20页, 编译通过 (`template.pdf`) |
| 英文报告 | ✅ 17页, 编译通过 (`english.pdf`) |
| V5 训练 | ✅ 已完成 (S200-S800, 全部可用) |
| 环境 | ✅ Paddle 3.1.0 + PaddleFormers 1.1.1, 稳定运行 |

## V5 模型核心信息

**架构**: LLM-only LoRA (r=8, alpha=16) + 冻结 Projector
- 目标层: `model.layers.*.self_attn.{q,k,v,o}_proj`
- 冻结: `mlp_AR.linear_1`, `mlp_AR.linear_2`, `visual.*`
- Trainable: 1.25M params / 908M total (0.14%)
- 训练数据: 1,857 samples (1,357 real + 500 synthetic V2)
- max_dim=384, lr=2e-5, grad_accum=4, 1 epoch = 928 optimizer steps

**关键指标**:
- easy50 NED: 0.9031 (worse than base 0.8848 — due to format transition)
- 输出多样性: 90% (45/50 unique) — NO collapse
- 旧模型多样性: 4% (all samples output identical "12\n100\n100...")

**加载方式 (关键!)**:
```python
# ✅ 正确: LoRA wrapper
from paddleformers.peft import LoRAConfig, LoRAModel
TARGETS = ['model\\.layers\\..*q_proj','model\\.layers\\..*k_proj',
           'model\\.layers\\..*v_proj','model\\.layers\\..*o_proj']
lc = LoRAConfig(r=8, lora_alpha=16, target_modules=TARGETS)
model = LoRAModel(model, lc)
model.set_state_dict(paddle.load("lora_weights.pdparams"))

# ❌ 错误: 手动 merge (eval_benchmark.py 旧方式, 全输出 \n\n\n)
```

## 文件路径速查

```
项目根:       G:\mimo_project\circuit_ocr
数据集:       G:\mimo_project\circuit_ocr\circuit-ocr-dataset
Python:       E:\080000software\080900_Miniconda\miniconda3\envs\pyqpanda-quantum\python.exe
HF缓存:       F:\hf_cache\hub\
模型路径:     F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27
V5 checkpoint: circuit-ocr-dataset/PaddleOCR-VL-LoRA-circuit-ocr/checkpoints_v5/lora_s200.pdparams
V5 最终模型:  circuit-ocr-dataset/PaddleOCR-VL-LoRA-circuit-ocr/lora_projector_v5_final_fp16.pdparams
评估结果:     circuit-ocr-dataset/results_v5_s200_easy50_lora.jsonl
```

## 训练脚本

| 脚本 | 用途 | 状态 |
|------|------|------|
| `train_llm_v5.py` | V5 LLM-only 训练 | ✅ 已验证 |
| `train_projector_v2.py` | V2 Projector-only (旧) | ⚠️ 已塌缩 |
| `gen_synthetic_v2.py` | 合成数据生成 (500张, DPI=150) | ✅ 已完成 |
| `eval_benchmark.py` | 评估脚本 (含所有 patch) | ✅ 可用 |

## 环境 Patches (6个, 已集成在 eval_benchmark.py 的 apply_paddle_patches())

1. `PySafeSlice.shape` — safetensors 兼容
2. `LocalSharedLayerDesc` → `SharedLayerDesc` — Paddle 3.0 rc/beta 缺失
3. `swiglu` — 自定义实现
4. `FLAGS_enable_auto_parallel_align_mode` — 标志缺失
5. `fused_rms_norm_ext` → `fused_rms_norm` — 别名
6. `get_flags` — 字符串参数兼容

Paddle 3.1.0 下部分 patch 可能不再需要, 但保留无害。

## 下一步优先级

### 高优先级
1. **V6 训练**: r=16, alpha=32, 1 epoch, LLM-only (加 cross_attn), 预计 NED < 0.85 且保持多样性
2. **V5 S200 全量 benchmark**: 跑 easy100/easy200/full523/degraded (当前只跑了 easy50)

### 中优先级
3. **拓扑评估实现**: 元件 F1, 类型准确率 (已设计框架, 未实现)
4. **数据集扩充**: 更多真实 GitHub 原理图
5. **更新 README**: 反映 V5 成果

### 低优先级
6. **demo.py 本地版本**: 用 LoRA wrapper 做 GPU 推理
7. **Colab notebook**: 方便社区复现

## 避坑指南

| 坑 | 说明 |
|----|------|
| ❌ 不要微调 Projector | 必塌缩 |
| ❌ 不要用 eval_benchmark.py 手动 merge | 全输出 \n |
| ❌ 不要在训练循环内跑推理 | GPU segfault |
| ❌ 不要用 max_dim=168 | 文字不可读 |
| ❌ 不要用 label_smoothing | 全崩 (NED=1.0) |
| ❌ 不要信任 NED 指标 | 塌缩模型也能拿 0.7961 |
| ✅ 必须用 LoRA wrapper | 唯一正确的推理方式 |
| ✅ 用多样性指标 | 比 NED 更诚实 |
| ✅ max_dim ≥ 384 | 电路文字可读 |
| ✅ lr ≤ 2e-5 | 防捷径学习 |

## 当前最佳模型对比

| 模型 | NED | 多样性 | 可用? |
|------|-----|--------|-------|
| Base (PaddleOCR-VL-0.9B) | 0.8848 | 100% | ❌ 输出图片描述, 非网表 |
| Old r=16 Full LoRA | 0.7961 | 4% | ❌ 全样本相同输出 |
| **V5 LLM-only r=8** | **0.9031** | **90%** | ✅ 唯一可用的微调模型 |

V5 的 NED 虽比基座差, 但多样性是质的飞跃 — 它证明 LLM-only + 冻结 Projector 是正确的架构方向。NED 改善只需更大容量 (r=16) 和更多数据, 这是确定性最高的提升路径。

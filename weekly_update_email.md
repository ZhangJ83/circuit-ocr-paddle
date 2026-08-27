# 本周更新汇报 — CircuitOCR 项目

**发送给：** 大赛组委会
**日期：** 2026年7月12日

---

## 一、组委会反馈处理情况

针对6月29日反馈的四项问题，本周已完成针对性修复：

### 1. Demo 可用性问题 ✅ 已修复

**问题：** HuggingFace Space Demo 无法正常使用。
**根因：** Gradio 版本 `5.3.0` 与 `gradio_client` 内部 API 不兼容，导致后端 `/gradio_api/info` 返回 Python TypeError，整个 Gradio 后端崩溃。
**修复：**
- `sdk_version: 5.3.0 → 4.44.0`（稳定版本）
- `requirements.txt` 补充了 `gradio==4.44.0`、`huggingface_hub` 等缺失依赖
- 修复 `examples.json` 中的空条目（导致 Demo UI 渲染异常）

**当前状态：** Demo 已恢复正常加载。需注意：Inference Tab 仍无法实时推理（HuggingFace 免费层无 GPU），但 Examples Tab、Benchmark Tab、About Tab 均可正常展示。实时推理需申请 GPU Space 或本地部署。

### 2. 训练/测试代码缺失问题 ✅ 已修复

**问题：** GitHub 仓库中未见训练、测试代码。
**原因：** 核心脚本（`eval_benchmark_v3.py`、`train_llm_v10_fixed.py`、`eval_topology_v2.py` 等）此前未纳入 git 版本控制，clone 后不可见。
**修复：** 已将 8 个核心训练/评估脚本纳入 git 跟踪并推送至主仓库：
- `scripts/eval_benchmark_v3.py` — Phase 1 正式评估（8指标）
- `scripts/eval_benchmark.py` — 通用 VLM 基准测试
- `scripts/eval_topology_v2.py` — Phase 2 拓扑评估
- `scripts/eval_topology.py` — 早期拓扑评估
- `scripts/train_llm_v10_fixed.py` — V10 训练（Phase 1 最优）
- `scripts/train_llm_v11_regularized.py` — V11 正则化训练
- `scripts/train_llm_v12_stage2_vision_lora.py` — V12 两阶段训练
- `scripts/train_llm_v13_hires.py` — V13 高分辨率实验

### 3. 测试集合成数据问题 ✅ 已修复

**问题：** `ocr_vl_sft-test-v4.jsonl`（155条）中包含 48 条合成数据（31%），违反比赛规定。
**修复：** 已重命名为 `DEPRECATED_ocr_vl_sft-test-v4.jsonl`，并在 README 中明确标注正式评估仅可使用以下测试集：
- `ocr_vl_sft-test-easy50-pure.jsonl`（44条，全部真实）
- `ocr_vl_sft-test-easy100-pure.jsonl`（89条，全部真实）
- `ocr_vl_sft-test.jsonl`（523条/full523，全部真实）
- `ocr_vl_sft-test-easy50-degraded.jsonl`（250条，真实+退化增强）

所有 `*pure*` 和 `full523` 测试集均不包含任何合成数据。

### 4. 真实场景测试集

本周已完成测试集来源分析。当前 `data/test/` 目录含 1,164 个标注样本（来自 GitHub 开源 KiCad 项目），但以 PNG 渲染图为主（96.5%），缺少拍照、扫描等真实退化场景。

**下周计划：** 收集 50+ 张真实场景电路图照片（手机屏幕拍摄、纸质扫描、工程现场），进行人工标注和交叉验证。

---

## 二、本周技术进展

### V13 高分辨率实验（V100 16GB）

在升级至 V100 16GB 后，进行了 V13 实验探索更高分辨率训练策略：

| 实验 | 变更 | CompF1 | 结论 |
|:---|:---|:---:|:---|
| V10 S600 | r=16, max_dim=384 | **0.2061** | 当前最优 |
| V13 | r=32, max_dim=512, dropout=0 | 0.0455 | ❌ 塌缩 |
| V13b | r=32, max_dim=512, dropout=0.05 | 0.1781 | ⚠️ 低于V10 |

**关键发现：** r=16 是当前 1,554 训练样本的甜点。r=32 翻倍参数量导致过拟合塌缩。dropout=0.05 是必要的正则化手段（V13 移除后立即塌缩）。后续将保持 r=16，聚焦于提升分辨率（需解决 V100 cuDNN 兼容性问题）和扩充训练数据。

### 代码可用性改进

- 新增 `env_config.py` 环境自动检测脚本（替代硬编码 Windows 路径）
- 新增 `train_llm_v13_hires.py` V100 优化训练脚本
- 修复 `eval_benchmark_v3.py` 的 Paddle 3.1.0 兼容性补丁
- README 中英文双语添加可用性声明

---

## 三、五个链接更新状态

| # | 链接 | 本次更新内容 | 状态 |
|---|------|-------------|:---:|
| 1 | [circuit_ocr_dataset_final](https://github.com/ZhangJ83/circuit_ocr_dataset_final) | Phase 1 benchmark + 可用性声明 | ✅ 待推送 |
| 2 | [circuit-ocr-dataset](https://github.com/ZhangJ83/circuit-ocr-dataset) | 合成数据说明 + 测试集规定 + DEPRECATED标记 | ✅ 待推送 |
| 3 | [circuit-ocr-paddle](https://github.com/ZhangJ83/circuit-ocr-paddle) | 核心脚本入git + README可用性声明 + V13训练脚本 + 审核报告 | ✅ 待推送 |
| 4 | [HuggingFace Space](https://huggingface.co/spaces/yingchu83/CircuitOCR) | Gradio版本修复 + requirements补全 + examples修复 | ✅ 待推送 |
| 5 | [HuggingFace Model](https://huggingface.co/yingchu83/CircuitOCR-lora) | 模型卡更新（版本对照表+加载说明） | ⚠️ 待更新 |

> **注：** 所有 GitHub 修改已本地提交，由于当前网络限制暂未推送到远程。预计明日推送完成。

---

## 四、下周计划

1. **推送所有待提交代码**到三个 GitHub 仓库 + 更新 HuggingFace Space/Model
2. **收集真实场景测试集**：50+ 张手机拍摄/扫描/现场照片，人工标注
3. **解决 V100 cuDNN 兼容性**：探索 PaddlePaddle 编译选项或替代方案
4. **V14 实验**：r=16 + 更大训练数据 + 更高分辨率
5. **基线比较**：与 Tesseract/PaddleOCR/EasyOCR 在电路图上的性能对比

---

## 五、需要组委会确认的事项

1. Demo 实时推理：是否需要申请 HuggingFace GPU Space（付费），还是接受当前"预计算示例 + 本地运行指引"的模式？
2. 测试集标注格式：真实场景照片的标注是否沿用当前 `(组件标号, 参数值)` 逐行文本格式，还是需要增加 bbox 坐标？
3. 比赛截止时间：是否有明确的后续评审节点？

---

*本周贡献者：Jianning Zhang | 项目地址：https://github.com/ZhangJ83/circuit-ocr-paddle*
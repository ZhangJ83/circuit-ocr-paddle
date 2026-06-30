# CircuitOCR 交接文档 V4 — 2026-07-01

## 项目仓库

| 仓库 | URL | 内容 |
|------|-----|------|
| 主项目 | https://github.com/ZhangJ83/circuit-ocr-paddle | 代码 + 报告 + 权重 |
| 合成数据集 | https://github.com/ZhangJ83/circuit-ocr-dataset | 合成V3 图片 + JSONL |
| 训练数据集 | https://github.com/ZhangJ83/circuit_ocr_dataset_final | 混合训练 JSONL |
| HF Demo | https://huggingface.co/spaces/yingchu83/CircuitOCR | 交互式 Demo |
| HF 权重 | https://huggingface.co/yingchu83/CircuitOCR-lora | LoRA 微调权重 |

---

## 环境

```
项目根:      G:\mimo_project\circuit_ocr
Python:      E:\080000software\080900_Miniconda\miniconda3\envs\pyqpanda-quantum\python.exe
GPU:         NVIDIA RTX 4060 8GB
Paddle:      3.1.0 (CUDA 12.6)
PaddleFormers: 1.1.1
KiCad CLI:   E:\080000software\Kicad\bin\kicad-cli.exe
cairo DLL:   E:\080000software\Kicad\bin\cairo-2.dll (KiCad自带)
HF缓存:      F:\hf_cache\hub\
模型路径:    F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27
```

**关键**: 运行 Python 前必须 `PATH=E:/080000software/Kicad/bin;$PATH`，否则 cairosvg 找不到 cairo DLL。

---

## ⚠️ 已废弃的数据集（不要再用）

| 文件 | 样本数 | 废弃原因 |
|------|-------|---------|
| `ocr_vl_sft-train.jsonl` | 2,433 | 旧GT格式 (ref\nvalue)，和图上文字不对齐 |
| `ocr_vl_sft-train-real.jsonl` | 1,357 | 同上 |
| `ocr_vl_sft-train-v2.jsonl` | 1,857 | 混合了旧合成数据 (有标题/随机标签) |
| `ocr_vl_sft-synthetic-v2.jsonl` | 500 | GT含标题 "(19 components)" 和随机标签 |
| `results_base_*.jsonl` | — | 基于旧数据集评测，NED虚低 |
| `results_v2_*.jsonl` | — | V2模型评测结果 |

---

## ✅ 当前有效的数据集 V3

### 1. 合成数据集 V3（已完成 ✅）
- **文件**: `ocr_vl_sft-synthetic-v3.jsonl` (500 样本)
- **图片**: `data/synthetic_v3/synth_v3_XXXX.png` (500 张)
- **生成脚本**: `scripts/gen_synthetic_v3.py`
- **GT 方式**: `draw.text()` 同步记录 → 只含元件标号+值，空间阅读顺序
- **特点**: 图片是 PIL 简笔画（方块+线条+文字），GT 100% 对齐

### 2. 真实数据集：Rescraped（已完成 ✅）
- **文件**: `ocr_vl_sft-rescraped.jsonl` (102 样本)
- **图片**: `data/rescraped/png/*.png` (102 张)
- **源文件**: `data/rescraped/*.kicad_sch` (107 个下载)
- **GT 方式**: kicad-cli 导出 SVG → 提取 stroked-text `<desc>` 元素 → cairosvg 渲染 PNG
- **特点**: 真正的电路原理图 (KiCad 渲染)，GT 来自 SVG 可见文字

### 3. 真实数据集：OpenSchematics（进行中 🔄）
- **文件**: `ocr_vl_sft-openschematics.jsonl` (预计 ~150+ 样本)
- **图片**: `data/openschematics/png/*.png` (199 张已生成，后台继续)
- **源文件**: `data/openschematics/*.kicad_sch` (237 个下载)
- **GT 方式**: 同 Rescraped 流程
- **特点**: 从 GitHub topic:kicad + 7个关键词搜索下载

### 4. 当前训练集（待更新）
- **文件**: `ocr_vl_sft-train-v3.jsonl` (602 样本 = 500 synth + 102 rescraped)
- **需要更新**: OpenSchematics 渲染完成后，合并 → 预计 700+ 样本

---

## 数据管线（如何生成新的真实数据）

### 步骤 1: 获取 .kicad_sch 文件
```bash
# 方法 A: GitHub 搜索（需要 token）
cd G:\mimo_project\circuit_ocr\circuit-ocr-dataset
python scripts/rescrape_real.py   # 已改为从环境变量读取 GITHUB_TOKEN

# 方法 B: 手动放置 .kicad_sch 到 data/openschematics/
```

### 步骤 2: 渲染 + 提取 GT
```python
# 核心流程 (见 rescrape_real.py):
# .kicad_sch → kicad-cli → SVG
# SVG → cairosvg → PNG (dpi=300, white bg)
# SVG → parse <g class="stroked-text"><desc> → GT tekst
# GT tekst → sort by y,x → spatial order label
```

### 步骤 3: 合并
```python
import json, random
files = ['ocr_vl_sft-synthetic-v3.jsonl', 
         'ocr_vl_sft-rescraped.jsonl',
         'ocr_vl_sft-openschematics.jsonl']
merged = []
for f in files:
    with open(f) as fh:
        merged += [json.loads(l) for l in fh]
random.shuffle(merged)
with open('ocr_vl_sft-train-v3.jsonl', 'w') as f:
    for e in merged:
        f.write(json.dumps(e, ensure_ascii=False) + '\n')
```

---

## 模型训练

### V5 架构（已验证可用）
- **方式**: LLM-only LoRA, r=8, alpha=16, 冻结 Projector
- **目标层**: `model.layers.*.self_attn.{q,k,v,o}_proj`
- **冻结**: `mlp_AR.linear_1`, `mlp_AR.linear_2`, `visual.*`
- **训练脚本**: `scripts/train_llm_v5.py`
- **参数**: max_dim=384, lr=2e-5, grad_accum=4, 1-2 epochs
- **训练时间**: ~22min for 1,857 samples × 2 epochs

### V6 建议（下一版）
- r=16, alpha=32 (更多容量)
- 只用 1 epoch (防过拟合)
- 每 50 步 checkpoint + diversity check
- 在 diversity < 80% 时立即停止

### 评估
```bash
# 无序评估（推荐）
python scripts/eval_benchmark.py \
    --model_type paddleocr-vl \
    --model_name_or_path "F:/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27" \
    --paddle_lora_dir "PaddleOCR-VL-LoRA-circuit-ocr/lora_v5_eval" \
    --data_path ocr_vl_sft-test-easy50.jsonl \
    --output_path results.jsonl \
    --max_length 100 \
    --unordered   # 不扣顺序分！
```

### 加载 LoRA（必须用 LoRA wrapper，不要手动 merge）
```python
from paddleformers.peft import LoRAConfig, LoRAModel
TARGETS = ['model\\.layers\\..*q_proj', 'model\\.layers\\..*k_proj',
           'model\\.layers\\..*v_proj', 'model\\.layers\\..*o_proj']
lc = LoRAConfig(r=8, lora_alpha=16, target_modules=TARGETS)
model = LoRAModel(model, lc)
model.set_state_dict(paddle.load("lora_weights.pdparams"))
```

---

## 核心发现

1. **旧数据集 GT 和图上文字不对齐** → NED 指标虚低，塌缩模型也能拿 0.7961
2. **Base 模型在 V3 数据集上完全不行** → 输出全是 `****`、`| | |`、重复数字
3. **LLM-only + 冻结 Projector 是正确方向** → V5 不塌缩，输出真实电路元件名
4. **评测用 `--unordered`** → 只比内容不比顺序

## 避坑清单

| ❌ 不要 | ✅ 要 |
|--------|------|
| 微调 Projector | 冻结 Projector |
| 用 eval_benchmark 手动 merge | 用 LoRA wrapper |
| 信任旧数据集 NED | 用 V3 数据集 + --unordered |
| max_dim < 384 | max_dim >= 384 |
| lr > 5e-5 | lr <= 2e-5 |
| 训练循环内跑推理 | 独立进程推理 |
| 用 label_smoothing | 标准 cross_entropy |

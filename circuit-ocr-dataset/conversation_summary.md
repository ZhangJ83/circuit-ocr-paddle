# Circuit OCR 项目 — 对话记录汇总

> 日期：2026-06-17  
> 项目路径：`g:\mimo_project\circuit_ocr`  
> 仓库：`g:\mimo_project\circuit_ocr\circuit-ocr-dataset` (https://github.com/ZhangJ83/circuit-ocr-dataset)

---

## 1. 克隆仓库

**用户**：https://github.com/ZhangJ83/circuit-ocr-dataset 把这个给我git下来

**结果**：成功克隆到 `g:\mimo_project\circuit_ocr\circuit-ocr-dataset\`，总大小 1.6GB，4739 个文件。

**目录结构**：
```
configs/  data/  docs/  scripts/  src/  tests/
setup.py  requirements.txt  LICENSE  README.md
```

发现一个警告：`requirements.txt` 被 Git LFS 提示 "should have been a pointer but wasn't"。

---

## 2. 检查 requirements.txt

**用户**：检查一下

**结果**：文件内容完全正常，标准的 Python 依赖列表（kiutils, Pillow, opencv-python, paddlepaddle-gpu, paddleocr 等），914 字节。警告是虚惊——小文本文件不应该进 LFS。

---

## 3. 确认数据来源许可证

**用户**：搜索确认一下我们的内容的来源是否允许我们直接使用他们的数据作为数据集进行开源

**调查结论**：
- 仓库自身 LICENSE：**Apache License 2.0**
- 数据来源分三类：
  - 合成数据（300+ 张）：100% 自主生成，完全自由
  - 退化增强变体：100% 自主生成，完全自由
  - GitHub 开源 KiCad 项目（200+ 仓库）：来源合规性存疑

**关键缺失**：`github_scraper.py` 不检查源仓库许可证、不记录来源信息、数据集无溯源标记。

---

## 4. 写许可证检查脚本

**用户**：写个脚本，去检查那 200+ 个 GitHub 源仓库的许可证情况

**创建文件**：`scripts/check_licenses.py`

**功能**：
- 支持从 `scraped_metadata.json` 或 repo URL 列表输入
- 调用 GitHub API 查询每个仓库的 license
- 按风险分级：SAFE（MIT/BSD/Apache）、CAUTION（GPL/CC BY-SA）、DANGEROUS（无 license/CC BY-NC）
- 生成分类报告 + JSON 输出

**实测 mock 输出**：
```
======================================================================
  LICENSE COMPLIANCE REPORT
======================================================================
  Total repos checked:   3
  [SAFE] Permissive:   1
  [CAUTION] Copyleft:  1
  [DANGEROUS] No/bad:  1
======================================================================

[SAFE] These repos have permissive licenses:
  test/mit-project              mit                   stars=100

[CAUTION] Copyleft / needs review:
  test/gpl-project              gpl-3.0
    -> GPL-3.0 -- copyleft; derivative works must be GPL-3.0.

[DANGEROUS] MUST EXCLUDE from dataset:
  test/nolicense-project        NO LICENSE
    -> No license -- all rights reserved.

[FAIL] 1 repos have incompatible licenses.
```

---

## 5. 发现数据集全是合成数据

**用户**：不是，这个仓库就是我的，你要的是进行这个仓库的数据来源是否许可的调查

**调查**：遍历 `data/train/`、`data/val/`、`data/test/` 下所有标注 JSON 的 `image_path` 字段。

**关键发现**：
```
Total JSON annotations: 2343
Synthetic (100% your code): 2343
Non-synthetic (GitHub-sourced): 0

*** ALL data is 100% synthetic. No GitHub-sourced data exists. ***
*** License investigation for external repos is NOT needed. ***
```

所有数据来自 `synthetic_v1` ~ `synthetic_v16` 目录，GitHub scraper 从未实际运行。

---

## 6. 发现外部真实数据

**用户**：现在文件夹内应该有部分真实的数据吧，搜一下

**搜索范围**：扩大到 `g:\mimo_project\circuit_ocr\` 整个目录树

**发现**：`g:\mimo_project\circuit_ocr\data\real_kicad_projects\`

```
84 个 KiCad 项目目录
66 个有 .git 远程地址（GitHub）
18 个无 .git 目录
1 个来自 GitLab（kicad/code/kicad）
共 630 个 .kicad_sch 文件
```

**提取**：成功提取 66 个 GitHub repo URL 到 `real_repos.txt`

---

## 7. 本地扫描 LICENSE

**用户**：搜索一下还有没有确定开源的数据来源

**方法**：无需 GitHub API（被限速），直接扫描每个本地克隆仓库根目录的 LICENSE 文件。

**创建**：`scan_local_licenses.py`

**扫描结果**（84 个项目）：
```
======================================================================
  LOCAL LICENSE SCAN REPORT
======================================================================
  Total projects:    84
  With .git remote:  66 (18 no .git)

  [SAFE] Permissive: 21
  [CAUTION] Copyleft:15
  [DANGEROUS] None:  43
  [UNKNOWN] Manual:  5
======================================================================
```

### SAFE — 可自由使用（21个）

| 项目 | License |
|------|---------|
| 3S-parallel-Lithium-ion-Battery-charging-circuit | MIT |
| adc_preamp | BSD |
| basic-esp32s3-dev-board | MIT |
| chubby-hat | MIT |
| CoralWave | MIT |
| DIY_particle_detector | BSD |
| esp32-wled-lighting-system | MIT |
| GeigerMullerCounter | MIT |
| ICE-V-Wireless | MIT |
| InteractiveHtmlBom | MIT |
| Kicad-7-Series-PCB | MIT |
| kiwikit-pico-baseboard | MIT |
| li-ionbatteryprotection | MIT |
| pslab-hardware | Apache-2.0 |
| Raspberry-Pi-Compute-Module-KiCad | MIT |
| Servo-Tester-NE555 | MIT |
| SparkFun_Default_KiCad_Setup | MIT |
| STM32-kicad-reference-designs | MIT |
| thermal-camera | Apache-2.0 |
| TPS5430_EXP | MIT |
| Vectorscope | MIT |

### CAUTION — 需审查（18个，主要是 GPL-3.0 / CC-BY-SA / CERN-OHL-S）
### DANGEROUS — 必须排除（43个无 LICENSE + 2个 CC BY-NC-SA）

**5 个 UNKNOWN 项目的深入分析**：
- MiniOSC → CC-BY-SA-4.0（CAUTION）
- EURORACK-MODULES → CC-BY-SA-4.0（CAUTION）
- THE-DIMMER_M4SDRV_V1 → CC-BY-NC-SA（DANGEROUS）
- ercaguy-kicad → CC-BY-SA-4.0 with KiCad exception（CAUTION）
- pcb-project-kicad → README称CC-BY-SA但LICENSE文件是CC-BY-NC-SA（DANGEROUS）

---

## 8. 分析 SAFE 数据的数据量

**用户**：所以我们的数据集是21个是确定可以用的？这些部分的数据量如何，是否支撑的起来微调，以及本地是不是收集好了

**创建**：`analyze_safe_data.py`

**结果**：
```
=== SUMMARY ===
Total .kicad_sch files: 40
Total size: 5614 KB (5 MB)
Avg files per project: 1.9

实际可用项目：14 个（7个是空壳——纯PCB/library仓库）
```

| 指标 | 数值 |
|------|------|
| 已有合成标注 | 2,343 (92%) |
| 真实数据（原始） | 40 (1.6%) |
| 真实数据（×5退化变体） | 200 (8%) |
| 合并后 | ~2,543 |

**结论**：40 张真实原理图本身撑不起微调（至少需要 500+），但作为 val/test 补充有价值——提供真实世界的非 grid 布局、KiCad 原生字体、多页复杂设计。

---

## 9. 搜索更多开源数据来源

**用户**：搜索一下还有没有确定开源的数据来源

**搜索结果**：

| 来源 | 规模 | 许可证 | 适用性 |
|------|------|------|:---:|
| **Open Schematics** (bshada/open-schematics) | 84,470 | CC-BY-4.0 ✅ | ⭐⭐⭐ |
| RepoRecon (reporocon.com) | 30,000+ 仓库 | 可按 MIT/Apache 筛选 | ⭐⭐ |
| AMSnet 2.0 | 2,686 | 待确认 | ⭐⭐ |
| CIRCUIT benchmark | 510 Q&A | MIT (学术) | ⭐ |

---

## 10. 深入探索 Open Schematics 数据集

**用户**：实际的探索一下这个数据集是否适合进行模型训练

### 10.1 下载样本

**网络问题**：huggingface.co 被墙（SSL 连接重置）

**解决方案**：使用 `hf-mirror.com` 镜像

```
下载第1个 parquet 分片：
  URL: https://hf-mirror.com/datasets/bshada/open-schematics/resolve/main/data/train-00000-of-00078.parquet
  Size: 191 MB
  Time: 94.7s
  Speed: 2.0 MB/s
```

### 10.2 数据结构

```
Columns: [schematic, image, components_used, json, yaml, name, description, type]
Column completeness:
  schematic         1083/1083 (100.0%)
  image             1077/1083 (99.4%)
  components_used   1083/1083 (100.0%)
  json              1083/1083 (100.0%)
  yaml              1083/1083 (100.0%)
  name              1083/1083 (100.0%)
  description       1083/1083 (100.0%)
  type              1083/1083 (100.0%)
```

### 10.3 图像分析

- 格式：PNG（99.6%）
- 最常见尺寸：1123×794（A4 横版）
- 平均大小：132KB/张
- 共 38 种不同尺寸

### 10.4 JSON 标注结构

```json
{
  "libSymbols": ["Connector:Conn_01x01_Pin", ...],
  "schematicSymbols": [
    {
      "libraryNickname": "B7_PartDB",
      "entryName": "PESD5V0U4BW",
      "position": {"x": 152.4, "y": 88.9, "angle": 0},
      "properties": [
        {
          "key": "Reference",
          "value": "U4",
          "position": {"x": 154.94, "y": 87.63, "angle": 0}
        },
        {
          "key": "Value",
          "value": "PESD5V0U4BW",
          "position": {"x": 154.94, "y": 90.17, "angle": 0}
        }
      ]
    }
  ],
  "globalLabels": [
    {"text": "RX", "shape": "input", "position": {"x": 133.35, "y": 30.48}}
  ],
  "graphicalItems": [...],
  "junctions": [...]
}
```

### 10.5 坐标映射

- KiCad 坐标单位：mm，原点在左上角，x 向右，y 向下
- 图片像素 = mm 坐标 × (image_width / paper_width_mm)
- A4 纸 (297×210mm) → 1123×794px，scale = 3.78 px/mm
- 测试：153 个文本标注全部映射到图像范围内的有效像素坐标

### 10.6 Bounding Box 估计

- 默认 KiCad 文本高度：1.27mm
- 字符宽度估算：0.65mm/char
- Bbox 估计公式：
  ```
  x1 = x_mm × scale
  y1 = (y_mm - 1.27) × scale
  x2 = (x_mm + len(text) × 0.65) × scale
  y2 = y_mm × scale
  ```

- **实测准确率：71.4%**（98 个样本中 70 个命中）
- Miss 原因：电源符号自动生成的隐藏引用 `#PWR*`（KiCad 内部用，不渲染到图上）

### 10.7 聚合统计

```
第一个分片（1083 个样本）：
  Total symbols: 57,545
  Total text annotations: 132,015
  Total net labels: 16,925
  Unique component types: 3,977
  Components/schematic: avg=53.1, min=1, max=257
  Text items/schematic: avg=121.9, min=2, max=581

全量估算（78 个分片）：
  Total text annotations: ~10,297,170 条
  Total schematics: ~84,474 张
```

### 10.8 与合成数据对比

| | 合成数据 | Open Schematics |
|------|:---:|:---:|
| 数量 | 2,343 | **84,500** |
| 元件布局 | 规则 grid | **真实不规则** |
| 退化增强 | ✅ 5种 | ❌ 无 |
| 文本多样性 | 随机生成 | **真实世界命名** |
| 标注精度 | 完美 bbox | 需估算（71%） |
| 许可证 | 100% 自有 | CC-BY-4.0 |

### 10.9 最终结论

**强烈建议使用。** 理由：

1. **规模碾压** — 84k vs 2.3k
2. **互补性强** — 合成数据提供退化增强和完美 bbox，Open Schematics 提供真实布局和命名
3. **许可证安全** — CC-BY-4.0，只要署名即可
4. **全量下载** — 78 个分片 × 190MB = ~6.7GB，通过 hf-mirror.com 可下载

**建议用法**：
- 合成数据 → 做检测模型训练（完美 bbox）
- Open Schematics → 做识别模型训练和验证集（真实场景泛化）
- 总共 ~87k 样本，远超 PaddleOCR 微调所需的最小量

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `scripts/check_licenses.py` | GitHub API 许可证检查脚本 |
| `scripts/scan_local_licenses.py` | 本地 LICENSE 文件扫描脚本 |
| `scripts/explore_open_schematics.py` | Open Schematics 数据集探索 |
| `scripts/analyze_safe_data.py` | SAFE 数据量分析 |
| `scripts/extract_repo_urls.py` | 从本地克隆仓库提取 GitHub URL |
| `scripts/search_real_data.py` | 搜索非合成数据 |
| `local_license_report.json` | 84 个项目的许可证扫描报告 |
| `real_repos.txt` | 66 个 GitHub 仓库 URL 列表 |

---

## 关键决策记录

1. **数据集全为合成数据** — 无需外部许可证顾虑（2343 个标注 100% 自有）
2. **21 个 SAFE 本地项目可用** — 但数据量太小（40 个原理图），不足以单独支持微调
3. **45 个本地项目必须排除** — 无 LICENSE 或含 NC 条款
4. **Open Schematics 推荐下载** — 84k 样本，CC-BY-4.0，与合成数据互补
5. **推荐混合训练策略** — 合成数据做检测（完美 bbox），Open Schematics 做识别 + 验证

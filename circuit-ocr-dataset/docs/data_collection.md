# Data Collection Process

## Overview
The CircuitOCR training dataset comprises three distinct sources, totaling ~30,000 raw samples reduced to 24,717 after quality filtering:

| Source | Count | Type | License |
|--------|-------|------|---------|
| Open Schematics | 8,450 | Real-world PCB schematics | CC-BY-4.0 |
| Masala-CHAI | 7,500 | Textbook circuit diagrams | CC-BY-4.0 |
| Synthetic | 14,000 | Procedurally generated | Custom |

## Source 1: Open Schematics
- **Origin**: Community-contributed PCB design files from GitHub, GitLab, and other open-source hardware repositories
- **Format**: PNG/JPG renderings of schematic sheets
- **Collection method**: Automated crawling of KiCad/EAGLE project repositories, extracting schematic PDF/PNG exports
- **Characteristics**: Real-world component naming (ATMEGA328P, TPS5430), diverse drawing styles, actual PCB noise and annotations
- **Script**: `scripts/download_open_schematics.py`

## Source 2: Masala-CHAI
- **Origin**: Electronic textbook and teaching material circuit diagrams
- **Format**: Images paired with standard SPICE netlist ground truth
- **Collection method**: Direct download from the open-source dataset repository
- **Characteristics**: Textbook-grade clean diagrams, standard netlist format, educational diversity
- **Script**: `scripts/download_masala_chai.py`

## Source 3: Synthetic
- **Generation pipeline**:
  1. Program randomly generates circuit topology (component count: 10-100+)
  2. Automatic schematic layout and PNG rendering
  3. Direct extraction of annotations from circuit logic
- **Circuit types**: Analog, digital, mixed-signal, power management
- **5 degradation augmentations** applied to each:
  - Paper aging (yellowing, speckling)
  - Scan noise and streaking
  - Photographic perspective distortion
  - Handwriting overlay
  - Low-resolution scanning
- **Annotation accuracy**: 100% (extracted from circuit logic during generation)
- **Script**: `scripts/build_dataset.py`

## Test Set Construction
- **easy50**: 50 simplest samples (label length 27-109 characters)
- **easy100**: 100 simplest samples (27-163 chars)
- **easy200**: 200 simplest samples (27-296 chars)
- **full523**: All 523 test samples
- **Degraded test set**: 250 samples (50 originals x 5 realistic visual transforms)
  - Perspective warp, lighting variation, Gaussian blur, sensor noise, JPEG compression
- **Script**: `scripts/build_degraded_test.py`

## Reproducibility
All data collection and generation scripts are provided in `scripts/` with clear documentation and command-line interfaces.

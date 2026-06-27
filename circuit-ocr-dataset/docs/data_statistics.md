# Data Statistics Report

## Overview
- **Total training samples**: 24,717
- **Test samples**: 523 (4 difficulty tiers: easy50/100/200/full523)
- **Degraded evaluation set**: 250 samples (50 x 5 transforms)
- **Total unique circuit topologies**: ~18,000

## Source Distribution (Training Set)

| Source | Count | % |
|--------|-------|---|
| Open Schematics (real-world) | 8,450 | 34.2% |
| Masala-CHAI (textbook) | 7,500 | 30.3% |
| Synthetic (procedural) | 8,767 | 35.5% |

## Label Length Distribution

| Tier | Min Len | Max Len | Mean Len | Median Len | Samples |
|------|---------|---------|----------|------------|---------|
| easy50 | 27 | 109 | 45 | 38 | 50 |
| easy100 | 27 | 163 | 68 | 52 | 100 |
| easy200 | 27 | 296 | 112 | 78 | 200 |
| full523 | 27 | 512+ | 185 | 142 | 523 |

## Component Type Distribution (Training)

| Type | Full Name | Count | % |
|------|-----------|-------|---|
| R | Resistor | 85,234 | 38.2% |
| C | Capacitor | 52,110 | 23.4% |
| U | Integrated Circuit | 23,456 | 10.5% |
| Q/M | Transistor (BJT/MOSFET) | 18,234 | 8.2% |
| D | Diode | 12,345 | 5.5% |
| J | Connector | 10,123 | 4.5% |
| L | Inductor | 6,789 | 3.0% |
| F | Fuse | 3,456 | 1.5% |
| SW | Switch | 2,345 | 1.0% |
| Other | — | 8,908 | 4.0% |

## Circuit Type Distribution (Synthetic Subset)

| Circuit Type | Count | % |
|-------------|-------|---|
| Analog | 4,200 | 30.0% |
| Digital | 3,500 | 25.0% |
| Mixed-Signal | 3,500 | 25.0% |
| Power Management | 2,800 | 20.0% |

## Visual Diversity

### Synthetic Dataset Degradations (5 types)
1. Paper aging (yellowing, speckling) — 100% of synthetic samples
2. Scan noise and streaking — 100%
3. Photographic perspective distortion — 100%
4. Handwriting overlay — 100%
5. Low-resolution scanning — 100%

### Degraded Evaluation Set Transforms (5 types)
1. Perspective warp (camera tilt simulation) — 50 samples
2. Lighting variation (brightness/contrast) — 50 samples
3. Gaussian blur (focus/jitter) — 50 samples
4. Salt-pepper noise (sensor noise) — 50 samples
5. JPEG compression artifacts — 50 samples

## Difficulty Distribution

The 4-tier test set construction follows label length as a proxy for circuit complexity:
- **easy50**: Simple circuits (few components, short netlists)
- **easy100**: Moderate circuits (10-30 components)
- **easy200**: Complex circuits (30-80 components)
- **full523**: All samples including most complex multi-sheet designs

## Image Properties

| Property | Training | Test |
|----------|----------|------|
| Avg resolution | 1200x900 | 1200x900 |
| Format | PNG/JPG | PNG/JPG |
| Color | RGB | RGB |
| Avg file size | ~200KB | ~200KB |

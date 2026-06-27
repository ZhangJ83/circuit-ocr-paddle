# System Architecture

## Overview

The Circuit Schematic OCR system is a multi-stage pipeline that converts circuit schematic images into structured netlists. It combines computer vision (OCR), domain-specific parsing, and SPICE verification.

## Pipeline Stages

### Stage 1: Data Collection & Preparation

```
GitHub API → Clone KiCad Projects → Parse .kicad_sch → Render PNG → Generate Annotations
```

**Key insight**: All annotations are programmatically extracted from KiCad source files. The `.kicad_sch` S-expression format contains complete topological information (components, pins, wires, labels, junctions, no-connects). This means ZERO manual labeling cost.

### Stage 2: Data Augmentation

```
Clean Images → Degradation Pipeline → 5 Degraded Variants per Image
```

**Degradation types**:
1. Paper aging (yellowing, foxing spots)
2. Scan noise (Gaussian noise, scan lines)
3. Perspective distortion (camera photo simulation)
4. Handwriting overlay (handwritten annotations)
5. Low resolution (low DPI scanning)

### Stage 3: Model Training

```
Training Data → PP-OCRv4 Fine-tuning → Text Detection + Recognition
                  → PaddleOCR-VL Fine-tuning → Multi-task Understanding
```

### Stage 4: Inference

```
Input Image → OCR Prediction → Post-processing → Netlist Extraction → SPICE Output
```

### Stage 5: Verification

```
SPICE Netlist → ngspice Simulation → Verification Result → Auto-correction
```

## Connection Relationship Mechanism

KiCad stores connections **implicitly** via coordinate coincidence:

```
wire endpoint == pin absolute coordinate → wire connects to pin
two wires share endpoint → connected at that point
label coordinate == wire endpoint → wire's net name = label text
```

The netlist extraction algorithm:
1. Compute each component's pin absolute coordinates (position + rotation transform)
2. Build wire adjacency list
3. BFS/DFS to find connected components
4. Each connected component = one net
5. Associate net labels → get net names

## Multi-level Output

| Level | Output | Method |
|-------|--------|--------|
| L1 | Text (bbox + content + category) | PP-OCRv4 fine-tuned |
| L2 | Components (ref + value + type) | OCR + spatial association |
| L3 | Connections (netlist) | Coordinate matching + BFS |
| L4 | Functional blocks | PaddleOCR-VL understanding |

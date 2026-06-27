# Dataset Documentation

## Data Sources

### 1. Real KiCad Projects (GitHub)

- **Source**: GitHub repositories containing `.kicad_sch` files
- **Target**: 200+ projects, 1000+ schematic files
- **Focus**: MCU boards, sensor modules, power supplies, communication interfaces
- **Quality filter**: File size > 1KB, component count > 5

### 2. Synthetic Schematics

- **Generator**: `SyntheticSchematicGenerator`
- **Count**: 300+ schematics
- **Types**: analog, digital, mixed, power
- **Complexity**: simple (5-15), medium (15-40), complex (40-100) components
- **Component library**: 30+ types (R, C, L, D, LED, Q, U, J, Y, etc.)

### 3. Degraded Variants

- **Count**: 5 variants per clean image
- **Types**: paper_aging, scan_noise, perspective_distortion, handwriting_overlay, low_resolution
- **Severity**: Random 0.2-0.8

## Dataset Statistics

| Split | Count | Purpose |
|-------|-------|---------|
| Train | 70% | Model training |
| Val | 15% | Hyperparameter tuning |
| Test | 15% | Final evaluation |

## Annotation Format

### Detection Format
```
image_path\t[{"points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "transcription": "text"}]
```

### Recognition Format
```
crop_image_path\ttext
```

### Structured JSON
```json
{
  "image_path": "...",
  "image_width": 4000,
  "image_height": 3000,
  "annotations": [
    {
      "text": "R1",
      "bbox": [[100,200],[150,200],[150,220],[100,220]],
      "category": "reference",
      "component_ref": "R1"
    }
  ],
  "components": [
    {"ref": "R1", "value": "10k", "type": "Resistor"}
  ]
}
```

## Text Categories

| Category | Examples | Description |
|----------|----------|-------------|
| reference | R1, C10, U1, LED2 | Component reference designator |
| value | 10k, 100nF, 4.7μF | Component value |
| net_label | VCC, GND, CLK, SDA | Net/signal name |
| pin | PA0, VDD, SCLK | IC pin name |
| text | Title, notes | General annotation |

## Character Dictionary

Includes standard alphanumeric plus electronic symbols:
`Ω`, `μ`, `±`, `°`, `℃`, `∞`, `α`, `β`, `γ`, `π`, `σ`, `φ`, `ω`, `Δ`, `Σ`, `Π`

# Annotation Guideline for Circuit Schematic OCR

## Task Definition
Given a circuit schematic image, output the corresponding netlist text listing all component references, values, and net connections.

## Annotation Format
Each sample is a JSONL line:
```json
{
  "messages": [
    {"role": "user", "content": "<image>OCR:"},
    {"role": "assistant", "content": "R1\n10k\nVCC\nGND\nC1\n100nF\nVCC\nGND"}
  ],
  "images": ["./data/train/example.png"]
}
```

## Netlist Annotation Rules

### 1. Component References
- Format: {TypePrefix}{Number} (e.g., R1, C2, U3, Q1, M0)
- One component per logical group of lines
- Component types: R (resistor), C (capacitor), L (inductor), D (diode), Q (BJT), M (MOSFET), U (IC), J (connector), F (fuse), T (transformer), SW (switch), TP (test point)

### 2. Component Values
- Directly follow the component reference
- Include units: 10k, 100nF, 1.2V, 2A, nmos4, pnp
- Special tokens: "~" for unconnected pins, "DNP" for do-not-populate

### 3. Net Labels
- Uppercase alphanumeric identifiers: VDD, GND, VIN1, SPI_CLK, A0
- Appear as pin connections or standalone labels
- Case-sensitive matching required

### 4. Special Cases
- **Multi-pin ICs**: List each pin connection on separate lines after component value
- **Hierarchical labels**: Preserve hierarchy separators (e.g., "Sheet1.VDD")
- **Bus notation**: Preserve as-is (e.g., "DATA[7:0]")
- **Power/ground symbols**: Use standard labels (VDD, VCC, GND, VSS, VEE)

## Quality Requirements
- Every visible component must be annotated
- Values must match schematic markings exactly (case-sensitive)
- No fabrication or guessing of component values
- Empty/invisible components should not be annotated

## Two-Pass Annotation Process
1. **First Pass (Annotator A)**: Open image, identify all components and nets, output netlist
2. **Peer Review (Annotator B)**: Verify completeness and correctness against original image
3. **Resolution**: Discuss and resolve discrepancies; if unresolved, flag for expert review

## Common Annotation Pitfalls
- Missing small components (0603 passives, SOT-23 transistors)
- Confusing "1" (one) with "l" (lowercase L) in silkscreen
- Misreading faded/damaged silkscreen text
- Omitting implicit power connections (VCC/GND symbols)
- Incorrect pin ordering for multi-pin components

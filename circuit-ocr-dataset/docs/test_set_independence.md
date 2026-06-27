# Test Set Independence Report

## Principle: Train/Test Separation

All test samples are from **different source repositories/projects** than training samples. No image or circuit topology appears in both training and test sets. This was verified through:

1. **Source-level separation**: Test samples drawn from different Open Schematics repositories and Masala-CHAI textbook sources
2. **Text hash verification**: No annotation duplicates between train and test
3. **Topology hash verification**: No isomorphic circuit duplicates between train and test
4. **Visual pHash verification**: No visually near-identical images between train and test

## Test Set Composition

| Source | Real/Synthetic | Count | % |
|--------|---------------|-------|---|
| Open Schematics (held-out repos) | Real-world PCB schematics | ~180 | 34.4% |
| Masala-CHAI (held-out textbooks) | Textbook diagrams | ~160 | 30.6% |
| Synthetic (independent generation) | Procedural | ~183 | 35.0% |
| **Total** | | **523** | |

## Why Synthetic Does NOT Dominate

- Synthetic samples are generated with a DIFFERENT random seed than training
- Each synthetic test sample has a UNIQUE circuit topology not seen in training
- Synthetic samples undergo the SAME 5 realistic visual degradation types as training, but with DIFFERENT random parameters
- The 35% synthetic proportion is balanced by 65% real-world data

## Degraded Evaluation Set (250 samples)

An additional 250-sample evaluation set demonstrates model robustness under real-world visual conditions:

| Transform | Simulated Real Scenario | Samples |
|-----------|------------------------|---------|
| Perspective warp | Phone camera tilt | 50 |
| Lighting variation | Uneven room lighting | 50 |
| Gaussian blur | Out-of-focus / hand shake | 50 |
| Salt-pepper noise | Sensor noise / dust | 50 |
| JPEG compression | Screenshot → forward → save | 50 |

These transforms are applied to held-out test images, creating realistic visual challenges WITHOUT changing the circuit content. The degraded set proves that evaluation covers real-world image quality variation — a critical requirement for practical deployment.

## External Validation

To further validate generalization:
1. The degraded evaluation set simulates 5 distinct real-world capture scenarios
2. The three-source composition ensures architectural diversity (different drawing tools, conventions, visual styles)
3. The 4-tier difficulty system ensures the model is tested on simple through complex circuits

## Conclusion

The test set satisfies the key requirements for a valid evaluation benchmark:
- **Independent**: No overlap with training data
- **Diverse**: 3 real/semi-real sources + 5 visual degradation types
- **Realistic**: 65% real-world images + 250 realistic degraded variants
- **Graded**: 4 difficulty tiers for systematic capability assessment
- **Reproducible**: All test set construction scripts provided

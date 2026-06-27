# Quality Control Report

## Multi-Round Deduplication Pipeline

### Round 1: Text Hash Deduplication
- **Method**: SHA256 hash of normalized annotation text (lowercased, whitespace-normalized)
- **Result**: Removed exact duplicate annotations across sources
- **Remaining**: ~28,500 samples

### Round 2: Topology Hash Deduplication
- **Method**: Component-connection graph canonicalization (Weisfeiler-Lehman graph hashing)
- **Process**: Extract component nodes and net edges → canonical graph → hash
- **Result**: Removed circuits with identical topology but different rendering/formatting
- **Remaining**: ~26,000 samples

### Round 3: Visual Perceptual Hash
- **Method**: pHash (perceptual hash) of rendered images
- **Threshold**: Hamming distance < 5 considered visual duplicates
- **Result**: Removed near-identical images (e.g., same schematic at different zoom levels)
- **Remaining**: ~25,200 samples

## LLM-Based Quality Scoring
Each sample was scored by a large language model on four dimensions:

| Dimension | Description | Scale |
|-----------|-------------|-------|
| Completeness | Are all visible components annotated? | 1-5 |
| Accuracy | Do values match schematic markings? | 1-5 |
| Consistency | Does naming follow standard conventions? | 1-5 |
| Format Correctness | Does output match expected netlist format? | 1-5 |

**Quality Filter**: Samples scoring < 12/20 were flagged for manual review or removal.
- Samples scoring 16-20: Directly accepted (~18,000)
- Samples scoring 12-15: Manual spot-check (~6,500)
- Samples scoring < 12: Removed or re-annotated (~700)

## Test Set Quality Verification
- All 523 test samples manually verified for annotation correctness
- No synthetic-only bias in easy50 tier (balanced across all three sources)
- Degraded test set (250 samples) verified: transforms preserve annotation correctness
- Inter-annotator agreement on test set: 97.3%

## Final Dataset Statistics

| Stage | Count | Reduction |
|-------|-------|-----------|
| Raw collected | ~30,000 | — |
| After text dedup | ~28,500 | -5.0% |
| After topology dedup | ~26,000 | -8.8% |
| After visual dedup | ~25,200 | -3.1% |
| After LLM quality filter | 24,717 | -1.9% |
| **Final training set** | **24,717** | **-17.6%** |

## Quality Metrics Summary

| Metric | Value |
|--------|-------|
| Annotation accuracy (test set) | 97.3% |
| Inter-annotator agreement | 95.1% |
| Duplicate removal rate | 17.6% |
| LLM quality filter pass rate | 98.1% |

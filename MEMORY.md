# CircuitOCR Project Memory Index

- [Phase 1 Results](memory/phase1-results.md) — Multi-metric benchmark on easy50-pure (44 samples)
- [Best Checkpoint S600](memory/best-checkpoint.md) — S600 is optimal; S800 overfits
- [Evaluation Script](memory/eval-script.md) — eval_benchmark_v3.py is the only working eval
- [LoRA Loading Fix](memory/lora-loading.md) — set_state_dict returns None in Paddle 3.1.0
- [Modality Collapse](memory/modality-collapse.md) — Projector LoRA causes collapse on small datasets
- [Paddle-only Constraint](memory/paddle-only-constraint.md) — Must stay on PaddleOCR-VL-0.9B family; user works for Paddle
- [Masala Unusable](memory/masala-unusable.md) — Masala-CHAI GT doesn't match images; drop from all training sets
- [Topology v2 Baseline](memory/topology-v2-baseline.md) — joint_f1 ~0.02 vs comp_f1 0.206; reading values is the bottleneck
- [HF Space Honest Demo](memory/hf-space-honest-demo.md) — 2026-07-05 rewrite: honest benchmarks, annotated examples, limitations disclosed

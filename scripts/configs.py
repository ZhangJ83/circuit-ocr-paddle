"""Experiment configurations for circuit OCR training."""
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Base config shared by all experiments
BASE_CONFIG = {
    "model_path": "/root/models/official_models/PaddleOCR-VL",
    "train_data": os.path.join(PROJECT_DIR, "output", "train_clean.jsonl"),
    "val_data": os.path.join(PROJECT_DIR, "output", "val_clean.jsonl"),
    "test_data": os.path.join(PROJECT_DIR, "output", "test_clean.jsonl"),
    "output_dir": os.path.join(PROJECT_DIR, "checkpoints"),
    "results_dir": os.path.join(PROJECT_DIR, "results"),

    # Training hyperparams
    "epochs": 3,
    "batch_size": 1,
    "grad_accum": 4,
    "base_lr": 2e-5,
    "warmup_steps": 100,
    "grad_clip": 1.0,
    "checkpoint_steps": 400,
    "val_samples": 30,

    # LoRA default
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "max_dim": 384,
    "freeze_projector": True,

    # Generation
    "max_new_tokens": 512,
    "repetition_penalty": 1.1,
}

def get_config(name: str, overrides: dict = None) -> dict:
    """Get experiment config. Deep-copies base and applies overrides."""
    import copy
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["name"] = name

    presets = {
        "baseline": {
            "desc": "LLM-Only LoRA r=16, 384px — document recommended baseline",
        },
        "hires": {
            "desc": "Higher resolution 512px — better small-text recognition",
            "max_dim": 512,
        },
        "large_rank": {
            "desc": "Larger LoRA rank r=32 — more capacity",
            "lora_r": 32, "lora_alpha": 64,
        },
        "hires_large": {
            "desc": "512px + r=32 — max single-GPU capacity",
            "max_dim": 512, "lora_r": 32, "lora_alpha": 64,
        },
        "projector_ablation": {
            "desc": "Projector+LLM LoRA — verify modality collapse",
            "freeze_projector": False,
        },
        "deep": {
            "desc": "Extended training 10 epochs",
            "epochs": 10,
        },
        "low_lr": {
            "desc": "Lower learning rate 1e-5 — cautious training",
            "base_lr": 1e-5, "warmup_steps": 200,
        },
    }

    if name in presets:
        for k, v in presets[name].items():
            if k != "desc":
                cfg[k] = v

    if overrides:
        cfg.update(overrides)

    return cfg

# Auto-pipeline experiment sequence
PIPELINE = [
    # Phase 1: Quick baselines
    {"name": "baseline", "epochs": 3, "deeper": "deep_baseline"},
    {"name": "hires", "epochs": 3, "deeper": "deep_hires"},
    {"name": "large_rank", "epochs": 3, "deeper": "deep_largerank"},
    {"name": "hires_large", "epochs": 3, "deeper": "deep_hireslarge"},
    {"name": "projector_ablation", "epochs": 3},  # verify collapse, no deep run

    # Phase 2: Deep training of best config (auto-selected)
    # These are placeholders — pipeline fills "deeper" based on best result
    {"name": "deep_baseline", "epochs": 10, "resume_from": "baseline"},
    {"name": "deep_hires", "epochs": 10, "resume_from": "hires"},
    {"name": "deep_largerank", "epochs": 10, "resume_from": "large_rank"},
    {"name": "deep_hireslarge", "epochs": 10, "resume_from": "hires_large"},

    # Phase 3: Final push (if time remains)
    {"name": "final", "epochs": 20, "resume_from": None},  # best deep config, continued
]

"""
三模型实时对比推理工具
======================
输入一张电路原理图，同时运行基座 / exp6 (v1) / Phase 1 (v2) 三个模型，
并排输出原始识别结果，适合答辩现场演示。

用法:
    python demo_compare.py <image_path>            # 单张图片
    python demo_compare.py                         # 使用内置测试图片
    python demo_compare.py --list                  # 列出内置测试图片
    python demo_compare.py --idx 0                 # 使用内置第 0 张测试图片
    python demo_compare.py --max-tokens 120         # 自定义最大 token 数

依赖: paddlepaddle, paddleformers (PaddleOCR-VL), PIL, Levenshtein
"""

import os, sys, json, re, time, argparse
from types import ModuleType
from io import BytesIO
from pathlib import Path

# ── Early patch for Paddle 3.1.0 ──
_d = ModuleType('dummy_flex_checkpoint')
_d.build_sharded_state_dict = lambda *a, **kw: None
sys.modules.setdefault('paddle.distributed.flex_checkpoint', _d)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', _d)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', _d)

# ── Paddle patches (from eval_benchmark) ──
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                       'circuit-ocr-dataset', 'scripts'))
    from eval_benchmark import apply_paddle_patches
    apply_paddle_patches()
except Exception:
    pass

import paddle
from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

# ═══════════════════════════════
# Paths
# ═══════════════════════════════
PROJECT_ROOT = Path(__file__).parent
BASE_MODEL_DIR = (r'F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL'
                  r'\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27')

CKPT_EXP6 = (PROJECT_ROOT / 'PaddleOCR-VL-LoRA-circuit-ocr'
             / 'lora_exp6_best.pdparams')       # v1: 1500 samples + synth text
CKPT_PHASE1 = (PROJECT_ROOT / 'checkpoints' / 'synth_pure_5k'
               / 'best.pdparams')                # v2: 5000 synth KiCad pre-training

# Built-in test images (from the 5 case study samples in the report)
BUILTIN_IMAGES = [
    PROJECT_ROOT / 'output' / 'review_1000' / 'images' / '1446.png',
    PROJECT_ROOT / 'output' / 'review_1000' / 'images' / '0385.png',
    PROJECT_ROOT / 'output' / 'review_1000' / 'images' / '0478.png',
    PROJECT_ROOT / 'output' / 'review_1000' / 'images' / '0640.png',
    PROJECT_ROOT / 'output' / 'review_1000' / 'images' / '1449.png',
]

# ═══════════════════════════════
# Model loading
# ═══════════════════════════════
MODEL = None
PROCESSOR = None
LORA_MODEL = None


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_models(lora_r: int = 16, lora_alpha: int = 32):
    """Load base model once, wrap with LoRA.  Call ONCE at startup."""
    global MODEL, PROCESSOR, LORA_MODEL

    if MODEL is not None:
        return

    log(f"Loading base model from {BASE_MODEL_DIR} ...")
    paddle.set_device('gpu')

    MODEL = AutoModelForConditionalGeneration.from_pretrained(
        BASE_MODEL_DIR,
        convert_from_hf=True,
        load_checkpoint_format='naive',
        low_cpu_mem_usage=True,
        dtype='bfloat16',
    )
    MODEL.config._attn_implementation = 'flashmask'
    MODEL.visual.config._attn_implementation = 'flashmask'

    # Apply LoRA wrapper
    lc = LoRAConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            '.*q_proj', '.*k_proj', '.*v_proj', '.*o_proj',
            '.*linear_1', '.*linear_2',
        ],
        lora_dropout=0.05,
    )
    LORA_MODEL = LoRAModel(MODEL, lc)
    if not hasattr(LORA_MODEL.model, 'full'):
        LORA_MODEL.model.full = lambda *a, **kw: iter(
            LORA_MODEL.model.named_parameters())

    PROCESSOR = AutoProcessor.from_pretrained(BASE_MODEL_DIR)
    LORA_MODEL.eval()

    # Count params
    trainable = sum(
        p.numel().item() for p in LORA_MODEL.parameters() if not p.stop_gradient)
    total = sum(p.numel().item() for p in LORA_MODEL.parameters())
    log(f"  Base model loaded.  LoRA trainable: {trainable:,} / {total:,}"
        f"  ({100*trainable/total:.2f}%)")


def load_ckpt(ckpt_path: str):
    """Load LoRA checkpoint weights into the already-loaded model."""
    state = paddle.load(str(ckpt_path))
    loaded = 0
    for k, p in LORA_MODEL.named_parameters():
        if k in state:
            v = state[k]
            if p.dtype != v.dtype:
                v = paddle.cast(v, p.dtype)
            if list(p.shape) == list(v.shape):
                p.set_value(v)
                loaded += 1
    log(f"  Loaded {loaded} params from {Path(ckpt_path).name}")


def clear_lora():
    """Zero out all LoRA parameters → effectively run as base model."""
    for p in LORA_MODEL.parameters():
        if not p.stop_gradient:
            p.set_value(paddle.zeros_like(p))


# ═══════════════════════════════
# Inference
# ═══════════════════════════════
def run_inference(
    image_path: str,
    prompt: str = '<image>OCR:',
    max_tokens: int = 100,
    dim: int = 384,
    repetition_penalty: float = 1.1,
) -> str:
    """Run single inference on one image.  Returns decoded text."""
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    if max(w, h) > dim:
        scale = dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Re-encode to avoid PIL compatibility issues
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=95)
    buf.seek(0)
    img = Image.open(buf)

    msgs = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': img},
            {'type': 'text', 'text': prompt.replace('<image>', '')},
        ]
    }]

    inp = PROCESSOR.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pd')

    input_ids = inp['input_ids']
    attn = inp['attention_mask']
    pv = inp.get('pixel_values')
    igt = inp.get('image_grid_thw')

    gen = []
    with paddle.no_grad():
        for _ in range(max_tokens):
            out = LORA_MODEL(
                input_ids=input_ids,
                attention_mask=attn,
                pixel_values=pv,
                image_grid_thw=igt,
            )
            logits_ = out[0] if isinstance(out, (list, tuple)) else out.logits
            ntl = logits_[:, -1, :]

            # Repetition penalty
            for tid in set(gen):
                sc = float(ntl[0, tid])
                ntl[0, tid] = sc * repetition_penalty if sc < 0 else sc / repetition_penalty

            nt = int(paddle.argmax(ntl, axis=-1).numpy()[0])
            if nt == PROCESSOR.tokenizer.eos_token_id:
                break
            gen.append(nt)
            input_ids = paddle.concat(
                [input_ids, paddle.to_tensor([[nt]])], axis=1)
            attn = paddle.concat(
                [attn, paddle.ones([1, 1], dtype=attn.dtype)], axis=1)

    img.close()
    return PROCESSOR.tokenizer.decode(gen, skip_special_tokens=True)


# ═══════════════════════════════
# Main entry
# ═══════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description='三模型实时对比推理 — 基座 vs exp6(v1) vs Phase1(v2)')
    parser.add_argument('image', nargs='?',
                        help='电路图图片路径（不指定则用内置测试图）')
    parser.add_argument('--list', action='store_true',
                        help='列出内置测试图片并退出')
    parser.add_argument('--idx', type=int, default=0,
                        help='内置测试图片索引 (0-4)')
    parser.add_argument('--max-tokens', type=int, default=100,
                        help='最大生成 token 数 (默认 100)')
    parser.add_argument('--dim', type=int, default=384,
                        help='图片最大边长 (默认 384)')
    parser.add_argument('--prompt', type=str, default='<image>OCR:',
                        help='推理 prompt')
    args = parser.parse_args()

    # List mode
    if args.list:
        print("内置测试图片 (来自报告5个案例):")
        for i, p in enumerate(BUILTIN_IMAGES):
            exists = "✓" if p.exists() else "✗"
            print(f"  [{i}] {exists} {p}")
        return

    # Determine image path
    if args.image:
        img_path = Path(args.image)
    else:
        img_path = BUILTIN_IMAGES[args.idx]

    if not img_path.exists():
        print(f"错误: 图片不存在: {img_path}")
        print("使用 --list 查看内置测试图片")
        sys.exit(1)

    # ── Load model ──
    load_models()

    # ── Run 3 models ──
    print()
    print("=" * 72)
    print(f"  图片: {img_path.name}")
    print(f"  路径: {img_path}")
    print(f"  max_tokens={args.max_tokens}, dim={args.dim}")
    print("=" * 72)

    # 1) Base model (zero LoRA)
    log("Running Base model (PaddleOCR-VL-0.9B, no LoRA) ...")
    clear_lora()
    t0 = time.time()
    out_base = run_inference(str(img_path), prompt=args.prompt,
                             max_tokens=args.max_tokens, dim=args.dim)
    t_base = time.time() - t0

    # 2) exp6 (v1)
    log("Running exp6 (v1, 1500 real + synth text) ...")
    load_ckpt(CKPT_EXP6)
    t0 = time.time()
    out_exp6 = run_inference(str(img_path), prompt=args.prompt,
                             max_tokens=args.max_tokens, dim=args.dim)
    t_exp6 = time.time() - t0

    # 3) Phase 1 (v2)
    log("Running Phase 1 (v2, 5000 synth KiCad) ...")
    load_ckpt(CKPT_PHASE1)
    t0 = time.time()
    out_phase1 = run_inference(str(img_path), prompt=args.prompt,
                               max_tokens=args.max_tokens, dim=args.dim)
    t_phase1 = time.time() - t0

    # ── Print results ──
    def _fmt(text: str, width: int = 24) -> str:
        """Format multi-line output for side-by-side display."""
        lines = text.strip().split('\n')
        return '\n'.join(
            f"  │ {line:<{width}s}" for line in lines[:20]
        ) + ('\n  │ ...' if len(lines) > 20 else '')

    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + "  三模型输出对比".center(62) + "║")
    print("╠" + "═" * 22 + "╦" + "═" * 22 + "╦" + "═" * 22 + "╣")
    print(f"║ {'Base (PaddleOCR-VL)':<20s} ║ {'exp6 (v1)':<20s} ║ {'Phase 1 (v2) ★':<20s} ║")
    print("╠" + "═" * 22 + "╬" + "═" * 22 + "╬" + "═" * 22 + "╣")

    base_lines = out_base.strip().split('\n')
    exp6_lines = out_exp6.strip().split('\n')
    ph1_lines = out_phase1.strip().split('\n')
    max_lines = min(max(len(base_lines), len(exp6_lines), len(ph1_lines)), 20)

    for i in range(max_lines):
        b = base_lines[i][:20] if i < len(base_lines) else ''
        e = exp6_lines[i][:20] if i < len(exp6_lines) else ''
        p = ph1_lines[i][:20] if i < len(ph1_lines) else ''
        print(f"║ {b:<20s} ║ {e:<20s} ║ {p:<20s} ║")

    print("╚" + "═" * 22 + "╩" + "═" * 22 + "╩" + "═" * 22 + "╝")
    print()

    # Timing
    print(f"  耗时: Base={t_base:.1f}s  exp6={t_exp6:.1f}s  Phase1={t_phase1:.1f}s")
    print(f"  总计: {t_base+t_exp6+t_phase1:.1f}s")
    print()

    # Full outputs
    def _print_full(label: str, text: str):
        print(f"── {label} ──")
        for line in text.strip().split('\n')[:30]:
            print(f"  {line}")
        if len(text.strip().split('\n')) > 30:
            print(f"  ... (共 {len(text.strip().split(chr(10)))} 行)")
        print()

    _print_full("Base (PaddleOCR-VL-0.9B, 无 LoRA)", out_base)
    _print_full("exp6 / v1 (1500 real + 300 synth text, CompF1≈0.119)", out_exp6)
    _print_full("Phase 1 / v2 ★ (5000 synth KiCad pre-train, CompF1≈0.304)", out_phase1)


if __name__ == '__main__':
    main()

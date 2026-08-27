"""
本地网页 Demo：拖拽电路图 → 三模型实时对比
============================================
启动:  python demo_server.py
访问:  http://localhost:8899
"""

import os, sys, json, time, io, base64
from types import ModuleType
from io import BytesIO
from pathlib import Path

# ── Early patches ──
_d = ModuleType('dummy_flex_checkpoint')
_d.build_sharded_state_dict = lambda *a, **kw: None
sys.modules.setdefault('paddle.distributed.flex_checkpoint', _d)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', _d)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', _d)

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
from flask import Flask, request, jsonify, send_from_directory

# ═══════════════════════════════
# Config
# ═══════════════════════════════
PROJECT_ROOT = Path(__file__).parent
BASE_MODEL_DIR = (r'F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL'
                  r'\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27')
CKPT_EXP6 = str(PROJECT_ROOT / 'PaddleOCR-VL-LoRA-circuit-ocr'
                / 'lora_exp6_best.pdparams')
CKPT_PHASE1 = str(PROJECT_ROOT / 'checkpoints' / 'synth_pure_5k'
                  / 'best.pdparams')

LORA_R = 16
LORA_ALPHA = 32
LORA_TARGETS = ['.*q_proj', '.*k_proj', '.*v_proj', '.*o_proj',
                '.*linear_1', '.*linear_2']
MAX_DIM = 384
MAX_TOKENS = 120

# ═══════════════════════════════
# Global model state
# ═══════════════════════════════
lora_model = None
processor = None

app = Flask(__name__)


def load_models():
    global lora_model, processor
    if lora_model is not None:
        return

    print(f"[init] Loading base model...")
    paddle.set_device('gpu')

    base = AutoModelForConditionalGeneration.from_pretrained(
        BASE_MODEL_DIR, convert_from_hf=True, load_checkpoint_format='naive',
        low_cpu_mem_usage=True, dtype='bfloat16')
    base.config._attn_implementation = 'flashmask'
    base.visual.config._attn_implementation = 'flashmask'

    lc = LoRAConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                    target_modules=LORA_TARGETS, lora_dropout=0.05)
    lora_model = LoRAModel(base, lc)
    if not hasattr(lora_model.model, 'full'):
        lora_model.model.full = lambda *a, **kw: iter(
            lora_model.model.named_parameters())

    processor = AutoProcessor.from_pretrained(BASE_MODEL_DIR)
    lora_model.eval()

    trainable = sum(p.numel().item() for p in lora_model.parameters()
                    if not p.stop_gradient)
    total = sum(p.numel().item() for p in lora_model.parameters())
    print(f"[init] Model ready. LoRA: {trainable:,}/{total:,}"
          f" ({100*trainable/total:.2f}%)")


def _load_weights(ckpt_path: str):
    state = paddle.load(ckpt_path)
    n = 0
    for k, p in lora_model.named_parameters():
        if k in state:
            v = state[k]
            if p.dtype != v.dtype:
                v = paddle.cast(v, p.dtype)
            if list(p.shape) == list(v.shape):
                p.set_value(v)
                n += 1
    return n


def _clear_lora():
    for p in lora_model.parameters():
        if not p.stop_gradient:
            p.set_value(paddle.zeros_like(p))


def run_one_inference(image_bytes: bytes, max_tokens: int = MAX_TOKENS,
                      dim: int = MAX_DIM) -> str:
    """Run inference on image bytes, return decoded text."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    w, h = img.size
    if max(w, h) > dim:
        scale = dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format='JPEG', quality=95)
    buf.seek(0)
    img = Image.open(buf)

    msgs = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img},
        {'type': 'text', 'text': 'OCR:'}]}]

    inp = processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pd')

    input_ids = inp['input_ids']
    attn = inp['attention_mask']
    pv = inp.get('pixel_values')
    igt = inp.get('image_grid_thw')

    gen = []
    with paddle.no_grad():
        for _ in range(max_tokens):
            out = lora_model(input_ids=input_ids, attention_mask=attn,
                             pixel_values=pv, image_grid_thw=igt)
            logits_ = out[0] if isinstance(out, (list, tuple)) else out.logits
            ntl = logits_[:, -1, :]
            for tid in set(gen):
                sc = float(ntl[0, tid])
                ntl[0, tid] = sc * 1.1 if sc < 0 else sc / 1.1
            nt = int(paddle.argmax(ntl, axis=-1).numpy()[0])
            if nt == processor.tokenizer.eos_token_id:
                break
            gen.append(nt)
            input_ids = paddle.concat(
                [input_ids, paddle.to_tensor([[nt]])], axis=1)
            attn = paddle.concat(
                [attn, paddle.ones([1, 1], dtype=attn.dtype)], axis=1)

    img.close()
    return processor.tokenizer.decode(gen, skip_special_tokens=True)


# ═══════════════════════════════
# Routes
# ═══════════════════════════════
@app.route('/')
def index():
    return send_from_directory(str(PROJECT_ROOT), 'demo_compare.html')


@app.route('/api/infer', methods=['POST'])
def api_infer():
    if 'image' not in request.files:
        return jsonify({'error': 'no image uploaded'}), 400

    file = request.files['image']
    image_bytes = file.read()

    results = {}
    timings = {}

    # 1) Base
    t0 = time.time()
    _clear_lora()
    results['base'] = run_one_inference(image_bytes)
    timings['base'] = round(time.time() - t0, 1)

    # 2) exp6 (v1)
    t0 = time.time()
    _load_weights(CKPT_EXP6)
    results['exp6'] = run_one_inference(image_bytes)
    timings['exp6'] = round(time.time() - t0, 1)

    # 3) Phase 1 (v2)
    t0 = time.time()
    _load_weights(CKPT_PHASE1)
    results['phase1'] = run_one_inference(image_bytes)
    timings['phase1'] = round(time.time() - t0, 1)

    # Encode image as base64 for display
    img_b64 = base64.b64encode(image_bytes).decode('utf-8')
    mime = file.content_type or 'image/png'

    return jsonify({
        'results': results,
        'timings': timings,
        'image_b64': img_b64,
        'image_mime': mime,
    })


if __name__ == '__main__':
    print("=" * 50)
    print("  CircuitOCR — 三模型实时对比 Demo")
    print("  启动中，请稍候...")
    print("=" * 50)
    load_models()
    print()
    print(f"  打开浏览器访问: http://localhost:8899")
    print(f"  拖拽电路图 PNG 到页面即可")
    print()
    app.run(host='0.0.0.0', port=8899, debug=False)

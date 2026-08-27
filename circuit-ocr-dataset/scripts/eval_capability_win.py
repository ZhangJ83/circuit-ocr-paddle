"""Task 5: General Capability Evaluation — Catastrophic Forgetting Check.
Compares PaddleOCR-VL base model vs LoRA fine-tuned model on general VQA tasks.
Uses the same working inference pattern as test_gen_win.py.
"""
import os, sys, json, time, re
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HOME"] = "F:/hf_cache/hub"
os.environ["PADDLE_HOME"] = "F:/paddle_cache"
os.environ["HF_HUB_CACHE"] = "F:/hf_cache/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["PATH"] = r"E:\080000software\080900_Miniconda\miniconda3\Library\bin;" + os.environ.get("PATH", "")

import paddle
import paddle.distributed.fleet.meta_parallel as mp
if not hasattr(mp, 'LocalSharedLayerDesc'):
    class _LocalSharedLayerDesc:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
    mp.LocalSharedLayerDesc = _LocalSharedLayerDesc

from types import ModuleType
try:
    import paddle.distributed.flex_checkpoint.dcp.sharded_weight
except Exception:
    dummy = ModuleType('dummy')
    for f in ['build_sharded_state_dict','create_sharded_weight_with_new_local',
              'reshape_sharded_weight','sharded_weight_parallel_cpu',
              'save_state_dict','load_state_dict']:
        setattr(dummy, f, lambda *a, **kw: None)
    for m in ['paddle.distributed.flex_checkpoint','paddle.distributed.flex_checkpoint.dcp',
              'paddle.distributed.flex_checkpoint.dcp.sharded_weight']:
        sys.modules.setdefault(m, dummy)

paddle.float8_e4m3fn = paddle.float32; paddle.float8_e5m2 = paddle.float32
paddle.LongTensor = paddle.Tensor
paddle.linalg.fp8_fp8_half_gemm_fused = None
paddle.Tensor.long = lambda s: s.astype("int64")
paddle.Tensor.float = lambda s: s.astype("float32")
paddle.Tensor.half = lambda s: s.astype("float16")

_old_reshape = paddle.Tensor.reshape
def _pr(self, *args, **kwargs):
    if args:
        if isinstance(args[0], paddle.dtype): return self.astype(args[0])
        if len(args) > 1: new_shape = list(args)
        elif len(args) == 1 and (isinstance(args[0], int) or hasattr(args[0], '__index__')):
            new_shape = [int(args[0])]
        else: new_shape = args[0]
        return _old_reshape(self, new_shape, **kwargs)
    return _old_reshape(self, **kwargs)
paddle.Tensor.reshape = _pr; paddle.Tensor.view = _pr
if not hasattr(paddle.Tensor, "repeat"): paddle.Tensor.repeat = paddle.Tensor.tile

_old_transpose = paddle.Tensor.transpose
def _pt(self, *args, **kwargs):
    if len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):
        dim0, dim1 = args[0], args[1]; ndim = self.ndim
        if dim0 < 0: dim0 += ndim
        if dim1 < 0: dim1 += ndim
        perm = list(range(ndim)); perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return _old_transpose(self, perm, **kwargs)
    return _old_transpose(self, *args, **kwargs)
paddle.Tensor.transpose = _pt

def _pms(self, mask, source):
    orig = self.shape; mask = mask.astype('bool')
    fs, fm, fsrc = self.flatten(), mask.flatten(), source.flatten()
    idx = paddle.nonzero(fm)
    scat = paddle.scatter_nd(idx, fsrc, fm.shape)
    return paddle.where(fm, scat, fs).reshape(orig)
paddle.Tensor.masked_scatter = _pms

_old_gf = paddle.base.framework.get_flags
paddle.base.framework.get_flags = lambda flags: {f: 2 if f == "FLAGS_flash_attn_version" else _old_gf([f]).get(f) for f in flags}
_old_sf = paddle.set_flags
paddle.set_flags = lambda d: _old_sf({k: v for k, v in d.items() if k != "FLAGS_flash_attn_version"}) if {k: v for k, v in d.items() if k != "FLAGS_flash_attn_version"} else None

_old_gelu = paddle.nn.functional.gelu
paddle.nn.functional.gelu = lambda x, approximate=False, name=None: _old_gelu(x, approximate == 'tanh' if isinstance(approximate, str) else approximate, name)

for nm in ['empty','zeros','ones','arange','full','randn','rand']:
    if hasattr(paddle, nm):
        of = getattr(paddle, nm)
        setattr(paddle, nm, lambda *a, _of=of, **kw: _of(*a, **{k: v for k, v in kw.items() if k != 'device'}))

paddle.nn.functional.swiglu = lambda *a, **kw: None

def _frms(x, w, eps=1e-6):
    v = paddle.mean(paddle.square(x), axis=-1, keepdim=True)
    r = paddle.rsqrt(v + eps); return (x * r * w, r)
paddle.incubate.nn.functional.fused_rms_norm_ext = _frms

def _fma(q, k, v, startend_row_indices=None, causal=True):
    qt, kt, vt = q.transpose([0,2,1,3]), k.transpose([0,2,1,3]), v.transpose([0,2,1,3])
    b, hq, lq, d = qt.shape; _, hk, lk, _ = kt.shape
    if hq != hk:
        nr = hq // hk
        kt = paddle.tile(kt.reshape([b,hk,1,lk,d]), [1,1,nr,1,1]).reshape([b,hq,lk,d])
        vt = paddle.tile(vt.reshape([b,hk,1,lk,d]), [1,1,nr,1,1]).reshape([b,hq,lk,d])
    uc = causal and lq == lk; am = None
    if causal and not uc:
        ri = paddle.arange(lq, dtype='int32').reshape([1,1,lq,1])
        ci = paddle.arange(lk, dtype='int32').reshape([1,1,1,lk])
        cb = ci <= (lk - lq + ri)
        am = paddle.where(cb, paddle.zeros([1,1,lq,lk], dtype=q.dtype), paddle.full([1,1,lq,lk], -1e9, dtype=q.dtype))
        if b > 1: am = paddle.tile(am, [b,1,1,1])
    try:
        return paddle.nn.functional.scaled_dot_product_attention(qt,kt,vt,attn_mask=am,is_causal=uc,training=False).transpose([0,2,1,3])
    except:
        scores = paddle.matmul(qt, kt.transpose([0,1,3,2])) / (d ** 0.5)
        if am is not None: scores = scores + am
        if uc:
            gq = paddle.arange(lq, dtype="int32").reshape([lq,1])
            gk = paddle.arange(lk, dtype="int32").reshape([1,lk])
            scores = paddle.where((gk-gq) <= (lk-lq), scores, paddle.to_tensor(-1e9, dtype=scores.dtype))
        return paddle.matmul(paddle.nn.functional.softmax(scores, axis=-1), vt).transpose([0,2,1,3])
paddle.nn.functional.flash_attention.flashmask_attention = _fma
paddle.incubate.tensor.manipulation.create_async_load = lambda *a, **kw: None
print("[Patches] OK", flush=True)

paddle.set_device("gpu")
print(f"GPU: {paddle.device.cuda.get_device_name(0)}", flush=True)

from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.generation import GenerationConfig
from paddleformers.peft import LoRAConfig, LoRAModel
from PIL import Image
from pathlib import Path
import numpy as np

LOCAL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
DATA_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
OUTPUT_FILE = f"{DATA_DIR}/results_capability_eval.jsonl"
LORA_DIR = f"{DATA_DIR}/PaddleOCR-VL-LoRA-circuit-ocr"

# === Test Cases for General Capability ===
# Using circuit schematic images (what we have), ask non-OCR questions
TEST_CASES = [
    # Basic visual perception
    {"type": "color", "prompt": "What colors are visible in this circuit diagram? List them."},
    {"type": "objects", "prompt": "What electronic components can you identify in this schematic? List the symbols you see."},
    {"type": "count", "prompt": "How many distinct components or nodes are visible in this circuit diagram? Give a number."},
    # Spatial reasoning
    {"type": "spatial", "prompt": "Describe the spatial layout of this schematic. Are components arranged horizontally, vertically, or both?"},
    # Category classification
    {"type": "category", "prompt": "What type of circuit diagram is this? (e.g., amplifier, power supply, logic gate, etc.) Briefly classify it."},
    # OCR baseline (our task)
    {"type": "ocr_baseline", "prompt": "OCR:"},
]

def load_model(use_lora=False):
    """Load base model with optional LoRA weights."""
    processor = AutoProcessor.from_pretrained(LOCAL_PATH)
    model = AutoModelForConditionalGeneration.from_pretrained(
        LOCAL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
        low_cpu_mem_usage=True, dtype="float32"
    )
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"

    if use_lora:
        lc = LoRAConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=['.*q_proj', '.*k_proj', '.*v_proj', '.*o_proj']
        )
        model = LoRAModel(model, lc)
        for lora_file in [
            f"{LORA_DIR}/lora_weights_f32.pdparams",
            f"{LORA_DIR}/final_model_light.pdparams"
        ]:
            if Path(lora_file).exists():
                print(f"  Loading LoRA: {lora_file}", flush=True)
                state = paddle.load(lora_file)
                model.set_state_dict(state)
                break
        else:
            print(f"  WARNING: No LoRA weights found in {LORA_DIR}")

    model.eval()
    return model, processor

def generate(model, processor, image, prompt, max_new_tokens=128):
    """Generate a response for an image + text prompt."""
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]}]
    inputs = processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pd"
    )
    gen_config = GenerationConfig(
        do_sample=False, bos_token_id=1, eos_token_id=2, pad_token_id=0
    )
    with paddle.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config, max_new_tokens=max_new_tokens)
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(outputs[0][0][input_len:], skip_special_tokens=True)

def load_test_images(n=10):
    """Load n test images from the dataset, resize to 256px max."""
    samples = [json.loads(l) for l in open(f"{DATA_DIR}/ocr_vl_sft-test.jsonl") if l.strip()]
    images = []
    for i in range(min(n, len(samples))):
        sample = samples[i]
        img_path = sample["images"][0]
        if not img_path.startswith("/"):
            img_path = f"{DATA_DIR}/{img_path.lstrip('./')}"
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        max_dim = 256
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        images.append({
            "path": img_path,
            "image": image,
            "label": sample["messages"][1]["content"][:100],
        })
    return images

def compute_text_similarity(t1, t2):
    """Simple token overlap similarity between two texts."""
    if not t1 or not t2:
        return 0.0
    tokens1 = set(re.findall(r'\w+', t1.lower()))
    tokens2 = set(re.findall(r'\w+', t2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    return len(intersection) / max(len(tokens1), len(tokens2))

def main():
    print("=" * 60)
    print("Task 5: General Capability Evaluation — Catastrophic Forgetting Check")
    print("=" * 60)

    test_images = load_test_images(n=5)  # Use 5 images
    print(f"\nLoaded {len(test_images)} test images")

    # Phase 1: Base model
    print("\n" + "=" * 60)
    print("[Phase 1/2] Evaluating BASE model...")
    print("=" * 60)
    base_model, processor = load_model(use_lora=False)
    base_results = []

    for img_idx, img_data in enumerate(test_images):
        for tc_idx, tc in enumerate(TEST_CASES):
            t0 = time.time()
            try:
                pred = generate(base_model, processor, img_data["image"], tc["prompt"])
                elapsed = time.time() - t0
            except Exception as e:
                pred = f"ERROR: {e}"
                elapsed = time.time() - t0

            result = {
                "model": "base",
                "image_idx": img_idx,
                "image_path": img_data["path"],
                "test_type": tc["type"],
                "prompt": tc["prompt"],
                "label": img_data["label"],
                "prediction": pred,
                "time_s": round(elapsed, 2),
            }
            base_results.append(result)
            print(f"  [BASE] img{img_idx} {tc['type']:15s} ({elapsed:.1f}s): {pred[:80]}", flush=True)

    del base_model
    paddle.device.cuda.empty_cache()

    # Phase 2: LoRA model
    print("\n" + "=" * 60)
    print("[Phase 2/2] Evaluating LoRA model...")
    print("=" * 60)
    lora_model, _ = load_model(use_lora=True)
    lora_results = []

    for img_idx, img_data in enumerate(test_images):
        for tc_idx, tc in enumerate(TEST_CASES):
            t0 = time.time()
            try:
                pred = generate(lora_model, processor, img_data["image"], tc["prompt"])
                elapsed = time.time() - t0
            except Exception as e:
                pred = f"ERROR: {e}"
                elapsed = time.time() - t0

            result = {
                "model": "lora",
                "image_idx": img_idx,
                "image_path": img_data["path"],
                "test_type": tc["type"],
                "prompt": tc["prompt"],
                "label": img_data["label"],
                "prediction": pred,
                "time_s": round(elapsed, 2),
            }
            lora_results.append(result)
            print(f"  [LORA] img{img_idx} {tc['type']:15s} ({elapsed:.1f}s): {pred[:80]}", flush=True)

    del lora_model
    paddle.device.cuda.empty_cache()

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS: Base vs LoRA Comparison")
    print("=" * 60)

    # Group by test_type
    by_type = {}
    for r in base_results + lora_results:
        t = r["test_type"]
        if t not in by_type:
            by_type[t] = {"base": [], "lora": []}
        by_type[t][r["model"]].append(r)

    for test_type in TEST_CASES:
        tc = test_type["type"]
        entries = by_type.get(tc, {"base": [], "lora": []})
        base_preds = [r["prediction"] for r in entries["base"] if "ERROR" not in r["prediction"]]
        lora_preds = [r["prediction"] for r in entries["lora"] if "ERROR" not in r["prediction"]]

        # For OCR baseline, compute similarity to label
        if tc == "ocr_baseline":
            base_sims = [compute_text_similarity(r["prediction"], r["label"])
                        for r in entries["base"] if "ERROR" not in r["prediction"]]
            lora_sims = [compute_text_similarity(r["prediction"], r["label"])
                        for r in entries["lora"] if "ERROR" not in r["prediction"]]
            avg_base = np.mean(base_sims) if base_sims else 0
            avg_lora = np.mean(lora_sims) if lora_sims else 0
            print(f"  {tc:15s}: base_sim={avg_base:.3f}  lora_sim={avg_lora:.3f}  "
                  f"delta={'+' if avg_lora>=avg_base else ''}{avg_lora-avg_base:+.3f}")
        else:
            # For general capability, check if predictions are similar (no catastrophic change)
            sims = []
            for i in range(min(len(base_preds), len(lora_preds))):
                sims.append(compute_text_similarity(base_preds[i], lora_preds[i]))
            avg_sim = np.mean(sims) if sims else 0
            avg_base_len = np.mean([len(p) for p in base_preds]) if base_preds else 0
            avg_lora_len = np.mean([len(p) for p in lora_preds]) if lora_preds else 0
            print(f"  {tc:15s}: base_len={avg_base_len:.0f}  lora_len={avg_lora_len:.0f}  "
                  f"base-lora_sim={avg_sim:.3f}  {'OK' if avg_sim > 0.3 else 'CHECK'}")

    # Save all results
    all_results = base_results + lora_results
    with open(OUTPUT_FILE, 'w') as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"\nResults saved to {OUTPUT_FILE}")
    print("=" * 60)
    print("Capability evaluation complete!")

if __name__ == "__main__":
    main()

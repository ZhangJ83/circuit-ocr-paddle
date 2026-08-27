#!/usr/bin/env python3
"""General Capability Evaluation — compare base model vs LoRA model.
Tests for catastrophic forgetting on general VQA tasks.
"""
import os, sys, json, time, argparse, random
from pathlib import Path
from datetime import datetime

os.environ.update({
    "KMP_DUPLICATE_LIB_OK": "TRUE", "HF_HOME": "/mnt/f/hf_cache/hub",
    "PADDLE_HOME": "/mnt/f/paddle_cache", "HF_HUB_CACHE": "/mnt/f/hf_cache/hub",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "FLAGS_allocator_strategy": "auto_growth",
})
DATASET_DIR = "/mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset"

# ====== FULL monkey-patches from train_lora_light.py ======
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
def _patched_reshape(self, *args, **kwargs):
    if args:
        if isinstance(args[0], paddle.dtype): return self.astype(args[0])
        if len(args) > 1: new_shape = list(args)
        elif len(args) == 1 and (isinstance(args[0], int) or hasattr(args[0], '__index__')):
            new_shape = [int(args[0])]
        else: new_shape = args[0]
        return _old_reshape(self, new_shape, **kwargs)
    return _old_reshape(self, **kwargs)
paddle.Tensor.reshape = _patched_reshape; paddle.Tensor.view = _patched_reshape
if not hasattr(paddle.Tensor, "repeat"): paddle.Tensor.repeat = paddle.Tensor.tile

_old_transpose = paddle.Tensor.transpose
def _patched_transpose(self, *args, **kwargs):
    if len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):
        dim0, dim1 = args[0], args[1]; ndim = self.ndim
        if dim0 < 0: dim0 += ndim
        if dim1 < 0: dim1 += ndim
        perm = list(range(ndim)); perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
        return _old_transpose(self, perm, **kwargs)
    return _old_transpose(self, *args, **kwargs)
paddle.Tensor.transpose = _patched_transpose

def _patched_masked_scatter(self, mask, source):
    orig = self.shape; mask = mask.astype('bool')
    flat_self, flat_mask, flat_src = self.flatten(), mask.flatten(), source.flatten()
    idx = paddle.nonzero(flat_mask)
    scat = paddle.scatter_nd(idx, flat_src, flat_mask.shape)
    return paddle.where(flat_mask, scat, flat_self).reshape(orig)
paddle.Tensor.masked_scatter = _patched_masked_scatter

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
# ====== End patches ======

# General capability test prompts
TEST_CASES = [
    {"id": "color",     "prompt": "What colors are present in this image? Describe briefly."},
    {"id": "objects",   "prompt": "List the main objects or elements visible in this image."},
    {"id": "count",     "prompt": "How many distinct components or elements can you count in this image?"},
    {"id": "spatial",   "prompt": "Describe the spatial layout of this image. What is on the left, right, top, bottom?"},
    {"id": "category",  "prompt": "What category or type of diagram does this image belong to?"},
    {"id": "ocr_baseline", "prompt": "Read all text and labels visible in this image."},
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_model(model_path, is_lora=False, lora_weights=None):
    from paddleformers.transformers import AutoModelForConditionalGeneration
    from paddleformers.generation import GenerationConfig

    log(f"  Loading model...")
    t0 = time.time()

    model = AutoModelForConditionalGeneration.from_pretrained(
        model_path, convert_from_hf=True, load_checkpoint_format='naive',
        low_cpu_mem_usage=True, dtype="bfloat16"
    )
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"

    if is_lora:
        from paddleformers.peft import LoRAConfig, LoRAModel
        lc = LoRAConfig(
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=['.*q_proj', '.*k_proj', '.*v_proj', '.*o_proj']
        )
        model = LoRAModel(model, lc)
        if lora_weights and Path(lora_weights).exists():
            state = paddle.load(lora_weights)
            model.set_state_dict(state)
            log(f"  LoRA weights loaded: {lora_weights}")
        else:
            log(f"  WARNING: LoRA weights not found at {lora_weights}")

    model.eval()
    log(f"  Model loaded in {time.time()-t0:.1f}s")
    return model


def run_eval(model, processor, test_cases, num_images=3):
    random.seed(42)

    data_file = f"{DATASET_DIR}/ocr_vl_sft-test.jsonl"
    samples = [json.loads(line) for line in open(data_file) if line.strip()]
    selected = random.sample(samples, min(num_images, len(samples)))

    results = []
    for si, sample in enumerate(selected):
        img_path = sample["images"][0]
        if not img_path.startswith('/'):
            img_path = f"{DATASET_DIR}/{img_path.lstrip('./')}"
        if not Path(img_path).exists():
            log(f"  SKIP: image not found: {img_path}")
            continue

        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        log(f"  Image {si+1}/5: {Path(img_path).name}")

        for tc in test_cases:
            t0 = time.time()
            try:
                msgs = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": tc["prompt"]},
                ]}]
                inputs = processor.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pd"
                )
                with paddle.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=128, do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
                    )
                response = processor.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                elapsed = time.time() - t0
                results.append({
                    "image": sample["images"][0],
                    "test_id": tc["id"],
                    "prompt": tc["prompt"],
                    "response": response.strip(),
                    "time": round(elapsed, 2),
                })
                log(f"    [{tc['id']}] ({elapsed:.1f}s) {response.strip()[:100]}")
            except Exception as e:
                import traceback
                log(f"    [{tc['id']}] ERROR: {e}")
                traceback.print_exc()
                results.append({
                    "image": sample["images"][0],
                    "test_id": tc["id"],
                    "prompt": tc["prompt"],
                    "response": f"ERROR: {e}",
                    "time": 0,
                })
            paddle.device.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--output", default="capability_eval.json")
    args = parser.parse_args()

    paddle.set_device('gpu')
    log(f"GPU: {paddle.device.cuda.get_device_name(0)}")

    MODEL_PATH = "/mnt/f/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27"
    LORA_DIR = f"{DATASET_DIR}/PaddleOCR-VL-LoRA-circuit-ocr"

    print("=" * 60)
    log("General Capability Evaluation — Catastrophic Forgetting Check")
    print("=" * 60)

    from paddleformers.transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 1. Base model
    print("\n[1/2] Evaluating BASE model...")
    base_model = load_model(MODEL_PATH, is_lora=False)
    base_results = run_eval(base_model, processor, TEST_CASES, args.num_images)
    del base_model
    paddle.device.cuda.empty_cache()

    # 2. LoRA model
    print("\n[2/2] Evaluating LoRA model...")
    lora_weights = f"{LORA_DIR}/final_model_light.pdparams"
    lora_model = load_model(MODEL_PATH, is_lora=True, lora_weights=lora_weights)
    lora_results = run_eval(lora_model, processor, TEST_CASES, args.num_images)
    del lora_model
    paddle.device.cuda.empty_cache()

    # Save results
    output = {
        "base_model_path": MODEL_PATH,
        "lora_dir": LORA_DIR,
        "num_images": args.num_images,
        "base_results": base_results,
        "lora_results": lora_results,
    }
    out_path = f"{DATASET_DIR}/{args.output}"
    json.dump(output, open(out_path, 'w'), indent=2, ensure_ascii=False)
    log(f"Results saved to {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 60)
    for i in range(0, min(len(base_results), len(lora_results)), len(TEST_CASES)):
        img = base_results[i]["image"] if i < len(base_results) else "?"
        print(f"\n--- Image: {img} ---")
        for j in range(len(TEST_CASES)):
            bi = i + j
            if bi >= len(base_results) or bi >= len(lora_results):
                break
            br = base_results[bi]
            lr = lora_results[bi]
            print(f"  [{br['test_id']}]")
            print(f"    BASE: {br['response'][:150]}")
            print(f"    LoRA: {lr['response'][:150]}")


if __name__ == "__main__":
    main()

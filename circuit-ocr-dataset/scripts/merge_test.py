"""Clean merge test: load saved LoRA + merge into base model + inference.
Assumes lora_final_fp16.pdparams already exists from training.
"""
import os, sys, json
os.environ.update({
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "HF_HOME": "F:/hf_cache/hub", "PADDLE_HOME": "F:/paddle_cache",
    "HF_HUB_CACHE": "F:/hf_cache/hub", "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1", "FLAGS_allocator_strategy": "auto_growth",
})

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()

import paddle; paddle.set_device("gpu")
import numpy as np
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig

DATASET_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
LORA_PATH = f"{DATASET_DIR}/PaddleOCR-VL-LoRA-circuit-ocr/lora_final_fp16.pdparams"

def log(msg):
    ts = __import__('datetime').datetime.now().strftime("%H:%M:%S")
    try: print(f"[{ts}] {msg}", flush=True)
    except: print(f"[{ts}] {msg.encode('ascii','replace').decode('ascii')}", flush=True)

# ── Load saved LoRA ──
log("[1/4] Loading saved LoRA weights...")
lora_saved = paddle.load(LORA_PATH)
log(f"  Loaded {len(lora_saved)} keys")

# Show key patterns
lora_A_keys = [k for k in lora_saved if k.endswith('.lora_A')]
log(f"  LoRA A keys: {len(lora_A_keys)}")
log(f"  Example A: {lora_A_keys[0]}")
log(f"  Example B: {lora_A_keys[0][:-7] + '.lora_B'}")

# ── Load base model ──
log("[2/4] Loading base model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Build base param lookup
base_params = {k: p for k, p in model.named_parameters()}
log(f"  Base params: {len(base_params)}")
log(f"  Example base key: {list(base_params.keys())[10]}")

# ── Merge ──
log("[3/4] Merging LoRA into base model...")
LORA_SCALE = 2.0  # alpha/r = 16/8
merged = 0
no_base = 0
shape_mismatch = 0

for lora_A_key in lora_A_keys:
    # Derive names — strip 'model.' prefix (LoRA wrapper adds it, base model doesn't have it)
    clean_key = lora_A_key[6:] if lora_A_key.startswith('model.') else lora_A_key
    base_hint = clean_key[:-7]  # remove '.lora_A'
    lora_B_key = lora_A_key[:-7] + '.lora_B'
    base_weight_key = base_hint + '.weight'

    if lora_B_key not in lora_saved:
        continue
    if base_weight_key not in base_params:
        no_base += 1
        if no_base <= 2:
            log(f"  No base param: {base_weight_key[-60:]}")
        continue

    A = lora_saved[lora_A_key]  # paddle tensor [hidden, r]
    B = lora_saved[lora_B_key]  # paddle tensor [r, hidden*groups]
    p_base = base_params[base_weight_key]

    # Convert to numpy for math
    A_np = A.numpy().astype(np.float32)
    B_np = B.numpy().astype(np.float32)
    W_np = p_base.numpy().astype(np.float32)

    A_hidden = A_np.shape[0]  # e.g. 1152

    # Paddle Linear stores W as [in_features, out_features]
    # A = [in, r], B = [r, out] → A@B = [in, out] — matches W directly!
    delta = A_np @ B_np  # [in_features, out_features]

    if delta.shape != W_np.shape:
        # Case 1: transposed
        if delta.shape[0] == W_np.shape[1] and delta.shape[1] == W_np.shape[0]:
            delta = delta.T
        # Case 2: GQA — delta col > W col (B covers all heads, W only KV heads)
        elif delta.shape[0] == W_np.shape[0] and delta.shape[1] > W_np.shape[1]:
            delta = delta[:, :W_np.shape[1]]
        # Case 3: tile rows
        elif delta.shape[0] < W_np.shape[0] and W_np.shape[0] % delta.shape[0] == 0:
            delta = np.tile(delta, (W_np.shape[0] // delta.shape[0], 1))

        if delta.shape != W_np.shape:
            shape_mismatch += 1
            if shape_mismatch <= 3:
                log(f"  Shape mismatch: {base_hint[-40:]} delta={delta.shape} W={W_np.shape}")
            continue

    W_new = W_np + delta * LORA_SCALE

    try:
        p_base.set_value(paddle.to_tensor(W_new.astype("float16"), dtype=p_base.dtype, place=p_base.place))
        merged += 1
    except Exception as e:
        if merged < 3:
            log(f"  set_value error: {e}")

log(f"  Merged: {merged} / {len(lora_A_keys)} (no_base={no_base} shape_mismatch={shape_mismatch})")

# ── Inference ──
log("[4/4] Testing inference...")
model.eval()

from PIL import Image
test_path = f"{DATASET_DIR}/ocr_vl_sft-test-easy50.jsonl"
with open(test_path, encoding="utf-8") as f:
    test_data = [json.loads(l) for l in f if l.strip()][:3]

for i, s in enumerate(test_data):
    img = Image.open(f"{DATASET_DIR}/{s['images'][0].lstrip('./')}").convert("RGB")
    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":s["messages"][0]["content"].replace("<image>","")}]}]
    inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")

    with paddle.no_grad():
        gen_out = model.generate(**inp, max_new_tokens=256, do_sample=False)

    if isinstance(gen_out, (list, tuple)):
        tok = gen_out[0]
    else:
        tok = gen_out
    if len(tok.shape) > 1:
        tok = tok[0]

    ids_list = [int(x) for x in tok.numpy().tolist() if int(x) > 0]
    resp = processor.tokenizer.decode(ids_list, skip_special_tokens=True)
    ref = s["messages"][1]["content"][:150]
    log(f"  [{i}] Pred ({len(resp.strip())} chars): {resp.strip()[:200]}")
    log(f"  [{i}] Ref  ({len(ref)} chars): {ref[:200]}")
    img.close()

log("=== DONE ===")

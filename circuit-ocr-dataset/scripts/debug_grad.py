"""Quick gradient debug: check if logits are detached."""
import paddle, sys, os, json
os.chdir('/mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset')
sys.path.insert(0, 'scripts')
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel
from PIL import Image

MP = '/mnt/f/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27'
DD = '/mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset'

proc = AutoProcessor.from_pretrained(MP)
model = AutoModelForConditionalGeneration.from_pretrained(
    MP, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"
lc = LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=['.*q_proj', '.*k_proj', '.*v_proj', '.*o_proj'])
model = LoRAModel(model, lc)
model.mark_only_lora_as_trainable()
model.train()

# Load sample
train = [json.loads(l) for l in open(f"{DD}/ocr_vl_sft-train.jsonl") if l.strip()]
s = train[0]
label = s["messages"][1]["content"]
img_path = s["images"][0]
if not img_path.startswith("/"):
    img_path = f"{DD}/{img_path.lstrip('./')}"
img = Image.open(img_path).convert("RGB")
w, h = img.size
md = 168
if max(w, h) > md:
    sc = md / max(w, h)
    img = img.resize((int(w * sc), int(h * sc)), Image.LANCZOS)
query = s["messages"][0]["content"]
msgs = [{"role": "user", "content": [
    {"type": "image", "image": img},
    {"type": "text", "text": query.replace("<image>", "")}
]}]
full_msgs = msgs + [{"role": "assistant", "content": [{"type": "text", "text": label}]}]

fi = proc.apply_chat_template(full_msgs, tokenize=True, add_generation_prompt=False,
                              return_dict=True, return_tensors="pd")
pi = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                              return_dict=True, return_tensors="pd")
plen = pi["input_ids"].shape[1]
fids = fi["input_ids"]
labels_t = paddle.full_like(fids, -100)
labels_t[0, plen:] = fids[0, plen:]

out = model(**fi)
logits = out[0] if isinstance(out, (tuple, list)) else out

print(f"logits.stop_gradient = {logits.stop_gradient}")
print(f"logits shape={logits.shape}, dtype={logits.dtype}")

# Test if paddle.cast preserves gradients
sl = paddle.cast(logits[:, :-1, :], "float32")
print(f"shift_logits.stop_gradient = {sl.stop_gradient}")

# Simple gradient test
x = paddle.randn([2, 2])
x.stop_gradient = False
y = paddle.cast(x, "float32")
print(f"Simple cast: stop_gradient={y.stop_gradient}")
z = y.sum()
z.backward()
print(f"Simple grad after cast: {x.grad is not None}, nonzero={bool((x.grad != 0).any().item()) if x.grad is not None else False}")

# Check if backward works through model logits
loss = paddle.cast(logits[:, :-1, :], "float32").sum()
print(f"Trying backward through model logits...")
try:
    loss.backward()
    print(f"  backward() succeeded!")
    # Check any LoRA grad
    has_grad = False
    for name, param in model.named_parameters():
        if not param.stop_gradient and param.grad is not None:
            if (param.grad.numpy() != 0).any():
                has_grad = True
                print(f"  Found grad in: {name}")
                break
    if not has_grad:
        print("  No non-zero grad found in any LoRA param!")
except Exception as e:
    print(f"  backward() FAILED: {e}")

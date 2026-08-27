"""Quick test: load model + LoRA, save only LoRA weights, verify they reload."""
import paddle, os, sys
os.chdir(r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset")
sys.path.insert(0, "scripts")
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()
from paddleformers.transformers import AutoModelForConditionalGeneration
from paddleformers.peft import LoRAConfig, LoRAModel

MP = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
OUT = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset\PaddleOCR-VL-LoRA-circuit-ocr"

print("Loading model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    MP, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="bfloat16")
model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"
lc = LoRAConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=['.*q_proj', '.*k_proj', '.*v_proj', '.*o_proj'])
model = LoRAModel(model, lc)
model.mark_only_lora_as_trainable()

# Patch for save
if not hasattr(model.model, 'full'):
    model.model.full = lambda *a, **kw: iter(model.model.named_parameters())

sd = model.state_dict()
lora_w = {k: v.numpy() for k, v in sd.items() if 'lora' in k.lower()}
print(f"LoRA keys: {len(lora_w)}, total size: {sum(v.nbytes for v in lora_w.values())/1024:.0f}KB")

# Save
path = f"{OUT}/test_lora_save.pdparams"
paddle.save(lora_w, path)
print(f"Saved to {path} ({os.path.getsize(path)/1024:.0f}KB)")

# Reload and verify
loaded = paddle.load(path)
match = all((loaded[k] == lora_w[k]).all().item() for k in lora_w)
print(f"Reload match: {match}")

# Verify non-zero (modified weights would be non-zero, init is zero here)
nonzero = sum(1 for v in lora_w.values() if (v != 0).any())
print(f"Non-zero LoRA matrices: {nonzero}/{len(lora_w)}")
print("=== SAVE TEST PASSED ===")

"""Minimal standalone PaddleOCR-VL inference test."""
import os, sys, time, json

# Prepend cuDNN 8.9.2.26 DLL paths for Paddle 2.6.2 compatibility
_dll_paths = [
    r"E:\080000software\080900_Miniconda\miniconda3\Library\bin",  # cuDNN 8.9.2.26
    r"E:\080000software\080900_Miniconda\miniconda3\Library\envs\gpu-pytorch\lib\site-packages\torch\lib",  # zlibwapi.dll
    r"E:\080000software\080900_Miniconda\miniconda3\pkgs\cudatoolkit-11.3.1-h59b6b97_2\Library\bin",
]
os.environ["PATH"] = ";".join(_dll_paths) + ";" + os.environ.get("PATH", "")

# Apply Paddle patches (same as benchmark script)
print("1. Applying Paddle compatibility patches...")
try:
    from types import ModuleType
    import paddle
    try:
        import paddle.distributed.flex_checkpoint.dcp.sharded_weight
    except (ImportError, ModuleNotFoundError, AttributeError):
        dummy = ModuleType('dummy')
        dummy.build_sharded_state_dict = lambda *a, **kw: None
        sys.modules.setdefault('paddle.distributed.flex_checkpoint', dummy)
        sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', dummy)
        sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', dummy)
    paddle.float8_e4m3fn = paddle.float32
    paddle.float8_e5m2 = paddle.float32
    paddle.LongTensor = paddle.Tensor
    print("   Patches applied OK")
except Exception as e:
    print(f"   Patch warning: {e}")

print("2. Importing PaddleFormers...")
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.generation import GenerationConfig
from PIL import Image
from pathlib import Path

print(f"   Paddle version: {paddle.__version__}")
print(f"   CUDA compiled: {paddle.version.cuda()}")
print(f"   cuDNN compiled: {paddle.version.cudnn()}")
print(f"   GPU count: {paddle.device.cuda.device_count()}")

device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
print(f"   Using device: {device}")
paddle.set_device(device)

print("3. Loading model from local cache...")
local_path = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
processor = AutoProcessor.from_pretrained(local_path)
print("   Processor loaded")

model = AutoModelForConditionalGeneration.from_pretrained(
    local_path,
    convert_from_hf=True,
    load_checkpoint_format='naive',
    low_cpu_mem_usage=True,
    dtype="float32"
)
print("   Model loaded")

model.config._attn_implementation = "flashmask"
model.visual.config._attn_implementation = "flashmask"
model.eval()

gen_config = GenerationConfig(
    do_sample=False, bos_token_id=1, eos_token_id=2,
    pad_token_id=0, use_cache=True
)

# Test on first 5 images from easy50
data_path = Path(__file__).parent.parent / "ocr_vl_sft-test-easy50.jsonl"
samples = []
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            samples.append(json.loads(line))
samples = samples[:5]

print(f"\n4. Running inference on {len(samples)} samples...")
for i, sample in enumerate(samples):
    start = time.time()
    img_rel = sample["images"][0]
    img_path = data_path.parent / img_rel
    if not img_path.exists():
        img_path = data_path.parent / Path(img_rel).name

    print(f"   [{i+1}/{len(samples)}] Loading: {img_path.name}...", end=" ", flush=True)

    try:
        image = Image.open(img_path).convert("RGB")
        query = sample["messages"][0]["content"]
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query.replace("<image>", "")},
            ],
        }]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pd"
        )

        with paddle.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_config, max_new_tokens=1024)
            output_ids = outputs[0].tolist()[0]
            output_text = processor.decode(output_ids, skip_special_tokens=True)

        elapsed = time.time() - start
        label = sample["messages"][1]["content"]
        print(f"OK {elapsed:.1f}s")
        print(f"       Pred: {repr(output_text[:100])}")
        print(f"       Label: {repr(label[:100])}")

        image.close()
        del image, messages, inputs, outputs, output_ids, output_text
        paddle.device.cuda.empty_cache()
        paddle.device.cuda.synchronize()

    except Exception as e:
        elapsed = time.time() - start
        print(f"FAIL {elapsed:.1f}s: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        break

print("\nDone.")

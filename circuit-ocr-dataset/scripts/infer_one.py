"""Single-sample inference runner for PaddleOCR-VL.
Usage: python infer_one.py <image_path> <prompt> <output_json_path>
Designed to be called as an independent subprocess so crashes don't cascade.
"""
import os, sys, json, time

# Prepend cuDNN 8.9.2.26 DLL paths
_dll_paths = [
    r"E:\080000software\080900_Miniconda\miniconda3\Library\bin",
    r"E:\080000software\080900_Miniconda\miniconda3\Library\envs\gpu-pytorch\lib\site-packages\torch\lib",
]
os.environ["PATH"] = ";".join(_dll_paths) + ";" + os.environ.get("PATH", "")

# Import and apply all Paddle patches from the benchmark script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_benchmark import apply_paddle_patches
apply_paddle_patches()

import paddle

# Import PaddleFormers
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.generation import GenerationConfig
from PIL import Image

def run_one(image_path, prompt, max_length=1024):
    paddle.set_device('gpu')
    local_path = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"

    processor = AutoProcessor.from_pretrained(local_path)
    model = AutoModelForConditionalGeneration.from_pretrained(
        local_path, convert_from_hf=True, load_checkpoint_format='naive',
        low_cpu_mem_usage=True, dtype="float32"
    )
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"
    model.eval()

    gen_config = GenerationConfig(
        do_sample=False, bos_token_id=1, eos_token_id=2,
        pad_token_id=0, use_cache=True
    )

    img = Image.open(image_path).convert("RGB")
    # Resize large images
    w, h = img.size
    max_dim = 768
    if w > max_dim or h > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt.replace("<image>", "")},
    ]}]

    inputs = processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pd"
    )

    with paddle.no_grad():
        outputs = model.generate(**inputs, generation_config=gen_config, max_new_tokens=max_length)
    text = processor.decode(outputs[0].tolist()[0], skip_special_tokens=True)
    return text

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: infer_one.py <image_path> <prompt> [output_path]"}))
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    try:
        start = time.time()
        pred = run_one(image_path, prompt)
        elapsed = time.time() - start
        result = {"prediction": pred, "elapsed": elapsed, "status": "ok"}
    except Exception as e:
        result = {"prediction": "", "elapsed": time.time() - start, "status": "error", "error": str(e)}

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
    else:
        print(json.dumps(result, ensure_ascii=False))

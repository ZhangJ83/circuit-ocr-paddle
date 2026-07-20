"""Standalone checkpoint evaluator. Loads a LoRA checkpoint and evaluates on test data."""
import os, sys, json, time, argparse
from pathlib import Path
from io import BytesIO

# Patches
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if "torchvision" in sys.modules and sys.modules["torchvision"] is None:
    del sys.modules["torchvision"]

from types import ModuleType
_dummy_fc = ModuleType('dummy_flex_checkpoint')
_dummy_fc.build_sharded_state_dict = lambda *a, **kw: None
for mod in ['paddle.distributed.flex_checkpoint',
            'paddle.distributed.flex_checkpoint.dcp',
            'paddle.distributed.flex_checkpoint.dcp.sharded_weight']:
    sys.modules.setdefault(mod, _dummy_fc)

import paddle; paddle.set_device("gpu")
if not hasattr(paddle, 'LongTensor'): paddle.LongTensor = paddle.Tensor
import paddle.nn.functional as F
if not hasattr(F, 'swiglu'):
    F.swiglu = lambda x: paddle.chunk(x,2,-1)[0] * F.silu(paddle.chunk(x,2,-1)[1])

from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_DIR, 'scripts'))
from eval_metrics import compute_all


def evaluate_checkpoint(checkpoint_path, data_path, output_path, max_dim=384, limit=None):
    """Full evaluation of a LoRA checkpoint on test data."""
    print(f"Loading model...")
    model_path = "PaddlePaddle/PaddleOCR-VL"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForConditionalGeneration.from_pretrained(
        model_path, convert_from_hf=True, load_checkpoint_format="naive",
        low_cpu_mem_usage=True, dtype="bfloat16")
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"

    # Apply LoRA wrapper
    lora_config = LoRAConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                             target_modules=[".*q_proj",".*k_proj",".*v_proj",".*o_proj",".*linear_1",".*linear_2"])
    model = LoRAModel(model, lora_config)
    model.eval()

    # Load weights
    if os.path.exists(checkpoint_path):
        lora_state = paddle.load(checkpoint_path)
        model_lora_params = {k: p for k, p in model.named_parameters() if 'lora_' in k}
        loaded = 0
        for ckpt_key, ckpt_value in lora_state.items():
            if ckpt_key in model_lora_params:
                model_lora_params[ckpt_key].set_value(paddle.cast(ckpt_value, model_lora_params[ckpt_key].dtype))
                loaded += 1
        print(f"Loaded {loaded}/{len(lora_state)} LoRA params")
    else:
        print(f"WARNING: checkpoint not found: {checkpoint_path}")

    # Load data
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): samples.append(json.loads(line))
    if limit: samples = samples[:limit]
    print(f"Evaluating {len(samples)} samples...")

    predictions, references = [], []
    eos_id = processor.tokenizer.eos_token_id or 2

    for i, s in enumerate(samples):
        try:
            query = s["messages"][0]["content"]
            if isinstance(query, list):
                query = next((c["text"] for c in query if c["type"] == "text"), "<image>OCR:")
            ref_text = s["messages"][1]["content"]

            img_path = s["images"][0]
            if not os.path.exists(img_path):
                local = img_path.replace("/root/circuit_ocr/", str(PROJECT_DIR) + "/")
                if os.path.exists(local): img_path = local

            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            scale = max_dim / max(w, h)
            if scale < 1: image = image.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            buf = BytesIO(); image.save(buf, format='JPEG', quality=95); buf.seek(0)
            image = Image.open(buf)

            messages = [{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":query.replace("<image>","")}]}]
            inp = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")

            input_ids = inp["input_ids"]; attention_mask = inp["attention_mask"]
            pixel_values = inp.get("pixel_values"); image_grid_thw = inp.get("image_grid_thw")

            generated = []
            with paddle.no_grad():
                for _ in range(512):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
                    logits = outputs[0] if isinstance(outputs,(list,tuple)) else outputs.logits
                    ntl = logits[:, -1, :]
                    for tid in set(generated):
                        score = float(ntl[0, tid])
                        ntl[0, tid] = score * 1.1 if score < 0 else score / 1.1
                    nt = int(paddle.argmax(ntl, axis=-1).numpy()[0])
                    if nt == eos_id: break
                    generated.append(nt)
                    t = paddle.to_tensor([[nt]], dtype=input_ids.dtype)
                    input_ids = paddle.concat([input_ids, t], axis=1)
                    attention_mask = paddle.concat([attention_mask, paddle.ones([1,1], dtype=attention_mask.dtype)], axis=1)

            pred_text = processor.tokenizer.decode(generated, skip_special_tokens=True)
            predictions.append(pred_text); references.append(ref_text)
            image.close()

            if (i+1) % 20 == 0:
                print(f"  [{i+1}/{len(samples)}]")

        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            predictions.append("[ERROR]")
            references.append(s["messages"][1]["content"])

    metrics = compute_all(predictions, references, label="test")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    result = {"config": {"checkpoint": checkpoint_path, "data": data_path, "max_dim": max_dim},
              "metrics": metrics, "results": [{"pred": p, "ref": r} for p, r in zip(predictions, references)]}

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved: {output_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_dim", type=int, default=384)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.data, args.output, args.max_dim, args.limit)

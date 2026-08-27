import sys
import os
import json
import paddle

# Setup path to import PaddleOCR-VL dependencies
sys.path.insert(0, r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset\scripts")
from eval_benchmark import apply_paddle_patches
apply_paddle_patches()

import paddle
paddle.set_device("gpu")
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel
from PIL import Image

MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
CKPT_PATH = r"G:\mimo_project\circuit_ocr\hf_model_clone\lora_v9_pure_final_fp16.pdparams"
DATASET_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"
EXAMPLES_JSON_PATH = r"G:\mimo_project\circuit_ocr\hf_space\examples.json"

print("Loading model and processor...")
model_base = AutoModelForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    convert_from_hf=True, 
    load_checkpoint_format="naive", 
    low_cpu_mem_usage=True, 
    dtype="bfloat16"
)
model_base.config._attn_implementation = "flashmask"
model_base.visual.config._attn_implementation = "flashmask"

TARGETS = [".*q_proj", ".*k_proj", ".*v_proj", ".*o_proj", ".*linear_1", ".*linear_2"]
lc = LoRAConfig(r=16, lora_alpha=32, target_modules=TARGETS)
model = LoRAModel(model_base, lc)
model.set_state_dict(paddle.load(CKPT_PATH))
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Load current examples
with open(EXAMPLES_JSON_PATH, "r", encoding="utf-8") as f:
    examples = json.load(f)

print(f"Loaded {len(examples)} examples.")

for idx, ex in enumerate(examples):
    # Map from relative path in hf_space to actual local path in dataset
    relative_img_path = ex["image"].lstrip("./")
    local_img_path = os.path.join(DATASET_DIR, relative_img_path)
    
    print(f"\nProcessing example {idx+1}/{len(examples)}: {local_img_path}")
    if not os.path.exists(local_img_path):
        print(f"Error: image {local_img_path} not found!")
        continue
        
    img = Image.open(local_img_path).convert("RGB")
    w, h = img.size
    scale = 384 / max(w, h)
    img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    
    # Prompt text for literal OCR
    prompt_text = "Perform literal OCR. Detect all printed text, reference designators, component values, and labels. Output them in top-to-bottom order, one text per line."
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt_text}]}]
    
    inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pd")
    
    input_ids = inp["input_ids"]
    attention_mask = inp["attention_mask"]
    pixel_values = inp.get("pixel_values")
    image_grid_thw = inp.get("image_grid_thw")
    
    generated = []
    with paddle.no_grad():
        for step in range(120): # increased max steps for complete prediction
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw
            )
            logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs.logits
            next_token_logits = logits[:, -1, :]
            next_token = int(paddle.argmax(next_token_logits, axis=-1).numpy()[0])
            if next_token == processor.tokenizer.eos_token_id:
                break
            generated.append(next_token)
            next_tensor = paddle.to_tensor([[next_token]], dtype=input_ids.dtype)
            input_ids = paddle.concat([input_ids, next_tensor], axis=1)
            attention_mask = paddle.concat([attention_mask, paddle.ones([1, 1], dtype=attention_mask.dtype)], axis=1)
            
    resp = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
    print(f"GT:\n{ex['gt']}\nPrediction:\n{resp}")
    
    # Overwrite prediction
    ex["v8_pred"] = resp

# Save updated examples back
with open(EXAMPLES_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(examples, f, ensure_ascii=False, indent=2)

print("\nSuccessfully updated examples.json with correct V9-Pure predictions!")

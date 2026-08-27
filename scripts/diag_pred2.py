"""Diagnostic v2: use apply_chat_template (official method)."""
import os,sys,json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
import paddle;paddle.set_device('gpu')
import paddle.nn.functional as F
_o=F.scaled_dot_product_attention
F.scaled_dot_product_attention=lambda *a,**kw:_o(*a,**{k:v for k,v in kw.items() if k!='enable_gqa'})
import torch
from paddleformers.transformers import AutoModelForConditionalGeneration,AutoProcessor
from paddleformers.peft import LoRAConfig,LoRAModel
from paddleformers.generation import GenerationConfig

M='/root/models/official_models/PaddleOCR-VL'
P='/root/circuit_ocr'

print('Loading base model (NO LoRA)...')
proc=AutoProcessor.from_pretrained(M,trust_remote_code=True)
model=AutoModelForConditionalGeneration.from_pretrained(M,load_checkpoint_format='safetensors',dtype='bfloat16')
model.config._attn_implementation='sdpa';model.visual.config._attn_implementation='sdpa'
model.eval()
gc=GenerationConfig(do_sample=False,bos_token_id=1,eos_token_id=2,pad_token_id=0,use_cache=True)

# Test on 1 validation sample using apply_chat_template
vs=json.loads(open(P+'/output/val_clean.jsonl').readline())
vip=vs['images'][0]
if not os.path.exists(vip):vip=vip.replace('/root/circuit_ocr/',P+'/')
vimg=Image.open(vip).convert('RGB')

print('Image size:',vimg.size)
gt=vs['messages'][1]['content']
print('GT:',gt[:60])

# Official method: apply_chat_template
# Try string format: <image> placeholder in text
img_inputs=proc.image_processor(images=[np.array(vimg)],return_tensors='np')
messages=[{
    'role':'user',
    'content':'<image>OCR:'
}]
inputs=proc.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pd')
# Also add pixel_values
inputs['pixel_values']=paddle.to_tensor(img_inputs['pixel_values']) if isinstance(img_inputs['pixel_values'],np.ndarray) else paddle.to_tensor(img_inputs['pixel_values'])
inputs['image_grid_thw']=paddle.to_tensor(img_inputs['image_grid_thw']) if isinstance(img_inputs['image_grid_thw'],np.ndarray) else paddle.to_tensor(img_inputs['image_grid_thw'])
print('Input keys:',list(inputs.keys()))
print('Input shape:',inputs['input_ids'].shape)

with paddle.no_grad():
    out=model.generate(**inputs,generation_config=gc,max_new_tokens=128)
pred=proc.decode(out[0].tolist()[0],skip_special_tokens=True)
print('PRED (base):',pred[:80])
print()

# Now try with S800 LoRA checkpoint
print('Loading S800 LoRA...')
for n,p in model.named_parameters():
    if 'mlp_AR' in n or 'projector' in n:p.stop_gradient=True
lc=LoRAConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=['.*q_proj','.*k_proj','.*v_proj','.*o_proj','.*linear_1','.*linear_2'])
model=LoRAModel(model,lc)
model.model.full=lambda *a,**kw:iter(model.model.named_parameters())
sd=paddle.load(P+'/checkpoints/baseline/checkpoint_s800.pdparams')
loaded=0
for n,p in model.named_parameters():
    if n in sd:
        try:p.set_value(paddle.cast(sd[n],p.dtype));loaded+=1
        except:pass
print('Loaded',loaded,'LoRA params')
model.eval()

inputs2=proc.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pd')
inputs2['pixel_values']=paddle.to_tensor(img_inputs['pixel_values']) if isinstance(img_inputs['pixel_values'],np.ndarray) else paddle.to_tensor(img_inputs['pixel_values'])
inputs2['image_grid_thw']=paddle.to_tensor(img_inputs['image_grid_thw']) if isinstance(img_inputs['image_grid_thw'],np.ndarray) else paddle.to_tensor(img_inputs['image_grid_thw'])
with paddle.no_grad():
    out=model.generate(**inputs2,generation_config=gc,max_new_tokens=128)
pred=proc.decode(out[0].tolist()[0],skip_special_tokens=True)
print('PRED (S800):',pred[:80])
print('DONE')

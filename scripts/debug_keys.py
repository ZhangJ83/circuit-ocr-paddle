"""Debug: check placeholder tokens."""
import os,sys,json
sys.modules.pop('torchvision',None)
import torchvision,torchvision.transforms
from mistral_common.tokens.tokenizers import utils as mu
if not hasattr(mu,'get_one_valid_tokenizer_file'):
    mu.get_one_valid_tokenizer_file=lambda d,e:list(mu._filter_valid_tokenizer_files(d,e))
import paddle;paddle.set_device('gpu')
import paddle.nn.functional as F
if not hasattr(F,'swiglu'):F.swiglu=lambda x:paddle.chunk(x,2,-1)[0]*F.silu(paddle.chunk(x,2,-1)[1])
_o=F.scaled_dot_product_attention
F.scaled_dot_product_attention=lambda *a,**kw:_o(*a,**{k:v for k,v in kw.items() if k!='enable_gqa'})
import numpy as np
from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration,AutoProcessor
sys.modules.pop('torchvision',None)
import torchvision,torchvision.transforms,torch
import transformers.utils.import_utils as tiu
tiu.is_torch_available=lambda:(True,'')
tiu.is_torchvision_available=lambda:(True,'')

M='/root/models/official_models/PaddleOCR-VL'
P='/root/circuit_ocr'

proc=AutoProcessor.from_pretrained(M,trust_remote_code=True)
model=AutoModelForConditionalGeneration.from_pretrained(M,load_checkpoint_format='safetensors',dtype='bfloat16')

# Check placeholder token
ph_id=proc.tokenizer.encode('<|placeholder|>',add_special_tokens=False)
print('placeholder token IDs:',ph_id)
print('placeholder token str:',proc.tokenizer.decode(ph_id))

# Check special tokens
print('bos_token:',proc.tokenizer.bos_token,proc.tokenizer.bos_token_id)
print('eos_token:',proc.tokenizer.eos_token,proc.tokenizer.eos_token_id)
print('pad_token:',proc.tokenizer.pad_token,proc.tokenizer.pad_token_id)

# Check model config
print('model.image_token_id:',getattr(model.config,'image_token_id','NOT SET'))

# Check processor config
print('processor image_processor type:',type(proc.image_processor).__name__)

# Test on 1 sample
val=json.loads(open(P+'/output/val_clean.jsonl').readline())
vimg=Image.open(val['images'][0]).convert('RGB')
w,h=vimg.size;s=384/max(w,h)
if s<1:vimg=vimg.resize((int(w*s),int(h*s)),Image.LANCZOS)

feats=proc.image_processor(images=[np.array(vimg)],return_tensors='np')
g=feats['image_grid_thw'][0];vn=max(1,int(g[1])*int(g[2])//4)
prompt=('<' + '|placeholder|' + '>')*vn + 'OCR:'

inp=proc(text=[prompt],images=[np.array(vimg)],return_tensors='np',padding=False,max_length=1024,truncation=True)
ids=inp['input_ids'][0]
print('\nprompt[:60]:',prompt[:60])
print('first 20 input_ids:',[int(x) for x in ids[:20]])
# Decode first tokens
for i in range(min(10,len(ids))):
    t=int(ids[i])
    print(f'  [{i}] id={t} -> {repr(proc.tokenizer.decode([t]))}')
print('...')
# Check last tokens
print('last 5:',[int(x) for x in ids[-5:]])
print('decoded last:',proc.tokenizer.decode([int(x) for x in ids[-5:]]))

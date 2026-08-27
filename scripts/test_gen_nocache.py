"""Test: does model.generate() work with use_cache=False?"""
import os,sys,json
sys.modules.pop('torchvision',None)
import torchvision,torchvision.transforms
from mistral_common.tokens.tokenizers import utils as mu
if not hasattr(mu,'get_one_valid_tokenizer_file'):mu.get_one_valid_tokenizer_file=lambda d,e:list(mu._filter_valid_tokenizer_files(d,e))
import paddle;paddle.set_device('gpu')
import paddle.nn.functional as F
if not hasattr(F,'swiglu'):F.swiglu=lambda x:paddle.chunk(x,2,-1)[0]*F.silu(paddle.chunk(x,2,-1)[1])
_o=F.scaled_dot_product_attention;F.scaled_dot_product_attention=lambda *a,**kw:_o(*a,**{k:v for k,v in kw.items() if k!='enable_gqa'})
import numpy as np;from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration,AutoProcessor
from paddleformers.generation import GenerationConfig
sys.modules.pop('torchvision',None);import torchvision,torchvision.transforms,torch
import transformers.utils.import_utils as tiu
tiu.is_torch_available=lambda:(True,'');tiu.is_torchvision_available=lambda:(True,'')

M='/root/models/official_models/PaddleOCR-VL';P='/root/circuit_ocr'
from paddleformers.peft import LoRAConfig,LoRAModel
proc=AutoProcessor.from_pretrained(M,trust_remote_code=True)
model=AutoModelForConditionalGeneration.from_pretrained(M,load_checkpoint_format='safetensors',dtype='bfloat16')
model.config._attn_implementation='sdpa';model.visual.config._attn_implementation='sdpa'
for n,p in model.named_parameters():
    if 'mlp_AR' in n or 'projector' in n:p.stop_gradient=True
lc=LoRAConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=['.*q_proj','.*k_proj','.*v_proj','.*o_proj','.*linear_1','.*linear_2'])
model=LoRAModel(model,lc)
# Load S800 checkpoint
sd=paddle.load(P+'/checkpoints/fast/checkpoint_s800.pdparams')
for n,p in model.named_parameters():
    if n in sd:
        try:p.set_value(paddle.cast(sd[n],p.dtype))
        except:pass
print('Loaded S800 LoRA weights')
model.eval()

val=json.loads(open(P+'/output/val_clean.jsonl').readline())
vip=val['images'][0]
vimg=Image.open(vip).convert('RGB')
vw,vh=vimg.size;s=384/max(vw,vh)
if s<1:vimg=vimg.resize((int(vw*s),int(vh*s)),Image.LANCZOS)
vimg_np=np.array(vimg)
feats=proc.image_processor(images=[vimg_np],return_tensors='np')
g=feats['image_grid_thw'][0];vn=max(1,int(g[1])*int(g[2])//4)
prompt=('<' + '|placeholder|' + '>')*vn + 'OCR:'
inp=proc(text=[prompt],images=[vimg_np],return_tensors='np',padding=True,max_length=2048,truncation=True)

def td(d):
    o={}
    for k,v in d.items():
        if isinstance(v,np.ndarray):o[k]=paddle.to_tensor(v)
        elif isinstance(v,torch.Tensor):o[k]=paddle.to_tensor(v.numpy())
        elif isinstance(v,list) and len(v)>0:
            if isinstance(v[0],np.ndarray):o[k]=paddle.to_tensor(np.array(v))
            else:o[k]=v
        else:o[k]=v
    return o

ipd=td(inp)
ipd['pixel_values']=paddle.to_tensor(feats['pixel_values'])
ipd['image_grid_thw']=paddle.to_tensor(feats['image_grid_thw'])

gc=GenerationConfig(do_sample=False,bos_token_id=1,eos_token_id=2,pad_token_id=0,use_cache=False)

print('Test: generate with use_cache=False')
import time;t0=time.time()
with paddle.no_grad():
    out=model.generate(**ipd,generation_config=gc,max_new_tokens=64)
elapsed=time.time()-t0
pred=proc.tokenizer.decode(out[0].tolist()[0],skip_special_tokens=True)
print('Pred:',repr(pred[:100]))
print('Time:',elapsed,'s')
print('GT:',val['messages'][1]['content'][:80])

"""Test model output format with and without LoRA."""
import os,sys,json,numpy as np
sys.modules.pop('torchvision',None);import torchvision,torchvision.transforms
from mistral_common.tokens.tokenizers import utils as mu
if not hasattr(mu,'get_one_valid_tokenizer_file'):mu.get_one_valid_tokenizer_file=lambda d,e:list(mu._filter_valid_tokenizer_files(d,e))
import paddle;paddle.set_device('gpu')
import paddle.nn.functional as F
if not hasattr(F,'swiglu'):F.swiglu=lambda x:paddle.chunk(x,2,-1)[0]*F.silu(paddle.chunk(x,2,-1)[1])
_o=F.scaled_dot_product_attention;F.scaled_dot_product_attention=lambda*a,**kw:_o(*a,**{k:v for k,v in kw.items() if k!='enable_gqa'})
from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration,AutoProcessor
from paddleformers.peft import LoRAConfig,LoRAModel
sys.modules.pop('torchvision',None);import torchvision,torchvision.transforms,torch as t
import transformers.utils.import_utils as tiu
tiu.is_torch_available=lambda:(True,'');tiu.is_torchvision_available=lambda:(True,'')

M='/root/models/official_models/PaddleOCR-VL';P='/root/circuit_ocr'
proc=AutoProcessor.from_pretrained(M,trust_remote_code=True)

# Load sample
val=json.loads(open(P+'/output/val_clean.jsonl').readline())
vimg=Image.open(val['images'][0]).convert('RGB')
w,h=vimg.size;s=384/max(w,h)
if s<1:vimg=vimg.resize((int(w*s),int(h*s)),Image.LANCZOS)
vimg_np=np.array(vimg)
feats=proc.image_processor(images=[vimg_np],return_tensors='np')
g=feats['image_grid_thw'][0];vn=max(1,int(g[1])*int(g[2])//4)
prompt=('<' + '|placeholder|' + '>')*vn + 'OCR:'
inp=proc(text=[prompt],images=[vimg_np],return_tensors='np',padding=True,max_length=2048,truncation=True)

def build_inputs():
    ipd={}
    for k,v in inp.items():
        if isinstance(v,np.ndarray):ipd[k]=paddle.to_tensor(v)
        elif isinstance(v,t.Tensor):ipd[k]=paddle.to_tensor(v.numpy())
        elif isinstance(v,list) and len(v)>0:
            if isinstance(v[0],np.ndarray):ipd[k]=paddle.to_tensor(np.array(v))
            else:ipd[k]=v
        else:ipd[k]=v
    ipd['pixel_values']=paddle.to_tensor(feats['pixel_values']) if isinstance(feats['pixel_values'],np.ndarray) else paddle.to_tensor(feats['pixel_values'].numpy())
    ipd['image_grid_thw']=paddle.to_tensor(feats['image_grid_thw']) if isinstance(feats['image_grid_thw'],np.ndarray) else paddle.to_tensor(feats['image_grid_thw'].numpy())
    return ipd

# Test 1: BASE model, no LoRA
print('=== BASE model (no LoRA) ===')
m1=AutoModelForConditionalGeneration.from_pretrained(M,load_checkpoint_format='safetensors',dtype='bfloat16')
m1.config._attn_implementation='sdpa';m1.visual.config._attn_implementation='sdpa'
m1.eval()
ipd=build_inputs()
with paddle.no_grad():
    o1=m1(**ipd)
    print('type:',type(o1).__name__)
    if hasattr(o1,'logits'):
        lt=o1.logits
        print('has logits, shape:',lt.shape)
        t1=int(paddle.argmax(lt[:,-1,:],axis=-1).numpy()[0])
        print('top token:',t1,repr(proc.tokenizer.decode([t1])))
    else:
        print('no logits attr:',str(o1)[:200])

# Test 2: WITH LoRA (no checkpoint, fresh init)
print('=== WITH LoRA (fresh) ===')
for n,p in m1.named_parameters():
    if 'mlp_AR' in n or 'projector' in n:p.stop_gradient=True
lc=LoRAConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=['.*q_proj','.*k_proj','.*v_proj','.*o_proj','.*linear_1','.*linear_2'])
m2=LoRAModel(m1,lc)
m2.eval()
ipd=build_inputs()
with paddle.no_grad():
    o2=m2(**ipd)
    print('type:',type(o2).__name__)
    if isinstance(o2,(list,tuple)):
        print('tuple len:',len(o2),[type(x).__name__ for x in o2])
        for i,x in enumerate(o2):
            if hasattr(x,'shape'):print(f'  [{i}] shape:',x.shape)
            else:print(f'  [{i}] scalar:',float(x.numpy()))
    elif hasattr(o2,'logits'):
        lt=o2.logits
        print('has logits, shape:',lt.shape)
        t2=int(paddle.argmax(lt[:,-1,:],axis=-1).numpy()[0])
        print('top token:',t2,repr(proc.tokenizer.decode([t2])))
    else:
        print('unknown output:',str(o2)[:200])

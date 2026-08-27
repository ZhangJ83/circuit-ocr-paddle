"""Test v2: train() vs eval(), with vs without labels"""
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
sys.modules.pop('torchvision',None);import torchvision,torchvision.transforms,torch
import transformers.utils.import_utils as tiu
tiu.is_torch_available=lambda:(True,'');tiu.is_torchvision_available=lambda:(True,'')

M='/root/models/official_models/PaddleOCR-VL';P='/root/circuit_ocr'
proc=AutoProcessor.from_pretrained(M,trust_remote_code=True)
model=AutoModelForConditionalGeneration.from_pretrained(M,load_checkpoint_format='safetensors',dtype='bfloat16')
model.config._attn_implementation='sdpa';model.visual.config._attn_implementation='sdpa'

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

# Test 1: eval(), no labels
model.eval()
print('Test 1: eval(), no labels')
with paddle.no_grad():
    o1=model(**ipd)
    l1=o1[0] if isinstance(o1,(list,tuple)) else o1.logits
    t1=int(paddle.argmax(l1[:,-1,:],axis=-1).numpy()[0])
print('  token:',t1,repr(proc.tokenizer.decode([t1])))

# Test 2: train(), no labels
model.train()
print('Test 2: train(), no labels')
o2=model(**ipd)
l2=o2[0] if isinstance(o2,(list,tuple)) else o2.logits
t2=int(paddle.argmax(l2[:,-1,:],axis=-1).numpy()[0])
print('  token:',t2,repr(proc.tokenizer.decode([t2])))

# Test 3: eval(), with actual labels
model.eval()
label_ids=proc.tokenizer.encode(val['messages'][1]['content'])+[2]
label_t=paddle.to_tensor(label_ids,dtype='int64')
plen=ipd['input_ids'].shape[1]
ipd['input_ids']=paddle.concat([ipd['input_ids'][0],label_t]).unsqueeze(0)
ipd['labels']=paddle.concat([paddle.full([plen],-100,dtype='int64'),label_t]).unsqueeze(0)
ipd['attention_mask']=paddle.ones([1,ipd['input_ids'].shape[1]],dtype='int64')
print('Test 3: eval(), WITH labels')
with paddle.no_grad():
    o3=model(**ipd)
    if isinstance(o3,(list,tuple)):
        loss=o3[0]
        logits=o3[1] if len(o3)>1 else None
        print('  loss:',float(loss.numpy()[0]))
        if logits is not None:
            t3=int(paddle.argmax(logits[:,-1,:],axis=-1).numpy()[0])
            print('  token:',t3,repr(proc.tokenizer.decode([t3])))
    else:
        print('  type:',type(o3))

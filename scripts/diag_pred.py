"""Diagnostic: what does the trained model actually predict?"""
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

M='/root/models/official_models/PaddleOCR-VL'
P='/root/circuit_ocr'
CKPT=P+'/checkpoints/baseline/checkpoint_s800.pdparams'

print('Loading...')
proc=AutoProcessor.from_pretrained(M,trust_remote_code=True)
model=AutoModelForConditionalGeneration.from_pretrained(M,load_checkpoint_format='safetensors',dtype='bfloat16')
model.config._attn_implementation='sdpa';model.visual.config._attn_implementation='sdpa'
for n,p in model.named_parameters():
    if 'mlp_AR' in n or 'projector' in n:p.stop_gradient=True
lc=LoRAConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=['.*q_proj','.*k_proj','.*v_proj','.*o_proj','.*linear_1','.*linear_2'])
model=LoRAModel(model,lc)

# Load S800 checkpoint
sd=paddle.load(CKPT)
for n,p in model.named_parameters():
    if n in sd:
        try:p.set_value(paddle.cast(sd[n],p.dtype))
        except:pass
model.eval()
print('S800 checkpoint loaded!')

# Test on 3 validation samples
val=[json.loads(l) for l in open(P+'/output/val_clean.jsonl')][:3]
for i,vs in enumerate(val):
    vip=vs['images'][0]
    if not os.path.exists(vip):vip=vip.replace('/root/circuit_ocr/',P+'/')
    vimg=Image.open(vip).convert('RGB')
    vw,vh=vimg.size;s=384/max(vw,vh)
    if s<1:vimg=vimg.resize((int(vw*s),int(vh*s)),Image.LANCZOS)
    feats=proc.image_processor(images=[np.array(vimg)],return_tensors='np')
    g=feats['image_grid_thw'][0];vn=max(1,int(g[1])*int(g[2])//4)
    prompt=('<' + '|placeholder|' + '>')*vn + 'OCR:'
    vinp=proc(text=[prompt],images=[np.array(vimg)],return_tensors='np',padding=False,max_length=1024,truncation=True)
    ipd={}
    for k,v in vinp.items():
        if isinstance(v,np.ndarray):ipd[k]=paddle.to_tensor(v)
        elif isinstance(v,torch.Tensor):ipd[k]=paddle.to_tensor(v.numpy())
        elif isinstance(v,list):
            if len(v)>0 and isinstance(v[0],np.ndarray):ipd[k]=paddle.to_tensor(np.array(v))
            else:ipd[k]=v
        else:ipd[k]=v
    # CRITICAL: add pixel_values from image_processor
    ipd['pixel_values']=paddle.to_tensor(feats['pixel_values']) if isinstance(feats['pixel_values'],np.ndarray) else paddle.to_tensor(feats['pixel_values'].numpy())
    ipd['image_grid_thw']=paddle.to_tensor(feats['image_grid_thw']) if isinstance(feats['image_grid_thw'],np.ndarray) else paddle.to_tensor(feats['image_grid_thw'].numpy())
    # Manual decode
    gen=[];eos=2
    with paddle.no_grad():
        for _ in range(256):
            vo=model(**ipd)
            vl=vo[0] if isinstance(vo,(list,tuple)) else vo.logits
            vt=vl[:,-1,:]
            for tid in set(gen):
                sc=float(vt[0,tid])
                vt[0,tid]=sc*1.1 if sc<0 else sc/1.1
            nt=int(paddle.argmax(vt,axis=-1).numpy()[0])
            if nt==eos:break
            gen.append(nt)
            ipd['input_ids']=paddle.concat([ipd['input_ids'],paddle.to_tensor([[nt]])],axis=1)
            ipd['attention_mask']=paddle.concat([ipd['attention_mask'],paddle.ones([1,1],dtype='int64')],axis=1)
    pred=proc.tokenizer.decode(gen,skip_special_tokens=True)
    gt=vs['messages'][1]['content']
    print('['+str(i)+'] GT:   '+gt[:80])
    print('['+str(i)+'] PRED: '+pred[:80])
    print()
print('DONE')

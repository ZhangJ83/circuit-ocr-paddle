"""Evaluate checkpoints using model.generate() — fast."""
import os,sys,json,time
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
import paddle;paddle.set_device('gpu')
import paddle.nn.functional as F
if not hasattr(F,'swiglu'):F.swiglu=lambda x:paddle.chunk(x,2,-1)[0]*F.silu(paddle.chunk(x,2,-1)[1])
_o=F.scaled_dot_product_attention
F.scaled_dot_product_attention=lambda *a,**kw:_o(*a,**{k:v for k,v in kw.items() if k!='enable_gqa'})
import torch
from paddleformers.transformers import AutoModelForConditionalGeneration,AutoProcessor
from paddleformers.peft import LoRAConfig,LoRAModel
from paddleformers.generation import GenerationConfig
from eval_metrics import compute_all

M='/root/models/official_models/PaddleOCR-VL'
P='/root/circuit_ocr'
MAX_DIM=384
VAL_SAMPLES=10

def to_pd(d):
    o={}
    for k,v in d.items():
        if isinstance(v,np.ndarray):o[k]=paddle.to_tensor(v)
        elif isinstance(v,torch.Tensor):o[k]=paddle.to_tensor(v.numpy())
        elif isinstance(v,list) and len(v)>0:
            if isinstance(v[0],np.ndarray):o[k]=paddle.to_tensor(np.array(v))
            elif isinstance(v[0],torch.Tensor):o[k]=paddle.to_tensor(np.array([x.numpy() for x in v]))
            else:o[k]=v
        else:o[k]=v
    return o

print('Loading model...')
sys.stdout.flush()
proc=AutoProcessor.from_pretrained(M,trust_remote_code=True)
model=AutoModelForConditionalGeneration.from_pretrained(M,load_checkpoint_format='safetensors',dtype='bfloat16')
model.config._attn_implementation='sdpa';model.visual.config._attn_implementation='sdpa'
for n,p in model.named_parameters():
    if 'mlp_AR' in n or 'projector' in n:p.stop_gradient=True
lc=LoRAConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=['.*q_proj','.*k_proj','.*v_proj','.*o_proj','.*linear_1','.*linear_2'])
model=LoRAModel(model,lc)
model.model.full=lambda *a,**kw:iter(model.model.named_parameters())
gc=GenerationConfig(do_sample=False,bos_token_id=1,eos_token_id=2,pad_token_id=0,use_cache=True)

val=[json.loads(l) for l in open(P+'/output/val_clean.jsonl')][:VAL_SAMPLES]

for step in [400,800,1200,1600,2000,2400]:
    ckpt=P+'/checkpoints/baseline/checkpoint_s'+str(step)+'.pdparams'
    if not os.path.exists(ckpt):
        print('S'+str(step)+': MISSING')
        continue
    t0=time.time()
    sd=paddle.load(ckpt)
    loaded=0
    for n,p in model.named_parameters():
        if n in sd:
            try:p.set_value(paddle.cast(sd[n],p.dtype));loaded+=1
            except:pass
    model.eval();preds=[];refs=[]
    with paddle.no_grad():
        for vs in val:
            try:
                vip=vs['images'][0]
                if not os.path.exists(vip):vip=vip.replace('/root/circuit_ocr/',P+'/')
                vimg=Image.open(vip).convert('RGB')
                vw,vh=vimg.size;s=MAX_DIM/max(vw,vh)
                if s<1:vimg=vimg.resize((int(vw*s),int(vh*s)),Image.LANCZOS)
                feats=proc.image_processor(images=[np.array(vimg)],return_tensors='np')
                g=feats['image_grid_thw'][0];vn=max(1,int(g[1])*int(g[2])//4)
                prompt=('<' + '|placeholder|' + '>')*vn + 'OCR:'
                vinp=proc(text=[prompt],images=[np.array(vimg)],return_tensors='np',padding=False,max_length=1024,truncation=True)
                vinp_pd=to_pd(vinp)
                vinp_pd['pixel_values']=paddle.to_tensor(feats['pixel_values']) if isinstance(feats['pixel_values'],np.ndarray) else paddle.to_tensor(feats['pixel_values'])
                vinp_pd['image_grid_thw']=paddle.to_tensor(feats['image_grid_thw']) if isinstance(feats['image_grid_thw'],np.ndarray) else paddle.to_tensor(feats['image_grid_thw'])
                out=model.generate(**vinp_pd,generation_config=gc,max_new_tokens=256)
                pred=proc.tokenizer.decode(out[0].tolist()[0],skip_special_tokens=True)
                preds.append(pred);refs.append(vs['messages'][1]['content'])
            except Exception as e:
                preds.append('[ERROR]');refs.append(vs['messages'][1]['content'])
    m=compute_all(preds,refs,label='s'+str(step))
    elapsed=time.time()-t0
    print('S'+str(step)+': jf1='+str(round(m['joint_f1'],4))+' CompF1='+str(round(m['component_f1'],4))+' NED='+str(round(m['ned'],4))+' ('+str(round(elapsed,1))+'s)')
    sys.stdout.flush()
print('DONE')

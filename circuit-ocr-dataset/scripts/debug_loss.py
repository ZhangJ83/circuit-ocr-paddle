#!/usr/bin/env python3
"""Debug: test compute_loss on a single sample."""
import os, sys, json
os.environ.update({"KMP_DUPLICATE_LIB_OK":"TRUE","HF_HOME":"/mnt/f/hf_cache/hub","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"})

import paddle
import paddle.distributed.fleet.meta_parallel as mp
if not hasattr(mp,'LocalSharedLayerDesc'):
    class _LocalSharedLayerDesc:
        def __init__(self,*a,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
    mp.LocalSharedLayerDesc=_LocalSharedLayerDesc

from types import ModuleType
try:import paddle.distributed.flex_checkpoint.dcp.sharded_weight
except:
    dummy=ModuleType('dummy')
    for f in ['build_sharded_state_dict','create_sharded_weight_with_new_local','reshape_sharded_weight','sharded_weight_parallel_cpu','save_state_dict','load_state_dict']:
        setattr(dummy,f,lambda *a,**kw:None)
    for m in ['paddle.distributed.flex_checkpoint','paddle.distributed.flex_checkpoint.dcp','paddle.distributed.flex_checkpoint.dcp.sharded_weight']:
        sys.modules.setdefault(m,dummy)

paddle.float8_e4m3fn=paddle.float32;paddle.float8_e5m2=paddle.float32
paddle.LongTensor=paddle.Tensor;paddle.linalg.fp8_fp8_half_gemm_fused=None
paddle.Tensor.long=lambda s:s.astype('int64')
paddle.Tensor.float=lambda s:s.astype('float32')
paddle.Tensor.half=lambda s:s.astype('float16')
_old_reshape=paddle.Tensor.reshape
def _pr(self,*a,**kw):
    if a:
        if isinstance(a[0],paddle.dtype):return self.astype(a[0])
        if len(a)>1:ns=list(a)
        elif len(a)==1 and(isinstance(a[0],int)or hasattr(a[0],'__index__')):ns=[int(a[0])]
        else:ns=a[0]
        return _old_reshape(self,ns,**kw)
    return _old_reshape(self,**kw)
paddle.Tensor.reshape=_pr;paddle.Tensor.view=_pr
if not hasattr(paddle.Tensor,'repeat'):paddle.Tensor.repeat=paddle.Tensor.tile
_old_transpose=paddle.Tensor.transpose
def _pt(self,*a,**kw):
    if len(a)==2 and isinstance(a[0],int)and isinstance(a[1],int):
        d0,d1=a[0],a[1];nd=self.ndim
        if d0<0:d0+=nd
        if d1<0:d1+=nd
        perm=list(range(nd));perm[d0],perm[d1]=perm[d1],perm[d0]
        return _old_transpose(self,perm,**kw)
    return _old_transpose(self,*a,**kw)
paddle.Tensor.transpose=_pt
paddle.Tensor.masked_scatter=lambda s,m,src:None
_old_gf=paddle.base.framework.get_flags
paddle.base.framework.get_flags=lambda flags:{f:2 if f=='FLAGS_flash_attn_version'else _old_gf([f]).get(f)for f in flags}
_old_sf=paddle.set_flags
paddle.set_flags=lambda d:_old_sf({k:v for k,v in d.items()if k!='FLAGS_flash_attn_version'})if{k:v for k,v in d.items()if k!='FLAGS_flash_attn_version'}else None
_old_gelu=paddle.nn.functional.gelu
paddle.nn.functional.gelu=lambda x,approximate=False,name=None:_old_gelu(x,approximate=='tanh'if isinstance(approximate,str)else approximate,name)
for nm in['empty','zeros','ones','arange','full','randn','rand']:
    if hasattr(paddle,nm):
        of=getattr(paddle,nm)
        setattr(paddle,nm,lambda*a,_of=of,**kw:_of(*a,**{k:v for k,v in kw.items()if k!='device'}))
paddle.nn.functional.swiglu=lambda*a,**kw:None
def _frms(x,w,eps=1e-6):
    v=paddle.mean(paddle.square(x),axis=-1,keepdim=True)
    r=paddle.rsqrt(v+eps);return(x*r*w,r)
paddle.incubate.nn.functional.fused_rms_norm_ext=_frms
def _fma(q,k,v,startend_row_indices=None,causal=True):
    qt,kt,vt=q.transpose([0,2,1,3]),k.transpose([0,2,1,3]),v.transpose([0,2,1,3])
    b,hq,lq,d=qt.shape;_,hk,lk,_=kt.shape
    if hq!=hk:
        nr=hq//hk
        kt=paddle.tile(kt.reshape([b,hk,1,lk,d]),[1,1,nr,1,1]).reshape([b,hq,lk,d])
        vt=paddle.tile(vt.reshape([b,hk,1,lk,d]),[1,1,nr,1,1]).reshape([b,hq,lk,d])
    try:return paddle.nn.functional.scaled_dot_product_attention(qt,kt,vt,is_causal=causal,training=False).transpose([0,2,1,3])
    except:
        scores=paddle.matmul(qt,kt.transpose([0,1,3,2]))/(d**0.5)
        attn=paddle.nn.functional.softmax(scores,axis=-1)
        return paddle.matmul(attn,vt).transpose([0,2,1,3])
paddle.nn.functional.flash_attention.flashmask_attention=_fma
paddle.incubate.tensor.manipulation.create_async_load=lambda*a,**kw:None

from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from PIL import Image
from pathlib import Path

DATASET_DIR='/mnt/g/mimo_project/circuit_ocr/circuit-ocr-dataset'
mpth='/mnt/f/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27'

paddle.set_device('gpu')
print('Loading processor...')
processor=AutoProcessor.from_pretrained(mpth)
print('Loading model...')
model=AutoModelForConditionalGeneration.from_pretrained(mpth,convert_from_hf=True,load_checkpoint_format='naive',low_cpu_mem_usage=True,dtype='bfloat16')
model.config._attn_implementation='flashmask'
model.visual.config._attn_implementation='flashmask'
print('Model loaded OK')

# Test single sample
train=[json.loads(l)for l in open(f'{DATASET_DIR}/ocr_vl_sft-train.jsonl')if l.strip()]
s=train[0]
img_path=s['images'][0]
if not img_path.startswith('/'):img_path=f"{DATASET_DIR}/{img_path.lstrip('./')}"
print(f"Image: {img_path}")
print(f"Exists: {Path(img_path).exists()}")

try:
    image=Image.open(img_path).convert('RGB')
    print(f"Image size: {image.size}")
    w,h=image.size
    if max(w,h)>224:
        scale=224.0/max(w,h)
        image=image.resize((int(w*scale),int(h*scale)),Image.LANCZOS)
        print(f"Resized: {image.size}")
    query=s['messages'][0]['content']
    label=s['messages'][1]['content']
    print(f"Query: {query[:60]}")
    print(f"Label len: {len(label)} chars")
    msgs=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':query.replace('<image>','')}]}]
    print("Apply chat template...")
    inputs=processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pd')
    for k,v in inputs.items():
        if hasattr(v,'shape'):print(f"  {k}: {v.shape}")
    print("Tokenize label...")
    lt=processor.tokenizer(label,return_tensors='pd',padding=False,truncation=True,max_length=2048)
    print(f"  label_ids: {lt['input_ids'][0].shape}")
    print("Forward...")
    out=model(**inputs,labels=lt['input_ids'][0].unsqueeze(0))
    if hasattr(out,'loss'):print(f"LOSS={float(out.loss):.6f}")
    else:print(f"No loss attr! type={type(out)}, keys={list(out.keys())if hasattr(out,'keys')else'N/A'}")
    image.close()
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()
print("Done")

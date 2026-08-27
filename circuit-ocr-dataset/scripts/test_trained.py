"""Minimal test: load trained safetensors model and test forward pass."""
import os, sys, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HOME"] = "F:/hf_cache/hub"
os.environ["PADDLE_HOME"] = "F:/paddle_cache"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["PATH"] = (
    r"E:\080000software\080900_Miniconda\miniconda3\Library\bin;"
    r"E:\080000software\080900_Miniconda\miniconda3\envs\pyqpanda-quantum\Lib\site-packages\torch\lib;"
    + os.environ.get("PATH", "")
)
sys.stdout.reconfigure(line_buffering=True)

def log(msg):
    print(msg, flush=True)

import paddle
import paddle.distributed.fleet.meta_parallel as mp
if not hasattr(mp, 'LocalSharedLayerDesc'):
    class _L: __init__=lambda s,*a,**kw:None; __enter__=lambda s:s; __exit__=lambda s,*a:None
    mp.LocalSharedLayerDesc=_L
from types import ModuleType
try: import paddle.distributed.flex_checkpoint.dcp.sharded_weight
except:
    d=ModuleType('d')
    for f in ['build_sharded_state_dict','create_sharded_weight_with_new_local','reshape_sharded_weight','sharded_weight_parallel_cpu','save_state_dict','load_state_dict']:
        setattr(d,f,lambda *a,**kw:None)
    for m in ['paddle.distributed.flex_checkpoint','paddle.distributed.flex_checkpoint.dcp','paddle.distributed.flex_checkpoint.dcp.sharded_weight']:
        sys.modules.setdefault(m,d)

paddle.float8_e4m3fn=paddle.float32; paddle.float8_e5m2=paddle.float32; paddle.LongTensor=paddle.Tensor
paddle.linalg.fp8_fp8_half_gemm_fused=None
paddle.Tensor.long=lambda s:s.astype("int64"); paddle.Tensor.float=lambda s:s.astype("float32"); paddle.Tensor.half=lambda s:s.astype("float16")

_old_r=paddle.Tensor.reshape
def _pr(self,*args,**kwargs):
    if args:
        if isinstance(args[0],paddle.dtype): return self.astype(args[0])
        if len(args)>1: return _old_r(self,list(args),**kwargs)
        if isinstance(args[0],int): return _old_r(self,[int(args[0])],**kwargs)
        return _old_r(self,args[0],**kwargs)
    return _old_r(self,**kwargs)
paddle.Tensor.reshape=_pr; paddle.Tensor.view=_pr
if not hasattr(paddle.Tensor,"repeat"): paddle.Tensor.repeat=paddle.Tensor.tile

_old_t=paddle.Tensor.transpose
def _pt(self,*args,**kwargs):
    if len(args)==2 and isinstance(args[0],int) and isinstance(args[1],int):
        d0,d1=args[0],args[1]; nd=self.ndim
        if d0<0: d0+=nd
        if d1<0: d1+=nd
        perm=list(range(nd)); perm[d0],perm[d1]=perm[d1],perm[d0]
        return _old_t(self,perm,**kwargs)
    return _old_t(self,*args,**kwargs)
paddle.Tensor.transpose=_pt

def _pms(self,mask,source):
    orig=self.shape; mask=mask.astype('bool')
    fs,fm,fsrc=self.flatten(),mask.flatten(),source.flatten()
    idx=paddle.nonzero(fm); scat=paddle.scatter_nd(idx,fsrc,fm.shape)
    return paddle.where(fm,scat,fs).reshape(orig)
paddle.Tensor.masked_scatter=_pms

_old_gf=paddle.base.framework.get_flags
paddle.base.framework.get_flags=lambda flags:{f:2 if f=="FLAGS_flash_attn_version" else _old_gf([f]).get(f) for f in flags}
_old_sf=paddle.set_flags
paddle.set_flags=lambda d:_old_sf({k:v for k,v in d.items() if k!="FLAGS_flash_attn_version"}) if {k:v for k,v in d.items() if k!="FLAGS_flash_attn_version"} else None

_old_gelu=paddle.nn.functional.gelu
paddle.nn.functional.gelu=lambda x,approximate=False,name=None:_old_gelu(x,approximate=='tanh' if isinstance(approximate,str) else approximate,name)

for nm in ['empty','zeros','ones','arange','full','randn','rand']:
    if hasattr(paddle,nm):
        of=getattr(paddle,nm)
        setattr(paddle,nm,lambda *a,_of=of,**kw:_of(*a,**{k:v for k,v in kw.items() if k!='device'}))

paddle.nn.functional.swiglu=lambda *a,**kw:None

def _frms(x,w,eps=1e-6):
    v=paddle.mean(paddle.square(x),axis=-1,keepdim=True)
    r=paddle.rsqrt(v+eps); return (x*r*w,r)
paddle.incubate.nn.functional.fused_rms_norm_ext=_frms

def _fma(q,k,v,startend_row_indices=None,causal=True):
    qt,kt,vt=q.transpose([0,2,1,3]),k.transpose([0,2,1,3]),v.transpose([0,2,1,3])
    b,hq,lq,d=qt.shape; _,hk,lk,_=kt.shape
    if hq!=hk:
        nr=hq//hk
        kt=paddle.tile(kt.reshape([b,hk,1,lk,d]),[1,1,nr,1,1]).reshape([b,hq,lk,d])
        vt=paddle.tile(vt.reshape([b,hk,1,lk,d]),[1,1,nr,1,1]).reshape([b,hq,lk,d])
    uc=causal and lq==lk; am=None
    if causal and not uc:
        ri=paddle.arange(lq,dtype='int32').reshape([1,1,lq,1])
        ci=paddle.arange(lk,dtype='int32').reshape([1,1,1,lk])
        cb=ci<=(lk-lq+ri)
        am=paddle.where(cb,paddle.zeros([1,1,lq,lk],dtype=q.dtype),paddle.full([1,1,lq,lk],-1e9,dtype=q.dtype))
        if b>1: am=paddle.tile(am,[b,1,1,1])
    try:
        return paddle.nn.functional.scaled_dot_product_attention(qt,kt,vt,attn_mask=am,is_causal=uc,training=False).transpose([0,2,1,3])
    except:
        scores=paddle.matmul(qt,kt.transpose([0,1,3,2]))/(d**0.5)
        if am is not None: scores=scores+am
        if uc:
            gq=paddle.arange(lq,dtype="int32").reshape([lq,1])
            gk=paddle.arange(lk,dtype="int32").reshape([1,lk])
            scores=paddle.where((gk-gq)<=(lk-lq),scores,paddle.to_tensor(-1e9,dtype=scores.dtype))
        return paddle.matmul(paddle.nn.functional.softmax(scores,axis=-1),vt).transpose([0,2,1,3])
paddle.nn.functional.flash_attention.flashmask_attention=_fma
paddle.incubate.tensor.manipulation.create_async_load=lambda *a,**kw:None

import numpy as np, tempfile
from safetensors.numpy import save_file, safe_open
tmp_path=tempfile.mktemp(suffix='.safetensors')
save_file({'dummy':np.zeros((1,))},tmp_path)
with safe_open(tmp_path,framework='np') as f:
    PySafeSlice=type(f.get_slice('dummy'))
    setattr(PySafeSlice,'shape',property(lambda self:self.get_shape()))
os.remove(tmp_path)

log("[0] Patches OK, setting GPU")
paddle.set_device("gpu")

from paddleformers.transformers import AutoModelForConditionalGeneration
from paddleformers.generation import GenerationConfig

# Test 1: Original base model
BASE_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
TRAINED_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL-LoRA-circuit-ocr"

log("[1] Loading BASE model...")
m1 = AutoModelForConditionalGeneration.from_pretrained(
    BASE_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="float32"
)
m1.config._attn_implementation = "flashmask"
m1.visual.config._attn_implementation = "flashmask"
m1.eval()

log("[2] BASE forward test...")
ids = paddle.to_tensor([[1, 2, 3, 4, 5]], dtype="int64")
with paddle.no_grad():
    o1 = m1(input_ids=ids, use_cache=False)
log(f"  BASE logits shape: {list(o1[0].shape)}")

del m1
paddle.device.cuda.empty_cache()

log("[3] Loading TRAINED model...")
m2 = AutoModelForConditionalGeneration.from_pretrained(
    TRAINED_PATH, convert_from_hf=True, load_checkpoint_format="naive",
    low_cpu_mem_usage=True, dtype="float32"
)
m2.config._attn_implementation = "flashmask"
m2.visual.config._attn_implementation = "flashmask"
m2.eval()

log("[4] TRAINED forward test...")
try:
    with paddle.no_grad():
        o2 = m2(input_ids=ids, use_cache=False)
    log(f"  TRAINED logits shape: {list(o2[0].shape)}")
    log("[5] SUCCESS! Trained model inference works!")
except Exception as e:
    log(f"  TRAINED forward CRASH: {e}")

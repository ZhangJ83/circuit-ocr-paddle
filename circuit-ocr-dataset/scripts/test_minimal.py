"""Minimal test: does modifying any model weight crash forward?"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HOME"] = "F:/hf_cache/hub"
os.environ["PADDLE_HOME"] = "F:/paddle_cache"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

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

paddle.set_device("gpu")
import numpy as np
from paddleformers.transformers import AutoModelForConditionalGeneration

LOCAL=r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"

print("[1] Load model...", flush=True)
model=AutoModelForConditionalGeneration.from_pretrained(LOCAL,convert_from_hf=True,load_checkpoint_format="naive",low_cpu_mem_usage=True,dtype="float32")
model.config._attn_implementation="flashmask"
model.visual.config._attn_implementation="flashmask"

# Dummy forward
print("[2] Base forward...", flush=True)
ids=paddle.to_tensor([[1,2,3,4,5]],dtype="int64")
with paddle.no_grad():
    out=model(input_ids=ids,use_cache=False)
print(f"  OK, logits={list(out[0].shape)}", flush=True)

# Test 1: set_value on state_dict
print("[3] set_value test...", flush=True)
sd=model.state_dict()
k0=list(sd.keys())[0]
orig_val=sd[k0].numpy().copy()
sd[k0].set_value(paddle.to_tensor(orig_val*0.5,place=sd[k0].place))
print(f"  Modified {k0}", flush=True)

print("[4] Forward after set_value...", flush=True)
try:
    with paddle.no_grad(): out2=model(input_ids=ids,use_cache=False)
    print(f"  OK, logits={list(out2[0].shape)}", flush=True)
except Exception as e:
    print(f"  CRASH: {e}", flush=True)

# Test 2: named_parameters direct
print("[5] named_parameters test...", flush=True)
first_param=None
first_name=None
for n,p in model.named_parameters():
    first_name=n; first_param=p; break
val=first_param.numpy().copy()
first_param.set_value(paddle.to_tensor(val*0.5,place=first_param.place))
print(f"  Modified {first_name}", flush=True)

print("[6] Forward after param.set_value...", flush=True)
try:
    with paddle.no_grad(): out3=model(input_ids=ids,use_cache=False)
    print(f"  OK, logits={list(out3[0].shape)}", flush=True)
except Exception as e:
    print(f"  CRASH: {e}", flush=True)

print("[7] DONE", flush=True)

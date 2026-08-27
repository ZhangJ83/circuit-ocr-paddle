"""One-shot benchmark: load model once, manual decode all samples across all tiers.

Key insight: manual decode with image inputs DOES work (test_trained_img.py proved it).
We just need to avoid generate() and handle per-sample crashes gracefully.
"""
import os, sys, json, time, gc
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HOME"] = "F:/hf_cache/hub"
os.environ["PADDLE_HOME"] = "F:/paddle_cache"
os.environ["HF_HUB_CACHE"] = "F:/hf_cache/hub"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["PATH"] = (
    r"E:\080000software\080900_Miniconda\miniconda3\Library\bin;"
    r"E:\080000software\080900_Miniconda\miniconda3\envs\pyqpanda-quantum\Lib\site-packages\torch\lib;"
    + os.environ.get("PATH", "")
)

sys.stdout.reconfigure(line_buffering=True)

# ========== PATCHES (copied from eval_benchmark.py apply_paddle_patches) ==========
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
    v=paddle.mean(paddle.square(x),axis=-1,keepdim=True); r=paddle.rsqrt(v+eps); return (x*r*w,r)
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
        ri=paddle.arange(lq,dtype='int32').reshape([1,1,lq,1]); ci=paddle.arange(lk,dtype='int32').reshape([1,1,1,lk])
        cb=ci<=(lk-lq+ri)
        am=paddle.where(cb,paddle.zeros([1,1,lq,lk],dtype=q.dtype),paddle.full([1,1,lq,lk],-1e9,dtype=q.dtype))
        if b>1: am=paddle.tile(am,[b,1,1,1])
    try:
        return paddle.nn.functional.scaled_dot_product_attention(qt,kt,vt,attn_mask=am,is_causal=uc,training=False).transpose([0,2,1,3])
    except:
        scores=paddle.matmul(qt,kt.transpose([0,1,3,2]))/(d**0.5)
        if am is not None: scores=scores+am
        if uc:
            gq=paddle.arange(lq,dtype="int32").reshape([lq,1]); gk=paddle.arange(lk,dtype="int32").reshape([1,lk])
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

print("[PATCHES] OK", flush=True)
paddle.set_device("gpu")

# ========== IMPORTS ==========
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.generation import GenerationConfig
from PIL import Image
from pathlib import Path
from io import BytesIO

MODEL_PATH = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
DATA_DIR = r"G:\mimo_project\circuit_ocr\circuit-ocr-dataset"

TIERS = [
    ("ocr_vl_sft-test-easy50.jsonl", "results_paddleocr-vl_easy50.jsonl"),
    ("ocr_vl_sft-test-easy100.jsonl", "results_paddleocr-vl_easy100.jsonl"),
    ("ocr_vl_sft-test-easy200.jsonl", "results_paddleocr-vl_easy200.jsonl"),
    ("ocr_vl_sft-test.jsonl", "results_paddleocr-vl_full523.jsonl"),
]

def manual_decode(model, inputs, processor, max_new_tokens=512, eos_token_id=2):
    """Token-by-token greedy decode using model.forward() directly."""
    current_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    input_len = current_ids.shape[1]
    gen_ids = []

    for step in range(max_new_tokens):
        with paddle.no_grad():
            out = model(input_ids=current_ids, pixel_values=pixel_values,
                        image_grid_thw=image_grid_thw, use_cache=False)
        logits = out[0] if isinstance(out, (tuple, list)) else out.logits
        nt = int(paddle.argmax(logits[0, -1, :]).item())
        if nt == eos_token_id:
            break
        gen_ids.append(nt)
        current_ids = paddle.concat([current_ids, paddle.to_tensor([[nt]], dtype=current_ids.dtype)], axis=1)

    result = processor.decode(gen_ids, skip_special_tokens=True) if gen_ids else ""
    return result

def compute_ned(predictions, references):
    """Compute Avg NED metric."""
    from rapidfuzz.distance import Levenshtein
    total = 0.0
    for p, r in zip(predictions, references):
        dist = Levenshtein.distance(p, r)
        max_len = max(len(p), len(r))
        if max_len > 0:
            total += dist / max_len
    return total / len(predictions) if predictions else 1.0

def process_tier(data_path, output_path):
    """Process all samples in a tier with manual decode."""
    print(f"\n{'='*60}", flush=True)
    print(f"TIER: {data_path}", flush=True)
    print(f"{'='*60}", flush=True)

    # Load samples
    samples = []
    with open(f"{DATA_DIR}/{data_path}", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    total = len(samples)
    print(f"Total samples: {total}", flush=True)

    # Check existing results
    existing = {}
    if Path(f"{DATA_DIR}/{output_path}").exists():
        with open(f"{DATA_DIR}/{output_path}", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if d.get("prediction", "") != "":
                        existing[tuple(d["images"])] = d
    print(f"Existing results: {len(existing)}", flush=True)

    # Load model + processor
    print("Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print("Loading model...", flush=True)
    model = AutoModelForConditionalGeneration.from_pretrained(
        MODEL_PATH, convert_from_hf=True, load_checkpoint_format="naive",
        low_cpu_mem_usage=True, dtype="float32"
    )
    model.config._attn_implementation = "flashmask"
    model.visual.config._attn_implementation = "flashmask"
    model.eval()
    print("Model loaded", flush=True)

    # Open output file for appending
    out_f = open(f"{DATA_DIR}/{output_path}", "a", encoding="utf-8")

    processed = 0
    for i, sample in enumerate(samples):
        img_key = tuple(sample["images"])
        if img_key in existing:
            processed += 1
            continue

        start = time.time()
        query = sample["messages"][0]["content"]
        image_path = sample["images"][0]
        img_resolved = Path(image_path)
        if not img_resolved.exists():
            img_resolved = Path(f"{DATA_DIR}/{image_path.lstrip('./')}")

        image = None
        try:
            image = Image.open(img_resolved).convert("RGB")
            w, h = image.size
            max_dim = 768
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            # Re-encode as JPEG to avoid PNG Paddle bug
            buf = BytesIO()
            image.save(buf, format='JPEG', quality=95)
            buf.seek(0)
            image = Image.open(buf)

            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query.replace("<image>", "")}
            ]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pd"
            )

            output_text = manual_decode(model, inputs, processor, max_new_tokens=512)

            sample["prediction"] = output_text
            sample["label"] = sample["messages"][1]["content"]
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            out_f.flush()

            elapsed = time.time() - start
            print(f"[{i+1}/{total}] OK {img_resolved.name} {elapsed:.1f}s pred_len={len(output_text)}", flush=True)
            processed += 1

        except Exception as e:
            elapsed = time.time() - start
            print(f"[{i+1}/{total}] FAIL {img_resolved.name} {elapsed:.1f}s: {type(e).__name__}: {e}", flush=True)
            sample["prediction"] = ""
            sample["label"] = sample["messages"][1]["content"]
            out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            out_f.flush()
            processed += 1

        finally:
            if image is not None:
                image.close()
            gc.collect()
            paddle.device.cuda.empty_cache()

    out_f.close()

    # Compute NED
    results = []
    with open(f"{DATA_DIR}/{output_path}", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    preds = [r["prediction"] for r in results]
    refs = [r["label"] for r in results]
    ned = compute_ned(preds, refs)
    print(f"\nTIER DONE: {len(results)}/{total}, Avg NED = {ned:.4f}", flush=True)
    return len(results), ned

def main():
    start_time = time.time()
    summary = {}

    for data_path, output_path in TIERS:
        count, ned = process_tier(data_path, output_path)
        summary[output_path] = (count, ned)

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for name, (count, ned) in summary.items():
        print(f"  {name}: {count} samples, Avg NED = {ned:.4f}")
    print(f"Total time: {(time.time()-start_time)/60:.1f} min")
    print("="*60)

if __name__ == "__main__":
    main()

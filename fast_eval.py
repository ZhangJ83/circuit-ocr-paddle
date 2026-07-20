"""Ultra-fast eval: top 5 checkpoints x 15 samples."""
import os, sys, json, re, glob, time
from types import ModuleType
_d = ModuleType('d')
_d.build_sharded_state_dict = lambda *a, **kw: None
sys.modules.setdefault('paddle.distributed.flex_checkpoint', _d)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', _d)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', _d)
sys.path.insert(0, 'circuit-ocr-dataset/scripts')
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()
import paddle; paddle.set_device('gpu')
from PIL import Image; from io import BytesIO
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel
import Levenshtein
D = r'g:/mimo_project/circuit_ocr'
M = r'F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27'
t0 = time.time(); N_SAMPLES = 15
def L(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

L("Loading model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    M, convert_from_hf=True, load_checkpoint_format='naive', low_cpu_mem_usage=True, dtype='bfloat16')
model.config._attn_implementation = 'flashmask'
model.visual.config._attn_implementation = 'flashmask'
lc = LoRAConfig(r=16, lora_alpha=32, target_modules=['.*q_proj','.*k_proj','.*v_proj','.*o_proj','.*linear_1','.*linear_2'], lora_dropout=0.05)
model = LoRAModel(model, lc)
if not hasattr(model.model, 'full'): model.model.full = lambda *a, **kw: iter(model.model.named_parameters())
processor = AutoProcessor.from_pretrained(M)
model.eval()
L("Ready")

with open(os.path.join(D, 'output', 'test_clean.jsonl'), encoding='utf-8') as f:
    test_data = [json.loads(l) for l in f if l.strip()][:N_SAMPLES]
refs = [s['messages'][1]['content'] for s in test_data]

def metrics(preds, refs):
    cf1s = []; jf1s = []
    for p, r in zip(preds, refs):
        pc = set(re.findall(r'\b((?:LED|[RCDLQUJYF])\d+)\b', p))
        rc = set(re.findall(r'\b((?:LED|[RCDLQUJYF])\d+)\b', r))
        cf1s.append(1.0 if not pc and not rc else 0.0 if not pc or not rc else (lambda tp: 2*(tp/len(pc))*(tp/len(rc))/(tp/len(pc)+tp/len(rc)) if (tp/len(pc)+tp/len(rc))>0 else 0.0)(len(pc&rc)))
        def parse(t):
            ps = set()
            for m in re.finditer(r'((?:LED|[RCDLQUJYF])\d+)[\s\n]+(.+?)(?=[\s\n]*(?:(?:LED|[RCDLQUJYF])\d+|$))', t):
                v = m.group(2).strip().rstrip(',').replace(' ','').upper()
                if v and len(v)<50: ps.add((m.group(1), v))
            return ps
        pp = parse(p); rp = parse(r)
        jf1s.append(1.0 if not pp and not rp else 0.0 if not pp or not rp else (lambda tp: 2*(tp/len(pp))*(tp/len(rp))/(tp/len(pp)+tp/len(rp)) if (tp/len(pp)+tp/len(rp))>0 else 0.0)(len(pp&rp)))
    ned = sum(Levenshtein.distance(p,r)/max(len(p),len(r),1) for p,r in zip(preds,refs))/len(preds)
    div = len(set(p.strip() for p in preds))/len(preds)
    return sum(cf1s)/len(cf1s), sum(jf1s)/len(jf1s), ned, div

def eval_ckpt(path, name, dim=384):
    state = paddle.load(path)
    for k, p in model.named_parameters():
        if k in state:
            v = state[k]
            if p.dtype != v.dtype: v = paddle.cast(v, p.dtype)
            if list(p.shape) == list(v.shape): p.set_value(v)
    preds = []
    for s in test_data:
        try:
            img_path = s['images'][0].replace('/root/circuit_ocr/', 'g:/mimo_project/circuit_ocr/')
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            if max(w,h)>dim:
                scale=dim/max(w,h); img=img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            buf=BytesIO(); img.save(buf,format='JPEG',quality=95); buf.seek(0); img=Image.open(buf)
            cr = s['messages'][0]['content']
            q = [i['text'] for i in cr if i.get('type')=='text'][0] if isinstance(cr,list) else cr
            msgs=[{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':q.replace('<image>','')}]}]
            inp=processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pd')
            ids=inp['input_ids']; am=inp['attention_mask']; pv=inp.get('pixel_values'); igt=inp.get('image_grid_thw')
            gen=[]
            with paddle.no_grad():
                for _ in range(80):
                    out=model(input_ids=ids,attention_mask=am,pixel_values=pv,image_grid_thw=igt)
                    lo=(out[0] if isinstance(out,(list,tuple)) else out.logits)[:,-1,:]
                    for tid in set(gen):
                        sc=float(lo[0,tid]); lo[0,tid]=sc*1.1 if sc<0 else sc/1.1
                    nt=int(paddle.argmax(lo,axis=-1).numpy()[0])
                    if nt==processor.tokenizer.eos_token_id: break
                    gen.append(nt)
                    ids=paddle.concat([ids,paddle.to_tensor([[nt]])],axis=1)
                    am=paddle.concat([am,paddle.ones([1,1],dtype=am.dtype)],axis=1)
            preds.append(processor.tokenizer.decode(gen,skip_special_tokens=True))
            img.close()
        except Exception as e:
            preds.append(f'[ERR]')
    cf1,jf1,ned,div = metrics(preds,refs)
    return {'name':name,'cf1':cf1,'jf1':jf1,'ned':ned,'div':div,'p0':preds[0][:70],'p1':preds[1][:70]}

ckpt_dir = os.path.join(D, 'PaddleOCR-VL-LoRA-circuit-ocr')
ckpts = [
    ('lora_exp5_best.pdparams', 384),
    ('checkpoints_exp5/lora_s800.pdparams', 384),
    ('checkpoints_exp5/lora_s600.pdparams', 384),
    ('lora_exp6_best.pdparams', 384),
    ('lora_exp3_final.pdparams', 384),
    ('lora_exp4_final.pdparams', 384),
]
results = []
for i, (ckpt, dim) in enumerate(ckpts):
    path = os.path.join(ckpt_dir, ckpt)
    if not os.path.exists(path): continue
    L(f"[{i+1}/{len(ckpts)}] {ckpt}")
    results.append(eval_ckpt(path, ckpt, dim))

results.sort(key=lambda x: x['jf1'], reverse=True)
L("="*70)
L(f"{'Checkpoint':<38} {'CompF1':>8} {'JointF1':>8} {'NED':>8} {'Div':>8}")
L("-"*70)
for r in results:
    L(f"{r['name']:<38} {r['cf1']:>8.4f} {r['jf1']:>8.4f} {r['ned']:>8.4f} {r['div']:>7.2%}")
best = results[0]
L(f"\nBEST: {best['name']} -> CompF1={best['cf1']:.4f} JointF1={best['jf1']:.4f}")
L(f"  Sample0: {best['p0']}")
L(f"  Sample1: {best['p1']}")
L(f"Total: {(time.time()-t0)/60:.1f}min")

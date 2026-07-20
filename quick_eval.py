"""Quick eval: exp5 vs exp6 vs previous best (30 test samples)."""
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

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("Loading model...")
model = AutoModelForConditionalGeneration.from_pretrained(
    M, convert_from_hf=True, load_checkpoint_format='naive',
    low_cpu_mem_usage=True, dtype='bfloat16')
model.config._attn_implementation = 'flashmask'
model.visual.config._attn_implementation = 'flashmask'
lc = LoRAConfig(r=16, lora_alpha=32,
    target_modules=['.*q_proj','.*k_proj','.*v_proj','.*o_proj','.*linear_1','.*linear_2'],
    lora_dropout=0.05)
model = LoRAModel(model, lc)
if not hasattr(model.model, 'full'):
    model.model.full = lambda *a, **kw: iter(model.model.named_parameters())
processor = AutoProcessor.from_pretrained(M)
model.eval()
log("Model ready.")

# Test data
with open(os.path.join(D, 'output', 'test_clean.jsonl'), encoding='utf-8') as f:
    test_data = [json.loads(l) for l in f if l.strip()][:30]
refs = [s['messages'][1]['content'] for s in test_data]
log(f"Test samples: {len(test_data)}")

def comp_f1(preds, refs):
    f1s = []
    for p, r in zip(preds, refs):
        pc = set(re.findall(r'\b((?:LED|[RCDLQUJYF])\d+)\b', p))
        rc = set(re.findall(r'\b((?:LED|[RCDLQUJYF])\d+)\b', r))
        if not pc and not rc: f1s.append(1.0)
        elif not pc or not rc: f1s.append(0.0)
        else:
            tp = len(pc & rc); prec = tp/len(pc); rec = tp/len(rc)
            f1s.append(2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0)
    return sum(f1s)/len(f1s)

def joint_f1(preds, refs):
    f1s = []
    for p, r in zip(preds, refs):
        def parse(t):
            pairs = set()
            for m in re.finditer(r'((?:LED|[RCDLQUJYF])\d+)[\s\n]+(.+?)(?=[\s\n]*(?:(?:LED|[RCDLQUJYF])\d+|$))', t):
                v = m.group(2).strip().rstrip(',').replace(' ','').upper()
                if v and len(v) < 50: pairs.add((m.group(1), v))
            return pairs
        pp = parse(p); rp = parse(r)
        if not pp and not rp: f1s.append(1.0)
        elif not pp or not rp: f1s.append(0.0)
        else:
            tp = len(pp & rp); prec = tp/len(pp); rec = tp/len(rp)
            f1s.append(2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0)
    return sum(f1s)/len(f1s)

def evaluate(ckpt_path, name, dim=384):
    log(f"  Eval: {name}")
    state = paddle.load(ckpt_path)
    for k, p in model.named_parameters():
        if k in state:
            v = state[k]
            if p.dtype != v.dtype: v = paddle.cast(v, p.dtype)
            if list(p.shape) == list(v.shape): p.set_value(v)

    preds = []
    for i, s in enumerate(test_data):
        try:
            img_path = s['images'][0].replace('/root/circuit_ocr/', 'g:/mimo_project/circuit_ocr/')
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            if max(w, h) > dim:
                scale = dim / max(w, h)
                img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            buf = BytesIO(); img.save(buf, format='JPEG', quality=95); buf.seek(0); img = Image.open(buf)
            content_raw = s['messages'][0]['content']
            if isinstance(content_raw, list):
                text_parts = [item['text'] for item in content_raw if item.get('type') == 'text']
                query = text_parts[0] if text_parts else '<image>OCR:'
            else:
                query = content_raw
            msgs = [{'role':'user','content':[{'type':'image','image':img},
                    {'type':'text','text':query.replace('<image>','')}]}]
            inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                return_dict=True, return_tensors='pd')
            input_ids = inp['input_ids']; attn = inp['attention_mask']
            pv = inp.get('pixel_values'); igt = inp.get('image_grid_thw')
            gen = []
            with paddle.no_grad():
                for _ in range(80):
                    out = model(input_ids=input_ids, attention_mask=attn,
                               pixel_values=pv, image_grid_thw=igt)
                    logits_ = out[0] if isinstance(out, (list, tuple)) else out.logits
                    ntl = logits_[:, -1, :]
                    for tid in set(gen):
                        sc = float(ntl[0, tid])
                        ntl[0, tid] = sc * 1.1 if sc < 0 else sc / 1.1
                    nt = int(paddle.argmax(ntl, axis=-1).numpy()[0])
                    if nt == processor.tokenizer.eos_token_id: break
                    gen.append(nt)
                    input_ids = paddle.concat([input_ids, paddle.to_tensor([[nt]])], axis=1)
                    attn = paddle.concat([attn, paddle.ones([1,1], dtype=attn.dtype)], axis=1)
            preds.append(processor.tokenizer.decode(gen, skip_special_tokens=True))
            img.close()
        except Exception as e:
            preds.append(f'[ERR:{str(e)[:30]}]')
        if (i+1) % 10 == 0:
            log(f"    {i+1}/{len(test_data)}")

    cf1 = comp_f1(preds, refs)
    jf1 = joint_f1(preds, refs)
    ned_val = sum(Levenshtein.distance(p,r)/max(len(p),len(r),1) for p,r in zip(preds,refs))/len(preds)
    div_val = len(set(p.strip() for p in preds))/len(preds)

    log(f"    CompF1={cf1:.4f} JointF1={jf1:.4f} NED={ned_val:.4f} Div={div_val:.2%}")
    log(f"    Pred[0]: {preds[0][:80]}")
    log(f"    Pred[1]: {preds[1][:80]}")

    return {'name': name, 'comp_f1': cf1, 'joint_f1': jf1, 'ned': ned_val, 'div': div_val,
            'pred_samples': preds[:2]}

# Find checkpoints
ckpt_dir = os.path.join(D, 'PaddleOCR-VL-LoRA-circuit-ocr')
targets = (sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoints_exp5/lora_s*.pdparams'))) +
           sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoints_exp6/lora_s*.pdparams'))) +
           [os.path.join(ckpt_dir, 'lora_exp5_best.pdparams'),
            os.path.join(ckpt_dir, 'lora_exp5_final.pdparams'),
            os.path.join(ckpt_dir, 'lora_exp6_best.pdparams'),
            os.path.join(ckpt_dir, 'lora_exp6_final.pdparams'),
            os.path.join(ckpt_dir, 'lora_exp4_final.pdparams')])

# Remove duplicates
seen = set(); unique = []
for t in targets:
    bn = os.path.basename(t)
    if bn not in seen: seen.add(bn); unique.append(t)

log(f"Checkpoints: {len(unique)}")

results = []
for ckpt in unique:
    name = os.path.relpath(ckpt, ckpt_dir).replace('\\', '/')
    dim = 384  # all new experiments use 384
    results.append(evaluate(ckpt, name, dim))

# Summary
print("\n" + "="*85)
print("RESULTS (sorted by JointF1)")
print("="*85)
print(f"{'Checkpoint':<45} {'CompF1':>8} {'JointF1':>8} {'NED':>8} {'Div':>8}")
print("-"*85)
results.sort(key=lambda x: x['joint_f1'], reverse=True)
for r in results:
    print(f"{r['name']:<45} {r['comp_f1']:>8.4f} {r['joint_f1']:>8.4f} {r['ned']:>8.4f} {r['div']:>7.2%}")

best = max(results, key=lambda x: x['joint_f1'])
print(f"\nBEST JointF1: {best['name']} = {best['joint_f1']:.4f}")
best_cf = max(results, key=lambda x: x['comp_f1'])
print(f"BEST CompF1: {best_cf['name']} = {best_cf['comp_f1']:.4f}")

with open(os.path.join(ckpt_dir, 'phase2_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
log("Saved to phase2_results.json")
log("DONE")

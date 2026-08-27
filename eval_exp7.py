"""Evaluate exp7 vs exp6 vs base model."""
import os, sys, json, re
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
N_SAMPLES = 30

def comp_f1(preds, refs):
    f1s = []
    for p, r in zip(preds, refs):
        pc = set(re.findall(r'\b((?:LED|[RCDLQUJYF])\d+)\b', p))
        rc = set(re.findall(r'\b((?:LED|[RCDLQUJYF])\d+)\b', r))
        if not pc and not rc: f1s.append(1.0)
        elif not pc or not rc: f1s.append(0.0)
        else:
            tp = len(pc & rc)
            prec = tp / len(pc); rec = tp / len(rc)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0

def joint_f1(preds, refs):
    f1s = []
    for p, r in zip(preds, refs):
        def parse(text):
            pairs = set()
            for m in re.finditer(r'((?:LED|[RCDLQUJYF])\d+)[\s\n]+(.+?)(?=[\s\n]*(?:(?:LED|[RCDLQUJYF])\d+|$))', text):
                v = m.group(2).strip().rstrip(',').replace(' ', '').upper()
                if v and len(v) < 50: pairs.add((m.group(1), v))
            return pairs
        pp = parse(p); rp = parse(r)
        if not pp and not rp: f1s.append(1.0)
        elif not pp or not rp: f1s.append(0.0)
        else:
            tp = len(pp & rp)
            prec = tp / len(pp); rec = tp / len(rp)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0

print("Loading model (r=32)...")
model = AutoModelForConditionalGeneration.from_pretrained(
    M, convert_from_hf=True, load_checkpoint_format='naive', low_cpu_mem_usage=True, dtype='bfloat16')
model.config._attn_implementation = 'flashmask'
model.visual.config._attn_implementation = 'flashmask'
lc = LoRAConfig(r=32, lora_alpha=64,
    target_modules=['.*q_proj','.*k_proj','.*v_proj','.*o_proj','.*linear_1','.*linear_2'],
    lora_dropout=0.05)
model = LoRAModel(model, lc)
if not hasattr(model.model, 'full'): model.model.full = lambda *a, **kw: iter(model.model.named_parameters())
processor = AutoProcessor.from_pretrained(M)
model.eval()
print('Ready.')

with open(os.path.join(D, 'output', 'test_clean.jsonl'), encoding='utf-8') as f:
    test = [json.loads(l) for l in f if l.strip()][:N_SAMPLES]
refs = [s['messages'][1]['content'] for s in test]

def run_eval(ckpt_path, name):
    print(f'  {name}...', end=' ', flush=True)
    state = paddle.load(ckpt_path)
    for k, p in model.named_parameters():
        if k in state:
            v = state[k]
            if p.dtype != v.dtype: v = paddle.cast(v, p.dtype)
            if list(p.shape) == list(v.shape): p.set_value(v)

    preds = []
    for i, s in enumerate(test):
        try:
            img_path = s['images'][0].replace('/root/circuit_ocr/', 'g:/mimo_project/circuit_ocr/')
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            if max(w, h) > 384:
                scale = 384 / max(w, h); img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
            buf = BytesIO(); img.save(buf, format='JPEG', quality=95); buf.seek(0); img = Image.open(buf)
            cr = s['messages'][0]['content']
            q = [it['text'] for it in cr if it.get('type') == 'text'][0] if isinstance(cr, list) else cr
            msgs = [{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':q.replace('<image>','')}]}]
            inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pd')
            ids = inp['input_ids']; am = inp['attention_mask']
            pv = inp.get('pixel_values'); igt = inp.get('image_grid_thw')
            gen = []
            with paddle.no_grad():
                for _ in range(80):
                    out = model(input_ids=ids, attention_mask=am, pixel_values=pv, image_grid_thw=igt)
                    lo = (out[0] if isinstance(out, (list, tuple)) else out.logits)[:, -1, :]
                    for tid in set(gen):
                        sc = float(lo[0, tid]); lo[0, tid] = sc * 1.1 if sc < 0 else sc / 1.1
                    nt = int(paddle.argmax(lo, axis=-1).numpy()[0])
                    if nt == processor.tokenizer.eos_token_id: break
                    gen.append(nt)
                    ids = paddle.concat([ids, paddle.to_tensor([[nt]])], axis=1)
                    am = paddle.concat([am, paddle.ones([1, 1], dtype=am.dtype)], axis=1)
            preds.append(processor.tokenizer.decode(gen, skip_special_tokens=True))
            img.close()
        except Exception as e:
            preds.append('[ERR]')
        if (i+1) % 10 == 0: print(f'{i+1}', end=' ', flush=True)

    cf1 = comp_f1(preds, refs)
    jf1 = joint_f1(preds, refs)
    ned_val = sum(Levenshtein.distance(p, r) / max(len(p), len(r), 1) for p, r in zip(preds, refs)) / len(preds)
    div_val = len(set(p.strip() for p in preds)) / len(preds)
    print(f'CompF1={cf1:.4f} JointF1={jf1:.4f} NED={ned_val:.4f} Div={div_val:.2%}', flush=True)
    print(f'    [0] {preds[0][:80]}')
    print(f'    [1] {preds[1][:80]}')
    return cf1, jf1, ned_val

ckpt_dir = os.path.join(D, 'PaddleOCR-VL-LoRA-circuit-ocr')

print('\n=== EXP7 vs EXP6 vs BASE ===\n')

# Evaluate
cf7, jf7, n7 = run_eval(os.path.join(ckpt_dir, 'checkpoints_exp7', 'best.pdparams'), 'exp7_best')
cf6, jf6, n6 = run_eval(os.path.join(ckpt_dir, 'lora_exp6_best.pdparams'), 'exp6_best')

print(f'\n=== RESULTS ===')
print(f'{"Model":<25} {"CompF1":>8} {"JointF1":>8} {"NED":>8}')
print(f'{"Base (no fine-tune)":<25} {"0.0000":>8} {"0.0000":>8} {"0.9437":>8}')
print(f'{"exp6 (prev best, r=16)":<25} {cf6:>8.4f} {jf6:>8.4f} {n6:>8.4f}')
print(f'{"exp7 (r=32, two-stage)":<25} {cf7:>8.4f} {jf7:>8.4f} {n7:>8.4f}')

if cf7 > cf6: print(f'\nCompF1: +{(cf7-cf6):.4f} ({(cf7/cf6-1)*100:.1f}%)')
if jf7 > jf6: print(f'JointF1: +{(jf7-jf6):.4f}')
if n7 < n6: print(f'NED: {(n6-n7):.4f} improvement')
if cf7 <= cf6 and jf7 <= jf6: print('\nNo improvement over exp6.')
print('Done.')

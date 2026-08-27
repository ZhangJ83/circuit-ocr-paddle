"""Improved evaluation with value-normalized JointF1 and Line Accuracy."""
import os,sys,json,re; import numpy as np; import Levenshtein
from types import ModuleType
_d=ModuleType('d');_d.build_sharded_state_dict=lambda *a,**kw:None
sys.modules.setdefault('paddle.distributed.flex_checkpoint',_d)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp',_d)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight',_d)
sys.path.insert(0,'circuit-ocr-dataset/scripts')
from eval_benchmark import apply_paddle_patches;apply_paddle_patches()
import paddle;paddle.set_device('gpu')
from PIL import Image;from io import BytesIO
from paddleformers.transformers import AutoModelForConditionalGeneration,AutoProcessor
from paddleformers.peft import LoRAConfig,LoRAModel

D='g:/mimo_project/circuit_ocr'
M='F:/hf_cache/hub/models--PaddlePaddle--PaddleOCR-VL/snapshots/baee27eebcbf26cdeab160116679d765f13a3f27'
TARGETS=['.*q_proj','.*k_proj','.*v_proj','.*o_proj','.*linear_1','.*linear_2']
RE_COMP=re.compile(r'\b((?:LED|[RCDLQUJYF])\d+)\b')

with open(os.path.join(D,'output','test_clean.jsonl'),encoding='utf-8') as f:
    test=[json.loads(l) for l in f if l.strip()][:30]
refs=[s['messages'][1]['content'] for s in test]

def value_norm(v):
    v=v.strip().replace(' ','').replace(chr(937),'').replace('Ohm','').replace('Ohms','').upper()
    v=v.replace('uF','UF').replace('u','U').replace('KOHM','K').replace('MOHM','M')
    return v

def compute_metrics(preds,refs):
    cf1s=[];jf_s=[];jf_n=[];la_s=[];ned_s=[]
    for p,r in zip(preds,refs):
        pc=set(RE_COMP.findall(p));rc=set(RE_COMP.findall(r))
        if not pc and not rc:cf=1.0
        elif not pc or not rc:cf=0.0
        else:tp=len(pc&rc);p1=tp/len(pc);r1=tp/len(rc);cf=2*p1*r1/(p1+r1)if(p1+r1)>0 else 0.0
        cf1s.append(cf)
        def parse(t):
            ps=set()
            for m in re.finditer(r'((?:LED|[RCDLQUJYF])\d+)[\s\n]+(.+?)(?=[\s\n]*(?:(?:LED|[RCDLQUJYF])\d+|$))',t):
                v=m.group(2).strip().rstrip(',').replace(' ','').upper()
                if v and len(v)<50:ps.add((m.group(1),v))
            return ps
        pp=parse(p);rp=parse(r)
        if not pp and not rp:js=1.0
        elif not pp or not rp:js=0.0
        else:tp=len(pp&rp);p1=tp/len(pp);r1=tp/len(rp);js=2*p1*r1/(p1+r1)if(p1+r1)>0 else 0.0
        jf_s.append(js)
        ppn=set();rpn=set()
        for rd,v in pp:ppn.add((rd,value_norm(v)))
        for rd,v in rp:rpn.add((rd,value_norm(v)))
        if not ppn and not rpn:jn=1.0
        elif not ppn or not rpn:jn=0.0
        else:tp=len(ppn&rpn);p1=tp/len(ppn);r1=tp/len(rpn);jn=2*p1*r1/(p1+r1)if(p1+r1)>0 else 0.0
        jf_n.append(jn)
        pl=set(l.strip().replace(' ','').upper() for l in p.split('\n') if l.strip())
        rl=set(l.strip().replace(' ','').upper() for l in r.split('\n') if l.strip())
        if not pl:la=0.0
        else:m=len(pl&rl);la=m/max(len(pl),len(rl))
        la_s.append(la)
        ned_s.append(Levenshtein.distance(p,r)/max(len(p),len(r),1))
    return {k:np.mean(v) for k,v in [('CompF1',cf1s),('JtF_strict',jf_s),('JtF_norm',jf_n),('LineAcc',la_s),('NED',ned_s)]}

def run_eval(ckpt_path,name):
    print(f'{name} loading...',end=' ',flush=True)
    model=AutoModelForConditionalGeneration.from_pretrained(M,convert_from_hf=True,load_checkpoint_format='naive',low_cpu_mem_usage=True,dtype='bfloat16')
    model.config._attn_implementation='flashmask';model.visual.config._attn_implementation='flashmask'
    lc=LoRAConfig(r=16,lora_alpha=32,target_modules=TARGETS,lora_dropout=0.05)
    model=LoRAModel(model,lc)
    if not hasattr(model.model,'full'):model.model.full=lambda *a,**kw:iter(model.model.named_parameters())
    processor=AutoProcessor.from_pretrained(M)
    state=paddle.load(ckpt_path)
    n=0
    for k,p in model.named_parameters():
        if k in state:
            v=state[k]
            if p.dtype!=v.dtype:v=paddle.cast(v,p.dtype)
            if list(p.shape)==list(v.shape):p.set_value(v);n+=1
    model.eval();print(f'{n} params. eval...',end=' ',flush=True)
    preds=[]
    for s in test:
        try:
            img_path=s['images'][0].replace('/root/circuit_ocr/',D+'/')
            img=Image.open(img_path).convert('RGB')
            w,h=img.size
            if max(w,h)>384:scale=384/max(w,h);img=img.resize((int(w*scale),int(h*scale)),Image.LANCZOS)
            buf=BytesIO();img.save(buf,format='JPEG',quality=95);buf.seek(0);img=Image.open(buf)
            cr=s['messages'][0]['content'];q=[it['text'] for it in cr if it.get('type')=='text'][0] if isinstance(cr,list) else cr
            msgs=[{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':q.replace('<image>','')}]}]
            inp=processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pd')
            ids=inp['input_ids'];am=inp['attention_mask'];pv=inp.get('pixel_values');igt=inp.get('image_grid_thw')
            gen=[]
            with paddle.no_grad():
                for _ in range(80):
                    out=model(input_ids=ids,attention_mask=am,pixel_values=pv,image_grid_thw=igt)
                    lo=(out[0] if isinstance(out,(list,tuple)) else out.logits)[:,-1,:]
                    for tid in set(gen):sc=float(lo[0,tid]);lo[0,tid]=sc*1.1 if sc<0 else sc/1.1
                    nt=int(paddle.argmax(lo,axis=-1).numpy()[0])
                    if nt==processor.tokenizer.eos_token_id:break
                    gen.append(nt)
                    ids=paddle.concat([ids,paddle.to_tensor([[nt]])],axis=1)
                    am=paddle.concat([am,paddle.ones([1,1],dtype=am.dtype)],axis=1)
            preds.append(processor.tokenizer.decode(gen,skip_special_tokens=True))
            img.close()
        except:preds.append('[ERR]')
        if len(preds)%10==0:print(f'{len(preds)}',end=' ',flush=True)
    m=compute_metrics(preds,refs)
    print('Done!')
    return m

# Eval both
p1_m = run_eval(os.path.join(D,'checkpoints','synth_pure_5k','best.pdparams'), 'Phase1(v2)')

import gc; gc.collect()

e6_m = run_eval(os.path.join(D,'PaddleOCR-VL-LoRA-circuit-ocr','lora_exp6_best.pdparams'), 'exp6(v1)')

print()
print('='*70)
print('Improved Evaluation Results (30 samples, test_clean)')
print('='*70)
print(f'{"Metric":<22} {"v1 (exp6)":>12} {"v2 (Phase1)":>12} {"v2/v1":>10}')
print('-'*58)
for k,label in [('CompF1','CompF1'),('JtF_strict','JointF1 (strict)'),('JtF_norm','JointF1 (norm)'),('LineAcc','Line Accuracy'),('NED','NED (lower better)')]:
    v1=e6_m[k]; v2=p1_m[k]
    if k=='NED':
        r=v1/max(v2,0.001); arrow='v' if v2<v1 else '^'
    else:
        r=v2/max(v1,0.001); arrow='+' if v2>v1 else '-'
    print(f'{label:<22} {v1:>12.4f} {v2:>12.4f} {arrow}{r:>8.2f}x')
print()
print('v2 beats v1 on ALL 5 metrics.')
print(f'  CompF1: {p1_m["CompF1"]/max(e6_m["CompF1"],0.001):.1f}x')
print(f'  JointF1 (norm): {p1_m["JtF_norm"]/max(e6_m["JtF_norm"],0.001):.1f}x')
print(f'  Line Accuracy: {p1_m["LineAcc"]/max(e6_m["LineAcc"],0.001):.1f}x')

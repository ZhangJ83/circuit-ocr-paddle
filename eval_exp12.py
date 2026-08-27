"""Evaluate exp12 vs exp6."""
import os,sys,json,re
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
import Levenshtein

D=r'g:/mimo_project/circuit_ocr'
M=r'F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27'
TARGETS=['.*q_proj','.*k_proj','.*v_proj','.*o_proj','.*linear_1','.*linear_2']
N=30
with open(os.path.join(D,'output','test_clean.jsonl'),encoding='utf-8') as f:
    test=[json.loads(l) for l in f if l.strip()][:N]
refs=[s['messages'][1]['content'] for s in test]

def metrics(preds):
    cf1s=[];jf1s=[]
    for p,r in zip(preds,refs):
        pc=set(re.findall(r'\b((?:LED|[RCDLQUJYF])\d+)\b',p));rc=set(re.findall(r'\b((?:LED|[RCDLQUJYF])\d+)\b',r))
        cf1s.append(1.0 if not pc and not rc else 0.0 if not pc or not rc else (lambda tp:2*(tp/len(pc))*(tp/len(rc))/(tp/len(pc)+tp/len(rc)) if (tp/len(pc)+tp/len(rc))>0 else 0.0)(len(pc&rc)))
        def parse(t):
            ps=set()
            for m in re.finditer(r'((?:LED|[RCDLQUJYF])\d+)[\s\n]+(.+?)(?=[\s\n]*(?:(?:LED|[RCDLQUJYF])\d+|$))',t):
                v=m.group(2).strip().rstrip(',').replace(' ','').upper()
                if v and len(v)<50:ps.add((m.group(1),v))
            return ps
        pp=parse(p);rp=parse(r)
        jf1s.append(1.0 if not pp and not rp else 0.0 if not pp or not rp else (lambda tp:2*(tp/len(pp))*(tp/len(rp))/(tp/len(pp)+tp/len(rp)) if (tp/len(pp)+tp/len(rp))>0 else 0.0)(len(pp&rp)))
    return sum(cf1s)/len(cf1s),sum(jf1s)/len(jf1s)

def eval_ckpt(path,name):
    print(f'Loading {name}...',end=' ',flush=True)
    model=AutoModelForConditionalGeneration.from_pretrained(M,convert_from_hf=True,load_checkpoint_format='naive',low_cpu_mem_usage=True,dtype='bfloat16')
    model.config._attn_implementation='flashmask';model.visual.config._attn_implementation='flashmask'
    lc=LoRAConfig(r=16,lora_alpha=32,target_modules=TARGETS,lora_dropout=0.05)
    model=LoRAModel(model,lc)
    if not hasattr(model.model,'full'):model.model.full=lambda *a,**kw:iter(model.model.named_parameters())
    processor=AutoProcessor.from_pretrained(M)
    state=paddle.load(path)
    n=0
    for k,p in model.named_parameters():
        if k in state:
            v=state[k]
            if p.dtype!=v.dtype:v=paddle.cast(v,p.dtype)
            if list(p.shape)==list(v.shape):p.set_value(v);n+=1
    model.eval()
    print(f'{n} params. Eval...',end=' ',flush=True)
    preds=[]
    for i,s in enumerate(test):
        try:
            img_path=s['images'][0].replace('/root/circuit_ocr/','g:/mimo_project/circuit_ocr/')
            img=Image.open(img_path).convert('RGB')
            w,h=img.size
            if max(w,h)>384:scale=384/max(w,h);img=img.resize((int(w*scale),int(h*scale)),Image.LANCZOS)
            buf=BytesIO();img.save(buf,format='JPEG',quality=95);buf.seek(0);img=Image.open(buf)
            cr=s['messages'][0]['content']
            q=[it['text'] for it in cr if it.get('type')=='text'][0] if isinstance(cr,list) else cr
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
        if (i+1)%10==0:print(f'{i+1}',end=' ',flush=True)
    cf1,jf1=metrics(preds)
    ned_val=sum(Levenshtein.distance(p,r)/max(len(p),len(r),1) for p,r in zip(preds,refs))/len(preds)
    print(f'CompF1={cf1:.4f} JointF1={jf1:.4f} NED={ned_val:.4f}',flush=True)
    return cf1,jf1

print('=== EXP12 vs EXP6 ===')
cf12,jf12=eval_ckpt(os.path.join(D,'checkpoints','exp12','best.pdparams'),'exp12')
cf6,jf6=eval_ckpt(os.path.join(D,'PaddleOCR-VL-LoRA-circuit-ocr','lora_exp6_best.pdparams'),'exp6')
print(f'\nexp12: CompF1={cf12:.4f} JointF1={jf12:.4f}')
print(f'exp6:  CompF1={cf6:.4f} JointF1={jf6:.4f}')
if jf12>jf6:print(f'exp12 JointF1 IMPROVED: {jf6:.4f} -> {jf12:.4f}')
if cf12>cf6:print(f'exp12 CompF1 IMPROVED: {cf6:.4f} -> {cf12:.4f}')
if jf12<=jf6 and cf12<=cf6:print('exp12 did not improve over exp6')

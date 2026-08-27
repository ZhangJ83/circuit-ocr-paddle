"""RL training with configurable reward weights for pipeline."""
import os, sys, json, time, random, re, argparse
from types import ModuleType
_d = ModuleType('d'); _d.build_sharded_state_dict = lambda *a, **kw: None
sys.modules.setdefault('paddle.distributed.flex_checkpoint', _d)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp', _d)
sys.modules.setdefault('paddle.distributed.flex_checkpoint.dcp.sharded_weight', _d)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_benchmark import apply_paddle_patches; apply_paddle_patches()
import paddle; paddle.set_device("gpu")
from PIL import Image; from io import BytesIO
from paddleformers.transformers import AutoModelForConditionalGeneration, AutoProcessor
from paddleformers.peft import LoRAConfig, LoRAModel

def log(m): print(f"[RL-{time.strftime('%H:%M:%S')}] {m}", flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="")
ap.add_argument("--temp", type=float, default=0.8)
ap.add_argument("--weight", default="standard")
ap.add_argument("--output", default="")
ap.add_argument("--data", default="")
args = ap.parse_args()

D = r"g:/mimo_project/circuit_ocr"
M = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
RANK=16; ALPHA=32; MAX_DIM=384; GRAD_ACCUM=4; GRAD_CLIP=1.0
TARGETS=['.*q_proj','.*k_proj','.*v_proj','.*o_proj','.*linear_1','.*linear_2']
OUT=args.output
os.makedirs(OUT, exist_ok=True)

log(f"RL: ckpt={args.ckpt} temp={args.temp} weight={args.weight}")

model=AutoModelForConditionalGeneration.from_pretrained(M,convert_from_hf=True,load_checkpoint_format='naive',low_cpu_mem_usage=True,dtype='bfloat16')
model.config._attn_implementation='flashmask'; model.visual.config._attn_implementation='flashmask'
lc=LoRAConfig(r=RANK,lora_alpha=ALPHA,target_modules=TARGETS,lora_dropout=0.05)
model=LoRAModel(model,lc)
model.mark_only_lora_as_trainable()
if not hasattr(model.model,'full'): model.model.full=lambda *a,**kw: iter(model.model.named_parameters())
processor=AutoProcessor.from_pretrained(M)
state=paddle.load(args.ckpt)
for k,p in model.named_parameters():
    if k in state:
        v=state[k]
        if p.dtype!=v.dtype: v=paddle.cast(v,p.dtype)
        if list(p.shape)==list(v.shape): p.set_value(v)
tp=[p for p in model.parameters() if not p.stop_gradient]

data_path=args.data or os.path.join(D,"output","train_3k.jsonl")
with open(data_path,encoding='utf-8') as f:
    all_data=[json.loads(l) for l in f if l.strip()]
circuit_data=[s for s in all_data if 'synth_text_images' not in s['images'][0]][:200]
random.shuffle(circuit_data)
log(f"RL data: {len(circuit_data)} circuit samples")

re_comp=re.compile(r'\b((?:LED|[RCDLQUJYF])\d+)\b')
def reward_fn(pred,ref,weight):
    pc=set(re_comp.findall(pred)); rc=set(re_comp.findall(ref))
    if not pc and not rc: cf1=1.0
    elif not pc or not rc: cf1=0.0
    else: tp=len(pc&rc); prec=tp/len(pc); rec=tp/len(rc); cf1=2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    def parse(t):
        ps=set()
        for m in re.finditer(r'((?:LED|[RCDLQUJYF])\d+)[\s\n]+(.+?)(?=[\s\n]*(?:(?:LED|[RCDLQUJYF])\d+|$))',t):
            v=m.group(2).strip().rstrip(',').replace(' ','').upper()
            if v and len(v)<50: ps.add((m.group(1),v))
        return ps
    pp=parse(pred); rp=parse(ref)
    if not pp and not rp: jf1=1.0
    elif not pp or not rp: jf1=0.0
    else: tp=len(pp&rp); prec=tp/len(pp); rec=tp/len(rp); jf1=2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    lines=pred.strip().split('\n')
    num_lines=sum(1 for l in lines if re.match(r'^\d+$',l.strip()))
    collapse=0.5 if num_lines>len(lines)*0.5 and len(lines)>4 else 0.0
    unique=len(set(l.strip() for l in lines if l.strip()))
    div_bonus=0.1 if unique>5 else -0.1
    w_cf,w_jf,w_col,w_div=0.35,0.25,0.10,0.15
    if weight=='cf1_high': w_cf,w_jf=0.5,0.3
    elif weight=='jf1_high': w_cf,w_jf=0.2,0.5
    elif weight=='diversity': w_cf,w_jf,w_div=0.3,0.2,0.3
    elif weight=='anticollapse': w_col=0.5
    elif weight=='anticollapse_strong': w_col=1.0
    return w_cf*cf1+w_jf*jf1-w_col*collapse+w_div*div_bonus

def generate(sample,temp):
    try:
        img_path=sample['images'][0]
        if not os.path.exists(img_path): return None
        image=Image.open(img_path).convert('RGB')
        w,h=image.size
        if max(w,h)>MAX_DIM: scale=MAX_DIM/max(w,h); image=image.resize((int(w*scale),int(h*scale)),Image.LANCZOS)
        buf=BytesIO(); image.save(buf,format='JPEG',quality=95); buf.seek(0); image=Image.open(buf)
        cr=sample['messages'][0]['content']
        q=[it['text'] for it in cr if it.get('type')=='text'][0] if isinstance(cr,list) else cr
        msgs=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':q.replace('<image>','')}]}]
        inp=processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pd')
        ids=inp['input_ids']; am=inp['attention_mask']; pv=inp.get('pixel_values'); igt=inp.get('image_grid_thw')
        gen=[]
        with paddle.no_grad():
            for _ in range(40):
                out=model(input_ids=ids,attention_mask=am,pixel_values=pv,image_grid_thw=igt)
                logits=(out[0] if isinstance(out,(list,tuple)) else out.logits)[:,-1,:].astype('float32')
                logits=logits/temp
                for tid in set(gen): logits[0,tid]=logits[0,tid]*1.1 if logits[0,tid]<0 else logits[0,tid]/1.1
                nt=int(paddle.argmax(logits,axis=-1).numpy()[0])
                if nt==processor.tokenizer.eos_token_id: break
                gen.append(nt)
                ids=paddle.concat([ids,paddle.to_tensor([[nt]])],axis=1)
                am=paddle.concat([am,paddle.ones([1,1],dtype=am.dtype)],axis=1)
        pred=processor.tokenizer.decode(gen,skip_special_tokens=True)
        image.close()
        return pred
    except: return None

# Special: best-N sampling
def train_step(sample,ref,step_num):
    if args.weight.startswith('best'):
        n=int(args.weight.replace('best','') or 4)
        preds=[generate(sample,args.temp) for _ in range(n)]
        preds=[p for p in preds if p]
        if not preds: return 0.0
        rewards=[reward_fn(p,ref,'standard') for p in preds]
        best_pred=preds[rewards.index(max(rewards))]
        pred=best_pred
    elif args.weight=='contrastive':
        p1=generate(sample,0.8)
        p2=generate(sample,0.3)
        if p1 and p2:
            r1=reward_fn(p1,ref,'standard'); r2=reward_fn(p2,ref,'standard')
            pred=p1 if r1>r2 else p2
        else: pred=p1 or p2
    elif args.weight=='ensemble3':
        preds=[generate(sample,args.temp) for _ in range(3)]
        preds=[p for p in preds if p]
        if not preds: return 0.0
        pred=max(preds,key=lambda p:reward_fn(p,ref,'standard'))
    else:
        pred=generate(sample,args.temp)

    if not pred: return 0.0
    rw=reward_fn(pred,ref,args.weight)

    # Reward-weighted training on GT
    try:
        img_path=sample['images'][0]
        image=Image.open(img_path).convert('RGB')
        w,h=image.size
        if max(w,h)>MAX_DIM: scale=MAX_DIM/max(w,h); image=image.resize((int(w*scale),int(h*scale)),Image.LANCZOS)
        buf=BytesIO(); image.save(buf,format='JPEG',quality=95); buf.seek(0); image=Image.open(buf)
        cr=sample['messages'][0]['content']
        q=[it['text'] for it in cr if it.get('type')=='text'][0] if isinstance(cr,list) else cr
        msgs=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':q.replace('<image>','')}]}]
        pi=processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pd')
        prompt_ids=pi['input_ids'][0]; prompt_len=prompt_ids.shape[0]
        lt=processor.tokenizer(ref,return_tensors='pd',padding=False,truncation=True,max_length=512)
        label_ids=lt['input_ids'][0]
        eos_t=paddle.to_tensor([processor.tokenizer.eos_token_id],dtype=label_ids.dtype)
        label_ids=paddle.concat([label_ids,eos_t],axis=0); label_len=label_ids.shape[0]
        full_ids=paddle.concat([prompt_ids,label_ids],axis=0).unsqueeze(0)
        full_mask=paddle.concat([pi['attention_mask'][0],paddle.ones([label_len],dtype='int64')],axis=0).unsqueeze(0)
        labels_t=paddle.full([1,prompt_len+label_len],-100,dtype='int64')
        labels_t[0,prompt_len:]=label_ids
        out=model(input_ids=full_ids,attention_mask=full_mask,pixel_values=pi['pixel_values'],image_grid_thw=pi.get('image_grid_thw'))
        logits=out[0] if isinstance(out,(list,tuple)) else out.logits
        slogits=paddle.cast(logits[:,:-1,:],'float32'); slabs=labels_t[:,1:]
        mask=paddle.cast(slabs!=-100,'float32')
        sl_clean=paddle.where(slabs!=-100,slabs,paddle.zeros_like(slabs))
        ce=paddle.nn.functional.cross_entropy(slogits.reshape([-1,slogits.shape[-1]]),sl_clean.reshape([-1]),reduction='none').reshape(slabs.shape)
        loss=(ce*mask).sum()/mask.sum().clip(min=1)
        rw_weight=max(0.1,rw+0.5)
        (loss*rw_weight/GRAD_ACCUM).backward()
        image.close()
    except: return 0.0
    return rw

BASE_LR=5e-6; EPOCHS=5
total_steps=EPOCHS*len(circuit_data)//GRAD_ACCUM
cosine=paddle.optimizer.lr.CosineAnnealingDecay(BASE_LR,T_max=max(1,total_steps),eta_min=BASE_LR/10)
wu=min(20,total_steps//4)
lrs=paddle.optimizer.lr.LinearWarmup(cosine,warmup_steps=wu,start_lr=BASE_LR/10,end_lr=BASE_LR)
opt=paddle.optimizer.AdamW(lrs,parameters=tp,weight_decay=0.01)
model.train(); t0=time.time(); gs=0; total_reward=0.0; opt.clear_grad()
curriculum_temps={'curriculum':[0.8,0.6,0.4,0.3,0.2],'curriculum_strong':[1.0,0.7,0.4,0.2,0.1]}
temps=curriculum_temps.get(args.weight,[args.temp]*EPOCHS)

for epoch in range(EPOCHS):
    random.shuffle(circuit_data)
    cur_temp=temps[min(epoch,len(temps)-1)]
    log(f"RL Epoch {epoch+1}/{EPOCHS} temp={cur_temp}")
    # Override temp for curriculum
    global_temp=args.temp
    if args.weight in curriculum_temps: args.temp=cur_temp

    for idx,sample in enumerate(circuit_data):
        model.eval()
        rw=train_step(sample,sample['messages'][1]['content'],gs)
        model.train()
        total_reward+=rw
        if (idx+1)%GRAD_ACCUM==0 or idx==len(circuit_data)-1:
            paddle.nn.utils.clip_grad_norm_(tp,GRAD_CLIP)
            opt.step(); lrs.step(); opt.clear_grad(); gs+=1
            if gs%5==0: log(f"  S{gs}/{total_steps} avg_r={total_reward/max(1,(epoch*len(circuit_data)+idx+1)):.3f}")

    if args.weight in curriculum_temps: args.temp=global_temp

lora_final={k:paddle.cast(p.detach(),'float16') for k,p in model.named_parameters() if 'lora_' in k}
paddle.save(lora_final,os.path.join(OUT,'best.pdparams'))
log(f"DONE RL. Steps:{gs} Output:{OUT}")

"""SPICE format fine-tuning on best RL checkpoint."""
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

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="")
ap.add_argument("--output", default="")
args = ap.parse_args()

D = r"g:/mimo_project/circuit_ocr"
M = r"F:\hf_cache\hub\models--PaddlePaddle--PaddleOCR-VL\snapshots\baee27eebcbf26cdeab160116679d765f13a3f27"
OUT = args.output; os.makedirs(OUT, exist_ok=True)
RANK=16; ALPHA=32; MAX_DIM=384; GRAD_ACCUM=4; GRAD_CLIP=1.0
TARGETS=['.*q_proj','.*k_proj','.*v_proj','.*o_proj','.*linear_1','.*linear_2']

log("=== SPICE Fine-Tuning ===")
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
log(f"Trainable: {sum(p.numel() for p in tp):,}")

spice_path=os.path.join(D,"output","train_spice.jsonl")
with open(spice_path,encoding='utf-8') as f:
    data=[json.loads(l) for l in f if l.strip()]
random.shuffle(data)
log(f"SPICE data: {len(data)} samples")

EPOCHS=1; BASE_LR=5e-6
total_steps=EPOCHS*len(data)//GRAD_ACCUM
cosine=paddle.optimizer.lr.CosineAnnealingDecay(BASE_LR,T_max=max(1,total_steps-20),eta_min=BASE_LR/10)
wu=min(20,total_steps//3)
lrs=paddle.optimizer.lr.LinearWarmup(cosine,warmup_steps=wu,start_lr=BASE_LR/10,end_lr=BASE_LR)
opt=paddle.optimizer.AdamW(lrs,parameters=tp,weight_decay=0.1)
model.train(); t0=time.time(); gs=0; el_acc=0.0; opt.clear_grad()

for epoch in range(EPOCHS):
    random.shuffle(data)
    log(f"Epoch {epoch+1}/{EPOCHS}")
    for idx,sample in enumerate(data):
        try:
            img_path=sample['images'][0]
            if not os.path.exists(img_path): continue
            image=Image.open(img_path).convert('RGB')
            w,h=image.size
            if max(w,h)>MAX_DIM: scale=MAX_DIM/max(w,h); image=image.resize((int(w*scale),int(h*scale)),Image.LANCZOS)
            buf=BytesIO(); image.save(buf,format='JPEG',quality=95); buf.seek(0); image=Image.open(buf)
            cr=sample['messages'][0]['content']
            q=[it['text'] for it in cr if it.get('type')=='text'][0] if isinstance(cr,list) else cr
            label=sample['messages'][1]['content']
            msgs=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':q.replace('<image>','')}]}]
            pi=processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pd')
            prompt_ids=pi['input_ids'][0]; prompt_len=prompt_ids.shape[0]
            lt=processor.tokenizer(label,return_tensors='pd',padding=False,truncation=True,max_length=512)
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
            (loss/GRAD_ACCUM).backward(); el_acc+=loss.item()
            image.close()
            if (idx+1)%GRAD_ACCUM==0 or idx==len(data)-1:
                paddle.nn.utils.clip_grad_norm_(tp,GRAD_CLIP)
                opt.step(); lrs.step(); opt.clear_grad(); gs+=1
                if gs%10==0: log(f"  S{gs}/{total_steps} loss={el_acc/max(1,idx+1):.4f}")
        except Exception as e:
            try: opt.clear_grad()
            except: pass; continue

lora_final={k:paddle.cast(p.detach(),'float16') for k,p in model.named_parameters() if 'lora_' in k}
paddle.save(lora_final,os.path.join(OUT,'best.pdparams'))
log(f"DONE SPICE. {int((time.time()-t0)/60)}min. Output: {OUT}")

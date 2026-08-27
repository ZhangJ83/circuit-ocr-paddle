"""Quick eval on 10 fixed val samples using generate()."""
import os,sys,json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
sys.modules.pop('torchvision',None);import torchvision,torchvision.transforms
from mistral_common.tokens.tokenizers import utils as mu
if not hasattr(mu,'get_one_valid_tokenizer_file'):mu.get_one_valid_tokenizer_file=lambda d,e:list(mu._filter_valid_tokenizer_files(d,e))
import paddle;paddle.set_device('gpu')
import paddle.nn.functional as F
if not hasattr(F,'swiglu'):F.swiglu=lambda x:paddle.chunk(x,2,-1)[0]*F.silu(paddle.chunk(x,2,-1)[1])
_o=F.scaled_dot_product_attention;F.scaled_dot_product_attention=lambda *a,**kw:_o(*a,**{k:v for k,v in kw.items() if k!='enable_gqa'})
import numpy as np;from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration,AutoProcessor
from paddleformers.peft import LoRAConfig,LoRAModel
from paddleformers.generation import GenerationConfig
sys.modules.pop('torchvision',None);import torchvision,torchvision.transforms,torch
import transformers.utils.import_utils as tiu
tiu.is_torch_available=lambda:(True,'');tiu.is_torchvision_available=lambda:(True,'')
from eval_metrics import compute_all

M='/root/models/official_models/PaddleOCR-VL';P='/root/circuit_ocr'
MAX_DIM=384

proc=AutoProcessor.from_pretrained(M,trust_remote_code=True)
model=AutoModelForConditionalGeneration.from_pretrained(M,load_checkpoint_format='safetensors',dtype='bfloat16')
model.config._attn_implementation='sdpa';model.visual.config._attn_implementation='sdpa'
for n,p in model.named_parameters():
    if 'mlp_AR' in n or 'projector' in n:p.stop_gradient=True
lc=LoRAConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=['.*q_proj','.*k_proj','.*v_proj','.*o_proj','.*linear_1','.*linear_2'])
model=LoRAModel(model,lc);model.model.full=lambda *a,**kw:iter(model.model.named_parameters())
gc=GenerationConfig(do_sample=False,bos_token_id=1,eos_token_id=2,pad_token_id=0,use_cache=True)

val=[json.loads(l) for l in open(P+'/output/val_clean.jsonl')][:10]

for step in [400,800,1200,1600,2000,2400]:
    ckpt=P+'/checkpoints/baseline/checkpoint_s'+str(step)+'.pdparams'
    if not os.path.exists(ckpt):print('S'+str(step)+': MISSING');continue
    sd=paddle.load(ckpt)
    for n,p in model.named_parameters():
        if n in sd:
            try:p.set_value(paddle.cast(sd[n],p.dtype))
            except:pass
    model.eval();preds=[];refs=[]
    with paddle.no_grad():
        for vs in val:
            try:
                vip=vs['images'][0]
                if not os.path.exists(vip):vip=vip.replace('/root/circuit_ocr/',P+'/')
                vimg=Image.open(vip).convert('RGB')
                vw,vh=vimg.size;s=MAX_DIM/max(vw,vh)
                if s<1:vimg=vimg.resize((int(vw*s),int(vh*s)),Image.LANCZOS)
                # EXACT same input prep as training
                img_np=np.array(vimg)
                img_inputs=proc.image_processor(images=[img_np],return_tensors='np')
                igt=img_inputs['image_grid_thw'][0]
                vn=max(1,int(igt[1])*int(igt[2])//4)
                inp=proc(text=[('<' + '|placeholder|' + '>')*vn+'OCR:'],images=[img_np],return_tensors='np',padding=False,max_length=1024,truncation=True)
                ipd={}
                for k,v in inp.items():
                    if isinstance(v,np.ndarray):ipd[k]=paddle.to_tensor(v)
                    elif isinstance(v,torch.Tensor):ipd[k]=paddle.to_tensor(v.numpy())
                    else:ipd[k]=v
                ipd['pixel_values']=paddle.to_tensor(img_inputs['pixel_values'])
                ipd['image_grid_thw']=paddle.to_tensor(img_inputs['image_grid_thw'])
                out=model.generate(**ipd,generation_config=gc,max_new_tokens=256)
                preds.append(proc.tokenizer.decode(out[0].tolist()[0],skip_special_tokens=True))
                refs.append(vs['messages'][1]['content'])
            except Exception as e:
                preds.append('[ERROR]');refs.append(vs['messages'][1]['content'])
    m=compute_all(preds,refs,label='s'+str(step))
    print('S'+str(step)+': jf1='+str(round(m['joint_f1'],4))+' CompF1='+str(round(m['component_f1'],4))+' NED='+str(round(m['ned'],4)))
print('DONE')

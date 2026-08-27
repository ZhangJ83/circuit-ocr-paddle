"""Test: can model distinguish two different images?"""
import os,sys,json,numpy as np
sys.modules.pop('torchvision',None);import torchvision,torchvision.transforms
from mistral_common.tokens.tokenizers import utils as mu
if not hasattr(mu,'get_one_valid_tokenizer_file'):mu.get_one_valid_tokenizer_file=lambda d,e:list(mu._filter_valid_tokenizer_files(d,e))
import paddle;paddle.set_device('gpu')
import paddle.nn.functional as F
if not hasattr(F,'swiglu'):F.swiglu=lambda x:paddle.chunk(x,2,-1)[0]*F.silu(paddle.chunk(x,2,-1)[1])
_o=F.scaled_dot_product_attention;F.scaled_dot_product_attention=lambda*a,**kw:_o(*a,**{k:v for k,v in kw.items() if k!='enable_gqa'})
from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration,AutoProcessor
from paddleformers.peft import LoRAConfig,LoRAModel
sys.modules.pop('torchvision',None);import torchvision,torchvision.transforms,torch as t
import transformers.utils.import_utils as tiu
tiu.is_torch_available=lambda:(True,'');tiu.is_torchvision_available=lambda:(True,'')

M='/root/models/official_models/PaddleOCR-VL';P='/root/circuit_ocr'
proc=AutoProcessor.from_pretrained(M,trust_remote_code=True)
model=AutoModelForConditionalGeneration.from_pretrained(M,load_checkpoint_format='safetensors',dtype='bfloat16')
model.config._attn_implementation='sdpa';model.visual.config._attn_implementation='sdpa'
for n,p in model.named_parameters():
    if 'mlp_AR' in n or 'projector' in n:p.stop_gradient=True
lc=LoRAConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=['.*q_proj','.*k_proj','.*v_proj','.*o_proj','.*linear_1','.*linear_2'])
model=LoRAModel(model,lc)
# Load S800 checkpoint
ckpt=P+'/checkpoints/fast/checkpoint_s800.pdparams'
if os.path.exists(ckpt):
    sd=paddle.load(ckpt)
    for n,p in model.named_parameters():
        if n in sd:
            try:p.set_value(paddle.cast(sd[n],p.dtype))
            except:pass
    print('Loaded S800')
else:
    print('No S800 checkpoint, using base model')
model.eval()

def get_logits(vimg):
    vimg_np=np.array(vimg)
    feats=proc.image_processor(images=[vimg_np],return_tensors='np')
    g=feats['image_grid_thw'][0];vn=max(1,int(g[1])*int(g[2])//4)
    prompt=('<' + '|placeholder|' + '>')*vn + 'OCR:'
    inp=proc(text=[prompt],images=[vimg_np],return_tensors='np',padding=True,max_length=2048,truncation=True)
    ipd={}
    for k,v in inp.items():
        if isinstance(v,np.ndarray):ipd[k]=paddle.to_tensor(v)
        elif isinstance(v,t.Tensor):ipd[k]=paddle.to_tensor(v.numpy())
        elif isinstance(v,list) and len(v)>0:
            if isinstance(v[0],np.ndarray):ipd[k]=paddle.to_tensor(np.array(v))
            else:ipd[k]=v
        else:ipd[k]=v
    ipd['pixel_values']=paddle.to_tensor(feats['pixel_values']) if isinstance(feats['pixel_values'],np.ndarray) else paddle.to_tensor(feats['pixel_values'].numpy())
    ipd['image_grid_thw']=paddle.to_tensor(feats['image_grid_thw']) if isinstance(feats['image_grid_thw'],np.ndarray) else paddle.to_tensor(feats['image_grid_thw'].numpy())
    with paddle.no_grad():
        o=model(**ipd)
        if isinstance(o,(list,tuple)):
            print('  output is tuple, len='+str(len(o)),[str(type(x))[-20:] for x in o])
            # First element is loss (scalar), second is logits
            if len(o)>1:l=o[1]
            else:l=o[0]
        elif hasattr(o,'logits'):
            l=o.logits
        else:
            l=o
    return l[:,-1,:]

# Load 2 different images
val=[json.loads(l) for l in open(P+'/output/val_clean.jsonl')][:2]
img1=Image.open(val[0]['images'][0]).convert('RGB')
img2=Image.open(val[1]['images'][0]).convert('RGB')
w1,h1=img1.size;s1=384/max(w1,h1)
if s1<1:img1=img1.resize((int(w1*s1),int(h1*s1)),Image.LANCZOS)
w2,h2=img2.size;s2=384/max(w2,h2)
if s2<1:img2=img2.resize((int(w2*s2),int(h2*s2)),Image.LANCZOS)

l1=get_logits(img1)
l2=get_logits(img2)

# Compare
diff=(l1-l2).abs().mean().numpy()[0]
print('Logits mean abs diff:',float(diff))
t1=int(paddle.argmax(l1,axis=-1).numpy()[0])
t2=int(paddle.argmax(l2,axis=-1).numpy()[0])
print('Image1 top token:',t1,repr(proc.tokenizer.decode([t1])))
print('Image2 top token:',t2,repr(proc.tokenizer.decode([t2])))
print('DIFFER?:',t1!=t2)
print('GT1:',val[0]['messages'][1]['content'][:60])
print('GT2:',val[1]['messages'][1]['content'][:60])

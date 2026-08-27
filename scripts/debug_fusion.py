"""Debug: check image fusion counts in model forward."""
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
sys.modules.pop('torchvision',None);import torchvision,torchvision.transforms,torch as t
import transformers.utils.import_utils as tiu
tiu.is_torch_available=lambda:(True,'');tiu.is_torchvision_available=lambda:(True,'')

M='/root/models/official_models/PaddleOCR-VL';P='/root/circuit_ocr'

proc=AutoProcessor.from_pretrained(M,trust_remote_code=True)
model=AutoModelForConditionalGeneration.from_pretrained(M,load_checkpoint_format='safetensors',dtype='bfloat16')
model.config._attn_implementation='sdpa';model.visual.config._attn_implementation='sdpa'
model.eval()

val=json.loads(open(P+'/output/val_clean.jsonl').readline())
vimg=Image.open(val['images'][0]).convert('RGB')
w,h=vimg.size;s=384/max(w,h)
if s<1:vimg=vimg.resize((int(w*s),int(h*s)),Image.LANCZOS)
vimg_np=np.array(vimg)

# Prepare inputs exactly as training does
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

# Count image tokens
img_tok_id=model.config.image_token_id
input_ids_np=ipd['input_ids'].numpy() if hasattr(ipd['input_ids'],'numpy') else ipd['input_ids'].numpy()
n_img_tok=int((input_ids_np==img_tok_id).sum())
print('image_token_id:',img_tok_id)
print('n_image_tokens:',n_img_tok)
print('pv shape:',ipd['pixel_values'].shape)
print('igt shape:',ipd['image_grid_thw'].shape)
print('input_ids shape:',ipd['input_ids'].shape)

# Check if pixel_values are actually non-zero
pv_np=ipd['pixel_values'].numpy() if hasattr(ipd['pixel_values'],'numpy') else ipd['pixel_values'].numpy()
print('pv mean:',float(pv_np.mean()),'max:',float(pv_np.max()),'min:',float(pv_np.min()))

# Forward pass
with paddle.no_grad():
    o=model(**ipd)
    logits=o[0] if isinstance(o,(list,tuple)) else o.logits
    # Check logits distribution
    lt=logits[:,-1,:]
    top5=paddle.topk(lt,5,axis=-1)
    print('Top 5 token IDs:',top5.indices.numpy()[0].tolist())
    print('Top 5 probs:',paddle.nn.functional.softmax(lt,axis=-1)[0,top5.indices[0]].numpy().tolist())
    for tid in top5.indices[0].numpy().tolist():
        print('  ',tid,repr(proc.tokenizer.decode([tid])))

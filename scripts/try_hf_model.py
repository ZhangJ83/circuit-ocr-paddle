"""Try loading model from HF with convert_from_hf=True"""
import os, sys
sys.modules.pop('torchvision',None)
import torchvision,torchvision.transforms
from mistral_common.tokens.tokenizers import utils as mu
if not hasattr(mu,'get_one_valid_tokenizer_file'):mu.get_one_valid_tokenizer_file=lambda d,e:list(mu._filter_valid_tokenizer_files(d,e))
import paddle;paddle.set_device('gpu')
import paddle.nn.functional as F
if not hasattr(F,'swiglu'):F.swiglu=lambda x:paddle.chunk(x,2,-1)[0]*F.silu(paddle.chunk(x,2,-1)[1])
_o=F.scaled_dot_product_attention;F.scaled_dot_product_attention=lambda *a,**kw:_o(*a,**{k:v for k,v in kw.items() if k!='enable_gqa'})
import numpy as np;from PIL import Image
from paddleformers.transformers import AutoModelForConditionalGeneration,AutoProcessor
sys.modules.pop('torchvision',None);import torchvision,torchvision.transforms,torch
import transformers.utils.import_utils as tiu
tiu.is_torch_available=lambda:(True,'');tiu.is_torchvision_available=lambda:(True,'')

# Try downloading from HF
print('Downloading from HF...')
try:
    model = AutoModelForConditionalGeneration.from_pretrained(
        "PaddlePaddle/PaddleOCR-VL", convert_from_hf=True, dtype="bfloat16")
    print('OK - HF download works!')
    # Quick test with image
    proc = AutoProcessor.from_pretrained("PaddlePaddle/PaddleOCR-VL", trust_remote_code=True)
    model.eval()
    import json
    val = json.loads(open('/root/circuit_ocr/output/val_clean.jsonl').readline())
    vimg = Image.open(val['images'][0]).convert('RGB')
    w,h=vimg.size;s=384/max(w,h)
    if s<1:vimg=vimg.resize((int(w*s),int(h*s)),Image.LANCZOS)
    feats=proc.image_processor(images=[np.array(vimg)],return_tensors='np')
    g=feats['image_grid_thw'][0];vn=max(1,int(g[1])*int(g[2])//4)
    prompt=('<' + '|placeholder|' + '>')*vn + 'OCR:'
    inp=proc(text=[prompt],images=[np.array(vimg)],return_tensors='np',padding=True,max_length=2048,truncation=True)
    ipd={}
    for k,v in inp.items():
        if isinstance(v,np.ndarray):ipd[k]=paddle.to_tensor(v)
        elif isinstance(v,torch.Tensor):ipd[k]=paddle.to_tensor(v.numpy())
        else:ipd[k]=v
    ipd['pixel_values']=paddle.to_tensor(feats['pixel_values'])
    ipd['image_grid_thw']=paddle.to_tensor(feats['image_grid_thw'])
    with paddle.no_grad():
        o=model(**ipd)
        l=o[0] if isinstance(o,(list,tuple)) else o.logits
        t=int(paddle.argmax(l[:,-1,:],axis=-1).numpy()[0])
    print('First token:',t,repr(proc.tokenizer.decode([t])))
    # Test WITHOUT images
    ipd2={k:v for k,v in ipd.items() if k not in ['pixel_values','image_grid_thw']}
    with paddle.no_grad():
        o2=model(**ipd2)
        l2=o2[0] if isinstance(o2,(list,tuple)) else o2.logits
        t2=int(paddle.argmax(l2[:,-1,:],axis=-1).numpy()[0])
    print('Without images:',t2,repr(proc.tokenizer.decode([t2])))
    print('SAME?:',t==t2)
except Exception as e:
    print('FAILED:',str(e)[:200])

"""Check if checkpoint keys match model keys."""
import paddle,os,sys
sys.modules.pop('torchvision',None)
import torchvision,torchvision.transforms
from mistral_common.tokens.tokenizers import utils as mu
if not hasattr(mu,'get_one_valid_tokenizer_file'):mu.get_one_valid_tokenizer_file=lambda d,e:list(mu._filter_valid_tokenizer_files(d,e))
paddle.set_device('gpu')
import paddle.nn.functional as F
if not hasattr(F,'swiglu'):F.swiglu=lambda x:paddle.chunk(x,2,-1)[0]*F.silu(paddle.chunk(x,2,-1)[1])
_o=F.scaled_dot_product_attention;F.scaled_dot_product_attention=lambda *a,**kw:_o(*a,**{k:v for k,v in kw.items() if k!='enable_gqa'})
from paddleformers.transformers import AutoModelForConditionalGeneration
from paddleformers.peft import LoRAConfig,LoRAModel
sys.modules.pop('torchvision',None)
import torchvision,torchvision.transforms,torch
import transformers.utils.import_utils as tiu
tiu.is_torch_available=lambda:(True,'');tiu.is_torchvision_available=lambda:(True,'')

# Load checkpoint
sd=paddle.load('checkpoints/baseline/checkpoint_s800.pdparams')
ckpt_keys=list(sd.keys())
print(len(ckpt_keys),'checkpoint keys')
for k in ckpt_keys[:5]:print('  CKPT:',k)
print('  ...')
for k in ckpt_keys[-3:]:print('  CKPT:',k)

# Load model with LoRA
M='/root/models/official_models/PaddleOCR-VL'
model=AutoModelForConditionalGeneration.from_pretrained(M,load_checkpoint_format='safetensors',dtype='bfloat16')
for n,p in model.named_parameters():
    if 'mlp_AR' in n or 'projector' in n:p.stop_gradient=True
lc=LoRAConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=['.*q_proj','.*k_proj','.*v_proj','.*o_proj','.*linear_1','.*linear_2'])
model=LoRAModel(model,lc)

model_keys=[n for n,p in model.named_parameters() if 'lora_' in n]
print(len(model_keys),'model lora keys')
for k in model_keys[:5]:print('  MODEL:',k)
print('  ...')
for k in model_keys[-3:]:print('  MODEL:',k)

# Check match
ms=set(model_keys)
matched=sum(1 for k in ckpt_keys if k in ms)
print('MATCHED:',matched,'/',len(ckpt_keys))

# Show mismatch
not_matched=[k for k in ckpt_keys if k not in ms]
if not_matched:
    print('NOT MATCHED examples:')
    for k in not_matched[:5]:print('  ',k)

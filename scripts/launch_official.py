"""Launch official paddleformers training with fsdp monkey-patch."""
import sys
from types import ModuleType

# Monkey-patch fsdp (not available in Paddle 3.2.0)
_fsdp = ModuleType('fsdp')
_fsdp.fully_shard = lambda *a, **kw: None
_fsdp.FullyShardedDataParallel = type('FSDP', (object,), {})
sys.modules['paddle.distributed.fsdp'] = _fsdp

# Also need fsdp.fully_shard at the package level
import paddle.distributed
paddle.distributed.fsdp = _fsdp

from paddleformers.cli.cli import main
sys.argv = [
    'paddleformers-cli', 'train', 'official_lora.yaml',
    'pre_alloc_memory=18'
]
main()

"""Debug flex_checkpoint state."""
import sys
for k in sorted(sys.modules):
    if 'flex' in k or 'sharded' in k.lower():
        print(f'{k}: {type(sys.modules[k]).__name__}')

# Try patching with the right approach
print("\n--- Patching ---")
from types import ModuleType, SimpleNamespace

# Force-overwrite all flex_checkpoint modules
fc = SimpleNamespace(
    build_sharded_state_dict=lambda *a, **kw: None,
    shard_weight=lambda *a, **kw: None,
    make_replicated_sharded_weight=lambda *a, **kw: None,
    ShardedStateDict=type('ShardedStateDict', (), {}),
    ShardedTensor=type('ShardedTensor', (), {}),
    ShardedWeight=type('ShardedWeight', (), {}),
    StateDictSaveHook=type('StateDictSaveHook', (), {}),
    state_dict_merge_fn=lambda *a, **kw: None,
)

for mod_name in ['paddle.distributed.flex_checkpoint',
                  'paddle.distributed.flex_checkpoint.dcp',
                  'paddle.distributed.flex_checkpoint.dcp.sharded_weight']:
    sys.modules[mod_name] = fc

print("Patch applied")
print(f"fc has ShardedWeight: {hasattr(fc, 'ShardedWeight')}")

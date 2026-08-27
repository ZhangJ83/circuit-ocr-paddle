"""Test that train() works end-to-end with 1 epoch."""
import sys, os
sys.path.insert(0, "scripts")
from configs import get_config
from train import train

cfg = get_config("baseline")
cfg["epochs"] = 1
cfg["val_samples"] = 3  # minimal val for testing
print("Calling train()...")
try:
    result = train(cfg)
    print(f"SUCCESS: {result}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

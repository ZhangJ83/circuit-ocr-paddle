"""Minimal: torchvision dummy -> can processor load?"""
import sys, os
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
from importlib.machinery import ModuleSpec

# Create torchvision dummy BEFORE any imports
def mkpkg(name):
    m = type(sys)(name); m.__version__ = "0.0.0"; m.__path__ = []
    m.__spec__ = ModuleSpec(name, None, is_package=True)
    return m
class IM: pass

# Set ALL torchvision modules as valid dummies
for name in ["torchvision", "torchvision.transforms", "torchvision.transforms.functional",
             "torchvision.transforms.v2", "torchvision.transforms.v2.functional"]:
    m = mkpkg(name)
    if name == "torchvision.transforms": m.InterpolationMode = IM
    sys.modules[name] = m

# Now try processor
import paddle; paddle.set_device("gpu")
from paddleformers.transformers import AutoProcessor

proc = AutoProcessor.from_pretrained(
    "/root/models/official_models/PaddleOCR-VL", trust_remote_code=True)
print("SUCCESS: Processor loaded")

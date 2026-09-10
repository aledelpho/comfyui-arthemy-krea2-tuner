"""Fake ComfyUI environment that keeps the REAL torch.

The no-torch `stub_env.py` next to this file is enough for the pure-python tests, but anything
that touches nn.Module trees needs a real torch. Run these with ComfyUI's own interpreter:

    <ComfyUI python> tests/test_baker.py
"""
import os, sys, types, tempfile
from unittest.mock import MagicMock

_TMP = tempfile.mkdtemp(prefix="fake_comfy_")

def _mock(name):
    m = MagicMock(); m.__name__ = name; m.__spec__ = MagicMock()
    sys.modules[name] = m
    return m

for _n in ("PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont",
           "safetensors", "safetensors.torch",
           "comfy", "comfy.lora", "comfy.model_patcher", "comfy.sd", "comfy.utils",
           "comfy.weight_adapter", "server"):
    _mock(_n)

# A REAL base class where ComfyUI has one: the suite subclasses WeightAdapterBase, and a
# MagicMock in its place would make `isinstance` checks raise instead of answering.
import torch as _torch

class _WeightAdapterBase:
    name = "stub"
    @classmethod
    def load(cls, *a, **kw): return None
    def calculate_weight(self, weight, key, strength, strength_model, offset, function,
                         intermediate_dtype=_torch.float32, original_weights=None):
        return weight

class _LoRAAdapter(_WeightAdapterBase):
    name = "lora"

sys.modules["comfy.weight_adapter"].WeightAdapterBase = _WeightAdapterBase
sys.modules["comfy.weight_adapter"].LoRAAdapter = _LoRAAdapter

fp = types.ModuleType("folder_paths")
fp.models_dir = os.path.join(_TMP, "models"); os.makedirs(fp.models_dir, exist_ok=True)
fp.folder_names_and_paths = {}
fp.get_folder_paths = lambda k: [os.path.join(fp.models_dir, k)]
fp.get_filename_list = lambda k: []
fp.get_full_path = lambda k, n: None
fp.supported_pt_extensions = {".safetensors"}
sys.modules["folder_paths"] = fp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

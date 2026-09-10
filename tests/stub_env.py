"""Minimal fake ComfyUI/torch environment so the suite module can be imported for tests."""
import os, sys, types, tempfile
from unittest.mock import MagicMock

_TMP = tempfile.mkdtemp(prefix="fake_comfy_")

def _mock_module(name):
    m = MagicMock()
    m.__name__ = name
    m.__spec__ = MagicMock()
    sys.modules[name] = m
    return m

for n in ("torch", "torch.nn", "numpy", "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont",
          "safetensors", "safetensors.torch", "comfy", "comfy.lora", "comfy.model_patcher",
          "comfy.sd", "comfy.utils", "comfy.weight_adapter", "server"):
    _mock_module(n)

fp = types.ModuleType("folder_paths")
fp.models_dir = os.path.join(_TMP, "models")
os.makedirs(fp.models_dir, exist_ok=True)
fp.folder_names_and_paths = {}
fp.get_folder_paths = lambda k: [os.path.join(fp.models_dir, k)]
fp.get_filename_list = lambda k: []
fp.get_full_path = lambda k, n: None
fp.supported_pt_extensions = {".safetensors"}
sys.modules["folder_paths"] = fp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

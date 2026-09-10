import sys
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("SKIP: needs real torch")
    sys.exit(0)

import os
import sys

_tests_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.dirname(_tests_dir)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
import stubs_real_torch, importlib
import comfy.lora
comfy.lora.calculate_weight = lambda patches, weight, key: weight + 0.5

m = importlib.import_module("Arthemy_Krea2_Tuner")

FAIL = []
def chk(c, label, extra=""):
    print(f"  {'ok  ' if c else 'FAIL'} {label}{'' if c else '   <- ' + str(extra)}")
    if not c: FAIL.append(label)

print("\n--- TEST 1: scaled-fp8 companion scale recognition ---")
# The spellings that actually occur. Checked against the ComfyUI source rather than assumed:
#   comfy/utils.py:1469  ->  a checkpoint on disk carries "<layer>.scale_weight", a SIBLING of
#                            "<layer>.weight", and the loader rewrites it to
#                            "<layer>.weight_scale".
# "<layer>.weight.scale_weight" - a suffix on the weight key - exists nowhere, and testing for
# it is how a wrong-shaped check passes its own test while missing every real checkpoint.
for spelling, scale_key in (
    ("on-disk sibling", "blocks.0.attn.wq.scale_weight"),
    ("loader-converted", "blocks.0.attn.wq.weight_scale"),
):
    sd_test = {"blocks.0.attn.wq.weight": torch.zeros((4, 4)), scale_key: torch.tensor(2.0)}
    chk(m.find_companion_scale("blocks.0.attn.wq.weight", sd_test) is not None,
        f"find_companion_scale finds the {spelling} form ({scale_key})")
    chk(m.is_skippable_for_tuning("blocks.0.attn.wq.weight", sd_test),
        f"and the weight is left alone as quantized ({spelling})")

chk(m.is_bookkeeping_sd_key("blocks.0.attn.wq.scale_weight"),
    "is_bookkeeping_sd_key recognizes .scale_weight")
chk(m.find_companion_scale("blocks.0.attn.wq.weight",
                           {"blocks.0.attn.wq.weight": torch.zeros((4, 4))}) is None,
    "and a plain bf16 weight has no companion")

print("\n--- TEST 2: dequantize_weight 2D broadcast ---")
w_fp8 = torch.ones((3, 2), dtype=torch.float8_e4m3fn)
s_1d = torch.tensor([2.0, 3.0, 4.0])  # out channels = 3
w_deq = m.dequantize_weight(w_fp8, scale=s_1d)
expected_col0 = torch.tensor([2.0, 3.0, 4.0])
chk(torch.allclose(w_deq[:, 0], expected_col0), "dequantize_weight applies out-channel scale along dim 0",
    f"got col0 = {w_deq[:, 0].tolist()} vs expected {expected_col0.tolist()}")

print("\n--- TEST 3: CLIP stream key preservation (no collision) ---")
clip_sd = {
    "clip_l.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight": torch.zeros((768, 768)),
    "clip_g.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight": torch.zeros((1280, 1280))
}
streamed_keys = [k for k, v in m.process_tensor_stream(clip_sd, lambda k, v: v, clean_keys=False)]
chk(len(streamed_keys) == 2 and streamed_keys[0] != streamed_keys[1],
    "CLIP stream preserves distinct clip_l and clip_g keys", streamed_keys)

print("\n--- TEST 4: ModelSaver and CLIPSaver calculate patches when present ---")
class FakePatcher:
    def __init__(self, sd):
        self.model = type("FakeModel", (), {"state_dict": lambda s: sd})()
        self.patches = {}
        self.backup = {}
        self.object_patches = {}

base_w = torch.full((4, 4), 2.0)
patcher = FakePatcher({"blocks.0.attn.wq.weight": base_w})
patcher.patches = {"blocks.0.attn.wq.weight": [(0.5,)]}

p_k = "blocks.0.attn.wq.weight"
patched_w = m.ComfyPatcherAdapter.calculate_safe_weight(patcher, p_k, base_w, model_sd={"blocks.0.attn.wq.weight": base_w})
chk(torch.allclose(patched_w, torch.full((4, 4), 2.5, dtype=torch.bfloat16)), "calculate_safe_weight applied patch calculation to base weight")

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
sys.exit(0 if not FAIL else 1)

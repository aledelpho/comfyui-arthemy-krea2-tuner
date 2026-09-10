"""The Model Baker must actually fold the patches into the module - and must never clear a
tuning it failed to fold.

The bug this pins: `isolate_and_assign_baked_weights` used to walk the module tree with
`clean_key(k)`, which strips the very prefix (`diffusion_model.`) that names the first hop, so
on a stock ComfyUI patcher nothing was ever located - and the function cleared `patches` anyway.

Needs a real torch, so run it with ComfyUI's own interpreter:
    <ComfyUI python> tests/test_baker.py
"""
import sys

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("SKIP: this test needs a real torch - run it with ComfyUI's interpreter.")
    sys.exit(0)

import stubs_real_torch, importlib
m = importlib.import_module("Arthemy_Krea2_Tuner")

FAIL = []
def chk(c, label, extra=""):
    print(f"  {'ok  ' if c else 'FAIL'} {label}{'' if c else '   <- ' + str(extra)}")
    if not c: FAIL.append(label)

class Diff(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(2)])
        self.register_buffer("scale_buf", torch.ones(4))

class BaseModelLike(nn.Module):
    """Mirrors comfy.model_base.BaseModel: the UNet hangs off a `diffusion_model` child,
    so every state_dict key is prefixed with it."""
    def __init__(self):
        super().__init__()
        self.diffusion_model = Diff()

class FakePatcher:
    def __init__(self, model):
        self.model = model
        self.patches = {}
        self.backup = {}
        self.object_patches = {}
        self.model_options = {}
        self.patches_uuid = None

KEY = "diffusion_model.blocks.0.weight"

def fresh():
    p = FakePatcher(BaseModelLike())
    p.patches = {KEY: [("dummy",)], "diffusion_model.blocks.1.weight": [("dummy",)]}
    return p

print("\n1. a baked tensor actually reaches the module")
p = fresh()
before = p.model.diffusion_model.blocks[0].weight.detach().clone()
baked = torch.full((4, 4), 7.0)
m.isolate_and_assign_baked_weights(p, {KEY: baked})
after = dict(p.model.state_dict())[KEY]
chk(torch.allclose(after, baked), "the parameter now holds the baked value",
    f"still {after.flatten()[:3].tolist()} (was {before.flatten()[:3].tolist()})")

print("\n2. the tuning is not destroyed when the bake fails")
p2 = fresh()
m.isolate_and_assign_baked_weights(p2, {"diffusion_model.nope.weight": torch.zeros(4, 4)})
chk(len(p2.patches) > 0, "patches survive a bake that located nothing",
    f"patches = {p2.patches}")

print("\n3. a baked key stops being a pending patch (no double application)")
p3 = fresh()
m.isolate_and_assign_baked_weights(p3, {KEY: torch.full((4, 4), 7.0)})
chk(KEY not in p3.patches, "the baked key was removed from patches", f"patches = {list(p3.patches)}")

print("\n4. the original module is not mutated (copy-on-write)")
p4 = fresh()
shared = p4.model
orig = shared.diffusion_model.blocks[0].weight.detach().clone()
m.isolate_and_assign_baked_weights(p4, {KEY: torch.full((4, 4), 7.0)})
chk(torch.allclose(shared.diffusion_model.blocks[0].weight, orig),
    "the pre-bake module object still holds its own weights")

print("\n5. buffers are assigned as buffers, not turned into Parameters")
p5 = fresh()
m.isolate_and_assign_baked_weights(p5, {"diffusion_model.scale_buf": torch.full((4,), 3.0)})
buf = p5.model.diffusion_model.scale_buf
chk(torch.allclose(buf, torch.full((4,), 3.0)) and not isinstance(buf, nn.Parameter),
    "buffer updated and still a plain tensor")

print("\n6. a partial bake keeps the patches it could not fold in")
p6 = fresh()
n = m.isolate_and_assign_baked_weights(p6, {KEY: torch.full((4, 4), 7.0),
                                            "diffusion_model.nope.weight": torch.zeros(4, 4)})
chk(n == 1, "reports 1 assigned, not 2", n)
chk(KEY not in p6.patches, "the folded key is gone from patches")
chk("diffusion_model.blocks.1.weight" in p6.patches, "the untouched key still applies",
    list(p6.patches))

print("\n7. nothing to bake but patches pending -> leave everything alone")
p7 = fresh()
m.isolate_and_assign_baked_weights(p7, {})
chk(len(p7.patches) == 2, "pending patches survive an empty bake", p7.patches)

print("\n8. nothing to bake and nothing pending -> clean up as before")
p8 = FakePatcher(BaseModelLike())
p8.patches = {}
p8.backup = {"x": 1}
m.isolate_and_assign_baked_weights(p8, {})
chk(p8.backup == {}, "the old clean-up path still runs")

print("\n9. fallback: a patcher whose .model IS the inner module")
class Direct(FakePatcher):
    pass
p9 = Direct(Diff())                      # no diffusion_model wrapper
p9.patches = {"diffusion_model.blocks.0.weight": [("dummy",)]}
n9 = m.isolate_and_assign_baked_weights(p9, {"diffusion_model.blocks.0.weight": torch.full((4, 4), 5.0)})
chk(n9 == 1 and torch.allclose(p9.model.blocks[0].weight, torch.full((4, 4), 5.0)),
    "the cleaned spelling is tried when the raw path misses", n9)

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)

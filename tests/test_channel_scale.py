"""The channel-scale adapter must compute exactly what the dense delta computed, while
carrying 24 KB instead of a copy of the model."""
import sys
try:
    import torch
except ImportError:
    print("SKIP: needs a real torch."); sys.exit(0)

import stubs_real_torch, importlib
m = importlib.import_module("Arthemy_Krea2_Tuner")

FAIL = []
def chk(c, label, extra=""):
    print(f"  {'ok  ' if c else 'FAIL'} {label}{'' if c else '   <- ' + str(extra)}")
    if not c: FAIL.append(label)

D = 64
torch.manual_seed(3)
s = torch.ones(D); s[:8] = 1.25; s[8:16] = 0.8

print("\n1. the API is present in this environment")
chk(m.HAS_WEIGHT_ADAPTER_API, "HAS_WEIGHT_ADAPTER_API")

print("\n2. exactly equal to the dense delta it replaces")
for axis, shape in ((0, (D, 128)), (1, (128, D))):
    W = torch.randn(*shape)
    ad = m.build_channel_scale_patch(s, axis=axis)
    got = ad.calculate_weight(W.clone(), "k", 1.0, 1.0, None, None)
    factor = s.unsqueeze(1) if axis == 0 else s.unsqueeze(0)
    dense = W + W * (factor - 1.0)          # what the node used to inject
    chk(torch.allclose(got, dense, atol=1e-6), f"axis {axis}: adapter == dense",
        (got - dense).abs().max().item())

print("\n3. square weight, both axes: order-independent product")
W = torch.randn(D, D)
w0 = m.build_channel_scale_patch(s, axis=0)
w1 = m.build_channel_scale_patch(s, axis=1)
a = w1.calculate_weight(w0.calculate_weight(W.clone(), "k", 1.0, 1.0, None, None), "k", 1.0, 1.0, None, None)
b = w0.calculate_weight(w1.calculate_weight(W.clone(), "k", 1.0, 1.0, None, None), "k", 1.0, 1.0, None, None)
chk(torch.allclose(a, b, atol=1e-6), "the two gains commute")
chk(torch.allclose(a, W * s.unsqueeze(1) * s.unsqueeze(0), atol=1e-6), "and give s_out * W * s_in")

print("\n4. strength interpolates the gain, it does not scale the weight")
W = torch.randn(D, 8)
ad = m.build_channel_scale_patch(s, axis=0)
half = ad.calculate_weight(W.clone(), "k", 0.5, 1.0, None, None)
chk(torch.allclose(half, W * (1.0 + (s - 1.0) * 0.5).unsqueeze(1), atol=1e-6), "strength 0.5")
chk(torch.allclose(ad.calculate_weight(W.clone(), "k", 0.0, 1.0, None, None), W), "strength 0 is a no-op")

print("\n5. it never re-applies strength_model (the caller already did)")
W = torch.randn(D, 8)
got = ad.calculate_weight(W.clone(), "k", 1.0, 2.0, None, None)
chk(torch.allclose(got, W * s.unsqueeze(1), atol=1e-6), "strength_model ignored inside the adapter")

print("\n6. a neutral vector produces no adapter at all")
chk(m.build_channel_scale_patch(torch.ones(D), axis=0) is None, "all-ones -> None")

print("\n7. a shape it does not describe is skipped, not broadcast onto")
W = torch.randn(D + 1, 8)
out = ad.calculate_weight(W.clone(), "narrowed", 1.0, 1.0, None, None)
chk(torch.allclose(out, W), "mismatched axis length leaves the weight alone")

print("\n8. `function` is honoured on the general path")
W = torch.randn(D, 8)
got = ad.calculate_weight(W.clone(), "k", 1.0, 1.0, None, lambda t: t * 0.0)
chk(torch.allclose(got, W), "a zeroing function cancels the gain")

print("\n9. the fast path is in-place: no full-size temporary")
W = torch.randn(D, 8)
ref = W
out = ad.calculate_weight(W, "k", 1.0, 1.0, None, None)
chk(out is ref, "returns the same tensor object")

print("\n10. payload: one shared vector, not one delta per key")
import sys as _s
scales = torch.ones(6144); scales[:614] = 1.15
ad0 = m.build_channel_scale_patch(scales, axis=0)
per_key = [ad0 for _ in range(84)]
adapter_bytes = ad0.scales.numel() * ad0.scales.element_size()
dense_bytes = 84 * 16384 * 6144 * 2          # bf16 delta per matrix, the old path
chk(len({id(a) for a in per_key}) == 1, "84 keys share one adapter object")
chk(adapter_bytes < 30_000, f"payload is {adapter_bytes/1024:.0f} KB", adapter_bytes)
print(f"     old dense path: {dense_bytes/1024**3:6.1f} GB     adapter: {adapter_bytes/1024:.0f} KB"
      f"     ratio 1 : {dense_bytes/adapter_bytes:,.0f}")

print("\n11. bf16 weights stay bf16 and are not upcast into a second copy")
W = torch.randn(D, 8).to(torch.bfloat16)
out = ad.calculate_weight(W.clone(), "k", 1.0, 1.0, None, None)
chk(out.dtype == torch.bfloat16, "dtype preserved", out.dtype)

print("\n12. it is not mistaken for an external LoRA")
off, is_lora, is_chaos, is_rot, *_ = m.parse_patch_entry((1.0, ad0, 1.0, None, None))
chk(not is_lora, "parse_patch_entry does not flag it as a LoRA")

print("\n13. survives ComfyUI's reconstruction (the two crashes this pins)")
# comfy/lora.py::prefetch_prepared_value gathers every tensor in `weights` into one VRAM
# buffer and copies into it ASYNCHRONOUSLY, then rebuilds the adapter around views onto that
# buffer. So: (a) the constructor must read no VALUE out of `weights` - the copy may not have
# landed - and (b) anything that is not weight data must not live there at all.
chk(len(ad0.weights) == 1, "weights holds only the scales vector", len(ad0.weights))
chk(type(ad0).AXIS == 0 and type(m.build_channel_scale_patch(scales, axis=1)).AXIS == 1,
    "the axis is carried by the class, not by the data")

class _Landmine(torch.Tensor):
    """A tensor whose .item() explodes - stands in for a buffer view the copy has not reached."""
    def item(self, *a, **k):
        raise AssertionError("the constructor must not read values out of `weights`")

mine = _Landmine(torch.zeros(4))
try:
    type(ad0)(set(), [mine])
    chk(True, "constructing over a not-yet-filled buffer reads nothing")
except AssertionError as e:
    chk(False, "constructing over a not-yet-filled buffer reads nothing", e)

# comfy/lora.py::prefetch_prepared_value does exactly this, from ModelPatcher.memory_required,
# on every module cast during sampling. Any state not inside `weights` is gone here.
rebuilt = type(ad0)(ad0.loaded_keys, ad0.weights)
chk(rebuilt.axis == ad0.axis, "axis survives", (rebuilt.axis, ad0.axis))
chk(torch.equal(rebuilt.scales, ad0.scales), "scales survive")
W = torch.randn(6144, 8)
chk(torch.allclose(rebuilt.calculate_weight(W.clone(), "k", 1.0, 1.0, None, None),
                   ad0.calculate_weight(W.clone(), "k", 1.0, 1.0, None, None)),
    "and it computes the same thing")
rebuilt2 = type(rebuilt)(rebuilt.loaded_keys, rebuilt.weights)
chk(rebuilt2.axis == ad0.axis, "twice over, too")
ad1 = m.build_channel_scale_patch(scales, axis=1)
r1 = type(ad1)(ad1.loaded_keys, ad1.weights)
chk(r1.axis == 1, "axis 1 is not silently reset to 0", r1.axis)
off, is_lora, *_ = m.parse_patch_entry((1.0, rebuilt, 1.0, None, None))
chk(not is_lora, "a reconstructed adapter is still not a LoRA")

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)

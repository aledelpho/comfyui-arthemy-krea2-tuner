"""The channel profile must be cached per (checkpoint, axis), invalidate itself, and measure
the quantity the gain will act on."""
import sys, os, tempfile
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

D, MLP = 64, 128
torch.manual_seed(11)

def make_sd(seed=0, fp8_scaled=False):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for b in range(3):
        sd[f"diffusion_model.blocks.{b}.attn.qkv.weight"]  = torch.randn(3*D, D, generator=g)
        sd[f"diffusion_model.blocks.{b}.attn.proj.weight"] = torch.randn(D, D, generator=g)
        sd[f"diffusion_model.blocks.{b}.mlp.gate.weight"]  = torch.randn(MLP, D, generator=g)
        sd[f"diffusion_model.blocks.{b}.mlp.down.weight"]  = torch.randn(D, MLP, generator=g)
    if fp8_scaled:
        # ComfyUI's scaled-fp8 layout: a companion scale makes the weight untouchable, so it
        # drops out of the profile too.
        for b in range(3):
            sd[f"diffusion_model.blocks.{b}.mlp.gate.scale_weight"] = torch.ones(MLP)   # sibling of .weight
    return sd

class P:
    def __init__(self, sd): self.backup = {}; self._sd = sd
    @property
    def model(self): raise AssertionError("profiling must not need patcher.model")

print("\n1. the residual width is read off the checkpoint")
sd = make_sd()
chk(m.Krea2Config.probe_hidden_size(sd) == D, "probe finds d_model", m.Krea2Config.probe_hidden_size(sd))
# One matrix is not an architecture: refusing to guess beats guessing wrong.
lonely = {"diffusion_model.x.weight": torch.randn(1024, 512)}
chk(m.Krea2Config.probe_hidden_size(lonely) == m.Krea2Config.HIDDEN_SIZE,
    "a single matrix falls back instead of guessing", m.Krea2Config.probe_hidden_size(lonely))
# but a real stack answers even at an unusual width
odd = {f"diffusion_model.blocks.{b}.attn.proj.weight": torch.randn(320, 320) for b in range(3)}
odd.update({f"diffusion_model.blocks.{b}.mlp.gate.weight": torch.randn(768, 320) for b in range(3)})
chk(m.Krea2Config.probe_hidden_size(odd) == 320, "and finds a non-Krea2 width",
    m.Krea2Config.probe_hidden_size(odd))
chk(m.Krea2Config.probe_hidden_size({}) == m.Krea2Config.HIDDEN_SIZE, "empty -> the documented fallback")

print("\n2. the metric follows the axis")
reads = m.channel_profile_matrices(sd, D, axis=1)
writes = m.channel_profile_matrices(sd, D, axis=0)
chk(all(sd[k].shape[1] == D for k in reads), "reads: every matrix has d_in == d_model")
chk(all(sd[k].shape[0] == D for k in writes), "writes: every matrix has d_out == d_model")
chk(set(reads) != set(writes), "the two sets differ", (len(reads), len(writes)))

print("\n3. a profile is a permutation of the channels")
order, note = m.resolve_channel_profile(P(sd), sd, D, axis=1)
chk(order is not None and sorted(order.tolist()) == list(range(D)), "every channel exactly once")
chk("computed" in note, "first call computes", note)

print("\n4. the second call comes from cache and is identical")
order2, note2 = m.resolve_channel_profile(P(sd), sd, D, axis=1)
chk(torch.equal(order, order2), "same ordering")
chk("cache" in note2, "and it says so", note2)

print("\n5. the cache separates axis, checkpoint and quantization")
o_w, n_w = m.resolve_channel_profile(P(sd), sd, D, axis=0)
chk(not torch.equal(order, o_w), "a different axis is a different profile")
sd_other = make_sd(seed=99)
o_other, n_other = m.resolve_channel_profile(P(sd_other), sd_other, D, axis=1)
chk("computed" in n_other, "a different finetune is not served from cache", n_other)
chk(not torch.equal(order, o_other), "and gives a different ordering")
sd_fp8 = make_sd(seed=0, fp8_scaled=True)
chk(len(m.channel_profile_matrices(sd_fp8, D, 1)) < len(reads),
    "a scaled-fp8 twin profiles fewer matrices")
o_fp8, n_fp8 = m.resolve_channel_profile(P(sd_fp8), sd_fp8, D, axis=1)
chk("computed" in n_fp8, "so it gets its own entry rather than the bf16 one", n_fp8)

print("\n5b. both scaled-fp8 spellings are recognised")
for spelling in ("scale_weight", "weight_scale"):
    probe_sd = make_sd(seed=0)
    probe_sd[f"diffusion_model.blocks.0.mlp.gate.{spelling}"] = torch.ones(MLP)
    chk(m.is_skippable_for_tuning("diffusion_model.blocks.0.mlp.gate.weight", probe_sd),
        f"<layer>.{spelling} marks the weight as quantized")
    chk(m.find_companion_scale("diffusion_model.blocks.0.mlp.gate.weight", probe_sd) is not None,
        f"and the scale is found for <layer>.{spelling}")
sd_conv = make_sd(seed=0)
sd_conv["diffusion_model.blocks.0.mlp.gate.weight_scale"] = torch.ones(MLP)
chk(m.is_skippable_for_tuning("diffusion_model.blocks.0.mlp.gate.weight", sd_conv),
    "the loader-converted <weight>_scale form too")

print("\n5c. the scope restricts both the profile and the injection")
mlp_only = m.channel_profile_matrices(sd, D, axis=1, scope="MLP (Gate/Up/Down)")
att_only = m.channel_profile_matrices(sd, D, axis=1, scope="ATTN (WQ/WK/WV/WO/Gate)")
all_r    = m.channel_profile_matrices(sd, D, axis=1, scope="All Components")
chk(all("mlp" in k for k in mlp_only), "MLP scope keeps only mlp matrices", mlp_only[:2])
chk(all("attn" in k for k in att_only), "ATTN scope keeps only attn matrices", att_only[:2])
chk(len(mlp_only) + len(att_only) == len(all_r), "and together they are the whole set",
    (len(mlp_only), len(att_only), len(all_r)))
chk(not set(mlp_only) & set(att_only), "with no overlap")
# writes side: only mlp.down has d_out == D among the MLP matrices
mlp_w = m.channel_profile_matrices(sd, D, axis=0, scope="MLP (Gate/Up/Down)")
chk(all(k.endswith("mlp.down.weight") for k in mlp_w), "MLP writes are the down projections", mlp_w[:2])

o_mlp, n_mlp = m.resolve_channel_profile(P(sd), sd, D, axis=1, scope="MLP (Gate/Up/Down)")
o_all, n_all = m.resolve_channel_profile(P(sd), sd, D, axis=1, scope="All Components")
# "All Components" was already computed above, so it comes back from cache - which is the
# point: the two scopes are separate entries, not one overwriting the other.
chk("computed" in n_mlp, "a new scope is computed, not served from another scope's entry", n_mlp)
chk("cache" in n_all, "while the scope computed earlier still hits its own entry", n_all)
chk(not torch.equal(o_mlp, o_all), "and the two orderings differ")
chk("cache" in m.resolve_channel_profile(P(sd), sd, D, axis=1, scope="MLP (Gate/Up/Down)")[1],
    "the new scope is cached from its second call on")
chk(m.channel_profile_fingerprint(sd, mlp_only, D, 1, "MLP (Gate/Up/Down)")
    != m.channel_profile_fingerprint(sd, mlp_only, D, 1, "All Components"),
    "the scope is part of the fingerprint even for the same key list")

print("\n6. fingerprints")
f1 = m.channel_profile_fingerprint(sd, reads, D, 1)
chk(f1 == m.channel_profile_fingerprint(sd, reads, D, 1), "stable across calls")
chk(f1 != m.channel_profile_fingerprint(sd, reads, D, 0), "axis is part of it")
chk(f1 != m.channel_profile_fingerprint(sd_other, m.channel_profile_matrices(sd_other, D, 1), D, 1),
    "content is part of it (same shapes, different weights)")

print("\n7. a corrupted cache file is refused, not trusted")
path = m._profile_cache_path(f1)
m._CHANNEL_PROFILE_MEM.clear()
open(path, "w").write('{"format": %d, "target_dim": %d, "axis": 1, "order": [0, 0, 0]}' % (m.CHANNEL_PROFILE_FORMAT, D))
chk(m.load_cached_channel_profile(f1, D) is None, "a non-permutation is rejected")
m._CHANNEL_PROFILE_MEM.clear()
open(path, "w").write("{ not json")
chk(m.load_cached_channel_profile(f1, D) is None, "garbage is rejected")
m._CHANNEL_PROFILE_MEM.clear()
o3, n3 = m.resolve_channel_profile(P(sd), sd, D, axis=1)
chk(torch.equal(o3, order), "and the profile is simply recomputed", n3)

print("\n8. a format bump invalidates every stored profile")
m._CHANNEL_PROFILE_MEM.clear()
import json as _j
d = _j.load(open(path)); d["format"] = m.CHANNEL_PROFILE_FORMAT - 1; _j.dump(d, open(path, "w"))
chk(m.load_cached_channel_profile(f1, D) is None, "older format ignored")

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)

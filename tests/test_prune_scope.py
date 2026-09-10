"""A scoped 5D injection must embed only the layers its scope can reach.

The point of the feature is preset weight: a 5D node restricted to one block used to embed
the WHOLE modifier - every block plus every CLIP layer. What is asserted here is the exact
layer set kept for each kind of scope, because "it got smaller" is not the property that
matters; "it kept precisely what the tuner will ask for at load time" is.
"""
import json, os, sys, tempfile
import stub_env, importlib          # noqa: F401
m = importlib.import_module("Arthemy_Krea2_Tuner")

FAIL = []
def check(cond, label, extra=""):
    (print(f"  ok   {label}") if cond else (FAIL.append(label), print(f"  FAIL {label}   <- {extra}")))

TENSORS = ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate", "mlp.gate", "mlp.up", "mlp.down")
SRC = "Comics_5D.json"


def payload():
    L = {}
    for b in range(28):
        for t in TENSORS:
            L[f"blocks.{b}.{t}"] = {"domain": "model", "d_out": 512, "d_in": 512,
                                    "singular_values": [1.0] * 5,
                                    "u_dct": [[0.01] * 16] * 5, "v_dct": [[0.01] * 16] * 5}
    for t in ("attn.wq", "mlp.up"):     # a block-less section: only "All Blocks" may keep it
        L[f"txtfusion.refiner_blocks.0.{t}"] = {"domain": "model", "d_out": 512, "d_in": 512,
                                                "singular_values": [1.0] * 5,
                                                "u_dct": [[0.01] * 16] * 5, "v_dct": [[0.01] * 16] * 5}
    for l in range(36):
        L[f"text_model.encoder.layers.{l}.self_attn.q_proj"] = {
            "domain": "clip", "d_out": 256, "d_in": 256, "singular_values": [1.0] * 5,
            "u_dct": [[0.01] * 16] * 5, "v_dct": [[0.01] * 16] * 5}
    return {"version": "5.0", "storage": "dct", "source_lora": "Comics.safetensors",
            "rank": 5, "harmonic_order": 16, "normalized": True, "layers": L}


TOTAL = len(payload()["layers"])          # 262


class FakePatcher:
    def __init__(self, recipes=None):
        self.patches = {}
        self.model_options = {"arthemy_5d_recipes": recipes or []}


def recipe(target, sub="All Components", sub_tensor=None):
    return {"domain": "model", "source_lora": SRC, "target_block": target,
            "sub_components": sub, "sub_tensor": sub_tensor or m.SUB_TENSOR_ANY,
            "master_multiplier": 1.0, "dimension_weights": [1.0, 0.5, 0.0, 0.0, 0.0]}


def save(recipes, prune=True):
    m._EMBEDDED_MODIFIERS[SRC] = payload()
    res = m.ArthemyKrea2PresetSaver().save_preset(
        FakePatcher(recipes), FakePatcher(), preset_name="p",
        subfolder_or_path=tempfile.mkdtemp(), embed_5d_modifiers=True, prune_5d_to_target=prune)
    path = res["result"][2]
    mod = list(json.load(open(path, encoding="utf-8"))["embedded_modifiers"].values())[0]
    keys = set(mod["layers"])
    return {
        "size": os.path.getsize(path),
        "keys": keys,
        "blocks": sorted({int(k.split(".")[1]) for k in keys if k.startswith("blocks.")}),
        "tensors": sorted({".".join(k.split(".")[2:]) for k in keys if k.startswith("blocks.")}),
        "clip": sum(1 for v in mod["layers"].values() if v["domain"] == "clip"),
        "loose": sorted(k for k in keys if k.startswith("txtfusion")),
    }


print("\n1. pruning off keeps everything (so the comparison below means something)")
full = save([recipe("All Blocks (0-27)")], prune=False)
check(len(full["keys"]) == TOTAL, f"all {TOTAL} layers kept", len(full["keys"]))

print("\n2. a whole-model recipe still drops the other DOMAIN")
r = save([recipe("All Blocks (0-27)")])
check(r["clip"] == 0, "no CLIP layers in a model-only preset", r["clip"])
check(len(r["loose"]) == 2, "block-less sections survive 'All Blocks'", r["loose"])

print("\n3. a block range keeps exactly that range")
r = save([recipe("Block_3 (All 10-14)")])
check(r["blocks"] == [10, 11, 12, 13, 14], "blocks 10-14 only", r["blocks"])
check(len(r["keys"]) == 5 * len(TENSORS), f"{5*len(TENSORS)} layers", len(r["keys"]))
check(r["loose"] == [], "block-less sections dropped: they belong to no block", r["loose"])
check(full["size"] / r["size"] > 5, "at least 5x lighter", full["size"] / r["size"])

print("\n4. sub_components narrows further")
r = save([recipe("Block_3 (All 10-14)", "MLP (Gate/Up/Down)")])
check(r["tensors"] == ["mlp.down", "mlp.gate", "mlp.up"], "MLP tensors only", r["tensors"])
check(len(r["keys"]) == 15, "15 layers", len(r["keys"]))

print("\n5. sub_tensor narrows to one tensor")
r = save([recipe("Block_3 (All 10-14)", "All Components", "MLP_down_proj")])
check(r["tensors"] == ["mlp.down"], "one tensor", r["tensors"])
check(len(r["keys"]) == 5, "5 layers", len(r["keys"]))
check(full["size"] / r["size"] > 40, "at least 40x lighter", full["size"] / r["size"])

print("\n6. a single sub-block keeps one block")
r = save([recipe("  ↳ Block_1C (2)")])
check(r["blocks"] == [2], "block 2 only", r["blocks"])

print("\n7. two nodes on the same modifier keep the UNION, never one of the two")
r = save([recipe("Block_1 (All 0-4)"), recipe("Block_5 (All 20-23)")])
check(r["blocks"] == [0, 1, 2, 3, 4, 20, 21, 22, 23], "both ranges present", r["blocks"])

print("\nALL PASS" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)

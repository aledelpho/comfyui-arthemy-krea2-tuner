"""End-to-end: Preset Saver writes a scoped payload, and the alias resolves back."""
import json, os, sys, tempfile
import stub_env, importlib
m = importlib.import_module("Arthemy_Krea2_Tuner")

FAIL = []
def check(cond, label, extra=""):
    (print(f"  ok   {label}") if cond else (FAIL.append(label), print(f"  FAIL {label} {extra}")))

class FakePatcher:
    def __init__(self, recipes): self.patches = {}; self.model_options = {"arthemy_5d_recipes": recipes}

def payload(order=8):
    layers = {}
    for b in range(28):
        for t in ("attn.wq", "attn.wk", "mlp.down"):
            layers[f"blocks.{b}.{t}"] = {"domain": "model", "d_out": 3072, "d_in": 3072,
                                         "singular_values": [1.0] * 5,
                                         "u_dct": [[0.01] * order] * 5, "v_dct": [[0.01] * order] * 5}
    for l in range(36):
        layers[f"text_model.encoder.layers.{l}.self_attn.q_proj"] = {
            "domain": "clip", "d_out": 1280, "d_in": 1280, "singular_values": [1.0] * 5,
            "u_dct": [[0.01] * order] * 5, "v_dct": [[0.01] * order] * 5}
    return {"version": "5.0", "storage": "dct", "source_lora": "Comics.safetensors",
            "rank": 5, "harmonic_order": order, "normalized": True, "layers": layers}

SRC = "Comics_5D.json"
m._EMBEDDED_MODIFIERS[SRC] = payload()      # stands in for the file on disk
recipe = {"domain": "model", "source_lora": SRC, "target_block": "  ↳ Block_3B (11)",
          "sub_components": "All Components", "sub_tensor": "ATTN_wq_query",
          "master_multiplier": 1.0, "dimension_weights": [1.0, .5, 0, 0, 0]}
out = tempfile.mkdtemp()

def run(prune):
    saver = m.ArthemyKrea2PresetSaver()
    res = saver.save_preset(FakePatcher([dict(recipe)]), FakePatcher([]),
                            preset_name=f"t_{int(prune)}", subfolder_or_path=out,
                            embed_5d_modifiers=True, prune_5d_to_target=prune)
    path = res["result"][2]
    return path, json.load(open(path, encoding="utf-8"))

print("\n1. pruning off (previous behaviour)")
p_off, d_off = run(False)
check(list(d_off["embedded_modifiers"]) == [SRC], "embedded under the plain name")
check(d_off["five_d_recipes"][0]["source_lora"] == SRC, "recipe still names the modifier")
check(len(d_off["embedded_modifiers"][SRC]["layers"]) == 120, "whole modifier embedded")

print("\n2. pruning on")
p_on, d_on = run(True)
alias = list(d_on["embedded_modifiers"])[0]
check(alias.startswith(SRC + m.SCOPE_ALIAS_SEP), "embedded under a scoped alias", alias)
check(d_on["five_d_recipes"][0]["source_lora"] == alias, "recipe re-pointed at the alias")
kept = d_on["embedded_modifiers"][alias]["layers"]
check(list(kept) == ["blocks.11.attn.wq"], "exactly the one tensor the recipe reaches", list(kept))
meta = d_on["embedded_modifiers"][alias]["arthemy_pruned"]
check(meta["kept_layers"] == 1 and meta["original_layers"] == 120, "prune metadata recorded", meta)
check(meta["scopes"][0]["sub_tensor"] == "ATTN_wq_query", "scope records the sub-tensor")
check(d_on["embedded_modifiers"][alias]["harmonic_order"] == 8, "top-level fields carried over")
a, b = os.path.getsize(p_off) / 1024.0, os.path.getsize(p_on) / 1024.0
check(b < a / 50, "preset is at least 50x smaller", f"{a:.0f} KB -> {b:.1f} KB")
print(f"     preset on disk: {a:.0f} KB -> {b:.1f} KB  ({a/b:.0f}x)")

print("\n3. the alias round-trips through a fresh session")
m._EMBEDDED_MODIFIERS.clear()
m.register_embedded_modifiers(d_on["embedded_modifiers"], origin="test preset")
check(m.get_cached_5d_dct(alias) is not None, "alias resolves to the scoped payload")
check(list(m.get_cached_5d_dct(alias)["layers"]) == ["blocks.11.attn.wq"], "with just its own layers")
check(m.get_cached_5d_dct(SRC) is None,
      "the plain name is NOT shadowed by the partial copy (it would resolve to the file on disk)")
m._EMBEDDED_MODIFIERS.clear()
m._EMBEDDED_MODIFIERS[SRC] = payload()
check(len(m.get_cached_5d_dct(alias)["layers"]) == 120,
      "an orphan alias falls back to the complete modifier")

print("\n4. a whole-model recipe still embeds normally")
wide = dict(recipe, target_block="All Blocks (0-27)", sub_tensor="All Sub-Tensors")
saver = m.ArthemyKrea2PresetSaver()
d = json.load(open(saver.save_preset(FakePatcher([wide]), FakePatcher([]), preset_name="t_wide",
                                     subfolder_or_path=out)["result"][2], encoding="utf-8"))
name = list(d["embedded_modifiers"])[0]
kept = d["embedded_modifiers"][name]["layers"]
check(len(kept) == 84 and all(e["domain"] == "model" for e in kept.values()),
      "every model layer kept, the unused CLIP half dropped", len(kept))
check(name.startswith(SRC + m.SCOPE_ALIAS_SEP),
      "still aliased: a payload missing its CLIP half is not the modifier on disk", name)

print("\n5. a recipe on each domain keeps both halves whole")
both = [dict(recipe, target_block="All Blocks (0-27)", sub_tensor="All Sub-Tensors"),
        dict(recipe, domain="clip", target_block="All Layers (0-59)", sub_tensor="All Sub-Tensors")]
d2 = json.load(open(saver.save_preset(FakePatcher([both[0]]), FakePatcher([both[1]]),
                                      preset_name="t_both", subfolder_or_path=out)["result"][2],
                    encoding="utf-8"))
check(list(d2["embedded_modifiers"]) == [SRC], "embedded under the plain name, unpruned",
      list(d2["embedded_modifiers"]))
check(len(d2["embedded_modifiers"][SRC]["layers"]) == 120, "all 120 layers")
check(all(r["source_lora"] == SRC for r in d2["five_d_recipes"]), "recipes untouched")

print("\n" + ("ALL PASS" if not FAIL else f"{len(FAIL)} FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)

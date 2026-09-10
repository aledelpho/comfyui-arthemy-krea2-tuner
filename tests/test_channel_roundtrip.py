"""A Channel Magnitude tuning must survive a preset save.

It is attached as a shared 1-D WeightAdapter, not as a tensor, so the Saver cannot serialize
it: without a recipe family the tuning vanished on save, and - worse - the Saver counted the
adapters as EXTERNAL LoRAs and reported "N LoRA tensors excluded" while writing an empty
preset. Both halves are pinned here.
"""
import json, sys, tempfile
import stub_env, importlib          # noqa: F401
m = importlib.import_module("Arthemy_Krea2_Tuner")

FAIL = []
def check(cond, label, extra=""):
    (print(f"  ok   {label}") if cond else (FAIL.append(label), print(f"  FAIL {label}   <- {extra}")))


class FakePatcher:
    def __init__(self, model_options=None, patches=None):
        self.patches = patches or {}
        self.model_options = model_options or {}


RECIPE = {"domain": "model", "mode": "Soft Value",
          "channel_path": m.CHANNEL_PATH_BOTH, "channel_scope": m.CHANNEL_SCOPE_DEFAULT,
          "bands": {"Band_01_Core_1pct": 0.30, "Band_12_Rare_100pct": -0.15}}

print("\n1. the family is declared, so Reset and Baker clear it too")
check("arthemy_channel_recipes" in m.ARTHEMY_RECIPE_KEYS, "arthemy_channel_recipes is a known family")

print("\n2. the Saver writes it")
out = tempfile.mkdtemp()
res = m.ArthemyKrea2PresetSaver().save_preset(
    FakePatcher({"arthemy_channel_recipes": [dict(RECIPE)]}), FakePatcher(),
    preset_name="chan", subfolder_or_path=out, embed_5d_modifiers=False)
data = json.load(open(res["result"][2], encoding="utf-8"))
check(len(data.get("channel_recipes", [])) == 1, "channel_recipes present", data.get("channel_recipes"))
got = (data.get("channel_recipes") or [{}])[0]
check(got.get("bands") == RECIPE["bands"], "band offsets survive verbatim", got.get("bands"))
check(got.get("channel_scope") == RECIPE["channel_scope"], "scope survives")
check(got.get("channel_path") == RECIPE["channel_path"], "path survives")
check(data["stats"].get("channel_recipes_count") == 1, "stats count it")

print("\n3. a channel-only graph is not reported as empty")
info = res["result"][3]
check("LoRA tensors excluded" not in info, "no spurious external-LoRA warning", info)

print("\n4. the Loader replays what the Saver wrote")
# The call itself needs a real model, so what is pinned here is the contract: the key the
# saver writes is the key the loader reads, and the node it replays through exists.
src = open(m.__file__.replace(".pyc", ".py"), encoding="utf-8").read()
check('data.get("channel_recipes"' in src, "loader reads the same key the saver writes")
check("ArthemyChannelMagnitudeTuner().tune_magnitude(" in src, "loader replays through the node")
check(hasattr(m.ArthemyChannelMagnitudeTuner, "tune_magnitude"), "that method exists")

print("\n5. an Arthemy granular adapter is never counted as an external LoRA")
class FakeAdapter:
    _is_arthemy_granular = True
scal, gran, lora, chaos = m.extract_patch_multipliers(
    FakePatcher(patches={"diffusion_model.blocks.0.attn.wq.weight": [(1.0, FakeAdapter())]}))
check(lora == 0, "lora_count stays 0", lora)
check(not scal and not gran, "and it is not mistaken for a scalar / granular patch")

print("\nALL PASS" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)

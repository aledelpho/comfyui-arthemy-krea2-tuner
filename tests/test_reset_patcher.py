"""The Reset Patcher clears its own work and nothing else.

Every assertion here corresponds to a line in ComfyUI's `comfy/model_patcher.py`:

  * `get_clone_model_override` (427-428) hands a clone the PARENT's model and the PARENT's
    `backup` dict, and `backup` is where the original weights live once ComfyUI has patched
    them in place. Rolling `patches_uuid` is what makes ComfyUI restore them (the
    `current_weight_patches_uuid != patches_uuid` test at 1258) - so clearing `backup` would
    leave that restore with nothing to restore from.
  * `clone()` shares `hook_backup` BY REFERENCE (492), and `unpatch_hooks()` (1682) writes
    into `self.model` with `copy_to_param` and then clears that shared dict.
  * nothing in this suite writes an `object_patch`.
"""
import sys, uuid
import stub_env, importlib          # noqa: F401
m = importlib.import_module("Arthemy_Krea2_Tuner")

FAIL = []
def check(cond, label, extra=""):
    (print(f"  ok   {label}") if cond else (FAIL.append(label), print(f"  FAIL {label}   <- {extra}")))


class FakePatcher:
    def __init__(self):
        self.patches = {"diffusion_model.blocks.0.attn.wq.weight": [(1.0, ("diff", 0.5))]}
        self.backup = {"diffusion_model.blocks.0.attn.wq.weight": "ORIGINAL"}
        self.object_patches = {"model_sampling": "someone else's"}
        self.hook_backup = {"diffusion_model.blocks.0.attn.wq.weight": "hook original"}
        self.model_options = {"arthemy_rotation_recipes": [{"a": 1}],
                              "arthemy_5d_recipes": [{"b": 2}, {"c": 3}],
                              "arthemy_channel_recipes": [{"d": 4}],
                              "unrelated_option": "keep me"}
        self.patches_uuid = uuid.uuid4()
        self.unpatch_hooks_called = 0

    def unpatch_hooks(self, whitelist_keys_set=None):
        self.unpatch_hooks_called += 1


p = FakePatcher()
before_uuid = p.patches_uuid
n_patches, n_recipes = m.ArthemyKrea2ResetPatcher._reset_patcher(p)

print("\n1. it clears what it owns")
check(p.patches == {}, "pending patches dropped", p.patches)
check(p.patches_uuid != before_uuid, "patches_uuid rolled, so ComfyUI unpatches and reloads")
check(all(rk not in p.model_options for rk in m.ARTHEMY_RECIPE_KEYS), "every recipe family gone",
      [rk for rk in m.ARTHEMY_RECIPE_KEYS if rk in p.model_options])

print("\n2. it leaves alone what belongs to ComfyUI or to other nodes")
check(p.backup == {"diffusion_model.blocks.0.attn.wq.weight": "ORIGINAL"},
      "backup kept - it is the ONLY way to undo an in-place patch", p.backup)
check(p.object_patches == {"model_sampling": "someone else's"},
      "object_patches kept (ModelSamplingAuraFlow, FreeU, ...)", p.object_patches)
check(p.hook_backup == {"diffusion_model.blocks.0.attn.wq.weight": "hook original"},
      "hook_backup kept - clone() shares it by reference", p.hook_backup)
check(p.unpatch_hooks_called == 0,
      "unpatch_hooks NOT called - it writes into the shared model", p.unpatch_hooks_called)
check(p.model_options.get("unrelated_option") == "keep me", "unrelated model_options kept")

print("\n3. it reports what it actually did")
check((n_patches, n_recipes) == (1, 4), "counts returned", (n_patches, n_recipes))
_, _, info = m.ArthemyKrea2ResetPatcher().reset(None, None, reset_model=False, reset_clip=False)
check("nothing was pending" in info, "an empty reset says so instead of 'clean'", info)

print("\nALL PASS" if not FAIL else f"\n{len(FAIL)} FAILED: {FAIL}")
sys.exit(1 if FAIL else 0)

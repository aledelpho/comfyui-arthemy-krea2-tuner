# Arthemy Krea-2 Suite — Architecture & Developer Notes

This document contains critical architectural decisions, memory management constraints, and engineering rules that govern the `Arthemy_Krea2_Tuner` codebase.

---

## 1. What This Package Is

A ComfyUI node suite that tunes a diffusion model **without retraining it**:
* **Scalar and per-channel gains**
* **Orthogonal rotations of dominant subspaces** (Lie algebra / SO(n) rotations)
* **5D Injections** (LoRAs compressed to top SVD directions with DCT compaction)

Everything is applied as **ComfyUI patches**, never by mutating the loaded checkpoint, and everything is reproducible from a lightweight JSON preset.

### Key Codebase Layout
* `Arthemy_Krea2_Tuner.py`: Node definitions, ComfyUI patch plumbing, presets, serialization, and visualizer backend.
* `arthemy_geometry_engine.py`: Pure-PyTorch numerics (SVD, Lie rotations, DCT, low-rank factorizations). Deliberately decoupled from ComfyUI.
* `web/js/arthemy_visualizer.js`: Interactive canvas panels and live graphical widgets on ComfyUI nodes.
* `presets/`: Reference tuning presets and reconstructions.
* `tests/`: Offline and PyTorch-based test suite.

---

## 2. Non-Negotiable Rules & Invariants

### Never mutate a shared module
A node clones the patcher and attaches patches. The only place that writes into the module tree is `isolate_and_assign_baked_weights`, which explicitly clones every level down the hierarchy.

### Never clear a patch you did not fold in
Losing a user's active tuning silently is unacceptable. `tests/test_baker.py` ensures that if a patch cannot be folded, it is preserved rather than dropped.

### A state_dict key IS the module path
Do not run a key through `clean_key` before walking the module tree: `clean_key` strips prefixes like `diffusion_model.` or the text-encoder wrapper, which represents the root hop in the model hierarchy.

### `WeightAdapter.weights` is weight data, nothing else
ComfyUI reconstructs an adapter as `type(value)(value.loaded_keys, value.weights)` (`comfy/lora.py::prefetch_prepared_value`) and gathers every tensor in `weights` into one VRAM buffer copied asynchronously.
* The adapter constructor must read **no value** out of `weights` during construction.
* Configuration data (such as the target axis) must live in the class definition (e.g. `ArthemyChannelRowScaleAdapter` vs `ArthemyChannelColScaleAdapter`).

### Every tuning needs a recipe family
A tuning attached as an adapter rather than a static delta tensor cannot be directly serialized by raw tensor dumps. It must:
1. `append_recipe` into a family listed in `ARTHEMY_RECIPE_KEYS`.
2. The Saver must collect that family.
3. The Loader must replay it.
`tests/test_node_contracts.py` enumerates and verifies all recipe families.

### The Preset Loader honours node-level rules
The Preset Loader must respect `is_skippable_for_tuning`. A preset saved on BF16 and replayed on scaled FP8 must never double-scale layers that other nodes refuse to touch.

### Recipe replays call nodes dynamically
Preset recipes are replayed by calling nodes. Any class or method renaming must be mirrored in the replay dispatch, or the Loader will raise `NameError`. `tests/test_node_contracts.py` walks `load_preset` and asserts that every replayed node exists.

### `object_patches` belongs to external nodes
No node in this suite writes to `object_patches`. `Reset` and `Baker` must never clear it (doing so would reset third-party nodes like `ModelSamplingAuraFlow` or `FreeU`).

### ModelPatcher clone semantics
`get_clone_model_override` (`comfy/model_patcher.py`) shares the parent's `model` and `backup` dict; `clone()` shares `hook_backup` by reference.
* **Reset must NOT clear `backup`**: That dict holds the original weights when ComfyUI patches in-place. Rolling `patches_uuid` triggers ComfyUI to restore them. Clearing it leaves restore with nothing, fusing the tuning permanently into memory.
* **Never call `unpatch_hooks()`**: It writes into `self.model` with `copy_to_param` and clears the shared `hook_backup`, affecting all branches of the graph.
* **Baker exception**: The Baker replaces `patcher.model` with a freshly cloned module tree; therefore, it MUST clear `backup` so ComfyUI does not overwrite newly baked weights with pre-bake originals.

### Report what happened, not what was attempted
`add_patches` silently drops keys that do not exist in the target model. Always count its return value.

---

## 3. Porting to a Different Base Model

When adapting this suite to other architectures (e.g. Z-Image, SDXL, Flux, or other DiT variants), work through the following checklist:

1. **Residual width**: `Krea2Config.HIDDEN_SIZE = 6144` is only the fallback. `Krea2Config.probe_hidden_size()` probes the checkpoint (most frequent dimension across 2-D weights with >= 2 sightings).
2. **Cached channel profiles**: Stored in `models/arthemy_profiles/channel_*.json`. Keyed by matrix fingerprints. If the profiling metric changes, bump `CHANNEL_PROFILE_FORMAT` to invalidate old caches safely.
3. **Block and layer maps**: Review `MODEL_TARGET_MAP`, `CLIP_TARGET_MAP`, `MODEL_BLOCK_LABEL_TO_IDX`, `CLIP_LAYER_LABEL_TO_IDX`, `MODEL_SECTION_KEYS`, `CLIP_SECTION_KEYS`.
4. **CLIP dual-tower indexing**: Text layers are indices 0–35; Qwen3-VL visual tower is mapped at +36 (`Visual_1A` is index 36). Review `Krea2TensorParser.extract_clip_layer_idx`, `RE_CLIP_VISUAL_BLOCKS`, and `probe_architecture`.
5. **Tensor naming**: `Krea2TensorParser.MODEL_SURGEON_MAP` and `CLIP_SURGEON_MAP` define sub-tensors (`attn.wq`, `mlp.down`, etc.).
6. **Quantization companions**: `companion_scale_candidates()` identifies scaled-FP8 companion scales (`<layer>.scale_weight` on disk and `<layer>.weight_scale` in ComfyUI memory).
7. **Block-less sections**: Prefixes `txtfusion`, `tmlp`, `first`, `last`, `tproj` have no block index; they participate only in "All Blocks" selections.

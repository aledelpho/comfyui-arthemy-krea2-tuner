# Arthemy Krea-2 Tuner Suite for ComfyUI

[![Version: 2.0.0](https://img.shields.io/badge/Version-2.0.0-purple.svg)](https://github.com/aledelpho/comfyui-arthemy-krea2-tuner)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom--Node-blue.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Tests: Passing](https://img.shields.io/badge/Tests-11%2F11%20Passing-brightgreen.svg)](tests/)

> **Release 2.0**: The comprehensive major release featuring zero-retraining geometric subspace steering, Lie algebra rotations, channel profiling, self-contained 5D LoRA compression, native memory safety, and interactive template workflows.

A high-precision ComfyUI node suite designed for fine-grained tuning, geometric subspace steering, and lightweight checkpoint manipulation for **Krea-2** and modern diffusion models **without retraining**.

---

## Highlights

* **Zero-Retraining Steering**: Steer diffusion aesthetics, composition, and prompt alignment using analytical geometry and spectral analysis.
* **Native Memory Safety**: Everything is applied as native ComfyUI patches. Base checkpoints remain strictly immutable in memory until explicitly saved or baked.
* **Lie Algebra Orthogonal Rotations**: Rotate dominant SVD weight subspaces in $SO(n)$ without altering the Frobenius norm of the matrix.
* **Channel Magnitude Profiling**: Dynamically profile the model's residual stream and apply direction-aware row/column gain vectors with microscopic overhead (~24 KB vs gigabytes of dense deltas).
* **5D LoRA Compression**: Compress dense LoRA updates into principal SVD directions with Discrete Cosine Transform (DCT) compaction, reducing preset footprints by up to 70x.
* **In-Place BF16/FP8 Baker & Savers**: Stream and fold active patches directly into clean checkpoints without triggering out-of-memory (OOM) errors or `ModelPatcherDynamic` assertion failures.
* **On-Node Interactive Visualizers**: Real-time canvas widgets on LiteGraph nodes that plot gain distributions and rotation angles directly in the ComfyUI interface.

---

## Node Catalog

Every node clones the ComfyUI patcher and attaches patches, so any number of them can be
chained in any order. Each one returns an `info` string ending in `Patches: N` — if `N` is 0,
that node did nothing, and the string says why.

Colour prefixes: 🟪 model · 🟨 text encoder · 🩷 LoRA.

### 🟪 Model Tools (diffusion backbone)

| Node | What it gives you |
|---|---|
| **🟪✨ Model Tuner** | One multiplier per section: `Text_Fusion`, `Time_Embed`, `Projection`, and `Block_1..6` = blocks 0-4, 5-9, 10-14, 15-19, 20-23, 24-27. `Soft Value` damps the slider to 10% (1.00 → ×1.10), `Real Value` is 1:1. |
| **🟪🔬 Model Sub-Block Tuner** | The same, one tensor family at a time: WQ/WK/WV/WO, attention gate, QK-norm scales, SwiGLU gate/up/down, time modulation, block norms — inside a block group or one single block. |
| **🟪🌪️ Model Sub-Block Chaos Tuner** | Reproducible pseudo-random perturbation of the families you choose. Per-family probability sliders, `chaos_strength`, and a CRC32 seed that reproduces across restarts. Rolled once per module, so paired tensors are never split. |
| **🟪📊 Channel Magnitude Tuner** | A *vertical* cut: ranks the residual-stream channels by energy, splits them into 12 logarithmic bands and scales each one. `channel_scope` picks the sub-network (MLP by default), `channel_path` picks whether the gain lands where it reads from the residual, where it writes back, or both. The ranking is profiled from the checkpoint and cached, so only the first move costs anything. |
| **🟪🌿 Model Axis Rotator** | Four independent rotation dials (±180°) inside each weight's dominant subspace — local in-plane, branch-into-trunk, long-range channel phase, and mirrored pairwise. Orthogonal, so the Frobenius norm is unchanged: it moves what a layer *means*, not how loud it is. `depth_reach` sets the rotated rank (8 / 16 / 32). |
| **🟪🌀 Model Chaos Rotator** | The same rotation, with random per-plane angles instead of dials. `chaos_strength` caps the angle as a fraction of 90°, `harmonic_coherence` snaps angles towards 45° multiples, the seed reproduces exactly. |
| **🟪🧭 Model Compass Rotator** | Rotation along one *chosen bearing*: `style_direction` (0-360°) blends continuously between local and long-range planes, `rotation_angle` (-90..+90) is how far you travel, `manifold` picks the output or the input space. Nearby bearings give related styles, so it can be swept like a map. |
| **🟪🧬 5D Model Tuner** | Injects a compressed LoRA back as a native low-rank patch, one SVD direction per slider, restricted to any block / component / single tensor. |

### 🟨 CLIP Tools (Qwen3-VL text encoder)

| Node | What it gives you |
|---|---|
| **🟨✨ CLIP Tuner** | `Embedding` plus seven layer groups: text layers 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-35. The vision tower (36-59) is deliberately left alone — it only runs when reference images are fed to the encoder. |
| **🟨🔬 CLIP Sub-Block Tuner** | Q/K/V/O projections, query norm, MLP gate/up/down and layer norms, inside a layer group or one single layer. Visual layers 36-59 are addressable here. |
| **🟨🌪️ CLIP Sub-Block Chaos Tuner** | Reproducible perturbation of the same families. The failure mode to watch for is the prompt being ignored, not the image degrading — keep `chaos_strength` around 0.05-0.15. |
| **🟨🌿 CLIP Axis Rotator** | The four rotation dials, on the text encoder. Far more sensitive than the model: 5-10° is already a lot. |
| **🟨🌀 CLIP Chaos Rotator** | Random-angle rotation of text-encoder subspaces: a different reading of the same prompt, at unchanged strength. |
| **🟨🧭 CLIP Compass Rotator** | Bearing-based rotation on the text encoder. |
| **🟨🧬 5D CLIP Tuner** | 5D injection into the text encoder. |

### 🩷 LoRA Tools

| Node | What it gives you |
|---|---|
| **🩷🔮 LoRA Block Loader** | Loads a LoRA with one keep/drop slider per block group. Keep a style LoRA only on blocks 24-27 and most of its character bleed goes away. |
| **🩷🔬 Load Sub-Block LoRA** | The same, per tensor family: attention only (composition without texture), MLP only (surface without layout). |
| **🩷🌪️ Load Sub-Block Chaos LoRA** | Loads a random subset of the LoRA's tensors, so it blends in without the usual all-or-nothing signature. `base_chance` plus per-family overrides, seeded. |
| **🩷🧬 LoRA-to-5D Extractor** | Compresses a LoRA into its N dominant SVD directions (1-8) as a portable `.json` modifier. `storage=auto` measures the fidelity a DCT would actually reach on that LoRA's tensor sizes and falls back to exact raw factors below 35%. It is an output node, so bypass it (Ctrl+B) once the modifier exists. |

### 🟪🟨 Presets, Savers, Baker & Utilities

| Node | What it gives you |
|---|---|
| **🟪🟨💾 Preset Saver** | Serialises every active tuning: scalar and granular patches, plus reproducible recipes for Chaos, Rotations, Channel Magnitude and 5D. `prune_5d_to_target` embeds only the layers each 5D node can actually reach — a 5D aimed at one block group makes the preset ~6.5× smaller, aimed at a single tensor ~48×. |
| **🟪🟨📂 Preset Loader** | Replays a preset by re-running the nodes that made it, with a global `strength_model` / `strength_clip` on top. A malformed recipe is skipped with a warning instead of taking the whole preset down. |
| **🟪🟨🔄 Reset Patcher** | Drops this suite's pending patches and recipes and rolls the patch id so ComfyUI restores anything it had already written in place. It deliberately leaves `object_patches`, hook patches and ComfyUI's own weight backup alone — FreeU, ModelSamplingAuraFlow and friends survive a reset. |
| **🟪📊 Model Visualizer / 🟨📊 CLIP Visualizer** | On-canvas plot of what is actually patched, per block or per layer, measured from the patcher rather than from the widget values. |
| **🟪💾 Model Saver / 🟨💾 CLIP Saver** | Stream a full `.safetensors` checkpoint of the tuned weights in BF16 or FP8_E4M3. They read patched weights directly, so there is no need to bake first. |
| **🟪🟨 Model Baker** | Folds every pending patch into a freshly cloned module tree in memory — no file written, and the base checkpoint is never mutated. One-way for the session: baked weights *are* the weights. |

> **Quantized checkpoints.** Every tuner refuses layers that carry a scaled-FP8 companion
> scale — patching them would apply that scale twice — and reports how many it skipped. The
> Savers handle those correctly by folding the scale into the weight.

---

## How to work with it

1. Fix the seed and the prompt, generate once. That image is your reference.
2. Add **one** node and move **one** widget. Generate again.
3. Read the node's `info` output. It ends with `Patches: N`; `N = 0` means nothing was applied,
   and the message says why (wrong block selection, quantized layers, a LoRA that carries no
   such tensor).
4. Keep what works and save it with the **Preset Saver**. Presets are text: they can be
   re-loaded at a different strength, stacked, or edited by hand.

Order matters only between a *scaling* node (Tuners, LoRA loaders) and an *additive* one
(Rotators, 5D). Two scalings commute; a scaling and a rotation do not.

---

## Installation

### Method 1: ComfyUI Manager (Recommended)
Search for `Arthemy Krea-2 Tuner` in the ComfyUI Manager and click **Install**.

### Method 2: Git Clone
Navigate to your ComfyUI `custom_nodes` directory:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Arthemy/ComfyUI-Arthemy-Krea2-Tuner.git
```
Install the lightweight dependencies (torch, safetensors, numpy, pillow are already bundled with standard ComfyUI environments):
```bash
pip install -r ComfyUI-Arthemy-Krea2-Tuner/requirements.txt
```

### Method 3: Comfy CLI
```bash
comfy node install arthemy-krea2-tuner
```

---

## Presets & Example Workflow

A preset is a small JSON file that records everything the suite is doing to a model, and
replays it. Presets live **inside this node's own folder**
(`custom_nodes/Arthemy_Krea2_Tuner/presets/`), so nothing is scattered through your root
`models/` directory: the **🟪🟨💾 Preset Saver** writes there and the **🟪🟨📂 Preset Loader**
reads from there.

What a preset carries:

* every scalar and granular patch, per tensor;
* reproducible recipes for Chaos perturbations, subspace rotations, Channel Magnitude bands
  and 5D injections — replayed by re-running the node that made them, not by storing tensors;
* optionally the 5D modifiers themselves (`embed_5d_modifiers`), so the preset reproduces on a
  machine that has never seen the source LoRA. With `prune_5d_to_target` on it embeds only the
  layers each 5D node can actually reach, which is what keeps that self-contained copy small.

Loading one is not the end of the chain: `strength_model` / `strength_clip` scale the whole
preset (0.50 for half, 1.50 to push it, negative to invert it), and you can keep tuning after
it. Presets are text, so they can also be edited by hand.

Shipped with the repository:

* **`presets/Arthemy_Comics_Preset.json`** — the Arthemy Comics tuning. Block and layer gains
  across the diffusion model and the Qwen3-VL text encoder, four style-compass rotations on
  blocks 10-14, and six scoped 5D injections carried inside the file, so it reproduces on a
  machine that has never seen the source LoRAs. It is a large file for that reason: the tuning
  itself is about 50 KB, the embedded 5D payloads are the rest.

A complete graph **is** included, in **[`example_workflows/`](example_workflows/)**: every node
in the suite, wired and laid out, with a note next to each one explaining what it does and how
to drive it. Drag the `.json` onto the ComfyUI canvas to open it.

---

## Developer Guide & Testing

The codebase includes an extensive offline and PyTorch-based test suite in `tests/`:

* **Contract Validation**: Ensures node inputs, outputs, defaults, and preset replay dispatches are consistent.
* **Memory Invariance**: Verifies that ComfyUI patcher backups are never corrupted.
* **Numerical Parity**: Validates that lightweight channel adapters match exact dense delta computations.

Seven of the eleven suites are pure Python and run anywhere (`python3 tests/test_node_contracts.py`);
the four that need real tensors skip themselves when torch is missing, so run the whole set with
ComfyUI's own interpreter:
```powershell
Get-ChildItem -Path tests -Filter "test_*.py" | ForEach-Object {
    & "path/to/comfyui/python.exe" $_.FullName
}
```

For detailed architectural notes, memory management invariants, and porting instructions to other architectures, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## License

This project is licensed under the [MIT License](LICENSE).

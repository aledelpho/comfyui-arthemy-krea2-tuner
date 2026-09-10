<p align="center">
  <img src="assets/Generation.webp" width="900" alt="Arthemy Krea-2 Tuner Generation Pipeline" />
</p>

<h1 align="center">Arthemy Krea-2 Tuner Suite</h1>

<p align="center">
  Zero-retraining weight steering, geometric subspace rotations, channel profiling, and 5D LoRA compaction for <b>Krea-2</b> and modern diffusion models in ComfyUI.
</p>

<p align="center">
  <a href="https://github.com/aledelpho/comfyui-arthemy-krea2-tuner/releases/tag/v2.0.0"><img alt="Version 2.0.0" src="https://img.shields.io/badge/version-2.0.0-purple?style=flat-square"></a>
  <img alt="ComfyUI custom node" src="https://img.shields.io/badge/ComfyUI-custom--node-6b46c1?style=flat-square">
  <img alt="Compatibility" src="https://img.shields.io/badge/compatible-Krea--2%20%7C%20Qwen3-1e88e5?style=flat-square">
  <a href="tests/"><img alt="Tests Passing" src="https://img.shields.io/badge/tests-11%2F11%20passing-brightgreen?style=flat-square"></a>
  <a href="#license"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green?style=flat-square"></a>
  <img alt="GitHub stars" src="https://img.shields.io/github/stars/aledelpho/comfyui-arthemy-krea2-tuner?style=flat-square">
</p>

---

> 🚀 **Version 2.0 Release**: A major evolution featuring zero-retraining Lie Algebra subspace rotations ($SO(n)$), residual channel magnitude profiling, 5D LoRA compression with scoped self-contained presets, native memory safety for `ModelPatcherDynamic`, and an offline automated test suite.

Edit a **Krea-2** model and its **Qwen3** text encoder by modulating internal weights, rotating singular subspaces, and profiling residual streams live in memory — **no training and no datasets required**. Move a dial, generate, and observe immediate stylistic and semantic shifts. When you land on a result you like, export it as a lightweight preset or bake it into a standalone `.safetensors` checkpoint.

<p align="center">
  <a href="assets/examples/XYBenchmark.webp">
    <img src="assets/examples/XYBenchmark.webp" width="850" alt="X/Y Tuning Benchmark Grid" />
  </a>
  <br>
  <sub><i>How different prompts behave under varying tuning configurations.</i></sub>
</p>

---

## Contents

- [Highlights](#highlights)
- [Installation](#installation)
- [Included Template Workflow](#included-template-workflow)
- [Node Catalog & Deep Dive](#node-catalog--deep-dive)
  - [🟪 Model Tools (Diffusion Backbone)](#-model-tools-diffusion-backbone)
  - [🟨 CLIP Tools (Qwen3-VL Text Encoder)](#-clip-tools-qwen3-vl-text-encoder)
  - [🩷 LoRA Tools & 5D Compaction](#-lora-tools--5d-compaction)
  - [🟪🟨 Presets, Savers & Baker](#-presets-savers--baker)
  - [Summary Architecture Matrix](#summary-architecture-matrix)
- [Reading the Controls](#reading-the-controls)
- [Presets & Example Presets](#presets--example-presets)
- [Visualizing Active Patches](#visualizing-active-patches)
- [Visual Tuning Showcase & Effects Gallery](#visual-tuning-showcase--effects-gallery)
  - [Model Tuning Effects (Blocks 1–6, Projections, Embeddings)](#-model-tuning-effects-blocks-16-text-fusion-time-embed-projection)
  - [CLIP Tuning Effects (Layers 1–7, Embedding)](#-clip-text-encoder-tuning-effects-layers-17-embedding)
  - [Deterministic Multi-Seed Persistence](#deterministic-multi-seed-persistence)
- [Developer Guide & Testing](#developer-guide--testing)
- [Troubleshooting](#troubleshooting)
- [Author & License](#author--license)

---

## Highlights

* **Zero-Retraining Steering**: Steer aesthetics, composition, and prompt fidelity through linear algebra, SVD subspace rotations, and residual channel profiling.
* **Native Memory Safety**: Applies non-destructive patches via ComfyUI's native patcher. Base checkpoints remain strictly immutable in memory until explicitly saved or baked.
* **Lie Algebra Orthogonal Rotations**: Rotate dominant SVD weight subspaces in $SO(n)$ without altering the Frobenius norm of the matrix.
* **Channel Magnitude Profiling**: Dynamically profile the model's residual stream and apply direction-aware row/column gain vectors with microscopic overhead (~24 KB vs gigabytes of dense deltas).
* **5D LoRA Compression**: Compress dense LoRA updates into principal SVD directions with Discrete Cosine Transform (DCT) compaction, reducing preset footprints by up to 70x.
* **In-Place BF16/FP8 Baker & Savers**: Stream and fold active patches directly into clean checkpoints without triggering out-of-memory (OOM) errors or `ModelPatcherDynamic` assertion failures.
* **On-Node Interactive Visualizers**: Real-time canvas widgets on LiteGraph nodes that plot gain distributions and rotation angles directly in the ComfyUI interface.

---

## Installation

### Method 1: ComfyUI Manager (Install via Git URL)
In **ComfyUI Manager**, click **Install via Git URL**, paste:
```text
https://github.com/aledelpho/comfyui-arthemy-krea2-tuner.git
```
and click **Install**.

### Method 2: Manual Installation (Git Clone)
Navigate to your ComfyUI `custom_nodes` folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/aledelpho/comfyui-arthemy-krea2-tuner.git
```
Dependencies (`torch`, `safetensors`, `numpy`, `Pillow`) are bundled by default with modern ComfyUI installations. If needed:
```bash
pip install -r comfyui-arthemy-krea2-tuner/requirements.txt
```
*Restart ComfyUI completely afterward.*

---

## Included Template Workflow

The suite includes a ready-to-run, modular reference workflow located in [`example_workflows/Arthemy_Krea2_Tuner_Workflow.json`](example_workflows/Arthemy_Krea2_Tuner_Workflow.json):

<p align="center">
  <img src="assets/Workflow.webp" width="900" alt="Arthemy Krea-2 Tuner Sandbox Workflow" />
</p>

### How to use the sandbox:
1. **Load the template:** Drag and drop `example_workflows/Arthemy_Krea2_Tuner_Workflow.json` onto the ComfyUI canvas.
2. **Left area:** Checkpoint loader, prompt encode, latent dimensions, and the **Reset Patcher**.
3. **Workspace slot:** Copy and paste any tuning tool from the organized banks below and chain it directly between the **Reset Patcher** and the **Visualizers**.
4. **Bottom banks:** All nodes categorized by colour: 🟪 Model Tuners · 🟨 CLIP Tuners · 🩷 LoRA & 5D · 🟪🟨 Rotators · Presets & Savers. Each node carries a dedicated README note explaining its exact inputs.
5. **Tune & Cook:** Adjust your sliders, observe live patch waveforms on the visualizers, and queue the generation!

---

## Node Catalog & Deep Dive

Every node clones the ComfyUI patcher and attaches patches in-place. They can be chained in any sequence. Each node outputs an `info` string ending in `Patches: N` confirming the exact count of modified tensors.

<p align="center">
  <img src="assets/ModelTuner.webp" width="650" alt="Model Tuner Nodes" />
</p>

### 🟪 Model Tools (Diffusion Backbone)

#### 1. 🟪✨ Model Tuner (Group-Level)
Provides one scalar multiplier per major section: `Text_Fusion`, `Time_Embed`, `Projection`, and `Block_1` through `Block_6` (blocks 0-4, 5-9, 10-14, 15-19, 20-23, 24-27). Moving a slider scales every tensor inside that group uniformly.
* **Workflow:** `MODEL` → `Model Tuner` → `KSampler`.

<p align="center">
  <a href="assets/examples/Block_3.webp">
    <img src="assets/examples/Block_3.webp" width="800" alt="Block Tuning Showcase - Block 3" />
  </a>
  <br>
  <sub><i>Block Tuning — Modulating an entire block (Block_3) across different values (-2.0 to +2.0) demonstrates how group-level tuning shapes visual structures and features.</i></sub>
</p>

#### 2. 🟪🔬 Model Sub-Block Tuner (Granular Tensor Type)
Narrows target execution: select a block group or a single block (`target_block`), then adjust fine-grained sliders for internal tensor families: `ATTN_wq_query`, `ATTN_wk_key`, `ATTN_wv_value`, `ATTN_wo_out`, `ATTN_q_norm`, `MLP_gate_swiglu`, `MLP_up_proj`, `MLP_down_proj`, `NORMS_block_scales`, etc.
* **Use case:** When a block improves layout but degrades texture, isolate the attention or feed-forward paths independently.

<p align="center">
  <a href="assets/examples/sub-block-Tuning.webp">
    <img src="assets/examples/sub-block-Tuning.webp" width="800" alt="Sub-Block Tuning Variations inside Block 3" />
  </a>
  <br>
  <sub><i>Sub-Block Tuning — Modulating specific sub-blocks within Block 3 influences fine details and layer-level characteristics.</i></sub>
</p>

#### 3. 🟪🌪️ Model Sub-Block Chaos Tuner (Seeded Perturbations)
Stochastic perturbation of chosen tensor families: each matched tensor is perturbed with probability `chance` by `chaos_strength`. Seeded with a 32-bit CRC seed to ensure identical reproductions across restarts.
* **Block-Level:** Perturbs entire module tensors coherently.
* **Element-Level (Sub-atomic):** Generates per-weight noise for textured variations.

<p align="center">
  <a href="assets/examples/chaos-block-Tuning.webp">
    <img src="assets/examples/chaos-block-Tuning.webp" width="800" alt="Chaos Tuning Variations" />
  </a>
  <br>
  <sub><i>Chaos Tuning — Seeded stochastic perturbations applied across sub-blocks introduce controlled stylistic variations.</i></sub>
</p>

#### 4. 🟪📊 Channel Magnitude Tuner (Vertical Energy Cuts)
A vertical rather than horizontal cut: profiles the residual stream ($d_{model} = 6144$ on Krea-2), ranks channels logarithmically by energy, cuts them into 12 logarithmic frequency bands, and applies direction-aware scaling.
* `channel_scope`: targets MLP or Attention sub-networks.
* `channel_path`: applies gains where the network reads from residual, writes back, or both.
* **Ultra-low footprint:** Generates ~24 KB rank vectors instead of multi-gigabyte dense matrices. Profiles are cached in `models/arthemy_profiles/`.

#### 5. 🟪🌿 Model Axis Rotator (Lie Algebra $SO(n)$)
Applies 4 independent rotation dials (±180°) inside the dominant SVD subspaces of the target weights: local in-plane, branch-into-trunk, long-range channel phase, and mirrored pairwise.
* **Norm Invariant:** Rotates within the special orthogonal group $SO(n)$, preserving the exact Frobenius norm of the base checkpoint. Moves what a layer *means*, not how loud it is.
* `depth_reach`: selects rotation rank (8, 16, or 32).

#### 6. 🟪🌀 Model Chaos Rotator & 🟪🧭 Model Compass Rotator
* **Chaos Rotator:** Rotates subspaces along pseudo-random angles with `harmonic_coherence` snapping toward 45° multiples.
* **Compass Rotator:** Rotates weights along a chosen continuous bearing `style_direction` (0–360°), allowing smooth spherical sweeps across stylistic manifolds.

#### 7. 🟪🧬 5D Model Tuner
Injects compressed LoRA representations directly back into diffusion layers as native low-rank patches, adjustable slider-by-slider for each dominant SVD direction.

---

### 🟨 CLIP Tools (Qwen3-VL Text Encoder)

<p align="center">
  <img src="assets/ClipTuner.webp" width="650" alt="CLIP Tuner Nodes" />
</p>

Tuning applied to the Qwen3-VL text encoder provides fine-grained steering over prompt comprehension and linguistic interpretation:
* **🟨✨ CLIP Tuner:** Group sliders covering `Embedding` and `Layer_1` through `Layer_7` (text layers 0-4 through 30-35).
* **🟨🔬 CLIP Sub-Block Tuner:** Granular control over `ATTN_q_proj`, `k_proj`, `v_proj`, `o_proj`, query norms, and MLP projections inside specific layers.
* **🟨🌪️ CLIP Sub-Block Chaos Tuner:** Seeded stochastic perturbations on linguistic projections.
* **🟨🌿 CLIP Axis Rotator:** Subspace Lie rotation dials on the text encoder (sensitive: 5°–10° produces noticeable semantic re-interpretations).
* **🟨🌀 CLIP Chaos Rotator & 🟨🧭 CLIP Compass Rotator:** Continuous bearing and random-plane subspace rotations on text embeddings.
* **🟨🧬 5D CLIP Tuner:** Re-injects compressed 5D LoRA adapters into linguistic layers.

---

### 🩷 LoRA Tools & 5D Compaction

<p align="center">
  <img src="assets/LoraTuner.webp" width="650" alt="LoRA Loader Nodes" />
</p>

* **🩷🔮 LoRA Block Loader:** Drop-in replacement for standard LoRA loaders with independent block-group multipliers. Isolating a style LoRA to blocks 24–27 preserves style while eliminating character bleed.
* **🩷🔬 Load Sub-Block LoRA:** Applies LoRA weights strictly to selected tensor families (e.g. attention only for composition, MLP only for texture).
* **🩷🌪️ Load Sub-Block Chaos LoRA:** Stochastically samples LoRA keys according to per-family probability sliders.
* **🩷🧬 LoRA-to-5D Extractor:** Compresses any external LoRA checkpoint into its $N$ dominant SVD directions (1–8) and exports a compact `.json` modifier into `models/arthemy_modifiers/`. Features automatic fidelity measurement with DCT compression fallback.

---

### 🟪🟨 Presets, Savers & Baker

#### 1. Presets (Lightweight & Self-Contained)
<p align="center">
  <img src="assets/Preset.webp" width="550" alt="Preset System" />
</p>

* **🟪🟨💾 Preset Saver:** Serializes all active scalar, granular, Chaos, Rotation, Channel Magnitude, and 5D configurations into a clean JSON preset located in `custom_nodes/Arthemy_Krea2_Tuner/presets/`.
  * `prune_5d_to_target`: When enabled, strips unused layers from embedded 5D modifiers, reducing file sizes from ~50x to ~70x.
* **🟪🟨📂 Preset Loader:** Replays presets with master `strength_model` and `strength_clip` multipliers. Safely skips incompatible scaled-FP8 companion scales.
* **Shipped Reference Preset:** Includes **`presets/Arthemy_Comics_Preset.json`**, delivering the complete Arthemy Comics aesthetic with embedded self-contained 5D payloads.

#### 2. Model Baker, Savers & Reset Patcher
<p align="center">
  <img src="assets/Saver.webp" width="450" alt="Model and CLIP Savers" />
</p>

* **🟪🟨 Model Baker:** Folds all pending in-memory patches directly into weight matrices in RAM. Leaves base checkpoints unharmed and enables downstream nodes to consume plain weights without patch overhead.
* **🟪💾 Model Saver / 🟨💾 CLIP Saver:** Directly stream full `.safetensors` checkpoints in **BF16** or **FP8_E4M3** without requiring pre-baking. Correctly fuses FP8 companion scales into quantized tensors.
* **🟪🟨🔄 Reset Patcher:** Safely drops pending suite patches and rolls patch IDs to force ComfyUI clean reloading. Leaves third-party hooks and `object_patches` (FreeU, AuraFlow) completely intact.

---

### Summary Architecture Matrix

| Domain | Group Level | Granular / Sub-Block | Subspace Rotations ($SO(n)$) | Stochastic Discovery | 5D Compaction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Diffusion Model** | `Model Tuner` | `Model Sub-Block Tuner` <br> `Channel Magnitude Tuner` | `Model Axis Rotator` <br> `Model Compass Rotator` | `Model Sub-Block Chaos` <br> `Model Chaos Rotator` | `5D Model Tuner` |
| **Qwen3 CLIP** | `CLIP Tuner` | `CLIP Sub-Block Tuner` | `CLIP Axis Rotator` <br> `CLIP Compass Rotator` | `CLIP Sub-Block Chaos` <br> `CLIP Chaos Rotator` | `5D CLIP Tuner` |
| **LoRA** | `LoRA Block Loader` | `Load Sub-Block LoRA` | — | `Load Sub-Block Chaos LoRA` | `LoRA-to-5D Extractor` |
| **Runtime & Disk** | `Preset Loader` | `Preset Saver` | `Reset Patcher` | `Model Baker` (RAM) | `Model / CLIP Savers` (Disk) |

---

## Reading the Controls

Slider controls represent offsets relative to base weights (`0.00` = neutral/unmodified):

| Mode | Formula | Application |
| :--- | :--- | :--- |
| **Soft Value** | `1.0 + (slider × 0.10)` | Delicate calibration — slider `1.0` applies +10% gain *(Recommended default)*. |
| **Real Value** | `1.0 + slider` | Direct multiplier — slider `1.0` doubles weight (+100%). For bold stylistic interventions. |

* **Rotation Dials:** Expressed in degrees (±180°). In CLIP, start between 5°–15°; in diffusion models, 10°–30° provides distinct stylistic shifts.
* **Channel Magnitude Bands:** Band 1 controls the top 1% energy channels (macro composition), while Band 12 governs the fine 18% tail (grain, micro-details).

---

## Visualizing Active Patches

<p align="center">
  <img src="assets/Preset.webp" width="550" alt="Visualization and Diagnostic Tools" />
</p>

* **🟪📊 Model Visualizer / 🟨📊 CLIP Visualizer:** Connect inline to inspect real-time bar graphs of patch magnitudes directly on the node canvas.
* Validates that patches have successfully attached to weights before spending GPU cycles on sampling.

---

## Visual Tuning Showcase & Effects Gallery

Visual reference gallery demonstrating how modulating individual Model blocks and CLIP layers steers generation outputs in isolation.

### Benchmark Baseline Prompt (Used across all examples)

```text
Western comics style, bold ink outlines, hatched shadows, eerie detached calm, seen from a dutch high angle close-up, upper body portrait, dynamic pose, dramatic angle, strong perspective. male human plague doctor, thinning gray hair slicked back, thin sparse eyebrows, pale sickly skin gradient, gaunt older adult, long thin gloved fingers, a wispy gray goatee, deep tired wrinkles, dull green eyes. narrow jaw, tall lanky frame, eerie detached calm stare. a long black waxed-leather coat with a high collar, a satchel of glass vials strapped across his chest. holding a bubbling green potion vial up to the light. Background: a dim candle-lit apothecary shop cluttered with shelves of jars and dried herbs. Lighting: flickering warm candlelight from below mixing with cool teal moonlight through a fogged window, creating dramatic contrast across his face.
```

### 🟪 Model Tuning Effects (Blocks 1–6, Text Fusion, Time Embed, Projection)

| | |
|---|---|
| **Block 1** <br><br> [![Block 1 Effect](assets/examples/Block_1.webp)](assets/examples/Block_1.webp) | **Block 2** <br><br> [![Block 2 Effect](assets/examples/Block_2.webp)](assets/examples/Block_2.webp) |
| **Block 3** <br><br> [![Block 3 Effect](assets/examples/Block_3.webp)](assets/examples/Block_3.webp) | **Block 4** <br><br> [![Block 4 Effect](assets/examples/Block_4.webp)](assets/examples/Block_4.webp) |
| **Block 5** <br><br> [![Block 5 Effect](assets/examples/Block_5.webp)](assets/examples/Block_5.webp) | **Block 6** <br><br> [![Block 6 Effect](assets/examples/Block_6.webp)](assets/examples/Block_6.webp) |
| **Text Fusion** <br><br> [![Text Fusion Effect](assets/examples/TextFusion.webp)](assets/examples/TextFusion.webp) | **Time Embed** <br><br> [![Time Embed Effect](assets/examples/Time_Embed.webp)](assets/examples/Time_Embed.webp) |
| **Projection** <br><br> [![Projection Effect](assets/examples/Projection.webp)](assets/examples/Projection.webp) | |

---

### 🟨 CLIP Text Encoder Tuning Effects (Layers 1–7, Embedding)

| | |
|---|---|
| **Layer 1** <br><br> [![Layer 1 Effect](assets/examples/Layer_1.webp)](assets/examples/Layer_1.webp) | **Layer 2** <br><br> [![Layer 2 Effect](assets/examples/Layer_2.webp)](assets/examples/Layer_2.webp) |
| **Layer 3** <br><br> [![Layer 3 Effect](assets/examples/Layer_3.webp)](assets/examples/Layer_3.webp) | **Layer 4** <br><br> [![Layer 4 Effect](assets/examples/Layer_4.webp)](assets/examples/Layer_4.webp) |
| **Layer 5** <br><br> [![Layer 5 Effect](assets/examples/Layer_5.webp)](assets/examples/Layer_5.webp) | **Layer 6** <br><br> [![Layer 6 Effect](assets/examples/Layer_6.webp)](assets/examples/Layer_6.webp) |
| **Layer 7** <br><br> [![Layer 7 Effect](assets/examples/Layer_7.webp)](assets/examples/Layer_7.webp) | **Embedding** <br><br> [![Embedding Effect](assets/examples/Embedding.webp)](assets/examples/Embedding.webp) |

---

### Deterministic Multi-Seed Persistence

Here you can see how feature isolation works in practice:
By identifying and amplifying only the specific Model blocks and CLIP layers responsible for generating the plague doctor's beak mask, the feature persists robustly and deterministically across completely different generation seeds.

<p align="center">
  <a href="assets/examples/SameTuning-DifferentSeed.webp">
    <img src="assets/examples/SameTuning-DifferentSeed.webp" width="850" alt="Same Tuning, Different Seed" />
  </a>
</p>

---

## Developer Guide & Testing

The repository comes equipped with an extensive 11-suite automated test suite located in `tests/`:

* **Contract Validation:** Checks inputs, outputs, types, and recipe replays (`test_node_contracts.py`).
* **Memory Invariance:** Verifies patcher clones and guarantees ComfyUI base weights are strictly preserved (`test_reset_patcher.py`, `test_prune.py`).
* **Numerical Parity:** Validates Lie algebra orthogonal rotation matrix properties and channel profiling fidelity (`test_channel_profile.py`, `test_channel_roundtrip.py`).
* **Checkpoints & Savers:** End-to-end verification of BF16 and scaled FP8 streaming encoders (`test_savers_patching.py`, `test_saver_e2e.py`).

Run the automated suite using ComfyUI's Python runtime:
```powershell
Get-ChildItem -Path tests -Filter "test_*.py" | ForEach-Object {
    & "path/to/comfyui/python.exe" $_.FullName
}
```

For complete architectural notes, memory management invariants, and custom node extensions, refer to **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## Troubleshooting

* **Nodes do not appear in ComfyUI:** Restart ComfyUI completely. Ensure the folder is placed in `ComfyUI/custom_nodes/Arthemy_Krea2_Tuner` without nested folders.
* **Sliders produce no visible change:** Confirm that you are running a **Krea-2** checkpoint with a **Qwen3** text encoder. On incompatible models, tensor prefixes will not match and `Patches: 0` will be returned in the node info string.
* **Out of Memory during save:** Model Saver and CLIP Saver stream tensors sequentially in chunks, but writing 13B models requires free RAM. Ensure background processes are minimized.
* **Scaled FP8 compatibility:** Quantized FP8 layers carrying separate `scale_weight` companions are automatically protected from double-scaling during tuning and correctly folded during checkpoint export.

---

## Author

**Arthemy** · [@aledelpho](https://github.com/aledelpho)

---

## License

Distributed under the **MIT License** — see [`LICENSE`](LICENSE) for details.
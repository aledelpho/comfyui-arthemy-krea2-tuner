<p align="center">
  <img src="assets/Generation.png" width="900" alt="Arthemy Krea-2 Tuner Generation Pipeline" />
</p>

<h1 align="center">Arthemy Krea-2 Tuner</h1>

<p align="center">
  Live weight tuning for <b>Krea-2</b> diffusion models and their <b>Qwen3</b> text encoder, directly inside ComfyUI.
</p>

<p align="center">
  <img alt="ComfyUI custom node" src="https://img.shields.io/badge/ComfyUI-custom--node-6b46c1?style=flat-square">
  <img alt="Compatibility" src="https://img.shields.io/badge/compatible-Krea--2%20%7C%20Qwen3-1e88e5?style=flat-square">
  <a href="#-license"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green?style=flat-square"></a>
  <img alt="GitHub stars" src="https://img.shields.io/github/stars/aledelpho/comfyui-arthemy-krea2-tuner?style=flat-square">
</p>

---

Edit a **Krea-2** model and its **CLIP** text encoder by multiplying specific internal weights up or down, live, with **no dataset and no retraining**. Move a slider, generate, and compare against the untouched model. When you land on a result you like, save it as a lightweight preset or bake it into a full `.safetensors` checkpoint.

Built for anyone who wants to push a model toward a specific look or behavior — a color palette, a rendering style, a recurring motif — by editing what's already inside it, instead of fine-tuning from scratch.

> [!NOTE]
> **Requires:** Krea-2 diffusion checkpoints + a Qwen3 text encoder (including Qwen3-VL 4B/8B/32B). Block counts and tensor names are matched to this architecture; these nodes won't affect other model families as-is.

<p align="center">
  <a href="assets/examples/SameTuning-DifferentSeed.png">
    <img src="assets/examples/SameTuning-DifferentSeed.png" width="800" alt="Same Tuning, Different Seed" />
  </a>
  <br>
  <sub><i>Same block/layer calibration, four different generation seeds — the isolated feature (here, the beak mask) holds up deterministically. Find the slice responsible for a trait, and it stays put.</i></sub>
</p>

---

## 📑 Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Picking the Right Node](#-picking-the-right-node)
  - [Model Tuning](#-tier-1--tuner-whole-groups)
  - [CLIP Tuning](#-clip-tuning-qwen3)
  - [LoRA Tuning](#-lora-tuning)
  - [At a Glance (Summary Table)](#-at-a-glance)
- [Reading the Sliders](#-reading-the-sliders)
- [Saving and Loading Calibrations](#-saving-and-loading-calibrations)
- [Seeing What's Currently Patched](#-seeing-whats-currently-patched)
- [Troubleshooting](#-troubleshooting)
- [Visual Tuning Showcase & Effects](#-visual-tuning-showcase--effects)
- [Author & License](#-author)

---

## 📦 Installation

### Option 1: Via ComfyUI Manager (Recommended)
Search for **"Arthemy Krea-2 Tuner"** in the ComfyUI Manager custom node browser and click **Install**.

### Option 2: Manual Installation
Clone this repository directly into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/aledelpho/comfyui-arthemy-krea2-tuner.git
```

*Restart ComfyUI completely afterward.*

---

## 🚀 Quick Start

1. **Load** your `MODEL` and `CLIP` normally in your workflow.
2. Add a **`🟦✨ Model Tuner`** node and route `MODEL` through it.
3. Move one of the block sliders (`Block_1` … `Block_6`) away from `0.00` (`0.00` = no change; positive values strengthen that group, negative values weaken it).
4. Connect the Tuner's `MODEL` output to your sampler and generate. Compare against the untouched model — with a slider away from `0.00`, you should see an immediate visual shift. If the outputs look identical, verify your checkpoint is Krea-2/Qwen3 (see [Troubleshooting](#-troubleshooting)).
5. *(Optional)* Chain a **`🟨✨ CLIP Tuner`** on the `CLIP` wire to adjust text encoder layers simultaneously.

---

## 🎛️ Picking the Right Node

Every domain — **Model**, **CLIP**, and **LoRA** — is tuned through three tiers of precision.

<p align="center">
  <img src="assets/ModelTuner.png" width="650" alt="Model Tuner Nodes" />
</p>

### 🟦 Tier 1 — Tuner (Whole Groups)
**`🟦✨ Model Tuner`** provides one slider per group of blocks: `Text_Fusion`, `Time_Embed`, `Projection`, and `Block_1` through `Block_6`. Moving a slider scales every tensor inside that group by the same offset.
* **Use case:** Broad exploration — *"Let's see which general region influences my output."*
* **Workflow:** `MODEL` → `🟦✨ Model Tuner` → `Sampler`. Nudge one slider at a time, generate, and compare.

### 🔬 Tier 2 — Sub-Block Tuner (Block + Tensor Type)
**`🟦🔬 Model Sub-Block Tuner`** narrows the target: select one block from the `target_block` dropdown, then adjust individual sliders for specific internal tensor types (`ATTN_wq_query`, `ATTN_wk_key`, `ATTN_wv_value`, `ATTN_wo_out`, `MLP_gate_swiglu`, `MLP_up_proj`, `MLP_down_proj`, `NORMS_block_scales`, etc.).
* **Use case:** Surgical isolation — *"Block 3 is promising, let's isolate attention layers from MLP layers."*
* **Workflow:** Follows a Tier 1 pass. Set `target_block` to the region identified in Tier 1 and sweep tensor-type sliders individually.

### 🌪️ Tier 3 — Chaos Tuner (Seeded Randomness)
**`🟦🌪️ Model Sub-Block Chaos Tuner`** uses the same targeting as Tier 2, but applies a seeded perturbation: with probability `chance`, each matched tensor is perturbed by `chaos_strength`. Same seed = deterministic, reproducible noise every time.
* **Block-Level mode:** Perturbs an entire tensor at once.
* **Element-Level (Sub-atomic) mode:** Rolls per individual weight for finer granularity.
* **Use case:** Creative discovery — *"Let's randomly nudge this sub-slice and lock the seed once an interesting style emerges."*

<p align="center">
  <a href="assets/examples/ChaosTuning.png">
    <img src="assets/examples/ChaosTuning.png" width="800" alt="Chaos Tuning Variations" />
  </a>
  <br>
  <sub><i>Different stochastic configurations, identical prompt — Chaos Tuning as a discovery tool for shading and line-weight variations.</i></sub>
</p>

---

### 🟨 CLIP Tuning (Qwen3)
<p align="center">
  <img src="assets/ClipTuner.png" width="650" alt="CLIP Tuner Nodes" />
</p>

Tuning applied to the Qwen3 text encoder follows the identical 3-tier structure:
* **`🟨✨ CLIP Tuner` (Tier 1):** One slider per group (`Embedding`, `Layer_1` through `Layer_7`).
* **`🟨🔬 CLIP Sub-Block Tuner` (Tier 2):** Select a specific layer, then adjust Qwen3-equivalent internal tensor sliders.
* **`🟨🌪️ CLIP Sub-Block Chaos Tuner` (Tier 3):** Seeded perturbation scoped to CLIP layers.

---

### 🟪 LoRA Tuning
<p align="center">
  <img src="assets/LoraTuner.png" width="650" alt="LoRA Loader Nodes" />
</p>

* **`🟪🔮 LoRA Block Loader` (Tier 1):** Drop-in replacement for standard LoRA loaders with per-section strength multipliers.
* **`🟪🔬 Load Sub-Block LoRA` (Tier 2):** Targets one block and tensor type (e.g., keep only attention layers of a LoRA and drop the rest).
* **`🟪🌪️ Load Sub-Block Chaos LoRA` (Tier 3):** Randomly retains or drops LoRA keys based on probability sliders and a seed.

---

### 📊 At a Glance

| Domain | Tier 1 — Group-level | Tier 2 — Block + Tensor Type | Tier 3 — Random / Seeded |
| :--- | :--- | :--- | :--- |
| 🟦 **Model** | `🟦✨ Model Tuner` | `🟦🔬 Model Sub-Block Tuner` | `🟦🌪️ Model Sub-Block Chaos Tuner` |
| 🟨 **CLIP** | `🟨✨ CLIP Tuner` | `🟨🔬 CLIP Sub-Block Tuner` | `🟨🌪️ CLIP Sub-Block Chaos Tuner` |
| 🟪 **LoRA** | `🟪🔮 LoRA Block Loader` | `🟪🔬 Load Sub-Block LoRA` | `🟪🌪️ Load Sub-Block Chaos LoRA` |

---

## 📐 Reading the Sliders

Tier 1 and Tier 2 sliders represent offsets from baseline (`0.00` = no modification). Two calculation modes are available per node:

| Mode | Formula | Use For |
| :--- | :--- | :--- |
| **Soft Value** | `1.0 + (slider × 0.10)` | Gentle adjustments — a slider of `1.0` moves the weight by +10%. *(Recommended default)* |
| **Real Value** | `1.0 + slider` | Direct multiplier — a slider of `1.0` doubles the weight (+100%). For aggressive changes. |

* **Chaos Sliders:** `chance` (`0.0`–`1.0`) defines the probability a tensor is touched; `chaos_strength` controls the perturbation amplitude.

### Optional Scripting Overrides
* **`vectors_override`**: Comma-separated list of raw values setting every group slider at once (34 values for Model Tuner, 8 for CLIP Tuner). Leave empty to use UI sliders.
* **`granular_json`**: JSON map targeting specific tensor-key substrings with custom weights:
  ```json
  {"blocks.3.attn.wq": 0.4, "blocks.3.mlp.down": -0.2}
  ```

---

## 💾 Saving and Loading Calibrations

Tuning occurs live in memory and **does not modify original checkpoint files**.

### 1. Presets (Lightweight JSON)
<p align="center">
  <img src="assets/Preset.png" width="550" alt="Preset System" />
</p>

* **`🟦🟨💾 Preset Saver`**: Captures all active patches and Chaos seeds into a small JSON file for deterministic reproduction.
* **`🟦🟨📂 Preset Loader`**: Loads saved presets onto fresh `MODEL`/`CLIP` streams with global `strength_model` and `strength_clip` dials.

> [!WARNING]
> Presets capture Tuner and Chaos offsets only — not active LoRAs. Use the **Model Baker** first if you wish to bake LoRA weights permanently into the preset.

<p align="center">
  <a href="assets/examples/XYBenchmark.png">
    <img src="assets/examples/XYBenchmark.png" width="800" alt="X/Y Tuning Benchmark Grid" />
  </a>
  <br>
  <sub><i>A calibration saved as a preset, reloaded across multiple prompts/seeds: the target aesthetic signature remains consistent.</i></sub>
</p>

### 2. Permanent Export (Full Checkpoints)
<p align="center">
  <img src="assets/Saver.png" width="450" alt="Model and CLIP Savers" />
</p>

* **`🟦🟨 Model Baker`**: Folds all active memory patches permanently into the model weights and clears the runtime patch list.
* **`🟦💾 Model Saver` / `🟨💾 CLIP Saver`**: Exports tuned models to `.safetensors` (BF16 or FP8), streamed directly to disk to prevent OOM errors.
* **`🟦🟨🔄 Reset Patcher`**: Clears all active patches and restores the model to clean baseline in memory.

---

## 👁️ Seeing What's Currently Patched

* **`🟦📊 Model Visualizer` / `🟨📊 CLIP Visualizer`**: Real-time graph nodes that render a bar chart of active weight modifications across blocks before rendering.

---

## 🛠️ Troubleshooting

* **Nodes don't appear in ComfyUI:** Restart the ComfyUI server process completely (refreshing the browser tab is not sufficient).
* **Sliders produce no visible change:** Ensure the loaded model is a **Krea-2** checkpoint with a **Qwen3** text encoder. On unsupported models, tensor names do not match and patches apply to 0 tensors.
* **Out of Memory (OOM) during baking/saving:** Model Baker and Savers use chunked streaming, but large checkpoints still require memory overhead. Close background GPU tasks before exporting.
* **Saved preset doesn't include my LoRA:** Presets only store offset multipliers. Use **`🟦🟨 Model Baker`** to fuse the LoRA into the model before saving.

---

## 🎨 Visual Tuning Showcase & Effects

A visual reference gallery showing how modulating each Model block and CLIP layer influences output in isolation.

<details>
<summary><b>📋 Benchmark Baseline Prompt</b> (Used across all examples)</summary>

```text
Western comics style, bold ink outlines, hatched shadows, eerie detached calm, seen from a dutch high angle close-up, upper body portrait, dynamic pose, dramatic angle, strong perspective. male human plague doctor, thinning gray hair slicked back, thin sparse eyebrows, pale sickly skin gradient, gaunt older adult, long thin gloved fingers, a wispy gray goatee, deep tired wrinkles, dull green eyes. narrow jaw, tall lanky frame, eerie detached calm stare. a long black waxed-leather coat with a high collar, a satchel of glass vials strapped across his chest. holding a bubbling green potion vial up to the light. Background: a dim candle-lit apothecary shop cluttered with shelves of jars and dried herbs. Lighting: flickering warm candlelight from below mixing with cool teal moonlight through a fogged window, creating dramatic contrast across his face.
```
</details>

<details>
<summary><b>🟦 Model Tuning Effects</b> (Blocks 1–6, Text Fusion, Time Embed, Projection)</summary>
<br>

| Block | Visual Result |
| :--- | :--- |
| **Block 1** | <a href="assets/examples/Block_1.png"><img src="assets/examples/Block_1.png" width="100%" alt="Block 1 Effect" /></a> |
| **Block 2** | <a href="assets/examples/Block_2.png"><img src="assets/examples/Block_2.png" width="100%" alt="Block 2 Effect" /></a> |
| **Block 3** | <a href="assets/examples/Block_3.png"><img src="assets/examples/Block_3.png" width="100%" alt="Block 3 Effect" /></a> |
| **Block 4** | <a href="assets/examples/Block_4.png"><img src="assets/examples/Block_4.png" width="100%" alt="Block 4 Effect" /></a> |
| **Block 5** | <a href="assets/examples/Block_5.png"><img src="assets/examples/Block_5.png" width="100%" alt="Block 5 Effect" /></a> |
| **Block 6** | <a href="assets/examples/Block_6.png"><img src="assets/examples/Block_6.png" width="100%" alt="Block 6 Effect" /></a> |
| **Text Fusion** | <a href="assets/examples/TextFusion.png"><img src="assets/examples/TextFusion.png" width="100%" alt="Text Fusion Effect" /></a> |
| **Time Embed** | <a href="assets/examples/Time_Embed.png"><img src="assets/examples/Time_Embed.png" width="100%" alt="Time Embed Effect" /></a> |
| **Projection** | <a href="assets/examples/Projection.png"><img src="assets/examples/Projection.png" width="100%" alt="Projection Effect" /></a> |

</details>

<details>
<summary><b>🟨 CLIP Text Encoder Tuning Effects</b> (Layers 1–7, Embedding)</summary>
<br>

| Layer | Visual Result |
| :--- | :--- |
| **Layer 1** | <a href="assets/examples/Layer_1.png"><img src="assets/examples/Layer_1.png" width="100%" alt="Layer 1 Effect" /></a> |
| **Layer 2** | <a href="assets/examples/Layer_2.png"><img src="assets/examples/Layer_2.png" width="100%" alt="Layer 2 Effect" /></a> |
| **Layer 3** | <a href="assets/examples/Layer_3.png"><img src="assets/examples/Layer_3.png" width="100%" alt="Layer 3 Effect" /></a> |
| **Layer 4** | <a href="assets/examples/Layer_4.png"><img src="assets/examples/Layer_4.png" width="100%" alt="Layer 4 Effect" /></a> |
| **Layer 5** | <a href="assets/examples/Layer_5.png"><img src="assets/examples/Layer_5.png" width="100%" alt="Layer 5 Effect" /></a> |
| **Layer 6** | <a href="assets/examples/Layer_6.png"><img src="assets/examples/Layer_6.png" width="100%" alt="Layer 6 Effect" /></a> |
| **Layer 7** | <a href="assets/examples/Layer_7.png"><img src="assets/examples/Layer_7.png" width="100%" alt="Layer 7 Effect" /></a> |
| **Embedding** | <a href="assets/examples/Embedding.png"><img src="assets/examples/Embedding.png" width="100%" alt="Embedding Effect" /></a> |

</details>

---

## 👤 Author

**Arthemy** · [@aledelpho](https://github.com/aledelpho)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

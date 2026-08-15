# ✨ Arthemy Krea-2 Tuner — ComfyUI Custom Node Suite

A high-precision model, CLIP, and LoRA tuning suite specifically engineered for **Krea-2** diffusion models and **Qwen3** text encoders inside [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

---

## 📐 The 3×3 Precision Grid

The suite is designed around 3 core domains (Model, CLIP, LoRA) structured across 3 distinct tiers of control and granularity:

| Domain | Tier 1: Tuner (Group / Block) | Tier 2: Sub-Block (Surgical Precision) | Tier 3: Sub-Block Chaos (Stochastic Discovery) |
|---|---|---|---|
| **🟦 Model** | `🟦✨ Model Tuner` | `🟦🔬 Model Sub-Block Tuner` | `🟦🌪️ Model Sub-Block Chaos Tuner` |
| **🟨 CLIP** | `🟨✨ CLIP Tuner` | `🟨🔬 CLIP Sub-Block Tuner` | `🟨🌪️ CLIP Sub-Block Chaos Tuner` |
| **🟪 LoRA** | `🟪🔮 LoRA Block Loader` | `🟪🔬 Load Sub-Block LoRA` | `🟪🌪️ Load Sub-Block Chaos LoRA` |

### 🎯 How to Choose Your Tool
- **Tier 1 (Tuner)**: Macro adjustments by block groups (e.g. `Block_1`–`Block_6`, `Text_Fusion`, `Time_Embed`, `Projection`).
- **Tier 2 (Sub-Block)**: Deterministic surgical precision targeting specific internal component types (`ATTN_wq_query`, `ATTN_wk_key`, `MLP_gate_swiglu`, `MLP_down_proj`, `NORMS`, etc.) across individual transformer blocks.
- **Tier 3 (Sub-Block Chaos)**: Stochastic discovery tool. Applies independent probability rolls per sub-tensor component to randomly discover novel stylistic or architectural variations.

---

## 🛠️ Savers, Bakers, Presets & Visualizers

| Node | Description |
|---|---|
| `🟦💾 Model Saver` | Memory-safe BF16/FP8 model baker & safetensors exporter with streaming tensor processing and auto-incrementing file naming. |
| `🟨💾 CLIP Saver` | Memory-safe BF16/FP8 exporter for Qwen3 / Qwen3-VL text encoder checkpoints. |
| `🟦🟨 Model Baker` | In-memory CPU patch baking with submodule parameter isolation — collapses active patches without mutating the cached base checkpoint. |
| `🟦🟨💾 Preset Saver` | Exports all active Model and CLIP tuning patches into a lightweight (~1–2 KB) `.json` preset file. |
| `🟦🟨📂 Preset Loader` | Loads and applies `.json` presets with independent `strength_model` and `strength_clip` global scaling multipliers. |
| `🟦📊 Model Visualizer` | Real-time visual graph rendering per-block weight offsets, LoRA presence, and chaos modifications on the LiteGraph Canvas. |
| `🟨📊 CLIP Visualizer` | Real-time visual diagram for Qwen3 text encoder layers. |
| `🟦🟨🔄 Reset Patcher` | Cleans and resets all active patches from model and CLIP patchers. |

---

## 🎛️ Tuning Modes & Value Interpretation

All tuner sliders operate as **offsets from baseline** (`0.00 = 1.0x / no change`):

| Mode | Formula | Use Case |
|---|---|---|
| **Soft Value** | `1.0 + (delta × 0.10)` | Precision micro-tuning (linear 10% dampening factor for subtle adjustments). |
| **Real Value** | `1.0 + delta` | Direct 1-to-1 linear multiplier scaling. |

### 📋 Advanced Overrides
- **Vectors Override**: Pass comma-separated float strings to configure all layer weights in a single string (34 values for Model, 8 values for CLIP).
- **Granular JSON**: Pass custom JSON key-value maps to target specific sub-tensor prefixes with custom weights.

---

## ⚙️ Architecture & Safety Standards

- **Native ComfyUI Patcher Integration** — Uses standard `add_patches` / `calculate_weight` without monkey-patching or unsafe private attribute overrides.
- **Submodule Parameter Isolation** — `ModelBaker` clones parameter dictionaries along modified module paths, preventing permanent mutation of the shared PyTorch `nn.Module` in memory.
- **FP8 Scale-Aware Dequantization** — Seamlessly applies companion `{key}_scale` and `weight_scale` tensors when dequantizing quantized models.
- **Dynamic Probing Engine** (`Krea2TensorParser`) — Programmatic state-dict architecture discovery with regex fallback.
- **Strict Key Resolution & $O(1)$ Caching** — Fast dictionary lookup prevents redundant `state_dict()` evaluations across large models.
- **Memory-Safe Streaming** (`process_tensor_stream`) — GC-aware streaming generator for OOM-safe model baking and saving.

---

## 🚀 Installation

### Via ComfyUI Manager (Recommended)
Search for **"Arthemy Krea-2 Tuner"** in the ComfyUI Manager's custom node browser.

### Manual Installation
1. Clone this repository into your ComfyUI `custom_nodes` folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/aledelpho/comfyui-arthemy-krea2-tuner.git
   ```
2. Restart ComfyUI.

---

## 👤 Author

**Arthemy** · [@aledelpho](https://github.com/aledelpho)

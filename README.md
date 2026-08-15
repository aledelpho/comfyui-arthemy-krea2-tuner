# ✨ Arthemy Krea-2 Tuner — ComfyUI Custom Node Suite

A high-precision model, CLIP, and LoRA tuning suite for **Krea-2** diffusion models inside [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

---

## 📐 The 3×3 Precision Grid

The suite is structured around 3 core domains across 3 levels of granularity:

| Domain | Tier 1: Tuner (Coarse Group/Block) | Tier 2: Sub-Block (Surgical Precision) | Tier 3: Sub-Block Chaos (Stochastic Discovery) |
|---|---|---|---|
| **🟦 Model** | `🟦✨ Model Tuner` | `🟦🔬 Model Sub-Block Tuner` | `🟦🌪️ Model Sub-Block Chaos Tuner` |
| **🟨 CLIP** | `🟨✨ CLIP Tuner` | `🟨🔬 CLIP Sub-Block Tuner` | `🟨🌪️ CLIP Sub-Block Chaos Tuner` |
| **🟪 LoRA** | `🟪🔮 LoRA Block Loader` | `🟪🔬 Load Sub-Block LoRA` | `🟪🌪️ Load Sub-Block Chaos LoRA` |

---

## 🛠️ Savers, Bakers & Utilities

| Node | Description |
|---|---|
| `🟦💾 Model Saver` | Memory-safe BF16 model baker & safetensors saver with streaming tensor processing |
| `🟨💾 CLIP Saver` | Memory-safe BF16 Qwen3 / Qwen3-VL text encoder saver |
| `🟦🟨 Model Baker` | In-place CPU BF16 patch baking directly into live model weights |
| `🟦📊 Model Visualizer` | Real-time visual graph rendering of per-block weight offsets, LoRA presence, and chaos modifications |
| `🟨📊 CLIP Visualizer` | Real-time visual graph for the Qwen3 text encoder layers |
| `🟦🟨🔄 Reset Patcher` | Clears all active patches from model and CLIP patchers |

---

## ⚙️ Architecture & Safety Standards

- **Native ComfyUI Patcher Integration** — Uses standard `add_patches` / `calculate_weight` without monkey-patching or unsafe cache overrides.
- **Dynamic Probing Engine** (`Krea2TensorParser`) — Programmatic state-dict architecture discovery with regex fallback.
- **Memory-Safe Streaming** (`process_tensor_stream`) — Memory threshold & GC-aware streaming generator for OOM-safe model baking and saving.
- **In-place Mutation Guard** — All patch tensors undergo strict `.clone().detach().contiguous()` to prevent VRAM buffer corruption.

---

## 🚀 Installation

### Manual
1. Clone or download this repository into your ComfyUI `custom_nodes` folder:
   ```
   ComfyUI/custom_nodes/comfyui-arthemy-krea2-tuner/
   ```
2. Restart ComfyUI.

### Via ComfyUI Manager
Search for **"Arthemy Krea-2 Tuner"** in the Manager's custom node browser.

---

## 🎛️ Soft Value vs. Real Value Mode

All tuner sliders operate as **offsets from baseline** (`0.00 = no change`).

| Mode | Formula | Use Case |
|---|---|---|
| **Soft Value** | `1.0 + delta × 0.10` | Precision micro-tuning (10% dampening factor) |
| **Real Value** | `1.0 + delta` | Direct 1-to-1 linear multiplier scaling |

---

## 👤 Author

**Arthemy** · [@aledelpho](https://github.com/aledelpho)

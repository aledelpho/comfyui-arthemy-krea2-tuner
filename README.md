# ✨ Arthemy Krea-2 Tuner — ComfyUI Custom Node Suite

A high-precision model and CLIP tuning suite for **Krea-2** diffusion models inside [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

---

## Features

| Node | Description |
|---|---|
| ✨ **Model Tuner** | Group-level weight scaling for all 28 Krea-2 blocks + text fusion layers |
| ✨ **CLIP Tuner** | Per-group layer scaling for the Qwen3 text encoder (embedding + 7 groups) |
| 🌪️ **Model Chaos Block Tuner** | Stochastic weight perturbation at Block-Level or Element-Level (sub-atomic) |
| 🌪️ **CLIP Chaos Block Tuner** | Same chaos engine applied to the CLIP text encoder |
| 🔬 **Model Block Surgeon Tuner** | Fine-grained sub-tensor targeting per attention/MLP component type per block |
| 🔬 **CLIP Block Surgeon Tuner** | Same surgeon precision for CLIP layers |
| 🌪️🔬 **Model/CLIP Chaos Block Surgeon** | Chaos tuning with surgeon-level sub-tensor targeting |
| 🔮 **LoRA Block Loader** | Block-level LoRA loading with per-block strength sliders |
| 🔮 **Isolated LoRA Block Loader** | Isolated LoRA loading: clones the patcher before applying so patches don't accumulate from upstream nodes |
| 🎲 **Load Chaos LoRA** | Stochastic LoRA application with chance and strength controls |
| 🎲🔬 **Load Chaos LoRA Block Surgeon** | Chaos LoRA with surgeon-level targeting |
| 💾 **Model Saver** | Memory-safe BF16 model baker & saver with streaming tensor processing |
| 💾 **CLIP Saver** | Same safe saver for CLIP/text encoder checkpoints |
| **Model Baker** | In-place CPU BF16 patch baking into model weights |
| 📊 **Model Visualizer** | Real-time visual graph of per-block weight offsets and LoRA presence |
| 📊 **CLIP Visualizer** | Same visual graph for the Qwen3 text encoder |
| 🔄 **Reset Patcher** | Clears all active patches from a model or CLIP patcher |

---

## Architecture

- **Native ComfyUI Patcher Integration** — uses `add_patches` / `calculate_weight` officially, no monkey-patching.
- **Dynamic Probing Engine** (`Krea2TensorParser`) — programmatic state-dict architecture discovery with regex fallback.
- **Memory-Safe Streaming** (`process_tensor_stream`) — GC-aware generator for OOM-safe baking and saving.
- **DRY BaseSurgeonTuner** — unified engine shared by all Surgeon and Chaos Surgeon tuners.
- **In-place mutation prevention** — all patch tensors are `.clone().detach().contiguous()` before injection.

---

## Installation

### Manual
1. Clone or download this repository into your ComfyUI `custom_nodes` folder:
   ```
   ComfyUI/custom_nodes/comfyui-arthemy-krea2-tuner/
   ```
2. Restart ComfyUI.

### Via ComfyUI Manager
Search for **"Arthemy Krea-2 Tuner"** in the Manager's custom node browser.

---

## Compatibility

- ComfyUI (latest)
- Krea-2 BF16 and FP8-scaled models
- Qwen3 (0.6B) and Qwen3-VL (4B) text encoders
- Python ≥ 3.10, PyTorch ≥ 2.1

---

## Soft Value vs. Real Value Mode

All tuner sliders operate as **offsets from baseline** (`0.00 = no change`).

| Mode | Formula | Use case |
|---|---|---|
| **Soft Value** | `1.0 + delta × 0.10` | Precision micro-tuning (10% dampening) |
| **Real Value** | `1.0 + delta` | Direct 1-to-1 linear scaling |

---

## Author

**Arthemy** · [@aledelpho](https://github.com/aledelpho)

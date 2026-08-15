from PIL import Image, ImageDraw, ImageFont
import numpy as np
"""
Arthemy Krea-2 Suite for ComfyUI

Architecture Overview:
1. Native ComfyUI Patcher Integration.
2. Dynamic Probing Engine in Krea2TensorParser (State-dict architecture probing with regex fallback).
3. Memory-Safe Generator & In-Place Processing (Out-Of-Memory prevention for Baker & Savers).
4. DRY BaseSurgeonTuner Architecture (Unified Model & CLIP Surgeon / Chaos Surgeon engine).
"""

import math
import os
import re
import gc
import json
import zlib
import time
import hashlib
import logging
from typing import Tuple, Dict, Any, Generator
import torch
import safetensors.torch
import comfy.lora
import comfy.model_patcher
import comfy.sd
import comfy.utils
import folder_paths

# Register arthemy_presets folder paths with ComfyUI
models_dir = folder_paths.models_dir
arthemy_presets_dir = os.path.join(models_dir, "arthemy_presets")
local_presets_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "presets")
os.makedirs(arthemy_presets_dir, exist_ok=True)
os.makedirs(local_presets_dir, exist_ok=True)

if "arthemy_presets" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["arthemy_presets"] = ([arthemy_presets_dir, local_presets_dir], {".json"})

class Krea2Config:
    """Global architecture configuration parameters and defaults."""
    MAX_UNET_BLOCKS: int = 28
    MAX_CLIP_LAYERS: int = 36
    DEFAULT_GC_THRESHOLD_MB: int = 500

def parse_granular_json(json_string: str) -> Dict[str, Any]:
    """Parses granular JSON configurations strictly, raising an explicit error for invalid syntax."""
    if not json_string or not json_string.strip():
        return {}
    try:
        parsed_data = json.loads(json_string)
        if not isinstance(parsed_data, dict):
            raise ValueError("JSON root must be a dictionary.")
        return parsed_data
    except Exception as e:
        raise ValueError(f"Arthemy Suite Error: Invalid Granular JSON syntax.\nDetails: {str(e)}")

def generate_fast_seed(key_string: str, base_seed: int) -> int:
    """Generates a fast, deterministic seed using CRC32."""
    key_hash = zlib.crc32(key_string.encode("utf-8"))
    return (key_hash ^ base_seed) % (2**31)

# Setup logger
logger = logging.getLogger("ArthemyKrea2Suite")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ==============================================================================
# HELPER DATA STRUCTURES & CONSTANTS
# ==============================================================================
DUMMY_TENSOR = torch.zeros(1, dtype=torch.float32)

MODEL_BLOCK_LABEL_TO_IDX = {
    "Block_1A": 0, "Block_1B": 1, "Block_1C": 2, "Block_1D": 3, "Block_1E": 4,
    "Block_2A": 5, "Block_2B": 6, "Block_2C": 7, "Block_2D": 8, "Block_2E": 9,
    "Block_3A": 10, "Block_3B": 11, "Block_3C": 12, "Block_3D": 13, "Block_3E": 14,
    "Block_4A": 15, "Block_4B": 16, "Block_4C": 17, "Block_4D": 18, "Block_4E": 19,
    "Block_5A": 20, "Block_5B": 21, "Block_5C": 22, "Block_5D": 23,
    "Block_6A": 24, "Block_6B": 25, "Block_6C": 26, "Block_6D": 27,
}

CLIP_LAYER_LABEL_TO_IDX = {
    "Layer_1A": 0, "Layer_1B": 1, "Layer_1C": 2, "Layer_1D": 3, "Layer_1E": 4,
    "Layer_2A": 5, "Layer_2B": 6, "Layer_2C": 7, "Layer_2D": 8, "Layer_2E": 9,
    "Layer_3A": 10, "Layer_3B": 11, "Layer_3C": 12, "Layer_3D": 13, "Layer_3E": 14,
    "Layer_4A": 15, "Layer_4B": 16, "Layer_4C": 17, "Layer_4D": 18, "Layer_4E": 19,
    "Layer_5A": 20, "Layer_5B": 21, "Layer_5C": 22, "Layer_5D": 23, "Layer_5E": 24,
    "Layer_6A": 25, "Layer_6B": 26, "Layer_6C": 27, "Layer_6D": 28, "Layer_6E": 29,
    "Layer_7A": 30, "Layer_7B": 31, "Layer_7C": 32, "Layer_7D": 33, "Layer_7E": 34, "Layer_7F": 35,
}

# ==============================================================================
# POINT 2: DYNAMIC PROBING ENGINE (Krea2TensorParser)
# ==============================================================================
class Krea2TensorParser:
    """Dynamic Probing Engine & Sub-Tensor Parser for Krea-2 Model & Qwen3 Text Encoder.
    Uses dynamic state_dict inspection to deduce architecture hierarchy programmatically,
    falling back to regex only if required."""

    MODEL_SURGEON_MAP = {
        "ATTN_wq_query": ("attn.wq.weight",),
        "ATTN_wk_key": ("attn.wk.weight",),
        "ATTN_wv_value": ("attn.wv.weight",),
        "ATTN_wo_out": ("attn.wo.weight",),
        "ATTN_gate_attn": ("attn.gate.weight",),
        "ATTN_qknorm_scales": ("attn.qknorm.qnorm.scale", "attn.qknorm.knorm.scale"),
        "MLP_gate_swiglu": ("mlp.gate.weight",),
        "MLP_up_proj": ("mlp.up.weight",),
        "MLP_down_proj": ("mlp.down.weight",),
        "MOD_lin_time": ("mod.lin",),
        "NORMS_block_scales": ("prenorm.scale", "postnorm.scale"),
    }

    CLIP_SURGEON_MAP = {
        "ATTN_q_proj": ("self_attn.q_proj.weight",),
        "ATTN_k_proj": ("self_attn.k_proj.weight",),
        "ATTN_v_proj": ("self_attn.v_proj.weight",),
        "ATTN_o_proj": ("self_attn.o_proj.weight",),
        "ATTN_q_norm": ("self_attn.q_norm.weight", "self_attn.k_norm.weight"),
        "MLP_gate_proj": ("mlp.gate_proj.weight",),
        "MLP_up_proj": ("mlp.up_proj.weight",),
        "MLP_down_proj": ("mlp.down_proj.weight",),
        "NORMS_layernorm": ("input_layernorm.weight", "post_attention_layernorm.weight"),
    }

    @staticmethod
    def clean_key(key: str) -> str:
        for prefix in ("model.diffusion_model.", "diffusion_model.", "cond_stage_model.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        return key

    @classmethod
    def probe_architecture(cls, state_dict: dict) -> dict:
        """Dynamic Probing: programmatically inspects state_dict keys and shapes
        to discover block hierarchy, layer counts, and tensor types."""
        discovered_blocks = set()
        discovered_layers = set()

        for k in state_dict.keys():
            ck = cls.clean_key(k)
            parts = ck.split(".")
            for i, p in enumerate(parts):
                if p.isdigit():
                    idx = int(p)
                    prev_part = parts[i-1] if i > 0 else ""
                    if "block" in prev_part or prev_part == "blocks":
                        discovered_blocks.add(idx)
                    elif "layer" in prev_part or prev_part == "layers":
                        discovered_layers.add(idx)

        num_model_blocks = max(discovered_blocks) + 1 if discovered_blocks else Krea2Config.MAX_UNET_BLOCKS
        num_clip_layers = max(discovered_layers) + 1 if discovered_layers else Krea2Config.MAX_CLIP_LAYERS

        return {
            "num_model_blocks": num_model_blocks,
            "num_clip_layers": num_clip_layers,
        }

    @classmethod
    def extract_model_block_idx(cls, clean_key: str):
        """Dynamic inspection of model block index with regex fallback."""
        if any(prefix in clean_key for prefix in ["txtfusion", "txtmlp", "tmlp", "tproj", "first", "last"]):
            return None, clean_key
        parts = clean_key.split(".")
        for i, p in enumerate(parts):
            if p.isdigit() and i > 0 and parts[i-1] in ("blocks", "block"):
                return int(p), ".".join(parts[i+1:])
        m = RE_MODEL_BLOCKS_FULL.search(clean_key)
        if m:
            return int(m.group(1)), m.group(2)
        return None, clean_key

    @classmethod
    def extract_clip_layer_idx(cls, clean_key: str):
        """Dynamic inspection of CLIP layer index with regex fallback."""
        parts = clean_key.split(".")
        for i, p in enumerate(parts):
            if p.isdigit() and i > 0 and ("layer" in parts[i-1] or parts[i-1] == "layers"):
                return int(p), ".".join(parts[i+1:])
        m = RE_CLIP_LAYERS_FULL.search(clean_key)
        if m:
            return int(m.group(1)), m.group(2)
        return None, clean_key

    @classmethod
    def match_model_sub_tensor(cls, sub_key: str):
        for widget_key, target_suffixes in cls.MODEL_SURGEON_MAP.items():
            for target_suffix in target_suffixes:
                if (sub_key == target_suffix 
                    or sub_key.endswith(f".{target_suffix}") 
                    or sub_key.startswith(f"{target_suffix}.") 
                    or f".{target_suffix}." in sub_key):
                    return widget_key
        return None

    @classmethod
    def match_clip_sub_tensor(cls, sub_key: str):
        for widget_key, target_suffixes in cls.CLIP_SURGEON_MAP.items():
            for target_suffix in target_suffixes:
                if (sub_key == target_suffix 
                    or sub_key.endswith(f".{target_suffix}") 
                    or sub_key.startswith(f"{target_suffix}.") 
                    or f".{target_suffix}." in sub_key):
                    return widget_key
        return None

    @classmethod
    def get_descriptive_model_surgeon_map(cls):
        return dict(cls.MODEL_SURGEON_MAP)

    @classmethod
    def get_descriptive_clip_surgeon_map(cls):
        return dict(cls.CLIP_SURGEON_MAP)

# ==============================================================================
# POINT 1: NATIVE COMFYUI PATCHER ADAPTER (No UUID Monkey-Patching)
# ==============================================================================
class ComfyPatcherAdapter:
    """Safe wrapper for ComfyUI ModelPatcher operations using official APIs."""

    @staticmethod
    def isolate_patcher(model_or_clip):
        """Creates a clean, isolated patcher clone natively using ComfyUI API."""
        patcher = get_patcher(model_or_clip)
        return patcher.clone()

    @staticmethod
    def calculate_safe_weight(model_or_clip, key: str, base_weight: torch.Tensor, model_sd: dict = None) -> torch.Tensor:
        """Calculates active weight safely without mutating model state or accessing private attributes.
        Resolves exact internal state_dict key name and reuses pre-computed model_sd to guarantee O(1) performance."""
        patcher = get_patcher(model_or_clip)
        if model_sd is None and hasattr(patcher, "model") and hasattr(patcher.model, "state_dict"):
            try:
                model_sd = patcher.model.state_dict()
            except Exception:
                model_sd = {}

        resolved_key = resolve_target_key(patcher, key, model_sd=model_sd)
        
        current_patches = []
        if hasattr(patcher, "patches"):
            if resolved_key in patcher.patches:
                current_patches = patcher.patches[resolved_key]
            elif key in patcher.patches:
                current_patches = patcher.patches[key]

        scale = None
        if model_sd:
            for scale_cand in (f"{resolved_key}_scale", f"{resolved_key}.weight_scale", f"{key}_scale", f"{key}.weight_scale"):
                if scale_cand in model_sd:
                    scale = model_sd[scale_cand]
                    break

        clean_base = dequantize_weight(get_clean_weight(patcher, resolved_key, base_weight), scale=scale)
        if current_patches:
            return comfy.lora.calculate_weight(current_patches, clean_base.clone(), resolved_key).to(torch.bfloat16)
        return clean_base

# Pre-compiled regular expressions for high-frequency state-dict iterations
RE_MODEL_BLOCKS_FULL = re.compile(r"blocks\.(\d+)\.(.+)$")
RE_CLIP_LAYERS_FULL = re.compile(r"layers\.(\d+)\.(.+)$")
RE_TXTFUSION_LAYERWISE = re.compile(r"txtfusion\.layerwise_blocks\.(\d+)\.")
RE_TXTFUSION_REFINER = re.compile(r"txtfusion\.refiner_blocks\.(\d+)\.")
RE_GENERAL_BLOCKS = re.compile(r"blocks\.(\d+)\.")

# Minimal 2-byte dummy tensor for VBAR ComfyUI patch alignment
DUMMY_PATCH_TENSOR = torch.zeros((1, 1), dtype=torch.bfloat16)

def sanitize_patch_tensor(tensor: torch.Tensor, target_dtype: torch.dtype = None, target_device: torch.device = None) -> torch.Tensor:
    """Rules 1, 2, 3: Memory Safety, Dtype/Device Preservation & Empty Buffer Guard.
    
    1. Memory Safety: Enforce contiguous memory layout and detachment via .clone().detach().contiguous()
    2. Dtype & Device: Strict casting to target model dtype (bf16) and device
    3. Sanity Check: Prevent empty buffers (nelement == 0) to avoid torch.frombuffer memory crashes
    """
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return None

    # Rule 3: Sanity check to prevent empty buffers
    if tensor.nelement() == 0:
        logger.warning("[Arthemy Patch Guard] Detected empty tensor (nelement == 0). Patch ignored to prevent memory crash.")
        return None

    # Rule 2: Preserve and cast to target dtype and device
    dtype = target_dtype if target_dtype is not None else tensor.dtype
    device = target_device if target_device is not None else tensor.device

    # Apply Rule 1: Contiguity and detachment
    return tensor.to(dtype=dtype, device=device).clone().detach().contiguous()

def resolve_target_key(patcher, k: str, model_sd: dict = None) -> str:
    """Finds the exact state_dict key in patcher.model matching k with O(1) cached lookup."""
    if model_sd is None:
        if patcher is None or not hasattr(patcher, "model"):
            return k
        try:
            model_sd = patcher.model.state_dict()
        except Exception:
            return k

    if k in model_sd:
        return k

    clean_k = Krea2TensorParser.clean_key(k)
    if clean_k in model_sd:
        return clean_k

    candidates = (
        f"diffusion_model.{clean_k}",
        f"cond_stage_model.{clean_k}",
        f"model.{clean_k}",
        f"model.language_model.{clean_k}",
        f"clip_model.{clean_k}",
        f"transformer.{clean_k}",
    )

    for cand in candidates:
        if cand in model_sd:
            return cand

    return k

def add_patches_to_front(model_patcher, patches: dict, strength_patch: float = 1.0, strength_model: float = 1.0):
    """Safely injects patches into a ModelPatcher using standard ComfyUI add_patches API.
    
    Dynamically resolves exact key names in the model/clip state_dict and formats 
    patch entries as a 1-tuple (diff,) so comfy.lora.calculate_weight sees len(v) == 1 and patch_type == 'diff'.
    Enforces contiguous memory layout, dtype matching, and empty buffer guards on all injected patches.
    """
    patcher = get_patcher(model_patcher)
    if patcher is None or not hasattr(patcher, "add_patches"):
        return []

    model_sd = patcher.model.state_dict() if hasattr(patcher, "model") and hasattr(patcher.model, "state_dict") else {}
    p_keys = set()

    for k, val in patches.items():
        target_k = resolve_target_key(patcher, k, model_sd=model_sd)
        base_w = model_sd.get(target_k, None)
        target_dtype = base_w.dtype if isinstance(base_w, torch.Tensor) else torch.bfloat16
        target_device = base_w.device if isinstance(base_w, torch.Tensor) else torch.device("cpu")

        scalar_mult = None
        tensor_payload = None

        if isinstance(val, (tuple, list)):
            if len(val) == 1:
                inner = val[0]
                if isinstance(inner, (int, float)):
                    scalar_mult = float(inner)
                elif isinstance(inner, torch.Tensor) and inner.numel() == 1:
                    scalar_mult = float(inner.item())
                elif isinstance(inner, torch.Tensor):
                    sanitary_t = sanitize_patch_tensor(inner, target_dtype, target_device)
                    if sanitary_t is not None:
                        tensor_payload = (sanitary_t,)
                else:
                    tensor_payload = (inner,)
            elif len(val) == 2:
                t_list = []
                valid_payload = True
                for item in val:
                    if isinstance(item, torch.Tensor):
                        san = sanitize_patch_tensor(item, target_dtype, target_device)
                        if san is None:
                            valid_payload = False
                            break
                        t_list.append(san)
                    else:
                        t_list.append(item)
                if valid_payload:
                    tensor_payload = tuple(t_list)
            else:
                inner = val[0]
                if isinstance(inner, torch.Tensor):
                    san = sanitize_patch_tensor(inner, target_dtype, target_device)
                    if san is not None:
                        tensor_payload = (san,)
                else:
                    tensor_payload = (inner,)
        elif isinstance(val, (int, float)):
            scalar_mult = float(val)
        elif isinstance(val, torch.Tensor):
            if val.numel() == 1:
                scalar_mult = float(val.item())
            else:
                san = sanitize_patch_tensor(val, target_dtype, target_device)
                if san is not None:
                    tensor_payload = (san,)

        if scalar_mult is not None:
            # Shape-matched base weight tensor for VBAR hostbuf memory alignment (> 0 bytes)
            if isinstance(base_w, torch.Tensor) and base_w.numel() > 1:
                patch_tensor = sanitize_patch_tensor(base_w, target_dtype, target_device)
            else:
                patch_tensor = DUMMY_PATCH_TENSOR

            if patch_tensor is not None and patch_tensor.nelement() > 0:
                formatted = {target_k: (patch_tensor,)}
                patcher.add_patches(formatted, strength_patch=0.0, strength_model=scalar_mult)
                p_keys.add(target_k)
            else:
                logger.warning(f"[Arthemy Patch Guard] Skipping scalar patch for key '{target_k}': empty tensor buffer.")

        elif tensor_payload is not None:
            formatted = {target_k: tensor_payload}
            patcher.add_patches(formatted, strength_patch=strength_patch, strength_model=strength_model)
            p_keys.add(target_k)

    return list(p_keys)

# ==============================================================================
# BASE NODE & UTILITIES
# ==============================================================================

def get_patcher(obj):
    """Safely retrieves the ModelPatcher/ModelPatcherDynamic object whether obj is a model/clip wrapper or patcher directly."""
    if obj is None:
        return None
    return getattr(obj, "patcher", obj)


class BaseKrea2Node:
    """Base class for all Arthemy Krea-2 custom nodes."""
    @classmethod
    def format_telemetry_info(cls, title: str, active_patches: int, details: str = "") -> str:
        msg = f"{title} | Active Patches: {active_patches}"
        if details:
            msg += f" | {details}"
        logger.info(f"[Arthemy Telemetry] {msg}")
        return msg

SOFT_DAMPENING_FACTOR = 0.10

def soft_target_weight(delta: float, mode: str = "Soft Value") -> float:
    """Calculates target weight according to selected mode:
    - Real Value: 1-to-1 direct linear scaling (1.0 + delta).
    - Soft Value: Precision micro-tuning with linear 10% dampening factor (1.0 + delta * 0.10).
    Guarantees 100% predictable linear behavior in both modes without non-linear distortion.
    """
    if mode == "Real Value":
        return 1.0 + delta
    return 1.0 + (delta * SOFT_DAMPENING_FACTOR)

def get_clean_weight(patcher, key: str, current_weight: torch.Tensor) -> torch.Tensor:
    if hasattr(patcher, "backup") and key in patcher.backup:
        b = patcher.backup[key]
        if isinstance(b, tuple) and len(b) > 0:
            return b[0] if isinstance(b[0], torch.Tensor) else current_weight
        if isinstance(b, torch.Tensor):
            return b
    return current_weight

def dequantize_weight(weight: Any, scale: Any = None) -> torch.Tensor:
    """Converts FP8 tensors to FP32, multiplying by companion scale factor if present."""
    if weight is None or not isinstance(weight, torch.Tensor):
        return weight
    if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        fp32_w = weight.to(torch.float32)
        if scale is not None:
            if isinstance(scale, torch.Tensor):
                scale_val = scale.to(device=weight.device, dtype=torch.float32)
            else:
                scale_val = float(scale)
            return fp32_w * scale_val
        return fp32_w
    return weight

def resolve_granular_weight(clean_key: str, granular_map: dict, default_val: float) -> float:
    if not granular_map:
        return default_val
    if clean_key in granular_map:
        return float(granular_map[clean_key])
    for pfx, val in granular_map.items():
        if pfx and (clean_key.startswith(pfx) or f".{pfx}" in clean_key):
            return float(val)
    return default_val

# ==============================================================================
# POINT 4: DRY UNIFIED BASE SURGEON TUNER (BaseSurgeonTuner)
# ==============================================================================
class BaseSurgeonTuner(BaseKrea2Node):
    """Abstract Base Class for Model and CLIP Block Surgeon and Chaos Surgeon Tuners.
    Encapsulates state_dict iteration, soft target weight calculation (linear micro-tuning),
    sub-key matching, seed hashing, and patch injection to enforce DRY architecture."""

    def _execute_surgeon_tuning(
        self,
        model_or_clip,
        is_clip: bool,
        selected_indices: set,
        surgeon_map: dict,
        kwargs: dict,
        mode: str = "Soft Value",
        vectors_override: str = "",
        granular_json: str = "",
        chaos_params: dict = None
    ):
        patcher = get_patcher(model_or_clip)
        clone_obj = model_or_clip.clone()
        base_sd = patcher.model.state_dict() if hasattr(patcher, "model") else {}
        valid_keys = getattr(patcher, "model_keys", None)

        granular_map = parse_granular_json(granular_json)

        slider_map = {w_key: kwargs.get(w_key, 0.0) for w_key in surgeon_map.keys()}
        patches_to_add = {}
        active_count = 0

        for k, base_weight in base_sd.items():
            if k.endswith(".position_ids") or k.endswith(".logit_scale") or k.endswith(".comfy_quant") or f"{k}_scale" in base_sd:
                continue
            if valid_keys is not None and k not in valid_keys:
                continue

            clean_k = Krea2TensorParser.clean_key(k)
            if is_clip:
                idx, sub_key = Krea2TensorParser.extract_clip_layer_idx(clean_k)
                matched_widget_key = Krea2TensorParser.match_clip_sub_tensor(sub_key)
            else:
                idx, sub_key = Krea2TensorParser.extract_model_block_idx(clean_k)
                matched_widget_key = Krea2TensorParser.match_model_sub_tensor(sub_key)

            if idx is None or idx not in selected_indices or not matched_widget_key:
                continue

            target_patch_key = f"diffusion_model.{clean_k}" if (not is_clip and not clean_k.startswith("diffusion_model.")) else clean_k

            if chaos_params is not None:
                chance_val = kwargs.get(matched_widget_key, 0.5)
                if chance_val <= 0.0: continue

                seed = chaos_params.get("seed", 42)
                chaos_strength = chaos_params.get("chaos_strength", 0.1)
                tune_mode = chaos_params.get("tune_mode", "Block-Level")

                fast_seed = generate_fast_seed(k, seed)
                rng = torch.Generator(device=base_weight.device)
                rng.manual_seed(fast_seed)

                if tune_mode == "Element-Level (Sub-atomic)":
                    mask = torch.rand(base_weight.shape, generator=rng, device=base_weight.device) < chance_val
                else:
                    v_rand = torch.rand(1, generator=rng, device=base_weight.device).item()
                    mask = torch.ones_like(base_weight, dtype=torch.bool) if v_rand < chance_val else torch.zeros_like(base_weight, dtype=torch.bool)

                if not torch.any(mask): continue

                w_active = ComfyPatcherAdapter.calculate_safe_weight(model_or_clip, target_patch_key, base_weight, model_sd=base_sd)
                delta = (w_active.to("cpu") * chaos_strength) * mask.to("cpu").to(torch.bfloat16)
                patches_to_add[target_patch_key] = (delta.to(torch.bfloat16),)
                active_count += 1
            else:
                raw_target = slider_map.get(matched_widget_key, 0.0)
                if granular_map:
                    raw_target = resolve_granular_weight(clean_k, granular_map, raw_target)

                target_w = soft_target_weight(raw_target, mode)
                strength = target_w - 1.0
                if strength != 0:
                    patches_to_add[target_patch_key] = (1.0 + strength,)
                    active_count += 1

        if patches_to_add:
            t_start = time.time()
            add_patches_to_front(clone_obj, patches_to_add, 1.0)

            # Record deterministic Chaos recipe for preset persistence
            if chaos_params is not None:
                clone_patcher = get_patcher(clone_obj)
                if hasattr(clone_patcher, "model_options"):
                    if "arthemy_chaos_recipes" not in clone_patcher.model_options:
                        clone_patcher.model_options["arthemy_chaos_recipes"] = []
                    else:
                        clone_patcher.model_options["arthemy_chaos_recipes"] = list(clone_patcher.model_options["arthemy_chaos_recipes"])
                    
                    recipe_entry = {
                        "domain": "clip" if is_clip else "model",
                        "selected_indices": sorted(list(selected_indices)),
                        "tune_mode": chaos_params.get("tune_mode", "Block-Level"),
                        "seed": chaos_params.get("seed", 42),
                        "chaos_strength": chaos_params.get("chaos_strength", 0.1),
                        "chances": {k: v for k, v in kwargs.items() if v > 0.0}
                    }
                    clone_patcher.model_options["arthemy_chaos_recipes"].append(recipe_entry)

            logger.info(f"[Arthemy Profiler] {'CLIP' if is_clip else 'Model'} Surgeon | "
                        f"{time.time()-t_start:.4f}s | {active_count} patches | selected: {sorted(selected_indices)}")

        desc_label = f"{'Layers' if is_clip else 'Blocks'} {sorted(selected_indices)}"
        return clone_obj, f"Arthemy Krea-2 {'CLIP' if is_clip else 'Model'} Surgeon ({desc_label}) | Patches: {active_count}"

# ==============================================================================
# POINT 3: MEMORY-SAFE GENERATOR & STREAMING PROCESSING
# ==============================================================================
def process_tensor_stream(
    tensor_dict: Dict[str, Any], 
    patch_fn, 
    target_dtype: torch.dtype = torch.bfloat16, 
    memory_threshold_mb: int = Krea2Config.DEFAULT_GC_THRESHOLD_MB
) -> Generator[Tuple[str, Any], None, None]:
    """Yields (clean_key, processed_tensor) while tracking cumulative memory.
    Triggers GC and clears CUDA cache only when processed data exceeds threshold_bytes."""
    threshold_bytes = memory_threshold_mb * 1024 * 1024
    accumulated_bytes = 0

    for k, v in tensor_dict.items():
        if k.endswith(".comfy_quant") or f"{k}_scale" in tensor_dict:
            continue
        clean_k = Krea2TensorParser.clean_key(k)

        if isinstance(v, torch.Tensor):
            accumulated_bytes += v.element_size() * v.nelement()

        processed = patch_fn(k, v)

        if isinstance(processed, torch.Tensor):
            if processed.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
                out_tensor = processed.cpu().clone()
            else:
                out_tensor = processed.to(target_dtype).cpu().clone()
            del processed
        else:
            out_tensor = processed

        yield clean_k, out_tensor

        if accumulated_bytes >= threshold_bytes:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            accumulated_bytes = 0

# ==============================================================================
# 1. ARTHEMY KREA2 MODEL TUNER
# ==============================================================================
class ArthemyKrea2ModelTuner(BaseKrea2Node):
    GROUP_MAP = {
        "Text_Fusion": ["txtfusion.", "txtmlp."],
        "Time_Embed": ["tmlp.", "tproj."],
        "Projection": ["first.", "last."],
        "Block_1": ["blocks.0.", "blocks.1.", "blocks.2.", "blocks.3.", "blocks.4."],
        "Block_2": ["blocks.5.", "blocks.6.", "blocks.7.", "blocks.8.", "blocks.9."],
        "Block_3": ["blocks.10.", "blocks.11.", "blocks.12.", "blocks.13.", "blocks.14."],
        "Block_4": ["blocks.15.", "blocks.16.", "blocks.17.", "blocks.18.", "blocks.19."],
        "Block_5": ["blocks.20.", "blocks.21.", "blocks.22.", "blocks.23."],
        "Block_6": ["blocks.24.", "blocks.25.", "blocks.26.", "blocks.27."],
    }
    ORDERED_KEYS = list(GROUP_MAP.keys())

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model": ("MODEL",), "mode": (["Soft Value", "Real Value"],),
                "vectors_override": ("STRING", {"default": "", "multiline": False}),
                "granular_json": ("STRING", {"default": "", "multiline": True}),
            }
        }
        for name in s.ORDERED_KEYS:
            inputs["required"][name] = ("FLOAT", {"default": 0.00, "min": -99.00, "max": 99.00, "step": 0.01,
                                                    "tooltip": "Offset from baseline (0.00 = no change). Displayed value + 1.00 = actual multiplier."})
        return inputs

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "tune_model"
    CATEGORY = "Arthemy/Krea2 Tuners"

    def tune_model(self, model, mode, vectors_override, granular_json="", **kwargs):
        def get_target_weight(delta): return soft_target_weight(delta, mode)
        final_weights = [1.0] * 34
        use_vector = False

        if vectors_override.strip():
            try:
                v_vals = [float(v.strip()) for v in vectors_override.split(',') if v.strip()]
                if len(v_vals) == 34:
                    final_weights = [get_target_weight(v) for v in v_vals]
                    use_vector = True
                else:
                    logger.warning(f"[Arthemy Model Tuner] vectors_override dimension mismatch: expected 34 values, got {len(v_vals)}. Override ignored.")
            except ValueError as e:
                logger.warning(f"[Arthemy Model Tuner] vectors_override parse error: {e}. Override ignored.")

        granular_map = parse_granular_json(granular_json)

        w_base = get_target_weight(0.0)
        m = model.clone()
        base_sd = m.model.state_dict()
        patches_to_add = {}
        active_patches = 0
        valid_keys = m.model_keys if hasattr(m, "model_keys") else None
        tasks = []

        for k, base_weight in base_sd.items():
            if k.endswith(".comfy_quant") or f"{k}_scale" in base_sd:
                continue

            target_weight = w_base
            clean_key = Krea2TensorParser.clean_key(k)

            if granular_map:
                raw_target = resolve_granular_weight(clean_key, granular_map, 0.0)
                target_weight = get_target_weight(raw_target)
            elif use_vector:
                if "txtfusion.layerwise_blocks" in clean_key:
                    match = RE_TXTFUSION_LAYERWISE.search(clean_key)
                    if match: target_weight = final_weights[28 + int(match.group(1))]
                elif "txtfusion.refiner_blocks" in clean_key:
                    match = RE_TXTFUSION_REFINER.search(clean_key)
                    if match: target_weight = final_weights[30 + int(match.group(1))]
                elif "txtfusion.projector" in clean_key:
                    target_weight = final_weights[32]
                elif "txtmlp" in clean_key:
                    target_weight = final_weights[33]
                else:
                    match = RE_GENERAL_BLOCKS.search(clean_key)
                    if match: target_weight = final_weights[int(match.group(1))]
            else:
                matched_group = None
                for group_name, prefixes in self.GROUP_MAP.items():
                    if any(clean_key.startswith(pfx) or f".{pfx}" in clean_key for pfx in prefixes):
                        matched_group = group_name
                        break
                if matched_group:
                    target_weight = get_target_weight(kwargs.get(matched_group, 0.0))

            strength = target_weight - 1.0
            if strength != 0:
                if valid_keys is not None and k not in valid_keys: continue
                tasks.append((k, base_weight, strength))

        if tasks:
            t_start = time.time()
            for k, base_weight, strength in tasks:
                patch_key = f"diffusion_model.{k}" if not k.startswith("diffusion_model.") else k
                patches_to_add[patch_key] = (1.0 + strength,)
                active_patches += 1
            logger.info(f"[Arthemy Profiler] Model Tuner finished | Total time: {time.time() - t_start:.4f}s | Layers processed: {active_patches}")

        if patches_to_add:
            add_patches_to_front(m, patches_to_add, 1.0)
        return (m, f"Krea-2 Model Tuned | Patches: {active_patches}")

# ==============================================================================
# 2. ARTHEMY KREA2 CLIP TUNER
# ==============================================================================
class ArthemyKrea2CLIPTuner(BaseKrea2Node):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",), "mode": (["Soft Value", "Real Value"],),
                "vectors_override": ("STRING", {"default": "", "multiline": False}),
                "granular_json": ("STRING", {"default": "", "multiline": True}),
                "Embedding": ("FLOAT", {"default": 0.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "Layer_1": ("FLOAT", {"default": 0.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "Layer_2": ("FLOAT", {"default": 0.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "Layer_3": ("FLOAT", {"default": 0.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "Layer_4": ("FLOAT", {"default": 0.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "Layer_5": ("FLOAT", {"default": 0.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "Layer_6": ("FLOAT", {"default": 0.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "Layer_7": ("FLOAT", {"default": 0.0, "min": -99.00, "max": 99.00, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("CLIP", "info")
    FUNCTION = "tune_clip"
    CATEGORY = "Arthemy/Krea2 Tuners"

    def tune_clip(self, clip, mode, vectors_override, granular_json="",
                  Embedding=0.0, Layer_1=0.0, Layer_2=0.0, Layer_3=0.0,
                  Layer_4=0.0, Layer_5=0.0, Layer_6=0.0, Layer_7=0.0, **kwargs):
        def get_target_weight(delta): return soft_target_weight(delta, mode)
        final_weights = [1.0] * 8
        use_vector = False

        if vectors_override.strip():
            try:
                v_vals = [float(v.strip()) for v in vectors_override.split(',') if v.strip()]
                if len(v_vals) == 8:
                    final_weights = [get_target_weight(v) for v in v_vals]
                    use_vector = True
                else:
                    logger.warning(f"[Arthemy CLIP Tuner] vectors_override dimension mismatch: expected 8 values, got {len(v_vals)}. Override ignored.")
            except ValueError as e:
                logger.warning(f"[Arthemy CLIP Tuner] vectors_override parse error: {e}. Override ignored.")

        granular_map = parse_granular_json(granular_json)

        w_vocab  = final_weights[0] if use_vector else get_target_weight(Embedding)
        w_b1     = final_weights[1] if use_vector else get_target_weight(Layer_1)
        w_b2     = final_weights[2] if use_vector else get_target_weight(Layer_2)
        w_b3     = final_weights[3] if use_vector else get_target_weight(Layer_3)
        w_b4     = final_weights[4] if use_vector else get_target_weight(Layer_4)
        w_b5     = final_weights[5] if use_vector else get_target_weight(Layer_5)
        w_b6     = final_weights[6] if use_vector else get_target_weight(Layer_6)
        w_b7     = final_weights[7] if use_vector else get_target_weight(Layer_7)

        c = clip.clone()
        c_p = get_patcher(c)
        base_sd = c_p.model.state_dict() if hasattr(c_p, 'model') else getattr(c, 'get_sd', lambda: {})()
        patches_to_add = {}
        active_patches = 0
        valid_keys = getattr(c_p, 'model_keys', None)



        def get_block_weight(idx):
            if idx < 5:  return w_b1
            if idx < 10: return w_b2
            if idx < 15: return w_b3
            if idx < 20: return w_b4
            if idx < 25: return w_b5
            if idx < 30: return w_b6
            return w_b7

        tasks = []
        for k, base_weight in base_sd.items():
            if k.endswith(".position_ids") or k.endswith(".logit_scale") or f"{k}_scale" in base_sd: continue

            target_scale = 1.0
            if granular_map:
                raw_target = resolve_granular_weight(k, granular_map, 0.0)
                target_scale = get_target_weight(raw_target)
            elif "embed_tokens" in k:
                target_scale = w_vocab
            else:
                idx, _ = Krea2TensorParser.extract_clip_layer_idx(Krea2TensorParser.clean_key(k))
                if idx is not None:
                    target_scale = get_block_weight(idx)

            strength = target_scale - 1.0
            if strength != 0:
                if valid_keys is not None and k not in valid_keys: continue
                tasks.append((k, base_weight, strength))

        if tasks:
            t_start = time.time()
            for k, base_weight, strength in tasks:
                patches_to_add[k] = (1.0 + strength,)
                active_patches += 1
            logger.info(f"[Arthemy Profiler] CLIP Tuner finished | Total time: {time.time() - t_start:.4f}s | Layers processed: {active_patches}")

        if patches_to_add:
            add_patches_to_front(c, patches_to_add, 1.0)
        return (c, f"Krea-2 CLIP Tuned | Patches: {active_patches}")



# ==============================================================================
# POINT 4 IMPLEMENTATIONS: SURGEON NODES DERIVED FROM BaseSurgeonTuner
# ==============================================================================
class ArthemyKrea2ModelBlockSurgeonTuner(BaseSurgeonTuner):
    MODEL_TARGET_MAP = {
        "All Blocks (0-27)": set(range(0, 28)),
        "Block_1 (All 0-4)": set(range(0, 5)),
        "  ↳ Block_1A (0)": {0},
        "  ↳ Block_1B (1)": {1},
        "  ↳ Block_1C (2)": {2},
        "  ↳ Block_1D (3)": {3},
        "  ↳ Block_1E (4)": {4},
        "Block_2 (All 5-9)": set(range(5, 10)),
        "  ↳ Block_2A (5)": {5},
        "  ↳ Block_2B (6)": {6},
        "  ↳ Block_2C (7)": {7},
        "  ↳ Block_2D (8)": {8},
        "  ↳ Block_2E (9)": {9},
        "Block_3 (All 10-14)": set(range(10, 15)),
        "  ↳ Block_3A (10)": {10},
        "  ↳ Block_3B (11)": {11},
        "  ↳ Block_3C (12)": {12},
        "  ↳ Block_3D (13)": {13},
        "  ↳ Block_3E (14)": {14},
        "Block_4 (All 15-19)": set(range(15, 20)),
        "  ↳ Block_4A (15)": {15},
        "  ↳ Block_4B (16)": {16},
        "  ↳ Block_4C (17)": {17},
        "  ↳ Block_4D (18)": {18},
        "  ↳ Block_4E (19)": {19},
        "Block_5 (All 20-23)": set(range(20, 24)),
        "  ↳ Block_5A (20)": {20},
        "  ↳ Block_5B (21)": {21},
        "  ↳ Block_5C (22)": {22},
        "  ↳ Block_5D (23)": {23},
        "Block_6 (All 24-27)": set(range(24, 28)),
        "  ↳ Block_6A (24)": {24},
        "  ↳ Block_6B (25)": {25},
        "  ↳ Block_6C (26)": {26},
        "  ↳ Block_6D (27)": {27},
    }
    SURGEON_MAP = Krea2TensorParser.get_descriptive_model_surgeon_map()

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model": ("MODEL",),
                "target_block": (list(s.MODEL_TARGET_MAP.keys()), {"default": "Block_1 (All 0-4)"}),
                "mode": (["Soft Value", "Real Value"],),
                "vectors_override": ("STRING", {"default": "", "multiline": False}),
                "granular_json": ("STRING", {"default": "", "multiline": True}),
            }
        }
        for name in s.SURGEON_MAP.keys():
            inputs["required"][name] = ("FLOAT", {"default": 0.0, "min": -99.00, "max": 99.00, "step": 0.01})
        return inputs

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "tune_model_surgeon"
    CATEGORY = "Arthemy/Krea2 Tuners"

    def tune_model_surgeon(self, model, target_block, mode, vectors_override, granular_json="", **kwargs):
        selected = self.MODEL_TARGET_MAP.get(target_block, None)
        if selected is None:
            # Graceful fallback for legacy workflows
            clean = target_block.strip().lstrip("↳").strip()
            for k, v in self.MODEL_TARGET_MAP.items():
                if clean in k:
                    selected = v
                    break
        if selected is None:
            selected = set(range(0, Krea2Config.MAX_UNET_BLOCKS))

        return self._execute_surgeon_tuning(
            model, is_clip=False, selected_indices=selected, surgeon_map=self.SURGEON_MAP,
            kwargs=kwargs, mode=mode, vectors_override=vectors_override, granular_json=granular_json
        )


class ArthemyKrea2CLIPBlockSurgeonTuner(BaseSurgeonTuner):
    CLIP_TARGET_MAP = {
        "All Layers (0-35)": set(range(0, 36)),
        "Layer_1 (All 0-4)": set(range(0, 5)),
        "  ↳ Layer_1A (0)": {0},
        "  ↳ Layer_1B (1)": {1},
        "  ↳ Layer_1C (2)": {2},
        "  ↳ Layer_1D (3)": {3},
        "  ↳ Layer_1E (4)": {4},
        "Layer_2 (All 5-9)": set(range(5, 10)),
        "  ↳ Layer_2A (5)": {5},
        "  ↳ Layer_2B (6)": {6},
        "  ↳ Layer_2C (7)": {7},
        "  ↳ Layer_2D (8)": {8},
        "  ↳ Layer_2E (9)": {9},
        "Layer_3 (All 10-14)": set(range(10, 15)),
        "  ↳ Layer_3A (10)": {10},
        "  ↳ Layer_3B (11)": {11},
        "  ↳ Layer_3C (12)": {12},
        "  ↳ Layer_3D (13)": {13},
        "  ↳ Layer_3E (14)": {14},
        "Layer_4 (All 15-19)": set(range(15, 20)),
        "  ↳ Layer_4A (15)": {15},
        "  ↳ Layer_4B (16)": {16},
        "  ↳ Layer_4C (17)": {17},
        "  ↳ Layer_4D (18)": {18},
        "  ↳ Layer_4E (19)": {19},
        "Layer_5 (All 20-24)": set(range(20, 25)),
        "  ↳ Layer_5A (20)": {20},
        "  ↳ Layer_5B (21)": {21},
        "  ↳ Layer_5C (22)": {22},
        "  ↳ Layer_5D (23)": {23},
        "  ↳ Layer_5E (24)": {24},
        "Layer_6 (All 25-29)": set(range(25, 30)),
        "  ↳ Layer_6A (25)": {25},
        "  ↳ Layer_6B (26)": {26},
        "  ↳ Layer_6C (27)": {27},
        "  ↳ Layer_6D (28)": {28},
        "  ↳ Layer_6E (29)": {29},
        "Layer_7 (All 30-35)": set(range(30, 36)),
        "  ↳ Layer_7A (30)": {30},
        "  ↳ Layer_7B (31)": {31},
        "  ↳ Layer_7C (32)": {32},
        "  ↳ Layer_7D (33)": {33},
        "  ↳ Layer_7E (34)": {34},
        "  ↳ Layer_7F (35)": {35},
    }
    SURGEON_MAP = Krea2TensorParser.get_descriptive_clip_surgeon_map()

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "clip": ("CLIP",),
                "target_layer": (list(s.CLIP_TARGET_MAP.keys()), {"default": "Layer_1 (All 0-4)"}),
                "mode": (["Soft Value", "Real Value"],),
                "vectors_override": ("STRING", {"default": "", "multiline": False}),
                "granular_json": ("STRING", {"default": "", "multiline": True}),
            }
        }
        for name in s.SURGEON_MAP.keys():
            inputs["required"][name] = ("FLOAT", {"default": 0.0, "min": -99.00, "max": 99.00, "step": 0.01})
        return inputs

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("CLIP", "info")
    FUNCTION = "tune_clip_surgeon"
    CATEGORY = "Arthemy/Krea2 Tuners"

    def tune_clip_surgeon(self, clip, target_layer, mode, vectors_override, granular_json="", **kwargs):
        selected = self.CLIP_TARGET_MAP.get(target_layer, None)
        if selected is None:
            clean = target_layer.strip().lstrip("↳").strip()
            for k, v in self.CLIP_TARGET_MAP.items():
                if clean in k:
                    selected = v
                    break
        if selected is None:
            selected = set(range(0, Krea2Config.MAX_CLIP_LAYERS))

        return self._execute_surgeon_tuning(
            clip, is_clip=True, selected_indices=selected, surgeon_map=self.SURGEON_MAP,
            kwargs=kwargs, mode=mode, vectors_override=vectors_override, granular_json=granular_json
        )


class ArthemyKrea2ModelChaosBlockSurgeonTuner(BaseSurgeonTuner):
    MODEL_TARGET_MAP = ArthemyKrea2ModelBlockSurgeonTuner.MODEL_TARGET_MAP
    _CHANCE_MAP = {v: f"{k}_chance" for k, v in Krea2TensorParser.MODEL_SURGEON_MAP.items()}

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model": ("MODEL",),
                "target_block": (list(s.MODEL_TARGET_MAP.keys()), {"default": "Block_1 (All 0-4)"}),
                "tune_mode": (["Block-Level", "Element-Level (Sub-atomic)"],),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "chaos_strength": ("FLOAT", {"default": 0.1, "min": -99.00, "max": 99.00, "step": 0.01}),
            }
        }
        for name in Krea2TensorParser.MODEL_SURGEON_MAP.keys():
            inputs["required"][f"{name}_chance"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01})
        return inputs

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "chaos_tune_model_surgeon"
    CATEGORY = "Arthemy/Krea2 Tuners"

    def chaos_tune_model_surgeon(self, model, target_block, tune_mode, seed, chaos_strength, **kwargs):
        selected = self.MODEL_TARGET_MAP.get(target_block, None)
        if selected is None:
            clean = target_block.strip().lstrip("↳").strip()
            for k, v in self.MODEL_TARGET_MAP.items():
                if clean in k:
                    selected = v
                    break
        if selected is None:
            selected = set(range(0, Krea2Config.MAX_UNET_BLOCKS))

        # Remap kwargs chances to standard surgeon_map names
        remapped_kwargs = {k.replace("_chance", ""): v for k, v in kwargs.items() if k.endswith("_chance")}

        chaos_params = {"tune_mode": tune_mode, "seed": seed, "chaos_strength": chaos_strength}
        return self._execute_surgeon_tuning(
            model, is_clip=False, selected_indices=selected, surgeon_map=Krea2TensorParser.MODEL_SURGEON_MAP,
            kwargs=remapped_kwargs, chaos_params=chaos_params
        )


class ArthemyKrea2CLIPChaosBlockSurgeonTuner(BaseSurgeonTuner):
    CLIP_TARGET_MAP = ArthemyKrea2CLIPBlockSurgeonTuner.CLIP_TARGET_MAP
    _CHANCE_MAP = {v: f"{k}_chance" for k, v in Krea2TensorParser.CLIP_SURGEON_MAP.items()}

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "clip": ("CLIP",),
                "target_layer": (list(s.CLIP_TARGET_MAP.keys()), {"default": "Layer_1 (All 0-4)"}),
                "tune_mode": (["Block-Level", "Element-Level (Sub-atomic)"],),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "chaos_strength": ("FLOAT", {"default": 0.1, "min": -99.00, "max": 99.00, "step": 0.01}),
            }
        }
        for name in Krea2TensorParser.CLIP_SURGEON_MAP.keys():
            inputs["required"][f"{name}_chance"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01})
        return inputs

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("CLIP", "info")
    FUNCTION = "chaos_tune_clip_surgeon"
    CATEGORY = "Arthemy/Krea2 Tuners"

    def chaos_tune_clip_surgeon(self, clip, target_layer, tune_mode, seed, chaos_strength, **kwargs):
        selected = self.CLIP_TARGET_MAP.get(target_layer, None)
        if selected is None:
            clean = target_layer.strip().lstrip("↳").strip()
            for k, v in self.CLIP_TARGET_MAP.items():
                if clean in k:
                    selected = v
                    break
        if selected is None:
            selected = set(range(0, Krea2Config.MAX_CLIP_LAYERS))

        remapped_kwargs = {k.replace("_chance", ""): v for k, v in kwargs.items() if k.endswith("_chance")}
        chaos_params = {"tune_mode": tune_mode, "seed": seed, "chaos_strength": chaos_strength}
        return self._execute_surgeon_tuning(
            clip, is_clip=True, selected_indices=selected, surgeon_map=Krea2TensorParser.CLIP_SURGEON_MAP,
            kwargs=remapped_kwargs, chaos_params=chaos_params
        )

# ==============================================================================
# POINT 3 IMPLEMENTATIONS: MEMORY-SAFE MODEL & CLIP SAVERS
# ==============================================================================
class ArthemyKrea2ModelSaver(BaseKrea2Node):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "output_checkpoint": ("STRING", {"default": "arthemy_krea2_model.safetensors"}),
                "precision": (["BF16", "FP8_E4M3"], {"default": "BF16"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "Arthemy/Krea2 Savers"

    def save(self, model, output_checkpoint, precision="BF16", prompt=None, extra_pnginfo=None):
        clean_name = output_checkpoint.strip()
        if clean_name.endswith(".safetensors"):
            clean_name = clean_name[:-len(".safetensors")]

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            clean_name, folder_paths.get_output_directory(), 0, 0
        )
        
        # Support auto-incrementing counter if format string or existing files exist
        if "%count%" in clean_name:
            file_name_out = clean_name.replace("%count%", f"{counter:05}") + ".safetensors"
        else:
            file_name_out = f"{filename}_{counter:05}_.safetensors"

        output_path = os.path.join(full_output_folder, file_name_out)
        os.makedirs(full_output_folder, exist_ok=True)
        logger.info(f"[ARTHEMY KREA2 MODEL SAVER] Stream-saving model: {file_name_out}")

        sd = model.model.state_dict()
        arch_info = Krea2TensorParser.probe_architecture(sd)
        if arch_info['num_model_blocks'] != Krea2Config.MAX_UNET_BLOCKS:
            logger.warning(f"[ARTHEMY KREA2 MODEL SAVER] ⚠️ Topology warning: detected {arch_info['num_model_blocks']} blocks (expected baseline: {Krea2Config.MAX_UNET_BLOCKS}).")
        else:
            logger.info(f"[ARTHEMY KREA2 MODEL SAVER] Topology validated: {arch_info['num_model_blocks']} blocks detected.")

        target_dtype = torch.float8_e4m3fn if precision == "FP8_E4M3" else torch.bfloat16
        final_sd = {}
        quant_layers = {}

        # Stream process tensors one by one to keep memory low
        def process_tensor(k, v):
            clean_k = Krea2TensorParser.clean_key(k)
            if k.endswith(".comfy_quant"):
                try:
                    quant_layers[clean_k[:-len(".comfy_quant")]] = json.loads(bytes(v.cpu().numpy()).decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError, AttributeError) as e:
                    logger.warning(f"[ARTHEMY KREA2 MODEL SAVER] Quantization metadata skip for '{k}': {type(e).__name__} - {e}")
                return None
            
            # Lookup scale if present
            scale = None
            for s_cand in (f"{k}_scale", f"{k}.weight_scale", f"{clean_k}_scale", f"{clean_k}.weight_scale"):
                if s_cand in sd:
                    scale = sd[s_cand]
                    break
            return dequantize_weight(v, scale=scale)

        for clean_key, tensor in process_tensor_stream(sd, process_tensor, target_dtype=target_dtype, memory_threshold_mb=Krea2Config.DEFAULT_GC_THRESHOLD_MB):
            if tensor is not None:
                final_sd[clean_key] = tensor

        metadata = {"format": "pt"}
        if len(quant_layers) > 0:
            metadata["_quantization_metadata"] = json.dumps({"layers": quant_layers})
        if extra_pnginfo and "workflow" in extra_pnginfo:
            metadata["workflow"] = json.dumps(extra_pnginfo["workflow"])

        safetensors.torch.save_file(final_sd, output_path, metadata=metadata)
        del final_sd
        gc.collect()
        logger.info(f"[ARTHEMY KREA2 MODEL SAVER] ✨ SUCCESS: {output_path}")
        return (output_path,)


class ArthemyKrea2CLIPSaver(BaseKrea2Node):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "output_checkpoint": ("STRING", {"default": "arthemy_qwen3_clip.safetensors"}),
                "precision": (["BF16", "FP8_E4M3"], {"default": "BF16"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save"
    CATEGORY = "Arthemy/Krea2 Savers"

    def save(self, clip, output_checkpoint, precision="BF16", prompt=None, extra_pnginfo=None):
        clean_name = output_checkpoint.strip()
        if clean_name.endswith(".safetensors"):
            clean_name = clean_name[:-len(".safetensors")]

        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            clean_name, folder_paths.get_output_directory(), 0, 0
        )
        
        if "%count%" in clean_name:
            file_name_out = clean_name.replace("%count%", f"{counter:05}") + ".safetensors"
        else:
            file_name_out = f"{filename}_{counter:05}_.safetensors"

        output_path = os.path.join(full_output_folder, file_name_out)
        os.makedirs(full_output_folder, exist_ok=True)
        logger.info(f"[ARTHEMY KREA2 CLIP SAVER] Stream-saving CLIP: {file_name_out}")

        clip_sd = clip.get_sd()
        arch_info = Krea2TensorParser.probe_architecture(clip_sd)
        if arch_info['num_clip_layers'] != Krea2Config.MAX_CLIP_LAYERS:
            logger.warning(f"[ARTHEMY KREA2 CLIP SAVER] ⚠️ Topology warning: detected {arch_info['num_clip_layers']} layers (expected baseline: {Krea2Config.MAX_CLIP_LAYERS}).")
        else:
            logger.info(f"[ARTHEMY KREA2 CLIP SAVER] Topology validated: {arch_info['num_clip_layers']} layers detected.")

        target_dtype = torch.float8_e4m3fn if precision == "FP8_E4M3" else torch.bfloat16
        final_sd = {}

        def process_clip_tensor(k, v):
            scale = None
            clean_k = Krea2TensorParser.clean_key(k)
            for s_cand in (f"{k}_scale", f"{k}.weight_scale", f"{clean_k}_scale", f"{clean_k}.weight_scale"):
                if s_cand in clip_sd:
                    scale = clip_sd[s_cand]
                    break
            return dequantize_weight(v, scale=scale)

        for clean_key, tensor in process_tensor_stream(clip_sd, process_clip_tensor, target_dtype=target_dtype, memory_threshold_mb=Krea2Config.DEFAULT_GC_THRESHOLD_MB):
            if tensor is not None:
                final_sd[clean_key] = tensor

        metadata = {"format": "pt"}
        safetensors.torch.save_file(final_sd, output_path, metadata=metadata)
        del final_sd
        gc.collect()
        logger.info(f"[ARTHEMY KREA2 CLIP SAVER] ✨ SUCCESS: {output_path}")
        return (output_path,)

# ==============================================================================
# LORA TRIO (Block Loader, Sub-Block, Sub-Block Chaos) & UTILITIES
# ==============================================================================
class ArthemyKrea2LoraBlockLoader(BaseKrea2Node):
    """Tier 1: Block-selective LoRA loading for Krea-2 and CLIP with per-section control."""
    GROUP_MAP = {
        "Text_Fusion": ["txtfusion.", "txtmlp."],
        "Time_Embed": ["tmlp.", "tproj."],
        "Projection": ["first.", "last."],
        "Block_1": ["blocks.0.", "blocks.1.", "blocks.2.", "blocks.3.", "blocks.4."],
        "Block_2": ["blocks.5.", "blocks.6.", "blocks.7.", "blocks.8.", "blocks.9."],
        "Block_3": ["blocks.10.", "blocks.11.", "blocks.12.", "blocks.13.", "blocks.14."],
        "Block_4": ["blocks.15.", "blocks.16.", "blocks.17.", "blocks.18.", "blocks.19."],
        "Block_5": ["blocks.20.", "blocks.21.", "blocks.22.", "blocks.23."],
        "Block_6": ["blocks.24.", "blocks.25.", "blocks.26.", "blocks.27."],
    }
    ORDERED_KEYS = list(GROUP_MAP.keys())

    @classmethod
    def INPUT_TYPES(s):
        lora_list = folder_paths.get_filename_list("loras")
        inputs = {
            "required": {
                "model": ("MODEL",), "clip": ("CLIP",),
                "lora_name": (lora_list,),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -99.00, "max": 99.00, "step": 0.01,
                                            "tooltip": "Global master multiplier for Model LoRA strength."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -99.00, "max": 99.00, "step": 0.01,
                                           "tooltip": "Global master multiplier for CLIP LoRA strength."}),
            }
        }
        for name in s.ORDERED_KEYS:
            inputs["required"][name] = ("FLOAT", {"default": 0.00, "min": -99.00, "max": 99.00, "step": 0.01,
                                                    "tooltip": f"LoRA multiplier for section {name} (0.00 = off, 1.00 = 100% strength)."})
        return inputs

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "info")
    FUNCTION = "load_lora"
    CATEGORY = "Arthemy/Krea2 LoRA"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, **kwargs):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, "LoRA bypassed (strength 0)")
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            raise FileNotFoundError(f"Arthemy Suite Error: LoRA '{lora_name}' not found in ComfyUI loras directory.")
        
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        filtered_lora = {}

        for k, v in lora.items():
            clean_k = Krea2TensorParser.clean_key(k)
            matched_group = None
            for group_name, prefixes in self.GROUP_MAP.items():
                if any(clean_k.startswith(pfx) or f".{pfx}" in clean_k for pfx in prefixes):
                    matched_group = group_name
                    break

            if matched_group:
                block_mult = kwargs.get(matched_group, 0.00)
                if block_mult != 0.00:
                    filtered_lora[k] = v * block_mult
            else:
                filtered_lora[k] = v

        if not filtered_lora:
            return (model, clip, f"LoRA '{lora_name}' bypassed (all selected block strengths are 0.00)")

        m, c = comfy.sd.load_lora_for_models(model, clip, filtered_lora, strength_model, strength_clip)
        active_sections = [sec for sec in self.ORDERED_KEYS if kwargs.get(sec, 0.00) != 0.00]
        sec_str = ", ".join(active_sections) if active_sections else "CLIP/Global"
        return (m, c, f"Loaded LoRA '{lora_name}' on sections [{sec_str}] ({len(filtered_lora)} keys)")


class ArthemyKrea2LoadSubBlockLora(BaseKrea2Node):
    """Tier 2: High-precision deterministic Sub-Block LoRA filtering and weighting.
    Allows targeting specific transformer blocks and individual component types
    (e.g., query/key attention, MLP down-projection) within the LoRA."""
    MODEL_TARGET_MAP = ArthemyKrea2ModelBlockSurgeonTuner.MODEL_TARGET_MAP
    SURGEON_MAP = Krea2TensorParser.get_descriptive_model_surgeon_map()

    @classmethod
    def INPUT_TYPES(s):
        lora_list = folder_paths.get_filename_list("loras")
        inputs = {
            "required": {
                "model": ("MODEL",), "clip": ("CLIP",), "lora_name": (lora_list,),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "target_block": (list(s.MODEL_TARGET_MAP.keys()), {"default": "Block_1 (All 0-4)"}),
            }
        }
        for name in s.SURGEON_MAP.keys():
            inputs["required"][name] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                                                  "tooltip": "Sub-tensor multiplier (0.0 = filter out entirely)."})
        return inputs

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "info")
    FUNCTION = "load_sub_block_lora"
    CATEGORY = "Arthemy/Krea2 LoRA"

    def load_sub_block_lora(self, model, clip, lora_name, strength_model, strength_clip, target_block, **kwargs):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, "LoRA bypassed (strength 0)")

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            raise FileNotFoundError(f"Arthemy Suite Error: LoRA '{lora_name}' not found in ComfyUI loras directory.")
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        selected_blocks = self.MODEL_TARGET_MAP.get(target_block, None)
        if selected_blocks is None:
            clean = target_block.strip().lstrip("↳").strip()
            for k, v in self.MODEL_TARGET_MAP.items():
                if clean in k:
                    selected_blocks = v
                    break
        if selected_blocks is None:
            selected_blocks = set(range(0, Krea2Config.MAX_UNET_BLOCKS))

        filtered_lora = {}
        for k, v in lora.items():
            clean_k = Krea2TensorParser.clean_key(k)
            idx, sub_key = Krea2TensorParser.extract_model_block_idx(clean_k)
            if idx is not None:
                if idx not in selected_blocks:
                    continue
                matched_widget = Krea2TensorParser.match_model_sub_tensor(sub_key)
                if matched_widget:
                    weight_mult = kwargs.get(matched_widget, 1.0)
                    if weight_mult <= 0.0:
                        continue
                    elif weight_mult != 1.0:
                        filtered_lora[k] = v * weight_mult
                        continue
            filtered_lora[k] = v

        m, c = comfy.sd.load_lora_for_models(model, clip, filtered_lora, strength_model, strength_clip)
        return (m, c, f"Sub-Block LoRA ({len(filtered_lora)} keys)")


class ArthemyKrea2LoadChaosLoraBlockSurgeon(BaseSurgeonTuner):
    """Tier 3: Stochastic Sub-Block Chaos LoRA filtering.
    Selectively and randomly activates LoRA key components according to per-sub-tensor chance rolls."""
    MODEL_TARGET_MAP = ArthemyKrea2ModelBlockSurgeonTuner.MODEL_TARGET_MAP
    SURGEON_MAP = Krea2TensorParser.get_descriptive_model_surgeon_map()

    @classmethod
    def INPUT_TYPES(s):
        lora_list = folder_paths.get_filename_list("loras")
        inputs = {
            "required": {
                "model": ("MODEL",), "clip": ("CLIP",), "lora_name": (lora_list,),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "target_block": (list(s.MODEL_TARGET_MAP.keys()), {"default": "Block_1 (All 0-4)"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "base_chance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
        for name in s.SURGEON_MAP.keys():
            inputs["required"][f"{name}_chance"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01})
        return inputs

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "info")
    FUNCTION = "load_chaos_lora_surgeon"
    CATEGORY = "Arthemy/Krea2 LoRA"

    def load_chaos_lora_surgeon(self, model, clip, lora_name, strength_model, strength_clip, target_block, seed, base_chance, **kwargs):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, "LoRA bypassed (strength 0)")

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            raise FileNotFoundError(f"Arthemy Suite Error: LoRA '{lora_name}' not found in ComfyUI loras directory.")
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        selected_blocks = self.MODEL_TARGET_MAP.get(target_block, None)
        if selected_blocks is None:
            clean = target_block.strip().lstrip("↳").strip()
            for k, v in self.MODEL_TARGET_MAP.items():
                if clean in k:
                    selected_blocks = v
                    break
        if selected_blocks is None:
            selected_blocks = set(range(0, Krea2Config.MAX_UNET_BLOCKS))

        filtered_lora = {}
        for k, v in lora.items():
            clean_k = Krea2TensorParser.clean_key(k)
            idx, sub_key = Krea2TensorParser.extract_model_block_idx(clean_k)
            if idx is not None and idx not in selected_blocks:
                continue

            matched_widget = Krea2TensorParser.match_model_sub_tensor(sub_key)
            if matched_widget:
                chance = kwargs.get(f"{matched_widget}_chance", base_chance)
            else:
                chance = base_chance

            if chance <= 0.0:
                continue

            fast_seed = generate_fast_seed(k, seed)
            rng = torch.Generator().manual_seed(fast_seed)
            if torch.rand(1, generator=rng).item() < chance:
                filtered_lora[k] = v

        m, c = comfy.sd.load_lora_for_models(model, clip, filtered_lora, strength_model, strength_clip)
        return (m, c, f"Sub-Block Chaos LoRA ({len(filtered_lora)} keys)")


class ArthemyKrea2ResetPatcher(BaseKrea2Node):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), "clip": ("CLIP",),
                "reset_model": ("BOOLEAN", {"default": True}),
                "reset_clip": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "info")
    FUNCTION = "reset"
    CATEGORY = "Arthemy/Krea2 Utilities"

    def reset(self, model, clip, reset_model=True, reset_clip=True):
        m = model.clone() if reset_model else model
        c = clip.clone() if reset_clip else clip
        m_p = get_patcher(m)
        c_p = get_patcher(c)
        if reset_model and m_p:
            if hasattr(m_p, "patches"): m_p.patches = {}
            if hasattr(m_p, "model_options"): m_p.model_options.pop("arthemy_chaos_recipes", None)
        if reset_clip and c_p:
            if hasattr(c_p, "patches"): c_p.patches = {}
            if hasattr(c_p, "model_options"): c_p.model_options.pop("arthemy_chaos_recipes", None)
        return (m, c, "Reset patchers clean.")

# ==============================================================================
# POINT 3 IMPLEMENTATION: MODEL BAKER WITH STREAM GENERATOR
# ==============================================================================
def isolate_and_assign_baked_weights(patcher, baked_weights: dict):
    """Safely isolates mutated parameters onto a cloned module hierarchy without in-place mutating shared base weights."""
    if not baked_weights or not hasattr(patcher, "model"):
        if hasattr(patcher, "patches"): patcher.patches = {}
        if hasattr(patcher, "backup"): patcher.backup = {}
        if hasattr(patcher, "object_patches"): patcher.object_patches = {}
        if hasattr(patcher, "model_options"): patcher.model_options.pop("arthemy_chaos_recipes", None)
        return
    
    import copy
    base_model = patcher.model
    cloned_top = copy.copy(base_model)
    cloned_top._modules = copy.copy(base_model._modules) if hasattr(base_model, "_modules") else {}
    cloned_top._parameters = copy.copy(base_model._parameters) if hasattr(base_model, "_parameters") else {}

    for k, new_tensor in baked_weights.items():
        clean_k = Krea2TensorParser.clean_key(k)
        parts = clean_k.split(".")
        curr = cloned_top
        
        for part in parts[:-1]:
            if hasattr(curr, part):
                sub = getattr(curr, part)
                if isinstance(sub, torch.nn.Module):
                    cloned_sub = copy.copy(sub)
                    cloned_sub._modules = copy.copy(sub._modules)
                    cloned_sub._parameters = copy.copy(sub._parameters)
                    setattr(curr, part, cloned_sub)
                    curr = cloned_sub
                else:
                    break
            elif isinstance(curr, (torch.nn.ModuleDict, dict)) and part in curr:
                curr = curr[part]
            else:
                break
                
        param_name = parts[-1]
        if hasattr(curr, param_name):
            setattr(curr, param_name, torch.nn.Parameter(new_tensor.clone().detach(), requires_grad=False))

    patcher.model = cloned_top
    patcher.patches = {}
    patcher.backup = {}
    patcher.object_patches = {}
    if hasattr(patcher, "model_options"):
        patcher.model_options.pop("arthemy_chaos_recipes", None)


class ArthemyKrea2ModelBaker(BaseKrea2Node):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "info")
    FUNCTION = "bake"
    CATEGORY = "Arthemy/Krea2 Utilities"

    def bake(self, model, clip):
        m_baked = model.clone()
        c_baked = clip.clone()

        c_p = get_patcher(c_baked)
        m_p = get_patcher(m_baked)
        c_sd = c_p.model.state_dict() if hasattr(c_p, "model") else {}
        m_sd = m_p.model.state_dict() if hasattr(m_p, "model") else {}

        n_model_patched = 0
        n_clip_patched = 0
        baked_model_weights = {}
        baked_clip_weights = {}

        # Stream process model tensors with O(1) key resolution
        def bake_model_tensor(k, v):
            nonlocal n_model_patched
            target_k = resolve_target_key(m_p, k, model_sd=m_sd)
            patches = getattr(m_p, "patches", {})
            if target_k in patches or k in patches:
                n_model_patched += 1
                patched_weight = ComfyPatcherAdapter.calculate_safe_weight(m_baked, target_k, v, model_sd=m_sd)
                baked_model_weights[target_k] = patched_weight
                return patched_weight
            return v

        for _, _ in process_tensor_stream(m_sd, bake_model_tensor, memory_threshold_mb=Krea2Config.DEFAULT_GC_THRESHOLD_MB):
            pass

        def bake_clip_tensor(k, v):
            nonlocal n_clip_patched
            target_k = resolve_target_key(c_p, k, model_sd=c_sd)
            patches = getattr(c_p, "patches", {})
            if target_k in patches or k in patches:
                n_clip_patched += 1
                patched_weight = ComfyPatcherAdapter.calculate_safe_weight(c_baked, target_k, v, model_sd=c_sd)
                baked_clip_weights[target_k] = patched_weight
                return patched_weight
            return v

        for _, _ in process_tensor_stream(c_sd, bake_clip_tensor, memory_threshold_mb=Krea2Config.DEFAULT_GC_THRESHOLD_MB):
            pass

        # Apply isolated parameters without mutating original shared module
        isolate_and_assign_baked_weights(m_p, baked_model_weights)
        isolate_and_assign_baked_weights(c_p, baked_clip_weights)

        info = f"Arthemy Krea-2 Model Baker: baked {n_model_patched} model parameters, {n_clip_patched} clip parameters."
        logger.info(f"[ARTHEMY KREA2 MODEL BAKER] {info}")
        return (m_baked, c_baked, info)

# ==============================================================================
# VISUALIZER NODES
# ==============================================================================

def parse_patch_entry(p):
    """Analyzes a patch entry tuple (strength_patch, diff, strength_model)
    and returns (offset_delta, is_lora, is_chaos)."""
    if not isinstance(p, tuple):
        return 0.0, False, False

    strength_patch = float(p[0]) if (len(p) >= 1 and isinstance(p[0], (int, float))) else 1.0
    diff = p[1] if len(p) >= 2 else p[0]
    strength_model = float(p[2]) if (len(p) >= 3 and isinstance(p[2], (int, float))) else 1.0

    offset_delta = 0.0
    is_lora = False
    is_chaos = False

    if strength_model != 1.0:
        offset_delta += (strength_model - 1.0)

    # Unwrap nested tuple structures used by ComfyUI LoRA patchers
    while isinstance(diff, (tuple, list)) and len(diff) == 1 and isinstance(diff[0], (tuple, list, torch.Tensor)):
        diff = diff[0]

    # Check for modern ComfyUI LoRAAdapter / WeightAdapter object
    diff_type_name = type(diff).__name__
    if "LoRA" in diff_type_name or "Lora" in diff_type_name or "Adapter" in diff_type_name or hasattr(diff, "weights") or hasattr(diff, "lora_a"):
        is_lora = True
    elif strength_patch != 0.0:
        if isinstance(diff, (tuple, list)):
            has_tensors = any(isinstance(x, torch.Tensor) for x in diff)
            if has_tensors or len(diff) >= 2:
                is_lora = True
            elif len(diff) == 1:
                val = diff[0]
                if isinstance(val, (int, float)):
                    offset_delta += (float(val) - 1.0) * strength_patch
                elif isinstance(val, torch.Tensor):
                    if val.numel() == 1:
                        offset_delta += (float(val.item()) - 1.0) * strength_patch
                    elif val.numel() > 1:
                        is_chaos = True
                        offset_delta += float(val.float().mean().item()) * strength_patch
        elif isinstance(diff, (int, float)):
            offset_delta += (float(diff) - 1.0) * strength_patch
        elif isinstance(diff, torch.Tensor):
            if diff.numel() == 1:
                offset_delta += (float(diff.item()) - 1.0) * strength_patch
            elif diff.numel() > 1:
                is_chaos = True
                offset_delta += float(diff.float().mean().item()) * strength_patch

    return offset_delta, is_lora, is_chaos


def render_visualizer_image(graph_data: list, title: str, is_clip: bool, visual_scale: float = 1.0, width: int = 960, height: int = 480) -> torch.Tensor:
    """Renders high-quality visualizer HUD image using PIL and converts to PyTorch IMAGE tensor [1, H, W, 3]."""
    img = Image.new("RGB", (width, height), color=(15, 23, 42)) # #0f172a
    draw = ImageDraw.Draw(img)

    base_color = (255, 215, 0) if is_clip else (0, 229, 255) # Gold for CLIP, Cyan for Model
    lora_color = (208, 0, 255) # Neon Purple

    padding_x = int(width * 0.03)
    top_y = int(height * 0.08)
    chart_w = width - (padding_x * 2)
    chart_h = height - top_y - int(height * 0.12)
    center_y = top_y + (chart_h // 2)

    # 1. Outer container box
    draw.rectangle([padding_x - 4, top_y - 20, padding_x + chart_w + 4, top_y + chart_h + 18], outline=base_color, width=2)

    # 2. Header title
    draw.text((padding_x + 4, top_y - 18), title, fill=base_color)

    # 3. Header legend & scale
    leg_x = padding_x + 180
    draw.text((leg_x, top_y - 18), "■ Base", fill=base_color)
    draw.text((leg_x + 75, top_y - 18), "■ LoRA", fill=lora_color)
    draw.text((leg_x + 150, top_y - 18), "^ Chaos", fill=(255, 255, 255, 200))

    scale_str = f"Scale: {visual_scale:.1f}x"
    draw.text((width - padding_x - 110, top_y - 18), scale_str, fill=(255, 255, 255, 180))

    # 4. Zero baseline axis
    for x in range(padding_x, padding_x + chart_w, 8):
        draw.line([(x, center_y), (min(x + 4, padding_x + chart_w), center_y)], fill=(255, 255, 255, 60), width=1)

    # 5. Render section graph waveform
    if graph_data:
        num_secs = len(graph_data)
        step_x = chart_w / num_secs
        max_offset_span = (chart_h / 2) - 6
        user_scale = max(0.01, min(99.0, visual_scale))
        scale_factor = (max_offset_span / 0.50) * (user_scale * 0.35)

        for idx, sec in enumerate(graph_data):
            x1 = padding_x + (idx * step_x)
            x2 = x1 + step_x

            raw_offset = sec.get("offset", 0.0)
            pixel_shift = raw_offset * scale_factor
            clamped_shift = max(-max_offset_span, min(max_offset_span, pixel_shift))
            y_val = center_y - clamped_shift

            is_lora = sec.get("is_lora", False)
            is_chaos = sec.get("is_chaos", False)
            current_color = lora_color if is_lora else base_color

            prev_shift = 0
            if idx > 0:
                prev_raw = graph_data[idx - 1].get("offset", 0.0)
                prev_shift = max(-max_offset_span, min(max_offset_span, prev_raw * scale_factor))
            prev_y = center_y - prev_shift

            if is_chaos:
                mid_x = x1 + (step_x / 2)
                zig_amp = 6 * (1.2 if is_lora else 1.0)
                pts = [
                    (x1, prev_y),
                    (x1 + step_x * 0.25, y_val - zig_amp),
                    (mid_x, y_val + zig_amp),
                    (x1 + step_x * 0.75, y_val - zig_amp),
                    (x2, y_val)
                ]
                draw.line(pts, fill=current_color, width=3 if is_lora else 2)
            else:
                draw.line([(x1, prev_y), (x1, y_val), (x2, y_val)], fill=current_color, width=3 if is_lora else 2)

    # 6. X-Axis Section Labels
    axis_y = top_y + chart_h + 4
    groups = [
        {"label": "B1", "endIdx": 4}, {"label": "B2", "endIdx": 9},
        {"label": "B3", "endIdx": 14}, {"label": "B4", "endIdx": 19},
        {"label": "B5", "endIdx": 23}, {"label": "B6", "endIdx": 27},
        {"label": "TF / TE / PR", "endIdx": 30}
    ] if not is_clip else [
        {"label": "L1", "endIdx": 4}, {"label": "L2", "endIdx": 9},
        {"label": "L3", "endIdx": 14}, {"label": "L4", "endIdx": 19},
        {"label": "L5", "endIdx": 24}, {"label": "L6", "endIdx": 29},
        {"label": "L7", "endIdx": 35}, {"label": "EM", "endIdx": 36}
    ]

    step_x = chart_w / (31 if not is_clip else 37)
    start_idx = 0
    for g in groups:
        end_x = padding_x + int((g["endIdx"] + 1) * step_x)
        start_x = padding_x + int(start_idx * step_x)
        mid_x = (start_x + end_x) // 2
        draw.text((mid_x - 10, axis_y), g["label"], fill=(255, 255, 255, 180))
        draw.line([(end_x, top_y + chart_h - 2), (end_x, top_y + chart_h + 5)], fill=(255, 255, 255, 50), width=1)
        start_idx = g["endIdx"] + 1

    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)

class ArthemyKrea2ModelVisualizer(BaseKrea2Node):
    OUTPUT_NODE = True
    SECTION_KEYS = [
        "Block_1A", "Block_1B", "Block_1C", "Block_1D", "Block_1E",
        "Block_2A", "Block_2B", "Block_2C", "Block_2D", "Block_2E",
        "Block_3A", "Block_3B", "Block_3C", "Block_3D", "Block_3E",
        "Block_4A", "Block_4B", "Block_4C", "Block_4D", "Block_4E",
        "Block_5A", "Block_5B", "Block_5C", "Block_5D",
        "Block_6A", "Block_6B", "Block_6C", "Block_6D",
        "Text_Fusion", "Time_Embed", "Projection"
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 99.00, "step": 0.1, "tooltip": "Visual height amplification scale."}),
                "image_width": ("INT", {"default": 960, "min": 256, "max": 3840, "step": 16, "tooltip": "Exported PNG render width."}),
                "image_height": ("INT", {"default": 480, "min": 128, "max": 2160, "step": 16, "tooltip": "Exported PNG render height."}),
            }
        }

    RETURN_TYPES = ("MODEL", "IMAGE", "STRING")
    RETURN_NAMES = ("MODEL", "IMAGE", "info")
    FUNCTION = "visualize"
    CATEGORY = "Arthemy/Visualizers"

    def visualize(self, model, scale=1.0, image_width=960, image_height=480):
        m_patcher = model.patcher if hasattr(model, "patcher") else model
        patches = getattr(m_patcher, "patches", {})

        sec_data = {s: {"block": s, "offset": 0.0, "is_lora": False, "is_chaos": False, "count": 0} for s in self.SECTION_KEYS}

        for pk, patch_list in patches.items():
            ck = Krea2TensorParser.clean_key(pk)
            sec_name = None
            if "txtfusion" in ck or "txtmlp" in ck:
                sec_name = "Text_Fusion"
            elif "tmlp" in ck or "tproj" in ck:
                sec_name = "Time_Embed"
            elif "first" in ck or "last" in ck:
                sec_name = "Projection"
            else:
                b_idx, _ = Krea2TensorParser.extract_model_block_idx(ck)
                if b_idx is not None:
                    for label, idx in MODEL_BLOCK_LABEL_TO_IDX.items():
                        if idx == b_idx:
                            sec_name = label
                            break

            if not sec_name or sec_name not in sec_data: continue
            sec = sec_data[sec_name]
            sec["count"] += 1

            for p in patch_list:
                off, is_l, is_c = parse_patch_entry(p)
                if is_l: sec["is_lora"] = True
                if is_c: sec["is_chaos"] = True
                sec["offset"] += off

        graph_data = []
        modified_sections = 0
        for s in self.SECTION_KEYS:
            d = sec_data[s]
            if d["count"] > 0:
                d["offset"] /= max(1, d["count"])
                modified_sections += 1
            graph_data.append({
                "block": d["block"], "offset": round(d["offset"], 4),
                "is_lora": d["is_lora"], "is_chaos": d["is_chaos"],
            })

        info = f"Model Visualizer: {modified_sections}/{len(self.SECTION_KEYS)} sections modified."
        logger.info(f"[ARTHEMY MODEL VISUALIZER] {info}")
        vis_image = render_visualizer_image(graph_data, "Krea-2 Model", is_clip=False, visual_scale=scale, width=image_width, height=image_height)
        return {"ui": {"graph_data": graph_data, "scale": [scale], "title": ["Krea-2 Model"]}, "result": (model, vis_image, info)}


class ArthemyKrea2CLIPVisualizer(BaseKrea2Node):
    OUTPUT_NODE = True
    SECTION_KEYS = [
        "Layer_1A", "Layer_1B", "Layer_1C", "Layer_1D", "Layer_1E",
        "Layer_2A", "Layer_2B", "Layer_2C", "Layer_2D", "Layer_2E",
        "Layer_3A", "Layer_3B", "Layer_3C", "Layer_3D", "Layer_3E",
        "Layer_4A", "Layer_4B", "Layer_4C", "Layer_4D", "Layer_4E",
        "Layer_5A", "Layer_5B", "Layer_5C", "Layer_5D", "Layer_5E",
        "Layer_6A", "Layer_6B", "Layer_6C", "Layer_6D", "Layer_6E",
        "Layer_7A", "Layer_7B", "Layer_7C", "Layer_7D", "Layer_7E", "Layer_7F",
        "Embedding"
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 99.00, "step": 0.1, "tooltip": "Visual height amplification scale."}),
                "image_width": ("INT", {"default": 960, "min": 256, "max": 3840, "step": 16, "tooltip": "Exported PNG render width."}),
                "image_height": ("INT", {"default": 480, "min": 128, "max": 2160, "step": 16, "tooltip": "Exported PNG render height."}),
            }
        }

    RETURN_TYPES = ("CLIP", "IMAGE", "STRING")
    RETURN_NAMES = ("CLIP", "IMAGE", "info")
    FUNCTION = "visualize"
    CATEGORY = "Arthemy/Visualizers"

    def visualize(self, clip, scale=1.0, image_width=960, image_height=480):
        c_patcher = clip.patcher if hasattr(clip, "patcher") else clip
        patches = getattr(c_patcher, "patches", {})

        sec_data = {s: {"block": s, "offset": 0.0, "is_lora": False, "is_chaos": False, "count": 0} for s in self.SECTION_KEYS}

        for pk, patch_list in patches.items():
            ck = Krea2TensorParser.clean_key(pk)
            sec_name = None
            l_idx, _ = Krea2TensorParser.extract_clip_layer_idx(ck)
            if l_idx is not None:
                for label, idx in CLIP_LAYER_LABEL_TO_IDX.items():
                    if idx == l_idx:
                        sec_name = label
                        break
            elif "embed_tokens" in ck: sec_name = "Embedding"

            if not sec_name or sec_name not in sec_data: continue
            sec = sec_data[sec_name]
            sec["count"] += 1

            for p in patch_list:
                off, is_l, is_c = parse_patch_entry(p)
                if is_l: sec["is_lora"] = True
                if is_c: sec["is_chaos"] = True
                sec["offset"] += off

        graph_data = []
        modified_sections = 0
        for s in self.SECTION_KEYS:
            d = sec_data[s]
            if d["count"] > 0:
                d["offset"] /= max(1, d["count"])
                modified_sections += 1
            graph_data.append({
                "block": d["block"], "offset": round(d["offset"], 4),
                "is_lora": d["is_lora"], "is_chaos": d["is_chaos"],
            })

        info = f"CLIP Visualizer: {modified_sections}/{len(self.SECTION_KEYS)} sections modified."
        logger.info(f"[ARTHEMY CLIP VISUALIZER] {info}")
        vis_image = render_visualizer_image(graph_data, "Qwen3 Text Encoder", is_clip=True, visual_scale=scale, width=image_width, height=image_height)
        return {"ui": {"graph_data": graph_data, "scale": [scale], "title": ["Qwen3 Text Encoder"]}, "result": (clip, vis_image, info)}

# ==============================================================================
# PRESET NODES (SAVER & LOADER)
# ==============================================================================

def extract_patch_multipliers(patcher) -> Tuple[Dict[str, float], int, int]:
    """Extracts pure scalar offset multipliers from active patches in a ModelPatcher.
    Returns (scalar_offsets_dict, lora_patches_count, chaos_patches_count)."""
    if patcher is None or not hasattr(patcher, "patches"):
        return {}, 0, 0
    extracted = {}
    lora_count = 0
    chaos_count = 0
    for k, patch_list in patcher.patches.items():
        if not patch_list:
            continue
        total_scalar_offset = 0.0
        has_scalar = False
        for p in patch_list:
            off, is_lora, is_chaos = parse_patch_entry(p)
            if is_lora:
                lora_count += 1
            elif is_chaos:
                chaos_count += 1
            else:
                total_scalar_offset += off
                has_scalar = True

        if has_scalar and round(total_scalar_offset, 6) != 0.0:
            clean_k = Krea2TensorParser.clean_key(k)
            extracted[clean_k] = round(total_scalar_offset, 6)
    return extracted, lora_count, chaos_count


class ArthemyKrea2PresetSaver(BaseKrea2Node):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "preset_name": ("STRING", {"default": "my_arthemy_preset"}),
            },
            "optional": {
                "author": ("STRING", {"default": "Arthemy"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("saved_path", "info")
    FUNCTION = "save_preset"
    CATEGORY = "Arthemy/Presets"

    def save_preset(self, model, clip, preset_name="my_arthemy_preset", author="Arthemy", **kwargs):
        m_p = get_patcher(model)
        c_p = get_patcher(clip)

        model_patches, m_lora, m_chaos = extract_patch_multipliers(m_p)
        clip_patches, c_lora, c_chaos = extract_patch_multipliers(c_p)

        # Collect deterministic Chaos recipes
        m_recipes = m_p.model_options.get("arthemy_chaos_recipes", []) if hasattr(m_p, "model_options") else []
        c_recipes = c_p.model_options.get("arthemy_chaos_recipes", []) if hasattr(c_p, "model_options") else []
        combined_chaos_recipes = list(m_recipes) + list(c_recipes)

        total_lora = m_lora + c_lora
        if total_lora > 0:
            logger.warning(f"[ARTHEMY PRESET SAVER] Warning: {total_lora} active LoRA patch tensors detected. "
                           "LoRAs are excluded from lightweight presets. Use Model Saver/Baker to fuse LoRA weights permanently.")

        if not model_patches and not clip_patches and not combined_chaos_recipes:
            logger.warning("[ARTHEMY PRESET SAVER] No active tunings or chaos recipes detected on Model or CLIP.")

        # Strict basename sanitization against path-traversal enforcing .json
        raw_name = os.path.basename(preset_name.strip())
        base_name, _ = os.path.splitext(raw_name)
        safe_base = re.sub(r'[^\w\-_\.]', '_', base_name)
        if not safe_base:
            safe_base = "arthemy_preset"
        safe_name = f"{safe_base}.json"

        save_dirs = folder_paths.get_folder_paths("arthemy_presets")
        save_dir = save_dirs[0] if save_dirs else arthemy_presets_dir
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, safe_name)

        preset_data = {
            "name": safe_base,
            "author": author.strip() if author else "Arthemy",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "2.0",
            "stats": {
                "model_patched_layers": len(model_patches),
                "clip_patched_layers": len(clip_patches),
                "chaos_recipes_count": len(combined_chaos_recipes),
                "excluded_lora_tensors": total_lora,
            },
            "model_patches": model_patches,
            "clip_patches": clip_patches,
            "chaos_recipes": combined_chaos_recipes,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(preset_data, f, indent=2, ensure_ascii=False)

        info = f"Preset saved: '{safe_name}' ({len(model_patches)} model, {len(clip_patches)} clip, {len(combined_chaos_recipes)} chaos recipes)"
        if total_lora > 0:
            info += f" | ⚠️ {total_lora} LoRA tensors excluded"
        logger.info(f"[ARTHEMY PRESET SAVER] {info} -> {file_path}")
        return (file_path, info)


class ArthemyKrea2PresetLoader(BaseKrea2Node):
    @classmethod
    def INPUT_TYPES(s):
        preset_files = folder_paths.get_filename_list("arthemy_presets")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "preset": (sorted(preset_files) if preset_files else ["None"],),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -99.00, "max": 99.00, "step": 0.01,
                                             "tooltip": "Global multiplier for model patches in this preset."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -99.00, "max": 99.00, "step": 0.01,
                                            "tooltip": "Global multiplier for CLIP patches in this preset."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "info")
    FUNCTION = "load_preset"
    CATEGORY = "Arthemy/Presets"

    def load_preset(self, model, clip, preset, strength_model=1.0, strength_clip=1.0):
        if not preset or preset == "None":
            return (model, clip, "No preset selected.")

        file_path = folder_paths.get_full_path("arthemy_presets", preset)
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Arthemy Suite Error: Preset '{preset}' not found.")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        m = model.clone()
        c = clip.clone()
        m_p = get_patcher(m)
        c_p = get_patcher(c)

        m_sd = m_p.model.state_dict() if hasattr(m_p, 'model') else {}
        c_sd = c_p.model.state_dict() if hasattr(c_p, 'model') else {}

        model_patches = data.get("model_patches", {})
        clip_patches = data.get("clip_patches", {})
        chaos_recipes = data.get("chaos_recipes", [])

        n_model_matched = 0
        n_model_unmatched = 0
        n_clip_matched = 0
        n_clip_unmatched = 0

        # 1. Apply scalar model patches with state-dict key resolution
        if model_patches and strength_model != 0.0:
            m_patches_to_add = {}
            for k, off in model_patches.items():
                target_k = resolve_target_key(m_p, k, model_sd=m_sd)
                scaled_off = off * strength_model
                if scaled_off != 0.0:
                    clean_k = Krea2TensorParser.clean_key(target_k)
                    patch_k = f"diffusion_model.{clean_k}"
                    m_patches_to_add[patch_k] = (1.0 + scaled_off,)
                    n_model_matched += 1
            if m_patches_to_add:
                add_patches_to_front(m, m_patches_to_add, 1.0)

        # 2. Apply scalar CLIP patches with state-dict key resolution
        if clip_patches and strength_clip != 0.0:
            c_patches_to_add = {}
            for k, off in clip_patches.items():
                target_k = resolve_target_key(c_p, k, model_sd=c_sd)
                scaled_off = off * strength_clip
                if scaled_off != 0.0:
                    c_patches_to_add[target_k] = (1.0 + scaled_off,)
                    n_clip_matched += 1
            if c_patches_to_add:
                add_patches_to_front(c, c_patches_to_add, 1.0)

        # 3. Deterministically regenerate Chaos recipes
        n_chaos_applied = 0
        for r in chaos_recipes:
            domain = r.get("domain", "model")
            selected_indices = set(r.get("selected_indices", []))
            tune_mode = r.get("tune_mode", "Block-Level")
            seed = r.get("seed", 42)
            base_strength = r.get("chaos_strength", 0.1)
            chances = r.get("chances", {})

            if domain == "model" and strength_model != 0.0:
                effective_strength = base_strength * strength_model
                chaos_p = {"tune_mode": tune_mode, "seed": seed, "chaos_strength": effective_strength}
                m, _ = BaseSurgeonTuner()._execute_surgeon_tuning(
                    m, is_clip=False, selected_indices=selected_indices,
                    surgeon_map=Krea2TensorParser.MODEL_SURGEON_MAP,
                    kwargs=chances, chaos_params=chaos_p
                )
                n_chaos_applied += 1
            elif domain == "clip" and strength_clip != 0.0:
                effective_strength = base_strength * strength_clip
                chaos_p = {"tune_mode": tune_mode, "seed": seed, "chaos_strength": effective_strength}
                c, _ = BaseSurgeonTuner()._execute_surgeon_tuning(
                    c, is_clip=True, selected_indices=selected_indices,
                    surgeon_map=Krea2TensorParser.CLIP_SURGEON_MAP,
                    kwargs=chances, chaos_params=chaos_p
                )
                n_chaos_applied += 1

        preset_name = data.get("name", preset)
        author = data.get("author", "Unknown")

        info_parts = [f"Loaded Preset '{preset_name}' by {author}"]
        info_parts.append(f"Model: {n_model_matched} scalar layers (x{strength_model:.2f})")
        info_parts.append(f"CLIP: {n_clip_matched} scalar layers (x{strength_clip:.2f})")
        if n_chaos_applied > 0:
            info_parts.append(f"Chaos: {n_chaos_applied} deterministic recipes")
        if n_model_unmatched > 0 or n_clip_unmatched > 0:
            info_parts.append(f"⚠️ Unmatched keys: {n_model_unmatched} model, {n_clip_unmatched} clip")

        info = " | ".join(info_parts)
        logger.info(f"[ARTHEMY PRESET LOADER] {info}")
        return (m, c, info)

# ==============================================================================
# NODE MAPPINGS & REGISTRATION (3x3 Grid + Savers & Utilities)
# ==============================================================================
NODE_CLASS_MAPPINGS = {
    # 🟦 Model Trio
    "ArthemyKrea2ModelTuner": ArthemyKrea2ModelTuner,
    "ArthemyKrea2ModelBlockSurgeonTuner": ArthemyKrea2ModelBlockSurgeonTuner,
    "ArthemyKrea2ModelChaosBlockSurgeonTuner": ArthemyKrea2ModelChaosBlockSurgeonTuner,

    # 🟨 CLIP Trio
    "ArthemyKrea2CLIPTuner": ArthemyKrea2CLIPTuner,
    "ArthemyKrea2CLIPBlockSurgeonTuner": ArthemyKrea2CLIPBlockSurgeonTuner,
    "ArthemyKrea2CLIPChaosBlockSurgeonTuner": ArthemyKrea2CLIPChaosBlockSurgeonTuner,

    # 🟪 LoRA Trio
    "ArthemyKrea2LoraBlockLoader": ArthemyKrea2LoraBlockLoader,
    "ArthemyKrea2LoadSubBlockLora": ArthemyKrea2LoadSubBlockLora,
    "ArthemyKrea2LoadChaosLoraBlockSurgeon": ArthemyKrea2LoadChaosLoraBlockSurgeon,

    # Savers & Utilities
    "ArthemyKrea2ModelSaver": ArthemyKrea2ModelSaver,
    "ArthemyKrea2CLIPSaver": ArthemyKrea2CLIPSaver,
    "ArthemyKrea2ModelBaker": ArthemyKrea2ModelBaker,
    "ArthemyKrea2ModelVisualizer": ArthemyKrea2ModelVisualizer,
    "ArthemyKrea2CLIPVisualizer": ArthemyKrea2CLIPVisualizer,
    "ArthemyKrea2ResetPatcher": ArthemyKrea2ResetPatcher,

    # Presets
    "ArthemyKrea2PresetSaver": ArthemyKrea2PresetSaver,
    "ArthemyKrea2PresetLoader": ArthemyKrea2PresetLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # 🟦 Model Trio
    "ArthemyKrea2ModelTuner": "🟦✨ Model Tuner",
    "ArthemyKrea2ModelBlockSurgeonTuner": "🟦🔬 Model Sub-Block Tuner",
    "ArthemyKrea2ModelChaosBlockSurgeonTuner": "🟦🌪️ Model Sub-Block Chaos Tuner",

    # 🟨 CLIP Trio
    "ArthemyKrea2CLIPTuner": "🟨✨ CLIP Tuner",
    "ArthemyKrea2CLIPBlockSurgeonTuner": "🟨🔬 CLIP Sub-Block Tuner",
    "ArthemyKrea2CLIPChaosBlockSurgeonTuner": "🟨🌪️ CLIP Sub-Block Chaos Tuner",

    # 🟪 LoRA Trio
    "ArthemyKrea2LoraBlockLoader": "🟪🔮 LoRA Block Loader",
    "ArthemyKrea2LoadSubBlockLora": "🟪🔬 Load Sub-Block LoRA",
    "ArthemyKrea2LoadChaosLoraBlockSurgeon": "🟪🌪️ Load Sub-Block Chaos LoRA",

    # Savers & Utilities
    "ArthemyKrea2ModelSaver": "🟦💾 Model Saver",
    "ArthemyKrea2CLIPSaver": "🟨💾 CLIP Saver",
    "ArthemyKrea2ModelBaker": "🟦🟨 Model Baker",
    "ArthemyKrea2ModelVisualizer": "🟦📊 Model Visualizer",
    "ArthemyKrea2CLIPVisualizer": "🟨📊 CLIP Visualizer",
    "ArthemyKrea2ResetPatcher": "🟦🟨🔄 Reset Patcher",

    # Presets
    "ArthemyKrea2PresetSaver": "🟦🟨💾 Preset Saver",
    "ArthemyKrea2PresetLoader": "🟦🟨📂 Preset Loader",
}

WEB_DIRECTORY = "./web"

import nodes
_web_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "web"))
if os.path.isdir(_web_path) and hasattr(nodes, "EXTENSION_WEB_DIRS"):
    nodes.EXTENSION_WEB_DIRS["Arthemy_Krea2_Tuner"] = _web_path

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

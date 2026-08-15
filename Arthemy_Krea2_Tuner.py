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
import folder_paths

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
                if sub_key == target_suffix or sub_key.endswith(f".{target_suffix}") or target_suffix in sub_key:
                    return widget_key
        return None

    @classmethod
    def match_clip_sub_tensor(cls, sub_key: str):
        for widget_key, target_suffixes in cls.CLIP_SURGEON_MAP.items():
            for target_suffix in target_suffixes:
                if sub_key == target_suffix or sub_key.endswith(f".{target_suffix}") or target_suffix in sub_key:
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
    def calculate_safe_weight(model_or_clip, key: str, base_weight: torch.Tensor) -> torch.Tensor:
        """Calculates active weight safely without mutating model state or accessing private attributes."""
        patcher = get_patcher(model_or_clip)
        current_patches = patcher.patches.get(key, [])
        clean_base = dequantize_weight(get_clean_weight(patcher, key, base_weight))
        if current_patches:
            return comfy.lora.calculate_weight(current_patches, clean_base.clone(), key).to(torch.bfloat16)
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
    """REGOLA 1, 2, 3: Memory Safety, Dtype/Device Preservation & Sanity Check.
    
    1. Memory Safety: Forzatura contiguità e distacco con .clone().detach().contiguous()
    2. Dtype & Device: Casting rigoroso al dtype (bf16) e device del modello di destinazione
    3. Sanity Check: Prevenzione buffer vuoti (nelement == 0) per evitare il crash di torch.frombuffer
    """
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return None

    # Regola 3: Sanity check prevenzione buffer vuoti
    if tensor.nelement() == 0:
        logger.warning("[Arthemy Patch Guard] Rilevato tensore vuoto (nelement == 0). Patch ignorata per prevenire crash di memoria.")
        return None

    # Regola 2: Controllo e conservazione del dtype/device originale
    dtype = target_dtype if target_dtype is not None else tensor.dtype
    device = target_device if target_device is not None else tensor.device

    # Eseguiamo il cast se necessario ed applichiamo la Regola 1 (contiguità e distacco)
    return tensor.to(dtype=dtype, device=device).clone().detach().contiguous()

def resolve_target_key(patcher, k: str) -> str:
    """Finds the exact state_dict key in patcher.model matching k."""
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

    candidates = [
        f"diffusion_model.{clean_k}",
        f"cond_stage_model.{clean_k}",
        f"model.{clean_k}",
        f"clip_model.{clean_k}",
        f"transformer.{clean_k}",
    ]

    for cand in candidates:
        if cand in model_sd:
            return cand

    return k

def add_patches_to_front(model_patcher, patches: dict, strength_patch: float = 1.0, strength_model: float = 1.0):
    """Injects patches into a ModelPatcher using standard ComfyUI add_patches API.
    
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
        target_k = resolve_target_key(patcher, k)
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
        return patcher.backup[key]
    return current_weight

def dequantize_weight(weight: torch.Tensor) -> torch.Tensor:
    if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return weight.to(torch.float32)
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

        granular_map = {}
        if granular_json and granular_json.strip():
            try:
                parsed = json.loads(granular_json)
                if isinstance(parsed, dict): granular_map = parsed
            except Exception as e:
                logger.error(f"[Arthemy Surgeon] JSON parse error: {e}")

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

                w_active = ComfyPatcherAdapter.calculate_safe_weight(model_or_clip, target_patch_key, base_weight)
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
            import time; t_start = time.time()
            add_patches_to_front(clone_obj, patches_to_add, 1.0)
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
            except ValueError: pass

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
            except ValueError: pass

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
# 3. ARTHEMY KREA2 MODEL CHAOS BLOCK TUNER
# ==============================================================================
class ArthemyKrea2ModelChaosBlockTuner(BaseKrea2Node):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "tune_mode": (["Block-Level", "Element-Level (Sub-atomic)"],),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "chaos_strength": ("FLOAT", {"default": 0.1, "min": -99.00, "max": 99.00, "step": 0.01}),
                "base_chance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Block_1_chance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Block_2_chance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Block_3_chance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Block_4_chance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Block_5_chance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Block_6_chance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Text_Fusion_chance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Time_Embed_chance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Projection_chance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "chaos_tune_model"
    CATEGORY = "Arthemy/Krea2 Tuners"

    def chaos_tune_model(self, model, tune_mode, seed, chaos_strength, base_chance=0.5,
                         Block_1_chance=0.0, Block_2_chance=0.0, Block_3_chance=0.0,
                         Block_4_chance=0.0, Block_5_chance=0.0, Block_6_chance=0.0,
                         Text_Fusion_chance=0.0, Time_Embed_chance=0.0, Projection_chance=0.0):
        m = model.clone()
        base_sd = m.model.state_dict()
        patches_to_add = {}
        active = 0
        valid_keys = m.model_keys if hasattr(m, "model_keys") else None

        def get_chance(prefix):
            if prefix.startswith("blocks."):
                idx = int(prefix.split(".")[1])
                if idx < 5:  return Block_1_chance or base_chance
                if idx < 10: return Block_2_chance or base_chance
                if idx < 15: return Block_3_chance or base_chance
                if idx < 20: return Block_4_chance or base_chance
                if idx < 24: return Block_5_chance or base_chance
                return Block_6_chance or base_chance
            elif "txtfusion" in prefix or "txtmlp" in prefix: return Text_Fusion_chance or base_chance
            elif "tmlp" in prefix or "tproj" in prefix: return Time_Embed_chance or base_chance
            elif "first" in prefix or "last" in prefix: return Projection_chance or base_chance
            return base_chance

        for k, base_weight in base_sd.items():
            if k.endswith(".comfy_quant") or f"{k}_scale" in base_sd: continue
            if valid_keys is not None and k not in valid_keys: continue

            ck = Krea2TensorParser.clean_key(k)
            target_chance = get_chance(ck)
            if target_chance <= 0.0: continue

            fast_seed = generate_fast_seed(k, seed)
            rng = torch.Generator(device=base_weight.device)
            rng.manual_seed(fast_seed)

            if tune_mode == "Element-Level (Sub-atomic)":
                mask = torch.rand(base_weight.shape, generator=rng, device=base_weight.device) < target_chance
            else:
                v_rand = torch.rand(1, generator=rng, device=base_weight.device).item()
                mask = torch.ones_like(base_weight, dtype=torch.bool) if v_rand < target_chance else torch.zeros_like(base_weight, dtype=torch.bool)

            if not torch.any(mask): continue

            pk = f"diffusion_model.{ck}" if not ck.startswith("diffusion_model.") else ck
            w = ComfyPatcherAdapter.calculate_safe_weight(m, pk, base_weight)
            delta = (w.to("cpu") * chaos_strength) * mask.to("cpu").to(torch.bfloat16)
            patches_to_add[pk] = (delta.to(torch.bfloat16),)
            active += 1

        if patches_to_add:
            add_patches_to_front(m, patches_to_add, 1.0)
        return (m, f"Krea-2 Model Chaos Block Tuner ({tune_mode}) | Patches: {active}")

# ==============================================================================
# 4. ARTHEMY KREA2 CLIP CHAOS BLOCK TUNER
# ==============================================================================
class ArthemyKrea2CLIPChaosBlockTuner(BaseKrea2Node):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "tune_mode": (["Block-Level", "Element-Level (Sub-atomic)"],),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "chaos_strength": ("FLOAT", {"default": 0.1, "min": -99.00, "max": 99.00, "step": 0.01}),
                "base_chance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Embedding_chance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Layer_1_chance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Layer_2_chance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Layer_3_chance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Layer_4_chance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Layer_5_chance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Layer_6_chance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Layer_7_chance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Final_Projection_chance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("CLIP", "info")
    FUNCTION = "chaos_tune_clip"
    CATEGORY = "Arthemy/Krea2 Tuners"

    def chaos_tune_clip(self, clip, tune_mode, seed, chaos_strength, base_chance=0.5,
                        Embedding_chance=0.0, Layer_1_chance=0.0, Layer_2_chance=0.0,
                        Layer_3_chance=0.0, Layer_4_chance=0.0, Layer_5_chance=0.0,
                        Layer_6_chance=0.0, Layer_7_chance=0.0, Final_Projection_chance=0.0):
        c = clip.clone()
        c_p = get_patcher(c)
        base_sd = c_p.model.state_dict() if hasattr(c_p, 'model') else getattr(c, 'get_sd', lambda: {})()
        patches_to_add = {}
        active = 0
        valid_keys = getattr(c_p, 'model_keys', None)

        for k, base_weight in base_sd.items():
            if k.endswith(".position_ids") or k.endswith(".logit_scale") or f"{k}_scale" in base_sd: continue
            if valid_keys is not None and k not in valid_keys: continue

            target_chance = base_chance
            if "embed_tokens" in k: target_chance = Embedding_chance or base_chance
            elif "norm.weight" in k or "final_layer_norm" in k: target_chance = Final_Projection_chance or base_chance
            else:
                idx, _ = Krea2TensorParser.extract_clip_layer_idx(Krea2TensorParser.clean_key(k))
                if idx is not None:
                    if idx < 5:   target_chance = Layer_1_chance or base_chance
                    elif idx < 10: target_chance = Layer_2_chance or base_chance
                    elif idx < 15: target_chance = Layer_3_chance or base_chance
                    elif idx < 20: target_chance = Layer_4_chance or base_chance
                    elif idx < 25: target_chance = Layer_5_chance or base_chance
                    elif idx < 30: target_chance = Layer_6_chance or base_chance
                    else:          target_chance = Layer_7_chance or base_chance

            if target_chance <= 0.0: continue

            fast_seed = generate_fast_seed(k, seed)
            rng = torch.Generator(device=base_weight.device)
            rng.manual_seed(fast_seed)

            if tune_mode == "Element-Level (Sub-atomic)":
                mask = torch.rand(base_weight.shape, generator=rng, device=base_weight.device) < target_chance
            else:
                v_rand = torch.rand(1, generator=rng, device=base_weight.device).item()
                mask = torch.ones_like(base_weight, dtype=torch.bool) if v_rand < target_chance else torch.zeros_like(base_weight, dtype=torch.bool)

            if not torch.any(mask): continue

            w_active = ComfyPatcherAdapter.calculate_safe_weight(c, k, base_weight)
            delta = (w_active.to("cpu") * chaos_strength) * mask.to("cpu").to(torch.bfloat16)
            patches_to_add[k] = (delta.to(torch.bfloat16),)
            active += 1

        if patches_to_add:
            add_patches_to_front(c, patches_to_add, 1.0)
        return (c, f"Krea-2 CLIP Chaos Block Tuner ({tune_mode}) | Patches: {active}")

# ==============================================================================
# POINT 4 IMPLEMENTATIONS: SURGEON NODES DERIVED FROM BaseSurgeonTuner
# ==============================================================================
class ArthemyKrea2ModelBlockSurgeonTuner(BaseSurgeonTuner):
    MODEL_BLOCK_RANGES = {
        "Block_1 (0-4)": set(range(0, 5)), "Block_2 (5-9)": set(range(5, 10)),
        "Block_3 (10-14)": set(range(10, 15)), "Block_4 (15-19)": set(range(15, 20)),
        "Block_5 (20-23)": set(range(20, 24)), "Block_6 (24-27)": set(range(24, 28)),
        "All Blocks (*)": set(range(0, Krea2Config.MAX_UNET_BLOCKS)),
    }
    SURGEON_MAP = Krea2TensorParser.get_descriptive_model_surgeon_map()

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model": ("MODEL",), "target_block": (list(s.MODEL_BLOCK_RANGES.keys()),),
                "sub_block": (["Block_1A (0)", "Block_1B (1)", "Block_1C (2)", "Block_1D (3)", "Block_1E (4)",
                               "Block_2A (5)", "Block_2B (6)", "Block_2C (7)", "Block_2D (8)", "Block_2E (9)",
                               "Block_3A (10)", "Block_3B (11)", "Block_3C (12)", "Block_3D (13)", "Block_3E (14)",
                               "Block_4A (15)", "Block_4B (16)", "Block_4C (17)", "Block_4D (18)", "Block_4E (19)",
                               "Block_5A (20)", "Block_5B (21)", "Block_5C (22)", "Block_5D (23)",
                               "Block_6A (24)", "Block_6B (25)", "Block_6C (26)", "Block_6D (27)",
                               "All Sub-Blocks (*)"],),
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

    def tune_model_surgeon(self, model, target_block, sub_block, mode, vectors_override, granular_json="", **kwargs):
        if sub_block != "All Sub-Blocks (*)":
            selected = {MODEL_BLOCK_LABEL_TO_IDX[sub_block.split(" ")[0]]}
        else:
            selected = self.MODEL_BLOCK_RANGES.get(target_block, set(range(0, Krea2Config.MAX_UNET_BLOCKS)))

        return self._execute_surgeon_tuning(
            model, is_clip=False, selected_indices=selected, surgeon_map=self.SURGEON_MAP,
            kwargs=kwargs, mode=mode, vectors_override=vectors_override, granular_json=granular_json
        )


class ArthemyKrea2CLIPBlockSurgeonTuner(BaseSurgeonTuner):
    CLIP_LAYER_RANGES = {
        "Layer_1 (0-4)": set(range(0, 5)), "Layer_2 (5-9)": set(range(5, 10)),
        "Layer_3 (10-14)": set(range(10, 15)), "Layer_4 (15-19)": set(range(15, 20)),
        "Layer_5 (20-24)": set(range(20, 25)), "Layer_6 (25-29)": set(range(25, 30)),
        "Layer_7 (30-35)": set(range(30, 36)), "All Layers (*)": set(range(0, Krea2Config.MAX_CLIP_LAYERS)),
    }
    SURGEON_MAP = Krea2TensorParser.get_descriptive_clip_surgeon_map()

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "clip": ("CLIP",), "target_layer": (list(s.CLIP_LAYER_RANGES.keys()),),
                "sub_layer": (["Layer_1A (0)", "Layer_1B (1)", "Layer_1C (2)", "Layer_1D (3)", "Layer_1E (4)",
                              "Layer_2A (5)", "Layer_2B (6)", "Layer_2C (7)", "Layer_2D (8)", "Layer_2E (9)",
                              "Layer_3A (10)", "Layer_3B (11)", "Layer_3C (12)", "Layer_3D (13)", "Layer_3E (14)",
                              "Layer_4A (15)", "Layer_4B (16)", "Layer_4C (17)", "Layer_4D (18)", "Layer_4E (19)",
                              "Layer_5A (20)", "Layer_5B (21)", "Layer_5C (22)", "Layer_5D (23)", "Layer_5E (24)",
                              "Layer_6A (25)", "Layer_6B (26)", "Layer_6C (27)", "Layer_6D (28)", "Layer_6E (29)",
                              "Layer_7A (30)", "Layer_7B (31)", "Layer_7C (32)", "Layer_7D (33)", "Layer_7E (34)", "Layer_7F (35)",
                              "All Sub-Layers (*)"],),
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

    def tune_clip_surgeon(self, clip, target_layer, sub_layer, mode, vectors_override, granular_json="", **kwargs):
        if sub_layer != "All Sub-Layers (*)":
            selected = {CLIP_LAYER_LABEL_TO_IDX[sub_layer.split(" ")[0]]}
        else:
            selected = self.CLIP_LAYER_RANGES.get(target_layer, set(range(0, Krea2Config.MAX_CLIP_LAYERS)))

        return self._execute_surgeon_tuning(
            clip, is_clip=True, selected_indices=selected, surgeon_map=self.SURGEON_MAP,
            kwargs=kwargs, mode=mode, vectors_override=vectors_override, granular_json=granular_json
        )


class ArthemyKrea2ModelChaosBlockSurgeonTuner(BaseSurgeonTuner):
    MODEL_BLOCK_RANGES = ArthemyKrea2ModelBlockSurgeonTuner.MODEL_BLOCK_RANGES
    _CHANCE_MAP = {v: f"{k}_chance" for k, v in Krea2TensorParser.MODEL_SURGEON_MAP.items()}

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model": ("MODEL",), "target_block": (list(s.MODEL_BLOCK_RANGES.keys()),),
                "sub_block": (["Block_1A (0)", "Block_1B (1)", "Block_1C (2)", "Block_1D (3)", "Block_1E (4)",
                               "Block_2A (5)", "Block_2B (6)", "Block_2C (7)", "Block_2D (8)", "Block_2E (9)",
                               "Block_3A (10)", "Block_3B (11)", "Block_3C (12)", "Block_3D (13)", "Block_3E (14)",
                               "Block_4A (15)", "Block_4B (16)", "Block_4C (17)", "Block_4D (18)", "Block_4E (19)",
                               "Block_5A (20)", "Block_5B (21)", "Block_5C (22)", "Block_5D (23)",
                               "Block_6A (24)", "Block_6B (25)", "Block_6C (26)", "Block_6D (27)",
                               "All Sub-Blocks (*)"],),
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

    def chaos_tune_model_surgeon(self, model, target_block, sub_block, tune_mode, seed, chaos_strength, **kwargs):
        if sub_block != "All Sub-Blocks (*)":
            selected = {MODEL_BLOCK_LABEL_TO_IDX[sub_block.split(" ")[0]]}
        else:
            selected = self.MODEL_BLOCK_RANGES.get(target_block, set(range(0, Krea2Config.MAX_UNET_BLOCKS)))

        # Remap kwargs chances to standard surgeon_map names
        remapped_kwargs = {k.replace("_chance", ""): v for k, v in kwargs.items() if k.endswith("_chance")}

        chaos_params = {"tune_mode": tune_mode, "seed": seed, "chaos_strength": chaos_strength}
        return self._execute_surgeon_tuning(
            model, is_clip=False, selected_indices=selected, surgeon_map=Krea2TensorParser.MODEL_SURGEON_MAP,
            kwargs=remapped_kwargs, chaos_params=chaos_params
        )


class ArthemyKrea2CLIPChaosBlockSurgeonTuner(BaseSurgeonTuner):
    CLIP_LAYER_RANGES = ArthemyKrea2CLIPBlockSurgeonTuner.CLIP_LAYER_RANGES
    _CHANCE_MAP = {v: f"{k}_chance" for k, v in Krea2TensorParser.CLIP_SURGEON_MAP.items()}

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "clip": ("CLIP",), "target_layer": (list(s.CLIP_LAYER_RANGES.keys()),),
                "sub_layer": (["Layer_1A (0)", "Layer_1B (1)", "Layer_1C (2)", "Layer_1D (3)", "Layer_1E (4)",
                              "Layer_2A (5)", "Layer_2B (6)", "Layer_2C (7)", "Layer_2D (8)", "Layer_2E (9)",
                              "Layer_3A (10)", "Layer_3B (11)", "Layer_3C (12)", "Layer_3D (13)", "Layer_3E (14)",
                              "Layer_4A (15)", "Layer_4B (16)", "Layer_4C (17)", "Layer_4D (18)", "Layer_4E (19)",
                              "Layer_5A (20)", "Layer_5B (21)", "Layer_5C (22)", "Layer_5D (23)", "Layer_5E (24)",
                              "Layer_6A (25)", "Layer_6B (26)", "Layer_6C (27)", "Layer_6D (28)", "Layer_6E (29)",
                              "Layer_7A (30)", "Layer_7B (31)", "Layer_7C (32)", "Layer_7D (33)", "Layer_7E (34)", "Layer_7F (35)",
                              "All Sub-Layers (*)"],),
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

    def chaos_tune_clip_surgeon(self, clip, target_layer, sub_layer, tune_mode, seed, chaos_strength, **kwargs):
        if sub_layer != "All Sub-Layers (*)":
            selected = {CLIP_LAYER_LABEL_TO_IDX[sub_layer.split(" ")[0]]}
        else:
            selected = self.CLIP_LAYER_RANGES.get(target_layer, set(range(0, Krea2Config.MAX_CLIP_LAYERS)))

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
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            output_checkpoint, folder_paths.get_output_directory(), 0, 0
        )
        if not output_checkpoint.endswith(".safetensors"):
            output_checkpoint += ".safetensors"
        output_path = os.path.join(full_output_folder, output_checkpoint)
        os.makedirs(full_output_folder, exist_ok=True)
        logger.info(f"[ARTHEMY KREA2 MODEL SAVER] Stream-saving model: {output_checkpoint}")

        sd = model.model.state_dict()
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
            return dequantize_weight(v)

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
        logger.info(f"[ARTHEMY KREA2 MODEL SAVER] ✨ SUCCESSO: {output_path}")
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
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            output_checkpoint, folder_paths.get_output_directory(), 0, 0
        )
        if not output_checkpoint.endswith(".safetensors"):
            output_checkpoint += ".safetensors"
        output_path = os.path.join(full_output_folder, output_checkpoint)
        os.makedirs(full_output_folder, exist_ok=True)
        logger.info(f"[ARTHEMY KREA2 CLIP SAVER] Stream-saving CLIP: {output_checkpoint}")

        clip_sd = clip.get_sd()
        target_dtype = torch.float8_e4m3fn if precision == "FP8_E4M3" else torch.bfloat16
        final_sd = {}

        def process_clip_tensor(k, v):
            return dequantize_weight(v)

        for clean_key, tensor in process_tensor_stream(clip_sd, process_clip_tensor, target_dtype=target_dtype, memory_threshold_mb=Krea2Config.DEFAULT_GC_THRESHOLD_MB):
            if tensor is not None:
                final_sd[clean_key] = tensor

        metadata = {"format": "pt"}
        safetensors.torch.save_file(final_sd, output_path, metadata=metadata)
        del final_sd
        gc.collect()
        logger.info(f"[ARTHEMY KREA2 CLIP SAVER] ✨ SUCCESSO: {output_path}")
        return (output_path,)

# ==============================================================================
# LORA BLOCK LOADERS (Isolated & Chaos)
# ==============================================================================
class ArthemyKrea2LoraBlockLoader(BaseKrea2Node):
    @classmethod
    def INPUT_TYPES(s):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",), "clip": ("CLIP",),
                "lora_name": (lora_list,),
                "strength_model": ("FLOAT", {"default": 0.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 0.0, "min": -99.00, "max": 99.00, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "info")
    FUNCTION = "load_lora"
    CATEGORY = "Arthemy/Krea2 LoRA"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, "LoRA bypassed (strength 0)")
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        m, c = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (m, c, f"Loaded LoRA: {lora_name}")


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
        if reset_model and m_p and hasattr(m_p, "patches"):
            m_p.patches = {}
        if reset_clip and c_p and hasattr(c_p, "patches"):
            c_p.patches = {}
        return (m, c, "Reset patchers clean.")


class ArthemyKrea2IsolatedLoraBlockLoader(BaseKrea2Node):
    """Isolated LoRA loader: clones the patcher before applying so the LoRA
    does not accumulate on top of any previously active patches from upstream nodes.
    Produces standard ComfyUI 2-tensor (lora_down, lora_up) patch entries, which
    are correctly detected as is_lora=True by the Visualizer's parse_patch_entry."""

    @classmethod
    def INPUT_TYPES(s):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",), "clip": ("CLIP",), "lora_name": (lora_list,),
                "strength_model": ("FLOAT", {"default": 0.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 0.0, "min": -99.00, "max": 99.00, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "info")
    FUNCTION = "load_lora"
    CATEGORY = "Arthemy/Krea2 LoRA"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, "Bypassed")
        m = ComfyPatcherAdapter.isolate_patcher(model)
        c = ComfyPatcherAdapter.isolate_patcher(clip)
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        res_m, res_c = comfy.sd.load_lora_for_models(m, c, lora, strength_model, strength_clip)
        return (res_m, res_c, f"Isolated LoRA: {lora_name}")


class ArthemyKrea2LoadChaosLoRA(BaseKrea2Node):
    @classmethod
    def INPUT_TYPES(s):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",), "clip": ("CLIP",), "lora_name": (lora_list,),
                "strength_model": ("FLOAT", {"default": 0.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 0.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "chaos_chance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "info")
    FUNCTION = "load_chaos_lora"
    CATEGORY = "Arthemy/Krea2 LoRA"

    def load_chaos_lora(self, model, clip, lora_name, strength_model, strength_clip, chaos_chance, seed):
        m = ComfyPatcherAdapter.isolate_patcher(model)
        c = ComfyPatcherAdapter.isolate_patcher(clip)
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        filtered_lora = {}
        for i, (k, v) in enumerate(lora.items()):
            key_hash = int(hashlib.md5(k.encode()).hexdigest(), 16) % (2**31)
            rng = torch.Generator().manual_seed((seed + key_hash) % (2**31))
            if torch.rand(1, generator=rng).item() < chaos_chance:
                filtered_lora[k] = v

        res_m, res_c = comfy.sd.load_lora_for_models(m, c, filtered_lora, strength_model, strength_clip)
        return (res_m, res_c, f"Chaos LoRA Loaded ({len(filtered_lora)} keys)")


class ArthemyKrea2LoadChaosLoraBlockSurgeon(BaseSurgeonTuner):
    @classmethod
    def INPUT_TYPES(s):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",), "clip": ("CLIP",), "lora_name": (lora_list,),
                "strength_model": ("FLOAT", {"default": 0.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 0.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "chaos_chance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "info")
    FUNCTION = "load_chaos_lora_surgeon"
    CATEGORY = "Arthemy/Krea2 LoRA"

    def load_chaos_lora_surgeon(self, model, clip, lora_name, strength_model, strength_clip, chaos_chance, seed):
        m = ComfyPatcherAdapter.isolate_patcher(model)
        c = ComfyPatcherAdapter.isolate_patcher(clip)
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        filtered_lora = {}
        for k, v in lora.items():
            key_hash = int(hashlib.md5(k.encode()).hexdigest(), 16) % (2**31)
            rng = torch.Generator().manual_seed((seed + key_hash) % (2**31))
            if torch.rand(1, generator=rng).item() < chaos_chance:
                filtered_lora[k] = v

        res_m, res_c = comfy.sd.load_lora_for_models(m, c, filtered_lora, strength_model, strength_clip)
        return (res_m, res_c, f"Chaos Surgeon LoRA ({len(filtered_lora)} keys)")

# ==============================================================================
# POINT 3 IMPLEMENTATION: MODEL BAKER WITH STREAM GENERATOR
# ==============================================================================
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

        m_sd = m_baked.model.state_dict()
        c_p = get_patcher(c_baked)
        m_p = get_patcher(m_baked)
        c_sd = c_p.model.state_dict() if hasattr(c_p, "model") else {}
        m_sd = m_p.model.state_dict() if hasattr(m_p, "model") else {}

        n_model_patched = 0
        n_clip_patched = 0

        # Stream process model tensors
        def bake_model_tensor(k, v):
            nonlocal n_model_patched
            pk = f"diffusion_model.{k}" if not k.startswith("diffusion_model.") else k
            if pk in getattr(m_p, "patches", {}):
                n_model_patched += 1
                return ComfyPatcherAdapter.calculate_safe_weight(m_baked, pk, v)
            return v

        for _, _ in process_tensor_stream(m_sd, bake_model_tensor, memory_threshold_mb=Krea2Config.DEFAULT_GC_THRESHOLD_MB):
            pass

        def bake_clip_tensor(k, v):
            nonlocal n_clip_patched
            if k in getattr(c_p, "patches", {}):
                n_clip_patched += 1
                return ComfyPatcherAdapter.calculate_safe_weight(c_baked, k, v)
            return v

        for _, _ in process_tensor_stream(c_sd, bake_clip_tensor, memory_threshold_mb=Krea2Config.DEFAULT_GC_THRESHOLD_MB):
            pass

        # Clear patches after baking
        if hasattr(m_p, "patches"): m_p.patches = {}
        if hasattr(c_p, "patches"): c_p.patches = {}
        

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

    # Only evaluate tensor payload for LoRA or Chaos if strength_patch != 0.0
    # (strength_patch == 0.0 indicates a scalar tuner patch where the 2D tensor is solely for VBAR memory alignment)
    if strength_patch != 0.0:
        if isinstance(diff, (tuple, list)):
            if len(diff) >= 2:
                if any(isinstance(x, torch.Tensor) and x.ndim >= 2 for x in diff):
                    is_lora = True
                    offset_delta += strength_patch * 0.10
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
    top_y = int(height * 0.22)
    chart_w = width - (padding_x * 2)
    chart_h = max(60, height - top_y - int(height * 0.15))
    center_y = top_y + (chart_h // 2)

    # 1. Outer container box
    draw.rectangle([padding_x - 4, top_y - 20, padding_x + chart_w + 4, top_y + chart_h + 30], outline=base_color, width=2)

    # 2. Header title
    draw.text((padding_x, top_y - 18), title, fill=base_color)

    # 3. Header legend & scale
    scale_str = f"Scale: {visual_scale:.1f}x"
    draw.text((width - padding_x - 100, top_y - 18), scale_str, fill=(255, 255, 255, 150))

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
    axis_y = top_y + chart_h + 8
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

# ==============================================================================
# NODE MAPPINGS & REGISTRATION
# ==============================================================================
NODE_CLASS_MAPPINGS = {
    "ArthemyKrea2ModelTuner": ArthemyKrea2ModelTuner,
    "ArthemyKrea2CLIPTuner": ArthemyKrea2CLIPTuner,
    "ArthemyKrea2ModelChaosBlockTuner": ArthemyKrea2ModelChaosBlockTuner,
    "ArthemyKrea2CLIPChaosBlockTuner": ArthemyKrea2CLIPChaosBlockTuner,
    "ArthemyKrea2ModelBlockSurgeonTuner": ArthemyKrea2ModelBlockSurgeonTuner,
    "ArthemyKrea2CLIPBlockSurgeonTuner": ArthemyKrea2CLIPBlockSurgeonTuner,
    "ArthemyKrea2ModelChaosBlockSurgeonTuner": ArthemyKrea2ModelChaosBlockSurgeonTuner,
    "ArthemyKrea2CLIPChaosBlockSurgeonTuner": ArthemyKrea2CLIPChaosBlockSurgeonTuner,
    "ArthemyKrea2ModelSaver": ArthemyKrea2ModelSaver,
    "ArthemyKrea2CLIPSaver": ArthemyKrea2CLIPSaver,
    "ArthemyKrea2LoraBlockLoader": ArthemyKrea2LoraBlockLoader,
    "ArthemyKrea2ResetPatcher": ArthemyKrea2ResetPatcher,
    "ArthemyKrea2IsolatedLoraBlockLoader": ArthemyKrea2IsolatedLoraBlockLoader,
    # Backwards-compatibility alias: old workflows using the Surgeon variant resolve to the merged class
    "ArthemyKrea2IsolatedLoraBlockSurgeonLoader": ArthemyKrea2IsolatedLoraBlockLoader,
    "ArthemyKrea2LoadChaosLoRA": ArthemyKrea2LoadChaosLoRA,
    "ArthemyKrea2LoadChaosLoraBlockSurgeon": ArthemyKrea2LoadChaosLoraBlockSurgeon,
    "ArthemyKrea2ModelBaker": ArthemyKrea2ModelBaker,
    "ArthemyKrea2ModelVisualizer": ArthemyKrea2ModelVisualizer,
    "ArthemyKrea2CLIPVisualizer": ArthemyKrea2CLIPVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArthemyKrea2ModelTuner": "✨ Arthemy Krea-2 Model Tuner",
    "ArthemyKrea2CLIPTuner": "✨ Arthemy Krea-2 CLIP Tuner",
    "ArthemyKrea2ModelChaosBlockTuner": "🌪️ Arthemy Krea-2 Model Chaos Block Tuner",
    "ArthemyKrea2CLIPChaosBlockTuner": "🌪️ Arthemy Krea-2 CLIP Chaos Block Tuner",
    "ArthemyKrea2ModelBlockSurgeonTuner": "🔬 Arthemy Krea-2 Model Block Surgeon Tuner",
    "ArthemyKrea2CLIPBlockSurgeonTuner": "🔬 Arthemy Krea-2 CLIP Block Surgeon Tuner",
    "ArthemyKrea2ModelChaosBlockSurgeonTuner": "🌪️🔬 Arthemy Krea-2 Model Chaos Block Surgeon",
    "ArthemyKrea2CLIPChaosBlockSurgeonTuner": "🌪️🔬 Arthemy Krea-2 CLIP Chaos Block Surgeon",
    "ArthemyKrea2ModelSaver": "💾 Arthemy Krea-2 Model Saver",
    "ArthemyKrea2CLIPSaver": "💾 Arthemy Krea-2 CLIP Saver",
    "ArthemyKrea2LoraBlockLoader": "🔮 Arthemy Krea-2 LoRA Block Loader",
    "ArthemyKrea2ResetPatcher": "🔄 Arthemy Krea-2 Reset Patcher",
    "ArthemyKrea2IsolatedLoraBlockLoader": "🔮 Arthemy Krea-2 Isolated LoRA Block Loader",
    "ArthemyKrea2IsolatedLoraBlockSurgeonLoader": "🔮 Arthemy Krea-2 Isolated LoRA Block Loader",
    "ArthemyKrea2LoadChaosLoRA": "🎲 Arthemy Krea-2 Load Chaos LoRA",
    "ArthemyKrea2LoadChaosLoraBlockSurgeon": "🎲🔬 Arthemy Krea-2 Load Chaos LoRA Block Surgeon",
    "ArthemyKrea2ModelBaker": "Arthemy Krea-2 Model Baker",
    "ArthemyKrea2ModelVisualizer": "📊 Arthemy Krea-2 Model Visualizer",
    "ArthemyKrea2CLIPVisualizer": "📊 Arthemy Krea-2 CLIP Visualizer",
}

WEB_DIRECTORY = "./web"

import nodes
_web_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "web"))
if os.path.isdir(_web_path) and hasattr(nodes, "EXTENSION_WEB_DIRS"):
    nodes.EXTENSION_WEB_DIRS["Arthemy_Krea2_Tuner"] = _web_path

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

"""
Arthemy Krea-2 Suite for ComfyUI

Architecture Overview:
1. Native ComfyUI Patcher Integration.
2. Dynamic Probing Engine in Krea2TensorParser (State-dict architecture probing with regex fallback).
3. Memory-Safe Generator & In-Place Processing (Out-Of-Memory prevention for Baker & Savers).
4. DRY BaseSurgeonTuner Architecture (Unified Model & CLIP Surgeon / Chaos Surgeon engine).
"""

import collections
import colorsys
import copy
import gc
import json
import logging
import os
import re
import sys
import time
import uuid
import zlib
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import torch
import safetensors.torch
from PIL import Image, ImageDraw, ImageFont

import comfy.lora
import comfy.model_patcher
import comfy.sd
import comfy.utils
import folder_paths

# Optional: only present inside a running ComfyUI server, never in the offline test
# harness (which imports this module standalone, with no "server" package on the path).
# Used solely to push a live rotation report to the frontend without touching a node's
# (obj, info) return contract, which the Preset Loader itself calls directly by node
# method - changing that tuple's shape would break every replay in this file, not just
# the tests that also call these methods directly.
try:
    from server import PromptServer
except Exception:
    PromptServer = None

# ==============================================================================
# LOGGER (declared before any helper that might log during module import)
# ==============================================================================
logger = logging.getLogger("ArthemyKrea2Suite")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ==============================================================================
# COMFYUI WEIGHT ADAPTER RESOLUTION (compatibility across ComfyUI versions)
# ==============================================================================
_WEIGHT_ADAPTER_MOD = None
for _mod_getter in (
    lambda: __import__("comfy.weight_adapter", fromlist=["LoRAAdapter"]),
    lambda: getattr(comfy.lora, "weight_adapter"),
):
    try:
        _candidate = _mod_getter()
        if getattr(_candidate, "LoRAAdapter", None) is not None:
            _WEIGHT_ADAPTER_MOD = _candidate
            break
    except Exception:
        continue

LORA_ADAPTER_CLS = getattr(_WEIGHT_ADAPTER_MOD, "LoRAAdapter", None) if _WEIGHT_ADAPTER_MOD else None
WEIGHT_ADAPTER_BASE_CLS = getattr(_WEIGHT_ADAPTER_MOD, "WeightAdapterBase", None) if _WEIGHT_ADAPTER_MOD else None

if LORA_ADAPTER_CLS is None:
    logger.error(
        "[Arthemy Krea-2 Suite] ComfyUI weight_adapter API not found. "
        "This suite requires a recent ComfyUI build (comfy.weight_adapter.LoRAAdapter). "
        "Rotation nodes will be disabled until ComfyUI is updated."
    )

# ==============================================================================
# GEOMETRY ENGINE (safe path-injected standalone & package import)
# ==============================================================================
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# Defaults keep the module importable (and every node visible with a clear error in its
# info string) even when the geometry engine is missing.
LowRankDelta = None
ROTATION_RANK_MAP = {"Light": 8, "Default": 16, "Heavy": 32}
_geom_ROTATION_MODE_OUTPUT = "Output Manifold (R @ W)"
_geom_ROTATION_MODE_INPUT = "Input Manifold (W @ R)"
STORAGE_DCT = "dct"
STORAGE_RAW = "raw"
dct_fidelity = None
build_modifier_from_factors = None
fast_style_compass_rotation = None
fast_dual_orthogonal_rotation = None
fast_chaos_orthogonal_rotation = None
extract_lora_to_5d_dct = None
synthesize_5d_patches_from_dct = None
peek_lora_vector_lengths = None


def normalize_signed_angle(deg: float) -> float:
    """Identical fallback to the geometry engine's, used when that engine is unavailable."""
    a = float(deg) % 360.0
    return a - 360.0 if a > 180.0 else a


GEOMETRY_ENGINE_ERROR = None

try:
    try:
        import arthemy_geometry_engine as _geom
    except Exception:
        from . import arthemy_geometry_engine as _geom  # type: ignore[no-redef]
    LowRankDelta = _geom.LowRankDelta
    ROTATION_RANK_MAP = _geom.ROTATION_RANK_MAP
    _geom_ROTATION_MODE_OUTPUT = _geom.ROTATION_MODE_OUTPUT
    _geom_ROTATION_MODE_INPUT = _geom.ROTATION_MODE_INPUT
    STORAGE_DCT = _geom.STORAGE_DCT
    STORAGE_RAW = _geom.STORAGE_RAW
    dct_fidelity = _geom.dct_fidelity
    build_modifier_from_factors = _geom.build_modifier_from_factors
    fast_style_compass_rotation = _geom.fast_style_compass_rotation
    fast_dual_orthogonal_rotation = _geom.fast_dual_orthogonal_rotation
    fast_chaos_orthogonal_rotation = _geom.fast_chaos_orthogonal_rotation
    extract_lora_to_5d_dct = _geom.extract_lora_to_5d_dct
    synthesize_5d_patches_from_dct = _geom.synthesize_5d_patches_from_dct
    peek_lora_vector_lengths = _geom.peek_lora_vector_lengths
    normalize_signed_angle = _geom.normalize_signed_angle
    _dct_matrix = getattr(_geom, "_dct_matrix", None)
except Exception as _e_geom:
    _dct_matrix = None
    GEOMETRY_ENGINE_ERROR = str(_e_geom)
    logger.error(f"[Arthemy Krea-2 Suite] Failed to import arthemy_geometry_engine: {_e_geom}. "
                 "Rotator and 5D nodes will report an error instead of silently doing nothing.")

# Unified single directory for arthemy_presets: strictly inside custom_nodes/Arthemy_Krea2_Tuner/presets
arthemy_presets_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "presets")
local_presets_dir = arthemy_presets_dir
os.makedirs(arthemy_presets_dir, exist_ok=True)

# Register presets directory so ComfyUI dropdowns read strictly from custom_nodes/Arthemy_Krea2_Tuner/presets
if "arthemy_presets" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["arthemy_presets"] = ([arthemy_presets_dir], {".json"})


class Krea2Config:
    """Global architecture configuration parameters and defaults.

    MAX_* are the *fallback* bounds for the stock Krea-2 / Qwen3 pair. They are only used
    when a dropdown label cannot be resolved; the real block and layer counts are probed
    from the state dict by `Krea2TensorParser.probe_architecture`.
    """
    MAX_UNET_BLOCKS: int = 28
    MAX_CLIP_LAYERS: int = 60
    DEFAULT_GC_THRESHOLD_MB: int = 500
    HIDDEN_SIZE: int = 6144
    MLP_SIZE: int = 16384

    @classmethod
    def probe_hidden_size(cls, state_dict: Dict[str, Any]) -> int:
        """The residual width, read off the checkpoint instead of assumed.

        HIDDEN_SIZE is the stock Krea-2 figure and stays as the fallback, but it must not be
        the answer: on another base (z-image, say) every vertical-tuning node would look for
        6144-wide matrices, find none, and report "no projection layers found" as if the model
        were odd rather than the constant being wrong.

        The residual width is the dimension that shows up most often across the 2-D weights -
        it appears on both sides of every attention projection and on one side of every MLP
        matrix, so it wins comfortably over the MLP width. Shapes only: no tensor is read.
        """
        counts: Dict[int, int] = {}
        for k, v in (state_dict or {}).items():
            if not isinstance(v, torch.Tensor) or v.ndim != 2:
                continue
            if is_bookkeeping_sd_key(k):
                continue
            for d in (int(v.shape[0]), int(v.shape[1])):
                if d >= 8:
                    counts[d] = counts.get(d, 0) + 1
        if not counts:
            return cls.HIDDEN_SIZE
        # Ties go to the SMALLER dimension: the residual width sits on both sides of every
        # attention projection, the MLP width on one side only, so a tie means we are not
        # looking at a transformer and the narrower axis is the safer guess.
        width, hits = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
        # One sighting is a coincidence, not an architecture.
        return width if hits >= 2 else cls.HIDDEN_SIZE

    @classmethod
    def probe(cls, state_dict: Dict[str, Any], is_clip: bool) -> int:
        """Number of blocks / layers actually present in this checkpoint."""
        try:
            info = Krea2TensorParser.probe_architecture(state_dict)
            return int(info["num_clip_layers"] if is_clip else info["num_model_blocks"])
        except Exception:
            return cls.MAX_CLIP_LAYERS if is_clip else cls.MAX_UNET_BLOCKS

def parse_granular_json(json_string: str) -> Dict[str, Any]:
    """Parses granular JSON configurations, supporting single dictionaries and automatically
    merging multiple consecutive JSON blocks pasted together."""
    if not json_string or not json_string.strip():
        return {}
    
    # 1. Try standard direct parse
    try:
        parsed_data = json.loads(json_string)
        if isinstance(parsed_data, dict):
            return parsed_data
    except Exception:
        pass

    # 2. Resilient multi-block JSON decoding (merges multiple consecutive { ... } { ... } blocks)
    merged = {}
    decoder = json.JSONDecoder()
    s = json_string.strip()
    idx = 0
    parsed_any = False
    
    while idx < len(s):
        while idx < len(s) and s[idx].isspace():
            idx += 1
        if idx >= len(s):
            break
        try:
            obj, end = decoder.raw_decode(s, idx)
            if isinstance(obj, dict):
                merged.update(obj)
                parsed_any = True
            if end <= idx:
                idx += 1
            else:
                idx = end
        except Exception as e:
            if not parsed_any:
                logger.warning(f"[Arthemy Granular JSON] Parse error on raw input: {e}")
                return {}
            break

    if parsed_any:
        return merged
    return {}

def generate_fast_seed(key_string: str, base_seed: int) -> int:
    """Generates a fast, deterministic seed using CRC32."""
    key_hash = zlib.crc32(key_string.encode("utf-8"))
    return (key_hash ^ base_seed) % (2**31)

# ==============================================================================
# HELPER DATA STRUCTURES & CONSTANTS
# ==============================================================================

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
    "Visual_1A": 36, "Visual_1B": 37, "Visual_1C": 38, "Visual_1D": 39, "Visual_1E": 40, "Visual_1F": 41,
    "Visual_2A": 42, "Visual_2B": 43, "Visual_2C": 44, "Visual_2D": 45, "Visual_2E": 46, "Visual_2F": 47,
    "Visual_3A": 48, "Visual_3B": 49, "Visual_3C": 50, "Visual_3D": 51, "Visual_3E": 52, "Visual_3F": 53,
    "Visual_4A": 54, "Visual_4B": 55, "Visual_4C": 56, "Visual_4D": 57, "Visual_4E": 58, "Visual_4F": 59,
}

# Reverse lookups (index -> label), so visualizers do not scan the whole label map per patch
MODEL_IDX_TO_LABEL = {v: k for k, v in MODEL_BLOCK_LABEL_TO_IDX.items()}
CLIP_IDX_TO_LABEL = {v: k for k, v in CLIP_LAYER_LABEL_TO_IDX.items()}

# ==============================================================================
# POINT 2: DYNAMIC PROBING ENGINE (Krea2TensorParser)
# ==============================================================================
class Krea2TensorParser:
    """Dynamic Probing Engine & Sub-Tensor Parser for Krea-2 Model & Qwen3 Text Encoder.
    Uses dynamic state_dict inspection to deduce architecture hierarchy programmatically,
    falling back to regex only if required."""

    MODEL_SURGEON_MAP = {
        "ATTN_wq_query": ("attn.wq.weight", "attn.q.weight", "attn.wq", "attn.q"),
        "ATTN_wk_key": ("attn.wk.weight", "attn.k.weight", "attn.wk", "attn.k"),
        "ATTN_wv_value": ("attn.wv.weight", "attn.v.weight", "attn.wv", "attn.v"),
        "ATTN_wo_out": ("attn.wo.weight", "attn.o.weight", "attn.wo", "attn.o"),
        "ATTN_gate_attn": ("attn.gate.weight", "attn.gate"),
        "ATTN_qknorm_scales": ("attn.qknorm.qnorm.scale", "attn.qknorm.knorm.scale"),
        "MLP_gate_swiglu": ("mlp.gate.weight", "mlp.gate"),
        "MLP_up_proj": ("mlp.up.weight", "mlp.up"),
        "MLP_down_proj": ("mlp.down.weight", "mlp.down"),
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
        for prefix in (
            "model.diffusion_model.", "diffusion_model.", "cond_stage_model.",
            "model.language_model.", "model.", "transformer.model.", "transformer."
        ):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if ".transformer.model." in key:
            key = key.split(".transformer.model.", 1)[1]
        elif ".transformer." in key:
            key = key.split(".transformer.", 1)[1]
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
                    if "layer" in prev_part or prev_part == "layers":
                        discovered_layers.add(idx)
                    elif prev_part == "blocks" and i > 1 and parts[i-2] == "visual":
                        discovered_layers.add(idx + 36)
                    elif "block" in prev_part or prev_part == "blocks":
                        discovered_blocks.add(idx)

        num_model_blocks = max(discovered_blocks) + 1 if discovered_blocks else Krea2Config.MAX_UNET_BLOCKS
        num_clip_layers = max(discovered_layers) + 1 if discovered_layers else Krea2Config.MAX_CLIP_LAYERS

        return {
            "num_model_blocks": num_model_blocks,
            "num_clip_layers": num_clip_layers,
        }

    @classmethod
    def extract_model_block_idx(cls, clean_key: str) -> Tuple[Optional[int], str]:
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
    def extract_clip_layer_idx(cls, clean_key: str) -> Tuple[Optional[int], str]:
        """Dynamic inspection of CLIP layer index with regex fallback."""
        parts = clean_key.split(".")
        for i, p in enumerate(parts):
            if p.isdigit() and i > 0:
                if "layer" in parts[i-1] or parts[i-1] == "layers":
                    return int(p), ".".join(parts[i+1:])
                elif parts[i-1] == "blocks" and i > 1 and parts[i-2] == "visual":
                    return int(p) + 36, ".".join(parts[i+1:])
        m = RE_CLIP_LAYERS_FULL.search(clean_key)
        if m:
            return int(m.group(1)), m.group(2)
        m_vis = RE_CLIP_VISUAL_BLOCKS.search(clean_key)
        if m_vis:
            return int(m_vis.group(1)) + 36, m_vis.group(2)
        return None, clean_key

    @classmethod
    def match_model_sub_tensor(cls, sub_key: str) -> Optional[str]:
        for widget_key, target_suffixes in cls.MODEL_SURGEON_MAP.items():
            for target_suffix in target_suffixes:
                if (sub_key == target_suffix 
                    or sub_key.endswith(f".{target_suffix}") 
                    or sub_key.startswith(f"{target_suffix}.") 
                    or f".{target_suffix}." in sub_key):
                    return widget_key
        return None

    @classmethod
    def match_clip_sub_tensor(cls, sub_key: str) -> Optional[str]:
        for widget_key, target_suffixes in cls.CLIP_SURGEON_MAP.items():
            for target_suffix in target_suffixes:
                if (sub_key == target_suffix 
                    or sub_key.endswith(f".{target_suffix}") 
                    or sub_key.startswith(f"{target_suffix}.") 
                    or f".{target_suffix}." in sub_key):
                    return widget_key
        return None

    @classmethod
    def get_descriptive_model_surgeon_map(cls) -> Dict[str, Tuple[str, ...]]:
        return dict(cls.MODEL_SURGEON_MAP)

    @classmethod
    def get_descriptive_clip_surgeon_map(cls) -> Dict[str, Tuple[str, ...]]:
        return dict(cls.CLIP_SURGEON_MAP)

# ==============================================================================
# POINT 1: NATIVE COMFYUI PATCHER ADAPTER (No UUID Monkey-Patching)
# ==============================================================================
class ComfyPatcherAdapter:
    """Safe wrapper for ComfyUI ModelPatcher operations using official APIs."""

    @staticmethod
    def calculate_safe_weight(model_or_clip: Any, key: str, base_weight: torch.Tensor, model_sd: Optional[Dict[str, Any]] = None) -> torch.Tensor:
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
            for scale_cand in (companion_scale_candidates(resolved_key)
                               + companion_scale_candidates(key)):
                if scale_cand in model_sd:
                    scale = model_sd[scale_cand]
                    break

        clean_base = dequantize_weight(get_clean_weight(patcher, resolved_key, base_weight), scale=scale)
        if current_patches:
            return comfy.lora.calculate_weight(current_patches, clean_base.float().clone(), resolved_key).to(torch.bfloat16)
        return clean_base

# Pre-compiled regular expressions for high-frequency state-dict iterations
RE_MODEL_BLOCKS_FULL = re.compile(r"blocks\.(\d+)\.(.+)$")
RE_CLIP_LAYERS_FULL = re.compile(r"layers\.(\d+)\.(.+)$")
RE_CLIP_VISUAL_BLOCKS = re.compile(r"visual\.blocks\.(\d+)\.(.+)$")
RE_TXTFUSION_LAYERWISE = re.compile(r"txtfusion\.layerwise_blocks\.(\d+)\.")
RE_TXTFUSION_REFINER = re.compile(r"txtfusion\.refiner_blocks\.(\d+)\.")
RE_GENERAL_BLOCKS = re.compile(r"blocks\.(\d+)\.")

# Minimal 2-byte dummy tensor for VBAR ComfyUI patch alignment
DUMMY_PATCH_TENSOR = torch.zeros((1, 1), dtype=torch.bfloat16)

# Custom markers propagated across tensor clones and inspected by visualizers / preset saver
ARTHEMY_TENSOR_ATTRS = (
    "_is_arthemy_rotation",
    "_arthemy_rotation_angle",
    "_arthemy_rotation_hue",
    "_arthemy_rotation_mode",
    "_arthemy_depth_reach",
    "_arthemy_subspace_rank",
    "_arthemy_relative_delta",
    "_is_arthemy_chaos",
    "_is_arthemy_five_d",
    "_is_arthemy_granular",
)

# Patch provenance kinds, shared by the injector, both visualizers and the preset saver.
KIND_GRANULAR = "granular"
KIND_ROTATION = "rotation"
KIND_FIVE_D = "five_d"
KIND_LORA = "lora"


def is_weight_adapter(obj: Any) -> bool:
    """True when obj is a ComfyUI WeightAdapter (LoRAAdapter & friends)."""
    if obj is None:
        return False
    if WEIGHT_ADAPTER_BASE_CLS is not None and isinstance(obj, WEIGHT_ADAPTER_BASE_CLS):
        return True
    return callable(getattr(obj, "calculate_weight", None))


def _stamp_arthemy_meta(obj: Any, meta: Dict[str, Any]) -> Any:
    """Attaches Arthemy provenance markers to an adapter / tensor.

    These are a convenience for downstream inspection only. They are NOT the source of
    truth for the visualizer: ComfyUI rebuilds adapters via
    `type(value)(loaded_keys, weights)` in its VRAM prefetch path, which drops any
    custom attribute. Section-level provenance lives in `model_options`
    (see `record_section_meta`).
    """
    if obj is None or not meta:
        return obj
    kind = meta.get("kind")
    try:
        if kind == KIND_ROTATION:
            setattr(obj, "_is_arthemy_rotation", True)
            setattr(obj, "_arthemy_rotation_angle", float(meta.get("angle", 0.0)))
            setattr(obj, "_arthemy_rotation_hue", float(meta.get("hue", 180.0)))
            setattr(obj, "_arthemy_rotation_mode", meta.get("rotation_type", "dual_lie"))
            setattr(obj, "_arthemy_depth_reach", meta.get("depth_reach", "Default"))
            setattr(obj, "_arthemy_subspace_rank", int(meta.get("subspace_rank", 0)))
            setattr(obj, "_arthemy_relative_delta", float(meta.get("relative_delta", 0.0)))
            if meta.get("is_chaos"):
                setattr(obj, "_is_arthemy_chaos", True)
        elif kind == KIND_FIVE_D:
            setattr(obj, "_is_arthemy_five_d", True)
            setattr(obj, "_arthemy_relative_delta", float(meta.get("relative_delta", 0.0)))
    except AttributeError:
        # Some torch builds refuse attribute assignment on certain tensor subclasses.
        pass
    return obj


# ==============================================================================
# CHANNEL-SCALE ADAPTER (vertical tuning without dense deltas)
# ==============================================================================
# Rescaling one axis of a weight by a per-channel vector is a broadcast, but ComfyUI's patch
# list can only express a per-tensor scalar (strength_model) or a per-element tensor. Building
# the per-element form materialises a delta the size of the model - ~23 GB in bf16 for Krea-2 -
# which then has to be moved to the GPU on every weight load: that is where the 250 s/iteration
# came from, and it is 2 bytes of payload for every 4 bytes of information the user typed.
#
# A WeightAdapter carries the 24 KB vector instead and does the broadcast inside the weight
# load, where ComfyUI has already allocated the working copy. One vector object is shared by
# every key it applies to, so the whole model costs one allocation, not eighty-four.

# `isinstance(..., type)`, not `is not None`: a ComfyUI build that exposes the name as
# something other than a class must fall back to the dense path rather than fail at import.
HAS_WEIGHT_ADAPTER_API = isinstance(WEIGHT_ADAPTER_BASE_CLS, type)
_ChannelAdapterBase = WEIGHT_ADAPTER_BASE_CLS if HAS_WEIGHT_ADAPTER_API else object


# Keys already warned about, so a mismatch logs once instead of once per module cast.
_CHANNEL_SCALE_WARNED: Set[str] = set()


class ArthemyChannelScaleAdapter(_ChannelAdapterBase):
    """Per-channel multiplicative gain along one axis of a 2-D weight.

    The axis lives in the CLASS, not in the data, and that is not a style choice.
    comfy/lora.py::prefetch_prepared_value reconstructs an adapter as

        type(value)(value.loaded_keys, prefetch_prepared_value(value.weights, ...))

    and every tensor inside `weights` is gathered into one VRAM buffer, copied there
    ASYNCHRONOUSLY on a stream; what comes back is a view into that buffer. So `weights` is
    reserved for weight data, and reading a value out of it at construction time (an
    `.item()` on an axis tensor, say) returns whatever is in the buffer at that instant -
    which is how this adapter first scaled the wrong axis and then indexed past the end of
    a shape. `type(value)` is the one thing the reconstruction preserves for free.

    Use the two concrete subclasses, never this base directly.
    """

    name = "arthemy_channel_scale"
    AXIS = 0

    def __init__(self, loaded_keys, weights):
        # Nothing is read out of `weights` here - see the class docstring.
        self.loaded_keys = loaded_keys if loaded_keys is not None else set()
        self.weights = weights

    @property
    def scales(self) -> torch.Tensor:
        # Read through `weights` rather than cached, so a reconstruction that re-homed the
        # tensor into the gathered buffer is picked up instead of the stale original.
        return self.weights[0]

    @property
    def axis(self) -> int:
        return self.AXIS

    @classmethod
    def from_scales(cls, scales: torch.Tensor, axis: int = 0) -> "ArthemyChannelScaleAdapter":
        vec = scales.detach().to(torch.float32).flatten().contiguous()
        target = ArthemyChannelRowScaleAdapter if int(axis) == 0 else ArthemyChannelColScaleAdapter
        return target(set(), [vec])

    # Class-level, so a ComfyUI reconstruction - which replays only loaded_keys and weights -
    # still carries the marker that keeps the visualizers and the Preset Saver from reporting
    # this as an external LoRA (see parse_patch_entry / ARTHEMY_TENSOR_ATTRS).
    _is_arthemy_granular = True

    @classmethod
    def load(cls, x, lora, alpha, dora_scale, loaded_keys=None):
        # Never loaded from a LoRA file; ComfyUI's adapter registry does not know this class.
        return None

    def to_train(self):
        raise NotImplementedError("ArthemyChannelScaleAdapter is inference-only.")

    def calculate_weight(self, weight, key, strength, strength_model=1.0, offset=None,
                         function=None, intermediate_dtype=torch.float32,
                         original_weights=None, **kwargs):
        # `strength_model` has already been applied to `weight` by comfy.lora.calculate_weight
        # before this runs - re-applying it here would square it.
        if strength == 0.0:
            return weight

        axis = self.axis
        if weight.ndim < 2 or weight.shape[axis] != self.scales.numel():
            # ComfyUI narrows `weight` itself when a patch carries an offset, so a mismatch here
            # means this vector does not describe the tensor in front of us. Skipping is the only
            # safe answer: broadcasting the wrong length would silently scale the wrong channels.
            if key not in _CHANNEL_SCALE_WARNED:
                _CHANNEL_SCALE_WARNED.add(key)
                logger.warning(f"[Arthemy Channel Scale] '{key}' has {tuple(weight.shape)} but the "
                               f"channel vector is {self.scales.numel()} long on axis {axis}; "
                               "this patch was skipped.")
            return weight

        # bf16/fp16 weights are scaled in their own dtype: a multiplicative gain is a relative
        # operation, so the single rounding it costs is the precision the weight already has,
        # and an fp32 upcast would allocate a second full-size copy of every layer.
        if weight.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            compute_dtype = weight.dtype
        else:
            compute_dtype = intermediate_dtype or torch.float32

        s = self.scales.to(device=weight.device, dtype=compute_dtype)
        if strength != 1.0:
            s = 1.0 + (s - 1.0) * float(strength)

        shape = [1] * weight.ndim
        shape[axis] = -1
        s = s.view(shape)

        # Fast path: with no transform to apply, this is an in-place broadcast multiply and
        # allocates nothing at all. `function` is an identity lambda in the common case, which
        # a one-element probe detects without assuming anything about the caller.
        if function is None:
            weight *= s
            return weight
        try:
            probe = torch.zeros(1, dtype=compute_dtype, device=weight.device)
            if function(probe) is probe:
                weight *= s
                return weight
        except Exception:
            pass

        # General path: express the gain as the additive delta every other adapter produces, so
        # `function` and `strength` compose here exactly as they do for a LoRA.
        diff = weight.to(compute_dtype) * (s - 1.0)
        weight += function(diff).to(weight.dtype)
        return weight


class ArthemyChannelRowScaleAdapter(ArthemyChannelScaleAdapter):
    """Scales ROWS: a matrix that writes into the residual stream (d_out = hidden)."""
    name = "arthemy_channel_scale_rows"
    AXIS = 0


class ArthemyChannelColScaleAdapter(ArthemyChannelScaleAdapter):
    """Scales COLUMNS: a matrix that reads from the residual stream (d_in = hidden)."""
    name = "arthemy_channel_scale_cols"
    AXIS = 1


def build_channel_scale_patch(scales: torch.Tensor, axis: int) -> Optional[Any]:
    """The adapter when this ComfyUI build supports one, otherwise None (dense fallback)."""
    if not HAS_WEIGHT_ADAPTER_API:
        return None
    if scales is None or not isinstance(scales, torch.Tensor) or scales.numel() == 0:
        return None
    vec = scales.detach().to(torch.float32).flatten().contiguous()
    if bool(torch.allclose(vec, torch.ones_like(vec))):
        return None
    return ArthemyChannelScaleAdapter.from_scales(vec, axis=axis)


ARTHEMY_PROFILES_DIRNAME = "arthemy_profiles"

# Bump when the meaning of a cached profile changes (metric, participating matrices, layout),
# so every stale file on disk is recomputed instead of silently believed.
CHANNEL_PROFILE_FORMAT = 4

# fingerprint -> ordering, for repeated executions inside one session.
_CHANNEL_PROFILE_MEM: "collections.OrderedDict[str, torch.Tensor]" = collections.OrderedDict()
_CHANNEL_PROFILE_MEM_MAX = 8


def get_arthemy_profiles_dir() -> str:
    """Where cached channel profiles live (next to the modifiers, in the models folder)."""
    prof_dir = os.path.join(folder_paths.models_dir, ARTHEMY_PROFILES_DIRNAME)
    os.makedirs(prof_dir, exist_ok=True)
    return prof_dir


def channel_profile_matrices(base_sd: Dict[str, Any], target_dim: int, axis: int,
                             scope: str = "All Components") -> List[str]:
    """The keys whose norms make up the profile for one axis, in a stable order.

    `axis=0` (writes) profiles the matrices that write into the residual, by ROW norm;
    `axis=1` (reads) the ones that read from it, by COLUMN norm. The metric has to match the
    axis the gain will act on: ranking channels by how strongly they are READ and then scaling
    how strongly they are WRITTEN measures one quantity and moves another.
    """
    keys = []
    for k, v in (base_sd or {}).items():
        if not isinstance(v, torch.Tensor) or v.ndim != 2:
            continue
        if is_skippable_for_tuning(k, base_sd):
            continue
        if int(v.shape[0 if axis == 0 else 1]) != int(target_dim):
            continue
        if not channel_scope_matches(k, scope):
            continue
        keys.append(k)
    return sorted(keys)


def channel_profile_fingerprint(base_sd: Dict[str, Any], keys: List[str], target_dim: int,
                                axis: int, scope: str = "All Components") -> str:
    """Identity of a profile, derived from the exact matrices that produce it.

    Not the checkpoint's filename: two finetunes share a name-shaped identity and every shape,
    so a name- or shape-only key would hand one model the other's bands. Not a full hash
    either - that is 12 GB of reading to save 18 seconds. Shapes and dtypes of every
    participating matrix, plus a content sample from a few of them, is enough to separate two
    finetunes AND to separate a checkpoint from its own FP8 twin (whose participating set
    differs, because the quantized matrices are skipped).
    """
    h = zlib.crc32(f"v{CHANNEL_PROFILE_FORMAT}|dim={int(target_dim)}|axis={int(axis)}"
                   f"|scope={scope}|n={len(keys)}".encode())
    for k in keys:
        v = base_sd.get(k)
        shape = tuple(int(x) for x in v.shape) if isinstance(v, torch.Tensor) else ()
        dtype = str(v.dtype) if isinstance(v, torch.Tensor) else "?"
        h = zlib.crc32(f"|{k}:{shape}:{dtype}".encode(), h)

    # Content sample: a fixed slice from a handful of evenly spaced matrices. Cheap (a few KB
    # read) and enough to tell two finetunes of the same architecture apart.
    if keys:
        step = max(1, len(keys) // 8)
        for k in keys[::step][:8]:
            v = base_sd.get(k)
            if not isinstance(v, torch.Tensor) or v.numel() == 0:
                continue
            try:
                sample = v.detach().flatten()[:256].to(torch.float32).cpu().numpy().tobytes()
                h = zlib.crc32(sample, h)
            except Exception:
                # A tensor that cannot be read (lazy, offloaded, exotic) contributes its
                # identity only. Weaker, never wrong: shapes and dtypes still separate.
                h = zlib.crc32(f"|unreadable:{k}".encode(), h)
    return f"{h & 0xFFFFFFFF:08x}"


def _profile_cache_path(fingerprint: str) -> str:
    return os.path.join(get_arthemy_profiles_dir(), f"channel_{fingerprint}.json")


def load_cached_channel_profile(fingerprint: str, target_dim: int) -> Optional[torch.Tensor]:
    """The stored ordering for this fingerprint, or None. Never raises."""
    cached = _CHANNEL_PROFILE_MEM.get(fingerprint)
    if cached is not None:
        _CHANNEL_PROFILE_MEM.move_to_end(fingerprint)
        return cached
    path = _profile_cache_path(fingerprint)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if int(data.get("format", -1)) != CHANNEL_PROFILE_FORMAT:
            return None
        if int(data.get("target_dim", -1)) != int(target_dim):
            return None
        order = data.get("order")
        if not isinstance(order, list) or len(order) != int(target_dim):
            return None
        t = torch.tensor(order, dtype=torch.long)
        # A hand-edited or truncated file must not reorder channels at random.
        if int(t.min()) < 0 or int(t.max()) >= int(target_dim) or int(torch.unique(t).numel()) != int(target_dim):
            logger.warning(f"[Arthemy Profiler] Cached profile '{os.path.basename(path)}' is not a "
                           "permutation of the channels; recomputing.")
            return None
    except Exception as e:
        logger.debug(f"[Arthemy Profiler] Could not read cached profile: {e}")
        return None
    _CHANNEL_PROFILE_MEM[fingerprint] = t
    while len(_CHANNEL_PROFILE_MEM) > _CHANNEL_PROFILE_MEM_MAX:
        _CHANNEL_PROFILE_MEM.popitem(last=False)
    return t


def store_cached_channel_profile(fingerprint: str, target_dim: int, axis: int,
                                 order: torch.Tensor, n_matrices: int,
                                 scope: str = "All Components") -> None:
    """Persists an ordering. A failure here costs 18 seconds next time, nothing else."""
    _CHANNEL_PROFILE_MEM[fingerprint] = order
    while len(_CHANNEL_PROFILE_MEM) > _CHANNEL_PROFILE_MEM_MAX:
        _CHANNEL_PROFILE_MEM.popitem(last=False)
    try:
        with open(_profile_cache_path(fingerprint), "w", encoding="utf-8") as f:
            json.dump({
                "format": CHANNEL_PROFILE_FORMAT,
                "fingerprint": fingerprint,
                "target_dim": int(target_dim),
                "axis": int(axis),
                "metric": "row_l2" if int(axis) == 0 else "col_l2",
                "scope": scope,
                "matrices": int(n_matrices),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "order": [int(x) for x in order.tolist()],
            }, f)
    except Exception as e:
        logger.warning(f"[Arthemy Profiler] Could not write the profile cache: {e}")


def compute_channel_profile(patcher: Any, base_sd: Dict[str, Any], keys: List[str],
                            target_dim: int, axis: int) -> Tuple[Optional[torch.Tensor], int]:
    """Average per-channel L2 over the participating matrices. Returns (ordering, n_used)."""
    accum = torch.zeros(int(target_dim), dtype=torch.float32)
    used = 0
    reduce_dim = 1 if int(axis) == 0 else 0     # rows for writes, columns for reads
    for k in keys:
        weight = base_sd.get(k)
        if not isinstance(weight, torch.Tensor):
            continue
        # FP8: dequantize with the companion scale, or the norms rank the codes, not the values.
        scale = find_companion_scale(k, base_sd)
        clean_w = dequantize_weight(get_clean_weight(patcher, k, weight), scale=scale)
        norms = torch.linalg.norm(clean_w.detach().to(torch.float32), dim=reduce_dim)
        if norms.numel() == int(target_dim):
            accum += norms
            used += 1
    if used == 0:
        return None, 0
    return torch.argsort(accum / float(used), descending=True), used


def resolve_channel_profile(patcher: Any, base_sd: Dict[str, Any], target_dim: int,
                            axis: int, scope: str = "All Components") -> Tuple[Optional[torch.Tensor], str]:
    """The channel ordering for one axis, from cache when it describes THIS checkpoint.

    Returns (ordering, note). The cache is keyed by a fingerprint of the participating
    matrices, so it invalidates itself on a different checkpoint, a different quantization or
    a different axis - nobody has to remember to clear it.
    """
    keys = channel_profile_matrices(base_sd, target_dim, axis, scope)
    if not keys:
        return None, "no participating matrices"
    fingerprint = channel_profile_fingerprint(base_sd, keys, target_dim, axis, scope)

    cached = load_cached_channel_profile(fingerprint, target_dim)
    if cached is not None:
        return cached, f"profile {fingerprint} from cache"

    t0 = time.time()
    order, used = compute_channel_profile(patcher, base_sd, keys, target_dim, axis)
    if order is None:
        return None, "no usable matrices"
    store_cached_channel_profile(fingerprint, target_dim, axis, order, used, scope)
    return order, f"profile {fingerprint} computed from {used} matrices in {time.time() - t0:.1f}s"


CHANNEL_PATH_WRITES = "Residual gain (writes only)"
CHANNEL_PATH_READS = "Channel attention (reads only)"
CHANNEL_PATH_BOTH = "Both (compounds)"
# Reads first: it is the default, after measuring that it gives the better prompt adherence
# on the real model, and that the profile metric is itself a read-side statistic.
# Both first: inside a scope it is the default, because it is the only setting that applies
# the gain on both sides of the nonlinearity.
CHANNEL_PATH_CHOICES = [CHANNEL_PATH_BOTH, CHANNEL_PATH_READS, CHANNEL_PATH_WRITES]

CHANNEL_SCOPE_DEFAULT = "MLP (Gate/Up/Down)"

CHANNEL_SCOPE_TOOLTIP = (
    "Which sub-network the band gain acts on. Uses the same component groups as the Surgeon "
    "nodes.\n"
    "MLP (default): the gain lands on both sides of the SwiGLU - gate/up on the way in, down on "
    "the way out. Measured as the sub-network that carries the interesting part of the change: "
    "at gain 1.20 it reproduces the whole-model MLP effect (x1.067) while leaving attention "
    "untouched (x0.997).\n"
    "ATTN: the same, on the attention projections.\n"
    "All Components: everything at once - the widest setting, and the one where a Both path "
    "compounds across sub-networks."
)


def channel_scope_matches(key: str, scope: str) -> bool:
    """True when `key` belongs to the chosen sub-network.

    Reuses `match_sub_component`, the same predicate the Surgeon and rotator nodes use, so the
    vertical tuners cannot drift into their own private idea of what "MLP" means.
    """
    return match_sub_component(Krea2TensorParser.clean_key(key), scope, is_clip=False)


CHANNEL_PATH_TOOLTIP = (
    "Which side of the residual stream the gain acts on, INSIDE the chosen scope.\n"
    "Both (default): the gain is applied where the sub-network reads from the residual AND "
    "where it writes back. For an MLP that means both sides of the nonlinearity, which changes "
    "the shape of its contribution and not merely its volume - silu(s*x) is not s*silu(x). This "
    "is the setting that reproduces the tool's original behaviour.\n"
    "Reads only: shifts how much the sub-network listens to those channels. Measurably better "
    "prompt adherence, but on an MLP it does only half the job (x1.044 against x1.067).\n"
    "Writes only: a linear gain after the nonlinearity, which never moves its operating point "
    "(x1.021). Useful analytically, weakest in practice.\n"
    "The non-linear runaway measured at high gains belongs to Both on All Components, where it "
    "compounds across sub-networks; inside one scope it is well behaved."
)


def channel_scale_axes_for_path(path_mode: str) -> Tuple[bool, bool]:
    """(scale_writes, scale_reads) for a channel_path widget value."""
    p = str(path_mode or CHANNEL_PATH_BOTH)
    if p.startswith("Both"):
        return True, True
    if p.startswith("Channel attention"):
        return False, True
    return True, False


def inject_channel_scales(model_clone: Any, patcher: Any, base_sd: Dict[str, Any],
                          channel_scales: torch.Tensor, target_dim: int, path_mode: str,
                          label: str, scope: str = "All Components") -> Tuple[int, int, str]:
    """Attaches the per-channel gain to every matrix the chosen path touches.

    Returns (n_applied, n_quant_skipped, note). One adapter object per axis is shared by every
    key on that axis, so the payload for a whole model is two 1-D vectors.
    """
    scale_writes, scale_reads = channel_scale_axes_for_path(path_mode)
    write_patch = build_channel_scale_patch(channel_scales, axis=0) if scale_writes else None
    read_patch = build_channel_scale_patch(channel_scales, axis=1) if scale_reads else None
    dense_fallback = not HAS_WEIGHT_ADAPTER_API

    patches_to_add: Dict[str, Any] = {}
    valid_keys = getattr(patcher, "model_keys", None)
    quant_skipped = 0

    for k, weight in base_sd.items():
        if is_skippable_for_tuning(k, base_sd):
            if not is_bookkeeping_sd_key(k):
                quant_skipped += 1
            continue
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            continue
        if valid_keys is not None and k not in valid_keys:
            continue

        if not channel_scope_matches(k, scope):
            continue

        d_out, d_in = weight.shape
        is_write = scale_writes and d_out == target_dim
        is_read = scale_reads and d_in == target_dim
        if not (is_write or is_read):
            continue

        clean_k = Krea2TensorParser.clean_key(k)
        patch_key = clean_k if clean_k.startswith("diffusion_model.") else f"diffusion_model.{clean_k}"

        if dense_fallback:
            # Pre-WeightAdapter ComfyUI: the dense delta is the only expressible form.
            scale = find_companion_scale(k, base_sd)
            clean_w = dequantize_weight(get_clean_weight(patcher, k, weight), scale=scale).detach().to(torch.float32)
            factor = torch.ones_like(clean_w)
            if is_write:
                factor *= channel_scales.unsqueeze(1)
            if is_read:
                factor *= channel_scales.unsqueeze(0)
            delta = (clean_w * (factor - 1.0)).to(torch.bfloat16)
            setattr(delta, "_is_arthemy_granular", True)
            patches_to_add[patch_key] = (delta,)
            continue

        # A square matrix on "Both" needs two entries; add_patches appends, so ComfyUI applies
        # them in order and the two gains compose - which is exactly what "Both" means.
        if is_write and is_read:
            inject_patches(model_clone, {patch_key: write_patch}, 1.0)
            patches_to_add[patch_key] = read_patch
        elif is_write:
            patches_to_add[patch_key] = write_patch
        else:
            patches_to_add[patch_key] = read_patch

    applied = inject_patches(model_clone, patches_to_add, 1.0)
    note = "dense fallback (this ComfyUI has no weight_adapter API)" if dense_fallback else "1-D adapter"
    if dense_fallback:
        logger.warning(f"[{label}] comfy.weight_adapter is unavailable, so the gain had to be "
                       "materialised as dense deltas. Update ComfyUI to get the 1-D path.")
    return len(applied), quant_skipped, note


def build_geometry_patch(payload: Any, layer_name: str = "") -> Tuple[Optional[Any], Dict[str, Any]]:
    """Converts a geometry-engine result into a ComfyUI patch value plus its metadata.

    This is the single place where the suite talks to ComfyUI's patch protocol.

    CRITICAL: modern ComfyUI has NO `"lora"` branch left in
    `comfy.lora.calculate_weight` - every low-rank patch type was moved into
    `comfy.weight_adapter`. A `("lora", (up, down, alpha, ...))` tuple therefore falls
    straight through to `logging.warning("patch type not recognized lora <key>")` and is
    discarded. Low-rank deltas MUST be handed over as a real `LoRAAdapter` instance.
    """
    if payload is None:
        return None, {}

    # 1. Native low-rank delta from the geometry engine (rotations, 5D synthesis)
    if LowRankDelta is not None and isinstance(payload, LowRankDelta):
        if LORA_ADAPTER_CLS is None:
            logger.warning("[Arthemy Geometry] ComfyUI weight_adapter API unavailable; low-rank patch skipped.")
            return None, {}
        adapter = LORA_ADAPTER_CLS(
            loaded_keys={layer_name or payload.meta.get("layer_name", "")},
            weights=(payload.A, payload.B, None, None, None, None),
        )
        return _stamp_arthemy_meta(adapter, payload.meta), dict(payload.meta)

    # 2. Already a WeightAdapter: pass through untouched. calculate_weight detects it
    #    *before* any tuple unwrapping, so it must never be wrapped in a tuple.
    if is_weight_adapter(payload):
        meta = {"kind": KIND_ROTATION} if getattr(payload, "_is_arthemy_rotation", False) else {"kind": KIND_LORA}
        return payload, meta

    # 3. Explicit ("diff"/"set", value) form is a valid ComfyUI patch value.
    if isinstance(payload, tuple) and len(payload) == 2 and isinstance(payload[0], str):
        if payload[0] == "lora":
            logger.error("[Arthemy Geometry] Refusing to inject a legacy ('lora', ...) tuple: "
                         "modern ComfyUI drops it silently. Return a LowRankDelta instead.")
            return None, {}
        return payload, {"kind": payload[0]}

    # 4. Dense delta tensor -> plain 1-tuple, which calculate_weight reads as "diff".
    if isinstance(payload, torch.Tensor):
        return (payload,), {"kind": KIND_GRANULAR}

    if isinstance(payload, (tuple, list)):
        inner = tuple(payload)
        if len(inner) == 1:
            if is_weight_adapter(inner[0]):
                return inner[0], {"kind": KIND_LORA}
            if isinstance(inner[0], torch.Tensor):
                return inner, {"kind": KIND_GRANULAR}
        logger.error(f"[Arthemy Geometry] Unsupported geometry payload of length {len(inner)} "
                     f"for '{layer_name}'; patch skipped.")
        return None, {}

    return payload, {}


def normalize_geometry_patch(payload: Any, layer_name: str = "") -> Optional[Any]:
    """Backwards-compatible wrapper returning only the patch value."""
    value, _meta = build_geometry_patch(payload, layer_name)
    return value


# ==============================================================================
# SECTION-LEVEL PATCH PROVENANCE (source of truth for the visualizers)
# ==============================================================================
SECTION_META_KEY = "arthemy_section_meta"


def record_section_meta(patcher: Any, domain: str, per_index: Dict[int, Dict[str, Any]]) -> None:
    """Merges per-block/per-layer provenance into model_options.

    Bounded by design (at most 28 blocks / 36 layers + 3 pseudo-sections), so unlike a
    per-tensor map it stays cheap even though ComfyUI deep-copies model_options on every
    downstream patcher clone. This is what lets the visualizer report the exact rotation
    angle of every single section instead of guessing it from a recipe.
    """
    if patcher is None or not hasattr(patcher, "model_options") or not per_index:
        return
    store = dict(patcher.model_options.get(SECTION_META_KEY, {}))
    domain_store = dict(store.get(domain, {}))
    for idx, meta in per_index.items():
        key = str(idx)
        current = dict(domain_store.get(key, {}))
        for mk, mv in meta.items():
            if mk == "angle":
                # Rotations compose: accumulate the magnitude rather than overwriting it,
                # so stacking two rotator nodes on the same block reads as the total.
                current["angle"] = round(float(current.get("angle", 0.0)) + float(mv), 2)
            elif mk == "relative_delta":
                current["relative_delta"] = round(float(current.get("relative_delta", 0.0)) + float(mv), 6)
            elif mk in ("is_rotation", "is_chaos", "is_five_d"):
                current[mk] = bool(current.get(mk, False) or mv)
            else:
                current[mk] = mv
        domain_store[key] = current
    store[domain] = domain_store
    patcher.model_options[SECTION_META_KEY] = store


def get_section_meta(patcher: Any, domain: str) -> Dict[str, Dict[str, Any]]:
    if patcher is None or not hasattr(patcher, "model_options"):
        return {}
    return dict(patcher.model_options.get(SECTION_META_KEY, {}).get(domain, {}))

def sanitize_patch_tensor(tensor: Optional[torch.Tensor], target_dtype: Optional[torch.dtype] = None, target_device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
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
    san = tensor.to(dtype=dtype, device=device).clone().detach().contiguous()
    for attr in ARTHEMY_TENSOR_ATTRS:
        if hasattr(tensor, attr):
            setattr(san, attr, getattr(tensor, attr))
    return san

# Bounded cache of per-state_dict lookup indexes.
# Keyed by (id(model_sd), len(model_sd), first_key) so a recycled id() cannot silently
# return a stale index for a different state dict.
_KEY_INDEX_CACHE: "collections.OrderedDict[Tuple[int, int, Optional[str]], Dict[str, Any]]" = collections.OrderedDict()
_KEY_INDEX_CACHE_MAX = 8


def _get_key_index(model_sd: Dict[str, Any]) -> Dict[str, Any]:
    """Builds (once per state_dict) a clean_key -> real_key index plus a resolution memo.

    Turns the previous O(n) per-key suffix scan into an amortized O(1) lookup, which matters
    because resolve_target_key is called inside loops over thousands of tensors.
    """
    try:
        first_key = next(iter(model_sd))
    except StopIteration:
        first_key = None
    cache_key = (id(model_sd), len(model_sd), first_key)

    cached = _KEY_INDEX_CACHE.get(cache_key)
    if cached is not None:
        _KEY_INDEX_CACHE.move_to_end(cache_key)
        return cached

    clean_index: Dict[str, str] = {}
    for sd_k in model_sd.keys():
        clean_index.setdefault(Krea2TensorParser.clean_key(sd_k), sd_k)

    entry = {"clean": clean_index, "memo": {}}
    _KEY_INDEX_CACHE[cache_key] = entry
    while len(_KEY_INDEX_CACHE) > _KEY_INDEX_CACHE_MAX:
        _KEY_INDEX_CACHE.popitem(last=False)
    return entry


def resolve_target_key(patcher: Any, k: str, model_sd: Optional[Dict[str, Any]] = None) -> str:
    """Finds the exact state_dict key in patcher.model matching k with amortized O(1) lookup."""
    if model_sd is None:
        if patcher is None or not hasattr(patcher, "model"):
            return k
        try:
            model_sd = patcher.model.state_dict()
        except Exception:
            return k

    if k in model_sd:
        return k

    index = _get_key_index(model_sd)
    memo = index["memo"]
    if k in memo:
        return memo[k]

    clean_k = Krea2TensorParser.clean_key(k)
    resolved = None

    if clean_k in model_sd:
        resolved = clean_k
    else:
        for cand in (
            f"diffusion_model.{clean_k}",
            f"cond_stage_model.{clean_k}",
            f"model.{clean_k}",
            f"model.language_model.{clean_k}",
            f"clip_model.{clean_k}",
            f"transformer.{clean_k}",
            f"transformer.model.{clean_k}",
        ):
            if cand in model_sd:
                resolved = cand
                break

    if resolved is None:
        # Index lookup: handles arbitrary text-encoder wrappers (qwen3vl_*.transformer.model. etc.)
        resolved = index["clean"].get(clean_k)

    if resolved is None:
        resolved = k

    memo[k] = resolved
    return resolved

def inject_patches(model_patcher: Any, patches: Dict[str, Any], strength_patch: float = 1.0, strength_model: float = 1.0) -> List[str]:
    """Safely injects patches into a ModelPatcher using the standard ComfyUI add_patches API.

    Dynamically resolves exact key names in the model/clip state_dict and formats
    patch entries as a 1-tuple (diff,) so comfy.lora.calculate_weight sees len(v) == 1 and patch_type == 'diff'.
    Enforces contiguous memory layout, dtype matching, and empty buffer guards on all injected patches.

    NOTE: ComfyUI's add_patches appends to the patch list; the returned list contains only the
    keys that were actually resolved and injected, so callers can report truthful counts.
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

        def _commit(payload, s_patch, s_model):
            accepted = patcher.add_patches({target_k: payload}, strength_patch=s_patch, strength_model=s_model)
            if accepted is None:
                if hasattr(patcher, "patches") and target_k in patcher.patches:
                    p_keys.add(target_k)
            elif isinstance(accepted, (list, tuple, set)):
                for ak in accepted:
                    p_keys.add(ak)
            else:
                p_keys.add(target_k)

        if isinstance(val, (tuple, list)) and len(val) == 1 and is_weight_adapter(val[0]):
            val = val[0]

        if is_weight_adapter(val):
            _commit(val, s_patch=strength_patch, s_model=strength_model)
            continue

        # ("lora", (up, down, alpha, ...)) / ("diff", tensor) style payloads are passed through verbatim
        if isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], str):
            _commit(val, s_patch=strength_patch, s_model=strength_model)
            continue

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
            # Minimal 2-byte dummy tensor for scalar multiplier without bloating RAM or overflowing HostBuffer
            formatted = (DUMMY_PATCH_TENSOR,)
            _commit(formatted, s_patch=0.0, s_model=scalar_mult)

        elif tensor_payload is not None:
            _commit(tensor_payload, s_patch=strength_patch, s_model=strength_model)

    patcher.patches_uuid = uuid.uuid4()
    return list(p_keys)


# Backwards-compatible alias (historical name; add_patches actually appends, it does not prepend)
add_patches_to_front = inject_patches

# ==============================================================================
# BASE NODE & UTILITIES
# ==============================================================================

def get_patcher(obj: Any) -> Any:
    """Safely retrieves the ModelPatcher/ModelPatcherDynamic object whether obj is a model/clip wrapper or patcher directly."""
    if obj is None:
        return None
    return getattr(obj, "patcher", obj)


# All model_options keys used by this suite to persist deterministic recipes.
ARTHEMY_RECIPE_KEYS = (
    "arthemy_chaos_recipes",
    "arthemy_rotation_recipes",
    "arthemy_chaos_rotation_recipes",
    "arthemy_5d_recipes",
    "arthemy_channel_recipes",
    "arthemy_section_meta",
)


def clear_arthemy_recipes(patcher: Any) -> None:
    """Drops every Arthemy recipe from a patcher (used by Reset Patcher and Baker).

    Leaving stale recipes behind would make a downstream Preset Saver persist tunings that
    are already fused into the weights, double-applying them on preset reload.
    """
    if patcher is None or not hasattr(patcher, "model_options"):
        return
    for rk in ARTHEMY_RECIPE_KEYS:
        patcher.model_options.pop(rk, None)


def append_recipe(patcher: Any, recipe_key: str, entry: Dict[str, Any]) -> None:
    """Appends a recipe to model_options without mutating a list shared with a parent clone."""
    if patcher is None or not hasattr(patcher, "model_options"):
        return
    existing = patcher.model_options.get(recipe_key, [])
    patcher.model_options[recipe_key] = list(existing) + [entry]


def resolve_target_map_entry(target_map: Dict[str, Any], label: str, fallback: Any) -> Any:
    """Resolves a dropdown label to its index set, tolerating legacy/renamed labels.

    `fallback` may be a callable, so an expensive default (probing the checkpoint for its
    real block count) is only computed when the label genuinely cannot be resolved.
    """
    selected = target_map.get(label, None)
    if selected is not None:
        return selected
    clean = (label or "").strip().lstrip("\u21b3").strip()
    if clean:
        for k, v in target_map.items():
            if clean in k:
                return v
    logger.warning(f"[Arthemy] Unrecognised target label '{label}'; falling back to the full range.")
    return fallback() if callable(fallback) else fallback


def normalize_depth_reach(value: Any) -> str:
    """Coerces any stored depth_reach (label, legacy int rank, None) to a valid label."""
    if isinstance(value, str) and value in ROTATION_RANK_MAP:
        return value
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return "Default"
    # Pick the preset whose rank is closest to the stored number.
    return min(ROTATION_RANK_MAP.items(), key=lambda kv: abs(kv[1] - rank))[0]


# Sub-component filter tokens, shared by every rotator / surgeon node so the Model and CLIP
# variants can no longer drift apart.
_SUBCOMPONENT_TOKENS = {
    False: {  # diffusion model
        "ATTN": ("attn",),
        "MLP": ("mlp",),
        "NORM": ("mod", "norm"),
    },
    True: {  # text encoder
        "ATTN": ("attn", "q_proj", "k_proj", "v_proj", "o_proj"),
        "MLP": ("mlp", "gate_proj", "up_proj", "down_proj"),
        "NORM": ("norm",),
    },
}


def match_sub_component(clean_key: str, sub_components: str, is_clip: bool) -> bool:
    """True when clean_key belongs to the requested sub-component group."""
    if not sub_components or sub_components.startswith("All"):
        return True
    tokens = _SUBCOMPONENT_TOKENS[bool(is_clip)]
    if sub_components.startswith("ATTN"):
        group = tokens["ATTN"]
    elif sub_components.startswith("MLP"):
        group = tokens["MLP"]
    elif sub_components.startswith("NORM") or "MOD" in sub_components:
        group = tokens["NORM"]
    else:
        return True
    return any(tok in clean_key for tok in group)


COMBO_EMPTY = "None"


def combo_options(items: Any) -> List[str]:
    """A combo list that is never empty, and whose first entry is always a valid default.

    ComfyUI cannot render a combo widget with zero options: the node becomes unconfigurable
    and queuing it fails validation instead of saying what is missing. An empty loras or
    modifiers folder is a perfectly ordinary state on a fresh install, so it has to produce a
    usable node - the placeholder then fails at execution time with a message that names the
    folder, which is the error the user can act on.
    """
    opts = [str(x) for x in (items or [])]
    return opts if opts else [COMBO_EMPTY]


def is_bookkeeping_sd_key(k: str) -> bool:
    """Non-weight tensors: registered buffers, quantization metadata blobs and companion scales.

    These are never tunable and never re-emitted as weights, in any node.
    """
    return (k.endswith(".position_ids")
            or k.endswith(".logit_scale")
            or k.endswith(".comfy_quant")
            or k.endswith("_scale")
            or k.endswith(".weight_scale")
            or k.endswith(".scale_weight")
            or k.endswith("_scale_weight"))


def is_skippable_for_tuning(k: str, state_dict: Dict[str, Any]) -> bool:
    """Keys that a tuner / surgeon / rotator must leave alone.

    On top of the bookkeeping tensors, this also excludes FP8 weights that carry a companion
    scale: a scalar multiplier or a bf16 delta applied there would land *before* the scale is
    applied, silently double-scaling the layer.

    NOTE: the Savers deliberately use a different rule - they fold the scale into the weight via
    dequantize_weight and DO emit it (see process_tensor_stream).
    """
    if is_bookkeeping_sd_key(k):
        return True
    return any(cand in state_dict for cand in companion_scale_candidates(k))


def warn_if_quantized_skipped(node_label: str, skipped: int) -> None:
    """Tells the user when a node was a no-op because the checkpoint is quantized."""
    if skipped:
        logger.warning(f"[{node_label}] {skipped} quantized tensor(s) (FP8 weight + companion scale) "
                       "were left untouched, because patching them would be applied on top of the "
                       "scale factor. Use a BF16 checkpoint to tune those layers.")


class BaseKrea2Node:
    """Base class for all Arthemy Krea-2 custom nodes."""
    pass

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

def get_clean_weight(patcher: Any, key: str, current_weight: torch.Tensor) -> torch.Tensor:
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
                if fp32_w.ndim > 1 and scale_val.ndim == 1:
                    # Reshape [out] to [out, 1, 1...] for proper broadcast across remaining dimensions
                    scale_val = scale_val.view(-1, *([1] * (fp32_w.ndim - 1)))
            else:
                scale_val = float(scale)
            return fp32_w * scale_val
        return fp32_w
    return weight

# Generic tensor-name tokens: matching these as a *suffix* would hit almost every tensor
# in the model, so they are only honoured on an exact key match.
_AMBIGUOUS_GRANULAR_TOKENS = frozenset({
    "weight", "bias", "scale", "alpha", "gate", "norm", "attn", "mlp", "mod",
})


def granular_key_matches(clean_key: str, pfx: str) -> bool:
    """Boundary-anchored match between a state-dict key and a granular_json selector.

    Prevents a selector such as "weight" from silently matching every tensor of the model
    (the old implementation used a bare endswith/startswith test).
    """
    if not pfx:
        return False
    if clean_key == pfx:
        return True
    if pfx in _AMBIGUOUS_GRANULAR_TOKENS:
        return False
    if pfx.endswith("."):
        return clean_key.startswith(pfx) or f".{pfx}" in clean_key
    return (clean_key.startswith(f"{pfx}.")
            or clean_key.endswith(f".{pfx}")
            or f".{pfx}." in clean_key)


def resolve_granular_entry(clean_key: str, granular_map: Dict[str, Any], default_val: Any = None) -> Any:
    if not granular_map:
        return default_val
    if clean_key in granular_map:
        return granular_map[clean_key]
    for pfx, val in granular_map.items():
        if granular_key_matches(clean_key, pfx):
            return val
    return default_val

# ==============================================================================
# ROTATION FACADE
# ==============================================================================
# All rotation geometry now lives in arthemy_geometry_engine, aligned with the dominant
# SVD subspace of each weight. The previous implementation here cached a QR basis of
# Gaussian noise (up to 256 MB of orthonormal random matrices) and rotated a *random*
# subspace, which conserved the norm but barely touched the directions the layer
# actually uses. Both the cache and the Woodbury identity are therefore gone.

ROTATION_MODE_OUTPUT = _geom_ROTATION_MODE_OUTPUT
ROTATION_MODE_INPUT = _geom_ROTATION_MODE_INPUT


def rotation_to_dense_delta(payload: Any, base_weight: torch.Tensor) -> Optional[torch.Tensor]:
    """Materializes the exact dense delta of a low-rank rotation (A @ B, reshaped like W).

    Needed only when a rotation has to be combined with a multiplicative scale, which a
    low-rank factorization cannot express on its own.
    """
    if payload is None:
        return None
    if LowRankDelta is not None and isinstance(payload, LowRankDelta):
        with torch.no_grad():
            return payload.dense(base_weight.shape)
    weights = getattr(payload, "weights", None)
    if not weights or len(weights) < 2:
        return None
    A, B = weights[0], weights[1]
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        return None
    with torch.no_grad():
        return (A.to(torch.float32) @ B.to(torch.float32)).reshape(base_weight.shape)


# Historical alias
rotation_adapter_to_dense_delta = rotation_to_dense_delta


def build_granular_patch(granular_val: Any, base_weight: torch.Tensor, mode: str = "Soft Value", default_scalar: float = 0.0) -> Optional[Tuple[Any, ...]]:
    """Builds a scalar multiplier tuple (1.0 + strength,) or an additive tensor diff tuple (diff_tensor,)
    from a granular value (scalar offset, list/tuple of floats, sparse channel dict, or rotation specification)."""
    if granular_val is None:
        if default_scalar == 0.0:
            return None
        target_w = soft_target_weight(default_scalar, mode)
        strength = target_w - 1.0
        return (1.0 + strength,) if strength != 0 else None

    # Case 1: List or tuple of channel values (additive delta or direct diff)
    if isinstance(granular_val, (list, tuple)):
        try:
            val_list = [float(x) for x in granular_val]
            if len(val_list) == base_weight.numel():
                t = torch.tensor(val_list, dtype=torch.bfloat16, device="cpu").view_as(base_weight)
                setattr(t, "_is_arthemy_granular", True)
                return (t,)
            elif len(val_list) > 0 and base_weight.ndim >= 1 and len(val_list) == base_weight.shape[-1]:
                t = torch.tensor(val_list, dtype=torch.bfloat16, device="cpu").expand_as(base_weight).contiguous()
                setattr(t, "_is_arthemy_granular", True)
                return (t,)
            else:
                logger.warning(f"[Arthemy Granular] Vector size mismatch: got {len(val_list)} values, expected {base_weight.numel()} for shape {tuple(base_weight.shape)}.")
                return None
        except (ValueError, TypeError) as e:
            logger.warning(f"[Arthemy Granular] Failed to parse array for granular patch: {e}")
            return None

    # Case 2: Dictionary (sparse channel deltas, full diff, or SO(N) Latent Space Rotation)
    elif isinstance(granular_val, dict):
        try:
            # Check for Rotation / Orthogonal specification in JSON
            if any(k in granular_val for k in ("rotation_angle", "rotate", "angle", "rot", "orthogonal_strength")):
                angle_deg = float(granular_val.get("rotation_angle", granular_val.get("rotate", granular_val.get("angle", 0.0))))
                rot_mode = str(granular_val.get("rotation_mode", granular_val.get("mode", ROTATION_MODE_OUTPUT)))
                rank = int(granular_val.get("subspace_rank", granular_val.get("rank", 8)))
                seed = int(granular_val.get("seed", 42))
                hue = float(granular_val.get("hue", granular_val.get("style_direction", 180.0)))
                depth = granular_val.get("depth_reach", granular_val.get("depth", "Custom"))
                scale_mult = float(granular_val.get("scale", granular_val.get("multiplier", 1.0)))

                if fast_style_compass_rotation is None:
                    logger.warning("[Arthemy Granular] Geometry engine unavailable; rotation entry ignored.")
                    return None

                delta_obj = fast_style_compass_rotation(
                    base_weight, angle_deg=angle_deg, hue=hue, depth=depth,
                    seed=seed, rank=rank, mode=rot_mode)
                if delta_obj is None:
                    logger.warning("[Arthemy Granular] Rotation skipped for a tensor "
                                   f"(shape {tuple(base_weight.shape)}, angle {angle_deg}).")
                    return None

                if scale_mult == 1.0:
                    # Native LoRAAdapter: exact, low-rank, applied by ComfyUI without
                    # materializing a full-size delta tensor.
                    patch_value, _meta = build_geometry_patch(delta_obj, "")
                    return (patch_value,) if patch_value is not None else None

                # A multiplicative scale cannot live inside a low-rank factorization, so the
                # rotation is materialized once and combined algebraically:
                # W_new = scale * (W + delta) -> delta_tot = (scale - 1) * W + scale * delta
                dense_delta = rotation_to_dense_delta(delta_obj, base_weight)
                if dense_delta is None:
                    logger.warning("[Arthemy Granular] Could not materialize rotation delta; "
                                   "applying rotation without the scale multiplier.")
                    patch_value, _meta = build_geometry_patch(delta_obj, "")
                    return (patch_value,) if patch_value is not None else None
                combined = ((scale_mult - 1.0) * base_weight.detach().to(torch.float32)
                            + scale_mult * dense_delta).to(torch.bfloat16)
                setattr(combined, "_is_arthemy_granular", True)
                setattr(combined, "_is_arthemy_rotation", True)
                setattr(combined, "_arthemy_rotation_angle", angle_deg)
                setattr(combined, "_arthemy_rotation_hue", hue)
                return (combined,)

            diff = torch.zeros(base_weight.shape, dtype=torch.bfloat16, device="cpu")
            flat_diff = diff.flatten()
            channels_dict = granular_val.get("channels", granular_val)
            applied = False
            for ch_k, ch_v in channels_dict.items():
                if str(ch_k).isdigit():
                    ch_idx = int(ch_k)
                    if 0 <= ch_idx < flat_diff.numel():
                        flat_diff[ch_idx] = float(ch_v)
                        applied = True
            if applied:
                setattr(diff, "_is_arthemy_granular", True)
                return (diff,)
            return None
        except Exception as e:
            logger.warning(f"[Arthemy Granular] Failed to parse dict for granular patch: {e}")
            return None

    # Case 3: Scalar (int or float) -> standard scalar multiplier
    elif isinstance(granular_val, (int, float)):
        raw_val = float(granular_val)
        target_w = soft_target_weight(raw_val, mode)
        strength = target_w - 1.0
        if strength != 0:
            return (1.0 + strength,)
        return None

    return None

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
        chaos_params: dict = None,
        base_sd: dict = None
    ):
        patcher = get_patcher(model_or_clip)
        clone_obj = model_or_clip.clone()
        if base_sd is None:
            base_sd = patcher.model.state_dict() if hasattr(patcher, "model") else {}
        valid_keys = getattr(patcher, "model_keys", None)

        granular_map = parse_granular_json(granular_json)

        slider_map = {w_key: kwargs.get(w_key, 0.0) for w_key in surgeon_map.keys()}

        # vectors_override: comma-separated list mapped positionally onto the sub-tensor sliders.
        # (Previously accepted by the signature and silently ignored.)
        if vectors_override and vectors_override.strip():
            try:
                v_vals = [float(v.strip()) for v in vectors_override.split(",") if v.strip()]
                if len(v_vals) == len(surgeon_map):
                    slider_map = {w_key: v_vals[i] for i, w_key in enumerate(surgeon_map.keys())}
                    logger.info(f"[Arthemy Surgeon] vectors_override applied to {len(v_vals)} sub-tensor sliders.")
                else:
                    logger.warning(f"[Arthemy Surgeon] vectors_override dimension mismatch: expected "
                                   f"{len(surgeon_map)} values, got {len(v_vals)}. Override ignored.")
            except ValueError as e:
                logger.warning(f"[Arthemy Surgeon] vectors_override parse error: {e}. Override ignored.")

        patches_to_add = {}
        active_count = 0
        quant_skipped = 0
        # Granular hits that fall outside the current block/layer selection. Honouring them is
        # the point of the change above, but it is worth saying out loud: a partial selector
        # such as "attn.wq" now reaches every block, not just the selected ones.
        granular_outside_selection: List[str] = []

        for k, base_weight in base_sd.items():
            if is_skippable_for_tuning(k, base_sd):
                if not is_bookkeeping_sd_key(k):
                    quant_skipped += 1
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

            in_selection = (idx is not None and idx in selected_indices and bool(matched_widget_key))
            target_patch_key = resolve_target_key(patcher, k, model_sd=base_sd)

            # GRANULAR FIRST, and independent of the block/layer dropdown.
            #
            # This check used to sit AFTER the selection gate below, which meant a granular
            # selector could only ever reach a tensor that (a) lives inside a numbered block,
            # (b) whose block is currently selected, and (c) matches one of this node's
            # sub-tensor sliders. A perfectly valid instruction such as
            #     {"txtfusion.projector.weight": {"8": -0.5, "9": -0.8}}
            # names a tensor with no block index at all, so it was dropped before the granular
            # map was even consulted - it worked in the Model/CLIP Tuner, which scans the whole
            # state dict, and silently did nothing here. Same JSON, same node family, two
            # different outcomes depending on where you pasted it.
            #
            # The rule is now one sentence in both nodes: granular_json is an explicit
            # by-name instruction and is honoured on every tensor it matches; the dropdown
            # governs the sliders only.
            if granular_map:
                granular_entry = resolve_granular_entry(clean_k, granular_map, None)
                if granular_entry is not None:
                    slider_default = slider_map.get(matched_widget_key, 0.0) if in_selection else 0.0
                    patch_val = build_granular_patch(granular_entry, base_weight, mode=mode,
                                                     default_scalar=slider_default)
                    if patch_val is not None:
                        patches_to_add[target_patch_key] = patch_val
                        active_count += 1
                        if not in_selection:
                            granular_outside_selection.append(clean_k)
                    continue

            if not in_selection:
                continue

            if chaos_params is not None:
                chance_val = float(kwargs.get(matched_widget_key, 0.0))
                if chance_val <= 0.0:
                    continue

                seed = chaos_params.get("seed", 42)
                chaos_strength = float(chaos_params.get("chaos_strength", 0.1))
                tune_mode = chaos_params.get("tune_mode", "Block-Level")

                fast_seed = generate_fast_seed(k, seed)
                # Always roll on CPU: CUDA and CPU generators produce different sequences for the
                # same seed, which would break the determinism promised by chaos presets.
                rng = torch.Generator(device="cpu")
                rng.manual_seed(fast_seed)

                if tune_mode == "Element-Level (Sub-atomic)":
                    mask = torch.rand(base_weight.shape, generator=rng, dtype=torch.float32) < chance_val
                    if not bool(torch.any(mask)):
                        continue
                    w_active = ComfyPatcherAdapter.calculate_safe_weight(
                        model_or_clip, target_patch_key, base_weight, model_sd=base_sd)
                    delta = ((w_active.detach().to("cpu", torch.float32) * chaos_strength) * mask).to(torch.bfloat16)
                    setattr(delta, "_is_arthemy_chaos", True)
                    patches_to_add[target_patch_key] = (delta,)
                    active_count += 1
                else:
                    # Block-Level: the whole tensor is either perturbed or untouched, which is
                    # exactly a scalar multiplier. Emitting a scalar instead of a dense
                    # full-size bf16 delta saves hundreds of MB of host RAM.
                    v_rand = torch.rand(1, generator=rng, dtype=torch.float32).item()
                    if v_rand >= chance_val:
                        continue
                    patches_to_add[target_patch_key] = (1.0 + chaos_strength,)
                    active_count += 1
            else:
                raw_target = slider_map.get(matched_widget_key, 0.0)
                if granular_map:
                    granular_entry = resolve_granular_entry(clean_k, granular_map, None)
                    if granular_entry is not None:
                        patch_val = build_granular_patch(granular_entry, base_weight, mode=mode, default_scalar=raw_target)
                        if patch_val is not None:
                            patches_to_add[target_patch_key] = patch_val
                            active_count += 1
                        continue

                target_w = soft_target_weight(raw_target, mode)
                strength = target_w - 1.0
                if strength != 0:
                    patches_to_add[target_patch_key] = (1.0 + strength,)
                    active_count += 1

        warn_if_quantized_skipped("Arthemy Surgeon", quant_skipped)

        if granular_outside_selection:
            shown = ", ".join(granular_outside_selection[:5])
            more = f" (+{len(granular_outside_selection) - 5} more)" if len(granular_outside_selection) > 5 else ""
            logger.info(
                f"[Arthemy Surgeon] granular_json also matched {len(granular_outside_selection)} tensor(s) "
                f"outside the current selection: {shown}{more}. Granular entries are explicit by-name "
                "instructions and apply wherever they match - exactly as they do in the Model/CLIP Tuner - "
                "while the dropdown keeps governing the sliders.")

        if patches_to_add:
            t_start = time.time()
            applied = inject_patches(clone_obj, patches_to_add, 1.0)
            active_count = len(applied)

            # Record the deterministic Chaos recipe for preset persistence
            if chaos_params is not None:
                append_recipe(get_patcher(clone_obj), "arthemy_chaos_recipes", {
                    "domain": "clip" if is_clip else "model",
                    "selected_indices": sorted(list(selected_indices)),
                    "tune_mode": chaos_params.get("tune_mode", "Block-Level"),
                    "seed": chaos_params.get("seed", 42),
                    "chaos_strength": chaos_params.get("chaos_strength", 0.1),
                    "chances": {w_key: float(kwargs.get(w_key, 0.0)) for w_key in surgeon_map.keys()},
                })

            logger.info(f"[Arthemy Profiler] {'CLIP' if is_clip else 'Model'} Surgeon | "
                        f"{time.time()-t_start:.4f}s | {active_count} patches | selected: {sorted(selected_indices)}")

        desc_label = f"{'Layers' if is_clip else 'Blocks'} {sorted(selected_indices)}"
        return clone_obj, f"Arthemy Krea-2 {'CLIP' if is_clip else 'Model'} Surgeon ({desc_label}) | Patches: {active_count}"

# ==============================================================================
# POINT 3: MEMORY-SAFE GENERATOR & STREAMING PROCESSING
# ==============================================================================
def process_tensor_stream(
    tensor_dict: Dict[str, Any],
    patch_fn: Callable[[str, Any], Optional[Any]],
    target_dtype: torch.dtype = torch.bfloat16,
    convert_dtype: bool = True,
    memory_threshold_mb: int = Krea2Config.DEFAULT_GC_THRESHOLD_MB,
    emit_quant_metadata: bool = False,
    clean_keys: bool = True,
) -> Generator[Tuple[str, Optional[Any]], None, None]:
    """Memory-safe generator that yields processed tensors one by one.

    Triggers GC and clears the CUDA cache only when processed data exceeds threshold_bytes.

    Filtering rules:
      * Companion scale tensors (``*_scale`` / ``*.weight_scale`` / ``*.scale_weight``) are dropped: their value is
        folded into the corresponding weight by ``dequantize_weight``. Emitting them would write
        scale factors into a checkpoint as if they were weights.
      * The quantized weights themselves are ALWAYS emitted (the previous implementation skipped
        every tensor that had a companion scale, producing checkpoints without any weights).
      * ``.comfy_quant`` metadata blobs are only forwarded to ``patch_fn`` when
        ``emit_quant_metadata`` is True, so the Model Saver can actually collect them.
      * ``convert_dtype=False`` skips the bf16/CPU conversion for consumers (e.g. the Baker)
        that only need the callback side effects.
    """
    threshold_bytes = memory_threshold_mb * 1024 * 1024
    accumulated_bytes = 0

    for k, v in tensor_dict.items():
        if (k.endswith("_scale") or k.endswith(".weight_scale")
                or k.endswith(".scale_weight") or k.endswith("_scale_weight")):
            continue
        if k.endswith(".comfy_quant") and not emit_quant_metadata:
            continue

        out_k = Krea2TensorParser.clean_key(k) if clean_keys else k

        if isinstance(v, torch.Tensor):
            accumulated_bytes += v.element_size() * v.nelement()

        processed = patch_fn(k, v)

        if isinstance(processed, torch.Tensor) and convert_dtype:
            if processed.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
                out_tensor = processed.to(device="cpu", copy=False).contiguous()
            else:
                actual_dtype = target_dtype
                if target_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    # FP8 max is 448; keep 1D tensors, norms, bias and modulators in BF16 to prevent overflow and NaN
                    if processed.ndim <= 1 or k.endswith(".bias") or "norm" in k or "mod" in k:
                        actual_dtype = torch.bfloat16
                out_tensor = processed.to(dtype=actual_dtype, device="cpu", copy=False).contiguous()
            del processed
        else:
            out_tensor = processed

        yield out_k, out_tensor

        if accumulated_bytes >= threshold_bytes:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            accumulated_bytes = 0


def companion_scale_candidates(key: str) -> Tuple[str, ...]:
    """Every spelling a scaled-FP8 companion scale can have for `key`.

    Two families, and they are shaped differently - which is the trap:

      * suffixes ON the weight key: `<...>.weight_scale`, `<...>_scale`. ComfyUI's loader
        rewrites a checkpoint's scales into `<layer>.weight_scale`, so `f"{key}_scale"`
        already covers the common case.
      * a SIBLING of the weight: `<layer>.scale_weight`, which is how the scale is spelled
        in a `_fp8_scaled` file on disk (comfy/utils.py rewrites it on load). Appending
        `.scale_weight` to the weight key produces `<layer>.weight.scale_weight`, which
        exists nowhere - the layer prefix has to be taken first.
    """
    cands = [f"{key}_scale", f"{key}.weight_scale"]
    if key.endswith(".weight"):
        layer = key[: -len(".weight")]
        cands += [f"{layer}.scale_weight", f"{layer}.weight_scale"]
    return tuple(cands)


def find_companion_scale(key: str, state_dict: Dict[str, Any]) -> Optional[Any]:
    """Returns the FP8 companion scale tensor for `key`, if the checkpoint carries one."""
    clean_k = Krea2TensorParser.clean_key(key)
    for cand in companion_scale_candidates(key) + companion_scale_candidates(clean_k):
        if cand in state_dict:
            return state_dict[cand]
    return None


def resolve_save_path(output_checkpoint: str) -> Tuple[str, str]:
    """Shared output-path resolution for the Model/CLIP savers. Returns (full_path, file_name)."""
    clean_name = output_checkpoint.strip()
    if clean_name.endswith(".safetensors"):
        clean_name = clean_name[:-len(".safetensors")]

    full_output_folder, filename, _counter, _subfolder, _prefix = folder_paths.get_save_image_path(
        clean_name, folder_paths.get_output_directory(), 0, 0
    )
    os.makedirs(full_output_folder, exist_ok=True)

    base_fn = os.path.basename(filename)
    # Check existing .safetensors files in destination directory to correctly increment counter
    cnt = 1
    try:
        existing = [f for f in os.listdir(full_output_folder) if f.startswith(base_fn) and f.endswith(".safetensors")]
        cnt = len(existing) + 1
        for ef in existing:
            digits = re.findall(r"\d+", ef)
            if digits:
                cnt = max(cnt, int(digits[-1]) + 1)
    except Exception:
        cnt = 1

    if "%count%" in clean_name:
        file_name_out = base_fn.replace("%count%", f"{cnt:05d}") + ".safetensors"
    else:
        file_name_out = f"{base_fn}_{cnt:05d}_.safetensors"

    return os.path.join(full_output_folder, file_name_out), file_name_out


# ==============================================================================
# 1. ARTHEMY KREA2 MODEL TUNER
# ==============================================================================
class ArthemyKrea2ModelTuner(BaseKrea2Node):
    """Tier 1: Macro Block-Level diffusion model weight scaling, vector overrides, and granular tuning."""
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

    # Vector layout: 28 transformer blocks + 2 txtfusion layerwise + 2 txtfusion refiner
    # + 1 txtfusion projector + 1 txtmlp = 34 slots.
    VECTOR_LEN = 34
    VEC_TXTFUSION_LAYERWISE_BASE = 28
    VEC_TXTFUSION_LAYERWISE_COUNT = 2
    VEC_TXTFUSION_REFINER_BASE = 30
    VEC_TXTFUSION_REFINER_COUNT = 2
    VEC_TXTFUSION_PROJECTOR = 32
    VEC_TXTMLP = 33

    def tune_model(self, model: Any, mode: str = "Soft Value", vectors_override: str = "", granular_json: str = "", **kwargs: float) -> Tuple[Any, str]:
        def get_target_weight(delta): return soft_target_weight(delta, mode)
        final_weights = [1.0] * self.VECTOR_LEN
        use_vector = False

        if vectors_override.strip():
            try:
                v_vals = [float(v.strip()) for v in vectors_override.split(',') if v.strip()]
                if len(v_vals) == self.VECTOR_LEN:
                    final_weights = [get_target_weight(v) for v in v_vals]
                    use_vector = True
                else:
                    logger.warning(f"[Arthemy Model Tuner] vectors_override dimension mismatch: expected "
                                   f"{self.VECTOR_LEN} values, got {len(v_vals)}. Override ignored.")
            except ValueError as e:
                logger.warning(f"[Arthemy Model Tuner] vectors_override parse error: {e}. Override ignored.")

        granular_map = parse_granular_json(granular_json)

        w_base = get_target_weight(0.0)
        m = model.clone()
        base_sd = m.model.state_dict()
        patches_to_add = {}
        active_patches = 0
        valid_keys = getattr(m, "model_keys", None)
        quant_skipped = 0
        t_start = time.time()

        for k, base_weight in base_sd.items():
            if is_skippable_for_tuning(k, base_sd):
                if not is_bookkeeping_sd_key(k):
                    quant_skipped += 1
                continue
            if valid_keys is not None and k not in valid_keys:
                continue

            clean_key = Krea2TensorParser.clean_key(k)
            patch_key = f"diffusion_model.{clean_key}" if not clean_key.startswith("diffusion_model.") else clean_key

            if granular_map:
                granular_entry = resolve_granular_entry(clean_key, granular_map, None)
                if granular_entry is not None:
                    patch_val = build_granular_patch(granular_entry, base_weight, mode=mode)
                    if patch_val is not None:
                        patches_to_add[patch_key] = patch_val
                        active_patches += 1
                    continue

            target_weight = w_base
            if use_vector:
                if "txtfusion.layerwise_blocks" in clean_key:
                    match = RE_TXTFUSION_LAYERWISE.search(clean_key)
                    if match:
                        sub_i = int(match.group(1))
                        if 0 <= sub_i < self.VEC_TXTFUSION_LAYERWISE_COUNT:
                            target_weight = final_weights[self.VEC_TXTFUSION_LAYERWISE_BASE + sub_i]
                        else:
                            logger.warning(f"[Arthemy Model Tuner] txtfusion.layerwise_blocks.{sub_i} exceeds the "
                                           f"{self.VEC_TXTFUSION_LAYERWISE_COUNT}-slot vector layout; left unscaled.")
                elif "txtfusion.refiner_blocks" in clean_key:
                    match = RE_TXTFUSION_REFINER.search(clean_key)
                    if match:
                        sub_i = int(match.group(1))
                        if 0 <= sub_i < self.VEC_TXTFUSION_REFINER_COUNT:
                            target_weight = final_weights[self.VEC_TXTFUSION_REFINER_BASE + sub_i]
                        else:
                            logger.warning(f"[Arthemy Model Tuner] txtfusion.refiner_blocks.{sub_i} exceeds the "
                                           f"{self.VEC_TXTFUSION_REFINER_COUNT}-slot vector layout; left unscaled.")
                elif "txtfusion.projector" in clean_key:
                    target_weight = final_weights[self.VEC_TXTFUSION_PROJECTOR]
                elif "txtmlp" in clean_key:
                    target_weight = final_weights[self.VEC_TXTMLP]
                else:
                    match = RE_GENERAL_BLOCKS.search(clean_key)
                    if match:
                        blk_i = int(match.group(1))
                        if 0 <= blk_i < Krea2Config.MAX_UNET_BLOCKS:
                            target_weight = final_weights[blk_i]
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
                patches_to_add[patch_key] = (1.0 + strength,)
                active_patches += 1

        warn_if_quantized_skipped("Arthemy Model Tuner", quant_skipped)

        if patches_to_add:
            active_patches = len(inject_patches(m, patches_to_add, 1.0))
            logger.info(f"[Arthemy Profiler] Model Tuner finished | Total time: {time.time() - t_start:.4f}s | "
                        f"Layers patched: {active_patches}")
        return (m, f"Krea-2 Model Tuned | Patches: {active_patches}")


# ==============================================================================
# 2. ARTHEMY KREA2 CLIP TUNER
# ==============================================================================
class ArthemyKrea2CLIPTuner(BaseKrea2Node):
    """Tier 1: Macro Layer-Level CLIP text encoder weight scaling, vector overrides, and granular tuning."""
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

    VECTOR_LEN = 8

    def tune_clip(self, clip: Any, mode: str = "Soft Value", vectors_override: str = "", granular_json: str = "",
                  Embedding: float = 0.0, Layer_1: float = 0.0, Layer_2: float = 0.0, Layer_3: float = 0.0,
                  Layer_4: float = 0.0, Layer_5: float = 0.0, Layer_6: float = 0.0, Layer_7: float = 0.0, **kwargs: float) -> Tuple[Any, str]:
        def get_target_weight(delta): return soft_target_weight(delta, mode)
        final_weights = [1.0] * self.VECTOR_LEN
        use_vector = False

        if vectors_override.strip():
            try:
                v_vals = [float(v.strip()) for v in vectors_override.split(',') if v.strip()]
                if len(v_vals) == self.VECTOR_LEN:
                    final_weights = [get_target_weight(v) for v in v_vals]
                    use_vector = True
                else:
                    logger.warning(f"[Arthemy CLIP Tuner] vectors_override dimension mismatch: expected "
                                   f"{self.VECTOR_LEN} values, got {len(v_vals)}. Override ignored.")
            except ValueError as e:
                logger.warning(f"[Arthemy CLIP Tuner] vectors_override parse error: {e}. Override ignored.")

        granular_map = parse_granular_json(granular_json)

        sliders = [Embedding, Layer_1, Layer_2, Layer_3, Layer_4, Layer_5, Layer_6, Layer_7]
        weights = final_weights if use_vector else [get_target_weight(s) for s in sliders]
        w_vocab = weights[0]

        c = clip.clone()
        c_p = get_patcher(c)
        base_sd = c_p.model.state_dict() if hasattr(c_p, 'model') else getattr(c, 'get_sd', lambda: {})()
        patches_to_add = {}
        active_patches = 0
        valid_keys = getattr(c_p, 'model_keys', None)
        quant_skipped = 0
        t_start = time.time()

        def get_block_weight(idx):
            # Layer groups: 0-4 (L1), 5-9 (L2), 10-14 (L3), 15-19 (L4), 20-24 (L5), 25-29 (L6), 30-35 (L7)
            # Indices 36-59 belong to the visual tower and are not affected by language layer sliders
            if idx > 35:
                return 1.0
            group = min(6, idx // 5)
            return weights[1 + group]

        for k, base_weight in base_sd.items():
            if is_skippable_for_tuning(k, base_sd):
                if not is_bookkeeping_sd_key(k):
                    quant_skipped += 1
                continue
            if valid_keys is not None and k not in valid_keys:
                continue

            clean_k = Krea2TensorParser.clean_key(k)
            if granular_map:
                granular_entry = resolve_granular_entry(clean_k, granular_map, None)
                if granular_entry is None:
                    granular_entry = resolve_granular_entry(k, granular_map, None)
                if granular_entry is not None:
                    patch_val = build_granular_patch(granular_entry, base_weight, mode=mode)
                    if patch_val is not None:
                        patches_to_add[k] = patch_val
                        active_patches += 1
                    continue

            target_scale = 1.0
            if "embed_tokens" in k:
                target_scale = w_vocab
            else:
                idx, _ = Krea2TensorParser.extract_clip_layer_idx(clean_k)
                if idx is not None:
                    target_scale = get_block_weight(idx)

            strength = target_scale - 1.0
            if strength != 0:
                patches_to_add[k] = (1.0 + strength,)
                active_patches += 1

        warn_if_quantized_skipped("Arthemy CLIP Tuner", quant_skipped)

        if patches_to_add:
            active_patches = len(inject_patches(c, patches_to_add, 1.0))
            logger.info(f"[Arthemy Profiler] CLIP Tuner finished | Total time: {time.time() - t_start:.4f}s | "
                        f"Layers patched: {active_patches}")
        return (c, f"Krea-2 CLIP Tuned | Patches: {active_patches}")


# ==============================================================================
# POINT 4 IMPLEMENTATIONS: SURGEON NODES DERIVED FROM BaseSurgeonTuner
# ==============================================================================
class ArthemyKrea2ModelBlockSurgeonTuner(BaseSurgeonTuner):
    """Tier 2: High-precision deterministic Sub-Block diffusion model component tuning.
    Allows adjusting specific internal components (attention heads, MLP SwiGLU, norms) for individual blocks."""
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

    def tune_model_surgeon(self, model: Any, target_block: str = "Block_1 (All 0-4)", mode: str = "Soft Value", vectors_override: str = "", granular_json: str = "", **kwargs: float) -> Tuple[Any, str]:
        selected = resolve_target_map_entry(self.MODEL_TARGET_MAP, target_block,
                                           set(range(0, Krea2Config.MAX_UNET_BLOCKS)))
        return self._execute_surgeon_tuning(
            model, is_clip=False, selected_indices=selected, surgeon_map=self.SURGEON_MAP,
            kwargs=kwargs, mode=mode, vectors_override=vectors_override, granular_json=granular_json
        )


class ArthemyKrea2CLIPBlockSurgeonTuner(BaseSurgeonTuner):
    """Tier 2: High-precision deterministic Sub-Block CLIP text encoder component tuning.
    Allows adjusting internal components (self-attention projections, MLP layers, layernorms) for individual layers."""
    CLIP_TARGET_MAP = {
        "All Layers (0-59)": set(range(0, 60)),
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
        "Visual_1 (All 36-41)": set(range(36, 42)),
        "  ↳ Visual_1A (36)": {36},
        "  ↳ Visual_1B (37)": {37},
        "  ↳ Visual_1C (38)": {38},
        "  ↳ Visual_1D (39)": {39},
        "  ↳ Visual_1E (40)": {40},
        "  ↳ Visual_1F (41)": {41},
        "Visual_2 (All 42-47)": set(range(42, 48)),
        "  ↳ Visual_2A (42)": {42},
        "  ↳ Visual_2B (43)": {43},
        "  ↳ Visual_2C (44)": {44},
        "  ↳ Visual_2D (45)": {45},
        "  ↳ Visual_2E (46)": {46},
        "  ↳ Visual_2F (47)": {47},
        "Visual_3 (All 48-53)": set(range(48, 54)),
        "  ↳ Visual_3A (48)": {48},
        "  ↳ Visual_3B (49)": {49},
        "  ↳ Visual_3C (50)": {50},
        "  ↳ Visual_3D (51)": {51},
        "  ↳ Visual_3E (52)": {52},
        "  ↳ Visual_3F (53)": {53},
        "Visual_4 (All 54-59)": set(range(54, 60)),
        "  ↳ Visual_4A (54)": {54},
        "  ↳ Visual_4B (55)": {55},
        "  ↳ Visual_4C (56)": {56},
        "  ↳ Visual_4D (57)": {57},
        "  ↳ Visual_4E (58)": {58},
        "  ↳ Visual_4F (59)": {59},
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

    def tune_clip_surgeon(self, clip: Any, target_layer: str = "Layer_1 (All 0-4)", mode: str = "Soft Value", vectors_override: str = "", granular_json: str = "", **kwargs: float) -> Tuple[Any, str]:
        selected = resolve_target_map_entry(self.CLIP_TARGET_MAP, target_layer,
                                            set(range(0, Krea2Config.MAX_CLIP_LAYERS)))
        return self._execute_surgeon_tuning(
            clip, is_clip=True, selected_indices=selected, surgeon_map=self.SURGEON_MAP,
            kwargs=kwargs, mode=mode, vectors_override=vectors_override, granular_json=granular_json
        )


class ArthemyKrea2ModelChaosBlockSurgeonTuner(BaseSurgeonTuner):
    """Tier 3: Stochastic Sub-Block Chaos diffusion model tuning.
    Randomly perturbs component subsets based on individual roll probabilities."""
    MODEL_TARGET_MAP = ArthemyKrea2ModelBlockSurgeonTuner.MODEL_TARGET_MAP

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

    def chaos_tune_model_surgeon(self, model: Any, target_block: str = "Block_1 (All 0-4)", tune_mode: str = "Block-Level", seed: int = 42, chaos_strength: float = 0.1, **kwargs: float) -> Tuple[Any, str]:
        selected = resolve_target_map_entry(self.MODEL_TARGET_MAP, target_block,
                                           set(range(0, Krea2Config.MAX_UNET_BLOCKS)))

        # Remap kwargs chances to standard surgeon_map names
        remapped_kwargs = {k.replace("_chance", ""): v for k, v in kwargs.items() if k.endswith("_chance")}

        chaos_params = {"tune_mode": tune_mode, "seed": seed, "chaos_strength": chaos_strength}
        return self._execute_surgeon_tuning(
            model, is_clip=False, selected_indices=selected, surgeon_map=Krea2TensorParser.MODEL_SURGEON_MAP,
            kwargs=remapped_kwargs, chaos_params=chaos_params
        )


class ArthemyKrea2CLIPChaosBlockSurgeonTuner(BaseSurgeonTuner):
    """Tier 3: Stochastic Sub-Block Chaos CLIP text encoder tuning.
    Randomly perturbs component subsets based on individual roll probabilities."""
    CLIP_TARGET_MAP = ArthemyKrea2CLIPBlockSurgeonTuner.CLIP_TARGET_MAP

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "clip": ("CLIP",),
                "target_layer": (list(s.CLIP_TARGET_MAP.keys()), {"default": "Layer_1 (All 0-4)", "tooltip": "Target CLIP layer group or individual layer."}),
                "tune_mode": (["Block-Level", "Element-Level (Sub-atomic)"], {"tooltip": "Chaos distribution mode across layers."}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Deterministic generator seed for chaos roll dice."}),
                "chaos_strength": ("FLOAT", {"default": 0.1, "min": -99.00, "max": 99.00, "step": 0.01, "tooltip": "Base multiplier perturbation magnitude."}),
            }
        }
        for name in Krea2TensorParser.CLIP_SURGEON_MAP.keys():
            inputs["required"][f"{name}_chance"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": f"Probability chance (0.00-1.00) to perturb {name}."})
        return inputs

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("CLIP", "info")
    FUNCTION = "chaos_tune_clip_surgeon"
    CATEGORY = "Arthemy/Krea2 Tuners"

    def chaos_tune_clip_surgeon(self, clip: Any, target_layer: str = "Layer_1 (All 0-4)", tune_mode: str = "Block-Level", seed: int = 42, chaos_strength: float = 0.1, **kwargs: float) -> Tuple[Any, str]:
        selected = resolve_target_map_entry(self.CLIP_TARGET_MAP, target_layer,
                                            set(range(0, Krea2Config.MAX_CLIP_LAYERS)))

        remapped_kwargs = {k.replace("_chance", ""): v for k, v in kwargs.items() if k.endswith("_chance")}
        chaos_params = {"tune_mode": tune_mode, "seed": seed, "chaos_strength": chaos_strength}
        return self._execute_surgeon_tuning(
            clip, is_clip=True, selected_indices=selected, surgeon_map=Krea2TensorParser.CLIP_SURGEON_MAP,
            kwargs=remapped_kwargs, chaos_params=chaos_params
        )

# ==============================================================================
# POINT 4B: HYPER-DIMENSIONAL LATENT SPACE ROTATOR & ORTHOGONAL SURGEON
# ==============================================================================

class RotationReport:
    """Outcome of one rotator pass, so nodes can report *measured* effect, not intent."""

    __slots__ = ("patched", "skipped_1d", "quant_skipped", "mean_angle", "mean_relative_delta", "per_index")

    def __init__(self):
        self.patched = 0
        self.skipped_1d = 0
        self.quant_skipped = 0
        self.mean_angle = 0.0
        self.mean_relative_delta = 0.0
        self.per_index: Dict[int, Dict[str, Any]] = {}

    def summary(self) -> str:
        if not self.patched:
            reason = ""
            if self.quant_skipped:
                reason = " (quantized checkpoint: nothing rotatable)"
            elif self.skipped_1d:
                reason = " (only 1-D norm/scale tensors matched: nothing to rotate)"
            return f"Rotated 0 layers{reason}"
        return (f"Rotated {self.patched} layers | avg {self.mean_angle:.1f}deg | "
                f"weight change {self.mean_relative_delta * 100:.2f}%")


def apply_rotation_over_state_dict(clone_obj: Any, is_clip: bool, selected_indices: set,
                                   sub_components: str, adapter_factory,
                                   base_sd: Optional[Dict[str, Any]] = None) -> RotationReport:
    """Shared driver for every rotator node.

    Iterates the state_dict once, filters by block/layer index and sub-component group, builds a
    patch through `adapter_factory(base_weight, clean_key)` and injects everything through
    inject_patches so that:
      * keys are resolved against the real state_dict (wrapped text encoders included);
      * the reported count reflects the patches actually accepted by the patcher;
      * the geometry engine's `LowRankDelta` becomes a native ComfyUI `LoRAAdapter`
        (the only low-rank format modern ComfyUI still applies);
      * per-section provenance (angle, measured relative weight change) is recorded in
        model_options, which is what the visualizers read.

    Returns a `RotationReport` instead of a bare count so a no-op is always explainable.
    """
    report = RotationReport()
    patcher = get_patcher(clone_obj)
    if base_sd is None:
        base_sd = patcher.model.state_dict() if hasattr(patcher, "model") else {}
    patches_to_add: Dict[str, Any] = {}
    angles: List[float] = []
    deltas: List[float] = []

    for k, base_weight in base_sd.items():
        if is_skippable_for_tuning(k, base_sd):
            if not is_bookkeeping_sd_key(k):
                report.quant_skipped += 1
            continue
        if not isinstance(base_weight, torch.Tensor):
            continue

        clean_k = Krea2TensorParser.clean_key(k)
        if is_clip:
            idx, _ = Krea2TensorParser.extract_clip_layer_idx(clean_k)
        else:
            idx, _ = Krea2TensorParser.extract_model_block_idx(clean_k)
        if idx is None or idx not in selected_indices:
            continue

        if not match_sub_component(clean_k, sub_components, is_clip):
            continue

        # An orthogonal rotation of a 1-D space is +/-1: norm scales and biases have no
        # plane to rotate. Counting them explains an otherwise mysterious "0 layers".
        if base_weight.ndim < 2:
            report.skipped_1d += 1
            continue

        payload, meta = build_geometry_patch(adapter_factory(base_weight, clean_k), clean_k)
        if payload is None:
            continue

        slot = report.per_index.setdefault(idx, {
            "is_rotation": True, "angle": 0.0, "relative_delta": 0.0,
            "hue": 180.0,
            "rotation_type": "dual_lie",
            "depth_reach": "Default",
            "tensors": 0,
        })
        slot["tensors"] += 1

        target_patch_key = resolve_target_key(patcher, k, model_sd=base_sd)
        patches_to_add[target_patch_key] = payload

        angle = float(meta.get("angle", 0.0))
        rel = float(meta.get("relative_delta", 0.0))
        angles.append(angle)
        deltas.append(rel)

        slot["angle"] = angle          # per-section angle is uniform within one node pass
        slot["relative_delta"] = max(slot["relative_delta"], rel)
        slot["hue"] = float(meta.get("hue", 180.0))
        slot["rotation_type"] = meta.get("rotation_type", "dual_lie")
        slot["depth_reach"] = meta.get("depth_reach", "Default")
        if meta.get("is_chaos"):
            slot["is_chaos"] = True

    warn_if_quantized_skipped("Arthemy Rotator", report.quant_skipped)

    if patches_to_add:
        report.patched = len(inject_patches(clone_obj, patches_to_add, 1.0))
        report.mean_angle = sum(angles) / max(1, len(angles))
        report.mean_relative_delta = sum(deltas) / max(1, len(deltas))

    record_section_meta(patcher, "clip" if is_clip else "model", report.per_index)
    return report


def send_live_rotation_report(unique_id: Optional[str], domain: str, is_chaos: bool,
                              selected_indices: set, report: "RotationReport",
                              params: Optional[Dict[str, Any]] = None) -> None:
    """Pushes the just-measured RotationReport to the node's own 3D panel over a plain
    ComfyUI websocket message, instead of through the node's return value.

    The rotator's (obj, info) tuple is called directly by the Preset Loader (to replay a
    recipe) as well as by the test suite, so changing its shape to the usual dict{"ui":
    ..., "result": ...} convention - the way the HUD visualizer sends its own live data -
    would break every one of those call sites. A side-channel message has no such
    contract to protect: nothing else in this file reads it, only the JS panel does.

    `params` echoes the settings this measurement was actually taken with, keyed by widget
    name. The panel compares them against the node's current widgets to decide whether the
    measured tree still describes the node, which it cannot do reliably by snapshotting the
    widgets when the message lands: by then the user may have already moved a slider.
    """
    if PromptServer is None or PromptServer.instance is None or not unique_id:
        return
    try:
        per_index = [
            {
                "idx": idx,
                "angle": round(float(v.get("angle", 0.0)), 2),
                "relative_delta": round(float(v.get("relative_delta", 0.0)), 6),
                "hue": round(float(v.get("hue", 180.0)), 1),
                "is_chaos": bool(v.get("is_chaos", False)),
                "tensors": int(v.get("tensors", 0)),
            }
            for idx, v in sorted(report.per_index.items())
        ]
        PromptServer.instance.send_sync("arthemy.rotation_report", {
            "node_id": str(unique_id),
            "domain": domain,
            "is_chaos": bool(is_chaos),
            "selected_indices": sorted(selected_indices),
            "per_index": per_index,
            "mean_angle": round(report.mean_angle, 2),
            "mean_relative_delta": round(report.mean_relative_delta, 6),
            "patched": report.patched,
            "skipped_1d": report.skipped_1d,
            "quant_skipped": report.quant_skipped,
            "params": {k: v for k, v in (params or {}).items()
                       if isinstance(v, (int, float, str, bool))},
        })
    except Exception as e:
        logger.debug(f"[Arthemy Rotator] Could not send live rotation report: {e}")


MODEL_SUBCOMPONENTS = ["All Components", "ATTN (WQ/WK/WV/WO/Gate)", "MLP (Gate/Up/Down)", "NORMS & MOD (Scales/Modulations)"]
CLIP_SUBCOMPONENTS = ["All Components", "ATTN (q_proj/k_proj/v_proj/o_proj)", "MLP (gate_proj/up_proj/down_proj)", "NORMS (LayerNorm Scales)"]

# One step finer than the sub-component groups above: the surgeon map's individual
# tensors (WQ alone, MLP-down alone, ...). The 5D Tuners take this as an optional extra
# narrowing, which also decides how little a scoped preset has to carry.
SUB_TENSOR_ANY = "All Sub-Tensors"
MODEL_SUB_TENSORS = [SUB_TENSOR_ANY] + list(Krea2TensorParser.MODEL_SURGEON_MAP.keys())
CLIP_SUB_TENSORS = [SUB_TENSOR_ANY] + list(Krea2TensorParser.CLIP_SURGEON_MAP.keys())


def is_sub_tensor_narrowed(sub_tensor: Any) -> bool:
    """True when a sub-tensor selection actually restricts anything."""
    v = str(sub_tensor or SUB_TENSOR_ANY).strip()
    return bool(v) and not v.startswith("All")


def match_sub_tensor_label(sub_key: str, is_clip: bool) -> Optional[str]:
    """Surgeon-map label for a sub-key taken from a 5D modifier.

    The surgeon maps are written against state_dict keys, so most of their entries end in
    '.weight'; a modifier's layer names are LoRA bases and never do. Matching the bare
    sub-key AND its '.weight' form is what makes the CLIP map (whose every entry carries
    the suffix) match at all - without it, every CLIP sub-tensor choice would silently
    select nothing.
    """
    fn = (Krea2TensorParser.match_clip_sub_tensor if is_clip
          else Krea2TensorParser.match_model_sub_tensor)
    return fn(sub_key) or fn(f"{sub_key}.weight")

DEPTH_REACH_CHOICES = list(ROTATION_RANK_MAP.keys())


class BaseRotatorNode(BaseKrea2Node):
    """Shared plumbing for all six rotator nodes.

    Collapses what used to be six near-identical 50-line classes (with the Model and CLIP
    variants already drifting apart) into one driver: input schema, dropdown resolution,
    recipe persistence, section provenance and info formatting all live here.

    Subclasses declare IS_CLIP / TARGET_MAP / RECIPE_KEY / LOG_TAG and implement
    `_factory(**kwargs)` returning the geometry callable plus the recipe payload.
    """

    IS_CLIP = False
    TARGET_MAP: Dict[str, Any] = {}
    RECIPE_KEY = "arthemy_rotation_recipes"
    LOG_TAG = "ARTHEMY ROTATOR"
    RETURN_NAMES = ("MODEL", "info")

    @classmethod
    def _io_name(cls) -> str:
        return "clip" if cls.IS_CLIP else "model"

    @classmethod
    def _target_name(cls) -> str:
        return "target_layer" if cls.IS_CLIP else "target_block"

    @classmethod
    def _common_inputs(cls) -> Dict[str, Any]:
        io_type = "CLIP" if cls.IS_CLIP else "MODEL"
        default_target = "Layer_1 (All 0-4)" if cls.IS_CLIP else "Block_1 (All 0-4)"
        subs = CLIP_SUBCOMPONENTS if cls.IS_CLIP else MODEL_SUBCOMPONENTS
        return {
            cls._io_name(): (io_type,),
            cls._target_name(): (list(cls.TARGET_MAP.keys()), {
                "default": default_target,
                "tooltip": "Target group, or a single block/layer, to rotate."}),
            "sub_components": (subs, {
                "default": "All Components",
                "tooltip": "Component subgroup inside the selection. NORMS are 1-D and cannot be rotated."}),
            "depth_reach": (DEPTH_REACH_CHOICES, {
                "default": "Default",
                "tooltip": f"Rotated subspace rank: {', '.join(f'{k}={v}' for k, v in ROTATION_RANK_MAP.items())}. "
                           "Higher = the rotation reaches more of the layer's behaviour."}),
        }

    def _apply(self, obj: Any, target_label: str, sub_components: str,
               factory, recipe: Dict[str, Any], headline: str,
               unique_id: Optional[str] = None,
               live_params: Optional[Dict[str, Any]] = None) -> Tuple[Any, str]:
        if GEOMETRY_ENGINE_ERROR is not None:
            msg = f"{headline} | ERROR: geometry engine unavailable ({GEOMETRY_ENGINE_ERROR})"
            logger.error(f"[{self.LOG_TAG}] {msg}")
            return (obj, msg)
        if LORA_ADAPTER_CLS is None:
            msg = (f"{headline} | ERROR: this ComfyUI build has no comfy.weight_adapter.LoRAAdapter, "
                   "so low-rank rotations cannot be applied. Update ComfyUI.")
            logger.error(f"[{self.LOG_TAG}] {msg}")
            return (obj, msg)

        clone_obj = obj.clone()
        patcher = get_patcher(clone_obj)
        # state_dict() rebuilds an OrderedDict over every tensor in the model, so it is
        # built once here and threaded through instead of being called again in the driver.
        base_sd = patcher.model.state_dict() if hasattr(patcher, "model") else {}
        # The fallback bound comes from the checkpoint itself, not a hardcoded 28/36, so an
        # unrecognised dropdown label cannot silently address blocks that do not exist.
        selected = resolve_target_map_entry(
            self.TARGET_MAP, target_label,
            lambda: set(range(0, Krea2Config.probe(base_sd, self.IS_CLIP))))

        report = apply_rotation_over_state_dict(
            clone_obj, is_clip=self.IS_CLIP, selected_indices=selected,
            sub_components=sub_components, adapter_factory=factory, base_sd=base_sd)

        # What the 3D panel needs to tell "these measurements still describe the node" from
        # "the user has changed something since". Keys MUST be widget names, which is why
        # each mixin passes them explicitly rather than reusing its recipe: the recipe keys
        # are the persisted-preset schema and do not always match (dual_rotation stores
        # structural_x while the widget is called structural_rot_x).
        panel_params = dict(live_params or {})
        panel_params[self._target_name()] = target_label
        panel_params["sub_components"] = sub_components
        send_live_rotation_report(
            unique_id, domain="clip" if self.IS_CLIP else "model",
            is_chaos=(recipe.get("type") == "chaos_rotation"),
            selected_indices=selected, report=report, params=panel_params)

        if report.patched:
            payload = dict(recipe)
            payload["domain"] = "clip" if self.IS_CLIP else "model"
            payload["target_block"] = target_label
            payload["sub_components"] = sub_components
            # Representative magnitude, so the Preset Loader and both visualizers can read a
            # single comparable "how much was this rotated" number for every rotation family.
            payload.setdefault("rotation_angle", round(report.mean_angle, 2))
            payload["measured_delta"] = round(report.mean_relative_delta, 6)
            append_recipe(get_patcher(clone_obj), self.RECIPE_KEY, payload)

        info = f"{headline} | {report.summary()}"
        logger.info(f"[{self.LOG_TAG}] {info}")
        return (clone_obj, info)


class _StyleCompassMixin:
    """Continuous 2D polar style compass: one tilt angle + one direction (hue)."""

    @classmethod
    def INPUT_TYPES(s):
        inputs = s._common_inputs()
        inputs["style_direction"] = ("FLOAT", {
            "default": 180.0, "min": 0.0, "max": 360.0, "step": 1.0,
            "tooltip": "Direction of travel on the style compass (0-360 deg). Blends continuously "
                       "between local and long-range rotation planes."})
        inputs["rotation_angle"] = ("FLOAT", {
            "default": 15.0, "min": -90.0, "max": 90.0, "step": 0.5,
            "tooltip": "Distance travelled from the base model (0 deg = base, 15 deg = gentle, "
                       "45 deg = distinct, 90 deg = maximum). Negative travels the same "
                       "distance the opposite way along the chosen direction."})
        inputs["seed"] = ("INT", {
            "default": 42, "min": 0, "max": 0xffffffffffffffff,
            "tooltip": "Reproducible style flavour: permutes the rotation planes identically "
                       "in every layer, so the style stays coherent across blocks."})
        inputs["manifold"] = ([_geom_ROTATION_MODE_OUTPUT, _geom_ROTATION_MODE_INPUT], {
            "default": _geom_ROTATION_MODE_OUTPUT,
            "tooltip": "Rotate the output/row space (what the layer emits) or the input/column "
                       "space (what the layer listens to)."})
        return {"required": inputs, "hidden": {"unique_id": "UNIQUE_ID"}}

    def _compass(self, obj, target_label, sub_components, depth_reach,
                 style_direction, rotation_angle, seed, manifold, unique_id=None):
        factory = lambda w, ck: fast_style_compass_rotation(
            w, angle_deg=rotation_angle, hue=style_direction, depth=depth_reach,
            seed=seed, mode=manifold, layer_name=ck)
        recipe = {
            "type": "style_compass",
            "style_direction": style_direction,
            "rotation_angle": rotation_angle,
            "depth_reach": depth_reach,
            "seed": seed,
            "manifold": manifold,
        }
        headline = (f"{'CLIP' if self.IS_CLIP else 'Model'} Compass Rotator "
                    f"(tilt {rotation_angle:.1f} deg, dir {style_direction:.0f} deg, {depth_reach})")
        live_params = {"style_direction": style_direction, "rotation_angle": rotation_angle,
                       "depth_reach": depth_reach, "seed": seed, "manifold": manifold}
        return self._apply(obj, target_label, sub_components, factory, recipe, headline,
                           unique_id, live_params)


class ArthemyKrea2LatentSpaceRotator(_StyleCompassMixin, BaseRotatorNode):
    """Compass Rotator (Model): aim by bearing + distance instead of by axis.

    Tier 4 continuous 2D polar rotation of the diffusion model. Named for how you steer it:
    `style_direction` is the bearing, `rotation_angle` the distance travelled from base.
    """

    IS_CLIP = False
    TARGET_MAP = ArthemyKrea2ModelBlockSurgeonTuner.MODEL_TARGET_MAP
    LOG_TAG = "ARTHEMY MODEL COMPASS"
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "rotate_latent_space"
    CATEGORY = "Arthemy/Krea2 Tuners"

    def rotate_latent_space(self, model: Any, target_block: str = "Block_1 (All 0-4)",
                            sub_components: str = "All Components", depth_reach: str = "Default",
                            style_direction: float = 180.0, rotation_angle: float = 15.0,
                            seed: int = 42, manifold: str = None, unique_id=None) -> Tuple[Any, str]:
        return self._compass(model, target_block, sub_components, depth_reach,
                             style_direction, rotation_angle, seed,
                             manifold or _geom_ROTATION_MODE_OUTPUT, unique_id)


class ArthemyKrea2CLIPSpaceRotator(_StyleCompassMixin, BaseRotatorNode):
    """Compass Rotator (CLIP): the same polar steering applied to the text encoder."""

    IS_CLIP = True
    TARGET_MAP = ArthemyKrea2CLIPBlockSurgeonTuner.CLIP_TARGET_MAP
    LOG_TAG = "ARTHEMY CLIP COMPASS"
    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("CLIP", "info")
    FUNCTION = "rotate_clip_space"
    CATEGORY = "Arthemy/Krea2 Tuners"

    def rotate_clip_space(self, clip: Any, target_layer: str = "Layer_1 (All 0-4)",
                          sub_components: str = "All Components", depth_reach: str = "Default",
                          style_direction: float = 180.0, rotation_angle: float = 10.0,
                          seed: int = 42, manifold: str = None, unique_id=None) -> Tuple[Any, str]:
        return self._compass(clip, target_layer, sub_components, depth_reach,
                             style_direction, rotation_angle, seed,
                             manifold or _geom_ROTATION_MODE_OUTPUT, unique_id)

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
        output_path, file_name_out = resolve_save_path(output_checkpoint)
        logger.info(f"[ARTHEMY KREA2 MODEL SAVER] Stream-saving model: {file_name_out}")

        sd = model.model.state_dict()
        arch_info = Krea2TensorParser.probe_architecture(sd)
        if arch_info['num_model_blocks'] != Krea2Config.MAX_UNET_BLOCKS:
            logger.warning(f"[ARTHEMY KREA2 MODEL SAVER] ⚠️ Topology warning: detected {arch_info['num_model_blocks']} blocks (expected baseline: {Krea2Config.MAX_UNET_BLOCKS}).")
        else:
            logger.info(f"[ARTHEMY KREA2 MODEL SAVER] Topology validated: {arch_info['num_model_blocks']} blocks detected.")

        patcher = get_patcher(model)
        patches = getattr(patcher, "patches", {}) or {}

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

            target_k = resolve_target_key(patcher, k, model_sd=sd)
            has_patch = (target_k in patches) or (k in patches)
            if has_patch:
                if find_companion_scale(k, sd) is not None:
                    logger.warning(f"[ARTHEMY KREA2 MODEL SAVER] Skipping patch calculation on quantized layer with companion scale: {k}")
                    return dequantize_weight(v, scale=find_companion_scale(k, sd))
                return ComfyPatcherAdapter.calculate_safe_weight(model, target_k, v, model_sd=sd)

            # Fold the companion FP8 scale into the weight instead of dropping the weight
            return dequantize_weight(v, scale=find_companion_scale(k, sd))

        for clean_key, tensor in process_tensor_stream(
            sd, process_tensor, target_dtype=target_dtype,
            memory_threshold_mb=Krea2Config.DEFAULT_GC_THRESHOLD_MB,
            emit_quant_metadata=True,
            clean_keys=True,
        ):
            if tensor is not None:
                final_sd[clean_key] = tensor

        metadata = {"format": "pt"}
        # Only write _quantization_metadata if actually saving in FP8 precision and quant_layers are present
        if precision == "FP8_E4M3" and len(quant_layers) > 0:
            metadata["_quantization_metadata"] = json.dumps({"layers": quant_layers})
        if extra_pnginfo and "workflow" in extra_pnginfo:
            metadata["workflow"] = json.dumps(extra_pnginfo["workflow"])

        tmp_output_path = output_path + ".tmp"
        try:
            safetensors.torch.save_file(final_sd, tmp_output_path, metadata=metadata)
            if os.path.exists(output_path):
                os.remove(output_path)
            os.replace(tmp_output_path, output_path)
        finally:
            if os.path.exists(tmp_output_path):
                try: os.remove(tmp_output_path)
                except OSError: pass
            del final_sd
            gc.collect()

        logger.info(f"[ARTHEMY KREA2 MODEL SAVER] SUCCESS: {output_path}")
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
        output_path, file_name_out = resolve_save_path(output_checkpoint)
        logger.info(f"[ARTHEMY KREA2 CLIP SAVER] Stream-saving CLIP: {file_name_out}")

        clip_sd = clip.get_sd()
        arch_info = Krea2TensorParser.probe_architecture(clip_sd)
        if arch_info['num_clip_layers'] != Krea2Config.MAX_CLIP_LAYERS:
            logger.warning(f"[ARTHEMY KREA2 CLIP SAVER] ⚠️ Topology warning: detected {arch_info['num_clip_layers']} layers (expected baseline: {Krea2Config.MAX_CLIP_LAYERS}).")
        else:
            logger.info(f"[ARTHEMY KREA2 CLIP SAVER] Topology validated: {arch_info['num_clip_layers']} layers detected.")

        patcher = get_patcher(clip)
        patches = getattr(patcher, "patches", {}) or {}

        target_dtype = torch.float8_e4m3fn if precision == "FP8_E4M3" else torch.bfloat16
        final_sd = {}

        def process_clip_tensor(k, v):
            target_k = resolve_target_key(patcher, k, model_sd=clip_sd)
            has_patch = (target_k in patches) or (k in patches)
            if has_patch:
                if find_companion_scale(k, clip_sd) is not None:
                    return dequantize_weight(v, scale=find_companion_scale(k, clip_sd))
                return ComfyPatcherAdapter.calculate_safe_weight(clip, target_k, v, model_sd=clip_sd)
            return dequantize_weight(v, scale=find_companion_scale(k, clip_sd))

        for clean_key, tensor in process_tensor_stream(
            clip_sd, process_clip_tensor, target_dtype=target_dtype,
            memory_threshold_mb=Krea2Config.DEFAULT_GC_THRESHOLD_MB,
            clean_keys=False,
        ):
            if tensor is not None:
                final_sd[clean_key] = tensor

        metadata = {"format": "pt"}
        if extra_pnginfo and "workflow" in extra_pnginfo:
            metadata["workflow"] = json.dumps(extra_pnginfo["workflow"])

        tmp_output_path = output_path + ".tmp"
        try:
            safetensors.torch.save_file(final_sd, tmp_output_path, metadata=metadata)
            if os.path.exists(output_path):
                os.remove(output_path)
            os.replace(tmp_output_path, output_path)
        finally:
            if os.path.exists(tmp_output_path):
                try: os.remove(tmp_output_path)
                except OSError: pass
            del final_sd
            gc.collect()

        logger.info(f"[ARTHEMY KREA2 CLIP SAVER] SUCCESS: {output_path}")
        return (output_path,)

# ==============================================================================
# LORA TRIO (Block Loader, Sub-Block, Sub-Block Chaos) & UTILITIES
# ==============================================================================

# Suffix markers used to split a LoRA key into (module_base, role).
# Order matters: the longest / most specific markers come first.
_LORA_ROLE_MARKERS = (
    ("lora_up", "up"),
    ("lora_down", "down"),
    ("lora_B", "up"),
    ("lora_A", "down"),
    ("lora_b", "up"),
    ("lora_a", "down"),
    ("hada_w1_a", "up"),
    ("hada_w1_b", "down"),
    ("hada_w2_a", "down"),
    ("hada_w2_b", "down"),
    ("lokr_w1_a", "up"),
    ("lokr_w1_b", "down"),
    ("lokr_w2_a", "down"),
    ("lokr_w2_b", "down"),
    ("lokr_w1", "up"),
    ("lokr_w2", "down"),
    ("oft_blocks", "single"),
    ("boft_blocks", "single"),
    ("diff_b", "single"),
    ("diff", "single"),
    ("alpha", "alpha"),
    ("dora_scale", "alpha"),
)


def split_lora_key(key: str) -> Tuple[str, str]:
    """Splits a LoRA state-dict key into (module_base, role).

    role is one of "up", "down", "alpha", "single" or "unknown". The module base is what all the
    tensors of the same adapted layer share, which is what block/sub-tensor filtering and chance
    rolls must operate on.

    Markers are matched on dot boundaries only, so a prefix such as "model.diffusion_model."
    can never be mistaken for the "diff" marker.
    """
    parts = key.split(".")
    for i, part in enumerate(parts):
        for marker, role in _LORA_ROLE_MARKERS:
            if part == marker:
                base = ".".join(parts[:i])
                return (base if base else key), role
    return key, "unknown"


def scale_lora_tensor(key: str, tensor: torch.Tensor, mult: float) -> torch.Tensor:
    """Applies a block multiplier to a LoRA tensor **linearly**.

    The LoRA delta is (roughly) alpha/rank * B @ A, so scaling *both* factors by m yields an m^2
    effect (m^3 once alpha is included). Only the "up"/B side - or a standalone full diff - is
    scaled here, so a multiplier of 0.5 really means half strength.
    """
    role = split_lora_key(key)[1]
    if role in ("up", "single", "unknown"):
        return tensor * mult
    return tensor


def match_lora_group(clean_key: str, group_map: Dict[str, List[str]]) -> Optional[str]:
    """Maps a cleaned LoRA key to one of the coarse GROUP_MAP sections."""
    for group_name, prefixes in group_map.items():
        if any(clean_key.startswith(pfx) or f".{pfx}" in clean_key for pfx in prefixes):
            return group_name
    return None


def warn_if_no_block_match(lora_name: str, matched: int, total: int) -> None:
    """Warns when block filtering matched nothing, which usually means an unsupported key format."""
    if total > 0 and matched == 0:
        logger.warning(
            f"[Arthemy LoRA] No block/layer index could be extracted from any key of '{lora_name}' "
            "(0 matches). The LoRA is probably stored in a naming convention this suite does not "
            "parse yet (e.g. kohya 'lora_unet_blocks_0_...' with underscores instead of dots), so "
            "per-block filtering had no effect and the LoRA was applied unfiltered."
        )


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
        lora_list = combo_options(folder_paths.get_filename_list("loras"))
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

    def load_lora(self, model: Any, clip: Any, lora_name: str, strength_model: float = 1.0, strength_clip: float = 1.0, **kwargs: float) -> Tuple[Any, Any, str]:
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, "LoRA bypassed (strength 0)")
        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            raise FileNotFoundError(f"Arthemy Suite Error: LoRA '{lora_name}' not found in ComfyUI loras directory.")

        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        # 1. Resolve the section multiplier once per *module* so both factors of a LoRA pair
        #    receive a consistent decision (and the multiplier is applied only once, linearly).
        module_mult: Dict[str, float] = {}
        matched_modules = 0
        for k in lora.keys():
            module_base, _role = split_lora_key(k)
            if module_base in module_mult:
                continue
            clean_base = Krea2TensorParser.clean_key(module_base)
            matched_group = match_lora_group(clean_base, self.GROUP_MAP)
            if matched_group:
                matched_modules += 1
                module_mult[module_base] = float(kwargs.get(matched_group, 0.00))
            else:
                module_mult[module_base] = 1.0  # ungrouped keys (CLIP/global) pass through untouched

        warn_if_no_block_match(lora_name, matched_modules, len(module_mult))

        # 2. Filter and scale, keeping every tensor of a surviving module together
        filtered_lora = {}
        for k, v in lora.items():
            module_base, _role = split_lora_key(k)
            mult = module_mult.get(module_base, 1.0)
            if mult == 0.0:
                continue
            filtered_lora[k] = v if mult == 1.0 else scale_lora_tensor(k, v, mult)

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
        lora_list = combo_options(folder_paths.get_filename_list("loras"))
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
                                                  "tooltip": "Sub-tensor multiplier (0.0 = filter out entirely, 1.0 = 100% strength)."})
        return inputs

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "info")
    FUNCTION = "load_sub_block_lora"
    CATEGORY = "Arthemy/Krea2 LoRA"

    def load_sub_block_lora(self, model: Any, clip: Any, lora_name: str, strength_model: float = 1.0, strength_clip: float = 1.0, target_block: str = "Block_1 (All 0-4)", **kwargs: float) -> Tuple[Any, Any, str]:
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, "LoRA bypassed (strength 0)")

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            raise FileNotFoundError(f"Arthemy Suite Error: LoRA '{lora_name}' not found in ComfyUI loras directory.")
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        selected_blocks = resolve_target_map_entry(self.MODEL_TARGET_MAP, target_block,
                                                   set(range(0, Krea2Config.MAX_UNET_BLOCKS)))

        # Decide per module (not per tensor) so LoRA pairs are never split apart
        module_decision: Dict[str, Optional[float]] = {}
        matched_modules = 0
        for k in lora.keys():
            module_base, _role = split_lora_key(k)
            if module_base in module_decision:
                continue
            clean_base = Krea2TensorParser.clean_key(module_base)
            idx, sub_key = Krea2TensorParser.extract_model_block_idx(clean_base)
            if idx is None:
                module_decision[module_base] = 1.0
                continue
            matched_modules += 1
            if idx not in selected_blocks:
                module_decision[module_base] = None
                continue
            matched_widget = Krea2TensorParser.match_model_sub_tensor(sub_key)
            if not matched_widget:
                module_decision[module_base] = 1.0
                continue
            weight_mult = float(kwargs.get(matched_widget, 0.00))
            module_decision[module_base] = None if weight_mult <= 0.0 else weight_mult

        warn_if_no_block_match(lora_name, matched_modules, len(module_decision))

        filtered_lora = {}
        for k, v in lora.items():
            module_base, _role = split_lora_key(k)
            mult = module_decision.get(module_base, 1.0)
            if mult is None:
                continue
            filtered_lora[k] = v if mult == 1.0 else scale_lora_tensor(k, v, mult)

        if not filtered_lora:
            return (model, clip, f"Sub-Block LoRA '{lora_name}' bypassed (nothing selected)")

        m, c = comfy.sd.load_lora_for_models(model, clip, filtered_lora, strength_model, strength_clip)
        return (m, c, f"Sub-Block LoRA ({len(filtered_lora)} keys)")


class ArthemyKrea2LoadChaosLoraBlockSurgeon(BaseSurgeonTuner):
    """Tier 3: Stochastic Sub-Block Chaos LoRA filtering.
    Selectively and randomly activates LoRA key components according to per-sub-tensor chance rolls."""
    MODEL_TARGET_MAP = ArthemyKrea2ModelBlockSurgeonTuner.MODEL_TARGET_MAP
    SURGEON_MAP = Krea2TensorParser.get_descriptive_model_surgeon_map()

    @classmethod
    def INPUT_TYPES(s):
        lora_list = combo_options(folder_paths.get_filename_list("loras"))
        inputs = {
            "required": {
                "model": ("MODEL",), "clip": ("CLIP",), "lora_name": (lora_list,),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -99.00, "max": 99.00, "step": 0.01}),
                "target_block": (list(s.MODEL_TARGET_MAP.keys()), {"default": "Block_1 (All 0-4)"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "base_chance": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
        for name in s.SURGEON_MAP.keys():
            inputs["required"][f"{name}_chance"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01})
        return inputs

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "info")
    FUNCTION = "load_chaos_lora_surgeon"
    CATEGORY = "Arthemy/Krea2 LoRA"

    def load_chaos_lora_surgeon(self, model: Any, clip: Any, lora_name: str, strength_model: float = 1.0, strength_clip: float = 1.0, target_block: str = "Block_1 (All 0-4)", seed: int = 42, base_chance: float = 0.20, **kwargs: float) -> Tuple[Any, Any, str]:
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, "LoRA bypassed (strength 0)")

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            raise FileNotFoundError(f"Arthemy Suite Error: LoRA '{lora_name}' not found in ComfyUI loras directory.")
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        selected_blocks = resolve_target_map_entry(self.MODEL_TARGET_MAP, target_block,
                                                   set(range(0, Krea2Config.MAX_UNET_BLOCKS)))

        # The chance is rolled ONCE PER MODULE. Rolling per tensor could keep lora_up while
        # dropping the matching lora_down, which breaks comfy.lora key pairing at load time.
        module_keep: Dict[str, bool] = {}
        matched_modules = 0
        for k in lora.keys():
            module_base, _role = split_lora_key(k)
            if module_base in module_keep:
                continue
            clean_base = Krea2TensorParser.clean_key(module_base)
            idx, sub_key = Krea2TensorParser.extract_model_block_idx(clean_base)
            if idx is not None:
                matched_modules += 1
                if idx not in selected_blocks:
                    module_keep[module_base] = False
                    continue
            else:
                # Modules without block index (e.g. CLIP text encoder, txtfusion, in/out projections)
                # are preserved intact, matching ArthemyKrea2LoadSubBlockLora behavior.
                module_keep[module_base] = True
                continue

            matched_widget = Krea2TensorParser.match_model_sub_tensor(sub_key)
            chance = float(kwargs.get(f"{matched_widget}_chance", base_chance)) if matched_widget else float(base_chance)
            if chance <= 0.0:
                module_keep[module_base] = False
                continue

            fast_seed = generate_fast_seed(module_base, seed)
            rng = torch.Generator(device="cpu").manual_seed(fast_seed)
            module_keep[module_base] = bool(torch.rand(1, generator=rng).item() < chance)

        warn_if_no_block_match(lora_name, matched_modules, len(module_keep))

        filtered_lora = {k: v for k, v in lora.items() if module_keep.get(split_lora_key(k)[0], False)}
        kept_modules = sum(1 for keep in module_keep.values() if keep)

        if not filtered_lora:
            return (model, clip, f"Sub-Block Chaos LoRA '{lora_name}' bypassed (no module survived the roll)")

        m, c = comfy.sd.load_lora_for_models(model, clip, filtered_lora, strength_model, strength_clip)
        return (m, c, f"Sub-Block Chaos LoRA ({kept_modules} modules / {len(filtered_lora)} keys)")


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

    @staticmethod
    def _reset_patcher(patcher: Any) -> Tuple[int, int]:
        """Drops this patcher's pending patches and recipes. Returns (patches, recipes).

        Three things a reset must NOT do, all of them learned from ComfyUI's own
        `model_patcher.py` rather than guessed:

        `backup` is NOT cleared. `ModelPatcher.clone()` hands the clone the parent's own
        model AND the parent's own backup dict (`get_clone_model_override`, line 427-428),
        and `backup` is where the ORIGINAL weights live once ComfyUI has patched them into
        the model in place. The reset is what makes ComfyUI restore them: rolling
        `patches_uuid` makes `current_weight_patches_uuid != patches_uuid` (line 1258), which
        is the exact condition that triggers `unpatch_model(unpatch_weights=True)`. Throwing
        the backup away would leave that restore with nothing to restore FROM - the previous
        tuning would stay fused into the live weights and the "reset" model would still carry
        it.

        `unpatch_hooks()` is NOT called. It writes the hook backups straight into
        `self.model` with `copy_to_param` and then clears `hook_backup` - and `clone()` shares
        `hook_backup` BY REFERENCE (line 492), so both the write and the clear reach every
        other branch of the graph. It is also unnecessary: ComfyUI calls it itself at the top
        of `load()` and inside `unpatch_model()`, before any patch is applied.

        `object_patches` is NOT cleared either. No node in this suite ever writes one, so
        everything in there belongs to somebody else - ModelSamplingAuraFlow's shift, FreeU,
        Differential Diffusion. Clearing it made "reset the Arthemy tuning" reset those too.
        """
        if patcher is None:
            return (0, 0)
        n_patches = len(getattr(patcher, "patches", {}) or {})
        n_recipes = 0
        if hasattr(patcher, "model_options"):
            n_recipes = sum(len(patcher.model_options.get(rk, []) or [])
                            for rk in ARTHEMY_RECIPE_KEYS if rk != "arthemy_section_meta")
        if hasattr(patcher, "patches"):
            patcher.patches = {}
        patcher.patches_uuid = uuid.uuid4()
        clear_arthemy_recipes(patcher)
        return (n_patches, n_recipes)

    def reset(self, model: Any, clip: Any, reset_model: bool = True, reset_clip: bool = True) -> Tuple[Any, Any, str]:
        m = model.clone() if (reset_model and model is not None) else model
        c = clip.clone() if (reset_clip and clip is not None) else clip
        mp = self._reset_patcher(get_patcher(m)) if reset_model else (0, 0)
        cp = self._reset_patcher(get_patcher(c)) if reset_clip else (0, 0)
        # Say what was actually dropped. "Reset patchers clean." was true of nothing in
        # particular and hid the case that matters: a reset that found nothing to reset.
        info = (f"Reset | Model: {mp[0]} patches, {mp[1]} recipes | "
                f"CLIP: {cp[0]} patches, {cp[1]} recipes")
        if not any(mp + cp):
            info += " | nothing was pending"
        info += " | hook-based and object patches from other nodes are left untouched"
        logger.info(f"[Arthemy Reset Patcher] {info}")
        return (m, c, info)

# ==============================================================================
# POINT 3 IMPLEMENTATION: MODEL BAKER WITH STREAM GENERATOR
# ==============================================================================
def _shallow_clone_module(module: torch.nn.Module) -> torch.nn.Module:
    """Copy-on-write clone of a single nn.Module level (children/params/buffers dicts only)."""
    cloned = copy.copy(module)
    for attr in ("_modules", "_parameters", "_buffers"):
        if hasattr(module, attr):
            setattr(cloned, attr, copy.copy(getattr(module, attr)))
    return cloned


def isolate_and_assign_baked_weights(patcher: Any, baked_weights: Dict[str, torch.Tensor]) -> int:
    """Isolates the baked parameters onto a cloned module hierarchy without in-place mutating
    shared base weights. Returns how many tensors were actually assigned.

    The keys of `baked_weights` are state_dict keys of `patcher.model`, and a state_dict key IS
    the module path to walk. They must NOT be cleaned first: `Krea2TensorParser.clean_key` strips
    `diffusion_model.` (and the text-encoder wrapper prefixes), which is exactly the first hop of
    the walk. Cleaning made every single tensor unlocatable on a stock ComfyUI patcher - whose
    `.model` is the BaseModel wrapper, not the UNet - while the code below still went on to clear
    `patches`: the Baker reported "baked N parameters" and handed back a model whose tuning had
    been silently deleted, with no patches and no recipes left to rebuild it from.

    Nothing is cleared that was not actually baked. Only the keys assigned here stop being pending
    patches, a partial bake keeps the rest of them, and a bake that locates nothing leaves the
    patcher exactly as it was.

    Traversal handles indexed containers (nn.ModuleList / nn.Sequential) and distinguishes
    parameters from buffers, so a buffer is never replaced by an nn.Parameter.
    """
    if not hasattr(patcher, "model"):
        return 0

    patches = getattr(patcher, "patches", None)

    if not baked_weights:
        # Nothing to bake is only equivalent to "already clean" when nothing was pending either.
        # When there ARE patches (e.g. every candidate was skipped as quantized), clearing them
        # would delete a tuning that was never folded into the weights.
        if patches:
            logger.warning("[Arthemy Baker] Nothing could be baked, so the pending patches were "
                           "left in place - this model still carries its tuning as patches.")
            return 0
        if hasattr(patcher, "patches"): patcher.patches = {}
        if hasattr(patcher, "backup"): patcher.backup = {}
        clear_arthemy_recipes(patcher)
        return 0

    cloned_top = _shallow_clone_module(patcher.model)

    def _assign_at(root: Any, path_parts: List[str], new_tensor: torch.Tensor) -> bool:
        curr = root
        for part in path_parts[:-1]:
            child = None
            if hasattr(curr, "_modules") and part in curr._modules:
                child = curr._modules[part]
            elif hasattr(curr, part) and isinstance(getattr(curr, part), torch.nn.Module):
                child = getattr(curr, part)

            if child is None or not isinstance(child, torch.nn.Module):
                return False

            cloned_child = _shallow_clone_module(child)
            if hasattr(curr, "_modules") and part in curr._modules:
                curr._modules[part] = cloned_child
            else:
                setattr(curr, part, cloned_child)
            curr = cloned_child

        leaf = path_parts[-1]
        target_tensor = new_tensor.detach().clone()
        if hasattr(curr, "_parameters") and leaf in curr._parameters:
            existing = curr._parameters[leaf]
            if existing is not None:
                target_tensor = target_tensor.to(dtype=existing.dtype, device=existing.device)
            curr._parameters[leaf] = torch.nn.Parameter(target_tensor, requires_grad=False)
            return True
        if hasattr(curr, "_buffers") and leaf in curr._buffers:
            existing = curr._buffers[leaf]
            if isinstance(existing, torch.Tensor):
                target_tensor = target_tensor.to(dtype=existing.dtype, device=existing.device)
            curr._buffers[leaf] = target_tensor
            return True
        return False

    assigned_keys: List[str] = []
    n_missing = 0

    for k, new_tensor in baked_weights.items():
        # The raw key first. The cleaned spelling is tried only as a fallback, for a patcher
        # whose `.model` already IS the inner module rather than a wrapper around it.
        cleaned = Krea2TensorParser.clean_key(k)
        candidates = [k] if cleaned == k else [k, cleaned]
        if any(_assign_at(cloned_top, cand.split("."), new_tensor) for cand in candidates):
            assigned_keys.append(k)
        else:
            n_missing += 1

    if not assigned_keys:
        logger.error(f"[Arthemy Baker] None of the {len(baked_weights)} baked tensor(s) could be "
                     "located in the module hierarchy. The bake was abandoned and the model was "
                     "left exactly as it was, patches included - nothing has been lost, but "
                     "nothing was folded in either.")
        return 0

    if n_missing:
        logger.warning(f"[Arthemy Baker] {n_missing} baked tensor(s) could not be located in the "
                       f"module hierarchy and were skipped ({len(assigned_keys)} assigned). Their "
                       "patches were left in place, so that part of the tuning still applies.")

    patcher.model = cloned_top

    # Only the keys that really landed stop being pending. Rebuilt as a new dict rather than
    # mutated in place, because a ModelPatcher clone can share this object with its parent.
    if isinstance(patches, dict):
        dropped = set()
        for k in assigned_keys:
            dropped.add(k)
            dropped.add(Krea2TensorParser.clean_key(k))
        patcher.patches = {pk: pv for pk, pv in patches.items() if pk not in dropped}

    # "Nothing left pending", not "everything we chose to bake landed". `n_missing` counts only
    # tensors that reached `baked_weights` and could not be located; a layer skipped as
    # quantized never gets there, so n_missing == 0 was true while its patch was still attached
    # - and clearing the recipes then threw away the rotations and 5D injections that are
    # reproduced ONLY from recipes and are never serialized as tensors.
    if not getattr(patcher, "patches", None):
        # A complete bake: the backup and the recipe log both describe work that is now part
        # of the weights themselves. `object_patches` is deliberately NOT touched - this suite
        # never writes one, so clearing it can only destroy another node's work.
        if hasattr(patcher, "backup"): patcher.backup = {}
        clear_arthemy_recipes(patcher)

    patcher.patches_uuid = uuid.uuid4()
    return len(assigned_keys)


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

    @staticmethod
    def _bake_domain(label: str, container: Any, patcher: Any) -> Tuple[Dict[str, torch.Tensor], int, int]:
        """Materializes every patched weight of one domain (model or clip).

        Iterates the state_dict directly instead of going through process_tensor_stream: the
        stream converted *every* tensor to bf16/CPU even though only patched ones are needed,
        which wasted a full dtype+contiguous copy per parameter.
        """
        sd = patcher.model.state_dict() if hasattr(patcher, "model") else {}
        patches = getattr(patcher, "patches", {}) or {}
        baked: Dict[str, torch.Tensor] = {}
        n_patched = 0
        n_skipped_quant = 0
        accumulated = 0
        threshold = Krea2Config.DEFAULT_GC_THRESHOLD_MB * 1024 * 1024

        for k, v in sd.items():
            if k.endswith(".comfy_quant") or k.endswith("_scale") or k.endswith(".weight_scale"):
                continue

            target_k = resolve_target_key(patcher, k, model_sd=sd)
            if target_k not in patches and k not in patches:
                continue

            if find_companion_scale(k, sd) is not None:
                # Baking a dequantized bf16 tensor back onto a quantized layer would leave the
                # companion scale in place and double-apply it at inference time.
                n_skipped_quant += 1
                continue

            patched_weight = ComfyPatcherAdapter.calculate_safe_weight(container, target_k, v, model_sd=sd)
            baked[target_k] = patched_weight.detach().to(device="cpu").contiguous()
            n_patched += 1

            if isinstance(v, torch.Tensor):
                accumulated += v.element_size() * v.nelement()
            if accumulated >= threshold:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                accumulated = 0

        if n_skipped_quant:
            logger.warning(f"[ARTHEMY KREA2 MODEL BAKER] {n_skipped_quant} quantized {label} layer(s) were "
                           "left un-baked (FP8 weights with a companion scale cannot be safely fused). "
                           "Use the Model/CLIP Saver on a non-quantized checkpoint to persist them.")
        return baked, n_patched, n_skipped_quant

    def bake(self, model: Any, clip: Any) -> Tuple[Any, Any, str]:
        m_baked = model.clone()
        c_baked = clip.clone()

        m_p = get_patcher(m_baked)
        c_p = get_patcher(c_baked)

        baked_model_weights, n_model_patched, m_skipped = self._bake_domain("model", m_baked, m_p)
        baked_clip_weights, n_clip_patched, c_skipped = self._bake_domain("clip", c_baked, c_p)

        # Apply isolated parameters without mutating the original shared module. The counts
        # reported are the ones that actually reached the module tree, not the ones collected -
        # a Baker that says "baked 900" while assigning 0 is how a lost tuning goes unnoticed.
        n_model_baked = isolate_and_assign_baked_weights(m_p, baked_model_weights)
        n_clip_baked = isolate_and_assign_baked_weights(c_p, baked_clip_weights)

        info = f"Arthemy Krea-2 Model Baker: baked {n_model_baked} model parameters, {n_clip_baked} clip parameters."
        if n_model_baked != n_model_patched or n_clip_baked != n_clip_patched:
            info += (f" | ⚠️ {n_model_patched - n_model_baked + n_clip_patched - n_clip_baked} "
                     "patched tensor(s) could not be located and still apply as patches")
        if m_skipped or c_skipped:
            info += f" | {m_skipped + c_skipped} quantized layer(s) skipped."
        logger.info(f"[ARTHEMY KREA2 MODEL BAKER] {info}")
        return (m_baked, c_baked, info)

# ==============================================================================
# VISUALIZER NODES
# ==============================================================================

def parse_patch_entry(p: Any) -> Tuple[float, bool, bool, bool, float, float, str]:
    """Analyzes a ComfyUI patch entry (strength_patch, diff, strength_model, ...).

    Returns (offset_delta, is_lora, is_chaos, is_rotation, rot_angle, rot_hue, rot_reach).
    """
    if not isinstance(p, (tuple, list)):
        return 0.0, False, False, False, 0.0, 180.0, "Default"

    strength_patch = float(p[0]) if (len(p) >= 1 and isinstance(p[0], (int, float))) else 1.0
    diff = p[1] if len(p) >= 2 else p[0]
    strength_model = float(p[2]) if (len(p) >= 3 and isinstance(p[2], (int, float))) else 1.0

    offset_delta = 0.0
    is_lora = False
    is_chaos = False
    is_rotation = bool(getattr(diff, "_is_arthemy_rotation", False))
    is_five_d = bool(getattr(diff, "_is_arthemy_five_d", False))
    rot_angle = getattr(diff, "_arthemy_rotation_angle", 0.0)
    rot_hue = getattr(diff, "_arthemy_rotation_hue", 180.0)
    rot_reach = getattr(diff, "_arthemy_depth_reach", "Default")

    if strength_model != 1.0:
        offset_delta += (strength_model - 1.0)

    # ("lora", payload) / ("diff", payload) style entries
    if isinstance(diff, (tuple, list)) and len(diff) == 2 and isinstance(diff[0], str):
        diff = diff[1]

    # Unwrap nested tuple structures used by ComfyUI LoRA patchers
    while isinstance(diff, (tuple, list)) and len(diff) == 1 and isinstance(diff[0], (tuple, list, torch.Tensor)):
        diff = diff[0]

    for candidate in ((diff,) if not isinstance(diff, (tuple, list)) else tuple(diff)):
        if getattr(candidate, "_is_arthemy_rotation", False):
            is_rotation = True
            rot_angle = getattr(candidate, "_arthemy_rotation_angle", rot_angle)
            rot_hue = getattr(candidate, "_arthemy_rotation_hue", rot_hue)
            rot_reach = getattr(candidate, "_arthemy_depth_reach", rot_reach)
        if getattr(candidate, "_is_arthemy_chaos", False):
            is_chaos = True
        if getattr(candidate, "_is_arthemy_five_d", False):
            is_five_d = True

    # Modern ComfyUI WeightAdapter object (LoRA, LoHa, rotation adapters, ...).
    # A 5D injection is a low-rank adapter too, but it is NOT an independent LoRA, so it
    # must never be reported as one (that is what painted 5D sections purple).
    if is_weight_adapter(diff) or hasattr(diff, "lora_a"):
        # A channel-scale adapter is suite-generated, like a rotation or a 5D injection: it must
        # not be reported as an external LoRA (which is what paints a section pink and triggers
        # the Preset Saver's "LoRAs excluded" warning).
        if not is_rotation and not is_five_d and not getattr(diff, "_is_arthemy_granular", False):
            is_lora = True
    elif strength_patch != 0.0:
        if isinstance(diff, (tuple, list)):
            has_tensors = any(isinstance(x, torch.Tensor) for x in diff)
            if has_tensors or len(diff) >= 2:
                if not is_rotation and not is_five_d:
                    is_lora = True
            elif len(diff) == 1:
                val = diff[0]
                if isinstance(val, (int, float)):
                    offset_delta += (float(val) - 1.0) * strength_patch
                elif isinstance(val, torch.Tensor):
                    if val.numel() == 1:
                        offset_delta += (float(val.item()) - 1.0) * strength_patch
                    elif val.numel() > 1:
                        if not is_rotation:
                            is_chaos = True
                        offset_delta += float(val.float().mean().item()) * strength_patch
        elif isinstance(diff, (int, float)):
            offset_delta += (float(diff) - 1.0) * strength_patch
        elif isinstance(diff, torch.Tensor):
            if diff.numel() == 1:
                offset_delta += (float(diff.item()) - 1.0) * strength_patch
            elif diff.numel() > 1:
                if not is_rotation:
                    is_chaos = True
                offset_delta += float(diff.float().mean().item()) * strength_patch

    return offset_delta, is_lora, is_chaos, is_rotation, rot_angle, rot_hue, rot_reach


def hue_to_rgb(hue: float) -> Tuple[int, int, int]:
    """Converts a continuous 2D Polar Compass Hue (0-360 deg) to a pure vibrant RGB triplet.

    Uses HLS with L=0.5 / S=1.0, which is the fully saturated hue circle.
    """
    h = (float(hue) % 360.0) / 360.0
    r, g, b = colorsys.hls_to_rgb(h, 0.50, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


# Brand hue of each domain, taken from the same hexes the JS palette uses
# (Model #6351cf -> 248.6 deg, CLIP #FFD700 -> 50.6 deg), and the half-width of the band a
# measured hue is allowed to move within.
MODEL_BRAND_HUE = 248.6
CLIP_BRAND_HUE = 50.6
HUE_BAND_HALF_WIDTH = 45.0


def domain_band_hue(measured_hue: float, is_clip: bool) -> float:
    """Folds a measured hue into a band centred on its own domain's brand hue.

    Drawing the raw measured hue looked informative until a Model rotation landed on hue 50
    - gold - and tinted a MODEL readout in the CLIP colour. Banding keeps the per-section
    variation while making it impossible for one domain to wear the other's identity:
    Model stays in 204-294, CLIP in 6-96. Mirrors domainBandHue() in the JS panel exactly,
    so the exported PNG and the live widget cannot disagree.
    """
    brand = CLIP_BRAND_HUE if is_clip else MODEL_BRAND_HUE
    m = float(measured_hue) % 360.0
    return (brand + (m / 360.0 - 0.5) * 2.0 * HUE_BAND_HALF_WIDTH) % 360.0


_VIS_FONT_CACHE: Dict[int, Any] = {}


def get_visualizer_font(size: int = 12) -> Any:
    """Returns a TrueType font when available, falling back to PIL's bitmap default.

    The bitmap default cannot render emoji or many symbols, which previously produced tofu
    glyphs (or an encoding error) in the HUD legend.
    """
    if size in _VIS_FONT_CACHE:
        return _VIS_FONT_CACHE[size]
    font = None
    for candidate in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            font = ImageFont.truetype(candidate, size)
            break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    _VIS_FONT_CACHE[size] = font
    return font


def render_visualizer_image(graph_data: List[Dict[str, Any]], title: str, is_clip: bool = False, visual_scale: float = 1.0, width: int = 960, height: int = 480) -> torch.Tensor:
    """Renders high-quality visualizer HUD image using PIL and converts to PyTorch IMAGE tensor [1, H, W, 3]."""
    img = Image.new("RGB", (width, height), color=(15, 23, 42)) # #0f172a
    draw = ImageDraw.Draw(img)
    font = get_visualizer_font(12)

    # Colour standards, kept byte-identical to ARTHEMY_PALETTE in the JS so the exported
    # PNG, the on-canvas HUD widget and the node's own title bar can never disagree.
    base_color = (255, 215, 0) if is_clip else (99, 81, 207)    # Gold for CLIP, purple for Model
    lora_color = (255, 82, 212)                                 # Pink: external LoRAs ONLY
    five_d_color = (234, 88, 12) if is_clip else (29, 78, 216)  # Orange for CLIP 5D, deep blue for Model 5D
    rot_color = (16, 185, 129)                                  # Emerald: rotations

    padding_x = int(width * 0.03)
    top_y = int(height * 0.08)
    chart_w = width - (padding_x * 2)
    chart_h = height - top_y - int(height * 0.12)
    center_y = top_y + (chart_h // 2)

    # 1. Outer container box
    draw.rectangle([padding_x - 4, top_y - 20, padding_x + chart_w + 4, top_y + chart_h + 18], outline=base_color, width=2)

    # 2. Header title
    draw.text((padding_x + 4, top_y - 18), title, fill=base_color, font=font)

    # 3. Header legend & scale (ASCII only: the bitmap fallback font has no emoji coverage)
    leg_x = padding_x + 170
    draw.text((leg_x, top_y - 18), "[#] Base", fill=base_color, font=font)
    draw.text((leg_x + 68, top_y - 18), "[#] LoRA", fill=lora_color, font=font)
    draw.text((leg_x + 136, top_y - 18), "[#] 5D", fill=five_d_color, font=font)
    draw.text((leg_x + 190, top_y - 18), "[o] Rotation", fill=rot_color, font=font)
    draw.text((leg_x + 272, top_y - 18), "[^] Chaos", fill=(255, 255, 255), font=font)

    scale_str = f"Scale: {visual_scale:.1f}x"
    draw.text((width - padding_x - 110, top_y - 18), scale_str, fill=(200, 200, 200), font=font)

    # 4. Zero baseline axis
    for x in range(padding_x, padding_x + chart_w, 8):
        draw.line([(x, center_y), (min(x + 4, padding_x + chart_w), center_y)], fill=(90, 90, 90), width=1)

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
            is_5d = sec.get("is_5d", False)
            is_chaos = sec.get("is_chaos", False)
            is_rot = sec.get("is_rotation", False) or (sec.get("rotation_angle", 0.0) != 0.0)
            rot_angle = sec.get("rotation_angle", 0.0)

            # Priority: an external LoRA is purple; a 5D harmonic injection is NOT an
            # independent LoRA, so it gets the domain-specific deep blue / orange instead.
            if is_lora:
                current_color = lora_color
            elif is_5d:
                current_color = five_d_color
            else:
                current_color = base_color
            emphasis = is_lora or is_5d

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
                draw.line(pts, fill=current_color, width=3 if emphasis else 2)
            else:
                draw.line([(x1, prev_y), (x1, y_val), (x2, y_val)], fill=current_color, width=3 if emphasis else 2)

            # Rotation badge: emerald marker + the actual angle of THIS section, with a
            # hue-tinted stem so the style direction stays readable at a glance.
            if is_rot and rot_angle != 0.0:
                mid_sec_x = x1 + (step_x / 2)
                badge_y = top_y + 12 if (idx % 2 == 0) else top_y + 26
                stem_color = hue_to_rgb(domain_band_hue(sec.get("rotation_hue", 180.0), is_clip))

                draw.line([(mid_sec_x, badge_y + 3), (mid_sec_x, center_y)], fill=stem_color, width=1)
                draw.ellipse([mid_sec_x - 3, badge_y - 3, mid_sec_x + 3, badge_y + 3],
                             fill=rot_color, outline=(255, 255, 255))
                ang_str = f"{rot_angle:.0f}°"
                draw.text((mid_sec_x - 8, badge_y - 13), ang_str, fill=rot_color, font=font)

            # 5D badge: a small filled square in the domain colour, plus the dimension count.
            if is_5d:
                mid_sec_x = x1 + (step_x / 2)
                tag_y = top_y + chart_h - 16 if (idx % 2 == 0) else top_y + chart_h - 28
                draw.rectangle([mid_sec_x - 3, tag_y - 3, mid_sec_x + 3, tag_y + 3], fill=five_d_color)
                dims = int(sec.get("five_d_dims", 0) or 0)
                if dims:
                    draw.text((mid_sec_x - 6, tag_y - 15), f"{dims}D", fill=five_d_color, font=font)

    # 6. X-Axis Section Labels (derived dynamically from graph_data points)
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

    num_points = max(len(graph_data), 1) if graph_data else (31 if not is_clip else 37)
    step_x = chart_w / num_points
    start_idx = 0
    for g in groups:
        end_x = padding_x + int((g["endIdx"] + 1) * step_x)
        start_x = padding_x + int(start_idx * step_x)
        mid_x = (start_x + end_x) // 2
        draw.text((mid_x - 10, axis_y), g["label"], fill=(200, 200, 200), font=font)
        draw.line([(end_x, top_y + chart_h - 2), (end_x, top_y + chart_h + 5)], fill=(80, 80, 80), width=1)
        start_idx = g["endIdx"] + 1

    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)


def clean_key_matches_projection(ck: str) -> bool:
    """Boundary-anchored check for the input/output projection tensors."""
    return (ck.startswith("first.") or ".first." in ck
            or ck.startswith("last.") or ".last." in ck)


MODEL_SECTION_KEYS = [
    "Block_1A", "Block_1B", "Block_1C", "Block_1D", "Block_1E",
    "Block_2A", "Block_2B", "Block_2C", "Block_2D", "Block_2E",
    "Block_3A", "Block_3B", "Block_3C", "Block_3D", "Block_3E",
    "Block_4A", "Block_4B", "Block_4C", "Block_4D", "Block_4E",
    "Block_5A", "Block_5B", "Block_5C", "Block_5D",
    "Block_6A", "Block_6B", "Block_6C", "Block_6D",
    "Text_Fusion", "Time_Embed", "Projection",
]

CLIP_SECTION_KEYS = [
    "Layer_1A", "Layer_1B", "Layer_1C", "Layer_1D", "Layer_1E",
    "Layer_2A", "Layer_2B", "Layer_2C", "Layer_2D", "Layer_2E",
    "Layer_3A", "Layer_3B", "Layer_3C", "Layer_3D", "Layer_3E",
    "Layer_4A", "Layer_4B", "Layer_4C", "Layer_4D", "Layer_4E",
    "Layer_5A", "Layer_5B", "Layer_5C", "Layer_5D", "Layer_5E",
    "Layer_6A", "Layer_6B", "Layer_6C", "Layer_6D", "Layer_6E",
    "Layer_7A", "Layer_7B", "Layer_7C", "Layer_7D", "Layer_7E", "Layer_7F",
    "Visual_1A", "Visual_1B", "Visual_1C", "Visual_1D", "Visual_1E", "Visual_1F",
    "Visual_2A", "Visual_2B", "Visual_2C", "Visual_2D", "Visual_2E", "Visual_2F",
    "Visual_3A", "Visual_3B", "Visual_3C", "Visual_3D", "Visual_3E", "Visual_3F",
    "Visual_4A", "Visual_4B", "Visual_4C", "Visual_4D", "Visual_4E", "Visual_4F",
    "Embedding",
]


def _new_section_slot(name: str) -> Dict[str, Any]:
    return {
        "block": name, "offset": 0.0, "scalar_count": 0,
        "is_lora": False, "is_chaos": False, "is_granular": False,
        "is_rotation": False, "rotation_angle": 0.0, "rotation_hue": 180.0,
        "rotation_type": "", "depth_reach": "Default", "relative_delta": 0.0,
        "is_five_d": False, "five_d_dims": 0, "five_d_source": "",
    }


def _section_for_model_key(ck: str) -> Optional[str]:
    if "txtfusion" in ck or "txtmlp" in ck:
        return "Text_Fusion"
    if "tmlp" in ck or "tproj" in ck:
        return "Time_Embed"
    if clean_key_matches_projection(ck):
        return "Projection"
    b_idx, _ = Krea2TensorParser.extract_model_block_idx(ck)
    return MODEL_IDX_TO_LABEL.get(b_idx) if b_idx is not None else None


def _section_for_clip_key(ck: str) -> Optional[str]:
    l_idx, _ = Krea2TensorParser.extract_clip_layer_idx(ck)
    if l_idx is not None:
        return CLIP_IDX_TO_LABEL.get(l_idx)
    if "embed_tokens" in ck or "embeddings" in ck:
        return "Embedding"
    return None


def _apply_recipe_indices(sec_data: Dict[str, Dict[str, Any]], target_map: Dict[str, Any],
                          idx_to_label: Dict[int, str], recipe: Dict[str, Any],
                          mutate) -> None:
    """Resolves a recipe's target dropdown label to sections and applies `mutate` to each."""
    sel = resolve_target_map_entry(target_map, recipe.get("target_block", ""), None)
    if not sel:
        return
    for idx in sel:
        label = idx_to_label.get(idx)
        if label in sec_data:
            mutate(sec_data[label])


def collect_visualizer_sections(patcher: Any, is_clip: bool) -> Tuple[List[Dict[str, Any]], int]:
    """Builds the visualizer graph data for one patcher.

    Single implementation shared by the Model and CLIP visualizers, which had drifted
    apart (only the Model one honoured projection keys, neither read the chaos-rotation
    or 5D recipe families, and both derived the rotation angle from a recipe field that
    the dual rotators never wrote - so every rotation showed up as "angle 0" and no badge
    was ever drawn).

    Provenance is read in three passes, from most to least authoritative:
      1. `arthemy_section_meta` - per-section facts recorded at injection time
         (exact angle, measured relative weight change, 5D flag).
      2. The recipe families - so a preset-loaded or bypassed graph still renders.
      3. The live patch list - for scalar offsets, granular deltas and external LoRAs.
    """
    domain = "clip" if is_clip else "model"
    section_keys = CLIP_SECTION_KEYS if is_clip else MODEL_SECTION_KEYS
    idx_to_label = CLIP_IDX_TO_LABEL if is_clip else MODEL_IDX_TO_LABEL
    target_map = (ArthemyKrea2CLIPBlockSurgeonTuner.CLIP_TARGET_MAP if is_clip
                  else ArthemyKrea2ModelBlockSurgeonTuner.MODEL_TARGET_MAP)
    section_for_key = _section_for_clip_key if is_clip else _section_for_model_key

    sec_data = {s: _new_section_slot(s) for s in section_keys}
    patches = getattr(patcher, "patches", {}) if patcher else {}
    model_options = getattr(patcher, "model_options", {}) if patcher else {}

    def recipes(key: str) -> List[Dict[str, Any]]:
        return [r for r in model_options.get(key, []) if r.get("domain", domain) == domain]

    # ---- Pass 1: authoritative per-section metadata recorded at injection time -------
    meta_labels: Set[str] = set()
    for idx_str, meta in get_section_meta(patcher, domain).items():
        try:
            label = idx_to_label.get(int(idx_str))
        except (TypeError, ValueError):
            label = None
        if label not in sec_data:
            continue
        meta_labels.add(label)
        slot = sec_data[label]
        if meta.get("is_rotation"):
            slot["is_rotation"] = True
            slot["rotation_angle"] = float(meta.get("angle", 0.0))
            slot["rotation_hue"] = float(meta.get("hue", 180.0))
            slot["rotation_type"] = meta.get("rotation_type", "")
            slot["depth_reach"] = meta.get("depth_reach", "Default")
            slot["relative_delta"] = float(meta.get("relative_delta", 0.0))
        if meta.get("is_chaos"):
            slot["is_chaos"] = True
        if meta.get("is_five_d"):
            slot["is_five_d"] = True
            slot["five_d_dims"] = int(meta.get("five_d_dims", 0) or 0)
            slot["five_d_source"] = meta.get("five_d_source", "")

    # ---- Pass 2: recipe families -----------------------------------------------------
    # Only a FALLBACK for sections that pass 1 did not cover (legacy sessions, presets with
    # an inline payload). A recipe only knows the dropdown label, so it would otherwise mark
    # all 28 blocks of "All Blocks (0-27)" even when the modifier touches six of them.
    # Every current code path (including the Preset Loader, which replays through the nodes)
    # records section metadata, so the recipe fallback only ever fires for graphs produced by
    # an older build of the suite.
    use_recipe_fallback = not meta_labels

    for r in (recipes("arthemy_rotation_recipes") + recipes("arthemy_chaos_rotation_recipes")
              if use_recipe_fallback else []):
        angle = float(r.get("rotation_angle", 0.0) or 0.0)
        if angle == 0.0:
            # Dual recipes store four axes; the compass stores one tilt. Fall back to the
            # dominant axis so the HUD always has a number to draw.
            angle = max(abs(float(r.get(k, 0.0) or 0.0))
                        for k in ("structural_x", "structural_y", "tensor_x", "tensor_y",
                                  "chaos_strength", "style_strength")) or 0.0
            if r.get("type") == "chaos_rotation":
                angle *= 90.0
        is_chaos = r.get("type") == "chaos_rotation" or "chaos_strength" in r
        hue = float(r.get("style_direction", r.get("hue", 180.0)) or 180.0)
        reach = str(r.get("depth_reach", "Default"))

        def mutate(slot, angle=angle, hue=hue, reach=reach, is_chaos=is_chaos):
            if slot["block"] in meta_labels:
                return
            slot["is_rotation"] = True
            if slot["rotation_angle"] == 0.0:
                slot["rotation_angle"] = round(angle, 2)
                slot["rotation_hue"] = hue
                slot["depth_reach"] = reach
            if is_chaos:
                slot["is_chaos"] = True

        _apply_recipe_indices(sec_data, target_map, idx_to_label, r, mutate)

    # Block-Level chaos tuning is injected as a scalar multiplier (memory optimization),
    # so its "chaos" nature is only knowable from the recipe.
    for r in recipes("arthemy_chaos_recipes"):
        for idx in r.get("selected_indices", []):
            label = idx_to_label.get(idx)
            if label in sec_data:
                sec_data[label]["is_chaos"] = True

    for r in (recipes("arthemy_5d_recipes") if use_recipe_fallback else []):
        # The bare modifier name, never the scoped alias a Preset Saver may have written:
        # the HUD names what the injection came FROM, not which slice of it travelled.
        def mutate_5d(slot, src=scope_alias_base(str(r.get("source_lora", "")))):
            if slot["block"] in meta_labels:
                return
            slot["is_five_d"] = True
            if not slot["five_d_source"]:
                slot["five_d_source"] = src

        _apply_recipe_indices(sec_data, target_map, idx_to_label, r, mutate_5d)

    # ---- Pass 3: live patch list ----------------------------------------------------
    for pk, patch_list in patches.items():
        ck = Krea2TensorParser.clean_key(pk)
        sec_name = section_for_key(ck)
        if not sec_name or sec_name not in sec_data:
            continue
        slot = sec_data[sec_name]

        for p in patch_list:
            off, is_l, is_c, is_r, r_ang, r_hue, r_reach = parse_patch_entry(p)
            if is_r:
                slot["is_rotation"] = True
                if r_ang != 0.0 and slot["rotation_angle"] == 0.0:
                    slot["rotation_angle"] = r_ang
                    slot["rotation_hue"] = r_hue
                    slot["depth_reach"] = r_reach
            elif is_l:
                # A low-rank adapter that is neither a rotation nor a 5D injection is an
                # actual external LoRA - the only thing that may be drawn in purple.
                if not slot["is_five_d"]:
                    slot["is_lora"] = True
            if is_c:
                slot["is_chaos"] = True
            if not is_l and not is_r:
                slot["offset"] += off
                slot["scalar_count"] += 1

    graph_data: List[Dict[str, Any]] = []
    modified = 0
    for s in section_keys:
        d = sec_data[s]
        if d["scalar_count"] > 0:
            d["offset"] /= max(1, d["scalar_count"])
            modified += 1
        elif d["is_lora"] or d["is_rotation"] or d["is_five_d"]:
            modified += 1
        graph_data.append({
            "block": d["block"],
            "offset": round(d["offset"], 4),
            "is_lora": d["is_lora"],
            "is_chaos": d["is_chaos"],
            "is_rotation": d["is_rotation"],
            "rotation_angle": round(float(d["rotation_angle"]), 2),
            "rotation_hue": round(float(d["rotation_hue"]), 1),
            "rotation_type": d["rotation_type"],
            "depth_reach": d["depth_reach"],
            "relative_delta": round(float(d["relative_delta"]), 6),
            "is_5d": d["is_five_d"],
            "five_d_dims": d["five_d_dims"],
            "five_d_source": d["five_d_source"],
        })
    return graph_data, modified


class BaseVisualizerNode(BaseKrea2Node):
    """Shared implementation of the Model and CLIP HUD visualizers."""

    OUTPUT_NODE = True
    IS_CLIP = False
    TITLE = "Krea-2 Model"
    LOG_TAG = "ARTHEMY MODEL VISUALIZER"

    @classmethod
    def _io_name(cls) -> str:
        return "clip" if cls.IS_CLIP else "model"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                s._io_name(): ("CLIP" if s.IS_CLIP else "MODEL",),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 99.00, "step": 0.1,
                                    "tooltip": "Visual height amplification scale."}),
                "image_width": ("INT", {"default": 960, "min": 256, "max": 3840, "step": 16,
                                        "tooltip": "Exported PNG render width."}),
                "image_height": ("INT", {"default": 480, "min": 128, "max": 2160, "step": 16,
                                         "tooltip": "Exported PNG render height."}),
            }
        }

    CATEGORY = "Arthemy/Visualizers"

    def _visualize(self, obj: Any, scale: float, image_width: int, image_height: int) -> Dict[str, Any]:
        patcher = get_patcher(obj)
        graph_data, modified = collect_visualizer_sections(patcher, self.IS_CLIP)

        # Real architecture of the loaded checkpoint, not the stock Krea-2/Qwen3 baseline
        # this canvas is laid out for. The section slots themselves stay fixed (the suite
        # only targets that one pair of architectures), but the axis can at least say
        # honestly how many of them are real for THIS checkpoint versus unused padding.
        baseline = Krea2Config.MAX_CLIP_LAYERS if self.IS_CLIP else Krea2Config.MAX_UNET_BLOCKS
        real_count = baseline
        try:
            sd = patcher.model.state_dict() if patcher is not None else None
            if sd:
                real_count = Krea2Config.probe(sd, self.IS_CLIP)
        except Exception as e:
            logger.debug(f"[{self.LOG_TAG}] architecture probe failed, assuming baseline ({baseline}): {e}")

        n_rot = sum(1 for g in graph_data if g["is_rotation"])
        n_5d = sum(1 for g in graph_data if g["is_5d"])
        n_lora = sum(1 for g in graph_data if g["is_lora"])
        info = (f"{'CLIP' if self.IS_CLIP else 'Model'} Visualizer: {modified}/{len(graph_data)} sections modified "
                f"| rotations: {n_rot} | 5D: {n_5d} | LoRA: {n_lora}")
        if real_count != baseline:
            noun = "layers" if self.IS_CLIP else "blocks"
            info += f" | ⚠️ {real_count}/{baseline} {noun} detected in this checkpoint"
        logger.info(f"[{self.LOG_TAG}] {info}")

        vis_image = render_visualizer_image(graph_data, self.TITLE, is_clip=self.IS_CLIP,
                                           visual_scale=scale, width=image_width, height=image_height)
        return {
            "ui": {"graph_data": graph_data, "scale": [scale], "title": [self.TITLE],
                   "real_count": [real_count], "baseline_count": [baseline]},
            "result": (obj, vis_image, info),
        }


class ArthemyKrea2ModelVisualizer(BaseVisualizerNode):
    IS_CLIP = False
    TITLE = "Krea-2 Model"
    LOG_TAG = "ARTHEMY MODEL VISUALIZER"
    SECTION_KEYS = MODEL_SECTION_KEYS
    MODEL_TARGET_MAP = ArthemyKrea2ModelBlockSurgeonTuner.MODEL_TARGET_MAP
    RETURN_TYPES = ("MODEL", "IMAGE", "STRING")
    RETURN_NAMES = ("MODEL", "IMAGE", "info")
    FUNCTION = "visualize"

    def visualize(self, model: Any, scale: float = 1.0, image_width: int = 960, image_height: int = 480):
        return self._visualize(model, scale, image_width, image_height)


class ArthemyKrea2CLIPVisualizer(BaseVisualizerNode):
    IS_CLIP = True
    TITLE = "Krea-2 CLIP"
    LOG_TAG = "ARTHEMY CLIP VISUALIZER"
    SECTION_KEYS = CLIP_SECTION_KEYS
    CLIP_TARGET_MAP = ArthemyKrea2CLIPBlockSurgeonTuner.CLIP_TARGET_MAP
    RETURN_TYPES = ("CLIP", "IMAGE", "STRING")
    RETURN_NAMES = ("CLIP", "IMAGE", "info")
    FUNCTION = "visualize"

    def visualize(self, clip: Any, scale: float = 1.0, image_width: int = 960, image_height: int = 480):
        return self._visualize(clip, scale, image_width, image_height)


# ==============================================================================
# PRESET NODES (SAVER & LOADER)
# ==============================================================================

def extract_patch_multipliers(patcher: Any) -> Tuple[Dict[str, float], Dict[str, Any], int, int]:
    """Extracts pure scalar offset multipliers AND granular deltas (sparse channel dict or dense
    vectors) from the active patches of a ModelPatcher.

    Returns (scalar_offsets, granular_patches, lora_patches_count, chaos_patches_count).
    Rotation adapters are intentionally ignored: they are persisted as deterministic recipes,
    so counting them as LoRA tensors would trigger a misleading "LoRAs excluded" warning.
    """
    if patcher is None or not hasattr(patcher, "patches"):
        return {}, {}, 0, 0
    extracted_scalars: Dict[str, float] = {}
    extracted_granular: Dict[str, Any] = {}
    lora_count = 0
    chaos_count = 0

    for k, patch_list in patcher.patches.items():
        if not patch_list:
            continue
        compound_mult = 1.0
        has_scalar = False
        clean_k = Krea2TensorParser.clean_key(k)

        for p in patch_list:
            if not isinstance(p, (tuple, list)) or len(p) == 0:
                continue

            # ComfyUI patch tuple layout: (strength_patch, value, strength_model, ...)
            strength_patch = float(p[0]) if isinstance(p[0], (int, float)) else 1.0
            diff = p[1] if len(p) > 1 else ()

            if isinstance(diff, (tuple, list)) and len(diff) == 2 and isinstance(diff[0], str):
                diff = diff[1]

            while isinstance(diff, (tuple, list)) and len(diff) == 1 and isinstance(diff[0], (tuple, list, torch.Tensor)):
                diff = diff[0]

            # Rotations and 5D injections come first: both are reproduced from deterministic
            # recipes and must never be serialized as tensors, nor counted as external LoRAs
            # (that produced a spurious "N LoRA tensors excluded" warning on every save).
            if getattr(diff, "_is_arthemy_rotation", False) or getattr(diff, "_is_arthemy_five_d", False):
                continue

            # Channel gains are WeightAdapter subclasses too, so they must be recognised
            # BEFORE the external-LoRA test below - `parse_patch_entry` already honours this
            # flag, and the two disagreeing is what made the Saver report a channel tuning as
            # "N LoRA tensors excluded" and then drop it.
            if getattr(diff, "_is_arthemy_granular", False):
                continue

            if is_weight_adapter(diff) or hasattr(diff, "lora_a"):
                lora_count += 1
                continue

            if getattr(diff, "_is_arthemy_chaos", False):
                chaos_count += 1
                continue

            # Granular tensor diffs (from granular_json or vectors_override)
            if isinstance(diff, torch.Tensor) and diff.numel() > 1 and strength_patch != 0.0:
                # Detect 1D broadcast across 2D matrix rows or columns to avoid 90MB JSON blowup
                t_vec = None
                if diff.ndim == 2:
                    if diff.shape[0] > 1 and torch.all(diff == diff[0:1, :]):
                        t_vec = diff[0, :]
                    elif diff.shape[1] > 1 and torch.all(diff == diff[:, 0:1]):
                        t_vec = diff[:, 0]
                elif diff.ndim == 1:
                    t_vec = diff

                eff_tensor = t_vec if t_vec is not None else diff
                flat = eff_tensor.detach().flatten().to(torch.float32)
                non_zero = torch.nonzero(flat).squeeze(-1)
                total_elements = flat.numel()
                nz_count = int(non_zero.numel())

                if nz_count == 0:
                    continue
                if nz_count <= 256 or nz_count < (0.20 * total_elements):
                    # Sparse representation: accumulate additions onto matching channel indices
                    idx_list = non_zero.tolist()
                    val_list = (flat[non_zero] * strength_patch).tolist()
                    ch_map = {str(i): round(v, 6) for i, v in zip(idx_list, val_list) if round(v, 6) != 0.0}
                    if ch_map:
                        current = extracted_granular.get(clean_k)
                        if not isinstance(current, dict):
                            if isinstance(current, list):
                                for i, v in enumerate(current):
                                    if v != 0.0:
                                        ch_map[str(i)] = round(ch_map.get(str(i), 0.0) + v, 6)
                            current = {}
                            extracted_granular[clean_k] = current
                        for idx_str, delta_val in ch_map.items():
                            current[idx_str] = round(current.get(idx_str, 0.0) + delta_val, 6)
                else:
                    # Dense vector representation: sum with existing entries if present
                    new_dense = [round(x, 6) for x in (flat * strength_patch).tolist()]
                    current = extracted_granular.get(clean_k)
                    if isinstance(current, list) and len(current) == len(new_dense):
                        extracted_granular[clean_k] = [round(a + b, 6) for a, b in zip(current, new_dense)]
                    elif isinstance(current, dict):
                        for str_i, val in current.items():
                            try:
                                i = int(str_i)
                                if i < len(new_dense):
                                    new_dense[i] = round(new_dense[i] + val, 6)
                            except ValueError:
                                pass
                        extracted_granular[clean_k] = new_dense
                    else:
                        extracted_granular[clean_k] = new_dense
            else:
                off, is_lora, is_c, *_ = parse_patch_entry(p)
                if is_lora:
                    lora_count += 1
                elif is_c:
                    chaos_count += 1
                else:
                    compound_mult *= (1.0 + off)
                    has_scalar = True

        if has_scalar and round(compound_mult - 1.0, 6) != 0.0:
            extracted_scalars[clean_k] = round(compound_mult - 1.0, 6)

    return extracted_scalars, extracted_granular, lora_count, chaos_count


class ArthemyKrea2PresetSaver(BaseKrea2Node):
    OUTPUT_NODE = True  # Ensures ComfyUI always executes the save even if outputs are unconnected

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "preset_name": ("STRING", {"default": "my_arthemy_preset", "tooltip": "Name of the preset file to save (will append .json automatically)."}),
            },
            "optional": {
                "subfolder_or_path": ("STRING", {"default": "", "tooltip": "Optional subfolder inside custom_nodes/Arthemy_Krea2_Tuner/presets/ (e.g. 'cinematic') OR a custom full folder path. If empty, saves to presets/ directly."}),
                "author": ("STRING", {"default": "Arthemy", "tooltip": "Author metadata written into the preset."}),
                "mirror_to_custom_nodes": ("BOOLEAN", {"default": False, "tooltip": "Legacy option (no-op: presets now save directly into custom_nodes/Arthemy_Krea2_Tuner/presets)."}),
                "embed_5d_modifiers": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Write the 5D modifiers themselves into the preset, so it reproduces "
                               "on any machine without the original .json or LoRA. Turn off for a "
                               "small preset that only references them by name (it will then only "
                               "reproduce where those files exist, unchanged)."}),
                "prune_5d_to_target": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Embed only the part of each 5D modifier its recipes can actually "
                               "reach. A 5D injection aimed at one block otherwise carries every "
                               "block of the model AND every CLIP layer into the preset, none of "
                               "which is ever synthesised on load. Turn this off to embed the "
                               "modifiers whole, e.g. to keep re-targeting them by hand after "
                               "loading the preset without the original files present."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "saved_path", "info")
    FUNCTION = "save_preset"
    CATEGORY = "Arthemy/Presets"

    def save_preset(self, model: Any, clip: Any, preset_name: str = "my_arthemy_preset", subfolder_or_path: str = "", author: str = "Arthemy", mirror_to_custom_nodes: bool = False, embed_5d_modifiers: bool = True, prune_5d_to_target: bool = True) -> Tuple[Any, Any, str, str]:
        m_p = get_patcher(model)
        c_p = get_patcher(clip)

        model_patches, model_granular, m_lora, m_chaos = extract_patch_multipliers(m_p)
        clip_patches, clip_granular, c_lora, c_chaos = extract_patch_multipliers(c_p)

        # Collect deterministic Chaos & Rotation recipes
        # (single accessor shared by every recipe family)
        def get_recipes(p, key):
            if p is None or not hasattr(p, "model_options"):
                return []
            return list(p.model_options.get(key, []))

        combined_chaos_recipes = get_recipes(m_p, "arthemy_chaos_recipes") + get_recipes(c_p, "arthemy_chaos_recipes")
        combined_rotation_recipes = get_recipes(m_p, "arthemy_rotation_recipes") + get_recipes(c_p, "arthemy_rotation_recipes")
        combined_chaos_rot_recipes = get_recipes(m_p, "arthemy_chaos_rotation_recipes") + get_recipes(c_p, "arthemy_chaos_rotation_recipes")
        combined_channel_recipes = get_recipes(m_p, "arthemy_channel_recipes")

        # 5D recipes stay slim - source reference plus dimension weights - because the payload
        # used to be embedded per recipe, which duplicated it once per entry AND got deep-copied
        # on every patcher clone downstream. The payloads instead go into ONE deduplicated
        # top-level map (see below), which costs nothing at runtime: the recipes carried on the
        # patcher never hold it, only the file on disk does.
        m_5d = []
        for src_patcher, default_domain in ((m_p, "model"), (c_p, "clip")):
            for entry in get_recipes(src_patcher, "arthemy_5d_recipes"):
                slim = {k: v for k, v in entry.items() if k != "dct_data"}
                slim.setdefault("domain", default_domain)
                m_5d.append(slim)

        # Modifiers travel WITH the preset unless the user opts out. A preset that only names
        # its modifier is half a recipe: replay it on another machine, or after the modifier
        # has been re-extracted or deleted, and it silently produces something else.
        embedded_modifiers: Dict[str, Any] = {}
        embed_skipped: List[str] = []
        prune_notes: List[str] = []
        if embed_5d_modifiers and m_5d:
            # Every scope a source is used at, in recipe order (a set here made the embedded
            # order - and so the preset's bytes - differ between two saves of the same graph).
            scopes_by_src: "collections.OrderedDict[str, List[Tuple[str, str, str, str]]]" = collections.OrderedDict()
            for fd in m_5d:
                src = str(fd.get("source_lora", "")).strip()
                if not src:
                    continue
                scope = (str(fd.get("domain", "model")),
                         str(fd.get("target_block", "")),
                         str(fd.get("sub_components", "All Components")),
                         str(fd.get("sub_tensor", SUB_TENSOR_ANY)))
                entry = scopes_by_src.setdefault(src, [])
                if scope not in entry:
                    entry.append(scope)

            for src, src_scopes in scopes_by_src.items():
                try:
                    payload = get_cached_5d_dct(src)
                except Exception as e:
                    payload = None
                    logger.warning(f"[ARTHEMY PRESET SAVER] Could not resolve '{src}' to embed: {e}")
                if not (isinstance(payload, dict) and payload.get("layers")):
                    embed_skipped.append(src)
                    continue

                store_name = src
                if prune_5d_to_target:
                    pruned, stats = prune_5d_payload_to_scopes(payload, src_scopes)
                    if pruned is not payload:
                        # The pruned payload is a different modifier from the one on disk, so it
                        # travels under a scoped alias and the recipes are re-pointed at it. The
                        # bare name is left alone: get_cached_5d_dct falls back to it if this
                        # preset is ever replayed without its payload.
                        store_name = f"{src}{SCOPE_ALIAS_SEP}{stats['tag']}"
                        for fd in m_5d:
                            if str(fd.get("source_lora", "")).strip() == src:
                                fd["source_lora"] = store_name
                        payload = pruned
                        prune_notes.append(f"{os.path.basename(src)} {stats['kept']}/{stats['total']} layers")
                embedded_modifiers[store_name] = payload

        total_lora = m_lora + c_lora
        if total_lora > 0:
            logger.warning(f"[ARTHEMY PRESET SAVER] Warning: {total_lora} active LoRA patch tensors detected. "
                           "LoRAs are excluded from lightweight presets. Use Model Saver/Baker to fuse LoRA weights permanently.")

        if not any((model_patches, clip_patches, model_granular, clip_granular,
                    combined_chaos_recipes, combined_rotation_recipes,
                    combined_chaos_rot_recipes, combined_channel_recipes, m_5d)):
            logger.warning("[ARTHEMY PRESET SAVER] No active tunings, rotations, or chaos recipes detected on Model or CLIP.")

        # Strict basename sanitization against path-traversal enforcing .json
        raw_name = os.path.basename(preset_name.strip())
        base_name, _ = os.path.splitext(raw_name)
        safe_base = re.sub(r'[^\w\-_\.]', '_', base_name)
        if not safe_base:
            safe_base = "arthemy_preset"
        safe_name = f"{safe_base}.json"

        # Determine target directory safely using os.path normalization
        clean_sub = subfolder_or_path.strip()
        if clean_sub and not any(clean_sub.startswith(x) for x in ["presets", "models/arthemy_presets", "custom_nodes/presets", "Default", "default"]):
            if os.path.isabs(clean_sub):
                save_dir = os.path.normpath(clean_sub)
            else:
                save_dir = os.path.normpath(os.path.join(arthemy_presets_dir, clean_sub))
        else:
            save_dir = arthemy_presets_dir

        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, safe_name)
        if os.path.exists(file_path):
            logger.info(f"[ARTHEMY PRESET SAVER] Overwriting existing preset file at: {file_path}")

        preset_data = {
            "name": safe_base,
            "author": author.strip() if author else "Arthemy",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": "2.1",
            # Presets written before rotations actually applied carry no marker. The loader
            # uses its absence to warn instead of quietly firing a rotation the author never
            # actually saw take effect.
            "suite_rotation_effective": True,
            "stats": {
                "model_patched_layers": len(model_patches),
                "model_granular_layers": len(model_granular),
                "clip_patched_layers": len(clip_patches),
                "clip_granular_layers": len(clip_granular),
                "chaos_recipes_count": len(combined_chaos_recipes),
                "rotation_recipes_count": len(combined_rotation_recipes),
                "chaos_rotation_recipes_count": len(combined_chaos_rot_recipes),
                "channel_recipes_count": len(combined_channel_recipes),
                "five_d_recipes_count": len(m_5d),
                "excluded_lora_tensors": total_lora,
            },
            "model_patches": model_patches,
            "model_granular_patches": model_granular,
            "clip_patches": clip_patches,
            "clip_granular_patches": clip_granular,
            "chaos_recipes": combined_chaos_recipes,
            "rotation_recipes": combined_rotation_recipes,
            "chaos_rotation_recipes": combined_chaos_rot_recipes,
            "channel_recipes": combined_channel_recipes,
            "five_d_recipes": m_5d,
            # name -> payload, deduplicated. Absent when nothing was embedded, so a slim
            # preset stays byte-identical to what previous versions wrote.
            **({"embedded_modifiers": embedded_modifiers} if embedded_modifiers else {}),
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(preset_data, f, indent=2, ensure_ascii=False)

        # Optional mirror copy to custom_nodes/presets (only when save_dir is a different external path)
        if mirror_to_custom_nodes and os.path.abspath(save_dir) != os.path.abspath(local_presets_dir):
            try:
                os.makedirs(local_presets_dir, exist_ok=True)
                local_file = os.path.join(local_presets_dir, safe_name)
                with open(local_file, "w", encoding="utf-8") as f:
                    json.dump(preset_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"[ARTHEMY PRESET SAVER] Failed to mirror preset copy to {local_presets_dir}: {e}")

        info = (f"Preset saved: '{safe_name}' ({len(model_patches)} model, {len(model_granular)} "
                f"model-granular, {len(clip_patches)} clip, {len(combined_chaos_recipes)} chaos, "
                f"{len(combined_rotation_recipes)} rotation, {len(combined_channel_recipes)} channel)")
        if embedded_modifiers:
            size_mb = os.path.getsize(file_path) / (1024.0 * 1024.0)
            info += (f" | {len(embedded_modifiers)} 5D modifier(s) embedded, self-contained "
                     f"({size_mb:.1f} MB)")
            if prune_notes:
                info += f" | scoped to target: {', '.join(prune_notes)}"
            if size_mb > 25.0:
                logger.warning(f"[ARTHEMY PRESET SAVER] '{safe_name}' is {size_mb:.1f} MB because it "
                               "carries its modifiers in full. Re-extract the source LoRA with "
                               "storage='auto' if a smaller file matters, or turn off "
                               "embed_5d_modifiers to reference them by name instead.")
        if embed_skipped:
            info += f" | ⚠️ could not embed: {', '.join(embed_skipped)}"
            logger.warning(f"[ARTHEMY PRESET SAVER] These 5D sources could not be resolved and are "
                           f"referenced by name only: {', '.join(embed_skipped)}. The preset will "
                           "need them present to reproduce.")
        if total_lora > 0:
            info += f" | ⚠️ {total_lora} LoRA tensors excluded"
        logger.info(f"[ARTHEMY PRESET SAVER] {info} -> {file_path}")
        return {"ui": {"text": [info]}, "result": (model, clip, file_path, info)}


class ArthemyKrea2PresetLoader(BaseKrea2Node):
    @classmethod
    def INPUT_TYPES(s):
        preset_files = folder_paths.get_filename_list("arthemy_presets")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "preset": (sorted(preset_files) if preset_files else ["None"], {"tooltip": "Select a preset from custom_nodes/Arthemy_Krea2_Tuner/presets."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -99.00, "max": 99.00, "step": 0.01,
                                             "tooltip": "Global multiplier for model patches in this preset."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -99.00, "max": 99.00, "step": 0.01,
                                            "tooltip": "Global multiplier for CLIP patches in this preset."}),
            },
            "optional": {
                "custom_path": ("STRING", {"default": "", "tooltip": "Direct path to a .json preset file OR a custom folder containing presets (if set, overrides the dropdown selection)."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "info")
    FUNCTION = "load_preset"
    CATEGORY = "Arthemy/Presets"

    def load_preset(self, model: Any, clip: Any, preset: str = "None", strength_model: float = 1.0, strength_clip: float = 1.0, custom_path: str = "") -> Tuple[Any, Any, str]:
        file_path = None

        if custom_path.strip():
            raw_path = custom_path.strip()
            abs_path = os.path.abspath(raw_path)
            if os.path.isfile(abs_path):
                file_path = abs_path
            elif os.path.isdir(abs_path):
                target_file = preset if preset.endswith(".json") else f"{preset}.json"
                candidate = os.path.join(abs_path, target_file)
                if os.path.isfile(candidate):
                    file_path = candidate
                else:
                    json_files = sorted(f for f in os.listdir(abs_path) if f.endswith(".json"))
                    if json_files:
                        file_path = os.path.join(abs_path, json_files[0])
                        logger.warning(f"[ARTHEMY PRESET LOADER] '{target_file}' not found in '{abs_path}'; "
                                       f"falling back to the first preset in alphabetical order: {json_files[0]}")
                    else:
                        raise FileNotFoundError(f"Arthemy Suite Error: No .json presets found in custom folder '{abs_path}'.")
            else:
                raise FileNotFoundError(f"Arthemy Suite Error: Custom path '{raw_path}' does not exist.")

        if not file_path:
            if not preset or preset == "None":
                return (model, clip, "No preset selected.")

            file_path = folder_paths.get_full_path("arthemy_presets", preset)
            if not file_path or not os.path.exists(file_path):
                # Direct search across registered directories
                for p_dir in folder_paths.get_folder_paths("arthemy_presets"):
                    candidate = os.path.join(p_dir, preset if preset.endswith(".json") else f"{preset}.json")
                    if os.path.isfile(candidate):
                        file_path = candidate
                        break

            if not file_path or not os.path.exists(file_path):
                raise FileNotFoundError(f"Arthemy Suite Error: Preset '{preset}' not found in any registered preset folders.")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        m = model.clone() if model is not None else None
        c = clip.clone() if clip is not None else None
        m_p = get_patcher(m)
        c_p = get_patcher(c)

        m_sd = m_p.model.state_dict() if (m_p and hasattr(m_p, 'model')) else {}
        c_sd = c_p.model.state_dict() if (c_p and hasattr(c_p, 'model')) else {}

        model_patches = data.get("model_patches", {})
        model_granular = data.get("model_granular_patches", {})
        clip_patches = data.get("clip_patches", {})
        clip_granular = data.get("clip_granular_patches", {})
        chaos_recipes = data.get("chaos_recipes", [])

        n_model_matched = 0
        n_model_unmatched = 0
        n_clip_matched = 0
        n_clip_unmatched = 0
        n_granular_applied = 0

        def build_granular_from_preset(g_val: Any, base_w: torch.Tensor, strength: float) -> Optional[Tuple[torch.Tensor]]:
            """Rebuilds a sparse channel dict or a dense vector stored in a preset."""
            if isinstance(g_val, dict):
                diff = torch.zeros(base_w.shape, dtype=torch.bfloat16, device="cpu")
                flat = diff.flatten()
                applied_ch = False
                for ch_k, ch_v in g_val.items():
                    if str(ch_k).isdigit():
                        ch_idx = int(ch_k)
                        if 0 <= ch_idx < flat.numel():
                            flat[ch_idx] = float(ch_v) * strength
                            applied_ch = True
                return (diff,) if applied_ch else None
            if isinstance(g_val, (list, tuple)):
                try:
                    val_list = [float(x) * strength for x in g_val]
                except (TypeError, ValueError):
                    return None
                if len(val_list) == base_w.numel():
                    return (torch.tensor(val_list, dtype=torch.bfloat16, device="cpu").view_as(base_w),)
                logger.warning(f"[ARTHEMY PRESET LOADER] Granular vector size mismatch "
                               f"({len(val_list)} values for shape {tuple(base_w.shape)}); entry skipped.")
            return None

        # Every tuner, surgeon, rotator and the channel tuner refuse FP8 weights that carry a
        # companion scale, because a multiplier applied there lands BEFORE the scale and the
        # layer ends up scaled twice. The Preset Loader did not, so a preset saved from a bf16
        # checkpoint and replayed on a scaled-FP8 one blew out exactly those layers, silently -
        # and the same graph built by hand would have warned. One rule, one place.
        n_model_quant_skipped = n_clip_quant_skipped = 0

        # 1. Apply scalar model patches with state-dict key resolution
        if model_patches and strength_model != 0.0:
            m_patches_to_add = {}
            for k, off in model_patches.items():
                target_k = resolve_target_key(m_p, k, model_sd=m_sd)
                if target_k in m_sd and is_skippable_for_tuning(target_k, m_sd):
                    n_model_quant_skipped += 1
                elif target_k in m_sd:
                    scaled_off = off * strength_model
                    if scaled_off != 0.0:
                        m_patches_to_add[target_k] = (1.0 + scaled_off,)
                        n_model_matched += 1
                else:
                    n_model_unmatched += 1
            if m_patches_to_add:
                inject_patches(m, m_patches_to_add, 1.0)

        # 2. Apply granular model patches (sparse channel deltas or dense vectors)
        if model_granular and strength_model != 0.0:
            m_granular_to_add = {}
            for k, g_val in model_granular.items():
                target_k = resolve_target_key(m_p, k, model_sd=m_sd)
                if target_k in m_sd and is_skippable_for_tuning(target_k, m_sd):
                    n_model_quant_skipped += 1
                elif target_k in m_sd:
                    payload = build_granular_from_preset(g_val, m_sd[target_k], strength_model)
                    if payload is not None:
                        m_granular_to_add[target_k] = payload
                        n_granular_applied += 1
            if m_granular_to_add:
                inject_patches(m, m_granular_to_add, 1.0)

        # 3. Apply scalar CLIP patches with state-dict key resolution
        if clip_patches and strength_clip != 0.0:
            c_patches_to_add = {}
            for k, off in clip_patches.items():
                target_k = resolve_target_key(c_p, k, model_sd=c_sd)
                if target_k in c_sd and is_skippable_for_tuning(target_k, c_sd):
                    n_clip_quant_skipped += 1
                elif target_k in c_sd:
                    scaled_off = off * strength_clip
                    if scaled_off != 0.0:
                        c_patches_to_add[target_k] = (1.0 + scaled_off,)
                        n_clip_matched += 1
                else:
                    n_clip_unmatched += 1
            if c_patches_to_add:
                inject_patches(c, c_patches_to_add, 1.0)

        # 4. Apply granular CLIP patches
        if clip_granular and strength_clip != 0.0:
            c_granular_to_add = {}
            for k, g_val in clip_granular.items():
                target_k = resolve_target_key(c_p, k, model_sd=c_sd)
                if target_k in c_sd and is_skippable_for_tuning(target_k, c_sd):
                    n_clip_quant_skipped += 1
                elif target_k in c_sd:
                    payload = build_granular_from_preset(g_val, c_sd[target_k], strength_clip)
                    if payload is not None:
                        c_granular_to_add[target_k] = payload
                        n_granular_applied += 1
            if c_granular_to_add:
                inject_patches(c, c_granular_to_add, 1.0)

        # 5. Deterministically regenerate Chaos recipes with memoized base_sd
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
                    kwargs=chances, chaos_params=chaos_p, base_sd=m_sd
                )
                n_chaos_applied += 1
            elif domain == "clip" and strength_clip != 0.0:
                effective_strength = base_strength * strength_clip
                chaos_p = {"tune_mode": tune_mode, "seed": seed, "chaos_strength": effective_strength}
                c, _ = BaseSurgeonTuner()._execute_surgeon_tuning(
                    c, is_clip=True, selected_indices=selected_indices,
                    surgeon_map=Krea2TensorParser.CLIP_SURGEON_MAP,
                    kwargs=chances, chaos_params=chaos_p, base_sd=c_sd
                )
                n_chaos_applied += 1

        # 6. Deterministically regenerate Rotation recipes.
        #    Recipes carry a "type": style-compass recipes (Latent Space Rotator) and dual
        #    Lie-rotation recipes (Model/CLIP Axis Rotator) need different reproducers. Without this
        #    dispatch every dual recipe was replayed as a 15 deg style-compass rotation.
        rotation_recipes = data.get("rotation_recipes", [])
        n_rotation_applied = 0

        # MIGRATION GUARD. Until the patch-format fix, every rotator produced a
        # ("lora", ...) payload that ComfyUI discarded with a log warning, so a preset
        # saved back then recorded rotation angles whose effect its author never saw.
        # Replaying them now applies a real rotation - which on a 16-rank subspace can
        # move the weights by tens of percent. Say so loudly rather than silently
        # changing what an old preset does.
        if rotation_recipes and not data.get("suite_rotation_effective", False):
            live_angles = []
            for _r in rotation_recipes:
                _a = max(abs(float(_r.get(k, 0.0) or 0.0)) for k in
                         ("rotation_angle", "structural_x", "structural_y", "tensor_x", "tensor_y"))
                if _a > 0.0:
                    live_angles.append(_a)
            if live_angles:
                logger.warning(
                    f"[ARTHEMY PRESET LOADER] '{os.path.basename(file_path)}' was saved before "
                    f"rotations were actually applied (no 'suite_rotation_effective' marker) and "
                    f"contains {len(live_angles)} rotation recipe(s) up to {max(live_angles):.1f} deg. "
                    "Those rotations were silently discarded when the preset was made, and WILL now "
                    "take effect. If the preset was tuned to look right without them, zero the angles "
                    "or re-save it.")

        # MIGRATION GUARD (signed angles). Angles are now folded into (-180, 180] before
        # the generator is built, because the four axes share plane indices and expm of the
        # accumulated generator is NOT 360-periodic: a recipe holding 355 on two axes used
        # to produce a rotation an order of magnitude larger than the small nudge 355 (=-5)
        # is meant to be. Replaying it now gives that small nudge instead - the intended
        # meaning, but a different picture from the one the preset was saved with.
        if rotation_recipes:
            wrapped = []
            for _r in rotation_recipes:
                axes = [float(_r.get(k, 0.0) or 0.0) for k in
                        ("structural_x", "structural_y", "tensor_x", "tensor_y")]
                # Only multi-axis recipes changed meaning; on a single axis 355 and -5 were
                # always the same rotation, so those replay exactly as before.
                if sum(1 for a in axes if a) > 1 and any(abs(a) > 180.0 for a in axes):
                    wrapped.append(max(abs(a) for a in axes))
            if wrapped:
                logger.warning(
                    f"[ARTHEMY PRESET LOADER] '{os.path.basename(file_path)}' has "
                    f"{len(wrapped)} multi-axis rotation recipe(s) with angles beyond 180 deg "
                    f"(up to {max(wrapped):.1f}). Angles are now read as signed (-180..180), so "
                    "355 means -5 rather than a full turn's worth of accumulated generator. The "
                    "rotation will be gentler than when this preset was saved; re-save it to "
                    "lock in the new reading.")

        for r in rotation_recipes:
            try:
                domain = r.get("domain", "model")
                target_block = r.get("target_block", "Block_1 (All 0-4)")
                sub_comp = r.get("sub_components", "All Components")
                depth_r = normalize_depth_reach(r.get("depth_reach", "Default"))
                recipe_type = r.get("type")
                if recipe_type is None:
                    # Legacy presets: infer from the stored fields
                    recipe_type = "dual_rotation" if any(
                        key in r for key in ("structural_x", "structural_y", "tensor_x", "tensor_y")
                    ) else "style_compass"

                eff_strength = strength_model if domain == "model" else strength_clip
                if eff_strength == 0.0:
                    continue

                if recipe_type == "dual_rotation":
                    s_x = float(r.get("structural_x") or 0.0) * eff_strength
                    s_y = float(r.get("structural_y") or 0.0) * eff_strength
                    t_x = float(r.get("tensor_x") or 0.0) * eff_strength
                    t_y = float(r.get("tensor_y") or 0.0) * eff_strength
                    if domain == "model":
                        m, _ = ArthemyKrea2ModelRotator().rotate_model(
                            m, target_block=target_block, sub_components=sub_comp, depth_reach=depth_r,
                            structural_rot_x=s_x, structural_rot_y=s_y,
                            tensor_rot_x=t_x, tensor_rot_y=t_y
                        )
                    else:
                        c, _ = ArthemyKrea2CLIPRotator().rotate_clip(
                            c, target_layer=target_block, sub_components=sub_comp, depth_reach=depth_r,
                            structural_rot_x=s_x, structural_rot_y=s_y,
                            tensor_rot_x=t_x, tensor_rot_y=t_y
                        )
                    n_rotation_applied += 1
                    continue

                # Compass Rotator (continuous 2D polar) rotation
                style_dir = _first_number(r.get("style_direction"), r.get("hue"), default=180.0)
                style_ang = float(r.get("rotation_angle") or r.get("style_strength") or 15.0)
                rot_seed = int(_first_number(r.get("seed"), default=42))
                manifold = r.get("manifold", ROTATION_MODE_OUTPUT)

                eff_ang = style_ang * eff_strength
                if domain == "model":
                    m, _ = ArthemyKrea2LatentSpaceRotator().rotate_latent_space(
                        m, target_block=target_block, sub_components=sub_comp, depth_reach=depth_r,
                        style_direction=style_dir, rotation_angle=eff_ang, seed=rot_seed, manifold=manifold
                    )
                else:
                    c, _ = ArthemyKrea2CLIPSpaceRotator().rotate_clip_space(
                        c, target_layer=target_block, sub_components=sub_comp, depth_reach=depth_r,
                        style_direction=style_dir, rotation_angle=eff_ang, seed=rot_seed, manifold=manifold
                    )
                n_rotation_applied += 1
            except (ValueError, TypeError) as e:
                logger.warning(f"[ARTHEMY PRESET LOADER] Skipping malformed rotation recipe: {e}")
                continue

        # 7. Replay Channel Magnitude recipes. Placed after the scalar and granular patches
        #    and before the rotations because a channel gain is a MULTIPLICATION, like those,
        #    while rotations and 5D are additive: two multiplications commute, a multiplication
        #    and an addition do not, so this is the position that reproduces the usual graph.
        n_channel_applied = 0
        for chr_ in data.get("channel_recipes", []) or []:
            try:
                if strength_model == 0.0:
                    continue
                bands = chr_.get("bands") or {}
                kw = {name: float(v) * strength_model for name, v in bands.items()
                      if name in ArthemyChannelMagnitudeTuner.BAND_NAMES}
                if not kw:
                    continue
                m, _ = ArthemyChannelMagnitudeTuner().tune_magnitude(
                    m,
                    mode=str(chr_.get("mode", "Soft Value")),
                    channel_path=str(chr_.get("channel_path", CHANNEL_PATH_BOTH)),
                    channel_scope=str(chr_.get("channel_scope", CHANNEL_SCOPE_DEFAULT)),
                    **kw)
                n_channel_applied += 1
            except Exception as e:
                logger.warning(f"[ARTHEMY PRESET LOADER] Skipping channel magnitude recipe: {e!r}")
                continue

        # 8. Apply Chaos Rotation recipes (before 5D, so all rotations sit together).
        chaos_rot_recipes = data.get("chaos_rotation_recipes", [])
        n_chaos_rot_applied = 0
        for cr in chaos_rot_recipes:
            try:
                c_domain = cr.get("domain", "model")
                c_tb = cr.get("target_block", "Block_1 (All 0-4)")
                c_sub = cr.get("sub_components", "All Components")
                c_depth = normalize_depth_reach(cr.get("depth_reach", "Default"))
                c_seed = int(_first_number(cr.get("seed"), default=42))
                c_str = float(cr.get("chaos_strength") or 0.35)
                c_coh = _first_number(cr.get("harmonic_coherence"), default=0.5)
                if c_domain == "model" and strength_model != 0.0:
                    m, _ = ArthemyKrea2ModelChaosRotator().chaos_rotate_model(
                        m, target_block=c_tb, sub_components=c_sub, depth_reach=c_depth,
                        seed=c_seed, chaos_strength=c_str * strength_model, harmonic_coherence=c_coh
                    )
                    n_chaos_rot_applied += 1
                elif c_domain == "clip" and strength_clip != 0.0:
                    c, _ = ArthemyKrea2CLIPChaosRotator().chaos_rotate_clip(
                        c, target_layer=c_tb, sub_components=c_sub, depth_reach=c_depth,
                        seed=c_seed, chaos_strength=c_str * strength_clip, harmonic_coherence=c_coh
                    )
                    n_chaos_rot_applied += 1
            except Exception as e:
                # Deliberately broad. A recipe family is replayed by CALLING a node, so a
                # rename inside the suite surfaces here as NameError / AttributeError, which
                # (ValueError, TypeError) does not catch - and an uncaught error aborts the
                # whole loader, discarding every recipe already applied above it. One bad
                # recipe must cost that recipe, never the preset.
                logger.warning(f"[ARTHEMY PRESET LOADER] Skipping chaos rotation recipe: {e!r}")
                continue

        # 9. Apply 5D Harmonic Tuner recipes.
        #    Slim recipes only reference the source modifier; the DCT payload is rebuilt through
        #    the shared cache. Replaying via the node itself keeps block/sub-component targeting
        #    and section provenance intact.
        five_d_recipes = data.get("five_d_recipes", [])
        n_5d_applied = 0

        # Register anything the preset carries BEFORE the recipes run, so the ordinary replay
        # path resolves it by name. Two shapes are accepted:
        #   - "embedded_modifiers": {name: payload}   - what the saver writes now
        #   - a per-recipe "dct_data"                 - what older presets carry
        # Routing both through the same registry means a legacy preset now honours its own
        # target_block, which the old direct-synthesis path silently ignored.
        n_embedded = register_embedded_modifiers(
            data.get("embedded_modifiers"), origin=f"preset '{os.path.basename(file_path)}'")
        legacy_inline = {}
        for fd in five_d_recipes:
            payload = fd.get("dct_data")
            src = str(fd.get("source_lora", "")).strip()
            if isinstance(payload, dict) and payload.get("layers"):
                legacy_inline[src or f"__inline_{len(legacy_inline)}__"] = payload
                if not src:
                    fd["source_lora"] = f"__inline_{len(legacy_inline) - 1}__"
        if legacy_inline:
            n_embedded += register_embedded_modifiers(legacy_inline, origin="a legacy inline preset")
            logger.info("[ARTHEMY PRESET LOADER] This preset carries its 5D payload inline in the "
                        "old per-recipe format. It is now replayed through the tuner node, so its "
                        "target_block / sub_components selection is honoured - the previous code "
                        "path applied it to every layer of the domain regardless.")

        # MIGRATION GUARD (5D). A DCT-mode modifier may carry a harmonic order too low to
        # be more than noise on its own tensor sizes (see dct_fidelity in the geometry
        # engine) - from before the extractor could warn about this, or from an explicit
        # override of the 'auto' default. Say so before injecting it, the same way the
        # rotation guard above warns about pre-fix presets rather than silently changing
        # what loading this preset does.
        if five_d_recipes and dct_fidelity is not None:
            low_fidelity_sources = []
            seen_sources = set()
            for fd in five_d_recipes:
                src = fd.get("source_lora", "")
                dct_payload = fd.get("dct_data") if fd.get("dct_data") is not None else None
                cache_key = src or f"<inline:{id(dct_payload)}>"
                if cache_key in seen_sources:
                    continue
                seen_sources.add(cache_key)
                if dct_payload is None and src:
                    dct_payload = get_cached_5d_dct(src)
                # payload_is_dct also recognises legacy v3 modifiers, which carry no
                # top-level "storage" key at all - they are the whole reason this guard
                # exists, so keying off that field alone would skip them.
                worst = payload_worst_fidelity(dct_payload)
                if worst is not None and worst < AUTO_STORAGE_FIDELITY_FLOOR:
                    low_fidelity_sources.append((src or "(inline)", worst))
            if low_fidelity_sources:
                detail = ", ".join(f"'{s}' (~{f * 100:.1f}%)" for s, f in low_fidelity_sources)
                logger.warning(
                    f"[ARTHEMY PRESET LOADER] '{os.path.basename(file_path)}' references "
                    f"{len(low_fidelity_sources)} 5D modifier(s) stored as harmonic DCT with fidelity "
                    f"below the {int(AUTO_STORAGE_FIDELITY_FLOOR * 100)}% floor: {detail}. The injected "
                    "direction will be mostly unrelated to the original LoRA/finetune. Re-extract with "
                    "storage='auto' (or 'raw factors') for an exact copy.")

        for fd in five_d_recipes:
            try:
                fd_domain = fd.get("domain", "model")
                eff = strength_model if fd_domain == "model" else strength_clip
                if eff == 0.0:
                    continue
                raw_dw = fd.get("dimension_weights")
                dim_w = list(raw_dw) if isinstance(raw_dw, (list, tuple)) else [0.0] * 5
                dim_w = [float(x) if x is not None and str(x).strip() != "" else 0.0 for x in dim_w]
                raw_mm = fd.get("master_multiplier", 1.0)
                master_mult = float(raw_mm if raw_mm is not None and str(raw_mm).strip() != "" else 1.0) * eff
                source_lora = fd.get("source_lora", "")
                target_lbl = fd.get("target_block", "All Layers (0-59)" if fd_domain == "clip" else "All Blocks (0-27)")
                sub_c = fd.get("sub_components", "All Components")
                sub_t = fd.get("sub_tensor", SUB_TENSOR_ANY)

                # Every source - on disk, embedded by this preset, or carried inline by an old one -
                # now resolves through the same lookup and replays through the tuner node, so all of
                # them honour target_block / sub_components identically.
                if get_cached_5d_dct(source_lora) is None:
                    logger.warning(f"[ARTHEMY PRESET LOADER] 5D recipe skipped: modifier / LoRA "
                                   f"'{source_lora}' is not available, and this preset does not carry "
                                   "it inline. Re-save the preset with embed_5d_modifiers enabled to "
                                   "make it self-contained.")
                    continue

                kw = {f"Dim_{i + 1:02d}": (dim_w[i] if i < len(dim_w) else 0.0) for i in range(5)}
                if fd_domain == "model":
                    m, _ = Arthemy5DTuner().tune_5d(m, source_lora=source_lora, target_block=target_lbl,
                                                    sub_components=sub_c, sub_tensor=sub_t,
                                                    master_multiplier=master_mult, **kw)
                else:
                    c, _ = Arthemy5DCLIPTuner().tune_5d_clip(c, source_lora=source_lora, target_layer=target_lbl,
                                                             sub_tensor=sub_t, sub_components=sub_c,
                                                             master_multiplier=master_mult, **kw)
                n_5d_applied += 1
            except (ValueError, TypeError) as e:
                logger.warning(f"[ARTHEMY PRESET LOADER] Skipping malformed 5D recipe: {e}")
                continue

        preset_name = data.get("name", preset)
        author = data.get("author", "Unknown")

        info_parts = [f"Loaded Preset '{preset_name}' by {author}"]
        info_parts.append(f"Model: {n_model_matched} scalar layers, {n_granular_applied} granular layers (x{strength_model:.2f})")
        info_parts.append(f"CLIP: {n_clip_matched} scalar layers (x{strength_clip:.2f})")
        if n_chaos_applied > 0:
            info_parts.append(f"Chaos: {n_chaos_applied} recipes")
        if n_rotation_applied > 0:
            info_parts.append(f"Rotations: {n_rotation_applied} recipes")
        if n_model_quant_skipped or n_clip_quant_skipped:
            info_parts.append(f"⚠️ {n_model_quant_skipped + n_clip_quant_skipped} quantized "
                              "layer(s) left untouched")
            logger.warning(f"[ARTHEMY PRESET LOADER] {n_model_quant_skipped + n_clip_quant_skipped} "
                           "layer(s) carry a companion scale (scaled FP8) and were left untouched; "
                           "patching them would double-apply that scale.")
        if n_chaos_rot_applied > 0:
            info_parts.append(f"Chaos rotations: {n_chaos_rot_applied} recipes")
        if n_channel_applied > 0:
            info_parts.append(f"Channel magnitude: {n_channel_applied} recipes")
        if n_5d_applied > 0:
            info_parts.append(f"5D: {n_5d_applied} recipes"
                              + (f" ({n_embedded} embedded)" if n_embedded else ""))
        if n_model_unmatched > 0 or n_clip_unmatched > 0:
            info_parts.append(f"⚠️ Unmatched keys: {n_model_unmatched} model, {n_clip_unmatched} clip")

        info = " | ".join(info_parts)
        logger.info(f"[ARTHEMY PRESET LOADER] {info}")
        return (m, c, info)


# ==============================================================================
# ROTATOR QUARTET (DUAL LIE & HARMONIC CHAOS)
# ==============================================================================

class _DualRotationMixin:
    """Four independent orthogonal rotation axes inside the dominant subspace."""

    @classmethod
    def INPUT_TYPES(s):
        inputs = s._common_inputs()
        for name, tip in (
            ("structural_rot_x", "Rotation *within* each component plane (adjacent principal directions). The most local control."),
            ("structural_rot_y", "Couples neighbouring components: tilts a branch into the residual trunk."),
            ("tensor_rot_x", "Long-range channel phase: couples each direction with the one half a subspace away."),
            ("tensor_rot_y", "Mirrored pairwise Givens rotation between channel pairs."),
        ):
            # Signed range: a negative angle is the exact inverse rotation of its positive
            # twin (expm(-A) == expm(A).T), so a small correction the other way is -5, not
            # the 355 the old 0-360 range forced the user to dial all the way round to.
            inputs[name] = ("FLOAT", {
                "default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5,
                "tooltip": tip + " Negative turns the opposite way by the same amount."})
        return {"required": inputs, "hidden": {"unique_id": "UNIQUE_ID"}}

    def _dual(self, obj, target_label, sub_components, depth_reach, sx, sy, tx, ty, unique_id=None):
        # Fold to the signed range HERE, not only inside the engine, so the persisted recipe,
        # the info headline and the 3D panel's params all quote the same number the rotation
        # was actually built from - a legacy 355 then reads out as -5 everywhere at once.
        sx, sy = normalize_signed_angle(sx), normalize_signed_angle(sy)
        tx, ty = normalize_signed_angle(tx), normalize_signed_angle(ty)
        rank = ROTATION_RANK_MAP.get(depth_reach, 16)
        factory = lambda w, ck: fast_dual_orthogonal_rotation(
            w, structural_x=sx, structural_y=sy, tensor_x=tx, tensor_y=ty,
            depth_rank=rank, layer_name=ck)
        recipe = {
            "type": "dual_rotation",
            "structural_x": sx, "structural_y": sy,
            "tensor_x": tx, "tensor_y": ty,
            "depth_reach": depth_reach,
        }
        headline = (f"{'CLIP' if self.IS_CLIP else 'Model'} Axis Rotator "
                    f"(struct {sx:.0f}/{sy:.0f} deg | tensor {tx:.0f}/{ty:.0f} deg, {depth_reach})")
        # Widget names, which differ from this recipe's persisted key names.
        live_params = {"structural_rot_x": sx, "structural_rot_y": sy,
                       "tensor_rot_x": tx, "tensor_rot_y": ty, "depth_reach": depth_reach}
        return self._apply(obj, target_label, sub_components, factory, recipe, headline,
                           unique_id, live_params)


class _ChaosRotationMixin:
    """Seed-deterministic harmonic chaos rotation."""

    RECIPE_KEY = "arthemy_chaos_rotation_recipes"

    @classmethod
    def INPUT_TYPES(s):
        inputs = s._common_inputs()
        inputs["seed"] = ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff,
                                  "tooltip": "Deterministic seed (CRC32-based, so it reproduces across ComfyUI restarts)."})
        inputs["chaos_strength"] = ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01,
                                             "tooltip": "Maximum per-plane rotation, as a fraction of 90 deg."})
        inputs["harmonic_coherence"] = ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                                 "tooltip": "Snaps the random plane angles towards 45 deg multiples."})
        return {"required": inputs, "hidden": {"unique_id": "UNIQUE_ID"}}

    def _chaos(self, obj, target_label, sub_components, depth_reach, seed, chaos_strength,
              harmonic_coherence, unique_id=None):
        rank = ROTATION_RANK_MAP.get(depth_reach, 16)
        factory = lambda w, ck: fast_chaos_orthogonal_rotation(
            w, seed=seed, chaos_strength=chaos_strength,
            harmonic_coherence=harmonic_coherence, depth_rank=rank, layer_name=ck)
        recipe = {
            "type": "chaos_rotation",
            "seed": seed,
            "chaos_strength": chaos_strength,
            "harmonic_coherence": harmonic_coherence,
            "depth_reach": depth_reach,
        }
        headline = (f"{'CLIP' if self.IS_CLIP else 'Model'} Chaos Rotator "
                    f"(seed {seed}, str {chaos_strength:.2f}, coh {harmonic_coherence:.2f}, {depth_reach})")
        live_params = {"seed": seed, "chaos_strength": chaos_strength,
                       "harmonic_coherence": harmonic_coherence, "depth_reach": depth_reach}
        return self._apply(obj, target_label, sub_components, factory, recipe, headline,
                           unique_id, live_params)


class ArthemyKrea2ModelRotator(_DualRotationMixin, BaseRotatorNode):
    """Axis Rotator (Model): four labelled rotation axes, structural and tensor."""

    IS_CLIP = False
    TARGET_MAP = ArthemyKrea2ModelBlockSurgeonTuner.MODEL_TARGET_MAP
    LOG_TAG = "ARTHEMY MODEL ROTATOR"
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "rotate_model"
    CATEGORY = "Arthemy/Krea2 Tuners"

    def rotate_model(self, model: Any, target_block: str = "Block_1 (All 0-4)",
                     sub_components: str = "All Components", depth_reach: str = "Default",
                     structural_rot_x: float = 0.0, structural_rot_y: float = 0.0,
                     tensor_rot_x: float = 0.0, tensor_rot_y: float = 0.0, unique_id=None) -> Tuple[Any, str]:
        return self._dual(model, target_block, sub_components, depth_reach,
                          structural_rot_x, structural_rot_y, tensor_rot_x, tensor_rot_y, unique_id)


class ArthemyKrea2ModelChaosRotator(_ChaosRotationMixin, BaseRotatorNode):
    """Model Chaos Rotator: seed-deterministic harmonic chaos rotation of the diffusion model."""

    IS_CLIP = False
    TARGET_MAP = ArthemyKrea2ModelBlockSurgeonTuner.MODEL_TARGET_MAP
    LOG_TAG = "ARTHEMY MODEL CHAOS ROTATOR"
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "chaos_rotate_model"
    CATEGORY = "Arthemy/Krea2 Tuners"

    def chaos_rotate_model(self, model: Any, target_block: str = "Block_1 (All 0-4)",
                           sub_components: str = "All Components", depth_reach: str = "Default",
                           seed: int = 42, chaos_strength: float = 0.35,
                           harmonic_coherence: float = 0.5, unique_id=None) -> Tuple[Any, str]:
        return self._chaos(model, target_block, sub_components, depth_reach,
                           seed, chaos_strength, harmonic_coherence, unique_id)


class ArthemyKrea2CLIPRotator(_DualRotationMixin, BaseRotatorNode):
    """Axis Rotator (CLIP): four labelled rotation axes, structural and tensor."""

    IS_CLIP = True
    TARGET_MAP = ArthemyKrea2CLIPBlockSurgeonTuner.CLIP_TARGET_MAP
    LOG_TAG = "ARTHEMY CLIP ROTATOR"
    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("CLIP", "info")
    FUNCTION = "rotate_clip"
    CATEGORY = "Arthemy/Krea2 Tuners"

    def rotate_clip(self, clip: Any, target_layer: str = "Layer_1 (All 0-4)",
                    sub_components: str = "All Components", depth_reach: str = "Default",
                    structural_rot_x: float = 0.0, structural_rot_y: float = 0.0,
                    tensor_rot_x: float = 0.0, tensor_rot_y: float = 0.0, unique_id=None) -> Tuple[Any, str]:
        return self._dual(clip, target_layer, sub_components, depth_reach,
                          structural_rot_x, structural_rot_y, tensor_rot_x, tensor_rot_y, unique_id)


class ArthemyKrea2CLIPChaosRotator(_ChaosRotationMixin, BaseRotatorNode):
    """CLIP Chaos Rotator: seed-deterministic harmonic chaos rotation of the text encoder."""

    IS_CLIP = True
    TARGET_MAP = ArthemyKrea2CLIPBlockSurgeonTuner.CLIP_TARGET_MAP
    LOG_TAG = "ARTHEMY CLIP CHAOS ROTATOR"
    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("CLIP", "info")
    FUNCTION = "chaos_rotate_clip"
    CATEGORY = "Arthemy/Krea2 Tuners"

    def chaos_rotate_clip(self, clip: Any, target_layer: str = "Layer_1 (All 0-4)",
                          sub_components: str = "All Components", depth_reach: str = "Default",
                          seed: int = 42, chaos_strength: float = 0.25,
                          harmonic_coherence: float = 0.5, unique_id=None) -> Tuple[Any, str]:
        return self._chaos(clip, target_layer, sub_components, depth_reach,
                           seed, chaos_strength, harmonic_coherence, unique_id)


# ==============================================================================
# LoRA-to-5D EXTRACTOR & 5D TUNERS (HARMONIC DCT MODIFIERS)
# ==============================================================================
# Bounded LRU cache of extracted 5D DCT payloads (each one can weigh several MB)
_DCT_CACHE: "collections.OrderedDict[str, Any]" = collections.OrderedDict()
_DCT_CACHE_MAX_ENTRIES = 4

# Modifiers supplied by a preset rather than found on disk, keyed by the name the recipe
# refers to. A preset that carries its own modifiers reproduces anywhere; without this a
# preset is only half of the recipe, and the other half is a file the recipient does not
# have. Deliberately NOT bounded like _DCT_CACHE: these are pinned for as long as the
# session lasts, because the preset that supplied them may be replayed at any time.
_EMBEDDED_MODIFIERS: Dict[str, Any] = {}


def _first_number(*candidates: Any, default: float) -> float:
    """First candidate that is a usable number, else `default`.

    `float(x or default)` is wrong for every widget whose minimum is 0: a saved
    `seed = 0`, `style_direction = 0` or `harmonic_coherence = 0` is falsy, so the preset
    replayed the DEFAULT instead of the value the user saved - and for `style_direction`
    that is 180 degrees, the exact opposite bearing, with no warning anywhere.
    """
    for c in candidates:
        if c is None:
            continue
        try:
            v = float(c)
        except (TypeError, ValueError):
            continue
        if v == v:                       # reject NaN, which compares unequal to itself
            return v
    return float(default)


def register_embedded_modifiers(payloads: Any, origin: str = "preset") -> int:
    """Makes preset-carried modifiers resolvable by name for the rest of the session.

    Returns how many were registered. Existing on-disk modifiers do NOT shadow these: a
    preset that ships a modifier is stating exactly which bytes it was built with, and
    honouring a same-named local file instead is how a preset silently stops reproducing.
    """
    if not isinstance(payloads, dict):
        return 0
    n = 0
    for name, payload in payloads.items():
        if not isinstance(name, str) or not isinstance(payload, dict):
            continue
        if "layers" not in payload:
            continue
        _EMBEDDED_MODIFIERS[name] = payload
        n += 1
    if n:
        logger.info(f"[Arthemy 5D] {n} modifier(s) supplied inline by {origin}; they now resolve "
                    "by name without needing the original file.")
        for name, payload in payloads.items():
            scope = payload.get("arthemy_pruned") if isinstance(payload, dict) else None
            if not isinstance(scope, dict):
                continue
            targets = ", ".join(describe_pruned_scope(s) for s in scope.get("scopes", [])
                                if isinstance(s, dict))
            logger.info(f"[Arthemy 5D] '{name}' is a scoped copy: {scope.get('kept_layers')} of "
                        f"{scope.get('original_layers')} layers, covering {targets or 'an unrecorded scope'}. "
                        f"It is registered under this scoped name only, so '{scope_alias_base(name)}' "
                        "keeps resolving to the complete modifier for every other node.")
    return n


# Separator between a modifier name and the scope tag of a preset-scoped copy of it.
# A pruned payload is NOT the modifier it was cut from, so it must never be registered
# under the modifier's own name: _EMBEDDED_MODIFIERS is pinned for the whole session and
# deliberately outranks the file on disk, so a partial payload registered under the bare
# name would make every OTHER 5D node in the session silently return 0 patches for every
# block the preset did not happen to use.
SCOPE_ALIAS_SEP = "#5dscope-"


def describe_pruned_scope(scope: Dict[str, Any]) -> str:
    """One-line rendering of a recorded prune scope, for logs and node reports."""
    parts = [str(scope.get("domain", "?")), str(scope.get("target", "?")),
             str(scope.get("sub_components", "?"))]
    st = scope.get("sub_tensor")
    if is_sub_tensor_narrowed(st):
        parts.append(str(st))
    return " / ".join(parts)


def scope_alias_base(name: str) -> str:
    """The modifier name a scoped alias was cut from ('X.json#5dscope-tag' -> 'X.json')."""
    return name.split(SCOPE_ALIAS_SEP, 1)[0] if SCOPE_ALIAS_SEP in name else name


def _scope_tag(scopes: List[Tuple[str, str, str, str]]) -> str:
    """Short, deterministic id for a set of (domain, target, subs, sub_tensor) scopes."""
    raw = "|".join(sorted(":".join(str(f) for f in scope) for scope in scopes))
    tag = re.sub(r"[^\w\-.]+", "_", raw).strip("_")
    if len(tag) > 48:
        tag = f"{zlib.crc32(raw.encode('utf-8')) & 0xFFFFFFFF:08x}"
    return tag or "scope"


def prune_5d_payload_to_scopes(payload: Any,
                               scopes: List[Tuple[str, str, str, str]]) -> Tuple[Any, Dict[str, Any]]:
    """Cuts a 5D payload down to the layers its recipes can actually reach.

    A 5D injection restricted to one block still embedded the WHOLE modifier in the
    preset - every block of the model plus every CLIP layer - because the block /
    sub-component selection was only ever applied at synthesis time. The recipe already
    records domain / target_block / sub_components / sub_tensor, so the same predicate can
    be applied at save time: what the filter would have thrown away is what the preset
    never needed to carry.

    The predicate is `Base5DTuner._layer_filter` itself, not a copy of its rules: the one
    thing this must never do is disagree with the filter the tuner applies on load.

    Returns `(payload, stats)`. `payload` is the ORIGINAL object (identity-comparable)
    whenever nothing could be dropped, so the caller can tell "pruned" from "unchanged".
    """
    layers = payload.get("layers") if isinstance(payload, dict) else None
    if not isinstance(layers, dict) or not layers or not scopes:
        return payload, {}

    # domain -> list of predicates, or True for "this whole domain stays"
    keepers: Dict[str, Any] = {}
    for domain, target_label, sub_components, sub_tensor in scopes:
        dom = "clip" if str(domain).lower() == "clip" else "model"
        if keepers.get(dom) is True:
            continue
        tuner = Arthemy5DCLIPTuner() if dom == "clip" else Arthemy5DTuner()
        selected = resolve_target_map_entry(tuner.TARGET_MAP, target_label, lambda: None)
        if selected is None:
            # An unrecognised label falls back to the full range on load, so it must keep
            # the full domain here too - pruning against a selection we could not resolve
            # is how a preset loses layers it will later ask for.
            keepers[dom] = True
            continue
        keepers.setdefault(dom, []).append(tuner._layer_filter(
            selected, sub_components,
            whole_model=str(target_label).strip().lower().startswith("all"),
            sub_tensor=sub_tensor))

    kept: Dict[str, Any] = {}
    for name, entry in layers.items():
        raw_dom = entry.get("domain", "model") if isinstance(entry, dict) else "model"
        dom = "clip" if str(raw_dom).lower() == "clip" else "model"
        preds = keepers.get(dom)
        if preds is None:
            continue  # no recipe touches this domain at all (the usual CLIP-side saving)
        if preds is True or any(pred(name) for pred in preds):
            kept[name] = entry

    stats = {"kept": len(kept), "total": len(layers), "tag": ""}
    if len(kept) == len(layers):
        return payload, stats
    if not kept:
        # Every layer filtered out means the recipes cannot reproduce anything from this
        # payload. That is a bug somewhere upstream, not a saving - keep the payload whole
        # rather than writing a preset that is guaranteed to inject nothing.
        logger.warning("[Arthemy 5D] Scope pruning would have dropped every layer of a modifier; "
                       "embedding it in full instead.")
        return payload, stats

    pruned = {k: v for k, v in payload.items() if k != "layers"}
    pruned["layers"] = kept
    # Freeze the fidelity of the FULL payload. payload_worst_fidelity reads it off the
    # longest vector still present, so pruning away the big tensors would otherwise make a
    # low-fidelity DCT modifier look better than it is, exactly in the guard that exists
    # to warn about it.
    if "fidelity_worst_full" not in pruned:
        worst = payload_worst_fidelity(payload)
        if worst is not None:
            pruned["fidelity_worst_full"] = round(float(worst), 6)
    pruned["arthemy_pruned"] = {
        "kept_layers": len(kept),
        "original_layers": len(layers),
        "scopes": [{"domain": d, "target": t, "sub_components": s, "sub_tensor": st}
                   for d, t, s, st in scopes],
    }
    stats["tag"] = _scope_tag(scopes)
    return pruned, stats


ARTHEMY_MODIFIERS_DIRNAME = "arthemy_modifiers"

STORAGE_MODE_RAW = "raw factors (exact, a few MB)"
STORAGE_MODE_DCT = "harmonic DCT (tiny, lossy)"
STORAGE_MODE_AUTO = "auto (raw unless DCT would stay faithful)"

# Below this direction-cosine fidelity, a DCT-truncated singular vector is closer to
# noise than to the real LoRA direction (see dct_fidelity's docstring). "auto" storage
# uses this as the cutoff between "DCT is fine, keep the file tiny" and "fall back to
# an exact raw copy".
AUTO_STORAGE_FIDELITY_FLOOR = 0.35

# The harmonic order "auto" reasons about. Both the Extractor widget default and the
# on-the-fly extraction in get_cached_5d_dct use this, because the raw-vs-DCT verdict is a
# function of the order as well as of the file: at order 16 the floor sits at 130-long
# vectors, at order 64 at 522. Two different defaults would make the same LoRA resolve
# differently depending on which node touched it first.
AUTO_STORAGE_DEFAULT_ORDER = 64


def payload_is_dct(payload: Any) -> bool:
    """True when a 5D modifier payload stores harmonic DCT coefficients.

    The top-level "storage" key only exists in payloads written since raw storage was
    added; v3 modifiers (`Comics_5D.json` and friends) have no such key and are always
    DCT. `synthesize_5d_patches_from_dct` already recognises the format per-layer, from
    the presence of u_raw/v_raw - so any check that trusts the top-level key alone
    silently exempts exactly the legacy files most likely to be low fidelity.
    """
    if not isinstance(payload, dict):
        return False
    storage = str(payload.get("storage") or "").lower()
    # Only an explicitly RECOGNISED storage string is authoritative. An empty or unknown
    # value falls through to the per-layer check below rather than answering False, which
    # would exempt the payload from the fidelity guard on nothing more than a typo.
    if storage.startswith("dct") or storage == "harmonic_dct":
        return True
    if storage.startswith("raw"):
        return False
    layers = payload.get("layers", {})
    if not isinstance(layers, dict):
        return False
    return any(isinstance(v, dict) and "u_dct" in v and "v_dct" in v for v in layers.values())


def resolve_auto_storage(lora_path: str, harmonic_order: int) -> Tuple[str, str]:
    """Picks raw-vs-DCT storage for a LoRA from a shape-only peek at its tensors.

    Returns `(storage_mode, human_note)`. Shared by the Extractor node and the on-the-fly
    extraction inside `get_cached_5d_dct`. The verdict depends on `harmonic_order` as well
    as on the file (the floor is `order / 0.1225` in vector length), so the two callers pass
    the same default - see AUTO_STORAGE_DEFAULT_ORDER - to keep "auto" meaning one thing.
    """
    if dct_fidelity is None or peek_lora_vector_lengths is None:
        return STORAGE_RAW, "auto -> raw factors (could not pre-check DCT fidelity) | "
    lens = peek_lora_vector_lengths(lora_path)
    worst_len = max(lens) if lens else 0
    predicted = dct_fidelity(harmonic_order, worst_len) if worst_len else 0.0
    if predicted >= AUTO_STORAGE_FIDELITY_FLOOR:
        return STORAGE_DCT, (f"auto -> DCT order {harmonic_order} (~{predicted * 100:.1f}% "
                             f"fidelity on this LoRA's longest vector, {worst_len}) | ")
    return STORAGE_RAW, (f"auto -> raw factors (DCT order {harmonic_order} would only reach "
                         f"~{predicted * 100:.1f}% fidelity on this LoRA's longest vector, "
                         f"{worst_len}, below the {int(AUTO_STORAGE_FIDELITY_FLOOR * 100)}% floor) | ")


def payload_worst_fidelity(payload: Any) -> Optional[float]:
    """Predicted direction cosine of the worst (longest) vector in a DCT payload, or None
    when the payload is not DCT / carries no usable dimensions.

    Modifier JSONs are files on disk that a user can hand-edit, so every field here is
    treated as untrusted: this runs inside the Preset Loader, which has no try/except of
    its own, and one `"d_out": null` must not turn a warning into a crashed node.
    """
    if dct_fidelity is None or not payload_is_dct(payload):
        return None
    try:
        # A scope-pruned payload records the fidelity of the payload it was cut from: the
        # longest vector may have been pruned away, and re-deriving the figure from what is
        # left would quietly report a fidelity this modifier never had.
        frozen = payload.get("fidelity_worst_full")
        if frozen is not None:
            try:
                return float(frozen)
            except (TypeError, ValueError):
                pass
        layers = payload.get("layers", {})
        if not isinstance(layers, dict):
            return None
        lens = set()
        for v in layers.values():
            if not isinstance(v, dict):
                continue
            for key in ("d_out", "d_in"):
                try:
                    n = int(v.get(key, 0))
                except (TypeError, ValueError):
                    continue
                if n > 0:
                    lens.add(n)
        if not lens:
            return None
        try:
            order = int(payload.get("harmonic_order", 16))
        except (TypeError, ValueError):
            order = 16
        return dct_fidelity(order, max(lens))
    except Exception as e:
        logger.debug(f"[Arthemy 5D] Could not assess modifier fidelity: {e}")
        return None

# Modifiers are searched in both places, exactly like presets. `folder_paths.models_dir`
# is the ComfyUI root's own `models/` folder, which under a launcher such as
# StabilityMatrix is NOT the shared model library the user sees in the UI - dropping a
# modifier in the wrong one used to make it silently invisible.
local_modifiers_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "modifiers")


def get_arthemy_modifiers_dir() -> str:
    """The directory new modifiers are written to (the ComfyUI models folder)."""
    mod_dir = os.path.join(folder_paths.models_dir, ARTHEMY_MODIFIERS_DIRNAME)
    os.makedirs(mod_dir, exist_ok=True)
    return mod_dir


def get_arthemy_modifier_dirs() -> List[str]:
    """Every directory a modifier may live in, in search order."""
    dirs = [get_arthemy_modifiers_dir()]
    if os.path.isdir(local_modifiers_dir):
        dirs.append(local_modifiers_dir)
    return dirs


def resolve_modifier_path(name: str) -> Optional[str]:
    """Finds a modifier by bare name across every search directory."""
    if not name:
        return None
    if os.path.isfile(name):
        return name
    for d in get_arthemy_modifier_dirs():
        cand = os.path.join(d, os.path.basename(name))
        if os.path.isfile(cand):
            return cand
    return None


def get_5d_modifier_list() -> List[str]:
    """Pre-extracted JSON modifiers first, then raw LoRA files (extracted on the fly)."""
    seen, json_mods = set(), []
    for d in get_arthemy_modifier_dirs():
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".json") and f not in seen:
                seen.add(f)
                json_mods.append(f)
    lora_list = [f for f in folder_paths.get_filename_list("loras") if not f.lower().endswith(".json")]
    return json_mods + lora_list


def get_cached_5d_dct(lora_name_or_path: str,
                      num_harmonic_coeffs: int = AUTO_STORAGE_DEFAULT_ORDER) -> Optional[Any]:
    """Returns the 5D harmonic DCT payload for a LoRA or a pre-extracted JSON modifier.

    Accepts a JSON modifier name, a LoRA name as listed by ComfyUI, or an absolute path.
    Both branches share the LRU cache (the JSON branch used to re-read and re-parse the
    file on every single node execution).
    """
    if extract_lora_to_5d_dct is None:
        return None
    clean_name = (lora_name_or_path or "").strip()
    if not clean_name:
        return None

    # Case 0: a modifier carried inside a preset. Checked BEFORE the folder lookup so a
    # self-contained preset reproduces on a machine that has never seen the source LoRA.
    embedded = _EMBEDDED_MODIFIERS.get(clean_name)
    if embedded is not None:
        return embedded

    # A preset-scoped alias whose preset has not been loaded in this session (a recipe
    # copied by hand, a preset re-saved without its payload). Falling back to the complete
    # modifier it was cut from reproduces the same injection: the tuner re-applies the very
    # filter the pruning used, so the extra layers are dropped again at synthesis time.
    if SCOPE_ALIAS_SEP in clean_name:
        base = scope_alias_base(clean_name)
        logger.info(f"[Arthemy 5D] Scoped modifier '{clean_name}' was not supplied by any preset in "
                    f"this session; falling back to the complete '{base}'.")
        return get_cached_5d_dct(base, num_harmonic_coeffs)

    # Case 1: pre-extracted JSON modifier
    if clean_name.lower().endswith(".json"):
        json_path = resolve_modifier_path(clean_name)
        if not json_path:
            logger.warning(f"[Arthemy 5D] Modifier '{clean_name}' not found in "
                           f"{get_arthemy_modifier_dirs()} and no preset has supplied it inline.")
            return None
        cached = _DCT_CACHE.get(json_path)
        if cached is not None:
            _DCT_CACHE.move_to_end(json_path)
            return cached
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"[Arthemy 5D] Failed to read JSON modifier '{json_path}': {e}")
            return None
        _DCT_CACHE[json_path] = data
        while len(_DCT_CACHE) > _DCT_CACHE_MAX_ENTRIES:
            _DCT_CACHE.popitem(last=False)
        return data

    # Case 2: safetensors LoRA, extracted on the fly and cached
    lora_path = clean_name if os.path.isfile(clean_name) else (folder_paths.get_full_path("loras", clean_name) or "")
    if not lora_path or not os.path.exists(lora_path):
        return None

    # The order is part of the identity of the extracted payload, so two calls asking for
    # different orders must not hand each other the same cached result.
    cache_key = f"{lora_path}#{int(num_harmonic_coeffs)}"
    cached = _DCT_CACHE.get(cache_key)
    if cached is not None:
        _DCT_CACHE.move_to_end(cache_key)
        return cached

    # Same "auto" rule the Extractor node applies. Without it, picking a raw .safetensors
    # LoRA straight from the 5D Tuner's dropdown always extracted as DCT - on a 36k-long
    # vector that is a few percent of the real direction, i.e. an injection unrelated to
    # the LoRA the user chose, with no warning anywhere.
    mode, auto_note = resolve_auto_storage(lora_path, num_harmonic_coeffs)
    try:
        dct_data = extract_lora_to_5d_dct(lora_path, num_harmonic_coeffs=num_harmonic_coeffs,
                                          storage=mode)
    except Exception as e:
        logger.warning(f"[Arthemy 5D] DCT extraction failed for '{lora_path}': {e}")
        return None
    logger.info(f"[Arthemy 5D] On-the-fly extraction of '{os.path.basename(lora_path)}': {auto_note}"
                f"{len(dct_data.get('layers', {}))} layers.")

    _DCT_CACHE[cache_key] = dct_data
    while len(_DCT_CACHE) > _DCT_CACHE_MAX_ENTRIES:
        _DCT_CACHE.popitem(last=False)
    return dct_data


class ArthemyLoRAto5DExtractor(BaseKrea2Node):
    """LoRA-to-5D Extractor: compresses any LoRA into a standalone harmonic DCT modifier."""
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        lora_list = combo_options(folder_paths.get_filename_list("loras"))
        return {
            "required": {
                "lora_name": (lora_list, {"default": lora_list[0]}),
            },
            "optional": {
                "custom_modifier_name": ("STRING", {"default": ""}),
                "storage": ([STORAGE_MODE_AUTO, STORAGE_MODE_RAW, STORAGE_MODE_DCT], {
                    "default": STORAGE_MODE_AUTO,
                    "tooltip": "Auto (recommended): checks what direction-cosine fidelity the "
                               "chosen DCT order would actually give on this LoRA's own tensor "
                               "sizes, and falls back to raw factors when that would be below "
                               f"{int(AUTO_STORAGE_FIDELITY_FLOOR * 100)}% - i.e. closer to noise "
                               "than to the real LoRA direction. Raw factors: a few MB, but an "
                               "exact rank-N copy. Harmonic DCT: tens of KB, but only about "
                               "sqrt(order / vector length) faithful - force this only when the "
                               "file size matters more than accuracy."}),
                "harmonic_order": ("INT", {"default": AUTO_STORAGE_DEFAULT_ORDER, "min": 4, "max": 2048, "step": 4,
                                           "tooltip": "DCT mode only. Coefficients kept per singular vector."}),
                "num_dimensions": ("INT", {"default": 5, "min": 1, "max": 8, "step": 1,
                                           "tooltip": "How many dominant SVD directions to keep."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("modifier_report",)
    FUNCTION = "extract_5d"
    CATEGORY = "Arthemy/Krea2 LoRA"

    def extract_5d(self, lora_name: str, custom_modifier_name: str = "",
                   storage: str = STORAGE_MODE_AUTO,
                   harmonic_order: int = AUTO_STORAGE_DEFAULT_ORDER, num_dimensions: int = 5) -> Tuple[str]:
        if extract_lora_to_5d_dct is None:
            return (f"Error: geometry engine unavailable ({GEOMETRY_ENGINE_ERROR}).",)
        if not lora_name:
            return ("Error: No LoRA selected.",)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if not lora_path or not os.path.exists(lora_path):
            return (f"Error: LoRA file '{lora_name}' not found.",)

        auto_note = ""
        if storage == STORAGE_MODE_AUTO:
            # Decide raw-vs-DCT before paying for the SVD, from a shape-only peek at this
            # LoRA's own tensors - the same dct_fidelity math the manual DCT path already
            # used to warn after the fact, just consulted first instead of after.
            mode, auto_note = resolve_auto_storage(lora_path, harmonic_order)
        else:
            mode = STORAGE_RAW if storage == STORAGE_MODE_RAW else STORAGE_DCT
        t_start = time.time()
        try:
            dct_data = extract_lora_to_5d_dct(lora_path, num_harmonic_coeffs=harmonic_order,
                                              num_dims=num_dimensions, storage=mode)
        except Exception as e:
            logger.warning(f"[ARTHEMY LoRA-to-5D EXTRACTOR] Extraction failed: {e}")
            return (f"Error extracting '{lora_name}': {e}",)

        layers = dct_data.get("layers", {}) if isinstance(dct_data, dict) else {}
        if not layers:
            return (f"Error: no LoRA up/down pairs found in '{lora_name}'.",)

        n_model = sum(1 for v in layers.values() if v.get("domain") == "model")
        n_clip = len(layers) - n_model

        base_name = os.path.splitext(os.path.basename(lora_path))[0]
        mod_name = custom_modifier_name.strip() or f"{base_name}_Harmonic5D"
        safe_mod_name = re.sub(r"[^\w\-_\.]", "_", os.path.basename(mod_name)) or "arthemy_modifier"
        mod_file = os.path.join(get_arthemy_modifiers_dir(), f"{safe_mod_name}.json")

        try:
            with open(mod_file, "w", encoding="utf-8") as f:
                json.dump(dct_data, f, indent=2)
        except Exception as e:
            logger.warning(f"[ARTHEMY LoRA-to-5D EXTRACTOR] Could not write '{mod_file}': {e}")
            return (f"Error writing modifier file: {e}",)

        # Refresh the cache under the modifier name too, so the 5D Tuners see it immediately.
        # Evict any stale pinned embedded copies so the re-extracted version takes effect at once.
        _DCT_CACHE[mod_file] = dct_data
        _DCT_CACHE[f"{safe_mod_name}.json"] = dct_data
        _DCT_CACHE[safe_mod_name] = dct_data
        _EMBEDDED_MODIFIERS.pop(f"{safe_mod_name}.json", None)
        _EMBEDDED_MODIFIERS.pop(safe_mod_name, None)
        while len(_DCT_CACHE) > _DCT_CACHE_MAX_ENTRIES:
            _DCT_CACHE.popitem(last=False)

        size_kb = os.path.getsize(mod_file) / 1024.0
        detail = f"raw float16 factors (exact rank {num_dimensions})" if mode == STORAGE_RAW else f"DCT order {harmonic_order}"
        # The exact resolved path, not a hand-built "models/..." guess: get_arthemy_modifiers_dir()
        # is the ComfyUI models folder, which under a launcher such as StabilityMatrix is NOT
        # the same as the shared model library the user browses in the UI - showing the real
        # path avoids the "which of my two model folders did this go into" confusion.
        report = (f"{auto_note}Extracted {num_dimensions}D modifier from '{lora_name}' -> "
                  f"{mod_file} "
                  f"({len(layers)} layers: {n_model} model / {n_clip} clip, "
                  f"{detail}, {size_kb / 1024.0:.2f} MB, {time.time() - t_start:.2f}s). To use it, pick "
                  f"'{safe_mod_name}.json' from the 5D Tuner's source_lora dropdown - it searches this "
                  f"same folder automatically.")

        if mode == STORAGE_DCT and dct_fidelity is not None:
            # A truncated DCT is a low-pass filter and singular vectors are not smooth, so
            # the retained fraction of the frequency band IS the fidelity. Say it out loud
            # rather than letting the node quietly inject an unrelated direction.
            lens = sorted({int(v.get("d_out", 0)) for v in layers.values()} |
                          {int(v.get("d_in", 0)) for v in layers.values()})
            if lens:
                worst = dct_fidelity(harmonic_order, max(lens))
                report += (f" | fidelity ~{worst * 100:.1f}% direction cosine on the longest "
                           f"vector ({max(lens)}); switch to raw factors for an exact copy")
                if worst < AUTO_STORAGE_FIDELITY_FLOOR:
                    logger.warning(f"[ARTHEMY LoRA-to-5D EXTRACTOR] DCT order {harmonic_order} keeps only "
                                   f"~{worst * 100:.1f}% of each singular direction on {max(lens)}-long "
                                   "vectors. The 5D injection will be mostly unrelated to this LoRA. "
                                   "Use storage='raw factors' (or 'auto') unless you specifically need a "
                                   "tiny file.")
        logger.info(f"[ARTHEMY LoRA-to-5D EXTRACTOR] {report}")
        return (report,)


class Base5DTuner(BaseKrea2Node):
    """Shared driver for the Model and CLIP 5D Tuners.

    Injects the 5 dominant harmonic dimensions of a LoRA as native low-rank adapters,
    restricted to the selected block/layer group and sub-component - the same targeting
    grammar as every other granular node in the suite.
    """

    IS_CLIP = False
    TARGET_MAP: Dict[str, Any] = {}
    LOG_TAG = "ARTHEMY 5D TUNER"
    NUM_DIMS = 5

    @classmethod
    def _io_name(cls) -> str:
        return "clip" if cls.IS_CLIP else "model"

    @classmethod
    def INPUT_TYPES(s):
        mod_list = combo_options(get_5d_modifier_list())
        io_type = "CLIP" if s.IS_CLIP else "MODEL"
        target_name = "target_layer" if s.IS_CLIP else "target_block"
        default_target = "All Layers (0-59)" if s.IS_CLIP else "All Blocks (0-27)"
        subs = CLIP_SUBCOMPONENTS if s.IS_CLIP else MODEL_SUBCOMPONENTS
        inputs = {
            s._io_name(): (io_type,),
            "source_lora": (mod_list, {"default": mod_list[0],
                                       "tooltip": "A pre-extracted .json modifier, or a LoRA that gets "
                                                  "compressed on the fly and cached."}),
            target_name: (list(s.TARGET_MAP.keys()), {"default": default_target,
                                                      "tooltip": "Where to inject the harmonic dimensions."}),
            "sub_components": (subs, {"default": "All Components",
                                      "tooltip": "Component subgroup inside the selection."}),
            "master_multiplier": ("FLOAT", {"default": 1.0, "min": -99.00, "max": 99.00, "step": 0.05,
                                            "tooltip": "Global multiplier applied on top of every dimension."}),
        }
        for i in range(1, s.NUM_DIMS + 1):
            inputs[f"Dim_{i:02d}"] = ("FLOAT", {
                "default": 0.0, "min": -99.00, "max": 99.00, "step": 0.05,
                "tooltip": f"Weight of harmonic dimension {i} (SVD direction {i} of the source LoRA)."})
        # Deliberately LAST: ComfyUI stores widget values positionally, so a new widget
        # inserted above the Dim_ rows would shift every saved workflow's dimensions by one.
        inputs["sub_tensor"] = (CLIP_SUB_TENSORS if s.IS_CLIP else MODEL_SUB_TENSORS, {
            "default": SUB_TENSOR_ANY,
            "tooltip": "Optional: inject into ONE tensor of the selection (WQ alone, MLP-down "
                       "alone, ...) instead of the whole sub-component group. Intersects with "
                       "sub_components - leave that on 'All Components' when using this. Note "
                       "that a LoRA only carries the tensors it was trained on: norm / mod "
                       "entries usually have no up/down pair, and a fused QKV cannot be split "
                       "into WQ/WK/WV. A Preset Saver with prune_5d_to_target on carries only "
                       "the tensors this selection reaches, so the narrower the choice, the "
                       "smaller the preset."})
        return {"required": inputs}

    def _layer_filter(self, selected_indices: set, sub_components: str, whole_model: bool = False,
                      sub_tensor: str = SUB_TENSOR_ANY):
        """Predicate over a modifier's clean layer name, mirroring the surgeon nodes' rules.

        `whole_model` is set when the target dropdown is the "All blocks / All layers" entry.
        Only then do the non-indexed sections (text fusion, time embedding, the input and
        output projections) come along: they belong to no block, so a block-scoped
        selection must not silently include them - and a whole-model selection must not
        silently exclude them, which is what happened before.

        `sub_tensor` narrows one step further, to a single surgeon-map tensor inside the
        selection. It is applied on top of `sub_components`, not instead of it, so the two
        widgets intersect rather than override each other.
        """
        is_clip = self.IS_CLIP
        narrow = is_sub_tensor_narrowed(sub_tensor)
        want = str(sub_tensor).strip()

        def _keep(clean_layer: str) -> bool:
            if is_clip:
                idx, sub_key = Krea2TensorParser.extract_clip_layer_idx(clean_layer)
            else:
                idx, sub_key = Krea2TensorParser.extract_model_block_idx(clean_layer)
            if idx is None:
                # The block-less sections have no sub-tensor identity in the surgeon map, so
                # a sub-tensor selection cannot mean them even on a whole-model target.
                return whole_model and not narrow
            if idx not in selected_indices:
                return False
            if not match_sub_component(clean_layer, sub_components, is_clip):
                return False
            if narrow and match_sub_tensor_label(sub_key, is_clip) != want:
                return False
            return True

        return _keep

    def _tune(self, obj: Any, source_lora: str, target_label: str, sub_components: str,
              master_multiplier: float, weights: List[float],
              sub_tensor: str = SUB_TENSOR_ANY) -> Tuple[Any, str]:
        domain = "clip" if self.IS_CLIP else "model"
        label = domain.upper() if self.IS_CLIP else "Model"

        if synthesize_5d_patches_from_dct is None:
            return (obj, f"5D Tuner error: geometry engine unavailable ({GEOMETRY_ENGINE_ERROR}).")
        if LORA_ADAPTER_CLS is None:
            return (obj, "5D Tuner error: this ComfyUI build has no comfy.weight_adapter.LoRAAdapter. Update ComfyUI.")
        if all(abs(w) < 1e-5 for w in weights) or abs(master_multiplier) < 1e-5:
            return (obj, f"5D {label} Tuner: bypassed (all dimensions at 0.0).")

        dct_data = get_cached_5d_dct(source_lora)
        if dct_data is None:
            return (obj, f"5D Tuner error: '{source_lora}' not found or not convertible to a 5D modifier.")

        clone_obj = obj.clone()
        _p = get_patcher(clone_obj)
        selected = resolve_target_map_entry(
            self.TARGET_MAP, target_label,
            lambda: set(range(0, Krea2Config.probe(
                _p.model.state_dict() if hasattr(_p, "model") else {}, self.IS_CLIP))))

        try:
            deltas = synthesize_5d_patches_from_dct(
                dct_data, weights, master_multiplier=master_multiplier, device="cpu",
                domain=domain,
                layer_filter=self._layer_filter(selected, sub_components,
                                                whole_model=target_label.strip().lower().startswith("all"),
                                                sub_tensor=sub_tensor))
        except Exception as e:
            logger.warning(f"[{self.LOG_TAG}] Synthesis failed: {e}")
            return (obj, f"5D Tuner error: {e}")

        if not deltas:
            n_dom = sum(1 for v in dct_data.get("layers", {}).values() if v.get("domain", "model") == domain)
            hint = (f"the modifier has no '{domain}' layers" if n_dom == 0
                    else "no modifier layer matched the selected blocks / sub-components")
            if n_dom and is_sub_tensor_narrowed(sub_tensor):
                hint += (f"; '{sub_tensor}' in particular exists in this modifier only if the "
                         "source LoRA trained that tensor unfused")
            # A payload a Preset Saver cut down to its own target says WHY there is nothing to
            # match here, instead of leaving the user to wonder why the same modifier that
            # works at one block does nothing at another.
            scope = dct_data.get("arthemy_pruned") if isinstance(dct_data, dict) else None
            if isinstance(scope, dict):
                targets = ", ".join(describe_pruned_scope(s) for s in scope.get("scopes", [])
                                    if isinstance(s, dict))
                hint += (f"; '{source_lora}' is a preset-scoped copy holding only "
                         f"{scope.get('kept_layers')} of {scope.get('original_layers')} layers"
                         + (f", for {targets}" if targets else "")
                         + f" - point this node at '{scope_alias_base(source_lora)}' for the complete modifier")
            return (clone_obj, f"5D {label} Tuner: 0 patches ({hint}).")

        patches: Dict[str, Any] = {}
        per_index: Dict[int, Dict[str, Any]] = {}
        for clean_layer, delta in deltas.items():
            key = f"{clean_layer}.weight"
            if not self.IS_CLIP and not key.startswith("diffusion_model."):
                key = f"diffusion_model.{key}"
            value, meta = build_geometry_patch(delta, clean_layer)
            if value is None:
                continue
            patches[key] = value
            if self.IS_CLIP:
                idx, _ = Krea2TensorParser.extract_clip_layer_idx(clean_layer)
            else:
                idx, _ = Krea2TensorParser.extract_model_block_idx(clean_layer)
            if idx is not None:
                slot = per_index.setdefault(idx, {"is_five_d": True, "five_d_dims": meta.get("active_dims", 0),
                                                  "five_d_source": meta.get("source_lora", source_lora)})
                slot["is_five_d"] = True

        applied = len(inject_patches(clone_obj, patches, 1.0)) if patches else 0

        if applied:
            record_section_meta(get_patcher(clone_obj), domain, per_index)
            # Slim recipe: the multi-megabyte DCT payload is NOT stored in model_options,
            # because ComfyUI deep-copies model_options on every downstream patcher clone
            # (and the Preset Saver would embed it in the JSON). It is rebuilt from cache.
            append_recipe(get_patcher(clone_obj), "arthemy_5d_recipes", {
                "domain": domain,
                "source_lora": source_lora,
                "target_block": target_label,
                "sub_components": sub_components,
                "sub_tensor": sub_tensor,
                "master_multiplier": master_multiplier,
                "dimension_weights": weights,
            })

        # Which of the two search folders (get_arthemy_modifier_dirs) this actually came
        # from - a bare filename in the dropdown says nothing about that, and the two
        # folders can silently both contain a same-named file.
        resolved_path = (resolve_modifier_path(source_lora) if source_lora.lower().endswith(".json")
                         else folder_paths.get_full_path("loras", source_lora))

        dim_str = ", ".join(f"D{i + 1}:{w:+.2f}" for i, w in enumerate(weights) if abs(w) > 1e-4)
        scope_str = f"{target_label} / {sub_components}"
        if is_sub_tensor_narrowed(sub_tensor):
            scope_str += f" / {sub_tensor}"
        info = (f"5D {label} Tuner [{dim_str}] x{master_multiplier:.2f} | {resolved_path or source_lora} | "
                f"{scope_str} | Patches: {applied}")
        logger.info(f"[{self.LOG_TAG}] {info}")
        return (clone_obj, info)

    def _collect_weights(self, kwargs: Dict[str, Any]) -> List[float]:
        return [float(kwargs.get(f"Dim_{i:02d}", 0.0) or 0.0) for i in range(1, self.NUM_DIMS + 1)]


class Arthemy5DTuner(Base5DTuner):
    """5D Tuner: harmonic modifier injection into the diffusion model."""

    IS_CLIP = False
    TARGET_MAP = ArthemyKrea2ModelBlockSurgeonTuner.MODEL_TARGET_MAP
    LOG_TAG = "ARTHEMY 5D MODEL TUNER"
    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "tune_5d"
    CATEGORY = "Arthemy/Krea2 Tuners"

    def tune_5d(self, model: Any, source_lora: str = "", target_block: str = "All Blocks (0-27)",
                sub_components: str = "All Components", master_multiplier: float = 1.0,
                sub_tensor: str = SUB_TENSOR_ANY, **kwargs: float) -> Tuple[Any, str]:
        return self._tune(model, source_lora, target_block, sub_components,
                          master_multiplier, self._collect_weights(kwargs), sub_tensor=sub_tensor)


class Arthemy5DCLIPTuner(Base5DTuner):
    """5D CLIP Tuner: harmonic modifier injection into the text encoder."""

    IS_CLIP = True
    TARGET_MAP = ArthemyKrea2CLIPBlockSurgeonTuner.CLIP_TARGET_MAP
    LOG_TAG = "ARTHEMY 5D CLIP TUNER"
    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("CLIP", "info")
    FUNCTION = "tune_5d_clip"
    CATEGORY = "Arthemy/Krea2 Tuners"

    def tune_5d_clip(self, clip: Any, source_lora: str = "", target_layer: str = "All Layers (0-59)",
                     sub_components: str = "All Components", master_multiplier: float = 1.0,
                     sub_tensor: str = SUB_TENSOR_ANY, **kwargs: float) -> Tuple[Any, str]:
        return self._tune(clip, source_lora, target_layer, sub_components,
                          master_multiplier, self._collect_weights(kwargs), sub_tensor=sub_tensor)


# ==============================================================================
# VERTICAL TUNING SUITE (Channel Magnitude)
# ==============================================================================

class ArthemyChannelMagnitudeTuner(BaseKrea2Node):
    """Vertical Tuning: Partition residual stream channels (d_model=6144) by static L2 energy into 12 logarithmic bands."""
    
    LOG_TIERS = [
        ("Band_01_Core_1pct", 0.01, "Top 1% highest-energy channels (Dominant macro-structure & primary subjects)"),
        ("Band_02_Top_3pct", 0.03, "1-3% energy tier (Primary forms & volume)"),
        ("Band_03_Top_6pct", 0.06, "3-6% energy tier (Secondary structures & composition)"),
        ("Band_04_Top_10pct", 0.10, "6-10% energy tier (Major anatomical & geometric layout)"),
        ("Band_05_Top_15pct", 0.15, "10-15% energy tier (Mid-range features & perspective)"),
        ("Band_06_Top_22pct", 0.22, "15-22% energy tier (Context, environment & shading)"),
        ("Band_07_Top_30pct", 0.30, "22-30% energy tier (Background composition & atmosphere)"),
        ("Band_08_Top_40pct", 0.40, "30-40% energy tier (Texture foundations)"),
        ("Band_09_Top_52pct", 0.52, "40-52% energy tier (Secondary details & surface tones)"),
        ("Band_10_Top_66pct", 0.66, "52-66% energy tier (Subtle artistic inflections)"),
        ("Band_11_Top_82pct", 0.82, "66-82% energy tier (Sparse style cues & stylistic nuance)"),
        ("Band_12_Rare_100pct", 1.00, "82-100% lowest-energy channels (Rare style traits, fine grain & micro-texture)"),
    ]
    BAND_NAMES = [t[0] for t in LOG_TIERS]

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": ("MODEL",),
                "mode": (["Soft Value", "Real Value"], {"default": "Soft Value"}),
            }
        }
        for name, _pct, desc in cls.LOG_TIERS:
            inputs["required"][name] = (
                "FLOAT",
                {"default": 0.00, "min": -99.00, "max": 99.00, "step": 0.01, "tooltip": desc}
            )
        # Deliberately last: ComfyUI stores widget values positionally, so a control inserted
        # above the band rows would shift every saved workflow's sliders by one.
        inputs["required"]["channel_path"] = (CHANNEL_PATH_CHOICES, {
            "default": CHANNEL_PATH_BOTH, "tooltip": CHANNEL_PATH_TOOLTIP})
        inputs["required"]["channel_scope"] = (MODEL_SUBCOMPONENTS, {
            "default": CHANNEL_SCOPE_DEFAULT, "tooltip": CHANNEL_SCOPE_TOOLTIP})
        return inputs

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "info")
    FUNCTION = "tune_magnitude"
    CATEGORY = "Arthemy/Krea2 Tuners"

    def tune_magnitude(self, model: Any, mode: str = "Soft Value",
                       channel_path: str = CHANNEL_PATH_BOTH,
                       channel_scope: str = CHANNEL_SCOPE_DEFAULT, **kwargs: float) -> Tuple[Any, str]:
        t_start = time.time()
        m = model.clone()
        patcher = get_patcher(m)
        base_sd = patcher.model.state_dict() if hasattr(patcher, "model") else {}

        active_sliders = {name: kwargs.get(name, 0.0) for name in self.BAND_NAMES if kwargs.get(name, 0.0) != 0.0}
        if not active_sliders:
            return (m, "Channel Magnitude Tuner: bypassed (all band offsets 0.0)")

        target_dim = Krea2Config.probe_hidden_size(base_sd)

        # Phase 1: rank the channels by energy. The metric follows the chosen path - row norms
        # of the writing matrices, column norms of the reading ones - and the result is cached
        # per (checkpoint, axis), so this is paid once per model rather than once per slider.
        # On "Both" the ranking comes from the read side: it is the axis that decides which
        # channels the sub-network listens to, and the write side then re-amplifies the same
        # channels. One ranking, not an average of two - averaging would blur the ordering the
        # bands are cut from.
        scale_writes, scale_reads = channel_scale_axes_for_path(channel_path)
        profile_axis = 1 if scale_reads else 0
        sorted_indices, profile_note = resolve_channel_profile(
            patcher, base_sd, target_dim, profile_axis, channel_scope)
        if sorted_indices is None:
            return (m, f"Channel Magnitude Tuner: no {target_dim}-wide matrices in "
                       f"'{channel_scope}' to profile ({profile_note})")

        # Phase 2: Build per-band logarithmic partition and resolve channel_scales vector.
        tier_indices: List[torch.Tensor] = []
        prev_idx = 0
        for _name, pct, _desc in self.LOG_TIERS:
            end_idx = int(round(pct * target_dim))
            tier_indices.append(sorted_indices[prev_idx:end_idx])
            prev_idx = end_idx

        channel_scales = torch.ones(target_dim, dtype=torch.float32)
        for i, (name, _pct, _desc) in enumerate(self.LOG_TIERS):
            slider_val = kwargs.get(name, 0.0)
            if slider_val != 0.0:
                scale_mult = soft_target_weight(slider_val, mode)
                channel_scales[tier_indices[i]] = scale_mult

        if torch.allclose(channel_scales, torch.ones_like(channel_scales)):
            return (m, "Channel Magnitude Tuner: bypassed (effective scales are 1.0)")

        # Phase 3: attach the gain as a shared 1-D adapter (see inject_channel_scales).
        applied, quant_skipped, note = inject_channel_scales(
            m, patcher, base_sd, channel_scales, target_dim, channel_path,
            "Arthemy Channel Magnitude Tuner", channel_scope)
        warn_if_quantized_skipped("Arthemy Channel Magnitude Tuner", quant_skipped)

        # The gain is a shared 1-D adapter, not a tensor the Preset Saver can serialize, so
        # without this recipe the tuning simply did not survive a save/load - the preset came
        # back empty and the saver even reported the adapters as "N LoRA tensors excluded".
        # The band offsets are what gets stored; the channel RANKING is deliberately not, so a
        # replay re-profiles the checkpoint it is actually loaded onto.
        if applied:
            append_recipe(patcher, "arthemy_channel_recipes", {
                "domain": "model",
                "mode": mode,
                "channel_path": channel_path,
                "channel_scope": channel_scope,
                "bands": {name: float(kwargs.get(name, 0.0)) for name in self.BAND_NAMES
                          if kwargs.get(name, 0.0) != 0.0},
            })

        elapsed = time.time() - t_start
        info = (f"Channel Magnitude Tuned | {channel_scope} | {channel_path} | "
                f"d_model {target_dim} | Patches: {applied} | Active Tiers: {len(active_sliders)} | "
                f"{note} | {profile_note} | Time: {elapsed:.3f}s")
        logger.info(f"[Arthemy Profiler] {info}")
        return (m, info)


# ==============================================================================
# NODE MAPPINGS & REGISTRATION (Suite Grid + Savers & Utilities)
# ==============================================================================
NODE_CLASS_MAPPINGS = {
    # 🟦 Model Tools (Tier 1-4 Quartet)
    "ArthemyKrea2ModelTuner": ArthemyKrea2ModelTuner,

    # 🟦 Model Tools (Tier 1-4 Quartet + New Rotators & 5D Tuner)
    "ArthemyKrea2ModelRotator": ArthemyKrea2ModelRotator,
    "ArthemyKrea2ModelChaosRotator": ArthemyKrea2ModelChaosRotator,
    "Arthemy5DTuner": Arthemy5DTuner,
    "Arthemy5DCLIPTuner": Arthemy5DCLIPTuner,

    # 🟪 Vertical Tuning Tools
    "ArthemyChannelMagnitudeTuner": ArthemyChannelMagnitudeTuner,

    # 🟨 CLIP Tools (Tier 1-4 Quartet + New Rotators)
    "ArthemyKrea2CLIPRotator": ArthemyKrea2CLIPRotator,
    "ArthemyKrea2CLIPChaosRotator": ArthemyKrea2CLIPChaosRotator,

    # 🟪 LoRA Tools
    "ArthemyLoRAto5DExtractor": ArthemyLoRAto5DExtractor,

    "ArthemyKrea2ModelBlockSurgeonTuner": ArthemyKrea2ModelBlockSurgeonTuner,
    "ArthemyKrea2ModelChaosBlockSurgeonTuner": ArthemyKrea2ModelChaosBlockSurgeonTuner,
    "ArthemyKrea2LatentSpaceRotator": ArthemyKrea2LatentSpaceRotator,

    # 🟨 CLIP Tools (Tier 1-4 Quartet)
    "ArthemyKrea2CLIPTuner": ArthemyKrea2CLIPTuner,
    "ArthemyKrea2CLIPBlockSurgeonTuner": ArthemyKrea2CLIPBlockSurgeonTuner,
    "ArthemyKrea2CLIPChaosBlockSurgeonTuner": ArthemyKrea2CLIPChaosBlockSurgeonTuner,
    "ArthemyKrea2CLIPSpaceRotator": ArthemyKrea2CLIPSpaceRotator,

    # 🟪 LoRA Tools (Tier 1-3 Trio)
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

    # Legacy Backward-Compatibility Aliases
    "ArthemyKrea2ModelBlockMerger": ArthemyKrea2ModelBlockSurgeonTuner,
    "ArthemyKrea2CLIPBlockMerger": ArthemyKrea2CLIPBlockSurgeonTuner,
    "ArthemyKrea2ModelChaosBlockMerger": ArthemyKrea2ModelChaosBlockSurgeonTuner,
    "ArthemyKrea2CLIPChaosBlockMerger": ArthemyKrea2CLIPChaosBlockSurgeonTuner,
    "ArthemyKrea2ModelChaosBlockTuner": ArthemyKrea2ModelChaosBlockSurgeonTuner,
    "ArthemyKrea2CLIPChaosBlockTuner": ArthemyKrea2CLIPChaosBlockSurgeonTuner,
    "ArthemyKrea2ModelRestorer": ArthemyKrea2ModelTuner,
    "ArthemyKrea2CLIPRestorer": ArthemyKrea2CLIPTuner,
    "ArthemyKrea2IsolatedLoraBlockLoader": ArthemyKrea2LoraBlockLoader,
    "ArthemyKrea2LoadChaosLoRA": ArthemyKrea2LoadChaosLoraBlockSurgeon,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # 🟪 Model Tools (Tier 1-4 Quartet)
    "ArthemyKrea2ModelTuner": "🟪✨ Model Tuner",

    # 🟪 Model Tools
    "ArthemyKrea2ModelRotator": "🟪🌿 Model Axis Rotator",
    "ArthemyKrea2ModelChaosRotator": "🟪🌀 Model Chaos Rotator",
    "Arthemy5DTuner": "🟪🧬 5D Model Tuner",
    "Arthemy5DCLIPTuner": "🟨🧬 5D CLIP Tuner",

    # 🟪 Vertical Tuning Tools
    "ArthemyChannelMagnitudeTuner": "🟪📊 Channel Magnitude Tuner",

    # 🟨 CLIP Tools
    "ArthemyKrea2CLIPRotator": "🟨🌿 CLIP Axis Rotator",
    "ArthemyKrea2CLIPChaosRotator": "🟨🌀 CLIP Chaos Rotator",

    # 🩷 LoRA Tools
    "ArthemyLoRAto5DExtractor": "🩷🧬 LoRA-to-5D Extractor",

    "ArthemyKrea2ModelBlockSurgeonTuner": "🟪🔬 Model Sub-Block Tuner",
    "ArthemyKrea2ModelChaosBlockSurgeonTuner": "🟪🌪️ Model Sub-Block Chaos Tuner",
    "ArthemyKrea2LatentSpaceRotator": "🟪🧭 Model Compass Rotator",

    # 🟨 CLIP Tools (Tier 1-4 Quartet)
    "ArthemyKrea2CLIPTuner": "🟨✨ CLIP Tuner",
    "ArthemyKrea2CLIPBlockSurgeonTuner": "🟨🔬 CLIP Sub-Block Tuner",
    "ArthemyKrea2CLIPChaosBlockSurgeonTuner": "🟨🌪️ CLIP Sub-Block Chaos Tuner",
    "ArthemyKrea2CLIPSpaceRotator": "🟨🧭 CLIP Compass Rotator",

    # 🩷 LoRA Tools (Tier 1-3 Trio)
    "ArthemyKrea2LoraBlockLoader": "🩷🔮 LoRA Block Loader",
    "ArthemyKrea2LoadSubBlockLora": "🩷🔬 Load Sub-Block LoRA",
    "ArthemyKrea2LoadChaosLoraBlockSurgeon": "🩷🌪️ Load Sub-Block Chaos LoRA",

    # Savers & Utilities
    "ArthemyKrea2ModelSaver": "🟪💾 Model Saver",
    "ArthemyKrea2CLIPSaver": "🟨💾 CLIP Saver",
    "ArthemyKrea2ModelBaker": "🟪🟨 Model Baker",
    "ArthemyKrea2ModelVisualizer": "🟪📊 Model Visualizer",
    "ArthemyKrea2CLIPVisualizer": "🟨📊 CLIP Visualizer",
    "ArthemyKrea2ResetPatcher": "🟪🟨🔄 Reset Patcher",

    # Presets
    "ArthemyKrea2PresetSaver": "🟪🟨💾 Preset Saver",
    "ArthemyKrea2PresetLoader": "🟪🟨📂 Preset Loader",

    # Legacy Display Aliases
    "ArthemyKrea2ModelBlockMerger": "🟪🔬 Model Sub-Block Tuner (Legacy)",
    "ArthemyKrea2CLIPBlockMerger": "🟨🔬 CLIP Sub-Block Tuner (Legacy)",
    "ArthemyKrea2ModelChaosBlockMerger": "🟪🌪️ Model Sub-Block Chaos Tuner (Legacy Merger)",
    "ArthemyKrea2CLIPChaosBlockMerger": "🟨🌪️ CLIP Sub-Block Chaos Tuner (Legacy Merger)",
    "ArthemyKrea2ModelChaosBlockTuner": "🟪🌪️ Model Sub-Block Chaos Tuner (Legacy)",
    "ArthemyKrea2CLIPChaosBlockTuner": "🟨🌪️ CLIP Sub-Block Chaos Tuner (Legacy)",
    "ArthemyKrea2ModelRestorer": "🟪✨ Model Tuner (Legacy)",
    "ArthemyKrea2CLIPRestorer": "🟨✨ CLIP Tuner (Legacy)",
    "ArthemyKrea2IsolatedLoraBlockLoader": "🩷🔮 LoRA Block Loader (Legacy)",
    "ArthemyKrea2LoadChaosLoRA": "🩷🌪️ Load Sub-Block Chaos LoRA (Legacy)",
}

# ComfyUI picks this up automatically from the package exports; manually writing into
# nodes.EXTENSION_WEB_DIRS as well used to register the same directory twice.
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']




"""
Arthemy Geometry Engine
=======================

Pure-torch geometry back-end for the Arthemy Krea-2 Suite. This module deliberately
does NOT import ComfyUI: it only produces *mathematical descriptions* of weight deltas,
so it can be unit-tested standalone and so that the ComfyUI-specific patch packaging
lives in exactly one place (`normalize_geometry_patch` in Arthemy_Krea2_Tuner.py).

Every public producer returns a `LowRankDelta` (or a dict of them), i.e. the exact
factorization

    W_new = W + A @ B

with `A: (d_out, r)` and `B: (r, d_in)`. The caller wraps it into a native ComfyUI
`LoRAAdapter`, which is the ONLY low-rank format that modern ComfyUI's
`comfy.lora.calculate_weight` still understands.

    HISTORICAL BUG (fixed): this module used to return raw
    `(mat_B, mat_A, alpha, None, None)` tuples, which the suite wrapped as
    `("lora", payload)`. ComfyUI moved every low-rank patch type into
    `comfy.weight_adapter`, so `calculate_weight` no longer has a "lora" branch and
    silently fell through to `logging.warning("patch type not recognized lora <key>")`.
    Result: every Rotator node and the 5D Tuner were complete no-ops.

Design notes
------------
* Rotations are exact elements of SO(n): the generator is skew-symmetric and
  `torch.matrix_exp` of a skew matrix is orthogonal to machine precision. The
  Frobenius norm of the rotated subspace is therefore conserved exactly.
* Rotations act inside the *dominant SVD subspace* of the weight, not a random
  subspace: rotating directions the layer actually uses is what makes the effect
  visible in the generated image.
* Angles are used verbatim. The previous implementation attenuated them through
  `sin(theta * (i+1) / rank)`, which turned a requested 90 deg into a few degrees
  per plane and made the whole node imperceptible even when the patch format was
  correct.
"""

import base64
import math
import os
import zlib
from typing import Dict, List, Optional, Tuple

import torch
import safetensors
import safetensors.torch

__all__ = [
    "LowRankDelta",
    "STORAGE_DCT",
    "STORAGE_RAW",
    "dct_fidelity",
    "pack_f16",
    "unpack_f16",
    "build_modifier_from_factors",
    "ROTATION_RANK_MAP",
    "ROTATION_MODE_OUTPUT",
    "ROTATION_MODE_INPUT",
    "fast_style_compass_rotation",
    "fast_dual_orthogonal_rotation",
    "fast_chaos_orthogonal_rotation",
    "extract_lora_to_5d_dct",
    "synthesize_5d_patches_from_dct",
    "peek_lora_vector_lengths",
    "normalize_signed_angle",
]


def normalize_signed_angle(deg: float) -> float:
    """Folds any angle into (-180, +180], the half-turn nearest zero.

    Rotation angles used to be folded with `% 360`, which turns a user's -5 into 355.
    For a SINGLE plane family those are the same rotation, so it looked harmless - but
    the four axes deliberately share plane indices (structural_x drives (0,1),(2,3)...
    while structural_y drives (1,2),(3,4)...), and `_skew_from_planes` ACCUMULATES into
    shared entries. The resulting generator is a general skew matrix, and `expm` of one
    is not 360-periodic: -5 on two axes came out ~12x LARGER than +5, a violent rotation
    where the user asked for a small nudge the other way.

    Signed angles fix that outright, because negating the generator is exactly the
    inverse rotation: expm(-A) == expm(A).T for any skew A, shared planes included.
    """
    a = float(deg) % 360.0
    return a - 360.0 if a > 180.0 else a


# Shared rank presets, so the nodes, the engine and the visualizer agree on what
# "Light / Default / Heavy" means.
ROTATION_RANK_MAP = {"Light": 8, "Default": 16, "Heavy": 32}

ROTATION_MODE_OUTPUT = "Output Manifold (R @ W)"
ROTATION_MODE_INPUT = "Input Manifold (W @ R)"

_MIN_ROTATABLE_DIM = 4
_EPS = 1e-8


class LowRankDelta:
    """Exact rank-r description of a weight delta: `W_new = W + A @ B`.

    `meta` carries provenance for the visualizer / preset saver (rotation kind,
    representative angle, hue, measured relative Frobenius change, ...).
    """

    __slots__ = ("A", "B", "meta")

    def __init__(self, A: torch.Tensor, B: torch.Tensor, meta: Optional[dict] = None):
        self.A = A
        self.B = B
        self.meta = dict(meta or {})

    @property
    def rank(self) -> int:
        return int(self.A.shape[1])

    def dense(self, shape=None) -> torch.Tensor:
        dense = self.A.to(torch.float32) @ self.B.to(torch.float32)
        return dense.reshape(shape) if shape is not None else dense

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LowRankDelta(rank={self.rank}, meta={self.meta})"


# ==============================================================================
# 1. INTERNAL HELPERS
# ==============================================================================

def _as_2d(weight: torch.Tensor) -> Optional[torch.Tensor]:
    """Flattens conv kernels to (d_out, d_in_total). Returns None for 0/1-D tensors.

    1-D tensors (RMSNorm scales, biases, modulation vectors) have no plane to rotate:
    an orthogonal rotation of a 1-D space is +/-1, so the NORMS group is intentionally
    left untouched by every rotator.
    """
    if weight is None or not isinstance(weight, torch.Tensor) or weight.ndim < 2:
        return None
    d_out = weight.shape[0]
    return weight.detach().to(torch.float32).reshape(d_out, -1)


def _resolve_rank(requested: int, min_dim: int) -> int:
    """Clamps the subspace rank to something the tensor can actually support (even, >= 2)."""
    rank = int(max(2, min(int(requested), min_dim)))
    if rank % 2:
        rank -= 1
    return max(2, rank)


_SVD_BASIS_SEED = 0x5EEDCAFE
_SVD_OVERSAMPLE = 8
_SVD_POWER_ITERS = 4


def _truncated_svd(W2d: torch.Tensor, rank: int,
                   niter: int = _SVD_POWER_ITERS) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic randomized truncated SVD (Halko range finder + power iterations).

    Two reasons this is hand-rolled instead of calling `torch.svd_lowrank`:

    1. DETERMINISM. `torch.svd_lowrank` draws its random projection from the *global*
       torch RNG and accepts no generator. Two runs of the same rotator on the same
       weight therefore found slightly different dominant subspaces, so a preset could
       never reproduce the image it was saved from (~0.7% drift per layer, compounding
       across a stack of rotators). Here the projection comes from a fixed-seed
       generator keyed only on the tensor shape, so the subspace is a pure function of
       the weight.
    2. ACCURACY. `svd_lowrank`'s default `niter=2` with `q == rank` is noticeably wrong
       on the tail singular values; oversampling by 8 and truncating afterwards gives a
       much better basis for the same practical cost.
    """
    m, n = W2d.shape
    min_dim = min(m, n)
    q = int(min(min_dim, rank + _SVD_OVERSAMPLE))

    g = torch.Generator(device="cpu").manual_seed((_SVD_BASIS_SEED ^ (m * 73856093) ^ (n * 19349663)) % (2 ** 31))
    omega = torch.randn(n, q, generator=g, dtype=torch.float32).to(W2d.device)

    Y = W2d @ omega
    for _ in range(max(0, niter)):
        Q, _ = torch.linalg.qr(Y)
        Y = W2d @ (W2d.transpose(-2, -1) @ Q)
    Q, _ = torch.linalg.qr(Y)

    B = Q.transpose(-2, -1) @ W2d                 # (q, n)
    Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Ub

    return U[:, :rank].contiguous(), S[:rank].contiguous(), Vh[:rank, :].t().contiguous()


def _skew_from_planes(rank: int, planes: List[Tuple[int, int, float]],
                      device, dtype=torch.float32) -> torch.Tensor:
    """Builds a skew-symmetric generator from explicit (i, j, angle_rad) planes.

    Angles accumulate when several plane sets touch the same pair, which is exactly
    what makes the four rotation axes composable.
    """
    A = torch.zeros((rank, rank), dtype=dtype, device=device)
    for i, j, ang in planes:
        if i == j or ang == 0.0 or i >= rank or j >= rank:
            continue
        A[i, j] += ang
        A[j, i] -= ang
    return A


def _finalize(U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, R: torch.Tensor,
              W2d: torch.Tensor, meta: dict, rotate_input: bool = False) -> Optional[LowRankDelta]:
    """Turns a subspace rotation R into the exact factorized delta.

    Output manifold (default) - rotate the row / output space:
        W ~ U diag(S) V^T   ->   W_new = U R diag(S) V^T
        delta = U (R - I) diag(S) V^T = A @ B,  A = U (R-I) diag(S),  B = V^T

    Input manifold - rotate the column / input space. Because
    `W (V R V^T + (I - V V^T)) = U diag(S) R V^T`, the same factorization applies with
    the singular values on the other side:
        delta = U diag(S) (R - I) V^T,  A = U diag(S) (R-I),  B = V^T
    """
    rank = R.shape[0]
    eye = torch.eye(rank, dtype=R.dtype, device=R.device)
    delta_R = R - eye
    if float(delta_R.abs().max()) < _EPS:
        return None

    if rotate_input:
        A = U @ (S.unsqueeze(1) * delta_R)
    else:
        A = U @ (delta_R * S.unsqueeze(0))
    B = V.t()

    # Measured relative Frobenius change: computed in O(r^3) from Grammians
    # sqrt(sum((A^T A) * (B B^T))) without materializing dense [m, n] product
    w_norm = float(torch.linalg.norm(W2d)) + _EPS
    AtA = A.t() @ A
    BBt = B @ B.t()
    d_norm = float(torch.sqrt(torch.clamp((AtA * BBt).sum(), min=0.0)))
    meta = dict(meta)
    meta["relative_delta"] = round(d_norm / w_norm, 6)
    meta["subspace_rank"] = rank

    return LowRankDelta(A.contiguous().to(torch.bfloat16),
                        B.contiguous().to(torch.bfloat16),
                        meta)


def _deterministic_seed(layer_name: str, seed: int) -> int:
    """CRC32-based, process-independent seed.

    `hash(str)` is randomized per interpreter run (PYTHONHASHSEED), so the previous
    implementation produced a different "deterministic" chaos rotation on every
    ComfyUI restart and no preset could ever reproduce a result.
    """
    return int((zlib.crc32((layer_name or "").encode("utf-8")) ^ (int(seed) & 0xFFFFFFFF)) % (2 ** 31))


# ==============================================================================
# 2A. CONTINUOUS 2D POLAR STYLE COMPASS (Latent / CLIP Space Rotator)
# ==============================================================================

def fast_style_compass_rotation(weight_tensor: torch.Tensor,
                                angle_deg: float = 15.0,
                                hue: float = 180.0,
                                depth: str = "Default",
                                seed: int = 42,
                                rank: int = 16,
                                mode: str = ROTATION_MODE_OUTPUT,
                                layer_name: str = "") -> Optional[LowRankDelta]:
    """Continuous 2D polar "style compass" rotation.

    ``angle_deg`` (tilt, 0-90) is how far from the base model you travel;
    ``hue`` (0-360) is the *direction* of travel and blends smoothly between two
    orthogonal plane families inside the dominant subspace:

        G(hue) = cos(hue) * G_local + sin(hue) * G_global
        R      = expm(radians(angle) * G(hue))

    ``G_local`` couples adjacent principal directions, ``G_global`` couples each
    direction with its counterpart half a subspace away. Both are unit-entry skew
    generators, so ``R`` is exactly orthogonal for every (angle, hue) pair and the
    Frobenius norm of the rotated subspace is conserved.

        PREVIOUS BEHAVIOUR: the basis came from a QR decomposition of Gaussian noise,
        i.e. a *random* 2r-dimensional subspace of a 3072-dimensional output space.
        Rotating directions the layer barely uses changes the weights by ~1% in a
        direction with almost no functional meaning, which is why the node "worked"
        yet produced near-identical images. Aligning the rotation with the dominant
        singular directions is what makes the same angle actually visible.

    ``seed`` only permutes the plane pairing, and does so identically for every layer,
    so a given seed is a reproducible style flavour that stays coherent across blocks.
    """
    W2d = _as_2d(weight_tensor)
    if W2d is None or float(angle_deg) == 0.0:
        return None

    d_out, d_in = W2d.shape
    min_dim = min(d_out, d_in)
    if min_dim < _MIN_ROTATABLE_DIM:
        return None

    rotate_input = isinstance(mode, str) and mode.strip().lower().startswith("input")
    requested = ROTATION_RANK_MAP.get(depth, rank) if isinstance(depth, str) else rank
    r = _resolve_rank(requested, min_dim)
    half = r // 2

    theta = math.radians(float(angle_deg))
    phi = math.radians(float(hue))
    c_local, c_global = math.cos(phi), math.sin(phi)

    # Layer-independent, seed-dependent pairing keeps the style coherent across blocks.
    g = torch.Generator(device="cpu").manual_seed(int(seed) % (2 ** 31))
    perm = torch.randperm(r, generator=g).tolist()

    with torch.no_grad():
        U, S, V = _truncated_svd(W2d, r)

        planes: List[Tuple[int, int, float]] = []
        n_pairs = max(1, half // 2)
        if abs(c_local) > _EPS:
            ang = theta * c_local
            # Local family uses the lower disjoint half of the subspace [0 .. half-1]
            planes += [(perm[2 * i], perm[2 * i + 1], ang) for i in range(n_pairs)]
        if abs(c_global) > _EPS:
            ang = theta * c_global
            # Global family uses the upper disjoint half of the subspace [half .. r-1]
            planes += [(perm[half + 2 * i], perm[half + 2 * i + 1], ang) for i in range(n_pairs)]

        A_skew = _skew_from_planes(r, planes, device=W2d.device)
        if float(A_skew.abs().max()) < _EPS:
            return None
        R = torch.matrix_exp(A_skew)

        meta = {
            "kind": "rotation",
            "rotation_type": "style_compass",
            "angle": round(float(angle_deg), 2),
            "hue": round(float(hue) % 360.0, 1),
            "rotation_mode": ROTATION_MODE_INPUT if rotate_input else ROTATION_MODE_OUTPUT,
            "depth_reach": depth,
            "seed": int(seed),
            "layer_name": layer_name,
        }
        return _finalize(U, S, V, R, W2d, meta, rotate_input=rotate_input)


# ==============================================================================
# 2B. DUAL ORTHOGONAL LIE ROTATION (Model / CLIP Axis Rotator)
# ==============================================================================

def fast_dual_orthogonal_rotation(weight_tensor: torch.Tensor,
                                  structural_x: float = 0.0,
                                  structural_y: float = 0.0,
                                  tensor_x: float = 0.0,
                                  tensor_y: float = 0.0,
                                  depth_rank: int = 16,
                                  layer_name: str = "") -> Optional[LowRankDelta]:
    """Exact, energy-conserving dual orthogonal Lie rotation of the dominant subspace.

    The four controls drive four *independent* families of rotation planes inside the
    rank-r dominant subspace, so each one has a distinct, reproducible signature:

    * ``structural_x`` - adjacent planes (0,1), (2,3), ... : rotation *within* each
      component plane. The most local, "voice of the layer" control.
    * ``structural_y`` - offset planes (1,2), (3,4), ... : couples neighbouring
      components, i.e. tilts a branch into the residual trunk.
    * ``tensor_x``     - long-range planes (i, i + r/2) : global channel phase.
    * ``tensor_y``     - mirrored planes (i, r-1-i) : pairwise channel Givens.

    Angles are used verbatim in radians (0-360 deg). `R = expm(skew)` is orthogonal to
    machine precision, so the Frobenius norm of the rotated subspace is preserved.
    """
    W2d = _as_2d(weight_tensor)
    if W2d is None:
        return None

    sx = normalize_signed_angle(structural_x)
    sy = normalize_signed_angle(structural_y)
    tx = normalize_signed_angle(tensor_x)
    ty = normalize_signed_angle(tensor_y)
    # abs(), not the raw values: with signed angles `max(-5, 0, 0, 0)` is 0, so the old
    # early-out silently threw away every rotation that only turned the negative way.
    if max(abs(sx), abs(sy), abs(tx), abs(ty)) == 0.0:
        return None

    d_out, d_in = W2d.shape
    min_dim = min(d_out, d_in)
    if min_dim < _MIN_ROTATABLE_DIM:
        return None

    rank = _resolve_rank(depth_rank, min_dim)
    half = rank // 2

    with torch.no_grad():
        U, S, V = _truncated_svd(W2d, rank)

        planes: List[Tuple[int, int, float]] = []
        if sx:
            ang = math.radians(sx)
            planes += [(2 * i, 2 * i + 1, ang) for i in range(half)]
        if sy:
            ang = math.radians(sy)
            planes += [(2 * i + 1, 2 * i + 2, ang) for i in range(half - 1)]
        if tx:
            ang = math.radians(tx)
            planes += [(i, i + half, ang) for i in range(half)]
        if ty:
            ang = math.radians(ty)
            # Cross-subspace rotation offset by half//2 to be strictly disjoint from sx, sy and tx at any rank
            shift = max(1, half // 2)
            planes += [(i, half + ((i + shift) % half), ang) for i in range(half)]

        A_skew = _skew_from_planes(rank, planes, device=W2d.device)
        if float(A_skew.abs().max()) < _EPS:
            return None
        R = torch.matrix_exp(A_skew)

        # Representative angle for the HUD: root-mean-square rotation angle across active planes.
        # Frobenius norm of A_skew with k planes of angle theta is sqrt(2 * k * theta^2).
        num_planes = max(1, len(planes))
        total_rad = float(torch.linalg.norm(A_skew) / math.sqrt(2.0 * num_planes))
        meta = {
            "kind": "rotation",
            "rotation_type": "dual_lie",
            "angle": round(math.degrees(total_rad), 2),
            "structural_x": sx, "structural_y": sy,
            "tensor_x": tx, "tensor_y": ty,
            # Hue lets the visualizer colour-code the flavour of the rotation:
            # structural-dominant vs tensor-dominant.
            "hue": round((sx + sy * 2.0 + tx * 3.0 + ty * 4.0) % 360.0, 1),
            "depth_reach": depth_rank,
            "layer_name": layer_name,
        }
        return _finalize(U, S, V, R, W2d, meta)


# ==============================================================================
# 3. HARMONIC CHAOS ROTATION (Model / CLIP Chaos Rotator)
# ==============================================================================

def fast_chaos_orthogonal_rotation(weight_tensor: torch.Tensor,
                                   seed: int = 42,
                                   chaos_strength: float = 0.35,
                                   harmonic_coherence: float = 0.5,
                                   depth_rank: int = 16,
                                   layer_name: str = "") -> Optional[LowRankDelta]:
    """Deterministic harmonic chaos rotation for spontaneous exploration.

    A random permutation of the subspace axes is paired up into r/2 disjoint planes and
    each plane gets its own random angle in +/- (chaos_strength * 90 deg).
    ``harmonic_coherence`` snaps those *angles* towards 45 deg multiples.

        The previous implementation snapped the raw entries of the skew matrix, which
        is not the same thing at all (the eigen-angles of a snapped matrix are not
        snapped), and it seeded the generator with `hash(layer_name)`, so nothing was
        actually reproducible across restarts.
    """
    W2d = _as_2d(weight_tensor)
    if W2d is None:
        return None
    if abs(float(chaos_strength)) < 1e-4:
        return None

    d_out, d_in = W2d.shape
    min_dim = min(d_out, d_in)
    if min_dim < _MIN_ROTATABLE_DIM:
        return None

    rank = _resolve_rank(depth_rank, min_dim)
    half = rank // 2
    coherence = max(0.0, min(1.0, float(harmonic_coherence)))

    g = torch.Generator(device="cpu").manual_seed(_deterministic_seed(layer_name, seed))
    perm = torch.randperm(rank, generator=g).tolist()
    raw = (torch.rand(half, generator=g, dtype=torch.float32) * 2.0 - 1.0)

    max_deg = 90.0 * abs(float(chaos_strength))
    snap_deg = 45.0
    while snap_deg > max_deg and snap_deg > 1.0:
        snap_deg /= 2.0
    angles_deg = []
    for v in raw.tolist():
        deg = v * max_deg
        if coherence > 0.0 and snap_deg > 0.0:
            snapped = round(deg / snap_deg) * snap_deg
            deg = (1.0 - coherence) * deg + coherence * snapped
        angles_deg.append(deg)

    with torch.no_grad():
        U, S, V = _truncated_svd(W2d, rank)
        planes = [(perm[2 * i], perm[2 * i + 1], math.radians(angles_deg[i])) for i in range(half)]
        A_skew = _skew_from_planes(rank, planes, device=W2d.device)
        if float(A_skew.abs().max()) < _EPS:
            return None
        R = torch.matrix_exp(A_skew)

        mean_abs = sum(abs(a) for a in angles_deg) / max(1, len(angles_deg))
        meta = {
            "kind": "rotation",
            "rotation_type": "chaos_lie",
            "angle": round(mean_abs, 2),
            "seed": int(seed),
            "chaos_strength": float(chaos_strength),
            "harmonic_coherence": coherence,
            # Chaos gets a warm hue band so it is visually distinct from dual rotations.
            "hue": float(_deterministic_seed(layer_name, seed) % 360),
            "depth_reach": depth_rank,
            "layer_name": layer_name,
            "is_chaos": True,
        }
        return _finalize(U, S, V, R, W2d, meta)


# ==============================================================================
# 4. HARMONIC DCT COMPRESSION OF A LoRA INTO 5 DIMENSIONS
# ==============================================================================

def _dct_matrix(N: int, num_coeffs: int, device="cpu") -> torch.Tensor:
    """Orthonormal DCT-II basis, `(num_coeffs, N)`.

    Building the basis once as a matrix replaces the previous per-coefficient Python
    loop (one `torch.cos` over N per coefficient, per vector, per layer) with a single
    matmul, which is what makes extraction of a full LoRA finish in seconds.
    """
    num_c = max(1, min(int(num_coeffs), N))
    n = torch.arange(N, dtype=torch.float32, device=device).unsqueeze(0)
    k = torch.arange(num_c, dtype=torch.float32, device=device).unsqueeze(1)
    basis = torch.cos((math.pi * k / N) * (n + 0.5))
    norm = torch.full((num_c, 1), math.sqrt(2.0 / N), dtype=torch.float32, device=device)
    norm[0, 0] = math.sqrt(1.0 / N)
    return basis * norm


def dct_1d(x: torch.Tensor, num_coeffs: int = 16) -> List[float]:
    """DCT-II of a 1-D tensor, truncated to the `num_coeffs` lowest frequencies."""
    x = x.detach().to(torch.float32).flatten()
    M = _dct_matrix(x.shape[0], num_coeffs, device=x.device)
    return [round(v, 6) for v in (M @ x).tolist()]


def idct_1d(coeffs: List[float], target_length: int, device="cpu",
            normalize: bool = True) -> torch.Tensor:
    """Reconstructs a 1-D wave from truncated DCT coefficients.

    `normalize=True` restores the unit norm of the reconstructed singular vector.
    Without it the reconstruction keeps only the energy of the retained harmonics
    (a few percent of a 3072-long singular vector), while the *full* singular value
    is still applied on top - so the synthesized delta magnitude was essentially
    arbitrary and mostly far too small to see.
    """
    M = _dct_matrix(target_length, len(coeffs), device=device)
    c = torch.tensor(coeffs, dtype=torch.float32, device=device)
    vec = M.t() @ c
    if normalize:
        n = torch.linalg.norm(vec)
        if float(n) > _EPS:
            vec = vec / n
    return vec


# ==============================================================================
# 4B. RAW FACTOR STORAGE
# ==============================================================================
# Why this exists. A truncated DCT is a low-pass filter, and the singular vectors of a
# weight delta are not smooth: their energy is spread across the whole frequency band.
# Keeping `order` of `N` coefficients therefore preserves only about `order / N` of the
# band, so the reconstructed direction has cosine ~= sqrt(order / N) with the real one.
# Measured on a 16384-long singular vector: order 16 -> 3.9% (theory 3.1%),
# order 1024 -> 25.2% (theory 25.0%). Energy goes as cos^2, so at the default order the
# harmonic modifier injects a direction ~99.9% uncorrelated with the one it claims to
# represent - for a real LoRA just as much as for a measured model residual.
#
# `storage="raw"` keeps the factors verbatim as base64-packed float16 instead. The file
# grows from tens of KB to a few MB, and in exchange the injection is faithful.
STORAGE_DCT = "dct"
STORAGE_RAW = "raw"


def dct_fidelity(order: int, length: int) -> float:
    """Expected cosine between a DCT-truncated singular vector and the original.

    Exposed so a node can warn honestly instead of silently injecting noise.
    """
    if length <= 0:
        return 0.0
    return min(1.0, math.sqrt(max(0, min(int(order), int(length))) / float(length)))


def pack_f16(mat: torch.Tensor) -> str:
    """Packs a 2-D float32 tensor as base64 float16 (row-major)."""
    arr = mat.detach().to(torch.float16).contiguous().cpu()
    return base64.b64encode(arr.numpy().tobytes()).decode("ascii")


def unpack_f16(blob: str, rows: int, cols: int, device="cpu") -> torch.Tensor:
    """Inverse of `pack_f16`, returned as float32."""
    raw = base64.b64decode(blob)
    t = torch.frombuffer(bytearray(raw), dtype=torch.float16)
    return t[: rows * cols].reshape(rows, cols).to(dtype=torch.float32, device=device)


_LORA_DOWN_SUFFIXES = ("lora_down.weight", "lora_A.weight", "lora_a.weight")
_LORA_UP_FOR_DOWN = {
    "lora_down.weight": "lora_up.weight",
    "lora_A.weight": "lora_B.weight",
    "lora_a.weight": "lora_b.weight",
}
_LORA_KEY_PREFIXES = (
    "diffusion_model.", "model.diffusion_model.", "lora_unet_", "transformer.",
    "text_encoders.", "cond_stage_model.", "lora_te_", "lora_te1_", "model.",
)
_CLIP_HINTS = ("text_encoder", "text_model", "lora_te", "cond_stage", "language_model", "t5", "qwen")


def _clean_lora_base(key: str) -> str:
    base = key
    for suffix in _LORA_DOWN_SUFFIXES:
        if base.endswith("." + suffix):
            base = base[: -len(suffix) - 1]
            break
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    for pfx in _LORA_KEY_PREFIXES:
        if base.startswith(pfx):
            base = base[len(pfx):]
    return base.strip(".").replace("_lora", "")


def _guess_domain(key: str) -> str:
    low = key.lower()
    return "clip" if any(h in low for h in _CLIP_HINTS) else "model"


def peek_lora_vector_lengths(lora_path: str) -> List[int]:
    """Shape-only scan of a LoRA's up/down pairs: the `d_out`/`d_in` every layer would get,
    without running the SVD `extract_lora_to_5d_dct` needs for the actual factors.

    Lets a caller decide raw-vs-DCT storage (via `dct_fidelity`) *before* paying for
    extraction, instead of extracting once to find out and possibly redoing it.

    Mirrors exactly how `extract_lora_to_5d_dct` derives its own dimensions:
    `delta = uw.flatten(1) @ dw.flatten(1)`, hence `d_out = up.shape[0]` and
    `d_in = prod(down.shape[1:])`. Reading only the down tensor - as the first version of
    this function did - never sees `d_out` at all (down is `(rank, d_in)`), so on any layer
    where `d_out > d_in` it under-reports the longest singular vector and the caller
    over-estimates DCT fidelity on precisely the vectors DCT truncates worst.
    """
    lengths: List[int] = []
    try:
        with safetensors.safe_open(lora_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for down_k in keys:
                up_k = None
                for suffix, up_suffix in _LORA_UP_FOR_DOWN.items():
                    if down_k.endswith(suffix):
                        up_k = down_k[: -len(suffix)] + up_suffix
                        break
                # No up partner: extract_lora_to_5d_dct skips this layer too, so it must
                # not contribute a length here either.
                if up_k is None or up_k not in keys:
                    continue
                try:
                    down_shape = f.get_slice(down_k).get_shape()
                    up_shape = f.get_slice(up_k).get_shape()
                except Exception:
                    continue
                if len(down_shape) < 2 or len(up_shape) < 1:
                    continue
                # d_in folds every trailing conv dimension in (in_ch * kh * kw), matching
                # flatten(start_dim=1); shape[1] alone undercounts convolutional layers.
                d_in = 1
                for s in down_shape[1:]:
                    d_in *= int(s)
                d_out = int(up_shape[0])
                # Same viability floor the extractor applies before running the SVD.
                if min(d_out, d_in) < 2:
                    continue
                lengths.append(d_out)
                lengths.append(d_in)
    except Exception:
        return []
    return lengths


def extract_lora_to_5d_dct(lora_path: str, num_harmonic_coeffs: int = 16,
                           num_dims: int = 5, storage: str = STORAGE_DCT) -> dict:
    """Compresses a LoRA into its `num_dims` dominant SVD directions.

    `storage=STORAGE_DCT` keeps each singular vector as `num_harmonic_coeffs` DCT
    harmonics: a modifier of a few dozen KB, but only ~sqrt(order/N) faithful to the
    real direction (see `dct_fidelity`).

    `storage=STORAGE_RAW` keeps the factors verbatim as base64 float16: a few MB, and
    an exact rank-`num_dims` reproduction of the LoRA's delta.

    Each layer entry records its `domain` ("model" / "clip") so the 5D Tuner nodes can
    target the right patcher.
    """
    storage = STORAGE_RAW if str(storage).lower().startswith("raw") else STORAGE_DCT
    dct_data = {
        "version": "5.0",
        "storage": storage,
        "source_lora": os.path.basename(lora_path),
        "rank": int(num_dims),
        "harmonic_order": int(num_harmonic_coeffs),
        "normalized": True,
        "layers": {},
    }

    with safetensors.safe_open(lora_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        down_keys = [k for k in keys if any(k.endswith(s) for s in _LORA_DOWN_SUFFIXES)]

        for down_k in sorted(down_keys):
            up_k = None
            alpha_k = None
            for suffix, up_suffix in _LORA_UP_FOR_DOWN.items():
                if down_k.endswith(suffix):
                    up_k = down_k[: -len(suffix)] + up_suffix
                    alpha_k = down_k[: -len(suffix)] + "alpha"
                    break
            if up_k is None or up_k not in keys:
                continue

            dw = f.get_tensor(down_k).to(torch.float32)
            uw = f.get_tensor(up_k).to(torch.float32)
            lora_rank = dw.shape[0]
            alpha = float(f.get_tensor(alpha_k).item()) if alpha_k in keys else float(lora_rank)
            scale = (alpha / lora_rank) if lora_rank > 0 else 1.0

            delta = (uw.flatten(start_dim=1) @ dw.flatten(start_dim=1)) * scale
            d_out, d_in = delta.shape
            min_dim = min(d_out, d_in)
            if min_dim < 2 or float(torch.linalg.norm(delta)) < 1e-5:
                del dw, uw, delta
                continue

            q_rank = min(int(num_dims), min_dim)
            U, S, V = _truncated_svd(delta, q_rank)

            clean_base = _clean_lora_base(down_k)
            entry = {
                "domain": _guess_domain(down_k),
                "d_out": int(d_out),
                "d_in": int(d_in),
                "singular_values": [round(float(x), 6) for x in S.tolist()],
            }
            if storage == STORAGE_RAW:
                entry["u_raw"] = pack_f16(U[:, :q_rank].t())   # (rank, d_out)
                entry["v_raw"] = pack_f16(V[:, :q_rank].t())   # (rank, d_in)
            else:
                u_mat = _dct_matrix(d_out, num_harmonic_coeffs)
                v_mat = _dct_matrix(d_in, num_harmonic_coeffs)
                entry["u_dct"] = [[round(v, 6) for v in (u_mat @ U[:, i]).tolist()] for i in range(q_rank)]
                entry["v_dct"] = [[round(v, 6) for v in (v_mat @ V[:, i]).tolist()] for i in range(q_rank)]
            dct_data["layers"][clean_base] = entry
            del U, S, V, delta, dw, uw

    return dct_data


def synthesize_5d_patches_from_dct(dct_data: dict,
                                   dimension_weights: List[float],
                                   master_multiplier: float = 1.0,
                                   device="cpu",
                                   domain: Optional[str] = None,
                                   layer_filter=None) -> Dict[str, LowRankDelta]:
    """Rebuilds the low-rank 5D perturbations from the harmonic DCT payload.

    Returns `{clean_layer_name: LowRankDelta}` keyed WITHOUT any framework prefix, so
    the calling node owns key resolution (`diffusion_model.` for the UNet, nothing for
    a wrapped text encoder). Previously this function hardcoded
    `f"diffusion_model.{layer}.weight"`, which made a CLIP 5D tuner impossible.

    `domain` filters entries by their recorded domain; `layer_filter(clean_layer)` is an
    optional predicate used by the nodes to honour block / sub-component selection.
    """
    if not dct_data or "layers" not in dct_data:
        return {}

    weights = [float(w) * float(master_multiplier) for w in list(dimension_weights)]
    if all(abs(w) < 1e-5 for w in weights):
        return {}

    # v3 payloads were written without unit-norm reconstruction; renormalizing them is
    # strictly an improvement, but flag it so the magnitude change is explainable.
    normalize = bool(dct_data.get("normalized", True))

    patches: Dict[str, LowRankDelta] = {}
    for clean_layer, l_info in dct_data["layers"].items():
        if domain is not None and l_info.get("domain", "model") != domain:
            continue
        if layer_filter is not None and not layer_filter(clean_layer):
            continue

        d_out = int(l_info["d_out"])
        d_in = int(l_info["d_in"])
        s_vals = l_info["singular_values"]
        is_raw = ("u_raw" in l_info and "v_raw" in l_info)

        if is_raw:
            n_dims = len(s_vals)
            U_raw = unpack_f16(l_info["u_raw"], n_dims, d_out, device=device)
            V_raw = unpack_f16(l_info["v_raw"], n_dims, d_in, device=device)
        else:
            u_dct_list = l_info["u_dct"]
            v_dct_list = l_info["v_dct"]
            n_dims = min(len(s_vals), len(u_dct_list), len(v_dct_list))

        u_cols, v_cols, scaled_s = [], [], []
        for i in range(n_dims):
            w_i = weights[i] if i < len(weights) else 0.0
            if abs(w_i) <= 1e-5 or abs(s_vals[i]) <= 1e-6:
                continue
            if is_raw:
                # Verbatim factors: only the float16 rounding needs correcting.
                u_v, v_v = U_raw[i], V_raw[i]
                un, vn = torch.linalg.norm(u_v), torch.linalg.norm(v_v)
                u_cols.append(u_v / un if float(un) > _EPS else u_v)
                v_cols.append(v_v / vn if float(vn) > _EPS else v_v)
            else:
                u_cols.append(idct_1d(u_dct_list[i], d_out, device=device, normalize=normalize))
                v_cols.append(idct_1d(v_dct_list[i], d_in, device=device, normalize=normalize))
            scaled_s.append(float(s_vals[i]) * w_i)

        if not u_cols:
            continue

        U_mat = torch.stack(u_cols, dim=1)
        V_mat = torch.stack(v_cols, dim=1)
        S_vec = torch.tensor(scaled_s, dtype=torch.float32, device=device)

        A = (U_mat * S_vec.unsqueeze(0)).contiguous().to(torch.bfloat16)
        B = V_mat.t().contiguous().to(torch.bfloat16)
        patches[clean_layer] = LowRankDelta(A, B, {
            "kind": "five_d",
            "active_dims": len(scaled_s),
            "domain": l_info.get("domain", "model"),
            "source_lora": dct_data.get("source_lora", ""),
            "storage": STORAGE_RAW if is_raw else STORAGE_DCT,
        })

    return patches


def build_modifier_from_factors(entries: Dict[str, dict], source_name: str,
                                num_dims: int = 5) -> dict:
    """Assembles a raw-storage modifier from factors measured elsewhere.

    `entries[clean_layer]` must carry `U` (rank, d_out), `V` (rank, d_in), `s` (rank,)
    and optionally `domain`. This is what lets a modifier be built from the *measured
    residual between two checkpoints* rather than from a LoRA file - the same tool,
    pointed at a finetune instead of an adapter.
    """
    out = {
        "version": "5.0",
        "storage": STORAGE_RAW,
        "source_lora": source_name,
        "rank": int(num_dims),
        "normalized": True,
        "layers": {},
    }
    for layer, e in entries.items():
        U = e["U"] if isinstance(e["U"], torch.Tensor) else torch.as_tensor(e["U"])
        V = e["V"] if isinstance(e["V"], torch.Tensor) else torch.as_tensor(e["V"])
        sv = e["s"] if isinstance(e["s"], torch.Tensor) else torch.as_tensor(e["s"])
        r = int(min(num_dims, U.shape[0], V.shape[0], sv.shape[0]))
        out["layers"][layer] = {
            "domain": e.get("domain", "model"),
            "d_out": int(U.shape[1]),
            "d_in": int(V.shape[1]),
            "singular_values": [round(float(x), 6) for x in sv[:r].tolist()],
            "u_raw": pack_f16(U[:r].to(torch.float32)),
            "v_raw": pack_f16(V[:r].to(torch.float32)),
        }
    return out

# FP8 E4M3: decode uint8->float32 and encode f32->nearest uint8 (all 256 codes).
# Matches OCP/FP8 E4M3 for reference roundtrip; close to CUDA static_cast<fp8e4m3>(f32).

from __future__ import annotations

import numpy as np

# Match transformer_engine::detail::TypeExtrema (common.h) / nvfp4.cu kernels
FP8_E4M3_FMAX: float = 448.0
FP4_E2M1_FMAX: float = 6.0
S_ENC_NUMER: float = FP8_E4M3_FMAX * FP4_E2M1_FMAX  # 2688
FLT_MAX: float = 3.402823466e38
TINY: float = 1.17549435e-38


def _decode_e4m3_byte(b: int) -> float:
    u = b & 0xFF
    sign = u >> 7
    exp = (u >> 3) & 0x0F
    man = u & 0x07
    if exp == 0:
        if man == 0:
            return 0.0
        v = (man / 8.0) * (2.0 ** (-6))
    else:
        v = (1.0 + man / 8.0) * (2.0 ** (exp - 7))
    return -v if sign else v


# Precompute 256 float32 values for all E4M3 codes
_E4M3_TABLE: np.ndarray = np.array(
    [_decode_e4m3_byte(i) for i in range(256)], dtype=np.float32
)


def f32_to_e4m3_u8(x: np.ndarray) -> np.ndarray:
    """Round each element to nearest fp8e4m3 (by L_inf on 256 codes). x can be any shape."""
    x = np.asarray(x, dtype=np.float32)
    flat = x.ravel()[:, None]
    d = np.abs(flat - _E4M3_TABLE[None, :])
    out = np.argmin(d, axis=1).astype(np.uint8)
    return out.reshape(x.shape)


def e4m3_u8_to_f32(b: np.ndarray) -> np.ndarray:
    return _E4M3_TABLE[np.asarray(b, dtype=np.int32) & 0xFF].astype(np.float32)


def compute_S_enc_from_amax_1x64_like_te(amax: float, eps: float = TINY) -> float:
    """
    Same as compute_global_encode_scaling_factor_FP4 in core_nvfp4.cuh, but *amax* is
    the 1x64 local max (replaces per-tensor global amax in this ref).
    """
    a = float(amax)
    if a <= 0.0 or not np.isfinite(a):
        return 1.0
    safe = max(a, eps)
    g = S_ENC_NUMER / safe
    return float(min(g, FLT_MAX))


def compute_S_dec_f32_before_cast_te(block_amax: float, S_enc: float) -> float:
    """
    Unquantized S_dec = block_amax * (S_enc / 6) before cast to e4m3
    (compute_decoding_scaling_factor, quantization_SF in core_nvfp4.cuh).
    """
    return float(np.float32(block_amax * (S_enc * (1.0 / FP4_E2M1_FMAX))))


def f32_e4m3_f32(x: float) -> float:
    """Cast pipeline f32 -> fp8e4m3 -> f32, numpy."""
    u = f32_to_e4m3_u8(np.array([x], dtype=np.float32))
    return float(e4m3_u8_to_f32(u)[0])

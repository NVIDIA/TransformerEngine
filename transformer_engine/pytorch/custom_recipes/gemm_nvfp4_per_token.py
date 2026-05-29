# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Reference + production GEMM for the NVFP4 per-token quantization scheme.

Per-token GEMM reuses cuBLAS LT NVFP4 (no TE fork) + a trailing row-amax
post-scale. Each side is a (data, scale, row_amax) triple matching what
tex.nvfp4_per_token_quantize emits. See include/transformer_engine/nvfp4_per_token.h.
"""

from __future__ import annotations

from typing import Optional

import torch

# get_cublas_workspace is imported lazily inside nvfp4_per_token_gemm to
# avoid a circular import with cpp_extensions.gemm at module load time.
from transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4 import cast_from_fp4x2
from transformer_engine.pytorch.custom_recipes.quantization_nvfp4_per_token import (
    BLOCK_K,
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    _AMAX_FLOOR,
    RefNVFP4TensorPerToken,
)


__all__ = [
    "dequantize_nvfp4_per_token",
    "nvfp4_per_token_gemm_dequant",
    "nvfp4_per_token_gemm",
]


# Reference: dequantize + reference matmul.


def _validate_per_token_triple(
    data: torch.Tensor, scale: torch.Tensor, row_amax: torch.Tensor, side: str
) -> int:
    """Sanity-check one (data, scale, row_amax) triple; return K."""
    if data.ndim != 2 or scale.ndim != 2 or row_amax.ndim != 1:
        raise ValueError(
            f"{side}: expected 2D data/scale + 1D row_amax, got dims "
            f"data={data.ndim}, scale={scale.ndim}, row_amax={row_amax.ndim}"
        )
    rows = data.shape[0]
    K = data.shape[1] * 2  # FP4 packs 2 values/byte.
    if K % BLOCK_K != 0:
        raise ValueError(f"{side}: K={K} must be a multiple of BLOCK_K={BLOCK_K}")
    if scale.shape != (rows, K // BLOCK_K):
        raise ValueError(f"{side}: scale shape {tuple(scale.shape)} != ({rows}, {K // BLOCK_K})")
    if row_amax.shape != (rows,):
        raise ValueError(f"{side}: row_amax shape {tuple(row_amax.shape)} != ({rows},)")
    return K


def dequantize_nvfp4_per_token(
    data: torch.Tensor, scale: torch.Tensor, row_amax: torch.Tensor
) -> torch.Tensor:
    """Dequantize a per-token NVFP4 (data, scale, row_amax) triple to fp32.

    x[i, k] = code[i, k] * s_dec[i, k//16] * row_amax[i] / (FP4_MAX * E4M3_MAX).
    """
    K = _validate_per_token_triple(data, scale, row_amax, "dequant")
    rows = data.shape[0]

    codes = data.contiguous().view(dtype=torch.uint8)
    qf = cast_from_fp4x2(codes, torch.float32)

    if scale.dtype == torch.float8_e4m3fn:
        s_dec = scale.to(torch.float32)
    else:
        s_dec = scale.view(torch.float8_e4m3fn).to(torch.float32)

    inv_outer = row_amax.to(torch.float32) / (FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX)
    per_block_decode = s_dec * inv_outer.unsqueeze(-1)
    per_elem_decode = per_block_decode.repeat_interleave(BLOCK_K, dim=1)
    assert per_elem_decode.shape == (rows, K)
    return qf * per_elem_decode


def nvfp4_per_token_gemm_dequant(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    a_row_amax: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    b_row_amax: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference C = A @ B^T via dequant-then-fp32-matmul.

    Agrees with the cuBLAS LT path at TF32 precision (~1e-3 relative).
    Exists as executable docs of the math chain and a sanity oracle.
    """
    K_a = _validate_per_token_triple(a_data, a_scale, a_row_amax, "A")
    K_b = _validate_per_token_triple(b_data, b_scale, b_row_amax, "B")
    if K_a != K_b:
        raise ValueError(f"K mismatch between A and B: {K_a} vs {K_b}")

    a_fp32 = dequantize_nvfp4_per_token(a_data, a_scale, a_row_amax)
    b_fp32 = dequantize_nvfp4_per_token(b_data, b_scale, b_row_amax)
    c = a_fp32 @ b_fp32.t()
    return c.to(out_dtype)


# Production wrapper: cuBLAS LT NVFP4 GEMM + per-token post-scale.


def nvfp4_per_token_gemm(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    a_row_amax: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.Tensor,
    b_row_amax: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    out_dtype: torch.dtype = torch.bfloat16,
    a_sf_swizzled: bool = False,
    b_sf_swizzled: bool = False,
) -> torch.Tensor:
    """Production C = alpha * (A @ B^T) via cuBLAS LT NVFP4 + per-token post-scale.

    Binding swizzles compact SFs in-flight, runs cuBLAS LT NVFP4 with operand
    amaxes pinned to 1.0, then applies the row_amax_A * row_amax_B post-scale.
    Output is bf16 (cuBLAS LT NVFP4 locks D to bf16/fp32); beta != 0 unsupported.

    ``a_sf_swizzled`` / ``b_sf_swizzled = True`` skips the in-binding swizzle
    for that operand (caller's SF is already in the cuBLAS LT swizzled layout
    e.g. from ``nvfp4_per_token_quantize(..., with_swizzle=True)``).
    """
    import transformer_engine_torch as tex  # type: ignore

    K_a = _validate_per_token_triple(a_data, a_scale, a_row_amax, "A")
    K_b = _validate_per_token_triple(b_data, b_scale, b_row_amax, "B")
    if K_a != K_b:
        raise ValueError(f"K mismatch between A and B: {K_a} vs {K_b}")
    K = K_a
    M = a_data.shape[0]
    N = b_data.shape[0]

    if K % 16 != 0:
        raise ValueError(f"K must be a multiple of 16 (got K={K})")
    # cuBLAS LT NVFP4 SF buffer is padded to (roundup(rows, 128), roundup(K/16, 4)).
    # Our compact quantize emits (rows, K/16); SF padding is a TODO so reject M/N < 128.
    if M < 128 or M % 128 != 0:
        raise ValueError(f"M must be a multiple of 128 (got M={M}); SF padding is a TODO.")
    if N < 128 or N % 128 != 0:
        raise ValueError(f"N must be a multiple of 128 (got N={N}); SF padding is a TODO.")
    if a_data.device != b_data.device:
        raise ValueError(
            f"A and B must be on the same device (got {a_data.device} vs {b_data.device})"
        )
    device = a_data.device

    if out is None:
        out_bf16 = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    else:
        if out.shape != (M, N):
            raise ValueError(f"out shape {tuple(out.shape)} != ({M}, {N})")
        if out.dtype != torch.bfloat16:
            raise ValueError(
                f"out dtype must be bf16 for in-place use, got {out.dtype}. "
                "(The binding produces bf16; pass `out=None` for non-bf16 dtypes "
                "and the result will be cast at the end.)"
            )
        out_bf16 = out

    if float(beta) != 0.0:
        raise ValueError(
            f"nvfp4_per_token_gemm: beta != 0 not yet supported, got beta={beta}. "
            "Use beta=0 and accumulate outside the call if needed."
        )

    a_data_u8 = a_data.contiguous().view(dtype=torch.uint8)
    b_data_u8 = b_data.contiguous().view(dtype=torch.uint8)

    # Binding expects uint8 SFs (accepts both e4m3 view and raw uint8 storage).
    a_scale_u8 = a_scale.contiguous().view(dtype=torch.uint8)
    b_scale_u8 = b_scale.contiguous().view(dtype=torch.uint8)
    a_scale_u8_flat = a_scale_u8.reshape(-1)
    b_scale_u8_flat = b_scale_u8.reshape(-1)

    a_row_amax_f32 = a_row_amax.to(torch.float32).contiguous()
    b_row_amax_f32 = b_row_amax.to(torch.float32).contiguous()

    # Lazy import to break the cpp_extensions.gemm circular import.
    from transformer_engine.pytorch.cpp_extensions.gemm import get_cublas_workspace

    workspace = get_cublas_workspace(device.index, ub=False, grouped_gemm=False)

    tex.nvfp4_per_token_gemm(
        a_data_u8,
        b_data_u8,
        a_scale_u8_flat,
        b_scale_u8_flat,
        a_row_amax_f32,
        b_row_amax_f32,
        out_bf16,
        workspace,
        M,
        N,
        K,
        float(alpha),
        float(beta),
        a_sf_swizzled=a_sf_swizzled,
        b_sf_swizzled=b_sf_swizzled,
    )

    return out_bf16 if out_dtype is torch.bfloat16 else out_bf16.to(out_dtype)

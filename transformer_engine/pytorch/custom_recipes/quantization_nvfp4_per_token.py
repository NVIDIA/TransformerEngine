# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import torch

from transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4 import cast_to_fp4x2

# Inner sub-block size along K is fixed by the NVFP4 spec (one E4M3
# ``s_dec`` per 16 FP4 samples); only the outer-amax granularity changes
# between per-token / per-tensor / blocked / 2D.
BLOCK_K: int = 16

# E2M1 / E4M3 numeric extrema (matches ``TypeExtrema`` in core_nvfp4.cuh).
FLOAT4_E2M1_MAX: float = 6.0
FLOAT8_E4M3_MAX: float = 448.0

# Matches the kernel's ``fmaxf(row_amax, 1e-12f)`` clamp on the divisor of
# ``compute_global_encode_scaling_factor_FP4``.
_AMAX_FLOOR: float = 1e-12


@dataclasses.dataclass
class RefNVFP4TensorPerToken:
    """Container for the per-token reference output.

    Attributes
    ----------
    data:
        Packed rowwise FP4 bytes, ``(M, N // 2)`` ``uint8``.
    scale:
        Per-1x16-block rowwise decode scale (E4M3), ``(M, N // 16)``
        ``float8_e4m3fn``.
    row_amax:
        Per-row outer amax, ``(M,)`` ``float32``. This replaces the
        per-tensor path's single-scalar ``amax`` and the blocked path's
        per-window ``window_amax``.
    columnwise_data, columnwise_scale, col_amax:
        Their columnwise (transposed) counterparts. Shapes are
        ``(N, M // 2)``, ``(N, M // 16)``, and ``(N,)`` respectively.
        ``None`` if columnwise was not requested.
    """

    data: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None
    row_amax: Optional[torch.Tensor] = None
    columnwise_data: Optional[torch.Tensor] = None
    columnwise_scale: Optional[torch.Tensor] = None
    col_amax: Optional[torch.Tensor] = None


class NVFP4QuantizerPerTokenRef:
    """Pure-PyTorch reference for the NVFP4 per-token cast kernel.

    Constructor takes the two output-direction switches (``rowwise`` and
    ``columnwise``). RHT, 2D scaling, and stochastic rounding are not
    exposed because the per-token CUDA kernel does not implement them
    (the per-token path is target-shape simple-and-fast: per-row outer
    + 1x16 inner SF, nothing else).

    The arithmetic chain (``S_enc``, ``s_dec``, ``block_scale``, FP4 cast)
    matches ``NVFP4Quantizer1x64Ref`` / ``NVFP4QuantizerBlockedRef``;
    only the outer-amax granularity differs:

      * 1x64Ref / BlockedRef : one outer amax per ``OUTER_K``-K-window
      * **PerTokenRef**      : one outer amax per row (full K window)
    """

    def __init__(
        self,
        rowwise: bool = True,
        columnwise: bool = False,
    ) -> None:
        if not rowwise and not columnwise:
            raise ValueError("At least one of rowwise / columnwise must be True.")
        self.rowwise = rowwise
        self.columnwise = columnwise

    def _quantize_2d(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the per-token reference math on a 2D input along its trailing dim.

        Returns ``(qx, sx, row_amax)`` where ``qx`` is ``(M, N // 2)``
        ``uint8``, ``sx`` is ``(M, N // BLOCK_K)`` ``float8_e4m3fn``,
        and ``row_amax`` is ``(M,)`` ``float32``.

        The columnwise pass is implemented by calling this routine on
        ``x.transpose(0, 1).contiguous()``.
        """
        if x.ndim != 2:
            raise ValueError(f"NVFP4QuantizerPerTokenRef expects a 2D tensor, got {x.ndim}D")
        M, N = x.shape
        if N % BLOCK_K != 0:
            raise ValueError(f"N={N} must be a multiple of BLOCK_K={BLOCK_K}")

        device = x.device
        fp32_max = torch.tensor(torch.finfo(torch.float32).max, device=device, dtype=torch.float32)
        fp4_max = torch.tensor(FLOAT4_E2M1_MAX, device=device, dtype=torch.float32)
        fp8_max = torch.tensor(FLOAT8_E4M3_MAX, device=device, dtype=torch.float32)

        n_blk = N // BLOCK_K
        x_fp32 = x.to(torch.float32).contiguous()
        x_blk = x_fp32.view(M, n_blk, BLOCK_K)

        # Outer = whole row. The kernel applies ``fmaxf(row_amax, 1e-12f)``
        # to the divisor; do the same here.
        row_amax = torch.amax(torch.abs(x_fp32), dim=-1)  # (M,) fp32 -- raw, pre-floor
        row_amax_safe = torch.clamp(row_amax, min=_AMAX_FLOOR).unsqueeze(-1)  # (M, 1)

        # Same ``compute_global_encode_scaling_factor_FP4`` form as the
        # per-tensor / blocked paths (just with ``row_amax`` instead of
        # ``window_amax`` / ``global_amax``).
        S_enc_row = (fp8_max * fp4_max) / row_amax_safe  # (M, 1)
        S_enc_row = torch.minimum(S_enc_row, fp32_max)
        S_enc_row = torch.where(
            (row_amax_safe == 0) | (S_enc_row == 0),
            torch.ones_like(S_enc_row),
            S_enc_row,
        )

        # Fold ``1 / fp4_max`` into the multiplier the same way the kernel
        # does in ``compute_decoding_scaling_factor`` (``S_enc * fp4_max_inv``).
        S_enc_row_mul_inv6 = S_enc_row * torch.reciprocal(fp4_max)  # (M, 1)

        # 1x16 block amax. Broadcast row's S_enc across n_blk blocks.
        vec_max = torch.amax(torch.abs(x_blk), dim=-1, keepdim=True)  # (M, n_blk, 1)
        S_enc_per_blk = S_enc_row.unsqueeze(-1)  # (M, 1, 1) -> broadcasts to (M, n_blk, 1)
        S_enc_per_blk_mul = S_enc_row_mul_inv6.unsqueeze(-1)

        # decode_scale = saturating_cast<fp8e4m3>(vec_max * S_enc / 6).
        # Kernel does NOT clamp before the cast; we clamp here because
        # PyTorch's ``.to(float8_e4m3fn)`` does not match CUDA's saturating
        # cast for values above FP8_MAX. After the explicit clamp the two
        # paths agree byte-for-byte.
        decode_scale_fp32 = vec_max * S_enc_per_blk_mul
        decode_scale_fp32 = torch.minimum(decode_scale_fp32, fp32_max)
        decode_scale_fp32 = torch.clamp(decode_scale_fp32, min=-fp8_max, max=fp8_max)
        decode_scale_e4m3 = decode_scale_fp32.to(torch.float8_e4m3fn)
        decode_scale_back_fp32 = decode_scale_e4m3.to(torch.float32)

        # block_scale = S_enc / s_dec, matching ``__fdiv_rn`` in the
        # kernel. All-zero blocks: s_dec saturates to 0, naive S_enc/0
        # would NaN; short-circuit to 0 to mirror the kernel.
        zero_blk = decode_scale_back_fp32 == 0
        denom = torch.where(
            zero_blk, torch.ones_like(decode_scale_back_fp32), decode_scale_back_fp32
        )
        encode_scale = S_enc_per_blk / denom
        encode_scale = torch.where(zero_blk, torch.zeros_like(encode_scale), encode_scale)
        encode_scale = torch.minimum(encode_scale, fp32_max)

        # Apply scale, clamp to FP4 range, pack two FP4 values per byte.
        scaled_x = x_blk * encode_scale
        clipped_x = torch.clamp(scaled_x, -fp4_max, fp4_max).reshape(M, N)
        qx = cast_to_fp4x2(clipped_x).contiguous()  # (M, N // 2)

        sx = decode_scale_e4m3.squeeze(-1).contiguous()  # (M, n_blk)
        row_amax_out = row_amax.to(torch.float32).contiguous()  # (M,) -- raw, no floor
        return qx, sx, row_amax_out

    def quantize(self, tensor: torch.Tensor) -> RefNVFP4TensorPerToken:
        """Quantize ``tensor`` and return a ``RefNVFP4TensorPerToken``."""
        out = RefNVFP4TensorPerToken()
        if self.rowwise:
            qx, sx, ra = self._quantize_2d(tensor)
            out.data = qx
            out.scale = sx
            out.row_amax = ra
        if self.columnwise:
            # The columnwise output is the rowwise quantization of the
            # transpose; both directions share the same math chain.
            qx_t, sx_t, ca = self._quantize_2d(tensor.transpose(0, 1).contiguous())
            out.columnwise_data = qx_t
            out.columnwise_scale = sx_t
            out.col_amax = ca
        return out


# ============================================================================
# Production wrapper (calls the CUDA kernel via the C-API binding).
# ============================================================================

# ----------------------------------------------------------------------------
# Shape / dtype gate shared by all three entries.
# ----------------------------------------------------------------------------
_PER_TOKEN_TILE: int = 128  # CHUNK_DIM_Y / CHUNK_DIM_X in the kernel


def _validate_per_token_input(x: torch.Tensor) -> Tuple[int, int]:
    """Enforce the per-token kernel's hard constraints. Returns ``(M, K)``."""
    if x.ndim != 2:
        raise ValueError(f"nvfp4_per_token expects a 2D tensor, got {x.ndim}D")
    if x.dtype != torch.bfloat16:
        raise ValueError(
            f"Per-token kernel is bf16-only; got dtype {x.dtype}. "
            "Non-bf16 inputs are not supported (no fallback path)."
        )
    M, K = x.shape
    if M % _PER_TOKEN_TILE != 0:
        raise ValueError(f"Per-token kernel requires M % {_PER_TOKEN_TILE} == 0; got M={M}")
    if K % _PER_TOKEN_TILE != 0:
        raise ValueError(f"Per-token kernel requires K % {_PER_TOKEN_TILE} == 0; got K={K}")
    return M, K


def nvfp4_per_token_quantize(
    x: torch.Tensor, *, rowwise: bool = True, columnwise: bool = False
) -> RefNVFP4TensorPerToken:
    """Production NVFP4 per-token cast through ``tex.nvfp4_per_token_quantize``.

    Backed by the TMA + mbarrier + 64x64 sub-tile pipeline
    (``common/cast/nvfp4/quantize_nvfp4_per_token.cu``). The C-API
    runs K1 (per-row + per-col amax) and K2 (FP4 + e4m3 SF encode) back-
    to-back on the same stream.

    Returns a ``RefNVFP4TensorPerToken`` populated with the kernel
    output (compact, non-swizzled scales). The Python-level container is
    the same as the reference for symmetry; only the source of the
    values differs.

    For cuBLAS LT consumption, the caller must swizzle the inner SF
    before forwarding to the GEMM; ``gemm_nvfp4_per_token`` handles
    this automatically.

    Raises ``ValueError`` on non-bf16 input or non-128-aligned shapes.
    """
    # Import lazily so the module does not require the binary at import time.
    # (Mirrors the pattern in ``gemm_nvfp4_blocked.py``.)
    import transformer_engine_torch as tex  # type: ignore

    if not (rowwise or columnwise):
        raise ValueError("At least one of rowwise / columnwise must be True.")
    M, K = _validate_per_token_input(x)

    device = x.device
    # Empty placeholders for the direction(s) we don't request -- the
    # binding still expects the argument slots (typed-empty is fine).
    empty = torch.empty(0, dtype=torch.uint8, device=device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=device)

    if rowwise:
        q_row = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
        s_dec_row = torch.empty((M, K // BLOCK_K), dtype=torch.uint8, device=device)
        row_amax = torch.empty((M,), dtype=torch.float32, device=device)
    else:
        q_row, s_dec_row, row_amax = empty, empty, empty_f32

    if columnwise:
        q_col = torch.empty((K, M // 2), dtype=torch.uint8, device=device)
        s_dec_col = torch.empty((K, M // BLOCK_K), dtype=torch.uint8, device=device)
        col_amax = torch.empty((K,), dtype=torch.float32, device=device)
    else:
        q_col, s_dec_col, col_amax = empty, empty, empty_f32

    tex.nvfp4_per_token_quantize(
        x, q_row, s_dec_row, row_amax, q_col, s_dec_col, col_amax, rowwise, columnwise
    )

    out = RefNVFP4TensorPerToken()
    if rowwise:
        out.data = q_row
        out.scale = s_dec_row.view(torch.float8_e4m3fn)
        out.row_amax = row_amax
    if columnwise:
        out.columnwise_data = q_col
        out.columnwise_scale = s_dec_col.view(torch.float8_e4m3fn)
        out.col_amax = col_amax
    return out


# ============================================================================
# Split entries (K1 = amax-only, K2 = encode-only).
#
# Diagnostic / benchmark interface, mirroring the production per-tensor
# kernel split (``HadamardAmaxTmaKernel`` for amax + the row_col_rht_gemm
# cast pass). Production callers should use ``nvfp4_per_token_quantize``
# above; the composite handles K1 + K2 ordering on the same stream.
# ============================================================================


def nvfp4_per_token_amax(
    x: torch.Tensor,
    *,
    rowwise: bool = True,
    columnwise: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Kernel 1 in isolation: per-row + per-col amax via TMA + atomicMax.
    Returns ``(row_amax, col_amax)``; either may be ``None`` if the
    corresponding direction is not requested.

    Lets the benchmark compare K1 wall-time against the production
    ``HadamardAmaxTmaKernel``. Production callers should use the
    composite ``nvfp4_per_token_quantize`` instead.

    Raises ``ValueError`` on non-bf16 input or non-128-aligned shapes.
    """
    import transformer_engine_torch as tex  # type: ignore

    if not (rowwise or columnwise):
        raise ValueError("At least one of rowwise / columnwise must be True.")
    M, K = _validate_per_token_input(x)

    device = x.device
    row_amax = (
        torch.empty((M,), dtype=torch.float32, device=device)
        if rowwise
        else torch.empty(0, dtype=torch.float32, device=device)
    )
    col_amax = (
        torch.empty((K,), dtype=torch.float32, device=device)
        if columnwise
        else torch.empty(0, dtype=torch.float32, device=device)
    )

    tex.nvfp4_per_token_amax(x, row_amax, col_amax, rowwise, columnwise)

    return (row_amax if rowwise else None, col_amax if columnwise else None)


def nvfp4_per_token_encode(
    x: torch.Tensor,
    *,
    row_amax: Optional[torch.Tensor] = None,
    col_amax: Optional[torch.Tensor] = None,
    rowwise: bool = True,
    columnwise: bool = True,
) -> RefNVFP4TensorPerToken:
    """Kernel 2 in isolation: FP4 + e4m3 SF encode given pre-filled
    amax buffer(s).

    ``row_amax`` of shape ``(M,)`` is required when ``rowwise=True``; same
    for ``col_amax`` of shape ``(K,)`` when ``columnwise=True``. The
    buffers are typically produced by a prior
    ``nvfp4_per_token_amax`` call.

    Lets the benchmark compare K2 wall-time against the production
    per-tensor cast pass. Production callers should use the composite
    ``nvfp4_per_token_quantize`` instead.

    Raises ``ValueError`` on non-bf16 input, non-128-aligned shapes, or
    missing / mis-shaped amax buffers.
    """
    import transformer_engine_torch as tex  # type: ignore

    if not (rowwise or columnwise):
        raise ValueError("At least one of rowwise / columnwise must be True.")
    M, K = _validate_per_token_input(x)
    if rowwise and (row_amax is None or row_amax.shape != (M,)):
        raise ValueError(f"row_amax must be (M={M},) fp32 when rowwise=True")
    if columnwise and (col_amax is None or col_amax.shape != (K,)):
        raise ValueError(f"col_amax must be (K={K},) fp32 when columnwise=True")

    device = x.device
    empty = torch.empty(0, dtype=torch.uint8, device=device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=device)

    if rowwise:
        q_row = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
        s_dec_row = torch.empty((M, K // BLOCK_K), dtype=torch.uint8, device=device)
        row_amax_t = row_amax  # type: ignore[assignment]
    else:
        q_row, s_dec_row, row_amax_t = empty, empty, empty_f32
    if columnwise:
        q_col = torch.empty((K, M // 2), dtype=torch.uint8, device=device)
        s_dec_col = torch.empty((K, M // BLOCK_K), dtype=torch.uint8, device=device)
        col_amax_t = col_amax  # type: ignore[assignment]
    else:
        q_col, s_dec_col, col_amax_t = empty, empty, empty_f32

    tex.nvfp4_per_token_encode(
        x,
        q_row,
        s_dec_row,
        row_amax_t,
        q_col,
        s_dec_col,
        col_amax_t,
        rowwise,
        columnwise,
    )

    out = RefNVFP4TensorPerToken()
    if rowwise:
        out.data = q_row
        out.scale = s_dec_row.view(torch.float8_e4m3fn)
        out.row_amax = row_amax_t
    if columnwise:
        out.columnwise_data = q_col
        out.columnwise_scale = s_dec_col.view(torch.float8_e4m3fn)
        out.col_amax = col_amax_t
    return out

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Reference implementation for NVFP4 hierarchical 1x64 cast (rowwise + columnwise).

The hierarchical 1x64 + 1x16 scheme replaces the per-tensor encoding scaling
factor used by stock NVFP4 with a per-1x64-K-window scaling factor; the four
1x16 sub-blocks inside a window share their parent ``S_enc``. The CUDA kernel
that implements this lives in
``transformer_engine/common/cast/nvfp4/quantize_nvfp4_1x64.cu`` and is
dispatched when ``NVTE_NVFP4_ROWWISE_1X64_LOCAL_ENCODE=1``.

This file mirrors that kernel's arithmetic in pure PyTorch so tests can
compare the kernel's output byte-for-byte against a Python oracle. The
arithmetic ordering and intermediate clamps are chosen to match what the
kernel does:

* ``S_enc_tile = (FP8_MAX*FP4_MAX) / max(tile_amax, 1e-12)`` clamped to
  ``fp32_max``;
* ``s_dec = saturating_cast<fp8e4m3>(vec_max * S_enc_tile / FP4_MAX)`` (the
  ``1/FP4_MAX`` is folded into the ``S_enc`` multiplier to match
  ``compute_decoding_scaling_factor``);
* ``block_scale = S_enc_tile / fp32(s_dec)`` with an explicit ``s_dec == 0``
  short-circuit to 0 (matches the kernel's ``s_dec_f == 0.f`` branch);
* ``q = round_fp4_satfinite(x_fp32 * block_scale)`` with values clamped to
  ``[-FP4_MAX, FP4_MAX]`` before packing.

For the columnwise (transposed) output the kernel runs the same math along
the original M direction with a 64x1 window; the reference implements this
by simply running the rowwise routine on ``x.T``. The window amax tensor
is exposed alongside ``data`` / ``scale`` so consumers can reconstruct
``S_enc_window`` (the per-block ``s_dec`` alone is not enough information
to dequantize correctly).

Only the non-RHT, non-2D, non-stochastic-rounding path is supported.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import torch

from transformer_engine.pytorch.custom_recipes.quantization_nvfp4 import cast_to_fp4x2

# Window/block geometry is fixed by the kernel design.
WINDOW_K: int = 64
BLOCK_K: int = 16
BLOCKS_PER_WINDOW: int = WINDOW_K // BLOCK_K  # 4

# E2M1 / E4M3 numeric extrema (matches ``TypeExtrema`` in core_nvfp4.cuh).
FLOAT4_E2M1_MAX: float = 6.0
FLOAT8_E4M3_MAX: float = 448.0

# Matches the kernel's ``fmaxf(tile_amax, 1e-12f)`` clamp guarding the divisor
# of ``compute_global_encode_scaling_factor_FP4``.
_TILE_AMAX_FLOOR: float = 1e-12


@dataclasses.dataclass
class RefNVFP4Tensor1x64:
    """Container for the hierarchical 1x64 reference output.

    Mirrors the subset of attributes that the bit-exact test inspects.

    Attributes
    ----------
    data:
        Packed rowwise FP4 bytes, ``(M, N // 2)`` ``uint8``.
    scale:
        Per-1x16-block rowwise decode scale (E4M3), ``(M, N // 16)``
        ``float8_e4m3fn``.
    window_amax_row:
        Per-1x64-window rowwise amax, ``(M, N // 64)`` ``float32``.
        ``S_enc_window`` is recoverable from this via
        ``compute_global_encode_scaling_factor_FP4(window_amax)``; consumers
        need it for correct dequantization (the per-block ``s_dec`` alone
        does not contain enough information).
    columnwise_data, columnwise_scale, window_amax_col:
        Their columnwise (transposed) counterparts. The transposed FP4 data
        has shape ``(N, M // 2)`` (matching the production cast+transpose
        layout), the columnwise scales ``(N, M // 16)``, and the columnwise
        window amax ``(N, M // 64)``. ``None`` if columnwise was not
        requested.
    """

    data: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None
    window_amax_row: Optional[torch.Tensor] = None
    columnwise_data: Optional[torch.Tensor] = None
    columnwise_scale: Optional[torch.Tensor] = None
    window_amax_col: Optional[torch.Tensor] = None


class NVFP4Quantizer1x64Ref:
    """Reference implementation of the hierarchical 1x64 cast kernel.

    Constructor takes flags matching the kernel's two output-direction
    switches; RHT, 2D scaling, and stochastic rounding are not exposed
    because the kernel rejects them at dispatch time and we do not want
    the reference to drift away from the kernel's actual capabilities.
    """

    def __init__(self, rowwise: bool = True, columnwise: bool = False) -> None:
        if not rowwise and not columnwise:
            raise ValueError("At least one of rowwise / columnwise must be True.")
        self.rowwise = rowwise
        self.columnwise = columnwise

    @staticmethod
    def _quantize_2d(
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the 1x64 reference math on a 2D input along its trailing dim.

        Returns ``(qx, sx, window_amax)`` where ``qx`` is ``(M, N // 2)``
        ``uint8``, ``sx`` is ``(M, N // BLOCK_K)`` ``float8_e4m3fn``, and
        ``window_amax`` is ``(M, N // WINDOW_K)`` ``float32``.

        The columnwise pass is implemented by calling this routine on
        ``x.T.contiguous()``; both passes therefore share a single
        arithmetic path, which is the cleanest way to keep the two
        directions consistent with the kernel (the kernel itself shares
        its math by re-using ``compute_decoding_scaling_factor`` /
        ``compute_global_encode_scaling_factor_FP4`` between passes).
        """
        if x.ndim != 2:
            raise ValueError(f"NVFP4Quantizer1x64Ref expects a 2D tensor, got {x.ndim}D")
        M, N = x.shape
        if N % WINDOW_K != 0:
            raise ValueError(
                f"N={N} must be a multiple of WINDOW_K={WINDOW_K} (kernel hard requirement)"
            )

        device = x.device
        fp32_max = torch.tensor(
            torch.finfo(torch.float32).max, device=device, dtype=torch.float32
        )
        fp4_max = torch.tensor(FLOAT4_E2M1_MAX, device=device, dtype=torch.float32)
        fp8_max = torch.tensor(FLOAT8_E4M3_MAX, device=device, dtype=torch.float32)

        Np = N
        n_win = Np // WINDOW_K
        n_blk = Np // BLOCK_K

        x_fp32 = x.to(torch.float32).contiguous()
        x_win = x_fp32.view(M, n_win, WINDOW_K)
        x_blk = x_fp32.view(M, n_blk, BLOCK_K)

        # 1x64 tile amax. The kernel applies ``fmaxf(tile_amax, 1e-12f)`` to
        # the divisor; do the same here. ``S_enc_tile`` is then computed
        # exactly like ``compute_global_encode_scaling_factor_FP4``: divide,
        # clamp to fp32_max, and fall back to 1.0 if the divisor or quotient
        # is zero (the latter branch is dead given the floor but is mirrored
        # for parity with the C++ helper).
        tile_amax = torch.amax(torch.abs(x_win), dim=-1, keepdim=True)  # (M, n_win, 1)
        tile_amax_safe = torch.clamp(tile_amax, min=_TILE_AMAX_FLOOR)

        S_enc_tile = (fp8_max * fp4_max) / tile_amax_safe
        S_enc_tile = torch.minimum(S_enc_tile, fp32_max)
        S_enc_tile = torch.where(
            (tile_amax_safe == 0) | (S_enc_tile == 0),
            torch.ones_like(S_enc_tile),
            S_enc_tile,
        )

        # Fold ``1 / fp4_max`` into the multiplier the same way the kernel
        # does in ``compute_decoding_scaling_factor`` (``S_enc * fp4_max_inv``).
        # Keeping the operation order identical is what makes the resulting
        # E4M3 scale bit-exact with the kernel.
        S_enc_tile_mul_inv6 = S_enc_tile * torch.reciprocal(fp4_max)

        # 1x16 block amax and per-block S_enc broadcast. Each 1x64 window
        # spans BLOCKS_PER_WINDOW (=4) consecutive 1x16 sub-blocks, so
        # ``repeat_interleave`` along the block axis aligns one S_enc_tile
        # to every block inside that window.
        vec_max = torch.amax(torch.abs(x_blk), dim=-1, keepdim=True)  # (M, n_blk, 1)
        S_enc_per_blk = S_enc_tile.repeat_interleave(BLOCKS_PER_WINDOW, dim=1)
        S_enc_per_blk_mul = S_enc_tile_mul_inv6.repeat_interleave(BLOCKS_PER_WINDOW, dim=1)

        # decode_scale = saturating_cast<fp8e4m3>(vec_max * S_enc_tile / 6).
        # The kernel does not clamp before the cast; we do, because PyTorch's
        # ``.to(float8_e4m3fn)`` does not match CUDA's saturating cast for
        # values above FP8_MAX. After the explicit clamp the two paths agree.
        decode_scale_fp32 = vec_max * S_enc_per_blk_mul
        decode_scale_fp32 = torch.minimum(decode_scale_fp32, fp32_max)
        decode_scale_fp32 = torch.clamp(decode_scale_fp32, min=-fp8_max, max=fp8_max)
        decode_scale_e4m3 = decode_scale_fp32.to(torch.float8_e4m3fn)
        decode_scale_back_fp32 = decode_scale_e4m3.to(torch.float32)

        # block_scale = S_enc_tile / s_dec, matching ``__fdiv_rn`` in the
        # kernel. All-zero blocks have ``s_dec == 0``, which would yield
        # ``+inf`` here and propagate NaN through the downstream multiply
        # (``cvt.rn.satfinite.e2m1x4.f32(NaN)`` saturates to +FP4_MAX on
        # SM10 -- we do NOT want that). Short-circuit to 0 to mirror the
        # kernel's ``s_dec_f == 0.f`` branch.
        zero_blk = decode_scale_back_fp32 == 0
        denom = torch.where(zero_blk, torch.ones_like(decode_scale_back_fp32), decode_scale_back_fp32)
        encode_scale = S_enc_per_blk / denom
        encode_scale = torch.where(zero_blk, torch.zeros_like(encode_scale), encode_scale)
        encode_scale = torch.minimum(encode_scale, fp32_max)

        # Apply scale, clamp to FP4 range, and pack two FP4 values per byte.
        scaled_x = x_blk * encode_scale
        clipped_x = torch.clamp(scaled_x, -fp4_max, fp4_max).reshape(M, Np)
        qx = cast_to_fp4x2(clipped_x).contiguous()  # (M, N // 2)

        sx = decode_scale_e4m3.squeeze(-1).contiguous()  # (M, n_blk)

        # Per-1x64-window amax, exposed for consumer-side dequantization.
        window_amax = tile_amax.squeeze(-1).to(torch.float32).contiguous()  # (M, n_win)

        return qx, sx, window_amax

    def quantize(self, tensor: torch.Tensor) -> RefNVFP4Tensor1x64:
        """Quantize ``tensor`` and return a ``RefNVFP4Tensor1x64``."""
        out = RefNVFP4Tensor1x64()
        if self.rowwise:
            qx, sx, win_amax = self._quantize_2d(tensor)
            out.data = qx
            out.scale = sx
            out.window_amax_row = win_amax
        if self.columnwise:
            # The columnwise output is the rowwise quantization of the
            # transpose; both directions share the same math and the same
            # ``s_dec``/``block_scale`` chain.
            qx_t, sx_t, win_amax_t = self._quantize_2d(tensor.transpose(0, 1).contiguous())
            out.columnwise_data = qx_t
            out.columnwise_scale = sx_t
            out.window_amax_col = win_amax_t
        return out

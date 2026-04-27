# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Reference implementation for NVFP4 rowwise 1x64 local-encode quantization.

The hierarchical 1x64 + 1x16 scheme replaces the per-tensor encoding scaling
factor used by stock NVFP4 with a per-1x64-K-window scaling factor; the four
1x16 sub-blocks inside a window share their parent ``S_enc``. The CUDA kernel
that implements this lives in
``transformer_engine/common/cast/nvfp4/quantize_nvfp4_1x64_rowwise.cu`` and is
dispatched when ``NVTE_NVFP4_ROWWISE_1X64_LOCAL_ENCODE=1``.

This file mirrors that kernel's arithmetic in pure PyTorch so tests can
compare the kernel's output byte-for-byte against a Python oracle (the same
bit-exact methodology used by ``NVFP4QuantizerRef`` for the production NVFP4
path). The arithmetic ordering and intermediate clamps are chosen to match
what the kernel does:

* ``S_enc_tile = (FP8_MAX*FP4_MAX) / max(tile_amax, 1e-12)`` clamped to
  ``fp32_max``;
* ``s_dec = saturating_cast<fp8e4m3>(vec_max * S_enc_tile / FP4_MAX)`` (the
  ``1/FP4_MAX`` is folded into the ``S_enc`` multiplier to match
  ``compute_decoding_scaling_factor``);
* ``block_scale = S_enc_tile / fp32(s_dec)`` (matches ``__fdiv_rn`` in the
  kernel);
* ``q = round_fp4_satfinite(x_fp32 * block_scale)`` with values clamped to
  ``[-FP4_MAX, FP4_MAX]`` before packing.

Only the rowwise, non-RHT, non-2D, non-stochastic-rounding path is supported,
matching the kernel's preconditions.
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
    """Container for the rowwise 1x64 reference output.

    Mirrors the subset of attributes that the bit-exact test inspects.
    Naming follows ``quantization_nvfp4.RefNVFP4Tensor`` so the test reads the
    same way as ``check_quantization_nvfp4_versus_reference``.

    Attributes
    ----------
    data:
        Packed FP4 bytes, ``(M, N // 2)`` ``uint8``.
    scale:
        Per-1x16-block decode scale (E4M3), ``(M, N // 16)`` ``float8_e4m3fn``.
    global_amax_row:
        Global tensor amax (1-D, single fp32 element). Equals the result of
        the kernel's ``atomicMaxFloat`` over all per-tile amax values.
    """

    data: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None
    global_amax_row: Optional[torch.Tensor] = None


class NVFP4Quantizer1x64Ref:
    """Reference implementation of the rowwise 1x64 local-encode kernel.

    The constructor takes no parameters because the kernel itself does not
    expose any -- columnwise output, RHT, 2D scaling, and stochastic rounding
    are all rejected at dispatch time. Surfacing those as ctor flags here
    would only invite the test to drift away from the kernel's actual
    capabilities.
    """

    def __init__(self) -> None:
        # No configurable knobs; see class docstring.
        pass

    @staticmethod
    def _quantize_rowwise(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the 1x64 reference math on a 2D input.

        Returns
        -------
        ``(qx, sx, global_amax)`` where shapes match the kernel's compact
        rowwise layout: ``qx`` is ``(M, N // 2)`` ``uint8``, ``sx`` is
        ``(M, N // BLOCK_K)`` ``float8_e4m3fn``, and ``global_amax`` is
        ``(1,)`` ``float32``.
        """
        if x.ndim != 2:
            raise ValueError(f"NVFP4Quantizer1x64Ref expects a 2D tensor, got {x.ndim}D")
        M, N = x.shape
        if N % BLOCK_K != 0:
            raise ValueError(
                f"N={N} must be a multiple of BLOCK_K={BLOCK_K} (kernel hard requirement)"
            )

        device = x.device
        fp32_max = torch.tensor(
            torch.finfo(torch.float32).max, device=device, dtype=torch.float32
        )
        fp4_max = torch.tensor(FLOAT4_E2M1_MAX, device=device, dtype=torch.float32)
        fp8_max = torch.tensor(FLOAT8_E4M3_MAX, device=device, dtype=torch.float32)

        # Pad K up to a multiple of WINDOW_K so the reshape into windows is
        # well-defined. The kernel itself supports a partial last window via
        # the ``win_len`` clamp; we emulate that by zero-padding here and
        # trimming the padded columns out of qx/sx at the end (the padded
        # blocks are uninitialised in the kernel output, so the test compares
        # only the un-padded prefix).
        pad_n = (WINDOW_K - N % WINDOW_K) % WINDOW_K
        if pad_n > 0:
            x_padded = torch.nn.functional.pad(x, (0, pad_n), mode="constant", value=0.0)
        else:
            x_padded = x.contiguous()
        Np = x_padded.shape[1]
        n_win = Np // WINDOW_K
        n_blk = Np // BLOCK_K

        x_padded_fp32 = x_padded.to(torch.float32)
        x_win = x_padded_fp32.view(M, n_win, WINDOW_K)
        x_blk = x_padded_fp32.view(M, n_blk, BLOCK_K)

        # 1x64 tile amax. The kernel applies fmaxf(tile_amax, 1e-12f) to the
        # divisor; do the same here. ``S_enc_tile`` is then computed exactly
        # like ``compute_global_encode_scaling_factor_FP4``: divide, clamp to
        # fp32_max, and fall back to 1.0 if the divisor or quotient is zero
        # (the latter branch is dead given the floor but is mirrored for
        # parity).
        tile_amax = torch.amax(torch.abs(x_win), dim=-1, keepdim=True)  # (M, n_win, 1)
        tile_amax_safe = torch.clamp(tile_amax, min=_TILE_AMAX_FLOOR)

        S_enc_tile = (fp8_max * fp4_max) / tile_amax_safe
        S_enc_tile = torch.minimum(S_enc_tile, fp32_max)
        S_enc_tile = torch.where(
            (tile_amax_safe == 0) | (S_enc_tile == 0),
            torch.ones_like(S_enc_tile),
            S_enc_tile,
        )

        # Fold (1 / fp4_max) into the multiplier the same way the kernel does
        # in ``compute_decoding_scaling_factor`` (``S_enc * fp4_max_inv``).
        # Keeping the operation order identical is what makes the resulting
        # E4M3 scale bit-exact with the kernel.
        S_enc_tile_mul_inv6 = S_enc_tile * torch.reciprocal(fp4_max)

        # 1x16 block amax and per-block S_enc broadcast. Each 1x64 window
        # spans exactly BLOCKS_PER_WINDOW (=4) consecutive 1x16 sub-blocks, so
        # repeat_interleave along the block axis aligns one S_enc_tile to
        # every block inside that window.
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
        # kernel. Padded sub-blocks have vec_max == 0 hence s_dec == 0, which
        # would yield +inf here and propagate NaN through the downstream
        # multiply. To keep the division warning-free we replace zero
        # divisors with 1.0, divide, then mask the result back to zero -- the
        # padded slots are trimmed out of the final comparison either way.
        zero_blk = decode_scale_back_fp32 == 0
        denom = torch.where(zero_blk, torch.ones_like(decode_scale_back_fp32), decode_scale_back_fp32)
        encode_scale = S_enc_per_blk / denom
        encode_scale = torch.where(zero_blk, torch.zeros_like(encode_scale), encode_scale)
        encode_scale = torch.minimum(encode_scale, fp32_max)

        # Apply scale, clamp to FP4 range, and pack two FP4 values per byte.
        scaled_x = x_blk * encode_scale
        clipped_x = torch.clamp(scaled_x, -fp4_max, fp4_max).reshape(M, Np)
        qx_packed_padded = cast_to_fp4x2(clipped_x)  # (M, Np // 2)

        sx_padded = decode_scale_e4m3.squeeze(-1)  # (M, n_blk)

        # Trim the K-direction padding so the returned tensors describe only
        # positions the kernel actually wrote to.
        qx = qx_packed_padded[:, : N // 2].contiguous()
        sx = sx_padded[:, : N // BLOCK_K].contiguous()

        # ``output.amax`` in the kernel accumulates ``atomicMaxFloat`` over
        # every per-tile amax, which is mathematically max-of-maxes -- i.e.
        # the global tensor amax. Compute that directly here.
        global_amax = torch.amax(torch.abs(x.to(torch.float32))).reshape(1)

        return qx, sx, global_amax

    def quantize(self, tensor: torch.Tensor) -> RefNVFP4Tensor1x64:
        """Quantize ``tensor`` and return a ``RefNVFP4Tensor1x64``."""
        qx, sx, global_amax = self._quantize_rowwise(tensor)
        return RefNVFP4Tensor1x64(data=qx, scale=sx, global_amax_row=global_amax)

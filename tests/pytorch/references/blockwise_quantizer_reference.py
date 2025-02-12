# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import dataclasses
import math
import torch
from typing import Optional, Protocol, Tuple


@dataclasses.dataclass()
class QuantizeResult:
    data: torch.Tensor
    scale: torch.Tensor
    data_t: Optional[torch.Tensor]
    scale_t: Optional[torch.Tensor]


# FIXME(kwyss): Put this in a common location for per-tensor current
# scaling reference
def _scale_from_amax_tensor(
    x_dtype: torch.dtype,
    amax: torch.Tensor,
    quant_dtype: torch.dtype,
    *,
    eps: float,
    pow_2_scales: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derives quantization and dequantization from amax and options.

    Reference implementation for scale calculation.

    Returns:
    - scale: quantization scales
    - scale_inv: dequantization scales
    - amax: Amax tensor with updates made for extrema values.
    """
    assert amax.dtype == torch.float, "amax must be a float tensor."
    fp8_max = torch.finfo(quant_dtype).max
    # Clamping amax to avoid division by small numbers
    amax = torch.max(amax, torch.tensor(eps))

    # Compute scale factor
    scale = torch.div(fp8_max, amax)
    # Note frexp doesn't give back inf for exponent with an inf input
    # We take care of inf before pow_2_scales
    scale = torch.where(scale == torch.inf, torch.finfo(x_dtype).max, scale)
    if pow_2_scales:
        # Calculate rounded down exponent
        _, exp = torch.frexp(scale)
        # Positive numbers are always returned as mant, exp with
        # a mantissa in [0.5, 1.0). Because a normal float has a mantissa with
        # hidden bit in [1.0, 2.0), the exponent will be off by exactly one because
        # of the shift. Subnormal and zero cases need not be considered because
        # the smallest possible result of fp8_max / amax is still normal.
        exp = exp - 1
        # No subnormals and zero.
        assert (exp > -127).all()
        unity = torch.tensor([1.0], device=exp.device)
        torch.ldexp(unity, exp, out=scale)
        # Case where amax is inf. The frexp, ldexp logic changes 0.0 scales
        # Return 0.0 for 0.0 scale for consistency with non-pow2 scale
        # calculation.
        scale = torch.where(amax == float("inf"), 0.0, scale)

    # Handle overflow cases for amax zero causing NaN
    scale = torch.where(amax == 0, 1.0, scale)

    # Compute scale_inv
    scale_inv = torch.reciprocal(scale)

    return scale, scale_inv, amax


@dataclasses.dataclass()
class CuBLASScaleMunger:

    def munge_scale_shapes_for_backend(
        self,
        unmunged: QuantizeResult,
        tile_shape: Tuple[int, int],
    ) -> QuantizeResult:
        """
        cuBLAS GEMMs requires 1x128 quantized tensors to be have scales transposed
        so that for an (M, N) tensor, the scales are (RounUpDiv(N, 128), M)

        For 128x128 quantized tensors, the GEMM expects (M, PadToAlign(RoundUpDivide(N, 128), 4))
        format. If RoundUpDivide(N, 128) is not divisible by 4, a transformation is required
        """
        if tile_shape[0] != 1:
            # 2D block quantized tensor needs padding for cuBLAS GEMM.
            def _munge_scale_tensor(s: torch.Tensor) -> torch.Tensor:
                M, K = s.shape
                if K % 4 == 0:
                    return s
                k_pad = 4 - (K % 4)
                return torch.nn.functional.pad(s, (0, k_pad), mode="constant", value=0).contiguous()

            s = _munge_scale_tensor(unmunged.scale)
            if unmunged.scale_t is None:
                s_t = None
            else:
                s_t = _munge_scale_tensor(unmunged.scale_t)
            return QuantizeResult(unmunged.data, s, unmunged.data_t, s_t)

        # 1D block quantized tensors needs transpose to prepare for the GEMM.
        s = unmunged.scale.transpose(-1, -2).contiguous()
        if unmunged.scale_t is None:
            s_t = None
        else:
            s_t = unmunged.scale_t.transpose(-1, -2).contiguous()
        return QuantizeResult(unmunged.data, s, unmunged.data_t, s_t)

    def demunge_scale_shape_from_backend(
        cls,
        qtensor_shape: Tuple[int, int],
        scales: torch.Tensor,
        tile_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Inverse operation of munge_scale_shapes_for_backend
        """
        if tile_shape[0] != 1:
            # 2D block quantized tensor may need padding stripped off
            derived_scale_k_shape = math.ceil(qtensor_shape[1] / tile_shape[1])
            M, K = scales.shape
            if derived_scale_k_shape == K:
                return scales
            else:
                return scales[:, :derived_scale_k_shape].contiguous()
        return scales.transpose(-1, -2).contiguous()


@dataclasses.dataclass()
class BlockwiseQuantizerReference:
    """
    A reference QuantizeOp for subchannel/block hybrid quantization.

    Defers to ref GEMMs and quantizization formatting based on the backend.
    """

    def __init__(self) -> None:
        self.scale_munger = CuBLASScaleMunger()

    @classmethod
    def _quantize_square_block_tiling(
        cls,
        x: torch.Tensor,
        quant_dtype: torch.dtype,
        tile_len: int,
        *,
        return_transpose: bool,
        pow_2_scales: bool,
        eps: float,
    ) -> QuantizeResult:
        M, K = x.shape

        pad_m_k = [0, 0]
        if K % tile_len != 0:
            pad_m_k[1] = tile_len - (K % tile_len)
        if M % tile_len != 0:
            pad_m_k[0] = tile_len - (M % tile_len)

        unpadded_m, unpadded_k = M, K
        if pad_m_k[0] != 0 or pad_m_k[1] != 0:
            x = torch.nn.functional.pad(
                x, (0, pad_m_k[1], 0, pad_m_k[0]), mode="constant", value=0
            ).contiguous()
            M, K = x.shape

        x_tiled = x.reshape(M // tile_len, tile_len, K // tile_len, tile_len)
        amax_grid = (
            torch.abs(x_tiled.transpose(-3, -2))
            .reshape(M // tile_len, K // tile_len, tile_len**2)
            .amax(dim=-1)
        ).float()
        dtype_max = torch.finfo(quant_dtype).max

        scale, scale_inv, _ = _scale_from_amax_tensor(
            x_dtype=x.dtype,
            amax=amax_grid,
            quant_dtype=quant_dtype,
            pow_2_scales=pow_2_scales,
            eps=eps,
        )
        qx = x_tiled * scale.reshape(M // tile_len, 1, K // tile_len, 1)
        qx = torch.clamp(qx, min=-dtype_max, max=dtype_max)
        qx = qx.to(dtype=quant_dtype)
        qx = qx.reshape(M, K)
        if unpadded_k != K or unpadded_m != M:
            qx = qx[:unpadded_m, :unpadded_k].contiguous()
        if return_transpose:
            # Valid because of square block sizes
            qx_t = qx.transpose(-1, -2).contiguous()
            scale_inv_t = scale_inv.transpose(-1, -2).contiguous()
        else:
            qx_t = None
            scale_inv_t = None

        return QuantizeResult(data=qx, scale=scale_inv, data_t=qx_t, scale_t=scale_inv_t)

    @classmethod
    def _quantize_vectorwise_reference(
        cls,
        x: torch.Tensor,
        quant_dtype: torch.dtype,
        tile_len: int,
        *,
        pow_2_scales: bool,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        M, K = x.shape
        dtype_max = torch.finfo(quant_dtype).max
        x_tiled = x.reshape(M, K // tile_len, tile_len)
        amax_grid = torch.abs(x_tiled).amax(dim=-1).float()
        scale, scale_inv, _ = _scale_from_amax_tensor(
            x_dtype=x.dtype,
            amax=amax_grid,
            quant_dtype=quant_dtype,
            pow_2_scales=pow_2_scales,
            eps=eps,
        )
        qx = x_tiled * scale.reshape(M, K // tile_len, 1)
        qx = torch.clamp(qx, min=-dtype_max, max=dtype_max)
        qx = qx.to(dtype=quant_dtype)
        qx = qx.reshape(M, K)
        return qx, scale_inv

    @classmethod
    def _quantize_vector_tiling(
        cls,
        x: torch.Tensor,
        quant_dtype: torch.dtype,
        tile_len: int,
        *,
        return_transpose: bool,
        pow_2_scales: bool,
        eps: float,
    ) -> QuantizeResult:
        M, K = x.shape

        if K % tile_len == 0:
            qref_input = x
        else:
            pad_amount = tile_len - (K % tile_len)
            pad = (0, pad_amount)
            qref_input = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        qout_padded, scale_inv = cls._quantize_vectorwise_reference(
            qref_input,
            quant_dtype,
            tile_len=tile_len,
            pow_2_scales=pow_2_scales,
            eps=eps,
        )
        if K % tile_len == 0:
            qout = qout_padded
        else:
            qout = qout_padded[:, :K].contiguous()

        if return_transpose:
            if M % tile_len == 0:
                qref_input = x.transpose(-1, -2).contiguous()
            else:
                amount_to_pad = tile_len - (M % tile_len)
                pad = (0, amount_to_pad)
                qref_input = torch.nn.functional.pad(
                    x.transpose(-1, -2), pad, mode="constant", value=0
                ).contiguous()
            qout_t_padded, scale_inv_t = cls._quantize_vectorwise_reference(
                qref_input,
                quant_dtype,
                tile_len=tile_len,
                pow_2_scales=pow_2_scales,
                eps=eps,
            )
            if M % tile_len == 0:
                qout_t = qout_t_padded
            else:
                qout_t = qout_t_padded[:, :M].contiguous()
        else:
            qout_t, scale_inv_t = None, None

        return QuantizeResult(data=qout, scale=scale_inv, data_t=qout_t, scale_t=scale_inv_t)

    def ref_dequantize_rowwise(
        self,
        q: torch.Tensor,
        quant_tile_shape: Tuple[int, int],
        s: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        assert q.dim() == 2
        q_M, q_K = q.shape
        s = self.scale_munger.demunge_scale_shape_from_backend((q_M, q_K), s, quant_tile_shape)
        assert len(s.shape) == 2
        m_tiles, k_tiles = s.shape
        M, K = q.shape
        unpadded_m, unpadded_k = M, K
        if M % quant_tile_shape[0] != 0 or K % quant_tile_shape[1] != 0:
            m_pad_amount = (quant_tile_shape[0] - (M % quant_tile_shape[0])) % quant_tile_shape[0]
            k_pad_amount = (quant_tile_shape[1] - (K % quant_tile_shape[1])) % quant_tile_shape[1]
            q = torch.nn.functional.pad(
                q, (0, k_pad_amount, 0, m_pad_amount), mode="constant", value=0
            ).contiguous()
            M, K = q.shape
        q_tiled = q.reshape(m_tiles, quant_tile_shape[0], k_tiles, quant_tile_shape[1])
        result = q_tiled.to(dtype) * s.reshape(m_tiles, 1, k_tiles, 1)
        result = result.view(M, K).to(dtype)
        if M != unpadded_m or K != unpadded_k:
            result = result[:unpadded_m, :unpadded_k].contiguous()
        return result

    def quantize(
        self,
        x: torch.Tensor,
        quant_dtype: torch.dtype,
        return_transpose: bool = False,
        eps: float = 0.0,
        pow_2_scales: bool = False,
        quant_tile_shape: Tuple[int, int] = (128, 128),
    ) -> QuantizeResult:
        # sanity checks
        assert x.dim() == 2
        assert x.dtype in (
            torch.float,
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ), "Unsupported input dtype."
        assert quant_dtype in (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ), "Unsupported quant dtype."

        assert quant_tile_shape in ((1, 128), (128, 128))
        if quant_tile_shape[0] == 1:
            # Quantize row-wise
            return self.scale_munger.munge_scale_shapes_for_backend(
                self._quantize_vector_tiling(
                    x,
                    quant_dtype,
                    tile_len=quant_tile_shape[1],
                    return_transpose=return_transpose,
                    pow_2_scales=pow_2_scales,
                    eps=eps,
                ),
                quant_tile_shape,
            )
        else:
            # Quantize block-wise
            return self.scale_munger.munge_scale_shapes_for_backend(
                self._quantize_square_block_tiling(
                    x,
                    quant_dtype,
                    tile_len=quant_tile_shape[0],
                    return_transpose=return_transpose,
                    pow_2_scales=pow_2_scales,
                    eps=eps,
                ),
                quant_tile_shape,
            )

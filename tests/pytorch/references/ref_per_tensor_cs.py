# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import transformer_engine_torch as tex

from transformer_engine.pytorch.constants import TE_DType_To_Torch


# compute amax and scale
def _ref_compute_amax_scale(x, quant_dtype, eps, pow_2_scales):
    x_fp32 = x.to(torch.float32)
    amax = torch.amax(torch.abs(x_fp32)).view(1)
    assert amax.dtype == torch.float, "amax must be a float tensor."
    fp8_max = torch.finfo(quant_dtype).max
    # Clamping amax to avoid division by small numbers
    amax = torch.max(amax, torch.tensor(eps))

    # Compute scale factor
    scale = torch.div(fp8_max, amax)
    # Note frexp doesn't give back inf for exponent with an inf input
    # We take care of inf before pow_2_scales
    # option1: set scale to fp32 max when scale is inf
    scale = torch.where(scale == torch.inf, torch.finfo(torch.float32).max, scale)
    # option2: when scale is inf, set scale to 1
    scale = torch.where(scale == torch.inf, 1.0, scale)
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
        # TODO: If/when adding a URM option an option is to cap to 126
        # rather than allowing the full range of FP32 (2 - 2^23) x 2^127
        # addresses cases where adding a mantissa overflows into inf scales.
        # Not necessary currently without additional scale smudging options.
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


def _multi_dim_transpose(tensor):
    # Get the number of dimensions
    dims = list(range(len(tensor.shape)))

    if len(dims) <= 1:
        return tensor

    # circular shift of shapes
    new_order = []
    new_order.append(dims[-1])
    for i in range(len(dims) - 1):
        new_order.append(dims[i])

    # Permute the tensor according to the new order
    output_tensor = tensor.permute(new_order).contiguous()

    return output_tensor


# current scaling reference quantization
def ref_per_tensor_cs_cast(
    tensor: torch.Tensor,
    fp8_dtype: tex.DType = tex.DType.kFloat8E4M3,
    return_transpose: bool = False,
    force_pow_2_scales: bool = False,
    amax_epsilon: float = 0.0,
) -> torch.Tensor:

    quant_dtype_torch = TE_DType_To_Torch[fp8_dtype]
    scale, scale_inv, _ = _ref_compute_amax_scale(
        tensor,
        quant_dtype_torch,
        amax_epsilon,
        force_pow_2_scales,
    )

    qx = (tensor.float() * scale).to(quant_dtype_torch)
    sx = scale_inv
    qx_t = None
    sx_t = None

    if tensor.shape == torch.Size([]):
        qx = qx.view([])

    if return_transpose:
        qx_t = _multi_dim_transpose(qx)
        sx_t = sx
    return qx, sx, qx_t, sx_t

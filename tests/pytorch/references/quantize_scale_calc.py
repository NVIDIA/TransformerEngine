# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import Tuple
import torch


def scale_from_amax_tensor(
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
    fp8_max = torch.tensor(torch.finfo(quant_dtype).max, dtype=torch.float32, device=amax.device)
    # Clamping amax to avoid division by small numbers
    eps_tensor = torch.tensor(eps, dtype=amax.dtype, device=amax.device)
    # `<` is false for NaN, matching the device function's epsilon floor.
    amax = torch.where(amax < eps_tensor, eps_tensor, amax)

    # Floor a positive amax at max_fp8 * FLT_MIN so scale_inv is at least FLT_MIN.
    guard = fp8_max * torch.finfo(torch.float32).tiny
    invalid = torch.isinf(amax) | torch.isnan(amax) | (amax == 0)
    floored = torch.where(amax < guard, guard, amax)
    safe_amax = torch.where(invalid, torch.ones_like(amax), floored)

    # Compute scale factor
    scale = torch.div(fp8_max, safe_amax)
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
        # a floored amax keeps the smallest fp8_max / amax at 2^126.
        exp = exp - 1
        # No subnormals and zero.
        assert (exp > -127).all()
        unity = torch.tensor([1.0], device=exp.device)
        torch.ldexp(unity, exp, out=scale)

    scale = torch.where(invalid, torch.ones_like(scale), scale)

    # Compute scale_inv
    scale_inv = torch.reciprocal(scale)

    return scale, scale_inv, amax

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import transformer_engine_torch as tex

from transformer_engine.pytorch.constants import TE_DType_To_Torch
from tests.pytorch.references.quantize_scale_calc import (
    scale_from_amax_tensor
)


# compute amax and scale
def _ref_compute_amax_scale(x, quant_dtype, eps, pow_2_scales):
    x_fp32 = x.to(torch.float32)
    amax = torch.amax(torch.abs(x_fp32)).view(1)
    return scale_from_amax_tensor(torch.float32,
                                  amax,
                                  quant_dtype,
                                  eps=eps,
                                  pow_2_scales=pow_2_scales)

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

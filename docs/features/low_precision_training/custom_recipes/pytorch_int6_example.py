# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_INT6_EXAMPLE
import dataclasses
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.quantized_tensor import Quantizer, QuantizedTensorStorage
from transformer_engine.pytorch.custom_recipes.quantization import MMParams, GEMMType


@dataclasses.dataclass
class Int6TensorStorage(QuantizedTensorStorage):
    """
    Custom tensor storage for Int6 quantization.

    The `custom = True` property triggers the custom GEMM dispatch path,
    routing GEMM operations to the quantizer's qgemm() method.
    """

    custom: bool = True

    data: torch.Tensor = None  # Rowwise quantized data
    data_t: torch.Tensor = None  # Columnwise quantized data (transposed)
    scale: torch.Tensor = None  # Rowwise scale
    scale_t: torch.Tensor = None  # Columnwise scale
    dtype: torch.dtype = None  # Original dtype
    original_shape: tuple = None  # Original tensor shape


class Int6Quantizer(Quantizer):
    """
    Custom Int6 quantizer with custom GEMM implementation.

    This quantizer demonstrates the full custom recipe flow:
    1. quantize_impl() converts high-precision tensor to Int6TensorStorage
    2. qgemm() performs matrix multiplication with dequantization
    """

    def quantize_impl(self, tensor: torch.Tensor) -> Int6TensorStorage:
        original_shape = tensor.shape
        original_dtype = tensor.dtype

        # Reshape to 2D for quantization
        if tensor.dim() > 2:
            tensor = tensor.view(-1, tensor.shape[-1])

        # Rowwise quantization (int6 range: [-32, 31])
        row_max = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = row_max / 31.0
        data = (tensor / scale).round().clamp(-32, 31)

        # Columnwise quantization for transpose
        col_max = tensor.abs().amax(dim=0, keepdim=True).clamp(min=1e-12)
        scale_t = col_max / 31.0
        data_t = (tensor / scale_t).round().clamp(-32, 31).T.contiguous()

        return Int6TensorStorage(
            data=data,
            data_t=data_t,
            scale=scale.squeeze(-1),
            scale_t=scale_t.squeeze(0),
            dtype=original_dtype,
            original_shape=original_shape,
            _quantizer=self,
        )

    def qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        m_params: MMParams,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor | None = None,
        gemm_type: GEMMType = GEMMType.FPROP,
        **kwargs,
    ) -> torch.Tensor:
        """
        Custom GEMM for Int6 tensors.

        Dequantizes inputs, performs matmul, and returns result.
        """
        # Dequantize: multiply quantized values by their scales
        x_hp = qx.float() * sx.unsqueeze(-1)
        w_hp = qw.float() * sw.unsqueeze(-1)

        # Matrix multiplication (x @ w.T for standard GEMM)
        result = torch.mm(x_hp, w_hp.T)

        # Add bias if present (only in FPROP)
        if bias is not None:
            result = result + bias

        return result.to(out_dtype)


def int6_factory(role: str):
    return Int6Quantizer(rowwise=True, columnwise=True)


custom_recipe = recipe.CustomRecipe(qfactory=int6_factory)

# Example usage:
model = te.Linear(64, 64).cuda()
x = torch.randn(32, 64, device="cuda").requires_grad_(True)

with te.autocast(enabled=True, recipe=custom_recipe):
    y = model(x)

loss = y.sum()
loss.backward()
# END_INT6_EXAMPLE

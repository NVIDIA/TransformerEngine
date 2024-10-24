# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for bias."""

from __future__ import annotations
from typing import Optional

import torch

from ...cpp_extensions import fp8_cast_transpose_bgrad_fused
from ...float8_tensor import Float8Tensor
from ...fp8 import FP8GlobalStateManager, get_fp8_te_dtype
from ..op import BasicOperation, OperationContext
from .._common import (
    canonicalize_device,
    canonicalize_dtype,
    convert_tensor,
    is_float8_tensor,
    reshape,
)


class Bias(BasicOperation):
    """Apply additive bias

    This is equivalent to the additive bias in `torch.nn.Linear`.

    Parameters
    ----------
    size: int
        Inner dimension of input tensor
    device: torch.device, default = default CUDA device
        Tensor device
    dtype: torch.dtype, default = default dtype
        Tensor datatype
    tensor_parallel: bool, default = `False`
        Whether to distribute input tensor and bias tensors along
        inner dimension
    tensor_parallel_group: torch.distributed.ProcessGroup, default = world group
        Process group for tensor parallelism

    """

    def __init__(
        self,
        size: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        tensor_parallel: bool = False,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        super().__init__()

        # Bias size
        self._size = size

        # Bias tensor device
        defer_param_init = False
        device = canonicalize_device(device)
        if device.type == "meta":
            defer_param_init = True
            device = canonicalize_device(None)
        if device.type != "cuda":
            raise ValueError(f"Only CUDA devices are supported (got {device})")
        self.device: torch.device = device

        # Tensor parallel configuration
        tensor_parallel_size = 1
        local_size = size
        if tensor_parallel:
            tensor_parallel_size = torch.distributed.get_world_size(tensor_parallel_group)
            tensor_parallel = tensor_parallel_size > 1
            if size % tensor_parallel_size != 0:
                raise ValueError(
                    "Invalid configuration for tensor parallelism "
                    f"({size=}, {tensor_parallel_size=})"
                )
            local_size //= tensor_parallel_size
        else:
            tensor_parallel_group = None
        self.tensor_parallel: bool = tensor_parallel
        self.tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = tensor_parallel_group
        self.tensor_parallel_size: int = tensor_parallel_size
        self.local_size: int = local_size

        # Initialize parameters if needed
        bias = torch.empty(
            local_size,
            device="meta",
            dtype=canonicalize_dtype(dtype),
        )
        bias = torch.nn.Parameter(bias)
        self.bias: torch.nn.Parameter
        self.register_parameter("bias", bias)
        if not defer_param_init:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameter buffers and values"""

        # Make sure parameter is initialized
        bias = self.bias
        if bias.device.type != self.device.type:
            bias = torch.empty_like(bias, device=self.device)
        else:
            bias = bias.to(device=self.device)

        # Initialize values
        bias.zero_()

        # Save updated parameter
        if not isinstance(bias, torch.nn.Parameter):
            bias = torch.nn.Parameter(bias)
        self.bias = bias

    def pre_forward(self) -> None:
        super().pre_forward()
        if self.bias.device.type == "meta":
            self.reset_parameters()

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
    ) -> torch.Tensor:

        # Get autocast dtype if needed
        dtype = None
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = self.bias.dtype

        # Apply bias
        x = convert_tensor(input_, dtype=dtype)
        b = reshape(self.bias, [1] * (x.dim() - 1) + [self.local_size])
        y = x + b

        # Save state for backward pass
        ctx.bias_requires_grad = self.bias.requires_grad
        ctx.with_fp8_compute = FP8GlobalStateManager.is_fp8_enabled()
        ctx.dtype = dtype
        ctx.prev_op = prev_op

        return y

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:

        # Check if FP8 is enabled
        with_fp8_grad_input = (
            ctx.with_fp8_compute
            and ctx.prev_op is not None
            and ctx.prev_op.num_fp8_scales("grad_output") > 0
            and grad_output.size(-1) % 16 == 0
            and grad_output.numel() // grad_output.size(-1) % 16 == 0
        )

        # Compute grad bias
        dy = grad_output
        db = None
        dx: torch.Tensor
        if not ctx.bias_requires_grad:
            # Trivial case: Don't compute bgrad, don't do anything
            # with dgrad
            dx = dy
        if not with_fp8_grad_input or is_float8_tensor(dy):
            # Non-FP8 case: Compute bgrad, don't do anything with
            # dgrad
            if dy.dim() > 1:
                db = dy.sum(tuple(range(dy.dim() - 1)))
            else:
                db = dy
            dx = dy
        else:
            # FP8 case: Call fused kernel to compute bgrad and cast
            # dgrad to FP8

            # Check grad output tensor
            output_dims = grad_output.size()
            dy = reshape(
                dy,
                (-1, output_dims[-1]),
                device=self.device,
                dtype=ctx.dtype,
            )

            # Call fused kernel for bgrad and casting dgrad to FP8
            fp8_meta = ctx.prev_op.get_fp8_meta("grad_output")
            fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(forward=False)
            fp8_dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=False)
            fp8_scale_inv = torch.empty([1], dtype=torch.float32, device=self.device)
            db, dx_data, dx_data_transpose = fp8_cast_transpose_bgrad_fused(
                dy,
                fp8_meta[fp8_meta_key],
                0,
                fp8_dtype,
                scale_inv=fp8_scale_inv,
            )

            # Construct grad input tensor
            if dx_data.size() != output_dims:
                dx_data = dx_data.reshape(output_dims)
            dx = Float8Tensor(
                data=dx_data,
                fp8_meta=fp8_meta,
                fp8_meta_forward=False,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
                fp8_scale_inv=fp8_scale_inv,
                dtype=ctx.dtype,
            )
            dx._transpose = dx_data_transpose
            dx._transpose_invalid = False

        return dx, (db,)

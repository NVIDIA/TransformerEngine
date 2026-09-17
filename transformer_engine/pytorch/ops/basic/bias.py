# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for bias."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch

import transformer_engine_torch as tex
from ...dynamo import TensorOrQuantized, TensorSpec
from ...quantization import FP8GlobalStateManager
from ..op import BasicOperation, OperationContext
from ...utils import canonicalize_device, canonicalize_dtype
from ...tensor import Quantizer


@dataclass(slots=True)
class BiasFwdArgs:
    """Tensor inputs and backward quantizer for the bias forward custom op."""

    input_: torch.Tensor
    bias: torch.Tensor
    grad_input_quantizer: Optional[Quantizer]


@dataclass(slots=True)
class BiasBwdArgs:
    """Output gradient and optional quantizer for the bias backward custom op."""

    grad_output: TensorOrQuantized
    grad_input_quantizer: Optional[Quantizer]


class Bias(BasicOperation):
    """Apply additive bias

    This is equivalent to the additive bias in ``torch.nn.Linear``.

    Parameters
    ----------
    size : int
        Inner dimension of input tensor
    device : torch.device, default = default CUDA device
        Tensor device
    dtype : torch.dtype, default = default dtype
        Tensor datatype
    tensor_parallel : bool, default = False
        Whether to distribute input tensor and bias tensors along
        inner dimension
    tensor_parallel_group : torch.distributed.ProcessGroup, default = world group
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
        if bias.device.type != "cuda":
            bias = torch.empty_like(bias, device=self.device)
        else:
            bias = bias.to(device=self.device)

        # Initialize values
        bias.zero_()

        # Save updated parameter
        if not isinstance(bias, torch.nn.Parameter):
            bias = torch.nn.Parameter(bias)
        self.bias = bias

    def pre_first_fuser_forward(self) -> None:
        super().pre_first_fuser_forward()
        if self.bias.device.type == "meta":
            self.reset_parameters()

    fwd_args_type = BiasFwdArgs
    bwd_args_type = BiasBwdArgs

    def pack_forward_args(
        self,
        basic_op_ctxs,
        input_,
        *,
        prev_op_grad_output_quantizer,
        basic_op_kwargs,
        **unused,  # pylint: disable=unused-argument
    ) -> BiasFwdArgs:
        if basic_op_kwargs[0]:
            raise ValueError("Bias forward does not expect keyword arguments")
        quantizer = prev_op_grad_output_quantizer
        if FP8GlobalStateManager.is_fp8_enabled():
            if FP8GlobalStateManager.get_fp8_recipe().backward_override is not None:
                quantizer = None
        return BiasFwdArgs(input_, self.bias, quantizer)

    @classmethod
    def forward_compute(cls, args: BiasFwdArgs):
        return args.input_ + args.bias, [()], ()

    @classmethod
    def forward_compute_fake(cls, args: BiasFwdArgs):
        return (
            TensorSpec(
                shape=args.input_.shape,
                dtype=torch.promote_types(args.input_.dtype, args.bias.dtype),
                device=args.input_.device,
            ),
            [()],
            (),
        )

    def forward_setup_context(self, basic_op_ctxs, args, aux) -> None:
        basic_op_ctxs[0].grad_input_quantizer = args.grad_input_quantizer

    def pack_backward_args(
        self, basic_op_ctxs, grad_output, **unused  # pylint: disable=unused-argument
    ) -> BiasBwdArgs:
        return BiasBwdArgs(grad_output, basic_op_ctxs[0].grad_input_quantizer)

    @classmethod
    def backward_compute(cls, args: BiasBwdArgs):
        dy = args.grad_output
        if dy.dim() == 1:
            return None, [(dy.clone(),)], [()]
        if args.grad_input_quantizer is None:
            return None, [(dy.sum(tuple(range(dy.dim() - 1))),)], [()]
        db, dx = tex.bgrad_quantize(dy, args.grad_input_quantizer)
        return dx, [(db,)], [()]

    @classmethod
    def backward_compute_fake(cls, args: BiasBwdArgs):
        dy = args.grad_output
        dx = None
        if len(dy.shape) > 1 and args.grad_input_quantizer is not None:
            dx = TensorSpec(
                shape=dy.shape,
                dtype=dy.dtype,
                device=dy.device,
                quantizer=args.grad_input_quantizer,
            )
        db = TensorSpec(shape=(dy.shape[-1],), dtype=dy.dtype, device=dy.device)
        return dx, [(db,)], [()]

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        x = input_
        b = self.bias.view([1] * (x.dim() - 1) + [self.local_size])

        if ctx.requires_grad:
            ctx.grad_input_quantizer = prev_op_grad_output_quantizer
            if FP8GlobalStateManager.is_fp8_enabled():
                fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
                if fp8_recipe.backward_override is not None:
                    ctx.grad_input_quantizer = None

        return x + b

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        dy = grad_output
        if dy.dim() > 1:
            quantizer = ctx.grad_input_quantizer
            if quantizer is None:
                db = dy.sum(tuple(range(dy.dim() - 1)))
            else:
                db, dy = tex.bgrad_quantize(dy, quantizer)
        else:
            db = dy
        return dy, (db,)

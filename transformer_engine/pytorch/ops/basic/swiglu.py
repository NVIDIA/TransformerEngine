# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for SwiGLU and variants."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...tensor import Float8CurrentScalingQuantizer, Quantizer
from ...utils import clear_tensor_data
from ..op import BasicOperation, OperationContext
from .._common import maybe_dequantize

__all__ = ["SwiGLU", "ClampedSwiGLU", "ScaledSwiGLU"]


class SwiGLU(BasicOperation):
    r"""Swish gated linear unit

    The input tensor is split into chunks :math:``a`` and :math:``b``
    along the last dimension and the following is computed:

    .. math::

       \text{SwiGLU}(a,b) = \text{SiLU}(a) * b

    where

    .. math::

       \text{SiLU}(x) = x \sigma(x) = \frac{x}{1+\exp(-x)}

    .. warning::

       Transformer Engine's gated activations and PyTorch's GLU
       activation follow opposite conventions for :math:``a`` and
       :math:``b``. Transformer Engine applies the gating function to
       the first half of the input tensor, while PyTorch applies it to
       the second half.

    The Sigmoid Linear Unit (SiLU) gating function is also known as
    the swish function. See
    `GLU Variants Improve Transformer <https://arxiv.org/abs/2002.05202>`__.

    Parameters
    ----------
    cache_quantized_input : bool, default = False
        Quantize input tensor when caching for use in the backward
        pass. This will typically reduce memory usage but require
        extra compute and increase numerical error. This feature is
        highly experimental.
    glu_interleave_size : int, optional
        When set, the GLU activations will use a block interleaved
        format. Instead of interpreting the input tensor as a
        concatenation of gates and linear units (e.g.
        :math:``[a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4]``
        in the above notation), it will be interpreted
        as alternating blocks of gates and linear units (e.g.
        :math:``[a_1, a_2, b_1, b_2, a_3, a_4, b_3, b_4]``
        when the interleave size is 2). This data format is highly
        experiental and is primarily intended to support some advanced
        fused kernels.

    """

    def __init__(
        self,
        *,
        cache_quantized_input: bool = False,
        glu_interleave_size: Optional[int] = None,
    ):
        super().__init__()
        self.cache_quantized_input: bool = cache_quantized_input
        self.glu_interleave_size: Optional[int] = glu_interleave_size

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:

        # Compute dtype
        dtype: torch.dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = input_.dtype
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise RuntimeError(f"Unsupported dtype ({dtype})")

        # Check input tensor
        input_ = maybe_dequantize(input_.contiguous(), dtype)

        # Remove interleaving if needed
        swiglu_in = input_
        if self.glu_interleave_size is not None:
            shape = swiglu_in.size()
            swiglu_in = swiglu_in.reshape(
                -1,
                shape[-1] // (2 * self.glu_interleave_size),
                2,
                self.glu_interleave_size,
            )
            swiglu_in = swiglu_in.transpose(1, 2).contiguous()
            swiglu_in = swiglu_in.view(shape)

        # Launch kernel
        out = tex.swiglu(swiglu_in, next_op_input_quantizer)

        # Quantize input to FP8 before caching if needed
        if self.cache_quantized_input:
            input_quantizer = Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E4M3,
                input_.device,
            )
            input_quantizer.set_usage(rowwise=True, columnwise=False)
            input_ = input_quantizer(input_)

        # Save state for backward pass
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(input_)
            ctx.save_for_backward(input_)
            ctx.dtype = dtype
            ctx.prev_op_grad_output_quantizer = prev_op_grad_output_quantizer

        return out

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:

        # Saved tensors from forward pass
        (input_,) = ctx.saved_tensors

        # Make sure tensors have correct dtypes
        x = maybe_dequantize(input_.contiguous(), ctx.dtype)
        dy = maybe_dequantize(grad_output.contiguous(), ctx.dtype)

        # Remove interleaving if needed
        swiglu_in = x
        if self.glu_interleave_size is not None:
            shape = swiglu_in.size()
            swiglu_in = swiglu_in.reshape(
                -1,
                shape[-1] // (2 * self.glu_interleave_size),
                2,
                self.glu_interleave_size,
            )
            swiglu_in = swiglu_in.transpose(1, 2).contiguous()
            swiglu_in = swiglu_in.view(shape)

        # Quantizer for grad input
        quantizer = ctx.prev_op_grad_output_quantizer
        if self.glu_interleave_size is not None:
            quantizer = None

        # Launch kernel
        grad_swiglu_in = tex.dswiglu(dy, swiglu_in, quantizer)

        # Apply interleaving if needed
        dx = grad_swiglu_in
        if self.glu_interleave_size is not None:
            shape = dx.size()
            dx = dx.reshape(
                -1,
                2,
                shape[-1] // (2 * self.glu_interleave_size),
                self.glu_interleave_size,
            )
            dx = dx.transpose(1, 2).contiguous()
            dx = dx.view(shape)

        # Clear input tensor if possible
        clear_tensor_data(input_)

        return dx, ()


class ClampedSwiGLU(BasicOperation):
    r"""GPT-OSS
    Implementation based on `GPT-OSS <https://github.com/openai/gpt-oss/blob/a0a84273e9e0c14a233cb9befdfd159c2bcfa6cd/gpt_oss/torch/model.py#L250>`__.

    This activation has two differences compared to the original SwiGLU
       1. Both gate and pre-activations are clipped based on parameter limit.
       2. Activation uses sigmoid(alpha * x) instead of sigmoid(x) used in Swish activation.

    .. warning::

       The input tensor is chunked along the last dimension to get
       gates/pre-activations which is different from GPT OSS
       implementation where the gates/pre-activations are assumed to
       be interleaved in the input tensor.

    Parameters
    ----------
    limit : float
        The clamp limit.
    alpha : float
        The scaling factor for the sigmoid function used in the activation.
    cache_quantized_input : bool, default = ``False``
        Quantize input tensor when caching for use in the backward pass.
    glu_interleave_size : int, optional
        When set, the GLU activations will use an experimental block
        interleaved format. See the corresponding option in the SwiGLU
        operation for more details.

    """

    def __init__(
        self,
        *,
        limit: float = 7.0,
        alpha: float = 1.702,
        cache_quantized_input: bool = False,
        glu_interleave_size: Optional[int] = None,
    ):
        super().__init__()
        self.limit: float = limit
        self.alpha: float = alpha
        self.cache_quantized_input: bool = cache_quantized_input
        self.glu_interleave_size: Optional[int] = glu_interleave_size

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:

        # Compute dtype
        dtype: torch.dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = input_.dtype
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise RuntimeError(f"Unsupported dtype ({dtype})")

        # Check input tensor
        x = maybe_dequantize(input_.contiguous(), dtype)

        # Remove interleaving if needed
        swiglu_in = input_
        if self.glu_interleave_size is not None:
            shape = swiglu_in.size()
            swiglu_in = swiglu_in.reshape(
                -1,
                shape[-1] // (2 * self.glu_interleave_size),
                2,
                self.glu_interleave_size,
            )
            swiglu_in = swiglu_in.transpose(1, 2).contiguous()
            swiglu_in = swiglu_in.view(shape)

        # Launch kernel
        out = tex.clamped_swiglu(
            swiglu_in,
            next_op_input_quantizer,
            limit=self.limit,
            alpha=self.alpha,
        )

        # Quantize input to FP8 before caching if needed
        if self.cache_quantized_input:
            input_quantizer = Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, x.device)
            input_quantizer.set_usage(rowwise=True, columnwise=False)
            x = input_quantizer(x)

        # Save state for backward pass
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(x)
            ctx.save_for_backward(x)
            ctx.dtype = dtype
            ctx.prev_op_grad_output_quantizer = prev_op_grad_output_quantizer

        return out

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:

        # Saved tensors from forward pass
        (input_,) = ctx.saved_tensors

        # Make sure tensors have correct dtypes
        x = maybe_dequantize(input_.contiguous(), ctx.dtype)
        dy = maybe_dequantize(grad_output.contiguous(), ctx.dtype)

        # Remove interleaving if needed
        swiglu_in = x
        if self.glu_interleave_size is not None:
            shape = swiglu_in.size()
            swiglu_in = swiglu_in.reshape(
                -1,
                shape[-1] // (2 * self.glu_interleave_size),
                2,
                self.glu_interleave_size,
            )
            swiglu_in = swiglu_in.transpose(1, 2).contiguous()
            swiglu_in = swiglu_in.view(shape)

        # Quantizer for grad input
        quantizer = ctx.prev_op_grad_output_quantizer
        if self.glu_interleave_size is not None:
            quantizer = None

        # Launch kernel
        grad_swiglu_in = tex.clamped_dswiglu(
            dy,
            swiglu_in,
            quantizer,
            limit=self.limit,
            alpha=self.alpha,
        )

        # Apply interleaving if needed
        dx = grad_swiglu_in
        if self.glu_interleave_size is not None:
            shape = dx.size()
            dx = dx.reshape(
                -1,
                2,
                shape[-1] // (2 * self.glu_interleave_size),
                self.glu_interleave_size,
            )
            dx = dx.transpose(1, 2).contiguous()
            dx = dx.view(shape)

        # Clear input tensor if possible
        clear_tensor_data(input_)

        return dx, ()


class ScaledSwiGLU(BasicOperation):
    r"""SwiGLU with post-scaling.

    If the SwiGLU output has shape ``(d_1, ..., d_n)``, it is
    multiplied with an extra input tensor of shape
    ``(d_1, ..., d_{n-1})``.

    Parameters
    ----------
    glu_interleave_size : int, optional
        When set, the GLU activations will use an experimental block
        interleaved format. See the corresponding option in the SwiGLU
        operation for more details.

    """

    # Operation expects scales
    num_extra_inputs: int = 1

    def __init__(self, glu_interleave_size: Optional[int] = None):
        super().__init__()
        self.glu_interleave_size: Optional[int] = glu_interleave_size

    def op_forward(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} operation has "
            f"{self.num_extra_inputs} extra tensor inputs "
            f"and {self.num_extra_outputs} extra tensor outputs. "
            "It overrides `fuser_forward` instead of `op_forward`."
        )

    def op_backward(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} operation has "
            f"{self.num_extra_inputs} extra tensor inputs "
            f"and {self.num_extra_outputs} extra tensor outputs. "
            "It overrides `fuser_backward` instead of `op_backward`."
        )

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:
        extra_input = basic_op_extra_inputs[0][0]

        # Determine compute dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        elif isinstance(input_, torch.Tensor):
            dtype = input_.dtype
        else:
            dtype = extra_input.dtype

        # Make sure inputs are in correct dtype
        input_ = maybe_dequantize(input_, dtype)
        scales = maybe_dequantize(extra_input, dtype)

        # Remove gate interleaving if needed
        swiglu_in = input_
        if self.glu_interleave_size is not None:
            shape = swiglu_in.size()
            swiglu_in = swiglu_in.reshape(
                -1,
                shape[-1] // (2 * self.glu_interleave_size),
                2,
                self.glu_interleave_size,
            )
            swiglu_in = swiglu_in.transpose(1, 2).contiguous()
            swiglu_in = swiglu_in.view(shape)

        # Compute scaled SwiGLU
        swiglu_out = tex.swiglu(swiglu_in, None)
        out = swiglu_out * scales.unsqueeze(-1)

        # Save state for backward pass
        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(input_)
            ctx.input_requires_grad = True
            ctx.extra_input_requires_grad = extra_input.requires_grad
            ctx.dtype = dtype
            ctx.save_for_backward(
                input_,
                scales if ctx.input_requires_grad else None,
            )

        return out, [()]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        Iterable[Iterable[Optional[torch.Tensor]]],
        Iterable[Iterable[Optional[torch.Tensor]]],
    ]:
        ctx = basic_op_ctxs[0]
        input_, scales = ctx.saved_tensors
        input_ = maybe_dequantize(input_, ctx.dtype)
        if scales is not None:
            scales = maybe_dequantize(scales, ctx.dtype)
        grad_output = maybe_dequantize(grad_output, ctx.dtype)

        # Remove gate interleaving if needed
        swiglu_in = input_
        if self.glu_interleave_size is not None:
            shape = swiglu_in.size()
            swiglu_in = swiglu_in.reshape(
                -1,
                shape[-1] // (2 * self.glu_interleave_size),
                2,
                self.glu_interleave_size,
            )
            swiglu_in = swiglu_in.transpose(1, 2).contiguous()
            swiglu_in = swiglu_in.view(shape)

        # Compute input grad
        grad_input = None
        if ctx.input_requires_grad:
            grad_swiglu_out = grad_output * scales.unsqueeze(-1)
            grad_swiglu_in = tex.dswiglu(grad_swiglu_out, swiglu_in, None)
            grad_input = grad_swiglu_in
            if self.glu_interleave_size is not None:
                shape = grad_input.size()
                grad_input = grad_input.reshape(
                    -1,
                    2,
                    shape[-1] // (2 * self.glu_interleave_size),
                    self.glu_interleave_size,
                )
                grad_input = grad_input.transpose(1, 2).contiguous()
                grad_input = grad_input.view(shape)

        # Compute scales grad by recomputing SwiGLU
        grad_extra_input = None
        if ctx.extra_input_requires_grad:
            swiglu_out = tex.swiglu(swiglu_in, None)
            grad_extra_input = torch.linalg.vecdot(swiglu_out, grad_output)

        # Clear input tensor if possible
        clear_tensor_data(ctx.saved_tensors[0])  # input_

        return grad_input, [()], [(grad_extra_input,)]

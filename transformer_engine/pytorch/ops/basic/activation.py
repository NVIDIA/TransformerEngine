# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operations for activation functions."""

from __future__ import annotations
from collections.abc import Callable
from typing import Optional

import torch

import transformer_engine_torch
from ...constants import TE_DType
from ...cpp_extensions import (
    geglu as tex_geglu,
    gelu as tex_gelu,
    reglu as tex_reglu,
    relu as tex_relu,
    swiglu as tex_swiglu,
)
from ...float8_tensor import Float8Tensor
from ...fp8 import FP8GlobalStateManager, get_fp8_te_dtype
from ..op import BasicOperation, OperationContext
from .._common import devices_match, is_float8_tensor


class _ActivationOperation(BasicOperation):
    r"""Apply activation function

    Activation functions are either element-wise unary functions or
    variants of the gated linear unit (GLU). Recall that GLU is
    computed by splitting the input tensor into chunks :math:`a` and
    :math:`b` along the last dimension and computing

    .. math::
       \text{GLU}(a,b) = \sigma(a) * b

    .. warning::

       Transformer Engine gated activations and PyTorch's GLU
       activation follow opposite conventions for :math:`a` and
       :math:`b`. Transformer Engine applies the gating function to
       the first half of the input tensor, while PyTorch applies it to
       the second half.

    """

    # Forward impl in transformer_engine.pytorch.cpp_extensions
    _forward_tex_function: Optional[Callable] = None
    # Backward impl in transformer_engine.pytorch.cpp_extensions
    _backward_tex_function: Optional[Callable] = None

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
    ) -> torch.Tensor:

        # Check input tensor
        x = input_
        if is_float8_tensor(x):
            x = x.from_float8()
        if x.device.type != "cuda":
            x = x.cuda()
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            x = x.float()
        if not x.is_contiguous():
            x = x.contiguous()

        # Check if FP8 is enabled
        with_fp8_output = False
        output_fp8_meta = None
        output_dtype = TE_DType[x.dtype]
        output_fp8_scale_inv = None
        if (
            FP8GlobalStateManager.is_fp8_enabled()
            and next_op is not None
            and next_op.num_fp8_scales("input") > 0
        ):
            with_fp8_output = True
            fp8_meta = next_op.get_fp8_meta("input")
            fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(forward=True)
            output_fp8_meta = fp8_meta[fp8_meta_key]
            output_dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
            output_fp8_scale_inv = torch.empty([1], dtype=torch.float32, device=x.device)

        # Launch kernel
        y = self.__class__._forward_tex_function(
            x,
            output_fp8_meta,
            0,
            output_dtype,
            scale_inv=output_fp8_scale_inv,
        )

        # Check output tensor
        if y.dim() != x.dim():
            y = y.reshape(list(x.shape[:-1]) + [-1])
        if with_fp8_output:
            y = Float8Tensor(
                data=y,
                fp8_meta=output_fp8_meta,
                fp8_meta_forward=True,
                fp8_meta_index=0,
                fp8_dtype=output_dtype,
                fp8_scale_inv=output_fp8_scale_inv,
                dtype=x.dtype,
            )

        # Save state for backward pass
        ctx.save_for_backward(x)
        ctx.prev_op = prev_op

        return y

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:

        # Saved tensors from forward pass
        (x,) = ctx.saved_tensors

        # Check grad output tensor
        dy = grad_output
        if is_float8_tensor(dy):
            dy = dy.from_float8()
        if not devices_match(dy.device, x.device) or dy.dtype != x.dtype:
            dy = dy.to(device=x.device, dtype=x.dtype)
        if not dy.is_contiguous():
            dy = dy.contiguous()

        # Launch kernel
        dx = self.__class__._backward_tex_function(dy, x, TE_DType[x.dtype])

        # Check grad input tensor
        if dx.size() != x.size():
            dx = dx.reshape(x.size())

        # Clear input tensor if possible
        if ctx.prev_op is not None:
            clear_tensor_data(x)

        return dx, ()


class GELU(_ActivationOperation):
    r"""Gaussian Error Linear Unit

    This computes the "tanh" approximation to GELU:

    .. math::

       \text{GELU}(x) \approx \frac{x}{2} \left( 1 + \tanh\left( 0.797x+0.036 x^3 \right) \right)

    See `Gaussian Error Linear Units (GELUs)<https://arxiv.org/abs/1606.08415>`__.

    """

    _forward_tex_function: Callable = tex_gelu
    _backward_tex_function: Callable = transformer_engine_torch.dgelu


class ReLU(_ActivationOperation):
    r"""Rectified linear unit

    .. math::

       \text{ReLU}(x) = \max(x,0)

    """

    _forward_tex_function: Callable = tex_relu
    _backward_tex_function: Callable = transformer_engine_torch.drelu


class GEGLU(_ActivationOperation):
    r"""Gaussian error gated linear unit

    The input tensor is split into chunks :math:`a` and :math:`b`
    along the last dimension and the following is computed:

    .. math::

       \text{GEGLU}(a,b) = \text{GELU}(a) * b

    where

    .. math::

       \text{GELU}(x) \approx \frac{x}{2} \left( 1 + \tanh\left( 0.797x+0.036 x^3 \right) \right)

    .. warning::

       Transformer Engine's gated activations and PyTorch's GLU
       activation follow opposite conventions for :math:`a` and
       :math:`b`. Transformer Engine applies the gating function to
       the first half of the input tensor, while PyTorch applies it to
       the second half.

    See `GLU Variants Improve Transformer<https://arxiv.org/abs/2002.05202>`__.

    """

    _forward_tex_function: Callable = tex_geglu
    _backward_tex_function: Callable = transformer_engine_torch.dgeglu


class ReGLU(_ActivationOperation):
    r"""Rectified gated linear unit

    The input tensor is split into chunks :math:`a` and :math:`b`
    along the last dimension and the following is computed:

    .. math::

       \text{ReGLU}(a,b) = \max(a,0) * b

    .. warning::

       Transformer Engine's gated activations and PyTorch's GLU
       activation follow opposite conventions for :math:`a` and
       :math:`b`. Transformer Engine applies the gating function to
       the first half of the input tensor, while PyTorch applies it to
       the second half.

    See `GLU Variants Improve Transformer<https://arxiv.org/abs/2002.05202>`__.

    """

    _forward_tex_function: Callable = tex_reglu
    _backward_tex_function: Callable = transformer_engine_torch.dreglu


class SwiGLU(_ActivationOperation):
    r"""Swish gated linear unit

    The input tensor is split into chunks :math:`a` and :math:`b`
    along the last dimension and the following is computed:

    .. math::

       \text{GEGLU}(a,b) = \text{SiLU}(a) * b

    where

    .. math::

       \text{SiLU}(x) = x \sigma(x) = \frac{x}{1+\exp(-x)}

    .. warning::

       Transformer Engine's gated activations and PyTorch's GLU
       activation follow opposite conventions for :math:`a` and
       :math:`b`. Transformer Engine applies the gating function to
       the first half of the input tensor, while PyTorch applies it to
       the second half.

    The Sigmoid Linear Unit (SiLU) gating function is also known as
    the swish function. See
    `GLU Variants Improve Transformer<https://arxiv.org/abs/2002.05202>`__
    and `Gaussian Error Linear Units (GELUs)<https://arxiv.org/abs/1606.08415>`__.

    """

    _forward_tex_function: Callable = tex_swiglu
    _backward_tex_function: Callable = transformer_engine_torch.dswiglu

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operations for activation functions."""

from __future__ import annotations
import abc
from typing import Optional

import torch

import transformer_engine_torch as tex
from ...fp8 import FP8GlobalStateManager
from ...tensor import QuantizedTensor
from ...tensor.float8_tensor import Float8CurrentScalingQuantizer
from ...utils import clear_tensor_data, devices_match
from ..op import BasicOperation, OperationContext
from .._common import reshape


class _ActivationOperation(BasicOperation, metaclass=abc.ABCMeta):
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

    Parameters
    ----------
    cache_quantized_input: bool, default = False
        Quantize input tensor when caching for use in the backward
        pass. This will typically reduce memory usage but require
        extra compute and increase numerical error. This feature is
        highly experimental.

    """

    def __init__(self, *, cache_quantized_input: bool = False):
        super().__init__()
        self.cache_quantized_input: bool = cache_quantized_input

    @abc.abstractmethod
    def _activation_forward_impl(self, *args, **kwargs) -> torch.Tensor:
        """Forward implementation

        Implementation from transformer_engine.pytorch.cpp_extensions.

        """

    @abc.abstractmethod
    def _activation_backward_impl(self, *args, **kwargs) -> torch.Tensor:
        """Backward implementation

        Implementation from transformer_engine_torch.

        """

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
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
        x = input_
        if isinstance(x, QuantizedTensor):
            x = x.dequantize()
        if x.device.type != "cuda":
            x = x.cuda()
        if x.dtype != dtype:
            x = x.to(dtype=dtype)
        if not x.is_contiguous():
            x = x.contiguous()

        # Check if quantized compute is enabled
        quantized_compute_enabled = FP8GlobalStateManager.is_fp8_enabled()
        quantizer = None
        if (
            quantized_compute_enabled
            and next_op is not None
            and next_op.num_quantizers("forward") > 0
        ):
            quantizer = next_op.get_quantizer("forward", 0)

        # Launch kernel
        y = self._activation_forward_impl(
            reshape(x, (-1, x.size(-1))),
            quantizer,
        )

        # Check output tensor
        if y.dim() != x.dim():
            y = y.reshape(list(x.shape[:-1]) + [-1])

        # Quantize input to FP8 before caching if needed
        if self.cache_quantized_input:
            input_quantizer = Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, x.device)
            input_quantizer.set_usage(rowwise=True, columnwise=False)
            x = input_quantizer(x)

        # Save state for backward pass
        ctx.save_for_backward(x.detach())
        ctx.quantized_compute_enabled = quantized_compute_enabled
        ctx.dtype = dtype
        ctx.prev_op = prev_op

        return y

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:

        # Saved tensors from forward pass
        (x,) = ctx.saved_tensors

        # Check input tensor
        if isinstance(x, QuantizedTensor):
            x = x.dequantize(dtype=ctx.dtype)
        elif x.dtype != ctx.dtype:
            x = x.to(dtype=ctx.dtype)
        if not x.is_contiguous():
            x = x.contiguous()

        # Check grad output tensor
        dy = grad_output
        if isinstance(dy, QuantizedTensor):
            dy = dy.dequantize(dtype=ctx.dtype)
        if not devices_match(dy.device, x.device) or dy.dtype != x.dtype:
            dy = dy.to(device=x.device, dtype=x.dtype)
        if not dy.is_contiguous():
            dy = dy.contiguous()

        # Check if quantized compute is enabled
        quantizer = None
        if (
            ctx.quantized_compute_enabled
            and ctx.prev_op is not None
            and ctx.prev_op.num_quantizers("backward") > 0
        ):
            quantizer = ctx.prev_op.get_quantizer("backward", 0)

        # Launch kernel
        dx = self._activation_backward_impl(
            reshape(dy, (-1, dy.size(-1))),
            reshape(x, (-1, x.size(-1))),
            quantizer,
        )

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

    def _activation_forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.gelu(*args, **kwargs)

    def _activation_backward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.dgelu(*args, **kwargs)


class ReLU(_ActivationOperation):
    r"""Rectified linear unit

    .. math::

       \text{ReLU}(x) = \max(x,0)

    """

    def _activation_forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.relu(*args, **kwargs)

    def _activation_backward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.drelu(*args, **kwargs)


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

    def _activation_forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.geglu(*args, **kwargs)

    def _activation_backward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.dgeglu(*args, **kwargs)


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

    def _activation_forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.reglu(*args, **kwargs)

    def _activation_backward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.dreglu(*args, **kwargs)


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

    def _activation_forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.swiglu(*args, **kwargs)

    def _activation_backward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.dswiglu(*args, **kwargs)

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operations for activation functions."""

from __future__ import annotations
import abc
from typing import Optional

import torch

import transformer_engine_torch as tex
from ...tensor.float8_tensor import Float8CurrentScalingQuantizer, Quantizer
from ...utils import clear_tensor_data
from ..op import BasicOperation, OperationContext
from .._common import maybe_dequantize

__all__ = [
    "GELU",
    "GEGLU",
    "QGELU",
    "QGEGLU",
    "ReLU",
    "ReGLU",
    "SReLU",
    "SReGLU",
    "SiLU",
    "SwiGLU",
]


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

        # Launch kernel
        y = self._activation_forward_impl(x, next_op_input_quantizer)

        # Quantize input to FP8 before caching if needed
        if self.cache_quantized_input:
            input_quantizer = Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, x.device)
            input_quantizer.set_usage(rowwise=True, columnwise=False)
            x = input_quantizer(x)

        # Save state for backward pass
        if ctx.requires_grad:
            ctx.save_for_backward(x)
            ctx.dtype = dtype
            ctx.prev_op_grad_output_quantizer = prev_op_grad_output_quantizer

        return y

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:

        # Saved tensors from forward pass
        (x,) = ctx.saved_tensors

        # Check input tensor
        x = maybe_dequantize(x.contiguous(), ctx.dtype)

        # Check grad output tensor
        dy = maybe_dequantize(grad_output.contiguous(), x.dtype)

        # Launch kernel
        dx = self._activation_backward_impl(dy, x, ctx.prev_op_grad_output_quantizer)

        # Clear input tensor if possible
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


class GEGLU(_ActivationOperation):
    r"""Gaussian Error Gated Linear Unit

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


class QGELU(_ActivationOperation):
    r"""Quick Gaussian Error Linear Unit

    Quick GELU from `HuggingFace<https://github.com/huggingface/transformers/blob/3e93dd295b5343557a83bc07b0b2ea64c926f9b4/src/transformers/activations.py#L90>`__
    and `paper<https://github.com/hendrycks/GELUs>`__.

    .. math::

       \text{QGELU}(x) \approx x * \sigma(1.702 * x)

    """

    def _activation_forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.qgelu(*args, **kwargs)

    def _activation_backward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.dqgelu(*args, **kwargs)


class QGEGLU(_ActivationOperation):
    r"""Quick Gaussian Error Gated Linear Unit

    The input tensor is split into chunks :math:`a` and :math:`b`
    along the last dimension and the following is computed:

    .. math::

       \text{QGEGLU}(a,b) = \text{QGELU}(a) * b

    where

    .. math::

       \text{QGELU}(x) \approx x * \sigma(1.702 * x)

    .. warning::

       Transformer Engine's gated activations and PyTorch's GLU
       activation follow opposite conventions for :math:`a` and
       :math:`b`. Transformer Engine applies the gating function to
       the first half of the input tensor, while PyTorch applies it to
       the second half.

    """

    def _activation_forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.qgeglu(*args, **kwargs)

    def _activation_backward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.dqgeglu(*args, **kwargs)


class ReLU(_ActivationOperation):
    r"""Rectified Linear Unit

    .. math::

       \text{ReLU}(x) = \max(x,0)

    """

    def _activation_forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.relu(*args, **kwargs)

    def _activation_backward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.drelu(*args, **kwargs)


class ReGLU(_ActivationOperation):
    r"""Rectified Gated Linear Unit

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


class SReLU(_ActivationOperation):
    r"""Squared Rectified Linear Unit

    .. math::

       \text{SReLU}(x) = \max(x^2,0)

    See `Primer: Searching for Efficient Transformers for Language Modeling<https://arxiv.org/abs/2109.08668v2>`__.

    """

    def _activation_forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.srelu(*args, **kwargs)

    def _activation_backward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.dsrelu(*args, **kwargs)


class SReGLU(_ActivationOperation):
    r"""Squared Rectified Gated Linear Unit

    The input tensor is split into chunks :math:`a` and :math:`b`
    along the last dimension and the following is computed:

    .. math::

       \text{SReGLU}(a,b) = \max(a^2,0) * b

    .. warning::

       Transformer Engine's gated activations and PyTorch's GLU
       activation follow opposite conventions for :math:`a` and
       :math:`b`. Transformer Engine applies the gating function to
       the first half of the input tensor, while PyTorch applies it to
       the second half.

    """

    def _activation_forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.sreglu(*args, **kwargs)

    def _activation_backward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.dsreglu(*args, **kwargs)


class SiLU(_ActivationOperation):
    r"""Sigmoid Linear Unit

    .. math::

       \text{SiLU}(x) = x \sigma(x) = \frac{x}{1+\exp(-x)}

    """

    def _activation_forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.silu(*args, **kwargs)

    def _activation_backward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex.dsilu(*args, **kwargs)


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

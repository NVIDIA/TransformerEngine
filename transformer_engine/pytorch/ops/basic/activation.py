# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operations for activation functions."""

from __future__ import annotations
import abc
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
    fp8_dswiglu_cast_transpose_fused,
)
from ...fp8 import FP8GlobalStateManager, get_fp8_te_dtype
from ...tensor import Float8Tensor, QuantizedTensor
from ...utils import clear_tensor_data, devices_match
from ..op import BasicOperation, OperationContext


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

    """

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

        # Check if FP8 is enabled
        fp8_enabled = FP8GlobalStateManager.is_fp8_enabled()
        with_fp8_output = False
        output_fp8_meta = None
        output_dtype = TE_DType[dtype]
        output_fp8_scale_inv = None
        if fp8_enabled and next_op is not None and next_op.num_fp8_scales("input") > 0:
            with_fp8_output = True
            fp8_meta = next_op.get_fp8_meta("input")
            fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(forward=True)
            output_fp8_meta = fp8_meta[fp8_meta_key]
            output_dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
            output_fp8_scale_inv = torch.empty([1], dtype=torch.float32, device=x.device)

        # Launch kernel
        y = self._activation_forward_impl(
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
                dtype=dtype,
            )

        # Save state for backward pass
        ctx.save_for_backward(x)
        ctx.fp8_enabled = fp8_enabled
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
        if isinstance(dy, QuantizedTensor):
            dy = dy.dequantize()
        if not devices_match(dy.device, x.device) or dy.dtype != x.dtype:
            dy = dy.to(device=x.device, dtype=x.dtype)
        if not dy.is_contiguous():
            dy = dy.contiguous()

        # Launch kernel
        dx = self._activation_backward_impl(dy, x, TE_DType[x.dtype])

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
        return tex_gelu(*args, **kwargs)

    def _activation_backward_impl(self, *args, **kwargs) -> torch.Tensor:
        return transformer_engine_torch.dgelu(*args, **kwargs)


class ReLU(_ActivationOperation):
    r"""Rectified linear unit

    .. math::

       \text{ReLU}(x) = \max(x,0)

    """

    def _activation_forward_impl(self, *args, **kwargs) -> torch.Tensor:
        return tex_relu(*args, **kwargs)

    def _activation_backward_impl(self, *args, **kwargs) -> torch.Tensor:
        return transformer_engine_torch.drelu(*args, **kwargs)


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
        return tex_geglu(*args, **kwargs)

    def _activation_backward_impl(self, *args, **kwargs) -> torch.Tensor:
        return transformer_engine_torch.dgeglu(*args, **kwargs)


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
        return tex_reglu(*args, **kwargs)

    def _activation_backward_impl(self, *args, **kwargs) -> torch.Tensor:
        return transformer_engine_torch.dreglu(*args, **kwargs)


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
        return tex_swiglu(*args, **kwargs)

    def _activation_backward_impl(self, *args, **kwargs) -> torch.Tensor:
        return transformer_engine_torch.dswiglu(*args, **kwargs)

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:

        # Saved tensors from forward pass
        (x,) = ctx.saved_tensors

        # Tensor attributes
        dtype = x.dtype
        device = x.device

        # Check grad output tensor
        dy = grad_output
        if isinstance(dy, QuantizedTensor):
            dy = dy.dequantize()
        if not devices_match(dy.device, device) or dy.dtype != dtype:
            dy = dy.to(device=device, dtype=dtype)
        if not dy.is_contiguous():
            dy = dy.contiguous()

        # Check if FP8 is enabled
        with_fp8_grad_input = False
        grad_input_fp8_meta = None
        grad_input_dtype = TE_DType[dtype]
        grad_input_fp8_scale_inv = None
        if (
            ctx.fp8_enabled
            and ctx.prev_op is not None
            and ctx.prev_op.num_fp8_scales("grad_output") > 0
        ):
            with_fp8_grad_input = True
            fp8_meta = ctx.prev_op.get_fp8_meta("grad_output")
            fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(forward=False)
            grad_input_fp8_meta = fp8_meta[fp8_meta_key]
            grad_input_dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=False)
            grad_input_fp8_scale_inv = torch.empty([1], dtype=torch.float32, device=device)

        # Launch kernel
        if with_fp8_grad_input:
            # Fused with FP8 cast-transpose
            input_dims = x.size()
            flat_input_dims = [x.numel() // input_dims[-1], input_dims[-1]]
            flat_output_dims = [flat_input_dims[0], flat_input_dims[1] // 2]
            dx = torch.empty(input_dims, dtype=torch.uint8, device=device)
            dx_t = torch.empty(
                (flat_input_dims[1], flat_input_dims[0]),
                dtype=torch.uint8,
                device=device,
            )
            fp8_dswiglu_cast_transpose_fused(
                dy.reshape(flat_output_dims),
                x.reshape(flat_input_dims),
                grad_input=dx.reshape(flat_input_dims),
                grad_input_transpose=dx_t,
                otype=grad_input_dtype,
                fp8_meta=grad_input_fp8_meta,
                fp8_meta_index=0,
                scale_inv=grad_input_fp8_scale_inv,
            )
            dx = Float8Tensor(
                data=dx,
                fp8_meta=grad_input_fp8_meta,
                fp8_meta_forward=True,
                fp8_meta_index=0,
                fp8_dtype=grad_input_dtype,
                fp8_scale_inv=grad_input_fp8_scale_inv,
                dtype=dtype,
            )
            dx._transpose = dx_t
            dx._transpose_invalid = False
        else:
            # Standard impl
            dx = self._activation_backward_impl(dy, x, TE_DType[dtype])
            if dx.size() != x.size():
                dx = dx.reshape(x.size())

        # Note: This fails if op is preceeded by an identity op like Quantize(forward=False)
        # # Clear input tensor if possible
        # if ctx.prev_op is not None:
        #     clear_tensor_data(x)

        return dx, ()

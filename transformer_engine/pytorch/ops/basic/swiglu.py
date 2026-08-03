# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for SwiGLU and variants."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...constants import DType
from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...tensor import Float8CurrentScalingQuantizer, Quantizer
from ...tensor.storage.grouped_tensor_storage import GroupedTensorStorage
from ...utils import clear_tensor_data
from ..op import BasicOperation, OperationContext
from .._common import maybe_dequantize

__all__ = ["SwiGLU", "ClampedSwiGLU", "ScaledSwiGLU", "ScaledClampedQGeGLU"]


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
                DType.kFloat8E4M3,
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
    glu_linear_offset : float
        Offset added to the linear (gate) component after clamping.
        Set to ``0.0`` to disable the offset.
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
        glu_linear_offset: float = 1.0,
        cache_quantized_input: bool = False,
        glu_interleave_size: Optional[int] = None,
    ):
        super().__init__()
        self.limit: float = limit
        self.alpha: float = alpha
        self.glu_linear_offset: float = glu_linear_offset
        self.cache_quantized_input: bool = cache_quantized_input
        self.glu_interleave_size: Optional[int] = glu_interleave_size

    def _tex_clamped_swiglu_forward(
        self,
        swiglu_in: torch.Tensor,
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        """Call :func:`tex.clamped_swiglu` with this op's ``limit`` / ``alpha`` / ``glu_linear_offset``."""
        return tex.clamped_swiglu(
            swiglu_in,
            next_op_input_quantizer,
            self.limit,
            self.alpha,
            self.glu_linear_offset,
        )

    def _tex_clamped_dswiglu(
        self,
        dy: torch.Tensor,
        swiglu_in: torch.Tensor,
        quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        """Call :func:`tex.clamped_dswiglu` with this op's ``limit`` / ``alpha`` / ``glu_linear_offset``."""
        return tex.clamped_dswiglu(
            dy,
            swiglu_in,
            quantizer,
            self.limit,
            self.alpha,
            self.glu_linear_offset,
        )

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

        # Launch kernel
        out = self._tex_clamped_swiglu_forward(swiglu_in, next_op_input_quantizer)

        # Quantize input to FP8 before caching if needed
        if self.cache_quantized_input:
            input_quantizer = Float8CurrentScalingQuantizer(DType.kFloat8E4M3, x.device)
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
        grad_swiglu_in = self._tex_clamped_dswiglu(dy, swiglu_in, quantizer)

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


class _ScaledGLU(BasicOperation):
    """SwiGLU-family activation with per-row scales (fused grouped MLP middle op)."""

    num_extra_inputs: int = 1

    def __init__(
        self,
        glu_interleave_size: Optional[int] = None,
        *,
        activation_recompute_in_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.glu_interleave_size: Optional[int] = glu_interleave_size
        self.activation_recompute_in_mlp: bool = activation_recompute_in_mlp

    def _scaled_glu_forward(
        self,
        input_: torch.Tensor,
        scales: torch.Tensor,
        quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        raise NotImplementedError

    def _grouped_scaled_glu_forward(
        self,
        input_: torch.Tensor,
        scales: torch.Tensor,
        quantizer: Optional[Quantizer],
        grouped_input: GroupedTensorStorage,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _scaled_glu_backward(
        self,
        grad_output: torch.Tensor,
        input_: torch.Tensor,
        scales: torch.Tensor,
        quantizer: Optional[Quantizer],
        *,
        compute_scale_grad: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError

    def _grouped_scaled_glu_backward(
        self,
        grad_output: torch.Tensor,
        input_: torch.Tensor,
        scales: torch.Tensor,
        quantizer: Optional[Quantizer],
        *,
        num_groups: int,
        first_dims: torch.Tensor,
        tensor_offsets: Optional[torch.Tensor],
        compute_scale_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError

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
        if self.activation_recompute_in_mlp:
            raise RuntimeError(
                f"{self.__class__.__name__}(activation_recompute_in_mlp=True) requires the "
                "fused grouped MLP path."
            )

        extra_input = basic_op_extra_inputs[0][0]

        # Determine compute dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        elif isinstance(input_, torch.Tensor):
            dtype = input_.dtype
        else:
            dtype = extra_input.dtype

        # Make sure inputs are in correct dtype
        grouped_input = input_ if isinstance(input_, GroupedTensorStorage) else None
        input_ = maybe_dequantize(input_, dtype)
        if isinstance(input_, GroupedTensorStorage):
            input_ = input_.rowwise_data.reshape(input_.logical_shape)
        scales = maybe_dequantize(extra_input, dtype)
        if grouped_input is None:
            out = self._scaled_glu_forward(input_, scales, next_op_input_quantizer)
        else:
            out = self._grouped_scaled_glu_forward(
                input_, scales, next_op_input_quantizer, grouped_input
            )

        # Save state for backward pass
        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(input_)
            ctx.input_requires_grad = True
            ctx.extra_input_requires_grad = extra_input.requires_grad
            ctx.dtype = dtype
            ctx.save_for_backward(
                grouped_input if grouped_input is not None else input_,
                scales if ctx.input_requires_grad or ctx.extra_input_requires_grad else None,
            )
            ctx.prev_op_grad_output_quantizer = prev_op_grad_output_quantizer

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
        if self.activation_recompute_in_mlp:
            raise RuntimeError(
                f"{self.__class__.__name__}(activation_recompute_in_mlp=True) requires the "
                "fused grouped MLP path."
            )

        ctx = basic_op_ctxs[0]
        input_, scales = ctx.saved_tensors
        grouped_input = input_ if isinstance(input_, GroupedTensorStorage) else None
        first_dims = grouped_input.first_dims if grouped_input is not None else None
        tensor_offsets = grouped_input.tensor_offsets if grouped_input is not None else None
        input_ = maybe_dequantize(input_, ctx.dtype)
        if isinstance(input_, GroupedTensorStorage):
            input_ = input_.rowwise_data.reshape(input_.logical_shape)
        if scales is not None:
            scales = maybe_dequantize(scales, ctx.dtype)
        grad_output = maybe_dequantize(grad_output, ctx.dtype)
        if isinstance(grad_output, GroupedTensorStorage):
            grad_output = grad_output.rowwise_data.reshape(grad_output.logical_shape)

        if grouped_input is None:
            grad_input, grad_extra_input = self._scaled_glu_backward(
                grad_output,
                input_,
                scales,
                ctx.prev_op_grad_output_quantizer,
                compute_scale_grad=ctx.extra_input_requires_grad,
            )
        else:
            grad_input, dense_grad_input, grad_extra_input = self._grouped_scaled_glu_backward(
                grad_output,
                input_,
                scales,
                ctx.prev_op_grad_output_quantizer,
                num_groups=int(first_dims.numel()),
                first_dims=first_dims,
                tensor_offsets=tensor_offsets,
                compute_scale_grad=ctx.extra_input_requires_grad,
            )
            # Preserve the pre-quantize result for the preceding
            # GroupedLinear's dbias/dscale reduction. ``grad_input`` remains
            # quantized for its dgrad and wgrad GEMMs.
            grad_input._dense_for_dbias = dense_grad_input
        if not ctx.input_requires_grad:
            grad_input = None

        # Clear input tensor if possible
        clear_tensor_data(ctx.saved_tensors[0])  # input_

        return grad_input, [()], [(grad_extra_input,)]


class ScaledSwiGLU(_ScaledGLU):
    r"""SwiGLU with post-scaling (matches cuDNN grouped GEMM ``act_func="swiglu"``).

    If the GLU output has shape ``(d_1, ..., d_n)``, it is multiplied
    with an extra input tensor of shape ``(d_1, ..., d_{n-1})``.

    Parameters
    ----------
    glu_interleave_size : int, optional
        When set, the GLU activations will use an experimental block
        interleaved format. See the corresponding option in the SwiGLU
        operation for more details.
    activation_recompute_in_mlp : bool, default = ``False``
        Enable fused grouped MLP kernels to recompute activation outputs
        during backward when supported instead of saving them.

    """

    def _scaled_glu_forward(
        self,
        input_: torch.Tensor,
        scales: torch.Tensor,
        quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        return tex.scaled_swiglu(
            input_,
            scales,
            quantizer,
            int(self.glu_interleave_size or 0),
        )

    def _grouped_scaled_glu_forward(
        self,
        input_: torch.Tensor,
        scales: torch.Tensor,
        quantizer: Optional[Quantizer],
        grouped_input: GroupedTensorStorage,
    ) -> torch.Tensor:
        return tex.grouped_scaled_swiglu(
            input_,
            scales.reshape(-1),
            quantizer,
            grouped_input.num_tensors,
            grouped_input.first_dims,
            None,
            int(self.glu_interleave_size or 0),
        )

    def _scaled_glu_backward(
        self,
        grad_output: torch.Tensor,
        input_: torch.Tensor,
        scales: torch.Tensor,
        quantizer: Optional[Quantizer],
        *,
        compute_scale_grad: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return tex.scaled_dswiglu(
            grad_output,
            input_,
            scales,
            quantizer,
            int(self.glu_interleave_size or 0),
            compute_scale_grad,
        )

    def _grouped_scaled_glu_backward(
        self,
        grad_output: torch.Tensor,
        input_: torch.Tensor,
        scales: torch.Tensor,
        quantizer: Optional[Quantizer],
        *,
        num_groups: int,
        first_dims: torch.Tensor,
        tensor_offsets: Optional[torch.Tensor],
        compute_scale_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return tex.grouped_scaled_dswiglu(
            grad_output,
            input_,
            scales.reshape(-1),
            quantizer,
            num_groups,
            first_dims,
            tensor_offsets,
            int(self.glu_interleave_size or 0),
            compute_scale_grad,
        )


class ScaledClampedQGeGLU(_ScaledGLU):
    r"""Clamped QGeGLU with post-scaling
    (matches cuDNN grouped GEMM ``act_func="geglu"``).

    Same layout and scaling contract as :class:`ScaledSwiGLU`, but the GLU
    uses :class:`ClampedSwiGLU` numerics (default ``limit`` / ``alpha`` match
    cuDNN).

    Parameters
    ----------
    glu_interleave_size : int, optional
        When set, the GLU activations will use an experimental block
        interleaved format. See :class:`ClampedSwiGLU`.
    activation_recompute_in_mlp : bool, default = ``False``
        Enable fused grouped MLP kernels to recompute activation outputs
        during backward when supported instead of saving them.
    limit : float, default ``7.0``
        Clamp limit (see :class:`ClampedSwiGLU`).
    alpha : float, default ``1.702``
        Sigmoid scale (see :class:`ClampedSwiGLU`).
    glu_linear_offset : float, default ``1.0``
        Offset added to the linear component after clamping
        (see :class:`ClampedSwiGLU`).

    """

    def __init__(
        self,
        glu_interleave_size: Optional[int] = None,
        *,
        activation_recompute_in_mlp: bool = False,
        limit: float = 7.0,
        alpha: float = 1.702,
        glu_linear_offset: float = 1.0,
    ) -> None:
        super().__init__(
            glu_interleave_size,
            activation_recompute_in_mlp=activation_recompute_in_mlp,
        )
        self._clamped: ClampedSwiGLU = ClampedSwiGLU(
            limit=limit,
            alpha=alpha,
            glu_linear_offset=glu_linear_offset,
        )

    def _scaled_glu_forward(
        self,
        input_: torch.Tensor,
        scales: torch.Tensor,
        quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        clamped = self._clamped
        return tex.scaled_clamped_swiglu(
            input_,
            scales,
            quantizer,
            clamped.limit,
            clamped.alpha,
            clamped.glu_linear_offset,
            int(self.glu_interleave_size or 0),
        )

    def _grouped_scaled_glu_forward(
        self,
        input_: torch.Tensor,
        scales: torch.Tensor,
        quantizer: Optional[Quantizer],
        grouped_input: GroupedTensorStorage,
    ) -> torch.Tensor:
        clamped = self._clamped
        return tex.grouped_scaled_clamped_swiglu(
            input_,
            scales.reshape(-1),
            quantizer,
            grouped_input.num_tensors,
            grouped_input.first_dims,
            None,
            clamped.limit,
            clamped.alpha,
            clamped.glu_linear_offset,
            int(self.glu_interleave_size or 0),
        )

    def _scaled_glu_backward(
        self,
        grad_output: torch.Tensor,
        input_: torch.Tensor,
        scales: torch.Tensor,
        quantizer: Optional[Quantizer],
        *,
        compute_scale_grad: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        clamped = self._clamped
        return tex.scaled_clamped_dswiglu(
            grad_output,
            input_,
            scales,
            quantizer,
            clamped.limit,
            clamped.alpha,
            clamped.glu_linear_offset,
            int(self.glu_interleave_size or 0),
            compute_scale_grad,
        )

    def _grouped_scaled_glu_backward(
        self,
        grad_output: torch.Tensor,
        input_: torch.Tensor,
        scales: torch.Tensor,
        quantizer: Optional[Quantizer],
        *,
        num_groups: int,
        first_dims: torch.Tensor,
        tensor_offsets: Optional[torch.Tensor],
        compute_scale_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        clamped = self._clamped
        return tex.grouped_scaled_clamped_dswiglu(
            grad_output,
            input_,
            scales.reshape(-1),
            quantizer,
            num_groups,
            first_dims,
            tensor_offsets,
            clamped.limit,
            clamped.alpha,
            clamped.glu_linear_offset,
            int(self.glu_interleave_size or 0),
            compute_scale_grad,
        )

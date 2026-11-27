# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusable operation for Layer Normalization."""

from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
import math
import os
from typing import Optional

import torch

from transformer_engine_torch import layernorm_bwd, layernorm_fwd
from ...constants import TE_DType
from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...dynamo import TensorOrQuantized, TensorSpec
from ...export import is_in_onnx_export_mode
from ...tensor import Quantizer
from ...utils import (
    canonicalize_device,
    canonicalize_dtype,
    clear_tensor_data,
    devices_match,
)
from ..op import BasicOperation
from .._common import (
    get_fused_normalization_quantizer,
    is_quantized_tensor,
    maybe_autocast_dtype,
    maybe_dequantize,
)


@dataclass(slots=True)
class LayerNormFwdArgs:
    """LayerNorm inputs, parameters and quantization configuration."""

    input_: TensorOrQuantized
    weight: torch.Tensor
    bias: torch.Tensor
    eps: float
    zero_centered_gamma: bool
    sm_margin: int
    dtype: torch.dtype
    output_quantizer: Optional[Quantizer]
    requires_grad: bool


@dataclass(slots=True)
class LayerNormBwdArgs:
    """LayerNorm gradient, saved tensors and parameters."""

    grad_output: TensorOrQuantized
    input_: TensorOrQuantized
    means: torch.Tensor
    rstdevs: torch.Tensor
    weight: torch.Tensor
    zero_centered_gamma: bool
    sm_margin: int
    dtype: torch.dtype


class LayerNorm(BasicOperation):
    r"""Layer Normalization

    Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__ .

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \varepsilon}} * \gamma + \beta

    :math:`\gamma` and :math:`\beta` are learnable affine transform
    parameters that match the inner-most dimensions of the input
    tensor.

    Parameters
    ----------
    normalized_shape : int or iterable of int
        Inner dimensions of input tensor
    eps : float, default = 1e-5
        A value added to the denominator of layer normalization for
        numerical stability
    device : torch.device, default = default CUDA device
        Tensor device
    dtype : torch.dtype, default = default dtype
        Tensor datatype
    zero_centered_gamma : bool, default = False
        If ``True``, the :math:`\gamma` parameter is initialized to
        zero and the calculation changes to

            .. math::
                y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \varepsilon}} * (1 + \gamma) + \beta

    sm_margin : int or dict, default = 0
        Number of SMs to exclude when launching CUDA kernels. This
        helps overlap with other kernels, e.g. communication kernels.
        For more fine-grained control, provide a dict with the SM
        margin at each compute stage ("forward", "backward",
        "inference").

    """

    def __init__(
        self,
        normalized_shape: Iterable[int] | int,
        *,
        eps: float = 1e-5,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        zero_centered_gamma: bool = False,
        sm_margin: int | dict[str, int] = 0,
    ) -> None:
        super().__init__()
        self.eps: float = eps
        self.zero_centered_gamma: bool = zero_centered_gamma

        # Parameter shape
        if not isinstance(normalized_shape, Iterable):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)

        # Parameter device
        defer_param_init = False
        device = canonicalize_device(device)
        if device.type == "meta":
            defer_param_init = True

        # Initialize parameters if needed
        dtype = canonicalize_dtype(dtype)
        weight = torch.empty(
            normalized_shape,
            device=device,
            dtype=dtype,
        )
        bias = torch.empty(
            normalized_shape,
            device=device,
            dtype=dtype,
        )
        weight = torch.nn.Parameter(weight)
        bias = torch.nn.Parameter(bias)
        self.weight: torch.nn.Parameter
        self.bias: torch.nn.Parameter
        self.register_parameter("weight", weight)
        self.register_parameter("bias", bias)
        if not defer_param_init:
            self.reset_parameters()

        # Number of SMs to exclude when launching CUDA kernels
        self._sm_margins: dict[str, int]
        if isinstance(sm_margin, dict):

            def getenv(name: str) -> int:
                return int(os.getenv(name, "0"))

            self._sm_margins = {
                "forward": sm_margin.get("forward", getenv("NVTE_FWD_LAYERNORM_SM_MARGIN")),
                "backward": sm_margin.get("backward", getenv("NVTE_BWD_LAYERNORM_SM_MARGIN")),
                "inference": sm_margin.get("inference", getenv("NVTE_INF_LAYERNORM_SM_MARGIN")),
            }
        else:

            def getenv(name: str) -> int:
                return int(os.getenv(name, str(sm_margin)))

            self._sm_margins = {
                "forward": getenv("NVTE_FWD_LAYERNORM_SM_MARGIN"),
                "backward": getenv("NVTE_BWD_LAYERNORM_SM_MARGIN"),
                "inference": getenv("NVTE_INF_LAYERNORM_SM_MARGIN"),
            }

    def reset_parameters(self) -> None:
        """Initialize parameter buffers and values"""

        # Parameter device
        weight = self.weight
        bias = self.bias
        device = weight.device
        if device.type == "meta":
            device = canonicalize_device(None)

        # Initialize param buffers
        if not devices_match(weight.device, device):
            weight = torch.empty_like(weight, device=device)
        if not devices_match(bias.device, device):
            bias = torch.empty_like(bias, device=device)

        # Initialize values
        if self.zero_centered_gamma:
            torch.nn.init.zeros_(weight)
        else:
            torch.nn.init.ones_(weight)
        torch.nn.init.zeros_(bias)

        # Save updated parameter
        if not isinstance(weight, torch.nn.Parameter):
            weight = torch.nn.Parameter(weight)
        if not isinstance(bias, torch.nn.Parameter):
            bias = torch.nn.Parameter(bias)
        self.weight = weight
        self.bias = bias

    fwd_args_type = LayerNormFwdArgs
    bwd_args_type = LayerNormBwdArgs

    def compile_unsupported_reason(self, mode: str) -> Optional[str]:
        if is_cpu_offload_enabled():
            return "CPU offloading"
        return super().compile_unsupported_reason(mode)

    def fuser_forward(self, basic_op_ctxs, input_, **kwargs):
        if is_in_onnx_export_mode():
            return self.op_onnx_forward(input_), [()]
        return super().fuser_forward(basic_op_ctxs, input_, **kwargs)

    def pack_forward_args(
        self,
        basic_op_ctxs,
        input_,
        *,
        next_op_input_quantizer,
        basic_op_kwargs,
        **_unused,
    ) -> LayerNormFwdArgs:
        if basic_op_kwargs[0]:
            raise ValueError("LayerNorm forward does not expect keyword arguments")
        if (
            torch.compiler.is_compiling()
            and not is_quantized_tensor(input_)
            and not input_.is_contiguous()
        ):
            raise RuntimeError("LayerNorm forward requires contiguous inputs under torch.compile")
        weight_dims = tuple(self.weight.size())
        input_dims = tuple(input_.size())
        if len(input_dims) < len(weight_dims) or input_dims[-len(weight_dims) :] != weight_dims:
            raise ValueError(
                f"Input tensor (shape={input_dims}) "
                f"and weight tensor (shape={weight_dims}) are not compatible"
            )
        requires_grad = basic_op_ctxs[0].requires_grad
        return LayerNormFwdArgs(
            input_,
            self.weight,
            self.bias,
            self.eps,
            self.zero_centered_gamma,
            self._sm_margins["forward" if requires_grad else "inference"],
            maybe_autocast_dtype(default_dtype=self.weight.dtype),
            get_fused_normalization_quantizer(next_op_input_quantizer),
            requires_grad,
        )

    @classmethod
    def forward_compute(
        cls, args: LayerNormFwdArgs, *, in_custom_op: bool = False
    ):  # pylint: disable=unused-argument
        weight_dims = args.weight.size()
        inner_dim = math.prod(weight_dims)
        x = maybe_dequantize(args.input_, args.dtype)
        w = maybe_dequantize(args.weight, args.dtype).view((inner_dim,))
        b = maybe_dequantize(args.bias, args.dtype).view((inner_dim,))
        x_kernel = (
            x if len(weight_dims) == 1 else x.view((*x.shape[: -len(weight_dims)], inner_dim))
        )
        y, means, rstdevs = layernorm_fwd(
            x_kernel,
            w,
            b,
            args.eps,
            None,
            args.output_quantizer,
            TE_DType[args.dtype],
            args.sm_margin,
            args.zero_centered_gamma,
        )
        out = y
        if len(weight_dims) > 1:
            out = y.view(args.input_.size())
            if is_quantized_tensor(out):
                quantizer = args.output_quantizer
                out.update_usage(
                    rowwise_usage=quantizer.rowwise_usage,
                    columnwise_usage=quantizer.columnwise_usage,
                )
        if not args.requires_grad:
            return out, [()], ()
        return out, [()], (x if x is not args.input_ else None, means, rstdevs)

    @classmethod
    def forward_compute_fake(cls, args: LayerNormFwdArgs):
        x = args.input_
        out = TensorSpec(
            shape=x.shape, dtype=args.dtype, device=x.device, quantizer=args.output_quantizer
        )
        if not args.requires_grad:
            return out, [()], ()
        saved_input = None
        if is_quantized_tensor(x) or x.dtype != args.dtype:
            saved_input = TensorSpec(shape=x.shape, dtype=args.dtype, device=x.device)
        rows = math.prod(x.shape[: len(x.shape) - len(args.weight.shape)])
        stats = TensorSpec(shape=(rows,), dtype=torch.float32, device=x.device)
        return out, [()], (saved_input, stats, stats)

    def forward_setup_context(self, basic_op_ctxs, args, aux) -> None:
        ctx = basic_op_ctxs[0]
        x = args.input_ if aux[0] is None else aux[0]
        means, rstdevs = aux[1:]
        if is_cpu_offload_enabled():
            mark_activation_offload(x, means, rstdevs)
        ctx.save_for_backward(x, means, rstdevs)
        ctx.dtype = args.dtype

    def pack_backward_args(self, basic_op_ctxs, grad_output, **_unused) -> LayerNormBwdArgs:
        ctx = basic_op_ctxs[0]
        x, means, rstdevs = ctx.saved_tensors
        return LayerNormBwdArgs(
            grad_output,
            x,
            means,
            rstdevs,
            self.weight,
            self.zero_centered_gamma,
            self._sm_margins["backward"],
            ctx.dtype,
        )

    @classmethod
    def backward_compute(cls, args: LayerNormBwdArgs, *, in_custom_op: bool = False):
        inner_dim = math.prod(args.weight.size())
        x = maybe_dequantize(args.input_, args.dtype).view((-1, inner_dim))
        dy = maybe_dequantize(args.grad_output, args.dtype).view(x.size())
        w = maybe_dequantize(args.weight, args.dtype).view((inner_dim,))
        dx, dw, db = layernorm_bwd(
            dy, x, args.means, args.rstdevs, w, args.sm_margin, args.zero_centered_gamma
        )
        if not in_custom_op:
            clear_tensor_data(args.input_, args.means, args.rstdevs)
        weight_dims = args.weight.size()
        return (
            dx.view(args.grad_output.size()),
            [(dw.view(weight_dims), db.view(weight_dims))],
            [()],
        )

    @classmethod
    def backward_compute_fake(cls, args: LayerNormBwdArgs):
        dy = args.grad_output
        dx = TensorSpec(shape=dy.shape, dtype=args.dtype, device=dy.device)
        dparam = TensorSpec(shape=args.weight.shape, dtype=args.dtype, device=dy.device)
        return dx, [(dparam, dparam)], [()]

    def op_onnx_forward(
        self,
        input_: torch.Tensor,
    ) -> torch.Tensor:
        """Every operand in this function has a defined ONNX translation."""
        weight = self.weight + 1 if self.zero_centered_gamma else self.weight
        return torch.nn.functional.layer_norm(
            input_, input_.shape[-1:], weight, self.bias, self.eps
        )

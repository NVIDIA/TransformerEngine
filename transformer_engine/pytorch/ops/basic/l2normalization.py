# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusable operation for L2 Normalization."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os

import torch

from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...dynamo import TensorOrQuantized, TensorSpec
from ...jit import (
    l2normalization_fused,
    l2normalization_fwd_fused,
    l2normalization_backward_fused,
    warmup_jit_l2normalization_all_dtypes,
)
from ...utils import clear_tensor_data
from ..op import BasicOperation
from .._common import is_quantized_tensor, maybe_dequantize


@dataclass(slots=True)
class L2NormalizationFwdArgs:
    """L2 normalization input and configuration."""

    input_: TensorOrQuantized
    eps: float
    requires_grad: bool


@dataclass(slots=True)
class L2NormalizationBwdArgs:
    """L2 normalization gradient and saved tensors."""

    grad_output: TensorOrQuantized
    input_: TensorOrQuantized
    rsqrt_norm: torch.Tensor
    eps: float


class L2Normalization(BasicOperation):
    r"""L2 Normalization

    Applies L2 normalization over the last dimension of input tensors.
    This is a parameter-free normalization that scales each vector to unit L2 norm.

    .. math::
        y = \frac{x}{\sqrt{\sum_{i} x_i^2 + \varepsilon}}

    This operation is used e.g. for query-key normalization in attention mechanisms.

    Parameters
    ----------
    eps : float, default = 1e-6
        A value added to the denominator for numerical stability
    seq_length : int, default = None
        sequence length of input samples. Needed for JIT Warmup, a technique where jit fused
        functions are warmed up before training to ensure same kernels are used for forward
        propagation and activation recompute phase.
    micro_batch_size : int, default = None
        batch size per training step. Needed for JIT Warmup, a technique where jit
        fused functions are warmed up before training to ensure same kernels are
        used for forward propagation and activation recompute phase.

    """

    def __init__(
        self,
        *,
        eps: float = 1e-6,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.eps: float = eps

        # JIT warmup for L2Normalization fused operations
        if seq_length and micro_batch_size:
            if torch.cuda.is_available() and bool(int(os.getenv("NVTE_TORCH_COMPILE", "1"))):
                # For L2Normalization, we don't know the hidden size until forward pass,
                # but we can warm up with common sizes. For QK normalization, this will be
                # the attention head dimension (hidden_size_per_attention_head), not the full
                # model hidden dimension. Common head dimensions are 32, 64, 80, 96, 128, 256.
                common_hidden_sizes = [32, 64, 80, 96, 128, 256]
                for hidden_size in common_hidden_sizes:
                    warmup_jit_l2normalization_all_dtypes(hidden_size, seq_length, micro_batch_size)

    fwd_args_type = L2NormalizationFwdArgs
    bwd_args_type = L2NormalizationBwdArgs

    def compile_unsupported_reason(self, mode: str) -> Optional[str]:
        if is_cpu_offload_enabled():
            return "CPU offloading"
        return super().compile_unsupported_reason(mode)

    def pack_forward_args(
        self,
        basic_op_ctxs,
        input_,
        *,
        basic_op_kwargs,
        **unused,  # pylint: disable=unused-argument
    ) -> L2NormalizationFwdArgs:
        if basic_op_kwargs[0]:
            raise ValueError("L2Normalization forward does not expect keyword arguments")
        if (
            torch.compiler.is_compiling()
            and not is_quantized_tensor(input_)
            and not input_.is_contiguous()
        ):
            raise RuntimeError(
                "L2Normalization forward requires contiguous inputs under torch.compile"
            )
        return L2NormalizationFwdArgs(input_, self.eps, basic_op_ctxs[0].requires_grad)

    @classmethod
    def forward_compute(
        cls, args: L2NormalizationFwdArgs, *, in_custom_op: bool = False
    ):  # pylint: disable=unused-argument
        x = maybe_dequantize(args.input_)
        if not args.requires_grad:
            return l2normalization_fused(x, args.eps), [()], ()
        y, rsqrt_norm = l2normalization_fwd_fused(x, args.eps)
        return y, [()], (x if x is not args.input_ else None, rsqrt_norm)

    @classmethod
    def forward_compute_fake(cls, args: L2NormalizationFwdArgs):
        x = args.input_
        y = TensorSpec(shape=x.shape, dtype=x.dtype, device=x.device)
        if not args.requires_grad:
            return y, [()], ()
        saved_input = TensorSpec(shape=x.shape, dtype=x.dtype, device=x.device)
        rsqrt_norm = TensorSpec(shape=(*x.shape[:-1], 1), dtype=torch.float32, device=x.device)
        return y, [()], (saved_input if is_quantized_tensor(x) else None, rsqrt_norm)

    def forward_setup_context(self, basic_op_ctxs, args, aux) -> None:
        ctx = basic_op_ctxs[0]
        x = args.input_ if aux[0] is None else aux[0]
        rsqrt_norm = aux[1]
        if is_cpu_offload_enabled():
            mark_activation_offload(x, rsqrt_norm)
        ctx.save_for_backward(x, rsqrt_norm)

    def pack_backward_args(
        self, basic_op_ctxs, grad_output, **unused  # pylint: disable=unused-argument
    ) -> L2NormalizationBwdArgs:
        x, rsqrt_norm = basic_op_ctxs[0].saved_tensors
        return L2NormalizationBwdArgs(grad_output, x, rsqrt_norm, self.eps)

    @classmethod
    def backward_compute(cls, args: L2NormalizationBwdArgs, *, in_custom_op: bool = False):
        x = maybe_dequantize(args.input_)
        dy = maybe_dequantize(args.grad_output)
        dx = l2normalization_backward_fused(dy, x, args.rsqrt_norm, args.eps)
        if not in_custom_op:
            clear_tensor_data(args.input_, args.rsqrt_norm)
        return dx, [()], [()]

    @classmethod
    def backward_compute_fake(cls, args: L2NormalizationBwdArgs):
        x = args.input_
        return TensorSpec(shape=x.shape, dtype=x.dtype, device=x.device), [()], [()]

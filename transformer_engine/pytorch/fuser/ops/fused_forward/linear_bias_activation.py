# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from collections.abc import Callable, Iterable
import contextlib
import math
from typing import Optional

import torch

from transformer_engine.pytorch.cpp_extensions import fp8_gemm, gemm
from transformer_engine.pytorch.distributed import (
    CudaRNGStatesTracker,
    gather_along_first_dim,
    reduce_scatter_along_first_dim,
)
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.fp8 import (
    FP8GlobalStateManager,
    get_fp8_te_dtype,
)
from transformer_engine.pytorch.fuser.ops.basic import BasicLinear, Bias
from transformer_engine.pytorch.fuser.ops.op import FusedOperation
from transformer_engine.pytorch.module.base import get_workspace
from .._common import (
    canonicalize_device,
    canonicalize_dtype,
    convert_tensor,
    fp8_cast_transpose,
    is_float8_tensor,
    reshape,
)


class ForwardLinearBiasActivation(FusedOperation):
    """Fused GEMM, bias, activation in the forward pass

    Bias and activation are both optional. Row tensor parallelism is
    not supported since that requires communication immediately after
    the GEMM.

    """

    def __init__(
        self,
        *,
        linear: BasicLinear,
        bias: Optional[Bias],
        activation: None,
    ) -> None:

        # Basic operations that comprise this fused operation
        op_idxs = dict(
            linear=0,
            bias=None,
            activation=None,
        )
        ops = [linear]
        if bias is not None:
            op_idxs["bias"] = len(ops)
            ops.append(bias)
        if activation is not None:
            op_idxs["activation"] = len(ops)
            ops.append(activation)

        # Initialize base class
        super().__init__(ops)

        # Index of each basic operations
        self._op_idxs: dict[str, Optional[int]] = op_idxs

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input: torch.Tensor,
        basic_op_prev_ops: list[Optional[BasicOperation]],
        basic_op_next_ops: list[Optional[BasicOperation]],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> torch.Tensor:

        # Get basic operations
        idx = self._op_idxs["linear"]
        linear_op = self.basic_ops[idx]
        linear_op_ctx = basic_op_ctxs[idx]
        linear_op_kwargs = basic_op_kwargs[idx]
        if self._op_idxs["bias"] is None:
            bias_op = None
        else:
            idx = self._op_idxs["bias"]
            bias_op = self.basic_ops[idx]
            if basic_op_kwargs[idx]:
                raise ValueError(
                    "Bias operation forward does not expect keyword arguments"
                )
        if self._op_idxs["activation"] is None:
            activation_op = None
        else:
            raise NotImplementedError("Activations are not yet supported")  ### TODO Implement

        # Tensor dims
        input_dims = input.size()
        weight_dims = linear_op.weight.size()

        # Check if FP8 is enabled
        with_fp8_compute = FP8GlobalStateManager.is_fp8_enabled()
        input_fp8_meta = None
        weight_fp8_meta = None
        output_fp8_meta = None
        grad_output_fp8_meta = None
        grad_input_fp8_meta = None
        if with_fp8_compute:
            input_fp8_meta = linear_op.get_fp8_meta("input")
            weight_fp8_meta = linear_op.get_fp8_meta("param")
            output_fp8_meta = None
            grad_output_fp8_meta = linear_op.get_fp8_meta("grad_output")
            prev_op = basic_op_prev_ops[0]
            if prev_op is not None and prev_op.num_fp8_scales("grad_output") > 0:
                grad_input_fp8_meta = prev_op.get_fp8_meta("grad_output")

        # Check input tensor
        if len(input_dims) == 0 or weight_dims[1] != input_dims[-1]:
            raise ValueError(
                f"Input tensor (shape={tuple(input.size())}) "
                f"and weight tensor (shape={tuple(linear_op.weight.size())}) "
                "are not compatible"
            )
        x_local = reshape(
            input,
            (-1, input_dims[-1]),
            device=linear_op.device,
            dtype=linear_op.dtype,
        )
        if with_fp8_compute and not is_float8_tensor(x_local):
            fp8_dtype = get_fp8_te_dtype(input_fp8_meta["recipe"], fprop_tensor=True)
            with_cast_transpose = linear_op.weight.requires_grad
            if linear_op.tensor_parallel_mode == "column" and linear_op.sequence_parallel:
                with_cast_transpose = False
            if with_cast_transpose:
                x_local = fp8_cast_transpose(
                    x_local,
                    fp8_meta=input_fp8_meta,
                    fp8_meta_forward=True,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                )
            else:
                x_local = Float8Tensor.to_float8(
                    x_local,
                    fp8_meta=input_fp8_meta,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                )
        elif not with_fp8_compute and is_float8_tensor(x_local):
            x_local = x_local.from_float8()
        x = x_local
        x_async = None
        if (
            linear_op.tensor_parallel_mode == "column"
            and linear_op.sequence_parallel
        ):
            x, x_async = gather_along_first_dim(
                x,
                linear_op.tensor_parallel_group,
                async_op=True,
            )

        # Check weight tensor
        ### TODO: Weight caching without FP8 params
        weight = linear_op.weight
        w = convert_tensor(
            weight,
            device=linear_op.device,
            dtype=linear_op.dtype,
            memory_format=torch.contiguous_format,
        )
        if with_fp8_compute and not is_float8_tensor(w):
            fp8_dtype = get_fp8_te_dtype(weight_fp8_meta["recipe"], fprop_tensor=True)
            w = Float8Tensor.to_float8(
                w,
                fp8_meta=weight_fp8_meta,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
            )
        elif not with_fp8_compute and is_float8_tensor(w):
            w = w.from_float8()

        # Check bias tensor
        b = None
        if bias_op is not None:
            b = convert_tensor(
                bias_op.bias,
                device=linear_op.device,
                dtype=linear_op.dtype,
                memory_format=torch.contiguous_format,
            )

        # Perform GEMM
        y = torch.empty(
            (x.size(0), weight_dims[0]),
            dtype=linear_op.dtype,
            device=linear_op.device,
        )
        if x_async is not None:
            x_async.wait()
        if with_fp8_compute:
            fp8_gemm(
                w._data,
                w._scale_inv,
                0,
                w._fp8_dtype,
                x._data,
                x._scale_inv,
                0,
                x._fp8_dtype,
                y.dtype,
                get_workspace(),
                out=y,
                bias=b,
                use_bias=(bias_op is not None),
            )
        else:
            gemm(
                w,
                x,
                y.dtype,
                get_workspace(),
                out=y,
                bias=b,
                use_bias=(bias_op is not None),
            )

        # Save state for backward pass
        linear_op_ctx.save_for_backward(x_local)
        linear_op_ctx.with_fp8_compute = with_fp8_compute
        linear_op_ctx.weight_fp8_meta = weight_fp8_meta
        linear_op_ctx.grad_output_fp8_meta = grad_output_fp8_meta
        linear_op_ctx.grad_input_fp8_meta = grad_input_fp8_meta
        linear_op_ctx.input_dims = input.size()
        linear_op_ctx.input_requires_grad = input.requires_grad
        linear_op_ctx.weight_requires_grad = weight.requires_grad

        # Reshape output tensor
        output_dims = list(input_dims)
        output_dims[0] = -1
        output_dims[-1] = y.size(-1)
        return reshape(y, output_dims)


def fuse_forward_linear_bias_activation(
    ops: list[tuple[FusableOperation, list[int]]],
) -> list[tuple[FusableOperation, list[int]]]:
    """Fuse GEMM, bias, activation in the forward pass

    Parameters
    ----------
    ops: list of tuples
        Forward pass operations and the indices of the corresponding
        basic operations.

    Returns
    -------
    ops: list of tuples
        Updated forward pass operations

    """

    # Scan through ops, fusing if possible
    out = []
    window = []
    while len(ops) >= 2:
        out.extend(window)

        # Check if first op is linear
        window, ops = ops[:1], ops[1:]
        op1, _ = window[0]
        if not isinstance(op1, BasicLinear):
            continue
        if op1.tensor_parallel_mode == "row":
            # Row tensor-parallelism requires communication after the
            # GEMM
            continue
        if op1.dtype not in (torch.float16, torch.bfloat16):
            # cuBLAS only supports fused GEMM+bias+activation with
            # FP16 and BF16 output
            continue

        # Check if second op is bias
        op2, _ = ops[0]
        if not isinstance(op2, Bias):
            continue
        window.extend(ops[:1])
        ops = ops[1:]

        # Replace window with fused op
        op = ForwardLinearBiasActivation(
            linear=window[0][0],
            bias=window[1][0],
            activation=None,
        )
        basic_op_idxs = [basic_op_idxs[0] for _, basic_op_idxs in window]
        window = [(op, basic_op_idxs)]

    # Return list of ops
    out.extend(window)
    out.extend(ops)
    return out

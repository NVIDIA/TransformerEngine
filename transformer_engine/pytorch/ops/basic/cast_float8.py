# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for identity."""

from __future__ import annotations
from typing import Optional

import torch

from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.fp8 import (
    FP8GlobalStateManager,
    get_fp8_te_dtype,
)
from transformer_engine.pytorch.ops.op import (
    BasicOperation,
    OperationContext,
)
from .._common import is_float8_tensor


class CastFloat8(BasicOperation):
    """Cast tensor to FP8

    Uses FP8 recipe from `fp8_autocast` context. When called outside
    of an `fp8_autocast` context, this is an identity operation.

    Parameters
    ----------
    forward: bool, default = `True`
        Perform FP8 cast in forward pass
    backward: bool, default = `False`
        Perform FP8 cast in backward pass

    """

    def __init__(
        self,
        forward: bool = True,
        backward: bool = False,
    ) -> None:
        super().__init__()
        self._cast_forward = forward
        self._cast_backward = backward

    def num_fp8_scales(self, mode: str) -> int:
        if mode == "input" and self._cast_forward:
            return 1
        if mode == "grad_output" and self._cast_backward:
            return 1
        return 0

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
    ) -> torch.Tensor:

        # Check if FP8 is enabled
        fp8_enabled = FP8GlobalStateManager.is_fp8_enabled()
        cast_forward = fp8_enabled and self._cast_forward
        cast_backward = fp8_enabled and self._cast_backward

        # Cast to FP8 if needed
        out = input_
        if cast_forward and not is_float8_tensor(out):
            fp8_meta = self.get_fp8_meta("input")
            fp8_dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
            out = Float8Tensor.to_float8(
                out,
                fp8_meta=fp8_meta,
                fp8_meta_forward=True,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
            )

        ctx.cast_backward = cast_backward
        return out

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        grad_input = grad_output
        if ctx.cast_backward and not is_float8_tensor(grad_input):
            fp8_meta = self.get_fp8_meta("grad_output")
            fp8_dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=False)
            grad_input = Float8Tensor.to_float8(
                grad_input,
                fp8_meta=fp8_meta,
                fp8_meta_forward=False,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
            )
        return grad_input, ()

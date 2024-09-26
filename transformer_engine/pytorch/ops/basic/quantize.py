# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for quantization."""

from __future__ import annotations
from typing import Optional

import torch

from ...fp8 import FP8GlobalStateManager, get_fp8_te_dtype
from ...tensor import Float8Tensor, QuantizedTensor
from ..op import BasicOperation, OperationContext


class Quantize(BasicOperation):
    """Quantize tensor data

    Uses FP8 recipe from `fp8_autocast` context. When called outside
    of an `fp8_autocast` context, this is an identity operation.

    Parameters
    ----------
    forward: bool, default = `True`
        Perform quantization in forward pass
    backward: bool, default = `False`
        Perform quantization in backward pass

    """

    def __init__(
        self,
        forward: bool = True,
        backward: bool = False,
    ) -> None:
        super().__init__()
        self._quantize_forward = forward
        self._quantize_backward = backward

    def num_fp8_scales(self, mode: str) -> int:
        if mode == "input" and self._quantize_forward:
            return 1
        if mode == "grad_output" and self._quantize_backward:
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
        quantize_forward = fp8_enabled and self._quantize_forward
        quantize_backward = fp8_enabled and self._quantize_backward

        # Quantize if needed
        out = input_
        if quantize_forward and not isinstance(out, QuantizedTensor):
            fp8_meta = self.get_fp8_meta("input")
            fp8_dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
            out = Float8Tensor.to_float8(
                out,
                fp8_meta=fp8_meta,
                fp8_meta_forward=True,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
            )

        ctx.quantize_backward = quantize_backward
        return out

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        grad_input = grad_output
        if ctx.quantize_backward and not isinstance(grad_input, QuantizedTensor):
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

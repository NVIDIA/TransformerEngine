# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused operation for forward GEMM + bias + add."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional, Union

import torch

from ...quantized_tensor import QuantizedTensorStorage
from ...quantization import FP8GlobalStateManager
from ...tensor import Quantizer
from ..basic import AddExtraInput, BasicLinear, Bias
from ..op import FusedOperation, FusibleOperation, OperationContext


class ForwardLinearBiasAdd(FusedOperation):
    """Fused forward GEMM + bias + add

    Bias is optional. Row tensor parallelism is not supported since
    that requires communication immediately after the GEMM.

    """

    def __init__(
        self,
        *,
        linear: BasicLinear,
        bias: Optional[Bias],
        add: AddExtraInput,
    ) -> None:

        # Basic operations that comprise this fused operation
        op_idxs = {"linear": 0, "bias": None, "add": None}
        ops = [linear]
        if bias is not None:
            op_idxs["bias"] = len(ops)
            ops.append(bias)
        op_idxs["add"] = len(ops)
        ops.append(add)

        # Initialize base class
        super().__init__(ops)

        # Index of each basic operations
        self._op_idxs: dict[str, Optional[int]] = op_idxs

    def fuser_forward_compute(
        self,
        input_: torch.Tensor,
        *,
        requires_grad: list[bool],
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[
        torch.Tensor,
        Iterable[Iterable[torch.Tensor]],
        list[tuple[Optional[Union[torch.Tensor, QuantizedTensorStorage]], ...]],
    ]:

        # Get basic operations
        linear_idx = self._op_idxs["linear"]
        linear_op = self.basic_ops[linear_idx]
        bias = None
        if self._op_idxs["bias"] is not None:
            bias_idx = self._op_idxs["bias"]
            bias = self.basic_ops[bias_idx].bias
            if basic_op_kwargs[bias_idx]:
                raise ValueError("Bias operation forward does not expect keyword arguments")

        # Check which grads are required
        input_requires_grad = requires_grad[linear_idx]
        weight_requires_grad = requires_grad[linear_idx] and linear_op.weight.requires_grad

        # Quantizers
        input_quantizer = linear_op.get_quantizer("forward", 0)
        weight_quantizer = linear_op.get_quantizer("forward", 1)
        output_quantizer = None
        with_quantized_compute = FP8GlobalStateManager.is_fp8_enabled()
        if with_quantized_compute:
            backward_override = FP8GlobalStateManager.get_fp8_recipe().backward_override
        else:
            backward_override = None

        # Linear forward
        output = basic_op_extra_inputs[self._op_idxs["add"]][0]
        output, x_local, w = BasicLinear._functional_forward(
            input=input_,
            weight=linear_op.weight,
            bias=bias,
            dtype=output.dtype,
            out=output,
            accumulate_into_out=True,
            tensor_parallel_mode=linear_op.tensor_parallel_mode,
            tensor_parallel_group=linear_op.tensor_parallel_group,
            sequence_parallel=linear_op.sequence_parallel,
            with_quantized_compute=with_quantized_compute,
            backward_override=backward_override,
            input_quantizer=input_quantizer,
            weight_quantizer=weight_quantizer,
            output_quantizer=output_quantizer,
            input_requires_grad=input_requires_grad,
            weight_requires_grad=weight_requires_grad,
        )

        # Determine tensors to save for backward pass
        if requires_grad[linear_idx]:
            if backward_override == "high_precision":
                saved_input = input_ if weight_requires_grad else None
                saved_weight = linear_op.weight if input_requires_grad else None
            else:
                saved_input = x_local
                saved_weight = w
            linear_tensors = (saved_input, saved_weight)
        else:
            linear_tensors = (None, None)

        tensors_to_save = [() for _ in range(len(self.basic_ops))]
        tensors_to_save[linear_idx] = linear_tensors

        return output, [() for _ in range(len(self.basic_ops))], tensors_to_save

    def fuser_forward_save_ctx(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        tensors_to_save: list[tuple[Optional[Union[torch.Tensor, QuantizedTensorStorage]], ...]],
        *,
        requires_grad: list[bool],
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> None:
        linear_idx = self._op_idxs["linear"]
        linear_op = self.basic_ops[linear_idx]
        linear_op.op_forward_save_ctx(
            basic_op_ctxs[linear_idx],
            input_,
            tensors_to_save[linear_idx],
            requires_grad=requires_grad[linear_idx],
            prev_op_grad_output_quantizer=prev_op_grad_output_quantizer,
        )
        if self._op_idxs["bias"] is not None:
            bias_idx = self._op_idxs["bias"]
            bias_op = self.basic_ops[bias_idx]
            bias_op.op_forward_save_ctx(
                basic_op_ctxs[bias_idx],
                input_,
                tensors_to_save[bias_idx],
                requires_grad=requires_grad[bias_idx],
                prev_op_grad_output_quantizer=linear_op.get_grad_output_quantizer(),
            )

    @staticmethod
    def fuse_forward_ops(
        ops: list[FusibleOperation],
        **unused,  # pylint: disable=unused-argument
    ) -> list[FusibleOperation]:
        """Apply operation fusion for forward pass.

        Parameters
        ----------
        ops : list of FusibleOperation
            Forward pass operations.

        Returns
        -------
        ops : list of FusibleOperation
            Updated forward pass operations

        """

        # Scan through ops, fusing if possible
        out = []
        window = []
        while ops:

            # Shift window
            out.extend(window)
            window = [ops[0]]
            ops = ops[1:]

            # Check if first op is linear
            if not isinstance(window[0], BasicLinear):
                continue
            if window[0].tensor_parallel_mode == "row":
                # Row tensor-parallelism requires communication after
                # the GEMM
                continue
            linear = window[0]

            # Check if next op is bias
            bias = None
            if ops and isinstance(ops[0], Bias):
                window.append(ops[0])
                ops = ops[1:]
                bias = window[-1]

            # Check if next op is in-place add extra input
            if ops and isinstance(ops[0], AddExtraInput) and ops[0]._in_place:
                window.append(ops[0])
                ops = ops[1:]
                add = window[-1]
            else:
                continue

            # Replace window with fused op
            op = ForwardLinearBiasAdd(linear=linear, bias=bias, add=add)
            window = [op]

        # Return list of ops
        out.extend(window)
        return out

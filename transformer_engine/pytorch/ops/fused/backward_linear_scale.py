# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused backward dgrad GEMM + scale."""

from __future__ import annotations
from typing import Optional

import torch

from ...module.base import get_dummy_wgrad
from ...utils import clear_tensor_data
from ..basic import BasicLinear, ConstantScale
from ..op import FusedOperation, FusibleOperation, OperationContext


class BackwardLinearScale(FusedOperation):
    """Fused backward dgrad GEMM + scale

    Column tensor parallelism is not supported since that requires
    communication immediately after the dgrad GEMM.

    """

    def __init__(
        self,
        *,
        scale: ConstantScale,
        linear: BasicLinear,
    ) -> None:
        super().__init__((linear, scale))

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        list[tuple[Optional[torch.Tensor], ...]],
        list[tuple[()]],
    ]:

        # Get basic operations
        linear_op = self.basic_ops[0]
        linear_op_ctx = basic_op_ctxs[0]
        scale_op = self.basic_ops[1]

        # Saved tensors from forward pass
        (x_local, w) = linear_op_ctx.saved_tensors

        # Megatron-LM wgrad fusion
        # Note: Get grad tensor from param so we can accumulate
        # directly into it.
        accumulate_into_main_grad = linear_op._accumulate_into_main_grad
        grad_weight = None
        if linear_op_ctx.weight_requires_grad and accumulate_into_main_grad:
            weight_param = linear_op.weight
            if hasattr(weight_param, "__fsdp_param__"):
                weight_param.main_grad = weight_param.get_main_grad()
            accumulate_into_main_grad = not getattr(weight_param, "overwrite_main_grad", False)
            if not hasattr(weight_param, "main_grad"):
                raise RuntimeError(
                    "BasicLinear op is configured with "
                    "accumulate_into_main_grad=True, "
                    "but weight parameter does not have main_grad attribute"
                )
            grad_weight = weight_param.main_grad.detach()
        else:
            accumulate_into_main_grad = False

        # Linear backward pass
        grad_input, grad_weight = BasicLinear._functional_backward(
            grad_output=grad_output,
            input=x_local,
            weight=w,
            input_requires_grad=linear_op_ctx.input_requires_grad,
            grad_input_alpha=scale_op.scale,
            weight_requires_grad=linear_op_ctx.weight_requires_grad,
            grad_weight_alpha=scale_op.scale,
            dtype=linear_op_ctx.dtype,
            grad_weight=grad_weight,
            accumulate_into_grad_weight=accumulate_into_main_grad,
            tensor_parallel_mode=linear_op.tensor_parallel_mode,
            tensor_parallel_group=linear_op.tensor_parallel_group,
            sequence_parallel=linear_op.sequence_parallel,
            with_quantized_compute=linear_op_ctx.with_quantized_compute,
            input_quantizer=linear_op_ctx.input_quantizer,
            weight_quantizer=linear_op_ctx.weight_quantizer,
            grad_output_quantizer=linear_op_ctx.grad_output_quantizer,
            grad_input_quantizer=linear_op_ctx.grad_input_quantizer,
        )

        # Clear input tensor if possible
        clear_tensor_data(x_local)

        # Megatron-LM wgrad fusion
        # Note: Return dummy tensor for grad weight if needed.
        if accumulate_into_main_grad:
            grad_weight = None
            weight_param = linear_op.weight
            if hasattr(weight_param, "grad_added_to_main_grad"):
                weight_param.grad_added_to_main_grad = True
                grad_weight = get_dummy_wgrad(
                    list(weight_param.size()),
                    weight_param.dtype,
                    zero=getattr(weight_param, "zero_out_wgrad", False),
                )

        return grad_input, [(grad_weight,), ()], [(), ()]

    @staticmethod
    def fuse_backward_ops(
        ops: list[FusibleOperation],
        **unused,  # pylint: disable=unused-argument
    ) -> list[FusibleOperation]:
        """Apply operation fusion for backward pass.

        Parameters
        ----------
        ops : list of FusibleOperation
            Backward pass operations.

        Returns
        -------
        ops : list of FusibleOperation
            Updated backward pass operations

        """

        # Scan through ops, fusing if possible
        out = []
        window, ops = ops[:2], ops[2:]
        while len(window) == 2:

            # Check if window matches pattern
            matches_pattern = True
            if not (isinstance(window[0], BasicLinear) and isinstance(window[1], ConstantScale)):
                matches_pattern = False
            elif window[0].tensor_parallel_mode == "column":
                # Column tensor-parallelism requires communication
                # after the dgrad GEMM
                matches_pattern = False

            if matches_pattern:
                # Construct fused op if window matches pattern
                op = BackwardLinearScale(linear=window[0], scale=window[1])
                window = [op]
            else:
                # Shift window if window doesn't match pattern
                out.extend(window[:-1])
                window = window[-1:]

            # Adjust window to expected size
            out.extend(window[:-2])
            window = window[-2:]
            while ops and len(window) < 2:
                window.append(ops[0])
                ops = ops[1:]

        # Return list of ops
        out.extend(window)
        return out

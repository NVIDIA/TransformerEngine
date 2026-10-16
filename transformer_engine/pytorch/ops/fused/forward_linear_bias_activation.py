# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused operation for forward GEMM + bias + activation."""

from __future__ import annotations
from typing import Optional

import torch

from ..basic import BasicLinear, Bias
from ..basic.basic_linear import BasicLinearFwdArgs
from ..op import FusedOperation, FusibleOperation


class ForwardLinearBiasActivation(FusedOperation):
    """Fused forward GEMM + bias + activation

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
        op_idxs = {"linear": 0, "bias": None, "activation": None}
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

    fwd_args_type = BasicLinearFwdArgs

    def compile_unsupported_reason(self, mode: str) -> Optional[str]:
        reason = super().compile_unsupported_reason(mode)
        if reason is not None:
            return reason
        return self.basic_ops[0].compile_unsupported_reason(mode)

    def pack_forward_args(self, basic_op_ctxs, input_, *, basic_op_kwargs, **kwargs):
        if self._op_idxs["activation"] is not None:
            raise NotImplementedError("Activations are not yet supported")
        args = self.basic_ops[0].pack_forward_args(
            basic_op_ctxs[:1], input_, basic_op_kwargs=basic_op_kwargs[:1], **kwargs
        )
        if self._op_idxs["bias"] is not None:
            idx = self._op_idxs["bias"]
            if basic_op_kwargs[idx]:
                raise ValueError("Bias forward does not expect keyword arguments")
            args.bias = self.basic_ops[idx].bias
        return args

    @classmethod
    def forward_compute(cls, args: BasicLinearFwdArgs, *, in_custom_op: bool = False):
        output, extras, aux = BasicLinear.forward_compute(args, in_custom_op=in_custom_op)
        if args.bias is not None:
            extras.append(())
        return output, extras, aux

    @classmethod
    def forward_compute_fake(cls, args: BasicLinearFwdArgs):
        output, extras, aux = BasicLinear.forward_compute_fake(args)
        if args.bias is not None:
            extras.append(())
        return output, extras, aux

    def forward_setup_context(self, basic_op_ctxs, args, aux) -> None:
        self.basic_ops[0].forward_setup_context(basic_op_ctxs[:1], args, aux)
        if self._op_idxs["bias"] is not None:
            basic_op_ctxs[self._op_idxs["bias"]].grad_input_quantizer = (
                args.grad_output_quantizer if args.backward_override is None else None
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
        window, ops = ops[:2], ops[2:]
        while len(window) == 2:

            # Check if window matches pattern
            matches_pattern = True
            if not (isinstance(window[0], BasicLinear) and isinstance(window[1], Bias)):
                matches_pattern = False
            elif window[0].tensor_parallel_mode == "row":
                # Row tensor-parallelism requires communication after
                # the GEMM
                matches_pattern = False
            elif window[0].weight.dtype not in (torch.float16, torch.bfloat16):
                # cuBLAS only supports fused GEMM+bias+activation with
                # FP16 and BF16 output
                matches_pattern = False

            if matches_pattern:
                # Construct fused op if window matches pattern
                op = ForwardLinearBiasActivation(
                    linear=window[0],
                    bias=window[1],
                    activation=None,
                )
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

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for all-reduce."""

from __future__ import annotations
from typing import Optional

import torch

from ...tensor import QuantizedTensor
from ..op import BasicOperation, OperationContext


class AllReduce(BasicOperation):
    """All-reduce tensor

    Equivalent to summing tensors from all processes. It is assumed
    that the output is used in operations that are redundantly
    computed on all processes, and hence that gradients are identical
    between processes.

    Parameters
    ----------
    process_group: torch.distributed.ProcessGroup, default = world group
        Process group for communication

    """

    def __init__(
        self,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        reduce_in_backward: bool = True,
    ) -> None:
        super().__init__()
        self.process_group: Optional[torch.distributed.ProcessGroup] = process_group
        self._reduce_in_backward: bool = reduce_in_backward

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
    ) -> torch.Tensor:

        # Trivial case
        if torch.distributed.get_world_size(self.process_group) == 1:
            return input_

        # Perform all-reduce
        x = input_
        if isinstance(x, QuantizedTensor):
            x = x.dequantize()
        x = x.contiguous()
        torch.distributed.all_reduce(x, group=self.process_group)
        return x

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        return grad_output, ()

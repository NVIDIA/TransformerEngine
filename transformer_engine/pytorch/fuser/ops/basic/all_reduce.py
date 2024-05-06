# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import torch

from transformer_engine.pytorch.fuser.ops.op import BasicOperation
from .._common import is_float8_tensor


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
        input: torch.Tensor,
    ) -> torch.Tensor:

        # Trivial case
        if torch.distributed.get_world_size(self.process_group) == 1:
            return input

        # Perform all-reduce
        x = input
        if is_float8_tensor(x):
            x = x.from_float8()
        x = x.contiguous()
        torch.distributed.all_reduce(x, group=self.process_group)
        return x

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        return grad_output, ()

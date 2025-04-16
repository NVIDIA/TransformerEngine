# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for all-gather."""

from __future__ import annotations
from typing import Optional

import torch

from ...distributed import gather_along_first_dim
from ...tensor import QuantizedTensor
from ..op import BasicOperation, OperationContext


class AllGather(BasicOperation):
    """All-gather tensor along outer dimension

    Equivalent to gathering tensors from all processes and
    concatenating along the first dimension.

    Parameters
    ----------
    process_group: torch.distributed.ProcessGroup, default = world group
        Process group for communication

    """

    def __init__(
        self,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        super().__init__()
        self.process_group: Optional[torch.distributed.ProcessGroup] = process_group
        self.process_group_size: int = torch.distributed.get_world_size(process_group)

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
    ) -> torch.Tensor:
        out: torch.Tensor
        if self.process_group_size == 1:
            out = input_.detach()
        else:
            out, _ = gather_along_first_dim(input_, self.process_group)
        return out

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:

        # Trivial case
        if self.process_group_size == 1:
            return grad_output, ()

        # Tensor dimensions
        output_dims = grad_output.size()
        if not output_dims or output_dims[0] % self.process_group_size != 0:
            raise RuntimeError(
                "Attempted to reduce-scatter a tensor "
                f"with shape={list(output_dims)} "
                f"over {self.process_group_size} processes"
            )
        input_dims = list(output_dims)
        input_dims[0] //= self.process_group_size

        # Check output gradient tensor
        dy = grad_output
        if isinstance(dy, QuantizedTensor):
            dy = dy.dequantize()
        dy = dy.contiguous()

        # Perform reduce-scatter
        dx = torch.empty(input_dims, dtype=dy.dtype, device=dy.device)
        torch.distributed.reduce_scatter_tensor(
            dx,
            dy,
            group=self.process_group,
        )
        return dx, ()

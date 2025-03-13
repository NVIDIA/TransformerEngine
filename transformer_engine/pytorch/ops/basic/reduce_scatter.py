# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for reduce-scatter."""

from __future__ import annotations
from typing import Optional

import torch

from ...distributed import gather_along_first_dim
from ...tensor import QuantizedTensor
from ..op import BasicOperation, OperationContext


class ReduceScatter(BasicOperation):
    """Reduce-scatter tensor along outer dimension

    Equivalent to summing tensors from all processes and splitting
    along the first dimension.

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

        # Trivial case
        if self.process_group_size == 1:
            return input_.detach()

        # Tensor dimensions
        input_dims = input_.size()
        if not input_dims or input_dims[0] % self.process_group_size != 0:
            raise RuntimeError(
                "Attempted to reduce-scatter a tensor "
                f"with shape={list(input_dims)} "
                f"over {self.process_group_size} processes"
            )
        output_dims = list(input_dims)
        output_dims[0] //= self.process_group_size

        # Check input tensor
        x = input_
        if isinstance(x, QuantizedTensor):
            x = x.dequantize()
        x = x.contiguous()

        # Perform reduce-scatter
        y = torch.empty(output_dims, dtype=x.dtype, device=x.device)
        torch.distributed.reduce_scatter_tensor(y, x, group=self.process_group)
        return y

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[()]]:
        grad_input: torch.Tensor
        if self.process_group_size == 1:
            grad_input = grad_output.detach()
        else:
            grad_input, _ = gather_along_first_dim(grad_output, self.process_group)
        return grad_input, ()

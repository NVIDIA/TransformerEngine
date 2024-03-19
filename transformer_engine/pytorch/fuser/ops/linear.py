# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Optional

import torch

from transformer_engine.pytorch.fuser.ops.op import FusedOperation
from transformer_engine.pytorch.fuser.ops.unfused import (
    AllReduce,
    Bias,
    ReduceScatter,
    UnfusedLinear,
)


class Linear(FusedOperation):
    """Apply linear transformation: :math:`y = x A^T + b`

    This is a drop-in replacement for `torch.nn.Linear`.

    Parameters
    ----------
    in_features: int
        Inner dimension of input tensor
    out_features: int
        Inner dimension of output tensor
    bias: bool, default = `True`
        Apply additive bias
    device: torch.device, default = default CUDA device
        Tensor device
    dtype: torch.dtype, default = default dtype
        Tensor datatype
    tensor_parallel_mode: {`None`, "column", "row"}, default = `None`
        Mode for tensor parallelism
    tensor_parallel_group: torch.distributed.ProcessGroup, default = world group
        Process group for tensor parallelism
    sequence_parallel: bool, default = `False`
        Whether to apply sequence parallelism together with tensor
        parallelism, i.e. distributing input or output tensors along
        outer dimension (sequence or batch dim) when not distributing
        along inner dimension (embedding dim)
    rng_state_tracker_function: callable
        Function that returns CudaRNGStatesTracker, which is used for
        model-parallel weight initialization

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        tensor_parallel_mode: Optional[str] = None,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        sequence_parallel: bool = False,
        rng_state_tracker_function: Optional[Callable[[], CudaRNGStatesTracker]] = None,
    ) -> None:

        # Tensor parallel configuration
        (
            tensor_parallel_mode,
            tensor_parallel_group,
            tensor_parallel_size,
            sequence_parallel,
            local_in_features,
            local_out_features,
        ) = UnfusedLinear._canonicalize_tensor_parallelism(
            mode=tensor_parallel_mode,
            process_group=tensor_parallel_group,
            sequence_parallel=sequence_parallel,
            in_features=in_features,
            out_features=out_features,
        )

        # Construct unfused ops
        ops = []
        if tensor_parallel_mode == "row":
            # Row TP: GEMM + bias + reduction
            ops.append(
                UnfusedLinear(
                    local_in_features,
                    local_out_features,
                    device=device,
                    dtype=dtype,
                    tensor_parallel_mode=None,
                    tensor_parallel_group=None,
                    sequence_parallel=None,
                    rng_state_tracker_function=rng_state_tracker_function,
                )
            )
            if bias:
                ops.append(
                    Bias(
                        local_out_features,
                        device=device,
                        dtype=dtype,
                        tensor_parallel=False,
                        tensor_parallel_group=None,
                    )
                )
            if sequence_parallel:
                ops.append(ReduceScatter(tensor_parallel_group))
            else:
                ops.append(AllReduce(tensor_parallel_group))
        else:
            # Column TP or no TP: (gather + GEMM) + bias
            ops.append(
                UnfusedLinear(
                    in_features,
                    out_features,
                    device=device,
                    dtype=dtype,
                    tensor_parallel_mode=tensor_parallel_mode,
                    tensor_parallel_group=tensor_parallel_group,
                    sequence_parallel=sequence_parallel,
                    rng_state_tracker_function=rng_state_tracker_function,
                )
            )
            if bias:
                ops.append(
                    Bias(
                        out_features,
                        device=device,
                        dtype=dtype,
                        tensor_parallel=(tensor_parallel_mode is not None),
                        tensor_parallel_group=tensor_parallel_group,
                    )
                )

        # Initialize base class
        super().__init__(ops)

        # Register parameters
        self.register_parameter("weight", self.unfused_ops[0].weight)
        self.register_parameter("bias", self.unfused_ops[1].bias if bias else None)

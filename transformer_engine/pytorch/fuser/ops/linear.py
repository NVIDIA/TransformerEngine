# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Optional

import torch

from transformer_engine.pytorch.fuser.ops.op import FusedOperation
from transformer_engine.pytorch.fuser.ops.unfused import Bias, UnfusedLinear


class Linear(FusedOperation):

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

        # Initialize unfused ops
        ### TODO Tensor/sequence parallelism
        ops = [
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
        ]
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

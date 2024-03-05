# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Optional

import torch

from .op import FusableOperation


class Linear(FusableOperation):

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
        linear_op = UnfusedLinear(
            in_features,
            out_features,
            device=device,
            dtype=dtype,
            tensor_parallel_mode=tensor_parallel_mode,
            tensor_parallel_group=tensor_parallel_group,
            sequence_parallel=sequence_parallel,
            rng_state_tracker_function=rng_state_tracker_function,
        )
        ops = [linear_op]
        if bias:
            bias_op = Bias(
                out_features,
                device=device,
                dtype=dtype,
                tensor_parallel=(tensor_parallel_mode is not None),
                tensor_parallel_group=tensor_parallel_group,
            )
            ops.append(bias_op)

        # Initialize base class
        super().__init__(ops)

        # Register parameters
        self.register_parameter("weight", linear_op.weight)
        if bias:
            self.register_parameter("bias", bias_op.bias)
        else:
            self.bias = torch.Tensor(
                device=linear_op.device,
                dtype=linear_op.dtype,
            )

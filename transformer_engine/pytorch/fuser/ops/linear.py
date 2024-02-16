# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from collections.abc import Callable, Iterable
import contextlib
import math
from typing import Optional

import torch

from ...cpp_extensions import gemm
from ...distributed import (
    CudaRNGStatesTracker,
    gather_along_first_dim,
    reduce_scatter_along_first_dim,
)
from ...fp8 import FP8GlobalStateManager
from ...module.base import get_workspace
from ._common import canonicalize_device, canonicalize_dtype
from .op import FusableOperation

class Linear(FusableOperation):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        tensor_parallel_mode: Optional[str] = None,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        sequence_parallel: bool = False,
        rng_state_tracker_function: Optional[Callable[[], CudaRNGStatesTracker]] = None,
    ) -> None:
        super().__init__()

        # Weight tensor dimensions
        self.in_features: int = in_features
        self.out_features: int = out_features

        # Check device
        defer_param_init = False
        device = canonicalize_device(device)
        if device.type == "meta":
            defer_param_init = True
            device = canonicalize_device(None)
        if device.type != "cuda":
            raise ValueError(f"Only CUDA devices are supported (got {device})")
        self.device: torch.device = device

        # Check dtype
        dtype = canonicalize_dtype(dtype)
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(
                f"Supported dtypes are float32, float16, bfloat16 (got {dtype})"
            )
        self.dtype: torch.dtype = canonicalize_dtype(dtype)

        # Configure tensor parallelism
        tensor_parallel_size: int = 1
        local_in_features: int = in_features
        local_out_features: int = out_features
        if tensor_parallel_mode is None:
            tensor_parallel_group = None
            sequence_parallel = False
        else:
            tensor_parallel_size = torch.distributed.get_world_size(tensor_parallel_group)
            if tensor_parallel_group == "column":
                if out_features % tensor_parallel_size != 0:
                    raise ValueError(
                        "Invalid configuration for tensor parallelism "
                        f"({tensor_parallel_mode=}, "
                        f"{out_features=}, "
                        f"{tensor_parallel_size=})"
                    )
                local_out_features /= tensor_parallel_size
            elif tensor_parallel_group == "row":
                if in_features % tensor_parallel_size != 0:
                    raise ValueError(
                        "Invalid configuration for tensor parallelism "
                        f"({tensor_parallel_mode=}, "
                        f"{in_features=}, "
                        f"{tensor_parallel_size=})"
                    )
                local_in_features /= tensor_parallel_size
            else:
                raise ValueError(
                    'Supported modes for tensor parallelism are "row" and "column" '
                    f"(got {tensor_parallel_mode=})"
                )
        self.tensor_parallel_mode: Optional[str] = tensor_parallel_mode
        self.tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = tensor_parallel_group
        self.tensor_parallel_size: int = tensor_parallel_size
        self.sequence_parallel: bool = sequence_parallel
        self.local_in_features: int = local_in_features
        self.local_out_features: int = local_out_features

        # Native FP8 parameters
        self._with_fp8_parameters = FP8GlobalStateManager.with_fp8_parameters()
        if self._with_fp8_parameters:
            defer_param_init = True

        # Initialize parameters if needed
        weight = torch.empty(
            local_out_features,
            local_in_features,
            device="meta" if defer_param_init else device,
            dtype=dtype,
        )
        weight = torch.nn.Parameter(weight)
        self.register_parameter("weight", weight)
        self._rng_state_tracker_function: Optional[Callable[[], CudaRNGStatesTracker]]
        self._rng_state_tracker_function = rng_state_tracker_function
        if not defer_param_init:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameter buffers and values"""

        # Make sure parameter is on a CUDA device
        weight = self.weight
        if weight.device.type != "cuda":
            weight = torch.empty_like(weight, device=self.device)

        # Initialize values
        init_context = contextlib.nullcontext
        if self._rng_state_tracker_function is not None:
            init_context = self._rng_state_tracker_function().fork
        with init_context():
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        # Cast to FP8 if needed
        ### TODO Fix
        # fp8_meta_index = ### TODO ?
        # if self._with_fp8_parameters:
        #     weight = Float8Tensor.to_float8(
        #         weight,
        #         fp8_meta=self.fp8_meta,
        #         fp8_meta_index=fp8_meta_index
        #     )

        # Save updated parameter
        if not isinstance(weight, torch.nn.Parameter):
            weight = torch.nn.Parameter(weight)
        self.weight = weight

    def _lazy_reset_parameters(self) -> None:
        """Initialize parameters if not already initialized"""
        if self.weight.device.type != "cuda":
            self.reset_parameters

    def unfused_op_forward(
        self,
        ctx: OperationContext,
        input: torch.Tensor,
    ) -> torch.Tensor:
        ### TODO FP8

        # Make sure parameters are ready
        self._lazy_reset_parameters()

        # Check tensors
        input_dims = input.size()
        if self.weight.size(1) != input_dims[-1]:
            raise ValueError(
                f"Input tensor (shape={tuple(input.size())}) "
                f"and weight tensor (shape={tuple(self.weight.size())}) "
                "are not compatible"
            )
        x = input.reshape(-1, input_dims[-1])
        x = x.to(
            device=self.device,
            dtype=self.dtype,
            memory_format=torch.contiguous_format,
        )
        w = self.weight.contiguous()

        # Gather sequence-parallel input if needed
        local_x = x
        if self.tensor_parallel_mode == "column" and self.sequence_parallel:
            x, _ = gather_along_first_dim(local_x, self.tensor_parallel_group)

        # Apply GEMM
        y = torch.empty(
            (x.size(0), w.size(0)),
            dtype=self.dtype,
            device=self.device,
        )
        gemm(
            w,
            x,
            self.dtype,
            get_workspace(),
            bias=None,
            use_bias=False,
            out=y,
        )

        # Reduce tensor-parallel output if needed
        if self.tensor_parallel_mode == "row":
            if sequence_parallel:
                y = reduce_scatter_along_first_dim(y, self.tensor_parallel_group)
            else:
                torch.distributed.all_reduce(y, group=self.tensor_parallel_group)

        ctx.save_for_backward(
            local_x,
        )
        ctx.input_dims = input_dims
        ctx.requires_dgrad = True  ### TODO input.requires_grad

        return y.reshape(-1, *input_dims[1:-1], y.size(-1))

    def unfused_op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:

        (
            local_x,
        ) = ctx.saved_tensors

        # Check tensors
        dy = grad_output.reshape(-1, grad_output.size(-1))
        dy = dy.to(
            device=self.weight.device,
            dtype=self.weight.dtype,
            memory_format=torch.contiguous_format,
        )

        # Helper function for async communication
        async_handle = None
        def wait_async_handle():
            nonlocal async_handle
            if async_handle is not None:
                async_handle.wait()
                async_handle = None

        # Gather sequence-parallel input if needed
        # Note: Try overlapping with dgrad
        wait_async_handle()
        x = local_x
        if (
            self.weight.requires_grad
            and self.tensor_parallel_mode == "column"
            and self.sequence_parallel
        ):
            x, async_handle = gather_along_first_dim(
                x,
                self.tensor_parallel_group,
                async_op=ctx.requires_dgrad,
            )

        # Apply dgrad GEMM
        dx = None
        if ctx.requires_dgrad:
            w = self.weight.contiguous()
            dx, _, _ = gemm(
                w,
                dy,
                self.dtype,
                get_workspace(),
                layout="NN",
                grad=True,
            )

        # Reduce tensor-parallel grad input if needed
        # Note: Try overlapping with wgrad
        wait_async_handle()
        if ctx.requires_dgrad and self.tensor_parallel_mode == "column":
            if self.sequence_parallel:
                dx, async_handle = reduce_scatter_along_first_dim(
                    dx,
                    self.tensor_parallel_group,
                    async_op=self.weight.requires_grad,
                )
            else:
                async_handle = torch.distributed.all_reduce(
                    dx,
                    group=self.tensor_parallel_group,
                    async_op=self.weight.requires_grad,
                )

        # Apply wgrad GEMM
        if self.weight.requires_grad:
            dw, _, _ = gemm(
                x,
                dy,
                self.dtype,
                get_workspace(),
                layout="NT",
                grad=True,
                # accumulate=accumulate_wgrad_into_param_main_grad
                # out=weight.main_grad if ctx.fuse_wgrad_accumulation else
            )

        wait_async_handle()
        if dx is not None:
            dx = dx.reshape(ctx.input_dims)
        return dx, [dw]

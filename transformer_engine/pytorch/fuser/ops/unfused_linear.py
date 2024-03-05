# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from collections.abc import Callable, Iterable
import contextlib
import math
from typing import Optional

import torch

from ...cpp_extensions import fp8_gemm, gemm
from ...distributed import (
    CudaRNGStatesTracker,
    gather_along_first_dim,
    reduce_scatter_along_first_dim,
)
from ...float8_tensor import Float8Tensor
from ...fp8 import FP8GlobalStateManager
from ...module.base import get_workspace
from ._common import (
    canonicalize_device,
    canonicalize_dtype,
    convert_tensor,
    is_float8_tensor,
)
from .op import FusableOperation


class UnfusedLinear(FusableOperation):

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

        # Weight tensor device
        defer_param_init = False
        device = canonicalize_device(device)
        if device.type == "meta":
            defer_param_init = True
            device = canonicalize_device(None)
        if device.type != "cuda":
            raise ValueError(f"Only CUDA devices are supported (got {device})")
        self.device: torch.device = device

        # Weight tensor datatype
        dtype = canonicalize_dtype(dtype)
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(
                f"Supported dtypes are float32, float16, bfloat16 (got {dtype})"
            )
        self.dtype: torch.dtype = canonicalize_dtype(dtype)

        # Tensor parallel configuration
        tensor_parallel_size = 1
        local_in_features = in_features
        local_out_features = out_features
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

        # Whether weight tensor is natively in FP8
        self._with_fp8_parameters = FP8GlobalStateManager.with_fp8_parameters()
        if self._with_fp8_parameters:
            defer_param_init = True

        # Initialize parameters if needed
        weight = torch.empty(
            local_out_features,
            local_in_features,
            device="meta",
            dtype=dtype,
        )
        weight = torch.nn.Parameter(weight)
        self.register_parameter("weight", weight)
        self._rng_state_tracker_function: Optional[Callable[[], CudaRNGStatesTracker]]
        self._rng_state_tracker_function = rng_state_tracker_function
        if not defer_param_init:
            self.reset_parameters()

    def num_fp8_scales(self, mode: str) -> int:
        if mode == "input":
            return 1
        if mode == "param":
            return 1
        if mode == "grad_output":
            return 1
        return 0

    def reset_parameters(self) -> None:
        """Initialize parameter buffers and values"""

        # Make sure parameter is initialized
        weight = self.weight
        if weight.device.type != "cuda" or is_float8_tensor(weight):
            weight = torch.empty_like(weight, device=self.device)
        weight = weight.to(device=self.device, dtype=self.dtype)

        # Initialize values
        init_context = contextlib.nullcontext
        if self._rng_state_tracker_function is not None:
            init_context = self._rng_state_tracker_function().fork
        with init_context():
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        # Cast to FP8 if needed
        if self._with_fp8_parameters:
            if self._fp8_metas is None:
                self._fp8_metas = self._make_fp8_metas()
            weight = Float8Tensor.to_float8(
                weight,
                fp8_meta=self._get_fp8_meta("param"),
                fp8_meta_index=0,
            )

        # Save updated parameter
        if not isinstance(weight, torch.nn.Parameter):
            weight = torch.nn.Parameter(weight)
        self.weight = weight

    def _pre_forward(self) -> None:
        super()._pre_forward()
        if self.weight.device.type == "meta":
            self.reset_parameters()

    def _unfused_op_forward(
        self,
        ctx: OperationContext,
        input: torch.Tensor,
    ) -> torch.Tensor:

        # Check if FP8 is enabled
        fp8_enabled = FP8GlobalStateManager.is_fp8_enabled()

        # Check input tensor
        input_dims = input.size()
        if self.weight.size(1) != input_dims[-1]:
            raise ValueError(
                f"Input tensor (shape={tuple(input.size())}) "
                f"and weight tensor (shape={tuple(self.weight.size())}) "
                "are not compatible"
            )
        x = convert_tensor(
            input,
            device=self.device,
            dtype=self.dtype,
            memory_format=torch.contiguous_format,
        )
        x = x.view(-1, input_dims[-1])
        if fp8_enabled and not is_float8_tensor(x):
            x = Float8Tensor.to_float8(
                x,
                fp8_meta=self._get_fp8_meta("input"),
                fp8_meta_index=0,
            )
        elif not fp8_enabled and is_float8_tensor(x):
            x = x.from_float8()

        # Gather sequence-parallel input if needed
        local_x = x
        async_handle = None
        if self.tensor_parallel_mode == "column" and self.sequence_parallel:
            x, async_handle = gather_along_first_dim(
                local_x,
                self.tensor_parallel_group,
                async_op=True,
            )

        # Check weight tensor
        ### TODO: Weight caching without FP8 params
        w = convert_tensor(
            self.weight,
            device=self.device,
            dtype=self.dtype,
            memory_format=torch.contiguous_format,
        )
        if fp8_enabled and not is_float8_tensor(w):
            w = Float8Tensor.to_float8(
                w,
                fp8_meta=self._get_fp8_meta("param"),
                fp8_meta_index=0,
            )
        elif not fp8_enabled and is_float8_tensor(w):
            w = w.from_float8()

        # Synchronize async communication
        if async_handle is not None:
            async_handle.wait()

        # Perform GEMM
        y = torch.empty(
            (x.size(0), w.size(0)),
            dtype=self.dtype,
            device=self.device,
        )
        if fp8_enabled:
            fp8_gemm(
                w._data,
                w._scale_inv,
                0,
                w._fp8_dtype,
                x._data,
                x._scale_inv,
                0,
                x._fp8_dtype,
                y.dtype,
                get_workspace(),
                out=y,
            )
        else:
            gemm(
                w,
                x,
                y.dtype,
                get_workspace(),
                out=y,
            )

        # Reduce tensor-parallel output if needed
        if self.tensor_parallel_mode == "row":
            if sequence_parallel:
                y = reduce_scatter_along_first_dim(y, self.tensor_parallel_group)
            else:
                torch.distributed.all_reduce(y, group=self.tensor_parallel_group)

        # Save state for backward pass
        ctx.save_for_backward(
            local_x,
        )
        ctx.fp8_enabled = fp8_enabled
        ctx.input_dims = input_dims
        ctx.requires_dgrad = True  ### TODO input.requires_grad

        # Reshape output tensor
        if len(input_dims) > 1:
            y = y.reshape(-1, *input_dims[1:-1], y.size(-1))
        else:
            y = y.reshape(-1)
        return y

    def _unfused_op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:

        # Load state from forward pass
        (
            local_x,
        ) = ctx.saved_tensors
        fp8_enabled = ctx.fp8_enabled

        # Helper function for async communication
        async_handle = None
        def wait_async_handle():
            nonlocal async_handle
            if async_handle is not None:
                async_handle.wait()
                async_handle = None

        # Gather sequence-parallel input if needed
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

        # Check grad output tensor
        ### TODO: fused cast-transpose
        dy = convert_tensor(
            grad_output,
            device=self.device,
            dtype=self.dtype,
            memory_format=torch.contiguous_format,
        )
        dy = dy.view(-1, dy.size(-1))
        if fp8_enabled and not is_float8_tensor(dy):
            dy = Float8Tensor.to_float8(
                dy,
                fp8_meta=self._get_fp8_meta("grad_output"),
                fp8_meta_forward=False,
                fp8_meta_index=0,
            )
        elif not fp8_enabled and is_float8_tensor(dy):
            dy = dy.from_float8()

        # Compute grad input
        dx = None
        if ctx.requires_dgrad:

            # Check weight tensor
            ### TODO: Weight caching without FP8 params
            ### TODO: FP8 transpose caching
            ### TODO: fused cast-transpose
            w, w_t = None, None
            if fp8_enabled:
                w_t = convert_tensor(
                    self.weight.transpose(),
                    device=self.device,
                    dtype=self.dtype,
                    memory_format=torch.contiguous_format,
                )
                if not is_float8_tensor(w_t):
                    w_t = Float8Tensor.to_float8(
                        w_t,
                        fp8_meta=self._get_fp8_meta("param"),
                        fp8_meta_index=0,
                    )
            else:
                w = convert_tensor(
                    self.weight,
                    device=self.device,
                    dtype=self.dtype,
                    memory_format=torch.contiguous_format,
                )
                if is_float8_tensor(w):
                    w = w.from_float8()

            # Perform dgrad GEMM
            dx = torch.empty(
                (dy.size(0), self.weight.size(1)),
                dtype=self.dtype,
                device=self.device,
            )
            if fp8_enabled:
                fp8_gemm(
                    w_t._data,
                    w_t._scale_inv,
                    0,
                    w_t._fp8_dtype,
                    dy._data,
                    dy._scale_inv,
                    0,
                    dy._fp8_dtype,
                    dx.dtype,
                    get_workspace(),
                    out=dx,
                )
            else:
                gemm(
                    w,
                    dy,
                    dx.dtype,
                    get_workspace(),
                    layout="NN",
                    out=dx,
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

        # Perform wgrad GEMM
        ### TODO: grad accumulation
        ### TODO: smarter handling of transpose
        dw = None
        if self.weight.requires_grad:
            dw = torch.empty(
                self.weight.size(),
                dtype=self.dtype,
                device=self.device,
                memory_format=torch.contiguous_format,
            )
            if fp8_enabled:
                fp8_gemm(
                    x.transpose()._data,
                    x._scale_inv,
                    0,
                    x._fp8_dtype,
                    dy.transpose()._data,
                    dy._scale_inv,
                    0,
                    dy._fp8_dtype,
                    dw.dtype,
                    get_workspace(),
                    out=dw,
                )
            else:
                gemm(
                    x,
                    dy,
                    dw.dtype,
                    get_workspace(),
                    out=dw,
                    layout="NT",
            )

        # Clean up and return grads
        wait_async_handle()
        if dx is not None:
            dx = dx.reshape(ctx.input_dims)
        return dx, [dw]

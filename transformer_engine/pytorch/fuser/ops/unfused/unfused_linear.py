# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from collections.abc import Callable, Iterable
import contextlib
import math
from typing import Optional

import torch

from transformer_engine.pytorch.cpp_extensions import fp8_gemm, gemm
from transformer_engine.pytorch.distributed import (
    CudaRNGStatesTracker,
    gather_along_first_dim,
    reduce_scatter_along_first_dim,
)
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.fp8 import (
    FP8GlobalStateManager,
    get_fp8_te_dtype,
)
from transformer_engine.pytorch.fuser.ops.op import UnfusedOperation
from transformer_engine.pytorch.module.base import get_workspace
from .._common import (
    canonicalize_device,
    canonicalize_dtype,
    convert_tensor,
    fp8_cast_transpose,
    is_float8_tensor,
)

def _wait_async(handle: Optional[Any]) -> None:
    """Wait for asynchronous communication to finish, if needed"""
    if handle is not None:
        handle.wait()


class UnfusedLinear(UnfusedOperation):
    """Apply linear transformation: :math:`y = x A^T`

    This is a drop-in replacement for `torch.nn.Linear` with
    `bias=False`.

    Parameters
    ----------
    in_features: int
        Inner dimension of input tensor
    out_features: int
        Inner dimension of output tensor
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
        Function that returns `CudaRNGStatesTracker`, which is used
        for model-parallel weight initialization

    """

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
        self.tensor_parallel_mode: Optional[str]
        self.tensor_parallel_group: Optional[torch.distributed.ProcessGroup]
        self.tensor_parallel_size: int
        self.sequence_parallel: bool
        self.local_in_features: int
        self.local_out_features: int
        (
            self.tensor_parallel_mode,
            self.tensor_parallel_group,
            self.tensor_parallel_size,
            self.sequence_parallel,
            self.local_in_features,
            self.local_out_features,
        ) = self._canonicalize_tensor_parallelism(
            mode=tensor_parallel_mode,
            process_group=tensor_parallel_group,
            sequence_parallel=sequence_parallel,
            in_features=in_features,
            out_features=out_features,
        )

        # Whether weight tensor is natively in FP8
        self._with_fp8_parameters = FP8GlobalStateManager.with_fp8_parameters()
        if self._with_fp8_parameters:
            self._fp8_metas = self._make_fp8_metas()

        # Initialize parameters if needed
        weight = torch.empty(
            self.local_out_features,
            self.local_in_features,
            device="meta",
            dtype=dtype,
        )
        weight = torch.nn.Parameter(weight)
        self.register_parameter("weight", weight)
        self._rng_state_tracker_function: Optional[Callable[[], CudaRNGStatesTracker]]
        self._rng_state_tracker_function = rng_state_tracker_function
        if not defer_param_init:
            self.reset_parameters()

    @classmethod
    def _canonicalize_tensor_parallelism(
        cls,
        *,
        mode: Optional[str],
        process_group: Optional[torch.distributed.ProcessGroup],
        sequence_parallel: bool,
        in_features: int,
        out_features: int,
    ) -> tuple[
        Optional[str],
        Optional[torch.distributed.ProcessGroup],
        int,
        bool,
        int,
        int,
    ]:
        """Check configuration for tensor parallelism

        Parameters
        ----------
        mode: {`None`, "column", "row"}
            Mode for tensor parallelism
        process_group: torch.distributed.ProcessGroup
            Process group for tensor parallelism
        sequence_parallel: bool
            Whether to apply sequence parallelism together with tensor
            parallelism, i.e. distributing input or output tensors
            along outer dimension (sequence or batch dim) when not
            distributing along inner dimension (embedding dim)
        in_features: int
            Inner dimension of global input tensor
        out_features: int
            Inner dimension of global output tensor

        Returns
        -------
        mode: {`None`, "column", "row"}
            Mode for tensor parallelism
        process_group: torch.distributed.ProcessGroup
            Process group for tensor parallelism
        group_size: int
            Size of tensor-parallel process group
        sequence_parallel: bool
            Whether to apply sequence parallelism
        local_in_features: int
            Inner dimension of local input tensor
        local_out_features: int
            Inner dimension of local output tensor

        """

        # Tensor-parallel group size
        if mode is None:
            group_size = 1
        else:
            group_size = torch.distributed.get_world_size(process_group)

        # Disable tensor parallelism if not needed
        if group_size == 1:
            mode = None
            process_group = None
            sequence_parallel = False

        # Determine local tensor dims
        local_in_features = in_features
        local_out_features = out_features
        if mode is None:
            pass
        elif mode == "column":
            # Distribute output tensor
            if out_features % group_size != 0:
                raise ValueError(
                    "Invalid configuration for tensor parallelism "
                    f"({mode=}, {out_features=}, {group_size=})"
                )
            local_out_features //= group_size
        elif mode == "row":
            # Distribute input tensor
            if in_features % group_size != 0:
                raise ValueError(
                    "Invalid configuration for tensor parallelism "
                    f"({mode=}, {in_features=}, {group_size=})"
                )
            local_in_features //= group_size
        else:
            raise ValueError(
                "Supported modes for tensor parallelism are "
                f'`None`, "row", and "column" (got {mode=})'
            )

        return (
            mode,
            process_group,
            group_size,
            sequence_parallel,
            local_in_features,
            local_out_features,
        )

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
            weight = Float8Tensor.to_float8(
                weight,
                fp8_meta=self.get_fp8_meta("param"),
                fp8_meta_index=0,
            )

        # Save updated parameter
        if not isinstance(weight, torch.nn.Parameter):
            weight = torch.nn.Parameter(weight)
        self.weight = weight

    def pre_forward(self) -> None:
        super().pre_forward()
        if self.weight.device.type == "meta":
            self.reset_parameters()

    def op_forward(
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
        local_x = convert_tensor(
            input,
            device=self.device,
            dtype=self.dtype,
            memory_format=torch.contiguous_format,
        )
        local_x = local_x.view(-1, input_dims[-1])  ### TODO Preserve transpose
        if fp8_enabled and not is_float8_tensor(local_x):
            fp8_meta = self.get_fp8_meta("input")
            fp8_dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
            with_cast_transpose = self.weight.requires_grad
            if self.tensor_parallel_mode == "column" and self.sequence_parallel:
                with_cast_transpose = False
            if with_cast_transpose:
                local_x = fp8_cast_transpose(
                    local_x,
                    fp8_meta=fp8_meta,
                    fp8_meta_forward=True,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                )
            else:
                local_x = Float8Tensor.to_float8(
                    local_x,
                    fp8_meta=fp8_meta,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                )
        elif not fp8_enabled and is_float8_tensor(local_x):
            local_x = local_x.from_float8()
        x = local_x
        x_async = None
        if self.tensor_parallel_mode == "column" and self.sequence_parallel:
            x, x_async = gather_along_first_dim(
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
            fp8_meta = self.get_fp8_meta("param")
            fp8_dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
            w = Float8Tensor.to_float8(
                w,
                fp8_meta=fp8_meta,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
            )
        elif not fp8_enabled and is_float8_tensor(w):
            w = w.from_float8()

        # Perform GEMM
        y = torch.empty(
            (x.size(0), w.size(0)),
            dtype=self.dtype,
            device=self.device,
        )
        x_async = _wait_async(x_async)
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
            if self.sequence_parallel:
                y, _ = reduce_scatter_along_first_dim(y, self.tensor_parallel_group)
            else:
                torch.distributed.all_reduce(y, group=self.tensor_parallel_group)

        # Save state for backward pass
        ctx.save_for_backward(
            local_x,
        )
        ctx.fp8_enabled = fp8_enabled
        ctx.input_dims = input_dims
        ctx.requires_dgrad = input.requires_grad

        # Reshape output tensor
        if len(input_dims) > 1:
            y = y.reshape(-1, *input_dims[1:-1], y.size(-1))
        else:
            y = y.reshape(-1)
        return y

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:

        # Load state from forward pass
        (
            local_x,
        ) = ctx.saved_tensors
        fp8_enabled = ctx.fp8_enabled

        # Check grad output tensor
        dy_async = None
        dy = convert_tensor(
            grad_output,
            device=self.device,
            dtype=self.dtype,
            memory_format=torch.contiguous_format,
        )
        dy = dy.view(-1, dy.size(-1))  ### TODO Preserve transpose
        if fp8_enabled and not is_float8_tensor(dy):
            fp8_meta = self.get_fp8_meta("grad_output")
            fp8_dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=False)
            with_cast_transpose = self.weight.requires_grad
            if self.tensor_parallel_mode == "row" and self.sequence_parallel:
                with_cast_transpose = False
            if with_cast_transpose:
                dy = fp8_cast_transpose(
                    dy,
                    fp8_meta=fp8_meta,
                    fp8_meta_forward=False,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                )
            else:
                dy = Float8Tensor.to_float8(
                    dy,
                    fp8_meta=fp8_meta,
                    fp8_meta_forward=False,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                )
        elif not fp8_enabled and is_float8_tensor(dy):
            dy = dy.from_float8()
        if self.tensor_parallel_mode == "row" and self.sequence_parallel:
            dy, dy_async = gather_along_first_dim(
                dy,
                self.tensor_parallel_group,
                async_op=True,
            )

        # Gather sequence-parallel input if needed
        x = local_x
        x_async = None
        if (
            self.weight.requires_grad
            and self.tensor_parallel_mode == "column"
            and self.sequence_parallel
        ):
            x, x_async = gather_along_first_dim(
                x,
                self.tensor_parallel_group,
                async_op=True,
            )

        # Compute grad input
        dx = None
        dx_async = None
        if ctx.requires_dgrad:

            # Check weight tensor
            ### TODO: Weight caching without FP8 params
            ### TODO Configurable FP8 transpose caching
            w = convert_tensor(
                self.weight,
                device=self.device,
                dtype=self.dtype,
                memory_format=torch.contiguous_format,
            )
            if fp8_enabled and not is_float8_tensor(w):
                fp8_meta = self.get_fp8_meta("param")
                fp8_dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
                w = fp8_cast_transpose(
                    w,
                    fp8_meta=fp8_meta,
                    fp8_meta_forward=True,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                )
            elif not fp8_enabled and is_float8_tensor(w):
                w = w.from_float8()

            # Perform dgrad GEMM
            dx = torch.empty(
                (dy.size(0), self.weight.size(1)),
                dtype=self.dtype,
                device=self.device,
            )
            dy_async = _wait_async(dy_async)
            if fp8_enabled:
                fp8_gemm(
                    w.transpose()._data,
                    w._scale_inv,
                    0,
                    w._fp8_dtype,
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
            if self.tensor_parallel_mode == "column":
                if self.sequence_parallel:
                    dx, dx_async = reduce_scatter_along_first_dim(
                        dx,
                        self.tensor_parallel_group,
                        async_op=True,
                    )
                else:
                    dx_async = torch.distributed.all_reduce(
                        dx,
                        group=self.tensor_parallel_group,
                        async_op=True,
                    )

        # Perform wgrad GEMM
        ### TODO: grad accumulation
        dw = None
        if self.weight.requires_grad:
            dw = torch.empty(
                self.weight.size(),
                dtype=self.dtype,
                device=self.device,
                memory_format=torch.contiguous_format,
            )
            dy_async = _wait_async(dy_async)
            x_async = _wait_async(x_async)
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
        _wait_async(dy_async)
        _wait_async(x_async)
        _wait_async(dx_async)
        if dx is not None:
            dx = dx.reshape(ctx.input_dims)
        return dx, [dw]

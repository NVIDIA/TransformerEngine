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
from transformer_engine.pytorch.fuser.ops.op import BasicOperation
from transformer_engine.pytorch.module.base import get_workspace
from .._common import (
    canonicalize_device,
    canonicalize_dtype,
    convert_tensor,
    fp8_cast_transpose,
    is_float8_tensor,
    reshape,
)

def _wait_async(handle: Optional[Any]) -> None:
    """Wait for asynchronous communication to finish, if needed"""
    if handle is not None:
        handle.wait()


class BasicLinear(BasicOperation):
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
    accumulate_into_main_grad: bool, default = `False`
        Whether to directly accumulate weight gradients into the
        weight's `main_grad` attribute instead of relying on PyTorch
        autograd. The weight's `main_grad` must be set externally and
        there is no guarantee that `grad` will be set or be
        meaningful.

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
        accumulate_into_main_grad: bool = False,
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

        # Whether to accumulate weight gradient into main_grad
        self._accumulate_into_main_grad = accumulate_into_main_grad

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

    @staticmethod
    def _functional_forward(
        ctx: OperationContext,
        input: torch.Tensor,
        weight: torch.Tensor,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        tensor_parallel_mode: Optional[str] = None,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        sequence_parallel: bool = False,
        input_fp8_meta: Optional[dict[str,Any]] = None,
        weight_fp8_meta: Optional[dict[str,Any]] = None,
        output_fp8_meta: Optional[dict[str,Any]] = None,
        grad_output_fp8_meta: Optional[dict[str,Any]] = None,
        grad_input_fp8_meta: Optional[dict[str,Any]] = None,
        accumulate_into_main_grad: bool = False,
    ) -> torch.Tensor:

        # Check device
        if device is None:
            device = weight.device
        device = canonicalize_device(device)
        if device.type != "cuda":
            raise ValueError(f"Only CUDA devices are supported (got {device})")

        # Check datatype
        if dtype is None:
            dtype = weight.dtype
        dtype = canonicalize_dtype(dtype)
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(
                f"Supported dtypes are float32, float16, bfloat16 (got {dtype})"
            )

        # Required grads
        requires_dgrad = input.requires_grad
        requires_wgrad = weight.requires_grad
        requires_backward = requires_dgrad or requires_wgrad

        # Check if FP8 is enabled
        with_fp8_compute = (
            FP8GlobalStateManager.is_fp8_enabled()
            and input_fp8_meta is not None
            and weight_fp8_meta is not None
        )
        if with_fp8_compute and requires_backward:
            with_fp8_compute = grad_output_fp8_meta is not None
        if not with_fp8_compute:
            input_fp8_meta = None
            weight_fp8_meta = None
            output_fp8_meta = None
            grad_output_fp8_meta = None
            grad_input_fp8_meta = None
        with_fp8_output = (
            with_fp8_compute
            and tensor_parallel_mode != "row"
            and output_fp8_meta is not None
        )

        # Tensor dims
        input_dims = input.size()
        weight_dims = weight.size()

        # Check input tensor
        if len(input_dims) == 0 or weight_dims[1] != input_dims[-1]:
            raise ValueError(
                f"Input tensor (shape={tuple(input.size())}) "
                f"and weight tensor (shape={tuple(weight.size())}) "
                "are not compatible"
            )
        x_local = reshape(
            input,
            (-1, input_dims[-1]),
            device=device,
            dtype=dtype,
        )
        if with_fp8_compute and not is_float8_tensor(x_local):
            fp8_dtype = get_fp8_te_dtype(
                input_fp8_meta["recipe"],
                fprop_tensor=True,
            )
            with_cast_transpose = requires_wgrad
            if tensor_parallel_mode == "column" and sequence_parallel:
                with_cast_transpose = False
            if with_cast_transpose:
                x_local = fp8_cast_transpose(
                    x_local,
                    fp8_meta=input_fp8_meta,
                    fp8_meta_forward=True,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                )
            else:
                x_local = Float8Tensor.to_float8(
                    x_local,
                    fp8_meta=input_fp8_meta,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                )
        elif not with_fp8_compute and is_float8_tensor(x_local):
            x_local = x_local.from_float8()
        x = x_local
        x_async = None
        if tensor_parallel_mode == "column" and sequence_parallel:
            x, x_async = gather_along_first_dim(
                x_local,
                tensor_parallel_group,
                async_op=True,
            )

        # Check weight tensor
        ### TODO: Weight caching without FP8 params
        if len(weight_dims) != 2:
            raise ValueError(
                f"Weight tensor is not 2D (shape={tuple(weight.size())})"
            )
        w = convert_tensor(
            weight,
            device=device,
            dtype=dtype,
            memory_format=torch.contiguous_format,
        )
        if with_fp8_compute and not is_float8_tensor(w):
            fp8_dtype = get_fp8_te_dtype(
                weight_fp8_meta["recipe"],
                fprop_tensor=True,
            )
            w = Float8Tensor.to_float8(
                w,
                fp8_meta=weight_fp8_meta,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
            )
        elif not with_fp8_compute and is_float8_tensor(w):
            w = w.from_float8()

        # Construct output tensor
        y = None
        if with_fp8_output:
            fp8_dtype = get_fp8_te_dtype(
                output_fp8_meta["recipe"],
                fprop_tensor=True,
            )
            data = torch.empty(
                (x.size(0), weight_dims[0]),
                dtype=torch.uint8,
                device=device,
            )
            y = Float8Tensor(
                data=data,
                fp8_meta=output_fp8_meta,
                fp8_meta_forward=True,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
                dtype=dtype,
            )
        else:
            y = torch.empty(
                (x.size(0), weight_dims[0]),
                dtype=dtype,
                device=device,
            )

        # Perform GEMM
        x_async = _wait_async(x_async)
        if with_fp8_compute:
            kwargs = dict(out=y)
            if with_fp8_output:
                fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                    forward=y._fp8_meta_forward,
                )
                kwargs.update(
                    dict(
                        out=y._data,
                        out_index=y._fp8_meta_index,
                        fp8_meta_tensor=y._fp8_meta[fp8_meta_key],
                        D_dtype=y._fp8_dtype,
                    )
                )
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
                **kwargs,
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
        if tensor_parallel_mode == "row":
            if sequence_parallel:
                y, _ = reduce_scatter_along_first_dim(y, tensor_parallel_group)
            else:
                torch.distributed.all_reduce(y, group=tensor_parallel_group)

        # Check buffer for wgrad fusion
        grad_weight = None
        if requires_wgrad and accumulate_into_main_grad:
            if not hasattr(weight, "main_grad"):
                raise RuntimeError(
                    "BasicLinear op is configured with "
                    "accumulate_into_main_grad=True, "
                    "but weight parameter does not have main_grad attribute"
                )
            grad_weight = weight.main_grad.detach()
        else:
            accumulate_into_main_grad = False

        # Save state for backward pass
        ctx.save_for_backward(
            x_local,
            weight.detach(),
            grad_weight,
        )
        ctx.device = device
        ctx.dtype = dtype
        ctx.tensor_parallel_mode = tensor_parallel_mode
        ctx.tensor_parallel_group = tensor_parallel_group
        ctx.sequence_parallel = sequence_parallel
        ctx.weight_fp8_meta = weight_fp8_meta
        ctx.grad_output_fp8_meta = grad_output_fp8_meta
        ctx.grad_input_fp8_meta = grad_input_fp8_meta
        ctx.accumulate_into_main_grad = accumulate_into_main_grad
        ctx.with_fp8_compute = with_fp8_compute
        ctx.input_dims = input_dims
        ctx.weight_dims = weight_dims
        ctx.requires_dgrad = requires_dgrad
        ctx.requires_wgrad = requires_wgrad

        # Reshape output tensor
        output_dims = list(input_dims)
        output_dims[0] = -1
        output_dims[-1] = weight_dims[0]
        return reshape(y, output_dims)

    @staticmethod
    def _functional_backward(
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:

        # Saved tensors
        x_local, weight, grad_weight = ctx.saved_tensors

        # Check if FP8 is enabled
        with_fp8_compute = ctx.with_fp8_compute
        with_fp8_grad_input = (
            with_fp8_compute
            and ctx.requires_dgrad
            and ctx.tensor_parallel_mode != "column"
            and ctx.grad_input_fp8_meta is not None
        )

        # Check grad output tensor
        dy_async = None
        dy = reshape(
            grad_output,
            (-1, grad_output.size(-1)),
            device=ctx.device,
            dtype=ctx.dtype,
        )
        if with_fp8_compute and not is_float8_tensor(dy):
            fp8_dtype = get_fp8_te_dtype(
                ctx.grad_output_fp8_meta["recipe"],
                fprop_tensor=False,
            )
            with_cast_transpose = ctx.requires_wgrad
            if ctx.tensor_parallel_mode == "row" and ctx.sequence_parallel:
                with_cast_transpose = False
            if with_cast_transpose:
                dy = fp8_cast_transpose(
                    dy,
                    fp8_meta=ctx.grad_output_fp8_meta,
                    fp8_meta_forward=False,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                )
            else:
                dy = Float8Tensor.to_float8(
                    dy,
                    fp8_meta=ctx.grad_output_fp8_meta,
                    fp8_meta_forward=False,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                )
        elif not with_fp8_compute and is_float8_tensor(dy):
            dy = dy.from_float8()
        if ctx.tensor_parallel_mode == "row" and ctx.sequence_parallel:
            dy, dy_async = gather_along_first_dim(
                dy,
                ctx.tensor_parallel_group,
                async_op=True,
            )

        # Gather sequence-parallel input if needed
        x = x_local
        x_async = None
        if (
            ctx.requires_wgrad
            and ctx.tensor_parallel_mode == "column"
            and ctx.sequence_parallel
        ):
            x, x_async = gather_along_first_dim(
                x,
                ctx.tensor_parallel_group,
                async_op=True,
            )

        # Compute grad input
        dx = None
        dx_async = None
        if ctx.requires_dgrad:

            # Check weight tensor
            ### TODO: Weight caching without FP8 params
            w = convert_tensor(
                weight,
                device=ctx.device,
                dtype=ctx.dtype,
                memory_format=torch.contiguous_format,
            )
            if with_fp8_compute and not is_float8_tensor(w):
                fp8_dtype = get_fp8_te_dtype(
                    ctx.weight_fp8_meta["recipe"],
                    fprop_tensor=True,
                )
                w = fp8_cast_transpose(
                    w,
                    fp8_meta=ctx.weight_fp8_meta,
                    fp8_meta_forward=True,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                )
            elif not with_fp8_compute and is_float8_tensor(w):
                w = w.from_float8()

            # Construct grad input tensor
            if with_fp8_grad_input:
                fp8_dtype = get_fp8_te_dtype(
                    ctx.grad_input_fp8_meta["recipe"],
                    fprop_tensor=False,
                )
                data = torch.empty(
                    (dy.size(0), ctx.weight_dims[1]),
                    dtype=torch.uint8,
                    device=ctx.device,
                )
                dx = Float8Tensor(
                    data=data,
                    fp8_meta=ctx.grad_input_fp8_meta,
                    fp8_meta_forward=False,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                    dtype=ctx.dtype,
                )
            else:
                dx = torch.empty(
                    (dy.size(0), ctx.weight_dims[1]),
                    dtype=ctx.dtype,
                    device=ctx.device,
                )

            # Perform dgrad GEMM
            dy_async = _wait_async(dy_async)
            if with_fp8_compute:
                kwargs = dict(out=dx)
                if with_fp8_grad_input:
                    fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                        forward=dx._fp8_meta_forward,
                    )
                    kwargs.update(
                        dict(
                            out=dx._data,
                            out_index=dx._fp8_meta_index,
                            fp8_meta_tensor=dx._fp8_meta[fp8_meta_key],
                            D_dtype=dx._fp8_dtype,
                        )
                    )
                fp8_gemm(
                    w.transpose_2d(cache=True),
                    w._scale_inv,
                    0,
                    w._fp8_dtype,
                    dy._data,
                    dy._scale_inv,
                    0,
                    dy._fp8_dtype,
                    dx.dtype,
                    get_workspace(),
                    **kwargs,
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
            if ctx.tensor_parallel_mode == "column":
                if ctx.sequence_parallel:
                    dx, dx_async = reduce_scatter_along_first_dim(
                        dx,
                        ctx.tensor_parallel_group,
                        async_op=True,
                    )
                else:
                    dx_async = torch.distributed.all_reduce(
                        dx,
                        group=ctx.tensor_parallel_group,
                        async_op=True,
                    )

        # Perform wgrad GEMM
        dw = None
        if ctx.requires_wgrad:
            if ctx.accumulate_into_main_grad:
                dw = grad_weight
            else:
                dw = torch.empty(
                    ctx.weight_dims,
                    dtype=ctx.dtype,
                    device=ctx.device,
                    memory_format=torch.contiguous_format,
                )
            dy_async = _wait_async(dy_async)
            x_async = _wait_async(x_async)
            if with_fp8_compute:
                fp8_gemm(
                    x.transpose_2d(cache=True),
                    x._scale_inv,
                    0,
                    x._fp8_dtype,
                    dy.transpose_2d(cache=True),
                    dy._scale_inv,
                    0,
                    dy._fp8_dtype,
                    dw.dtype,
                    get_workspace(),
                    accumulate=ctx.accumulate_into_main_grad,
                    out=dw,
                )
            else:
                gemm(
                    x,
                    dy,
                    x.dtype,
                    get_workspace(),
                    accumulate=ctx.accumulate_into_main_grad,
                    layout="NT",
                    out=dw,
            )
            if ctx.accumulate_into_main_grad:
                dw = None

        # Clean up and return grads
        _wait_async(dy_async)
        _wait_async(x_async)
        _wait_async(dx_async)
        if dx is not None:
            dx = reshape(dx, ctx.input_dims)
        return dx, dw

    def op_forward(
        self,
        ctx: OperationContext,
        input: torch.Tensor,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
    ) -> torch.Tensor:

        # FP8 metadata
        input_fp8_meta = None
        weight_fp8_meta = None
        output_fp8_meta = None
        grad_output_fp8_meta = None
        grad_input_fp8_meta = None
        if FP8GlobalStateManager.is_fp8_enabled():
            input_fp8_meta = self.get_fp8_meta("input")
            weight_fp8_meta = self.get_fp8_meta("param")
            if next_op is not None and next_op.num_fp8_scales("input") > 0:
                output_fp8_meta = next_op.get_fp8_meta("input")
            grad_output_fp8_meta = self.get_fp8_meta("grad_output")
            if prev_op is not None and prev_op.num_fp8_scales("grad_output") > 0:
                grad_input_fp8_meta = prev_op.get_fp8_meta("grad_output")

        # Call functional implementation
        return BasicLinear._functional_forward(
            ctx,
            input,
            self.weight,
            device=self.device,
            dtype=self.dtype,
            tensor_parallel_mode=self.tensor_parallel_mode,
            tensor_parallel_group=self.tensor_parallel_group,
            sequence_parallel=self.sequence_parallel,
            input_fp8_meta=input_fp8_meta,
            weight_fp8_meta=weight_fp8_meta,
            output_fp8_meta=output_fp8_meta,
            grad_output_fp8_meta=grad_output_fp8_meta,
            grad_input_fp8_meta=grad_input_fp8_meta,
            accumulate_into_main_grad=self._accumulate_into_main_grad,
        )

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:
        grad_input, grad_weight = BasicLinear._functional_backward(
            ctx,
            grad_output,
        )
        return grad_input, [grad_weight]

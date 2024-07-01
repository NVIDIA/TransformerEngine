# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for linear layer without bias."""

from __future__ import annotations
from collections.abc import Callable, Iterable
import contextlib
import math
from typing import Any, Optional

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
from transformer_engine.pytorch.module.base import get_workspace
from transformer_engine.pytorch.ops.op import (
    BasicOperation,
    OperationContext,
)
from .._common import (
    canonicalize_device,
    canonicalize_dtype,
    convert_tensor,
    is_float8_tensor,
    reshape,
)
from ...utils import clear_tensor_data


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
            raise ValueError(f"Supported dtypes are float32, float16, bfloat16 (got {dtype})")
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
        self.weight: torch.nn.Parameter
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
        if mode in ("input", "param", "grad_output"):
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
        input: torch.Tensor,  # pylint: disable=redefined-builtin
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        tensor_parallel_mode: Optional[str] = None,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        sequence_parallel: bool = False,
        with_fp8_compute: bool = False,
        input_fp8_meta: Optional[dict[str, Any]] = None,
        weight_fp8_meta: Optional[dict[str, Any]] = None,
        output_fp8_meta: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Functional API for forward pass

        Parameters
        ----------
        input: torch.Tensor
            Input tensor
        weight: torch.Tensor
            Weight tensor
        bias: torch.Tensor, optional
            Bias tensor
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
            parallelism, i.e. distributing input or output tensors
            along outer dimension (sequence or batch dim) when not
            distributing along inner dimension (embedding dim)
        with_fp8_compute: bool, default = `False`
            Whether to perform compute in FP8
        input_fp8_meta: dict, optional
            FP8 metadata for casting input tensor to FP8. Required for
            FP8 compute if input is not already in FP8.
        weight_fp8_meta: dict, optional
            FP8 metadata for casting weight tensor to FP8. Required for
            FP8 compute if weight is not already in FP8.
        output_fp8_meta: dict, optional
            FP8 metadata for casting output tensor to FP8

        Returns
        -------
        torch.Tensor
            Output tensor
        torch.Tensor
            Input tensor used in GEMM, possibly cast and reshaped from
            provided input tensor
        torch.Tensor
            Weight tensor used in GEMM, possibly cast and reshaped from
            provided weight tensor

        """

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
            raise ValueError(f"Supported dtypes are float32, float16, bfloat16 (got {dtype})")

        # Check tensor dims
        input_dims = tuple(input.size())
        weight_dims = tuple(weight.size())
        if len(weight_dims) != 2:
            raise ValueError(f"Weight tensor is not 2D (shape={weight_dims})")
        if len(input_dims) == 0 or weight_dims[1] != input_dims[-1]:
            raise ValueError(
                f"Input tensor (shape={input_dims}) "
                f"and weight tensor (shape={weight_dims}) "
                "are not compatible"
            )

        # Check if FP8 is enabled
        if with_fp8_compute:
            if input_fp8_meta is None and not is_float8_tensor(input):
                raise ValueError("No FP8 metadata was provided for casting input to FP8")
            if weight_fp8_meta is None and not is_float8_tensor(weight):
                raise ValueError("No FP8 metadata was provided for casting weight to FP8")
        else:
            input_fp8_meta = None
            weight_fp8_meta = None
            output_fp8_meta = None
        with_fp8_output = (
            with_fp8_compute and tensor_parallel_mode != "row" and output_fp8_meta is not None
        )

        # Check input tensor
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
            x_fp8 = Float8Tensor(
                data=torch.empty_like(x_local, dtype=torch.uint8),
                fp8_meta=input_fp8_meta,
                fp8_meta_forward=True,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
                fp8_scale_inv=torch.empty([1], dtype=torch.float32, device=device),
                dtype=dtype,
            )
            with_cast_transpose = weight.requires_grad
            if tensor_parallel_mode == "column" and sequence_parallel:
                with_cast_transpose = False
            if with_cast_transpose:
                x_fp8.cast_transpose_(x_local)
            else:
                x_fp8.copy_(x_local)
            x_local = x_fp8
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

        # Check bias tensor
        b = None
        if bias is not None:
            b = convert_tensor(
                bias,
                device=device,
                dtype=dtype,
                memory_format=torch.contiguous_format,
            )

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
        _wait_async(x_async)
        x_async = None
        if with_fp8_compute:
            kwargs = dict(
                out=y,
                bias=b,
                use_bias=(b is not None),
            )
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
                bias=b,
                use_bias=(b is not None),
            )

        # Reduce tensor-parallel output if needed
        if tensor_parallel_mode == "row":
            if sequence_parallel:
                y, _ = reduce_scatter_along_first_dim(y, tensor_parallel_group)
            else:
                torch.distributed.all_reduce(y, group=tensor_parallel_group)

        # Reshape output tensor
        output_dims = list(input_dims)
        output_dims[0] = -1
        output_dims[-1] = weight_dims[0]
        output = reshape(y, output_dims)

        return output, x_local, w

    @staticmethod
    def _functional_backward(
        grad_output: torch.Tensor,
        input: Optional[torch.Tensor],  # pylint: disable=redefined-builtin
        weight: Optional[torch.Tensor],
        input_dims: Iterable[int],
        weight_dims: Iterable[int],
        *,
        input_requires_grad: bool = True,
        weight_requires_grad: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        tensor_parallel_mode: Optional[str] = None,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        sequence_parallel: bool = False,
        with_fp8_compute: bool = False,
        input_fp8_meta: Optional[dict[str, Any]] = None,
        weight_fp8_meta: Optional[dict[str, Any]] = None,
        grad_output_fp8_meta: Optional[dict[str, Any]] = None,
        grad_input_fp8_meta: Optional[dict[str, Any]] = None,
        accumulate_into_grad_weight: bool = False,
        grad_weight: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Functional API for backward pass

        Parameters
        ----------
        grad_output: torch.Tensor
            Loss gradient w.r.t. output tensor
        input: torch.Tensor, optional
            Input tensor. Required to compute loss gradient w.r.t.
            weight.
        weight: torch.Tensor, optional
            Weight tensor. Required to compute loss gradient w.r.t.
            input.
        input_dims: iterable of int
            Input tensor dimensions
        weight_dims: iterable of int
            Weight tensor dimensions
        input_requires_grad: bool
            Whether to compute loss gradient w.r.t. input tensor
        weight_requires_grad: bool
            Whether to compute loss gradient w.r.t. weight tensor
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
            parallelism, i.e. distributing input or output tensors
            along outer dimension (sequence or batch dim) when not
            distributing along inner dimension (embedding dim)
        with_fp8_compute: bool, default = `False`
            Whether to perform compute in FP8
        input_fp8_meta: dict, optional
            FP8 metadata for casting input tensor to FP8. Required for
            FP8 compute if input is not already in FP8.
        weight_fp8_meta: dict, optional
            FP8 metadata for casting weight tensor to FP8. Required for
            FP8 compute if weight is not already in FP8.
        grad_output_fp8_meta: dict, optional
            FP8 metadata for casting loss gradient w.r.t. output
            tensor to FP8. Required if output grad is not already in
            FP8.
        grad_output_fp8_meta: dict, optional
            FP8 metadata for casting loss gradient w.r.t. input
            tensor to FP8
        accumulate_into_grad_weight: bool, default = `False`
            Accumulate into weight grad instead of overwriting
        grad_weight: torch.Tensor, optional
            Loss gradient w.r.t. weight tensor

        Returns
        -------
        torch.Tensor
            Loss gradient w.r.t. input tensor
        torch.Tensor
            Loss gradient w.r.t. weight tensor

        """

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
            raise ValueError(f"Supported dtypes are float32, float16, bfloat16 (got {dtype})")

        # Check tensor dims
        output_dims = tuple(grad_output.size())
        input_dims = tuple(input_dims)
        weight_dims = tuple(weight_dims)
        if len(weight_dims) != 2:
            raise ValueError(f"Weight tensor is not 2D (shape={weight_dims})")
        if len(input_dims) == 0 or weight_dims[1] != input_dims[-1]:
            raise ValueError(
                f"Input tensor (shape={input_dims}) "
                f"and weight tensor (shape={weight_dims}) "
                "are not compatible"
            )
        if weight_dims[0] != output_dims[-1]:
            raise ValueError(
                f"Grad output tensor (shape={output_dims}) "
                f"and weight tensor (shape={weight_dims}) "
                "are not compatible"
            )

        # Check if FP8 is enabled
        if with_fp8_compute:
            if grad_output_fp8_meta is None and not is_float8_tensor(grad_output):
                raise ValueError("No FP8 metadata was provided for casting output gradient to FP8")
        else:
            input_fp8_meta = None
            weight_fp8_meta = None
            grad_output_fp8_meta = None
            grad_input_fp8_meta = None
        with_fp8_grad_input = (
            with_fp8_compute
            and input_requires_grad
            and tensor_parallel_mode != "column"
            and grad_input_fp8_meta is not None
        )

        # Check grad output tensor
        dy_async = None
        dy = reshape(
            grad_output,
            (-1, output_dims[-1]),
            device=device,
            dtype=dtype,
        )
        if with_fp8_compute and not is_float8_tensor(dy):
            fp8_dtype = get_fp8_te_dtype(
                grad_output_fp8_meta["recipe"],
                fprop_tensor=False,
            )
            dy_fp8 = Float8Tensor(
                data=torch.empty_like(dy, dtype=torch.uint8),
                fp8_meta=grad_output_fp8_meta,
                fp8_meta_forward=False,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
                fp8_scale_inv=torch.empty([1], dtype=torch.float32, device=device),
                dtype=dtype,
            )
            with_cast_transpose = weight_requires_grad
            if tensor_parallel_mode == "row" and sequence_parallel:
                with_cast_transpose = False
            if with_cast_transpose:
                dy_fp8.cast_transpose_(dy)
            else:
                dy_fp8.copy_(dy)
            dy = dy_fp8
        elif not with_fp8_compute and is_float8_tensor(dy):
            dy = dy.from_float8()
        if tensor_parallel_mode == "row" and sequence_parallel:
            dy, dy_async = gather_along_first_dim(
                dy,
                tensor_parallel_group,
                async_op=True,
            )

        # Check input tensor
        x = None
        x_async = None
        if weight_requires_grad:
            if input is None:
                raise ValueError("Input tensor is required to compute weight grad")
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
                x_fp8 = Float8Tensor(
                    data=torch.empty_like(x_local, dtype=torch.uint8),
                    fp8_meta=input_fp8_meta,
                    fp8_meta_forward=True,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                    fp8_scale_inv=torch.empty([1], dtype=torch.float32, device=device),
                    dtype=dtype,
                )
                x_fp8.cast_transpose_(x_local)
                x_local = x_fp8
            elif not with_fp8_compute and is_float8_tensor(x_local):
                x_local = x_local.from_float8()
            x = x_local
            if tensor_parallel_mode == "column" and sequence_parallel:
                x, x_async = gather_along_first_dim(
                    x_local,
                    tensor_parallel_group,
                    async_op=True,
                )

        # Compute grad input
        dx = None
        dx_async = None
        if input_requires_grad:

            # Check weight tensor
            if weight is None:
                raise ValueError("Weight tensor is required to compute input grad")
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
                w_fp8 = Float8Tensor(
                    data=torch.empty_like(w, dtype=torch.uint8),
                    fp8_meta=weight_fp8_meta,
                    fp8_meta_forward=True,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                    fp8_scale_inv=torch.empty([1], dtype=torch.float32, device=device),
                    dtype=dtype,
                )
                w_fp8.cast_transpose_(w)
                w = w_fp8
            elif not with_fp8_compute and is_float8_tensor(w):
                w = w.from_float8()

            # Construct grad input tensor
            if with_fp8_grad_input:
                fp8_dtype = get_fp8_te_dtype(
                    grad_input_fp8_meta["recipe"],
                    fprop_tensor=False,
                )
                data = torch.empty(
                    (dy.size(0), weight_dims[1]),
                    dtype=torch.uint8,
                    device=device,
                )
                dx = Float8Tensor(
                    data=data,
                    fp8_meta=grad_input_fp8_meta,
                    fp8_meta_forward=False,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                    dtype=dtype,
                )
            else:
                dx = torch.empty(
                    (dy.size(0), weight_dims[1]),
                    dtype=dtype,
                    device=device,
                )

            # Perform dgrad GEMM
            _wait_async(dy_async)
            dy_async = None
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
                    w.transpose_2d(),
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
            if tensor_parallel_mode == "column":
                if sequence_parallel:
                    dx, dx_async = reduce_scatter_along_first_dim(
                        dx,
                        tensor_parallel_group,
                        async_op=True,
                    )
                else:
                    dx_async = torch.distributed.all_reduce(
                        dx,
                        group=tensor_parallel_group,
                        async_op=True,
                    )

        # Perform wgrad GEMM
        if not weight_requires_grad:
            grad_weight = None
        else:
            if grad_weight is None:
                if accumulate_into_grad_weight:
                    raise ValueError(
                        "Attempted to accumulate into grad weight buffer"
                        "without providing grad weight"
                    )
                grad_weight = torch.empty(
                    weight_dims,
                    dtype=dtype,
                    device=device,
                    memory_format=torch.contiguous_format,
                )
            _wait_async(dy_async)
            _wait_async(x_async)
            dy_async = None
            x_async = None
            if with_fp8_compute:
                fp8_gemm(
                    x.transpose_2d(),
                    x._scale_inv,
                    0,
                    x._fp8_dtype,
                    dy.transpose_2d(),
                    dy._scale_inv,
                    0,
                    dy._fp8_dtype,
                    grad_weight.dtype,
                    get_workspace(),
                    accumulate=accumulate_into_grad_weight,
                    out=grad_weight,
                )
            else:
                gemm(
                    x,
                    dy,
                    x.dtype,
                    get_workspace(),
                    accumulate=accumulate_into_grad_weight,
                    layout="NT",
                    out=grad_weight,
                )

        # Clean up and return grads
        _wait_async(dy_async)
        _wait_async(x_async)
        _wait_async(dx_async)
        grad_input = None
        if dx is not None:
            grad_input = reshape(dx, input_dims)
        return grad_input, grad_weight

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
    ) -> torch.Tensor:

        # FP8 metadata
        with_fp8_compute = FP8GlobalStateManager.is_fp8_enabled()
        input_fp8_meta = None
        weight_fp8_meta = None
        output_fp8_meta = None
        grad_output_fp8_meta = None
        grad_input_fp8_meta = None
        if with_fp8_compute:
            input_fp8_meta = self.get_fp8_meta("input")
            weight_fp8_meta = self.get_fp8_meta("param")
            if next_op is not None and next_op.num_fp8_scales("input") > 0:
                output_fp8_meta = next_op.get_fp8_meta("input")
            grad_output_fp8_meta = self.get_fp8_meta("grad_output")
            if prev_op is not None and prev_op.num_fp8_scales("grad_output") > 0:
                grad_input_fp8_meta = prev_op.get_fp8_meta("grad_output")

        # Linear forward
        output, x_local, _ = BasicLinear._functional_forward(
            input=input_,
            weight=self.weight,
            device=self.device,
            dtype=self.dtype,
            tensor_parallel_mode=self.tensor_parallel_mode,
            tensor_parallel_group=self.tensor_parallel_group,
            sequence_parallel=self.sequence_parallel,
            with_fp8_compute=with_fp8_compute,
            input_fp8_meta=input_fp8_meta,
            weight_fp8_meta=weight_fp8_meta,
            output_fp8_meta=output_fp8_meta,
        )

        # Save state for backward pass
        ctx.save_for_backward(x_local)
        ctx.with_fp8_compute = with_fp8_compute
        ctx.weight_fp8_meta = weight_fp8_meta
        ctx.grad_output_fp8_meta = grad_output_fp8_meta
        ctx.grad_input_fp8_meta = grad_input_fp8_meta
        ctx.input_dims = input_.size()
        ctx.input_requires_grad = input_.requires_grad
        ctx.weight_requires_grad = self.weight.requires_grad
        ctx.has_prev_op = prev_op is not None

        return output

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:

        # Saved tensors from forward pass
        (x_local,) = ctx.saved_tensors

        # wgrad fusion
        accumulate_into_main_grad = self._accumulate_into_main_grad
        grad_weight = None
        if ctx.weight_requires_grad and accumulate_into_main_grad:
            if not hasattr(self.weight, "main_grad"):
                raise RuntimeError(
                    "BasicLinear op is configured with "
                    "accumulate_into_main_grad=True, "
                    "but weight parameter does not have main_grad attribute"
                )
            grad_weight = self.weight.main_grad.detach()
        else:
            accumulate_into_main_grad = False

        # Linear backward pass
        grad_input, grad_weight = BasicLinear._functional_backward(
            grad_output=grad_output,
            input=x_local,
            weight=self.weight,
            input_dims=ctx.input_dims,
            weight_dims=self.weight.size(),
            input_requires_grad=ctx.input_requires_grad,
            weight_requires_grad=ctx.weight_requires_grad,
            device=self.device,
            dtype=self.dtype,
            tensor_parallel_mode=self.tensor_parallel_mode,
            tensor_parallel_group=self.tensor_parallel_group,
            sequence_parallel=self.sequence_parallel,
            with_fp8_compute=ctx.with_fp8_compute,
            weight_fp8_meta=ctx.weight_fp8_meta,
            grad_output_fp8_meta=ctx.grad_output_fp8_meta,
            grad_input_fp8_meta=ctx.grad_input_fp8_meta,
            accumulate_into_grad_weight=accumulate_into_main_grad,
            grad_weight=grad_weight,
        )

        # Clear input tensor if possible
        if ctx.has_prev_op:
            clear_tensor_data(x_local)

        if accumulate_into_main_grad:
            grad_weight = None
        return grad_input, [grad_weight]

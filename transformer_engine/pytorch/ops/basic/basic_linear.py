# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for linear layer without bias."""

from __future__ import annotations
from collections.abc import Callable, Iterable
import contextlib
import math
from typing import Any, Optional

import torch

from transformer_engine.pytorch.module.base import get_workspace
from ...cpp_extensions import general_gemm
from ...distributed import (
    CudaRNGStatesTracker,
    gather_along_first_dim,
    reduce_scatter_along_first_dim,
)
from ...fp8 import FP8GlobalStateManager
from ...module.base import _2X_ACC_FPROP, _2X_ACC_DGRAD, _2X_ACC_WGRAD
from ...tensor import Quantizer, QuantizedTensor
from ...tensor.float8_tensor import Float8Quantizer
from ...tensor.mxfp8_tensor import MXFP8Quantizer
from ...tensor._internal.float8_tensor_base import Float8TensorBase
from ..op import BasicOperation, OperationContext
from .._common import (
    canonicalize_device,
    canonicalize_dtype,
    devices_match,
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
    userbuffers_options, dict, optional
        Options for overlapping tensor-parallel communication with
        compute using Userbuffers. This feature is highly
        experimental.

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
        userbuffers_options: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        # Weight tensor dimensions
        self.in_features: int = in_features
        self.out_features: int = out_features

        # Weight tensor attributes
        device = canonicalize_device(device)
        dtype = canonicalize_dtype(dtype)
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Supported dtypes are float32, float16, bfloat16 (got {dtype})")

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

        # Whether weight tensor is natively quantized
        self._with_quantized_weight: bool = FP8GlobalStateManager.with_fp8_parameters()

        # Initialize parameters if needed
        weight = torch.empty(
            self.local_out_features,
            self.local_in_features,
            device=device,
            dtype=dtype,
        )
        weight = torch.nn.Parameter(weight)
        self.weight: torch.nn.Parameter
        self.register_parameter("weight", weight)
        self._rng_state_tracker_function: Optional[Callable[[], CudaRNGStatesTracker]]
        self._rng_state_tracker_function = rng_state_tracker_function
        if weight.device.type != "meta":
            self.reset_parameters()

        # Whether to accumulate weight gradient into main_grad
        self._accumulate_into_main_grad: bool = accumulate_into_main_grad

        # Userbuffers options
        self._userbuffers_options: Optional[dict[str, Any]] = userbuffers_options

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

    def num_quantizers(self, mode: str) -> int:
        if mode == "forward":
            return 2
        if mode == "backward":
            return 1
        return 0

    def reset_parameters(self) -> None:
        """Initialize parameter buffers and values"""

        # Parameter device
        weight = self.weight
        device = weight.device
        if device.type == "meta":
            device = canonicalize_device(None)

        # Allocate buffer if needed
        if isinstance(weight, QuantizedTensor):
            weight = torch.empty(
                weight.size(),
                dtype=weight.dtype,
                device=device,
            )
        elif not devices_match(weight.device, device):
            weight = torch.empty_like(weight, device=device)

        # Initialize values
        init_context = contextlib.nullcontext()
        if self._rng_state_tracker_function is not None:
            init_context = self._rng_state_tracker_function().fork()
        with init_context:
            torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        # Quantize if needed
        if self._with_quantized_weight:
            quantizer = self.get_quantizer("forward", 1)
            quantizer.set_usage(
                rowwise=True,
                columnwise=torch.is_grad_enabled(),
            )
            with torch.no_grad():
                weight = quantizer(weight)

        # Save updated parameter
        if not isinstance(weight, torch.nn.Parameter):
            weight = torch.nn.Parameter(weight)
        self.weight = weight

    def pre_forward(self, *args, **kwargs) -> None:
        super().pre_forward(*args, **kwargs)

        # Initialize weights if needed
        weight = self.weight
        if weight.device.type == "meta":
            self.reset_parameters()
            weight = self.weight

        # Configure quantizers
        if FP8GlobalStateManager.is_fp8_enabled():
            input_quantizer = self.get_quantizer("forward", 0)
            weight_quantizer = self.get_quantizer("forward", 1)
            grad_output_quantizer = self.get_quantizer("backward", 0)

            # Specify required tensor formats
            is_grad_enabled = torch.is_grad_enabled()
            weight_requires_grad = is_grad_enabled and weight.requires_grad
            input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
            weight_quantizer.set_usage(rowwise=True, columnwise=is_grad_enabled)
            grad_output_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)

            # Make sure weight tensor has correct quantizer
            # Note: Quantizer might have changed if quantization
            # recipe changed
            if isinstance(weight_quantizer, Float8Quantizer) and isinstance(
                weight, Float8TensorBase
            ):
                weight._quantizer = weight_quantizer

    @staticmethod
    def _functional_forward(
        input: torch.Tensor,  # pylint: disable=redefined-builtin
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,  # pylint: disable=unused-argument
        dtype: Optional[torch.dtype] = None,
        out: Optional[torch.Tensor] = None,
        accumulate_into_out: bool = False,
        tensor_parallel_mode: Optional[str] = None,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        sequence_parallel: bool = False,
        with_quantized_compute: bool = False,
        input_quantizer: Optional[Quantizer] = None,
        weight_quantizer: Optional[Quantizer] = None,
        output_quantizer: Optional[Quantizer] = None,
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
        out: torch.Tensor, optional
            Output tensor
        accumulate_into_out: bool, default = `False`
            Add result to output tensor instead of overwriting
        tensor_parallel_mode: {`None`, "column", "row"}, default = `None`
            Mode for tensor parallelism
        tensor_parallel_group: torch.distributed.ProcessGroup, default = world group
            Process group for tensor parallelism
        sequence_parallel: bool, default = `False`
            Whether to apply sequence parallelism together with tensor
            parallelism, i.e. distributing input or output tensors
            along outer dimension (sequence or batch dim) when not
            distributing along inner dimension (embedding dim)
        with_quantized_compute: bool, default = `False`
            Whether to perform compute with quantized data.
        input_quantizer: Quantizer, optional
            Builder class for quantized input tensor.
        weight_quantizer: Quantizer, optional
            Builder class for quantized weight tensor.
        output_quantizer: Quantizer, optional
            Builder class for quantized output tensor.

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

        # Check datatype
        if dtype is None:
            dtype = weight.dtype if out is None else out.dtype
        dtype = canonicalize_dtype(dtype)
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Supported dtypes are float32, float16, bfloat16 (got {dtype})")
        if out is not None and out.dtype != dtype:
            raise ValueError(f"Output tensor has invalid dtype (expected {dtype}, got {out.dtype})")

        # Check input tensor
        x_local = input
        x = None
        x_async = None
        with_x_all_gather = tensor_parallel_mode == "column" and sequence_parallel
        own_quantized_x_local = False
        if with_quantized_compute:
            if input_quantizer is None:
                raise ValueError("Missing quantizer for input tensor")
            input_quantizer.set_usage(rowwise=True)
            if with_x_all_gather:
                input_quantizer.set_usage(columnwise=False)
                x, x_async = gather_along_first_dim(
                    x_local,
                    tensor_parallel_group,
                    async_op=True,
                    quantizer=input_quantizer,
                )
            else:
                if not isinstance(x_local, QuantizedTensor):
                    x_local = input_quantizer(x_local)
                    own_quantized_x_local = True
                x = x_local
        else:
            if isinstance(x_local, QuantizedTensor):
                x_local = x_local.dequantize()
            if x_local.dtype != dtype:
                x_local = x_local.to(dtype=dtype)
            if with_x_all_gather:
                x, x_async = gather_along_first_dim(
                    x_local,
                    tensor_parallel_group,
                    async_op=True,
                )
            else:
                x = x_local

        # Check weight tensor
        w = weight
        w_is_quantized = isinstance(w, QuantizedTensor)
        if with_quantized_compute and not w_is_quantized:
            if weight_quantizer is None:
                raise ValueError("Missing quantizer for weight tensor")
            weight_quantizer.set_usage(rowwise=True)
            w = weight_quantizer(w)
        elif not with_quantized_compute and w_is_quantized:
            w = w.dequantize()
        if not with_quantized_compute and w.dtype != dtype:
            w = w.to(dtype=dtype)

        # Check output tensor
        y = out
        if y is None:
            if not with_quantized_compute:
                output_quantizer = None
            if tensor_parallel_mode == "row":
                output_quantizer = None
        elif isinstance(y, QuantizedTensor):
            if not with_quantized_compute:
                raise ValueError("Output tensor is quantized, but quantized compute is not enabled")
            if tensor_parallel_mode == "row":
                raise ValueError(
                    "Output tensor is quantized, "
                    "but row tensor parallelism does not support quantized output"
                )
            if output_quantizer is None:
                output_quantizer = getattr(y, "_quantizer", None)
            if output_quantizer is None:
                raise ValueError("Output tensor is quantized, but quantizer was not provided")
        else:
            output_quantizer = None
        if isinstance(output_quantizer, MXFP8Quantizer):
            raise RuntimeError(
                "Attempting to generate MXFP8 output tensor, "
                "but GEMM with MXFP8 output is not supported"
            )
        if output_quantizer is not None:
            output_quantizer.set_usage(rowwise=True, columnwise=False)

        # Check if accumulating into output tensor
        if accumulate_into_out:
            if y is None:
                raise ValueError(
                    "Attempted to accumulate into output tensor without providing output tensor"
                )
            if tensor_parallel_mode == "row":
                raise ValueError(
                    "Accumulating into output tensor is not supported with row tensor parallelism"
                )

        # Synchronize communication for input
        _wait_async(x_async)
        x_async = None

        # Perform GEMM
        y, *_ = general_gemm(
            w,
            x,
            get_workspace(),
            out_dtype=dtype,
            quantization_params=output_quantizer,
            accumulate=accumulate_into_out,
            out=y,
            bias=bias,
            use_split_accumulator=_2X_ACC_FPROP,
        )

        # Reduce tensor-parallel output if needed
        if tensor_parallel_mode == "row":
            if sequence_parallel:
                y, _ = reduce_scatter_along_first_dim(y, tensor_parallel_group)
            else:
                torch.distributed.all_reduce(y, group=tensor_parallel_group)

        # Configure input tensor for backward pass
        if own_quantized_x_local:
            ### TODO Restore once column-wise usage is supported by itself  # pylint: disable=fixme
            # x_local.update_usage(rowwise_usage=False)
            pass

        # Detach input tensor if needed
        # Note: PyTorch autograd produces esoteric errors if we save
        # input tensor as context for backward pass.
        if x_local is input:
            x_local = x_local.detach()

        return y, x_local, w

    @staticmethod
    def _functional_backward(
        grad_output: torch.Tensor,
        input: Optional[torch.Tensor],  # pylint: disable=redefined-builtin
        weight: Optional[torch.Tensor],
        *,
        input_requires_grad: bool = True,
        weight_requires_grad: bool = True,
        device: Optional[torch.device] = None,  # pylint: disable=unused-argument
        dtype: Optional[torch.dtype] = None,
        grad_weight: Optional[torch.Tensor] = None,
        accumulate_into_grad_weight: bool = False,
        grad_input: Optional[torch.Tensor] = None,
        accumulate_into_grad_input: bool = False,
        tensor_parallel_mode: Optional[str] = None,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        sequence_parallel: bool = False,
        with_quantized_compute: bool = False,
        input_quantizer: Optional[Quantizer] = None,
        weight_quantizer: Optional[Quantizer] = None,
        grad_output_quantizer: Optional[Quantizer] = None,
        grad_input_quantizer: Optional[Quantizer] = None,
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
        input_requires_grad: bool
            Whether to compute loss gradient w.r.t. input tensor
        weight_requires_grad: bool
            Whether to compute loss gradient w.r.t. weight tensor
        device: torch.device, default = default CUDA device
            Tensor device
        dtype: torch.dtype, default = default dtype
            Tensor datatype
        grad_weight: torch.Tensor, optional
            Loss gradient w.r.t. weight tensor
        accumulate_into_grad_weight: bool, default = `False`
            Add result to weight grad instead of overwriting
        grad_input: torch.Tensor, optional
            Loss gradient w.r.t. input tensor
        accumulate_into_grad_input: bool, default = `False`
            Add result to input grad instead of overwriting
        tensor_parallel_mode: {`None`, "column", "row"}, default = `None`
            Mode for tensor parallelism
        tensor_parallel_group: torch.distributed.ProcessGroup, default = world group
            Process group for tensor parallelism
        sequence_parallel: bool, default = `False`
            Whether to apply sequence parallelism together with tensor
            parallelism, i.e. distributing input or output tensors
            along outer dimension (sequence or batch dim) when not
            distributing along inner dimension (embedding dim)
        with_quantized_compute: bool, default = `False`
            Whether to perform compute with quantized data.
        input_quantizer: Quantizer, optional
            Builder class for quantized input tensor.
        weight_quantizer: Quantizer, optional
            Builder class for quantized weight tensor.
        grad_output_quantizer: Quantizer, optional
            Builder class for quantized loss gradient w.r.t. output
            tensor.
        grad_input_quantizer: dict, optional
            Builder class for quantized loss gradient w.r.t. input
            tensor.

        Returns
        -------
        torch.Tensor
            Loss gradient w.r.t. input tensor
        torch.Tensor
            Loss gradient w.r.t. weight tensor

        """

        # Check datatype
        if dtype is None:
            dtype = weight.dtype
        dtype = canonicalize_dtype(dtype)
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Supported dtypes are float32, float16, bfloat16 (got {dtype})")

        # Check grad output tensor
        dy_local = grad_output
        dy = None
        dy_async = None
        with_dy_all_gather = tensor_parallel_mode == "row" and sequence_parallel
        if with_quantized_compute:
            if grad_output_quantizer is None:
                raise ValueError("Missing quantizer for grad output tensor")
            grad_output_quantizer.set_usage(
                rowwise=input_requires_grad,
                columnwise=weight_requires_grad,
            )
            if with_dy_all_gather:
                dy, dy_async = gather_along_first_dim(
                    dy_local,
                    tensor_parallel_group,
                    async_op=True,
                    quantizer=grad_output_quantizer,
                )
            else:
                if not isinstance(dy_local, QuantizedTensor):
                    dy_local = grad_output_quantizer(dy_local)
                dy = dy_local
        else:
            if isinstance(dy_local, QuantizedTensor):
                dy_local = dy_local.dequantize()
            if dy_local.dtype != dtype:
                dy_local = dy_local.to(dtype=dtype)
            if with_dy_all_gather:
                dy, dy_async = gather_along_first_dim(
                    dy_local,
                    tensor_parallel_group,
                    async_op=True,
                )
            else:
                dy = dy_local

        # Check input tensor
        x = None
        x_async = None
        if weight_requires_grad:
            if input is None:
                raise ValueError("Input tensor is required to compute weight grad")
            x_local = input
            with_x_all_gather = tensor_parallel_mode == "column" and sequence_parallel
            if with_quantized_compute:
                if input_quantizer is None:
                    raise ValueError("Missing quantizer for input tensor")
                input_quantizer.set_usage(columnwise=True)
                if with_x_all_gather:
                    x, x_async = gather_along_first_dim(
                        x_local,
                        tensor_parallel_group,
                        async_op=True,
                        quantizer=input_quantizer,
                    )
                else:
                    if not isinstance(x_local, QuantizedTensor):
                        x_local = input_quantizer(x_local)
                    x = x_local
            else:
                if isinstance(x_local, QuantizedTensor):
                    x_local = x_local.dequantize()
                if x_local.dtype != dtype:
                    x_local = x_local.to(dtype=dtype)
                if with_x_all_gather:
                    x, x_async = gather_along_first_dim(
                        x_local,
                        tensor_parallel_group,
                        async_op=True,
                    )
                else:
                    x = x_local

        # Compute grad input
        dx = None
        dx_async = None
        if input_requires_grad:

            # Check weight tensor
            if weight is None:
                raise ValueError("Weight tensor is required to compute input grad")
            w = weight
            w_is_quantized = isinstance(w, QuantizedTensor)
            if with_quantized_compute and not w_is_quantized:
                if weight_quantizer is None:
                    raise ValueError("Missing quantizer for weight tensor")
                weight_quantizer.set_usage(columnwise=True)
                w = weight_quantizer(w)
            elif not with_quantized_compute and w_is_quantized:
                w = w.dequantize()
            if not with_quantized_compute and w.dtype != dtype:
                w = w.to(dtype=dtype)

            # Synchronize tensor-parallel communication
            _wait_async(dy_async)
            dy_async = None

            # Check grad input tensor
            dx = grad_input
            if dx is None:
                if not with_quantized_compute:
                    grad_input_quantizer = None
                if tensor_parallel_mode == "column":
                    grad_input_quantizer = None
            elif isinstance(dx, QuantizedTensor):
                if not with_quantized_compute:
                    raise ValueError(
                        "Grad input tensor is quantized, but quantized compute is not enabled"
                    )
                if tensor_parallel_mode == "column":
                    raise ValueError(
                        "Grad input tensor is quantized, "
                        "but column tensor parallelism does not support quantized grad input"
                    )
                if grad_input_quantizer is None:
                    grad_input_quantizer = getattr(dx, "_quantizer", None)
                if grad_input_quantizer is None:
                    raise ValueError(
                        "Grad input tensor is quantized, but quantizer was not provided"
                    )
            else:
                grad_input_quantizer = None
            if isinstance(grad_input_quantizer, MXFP8Quantizer):
                raise RuntimeError(
                    "Attempting to generate MXFP8 grad input tensor, "
                    "but GEMM with MXFP8 output is not supported"
                )

            # Check if accumulating into grad input tensor
            if accumulate_into_grad_input:
                if dx is None:
                    raise ValueError(
                        "Attempted to accumulate into grad input tensor "
                        "without providing grad input tensor"
                    )
                if tensor_parallel_mode == "column":
                    raise ValueError(
                        "Accumulating into grad input tensor "
                        "is not supported with column tensor parallelism"
                    )

            # Perform dgrad GEMM
            dx, *_ = general_gemm(
                w,
                dy,
                get_workspace(),
                out_dtype=dtype,
                quantization_params=grad_input_quantizer,
                accumulate=accumulate_into_grad_input,
                layout="NN",
                out=dx,
                use_split_accumulator=_2X_ACC_DGRAD,
                grad=True,
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

        # Compute grad weight
        dw = None
        if weight_requires_grad:

            # Synchronize tensor-parallel communication
            _wait_async(x_async)
            _wait_async(dy_async)
            x_async = None
            dy_async = None

            # Check grad input tensor
            dw = grad_weight
            dw_dtype = dtype
            if dw is None:
                if accumulate_into_grad_weight:
                    raise ValueError(
                        "Attempted to accumulate into grad weight tensor "
                        "without providing grad weight tensor"
                    )
            else:
                dw_dtype = dw.dtype

            # Perform wgrad GEMM
            dw, *_ = general_gemm(
                x,
                dy,
                get_workspace(),
                out_dtype=dw_dtype,
                accumulate=accumulate_into_grad_weight,
                layout="NT",
                out=dw,
                use_split_accumulator=_2X_ACC_WGRAD,
                grad=True,
            )

        # Clean up and return grads
        _wait_async(dy_async)
        _wait_async(x_async)
        _wait_async(dx_async)
        return dx, dw

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op: Optional[BasicOperation] = None,
        next_op: Optional[BasicOperation] = None,
    ) -> torch.Tensor:

        # Check which grads are required
        input_requires_grad = ctx.requires_grad and input_.requires_grad
        weight_requires_grad = ctx.requires_grad and self.weight.requires_grad

        # FP8 metadata
        with_quantized_compute = FP8GlobalStateManager.is_fp8_enabled()
        input_quantizer = None
        weight_quantizer = None
        output_quantizer = None
        grad_output_quantizer = None
        grad_input_quantizer = None
        if with_quantized_compute:

            # Get quantizers
            input_quantizer = self.get_quantizer("forward", 0)
            weight_quantizer = self.get_quantizer("forward", 1)
            if next_op is not None and next_op.num_quantizers("forward") > 0:
                output_quantizer = next_op.get_quantizer("forward", 0)
            grad_output_quantizer = self.get_quantizer("backward", 0)
            if prev_op is not None and prev_op.num_quantizers("backward") > 0:
                grad_input_quantizer = prev_op.get_quantizer("backward", 0)

            # Configure quantizers
            # Note: We cache the quantized input for backward pass,
            # but discard the quantized weights.
            input_quantizer.set_usage(columnwise=weight_requires_grad)
            weight_quantizer.set_usage(columnwise=False)

        # Get autocast dtype if needed
        dtype = None
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")

        # Linear forward
        output, x_local, _ = BasicLinear._functional_forward(
            input=input_,
            weight=self.weight,
            dtype=dtype,
            tensor_parallel_mode=self.tensor_parallel_mode,
            tensor_parallel_group=self.tensor_parallel_group,
            sequence_parallel=self.sequence_parallel,
            with_quantized_compute=with_quantized_compute,
            input_quantizer=input_quantizer,
            weight_quantizer=weight_quantizer,
            output_quantizer=output_quantizer,
        )

        # Save state for backward pass
        ctx.save_for_backward(x_local)
        ctx.with_quantized_compute = with_quantized_compute
        ctx.input_quantizer = input_quantizer
        ctx.weight_quantizer = weight_quantizer
        ctx.grad_output_quantizer = grad_output_quantizer
        ctx.grad_input_quantizer = grad_input_quantizer
        ctx.dtype = dtype
        ctx.input_requires_grad = input_requires_grad
        ctx.weight_requires_grad = weight_requires_grad
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
            input_requires_grad=ctx.input_requires_grad,
            weight_requires_grad=ctx.weight_requires_grad,
            dtype=ctx.dtype,
            grad_weight=grad_weight,
            accumulate_into_grad_weight=accumulate_into_main_grad,
            tensor_parallel_mode=self.tensor_parallel_mode,
            tensor_parallel_group=self.tensor_parallel_group,
            sequence_parallel=self.sequence_parallel,
            with_quantized_compute=ctx.with_quantized_compute,
            input_quantizer=ctx.input_quantizer,
            weight_quantizer=ctx.weight_quantizer,
            grad_output_quantizer=ctx.grad_output_quantizer,
            grad_input_quantizer=ctx.grad_input_quantizer,
        )

        # Clear input tensor if possible
        if ctx.has_prev_op:
            clear_tensor_data(x_local)

        if accumulate_into_main_grad:
            grad_weight = None
        return grad_input, [grad_weight]

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

from ...cpp_extensions import general_gemm
from ...distributed import (
    CudaRNGStatesTracker,
    gather_along_first_dim,
    reduce_scatter_along_first_dim,
)
from ...fp8 import FP8GlobalStateManager, Recipe
from ...module.base import (
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
    get_dummy_wgrad,
    get_workspace,
)
from ...tensor import Quantizer
from ...tensor.float8_tensor import Float8Quantizer
from ...tensor._internal.float8_tensor_base import Float8TensorBase
from ...utils import (
    canonicalize_device,
    canonicalize_dtype,
    clear_tensor_data,
    devices_match,
)
from ..op import BasicOperation, OperationContext
from .._common import maybe_dequantize, is_quantized_tensor


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
        meaningful. This is primarily intented to integrate with
        Megatron-LM.
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
        if is_quantized_tensor(weight):
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
            if quantizer is None:
                raise RuntimeError(
                    "Tried to quantize weight with deferred initialization "
                    "due to meta device, but no quantizer was available. "
                    "This is most likely because the weight was initialized "
                    "within fp8_model_init, but the forward pass was not "
                    "performed within fp8_autocast."
                )
            quantizer.set_usage(
                rowwise=True,
                columnwise=torch.is_grad_enabled(),
            )
            quantizer.internal = False
            with torch.no_grad():
                weight = quantizer(weight)

        # Save updated parameter
        if not isinstance(weight, torch.nn.Parameter):
            weight = torch.nn.Parameter(weight)
        self.weight = weight

    def pre_first_fuser_forward(self) -> None:
        super().pre_first_fuser_forward()
        if self.weight.device.type == "meta":
            self.reset_parameters()

    def reset_recipe_state(self, *, recipe: Optional[Recipe]) -> None:
        super().reset_recipe_state(recipe=recipe)

        # Input/grad output quantizers use internal tensors
        input_quantizer = self.get_quantizer("forward", 0)
        grad_output_quantizer = self.get_quantizer("backward", 0)
        if input_quantizer is not None:
            input_quantizer.internal = True
        if grad_output_quantizer is not None:
            grad_output_quantizer.internal = True

        # Handle weight quantizer
        # Note: This function may be called in base class constructor,
        # before any basic linear attrs have been set.
        weight_quantizer = self.get_quantizer("forward", 1)
        if weight_quantizer is None:
            pass
        elif is_quantized_tensor(getattr(self, "weight", None)):
            # Make sure weight param has correct quantizer
            weight_quantizer.set_usage(rowwise=True, columnwise=torch.is_grad_enabled())
            weight_quantizer.internal = False
            self.weight.update_quantizer(weight_quantizer.copy())
        else:
            # Use internal tensors if quantized weights will not be
            # exposed externally
            weight_quantizer.internal = (
                not FP8GlobalStateManager.with_fp8_parameters()
                and not getattr(self, "_with_quantized_weight", False)
            )

    @staticmethod
    def _functional_forward(
        input: torch.Tensor,  # pylint: disable=redefined-builtin
        weight: torch.Tensor,
        *,
        alpha: float = 1.0,
        bias: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,  # pylint: disable=unused-argument
        dtype: Optional[torch.dtype] = None,
        out: Optional[torch.Tensor] = None,
        beta: Optional[float] = None,
        accumulate_into_out: bool = False,
        tensor_parallel_mode: Optional[str] = None,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        sequence_parallel: bool = False,
        with_quantized_compute: bool = False,
        input_quantizer: Optional[Quantizer] = None,
        weight_quantizer: Optional[Quantizer] = None,
        output_quantizer: Optional[Quantizer] = None,
        input_requires_grad: bool = True,
        weight_requires_grad: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Functional API for forward pass

        Parameters
        ----------
        input: torch.Tensor
            Input tensor
        weight: torch.Tensor
            Weight tensor
        alpha: float, default = 1.0
            Scaling factor applied to the result of the GEMM
        bias: torch.Tensor, optional
            Bias tensor
        device: torch.device, default = default CUDA device
            Tensor device
        dtype: torch.dtype, default = infer from out or weight
            Tensor datatype
        out: torch.Tensor, optional
            Output tensor
        beta: float, optional
            Scaling factor applied to original value of out when accumulating into it
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
        input_requires_grad: bool, default = `True`
            Whether the loss gradient w.r.t. the input tensor is
            required in the backward pass.
        weight_requires_grad: bool, default = `True`
            Whether the loss gradient w.r.t. the weight tensor is
            required in the backward pass.

        Returns
        -------
        torch.Tensor
            Output tensor
        torch.Tensor, optional
            Input tensor, ready for use in backward pass. `None` is
            returned if loss gradient w.r.t. the weight tensor is not
            required.
        torch.Tensor, optional
            Weight tensor, ready for use in backward pass. `None` is
            returned if loss gradient w.r.t. the input tensor is not
            required.

        """

        # Check datatype
        if dtype is None:
            if out is not None and isinstance(out, torch.Tensor):
                dtype = out.dtype
            elif weight is not None and isinstance(weight, torch.Tensor):
                dtype = weight.dtype
            else:
                raise ValueError(
                    "Could not infer dtype from weight nor out and dtype was not provided"
                )
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Supported dtypes are float32, float16, bfloat16 (got {dtype})")
        if out is not None and out.dtype != dtype:
            raise ValueError(f"Output tensor has invalid dtype (expected {dtype}, got {out.dtype})")

        # Check input tensor
        x_local = input
        x = None
        x_async = None
        with_x_all_gather = tensor_parallel_mode == "column" and sequence_parallel
        if with_quantized_compute:
            if input_quantizer is None:
                raise ValueError("Missing quantizer for input tensor")
            input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
            if with_x_all_gather:
                input_quantizer.set_usage(columnwise=False)
                x, x_async = gather_along_first_dim(
                    x_local,
                    tensor_parallel_group,
                    async_op=True,
                    quantizer=input_quantizer,
                )
            else:
                if not is_quantized_tensor(x_local):
                    x_local = input_quantizer(x_local)
                x = x_local
        else:
            x_local = maybe_dequantize(x_local, dtype)

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
        if not with_quantized_compute:
            w = maybe_dequantize(w, dtype)
        elif with_quantized_compute and not is_quantized_tensor(w):
            if weight_quantizer is None:
                raise ValueError("Missing quantizer for weight tensor")
            weight_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
            w = weight_quantizer(w)

        # Check output tensor
        y = out
        if y is None:
            if not with_quantized_compute:
                output_quantizer = None
            if tensor_parallel_mode == "row":
                output_quantizer = None
        elif is_quantized_tensor(y):
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
        if output_quantizer is not None:
            if not isinstance(output_quantizer, Float8Quantizer):
                raise RuntimeError(
                    "Attempting to generate quantized output tensor with unsupported quantizer"
                )
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
            alpha=alpha,
            beta=beta,
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

        # Prepare weight tensor for backward pass
        if input_requires_grad:
            if w is not weight and with_quantized_compute and is_quantized_tensor(w):
                w.update_usage(rowwise_usage=False, columnwise_usage=True)
        else:
            w = None

        # Prepare input tensor for backward pass
        if weight_requires_grad:
            if with_quantized_compute and is_quantized_tensor(x_local):
                if not (isinstance(x_local, Float8TensorBase) and with_x_all_gather):
                    # FP8 does not support all-gather of transpose data
                    x_local.update_usage(rowwise_usage=False, columnwise_usage=True)
        else:
            x_local = None

        return y, x_local, w

    @staticmethod
    def _functional_backward(
        grad_output: torch.Tensor,
        input: Optional[torch.Tensor],  # pylint: disable=redefined-builtin
        weight: Optional[torch.Tensor],
        *,
        grad_input_alpha: Optional[float] = None,
        input_requires_grad: bool = True,
        grad_weight_alpha: Optional[float] = None,
        weight_requires_grad: bool = True,
        device: Optional[torch.device] = None,  # pylint: disable=unused-argument
        dtype: Optional[torch.dtype] = None,
        grad_weight: Optional[torch.Tensor] = None,
        grad_weight_beta: Optional[float] = None,
        accumulate_into_grad_weight: bool = False,
        grad_input: Optional[torch.Tensor] = None,
        grad_input_beta: Optional[float] = None,
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
        grad_input_alpha: float, optional
            Scaling factor applied to the result of the dgrad GEMM
        input_requires_grad: bool
            Whether to compute loss gradient w.r.t. input tensor
        grad_weight_alpha: float, optional
            Scaling factor applied to the result of the wgrad GEMM
        weight_requires_grad: bool
            Whether to compute loss gradient w.r.t. weight tensor
        device: torch.device, default = default CUDA device
            Tensor device
        dtype: torch.dtype, default = default dtype
            Tensor datatype
        grad_weight: torch.Tensor, optional
            Loss gradient w.r.t. weight tensor
        grad_weight_beta: float, optional
            Scaling factor applied to original value of grad_weight when accumulating into it
        accumulate_into_grad_weight: bool, default = `False`
            Add result to weight grad instead of overwriting
        grad_input: torch.Tensor, optional
            Loss gradient w.r.t. input tensor
        grad_input_beta: float, optional
            Scaling factor applied to original value of grad_input when accumulating into it
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
            if isinstance(weight, torch.Tensor):
                dtype = weight.dtype
            elif isinstance(grad_output, torch.Tensor):
                dtype = grad_output.dtype
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
                if not is_quantized_tensor(dy_local):
                    dy_local = grad_output_quantizer(dy_local)
                else:
                    dy_local.update_usage(
                        rowwise_usage=input_requires_grad,
                        columnwise_usage=weight_requires_grad,
                    )
                dy = dy_local
        else:
            dy_local = maybe_dequantize(dy_local, dtype)

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
                    if is_quantized_tensor(x_local):
                        x_local.update_usage(columnwise_usage=True)
                    else:
                        x_local = input_quantizer(x_local)
                    x = x_local
            else:
                x_local = maybe_dequantize(x_local, dtype)

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
            if with_quantized_compute:
                if is_quantized_tensor(w):
                    w.update_usage(columnwise_usage=True)
                else:
                    if weight_quantizer is None:
                        raise ValueError("Missing quantizer for weight tensor")
                    weight_quantizer.set_usage(columnwise=True)
                    w = weight_quantizer(w)
            else:
                w = maybe_dequantize(w, dtype)

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
            elif is_quantized_tensor(dx):
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
            if grad_input_quantizer is not None:
                if not isinstance(grad_input_quantizer, Float8Quantizer):
                    raise RuntimeError(
                        "Attempting to generate quantized grad input tensor "
                        "with unsupported quantizer"
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
                alpha=grad_input_alpha,
                beta=grad_input_beta,
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

            # Check grad weight tensor
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
                alpha=grad_weight_alpha,
                beta=grad_weight_beta,
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
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:

        # Check which grads are required
        input_requires_grad = ctx.requires_grad
        weight_requires_grad = ctx.requires_grad and self.weight.requires_grad

        # FP8 metadata
        input_quantizer = self.get_quantizer("forward", 0)
        weight_quantizer = self.get_quantizer("forward", 1)
        output_quantizer = next_op_input_quantizer
        grad_output_quantizer = self.get_quantizer("backward", 0)
        grad_input_quantizer = prev_op_grad_output_quantizer
        with_quantized_compute = FP8GlobalStateManager.is_fp8_enabled()
        if with_quantized_compute:
            # Configure quantizers
            # Note: We cache the quantized input for backward pass,
            # but discard the quantized weights.
            input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
            weight_quantizer.set_usage(rowwise=True, columnwise=False)

            recipe = FP8GlobalStateManager.get_fp8_recipe()
            if recipe.float8_current_scaling():
                input_quantizer.force_pow_2_scales = recipe.fp8_quant_fwd_inp.power_2_scale
                input_quantizer.amax_epsilon_scales = recipe.fp8_quant_fwd_inp.amax_epsilon
                weight_quantizer.force_pow_2_scales = recipe.fp8_quant_fwd_inp.power_2_scale
                weight_quantizer.amax_epsilon_scales = recipe.fp8_quant_fwd_inp.amax_epsilon
                grad_output_quantizer.force_pow_2_scales = recipe.fp8_quant_fwd_inp.power_2_scale
                grad_output_quantizer.amax_epsilon_scales = recipe.fp8_quant_fwd_inp.amax_epsilon
                if self.sequence_parallel and self.tensor_parallel_mode == "column":
                    input_quantizer.with_amax_reduction = True
                    input_quantizer.amax_reduction_group = self.tensor_parallel_group
                if self.sequence_parallel and self.tensor_parallel_mode == "row":
                    grad_output_quantizer.with_amax_reduction = True
                    grad_output_quantizer.amax_reduction_group = self.tensor_parallel_group

        # Get autocast dtype if needed
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = self.weight.dtype

        # Linear forward
        output, x_local, w = BasicLinear._functional_forward(
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
            input_requires_grad=input_requires_grad,
            weight_requires_grad=weight_requires_grad,
        )

        # Save state for backward pass
        if ctx.requires_grad:
            ctx.save_for_backward(x_local, w)
            ctx.with_quantized_compute = with_quantized_compute
            ctx.input_quantizer = input_quantizer
            ctx.weight_quantizer = weight_quantizer
            ctx.grad_output_quantizer = grad_output_quantizer
            ctx.grad_input_quantizer = grad_input_quantizer
            ctx.dtype = dtype
            ctx.input_requires_grad = input_requires_grad
            ctx.weight_requires_grad = weight_requires_grad

        return output

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:

        # Saved tensors from forward pass
        (x_local, w) = ctx.saved_tensors

        # Megatron-LM wgrad fusion
        # Note: Get grad tensor from param so we can accumulate
        # directly into it.
        accumulate_into_main_grad = self._accumulate_into_main_grad
        grad_weight = None
        if ctx.weight_requires_grad and accumulate_into_main_grad:
            weight_param = self.weight
            if hasattr(weight_param, "__fsdp_param__"):
                weight_param.main_grad = weight_param.get_main_grad()
            if not hasattr(weight_param, "main_grad"):
                raise RuntimeError(
                    "BasicLinear op is configured with "
                    "accumulate_into_main_grad=True, "
                    "but weight parameter does not have main_grad attribute"
                )
            grad_weight = weight_param.main_grad.detach()
        else:
            accumulate_into_main_grad = False

        # Linear backward pass
        grad_input, grad_weight = BasicLinear._functional_backward(
            grad_output=grad_output,
            input=x_local,
            weight=w,
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
        clear_tensor_data(x_local)

        # Megatron-LM wgrad fusion
        # Note: Return dummy tensor for grad weight if needed.
        if accumulate_into_main_grad:
            grad_weight = None
            weight_param = self.weight
            if hasattr(weight_param, "grad_added_to_main_grad"):
                weight_param.grad_added_to_main_grad = True
                grad_weight = get_dummy_wgrad(
                    list(weight_param.size()),
                    weight_param.dtype,
                    zero=getattr(weight_param, "zero_out_wgrad", False),
                )

        return grad_input, [grad_weight]

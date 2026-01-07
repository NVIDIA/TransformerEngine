# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for bias."""

from __future__ import annotations
from collections.abc import Iterable
import contextlib
import math
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...module.base import get_dummy_wgrad
from ...quantization import FP8GlobalStateManager
from ...tensor import Quantizer
from ...utils import (
    canonicalize_device,
    canonicalize_dtype,
    clear_tensor_data,
    devices_match,
)
from .._common import is_quantized_tensor
from ..op import BasicOperation, OperationContext


class GroupedLinear(BasicOperation):

    # Operation expects input split sizes
    num_extra_inputs: int = 1

    def __init__(
        self,
        group_size: int,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        rng_state_tracker_function: Optional[Callable[[], CudaRNGStatesTracker]] = None,
        accumulate_into_main_grad: bool = False,
    ) -> None:
        super().__init__()

        # Weight tensor dimensions
        self.group_size: int = group_size
        self.in_features: int = in_features
        self.out_features: int = out_features
        if self.group_size <= 0:
            raise ValueError(f"Invalid group size ({self.group_size})")
        if self.in_features <= 0:
            raise ValueError(f"Invalid input size ({self.in_features})")
        if self.out_features <= 0:
            raise ValueError(f"Invalid output size ({self.out_features})")

        # Weight tensor attributes
        device = canonicalize_device(device)
        dtype = canonicalize_dtype(dtype)
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Supported dtypes are float32, float16, bfloat16 (got {dtype})")

        # Initialize recipe state if needed for natively quantized weight
        self._with_quantized_weight: bool = FP8GlobalStateManager.with_fp8_parameters()
        if self._with_quantized_weight:
            self.reset_recipe_state(recipe=FP8GlobalStateManager.get_fp8_recipe())

        # RNG state tracker
        self._rng_state_tracker_function: Optional[Callable[[], CudaRNGStatesTracker]]
        self._rng_state_tracker_function = rng_state_tracker_function

        # Register weights
        self.weight0: torch.nn.Parameter
        for group_idx in range(self.group_size):
            weight_tensor = torch.empty(
                self.out_features,
                self.in_features,
                device=device,
                dtype=dtype,
            )
            self.register_parameter(
                f"weight{group_idx}",
                torch.nn.Parameter(weight_tensor),
            )

        # Register biases
        self.bias0: Optional[torch.nn.Parameter]
        for group_idx in range(self.group_size):
            bias_tensor = None
            if bias:
                bias_tensor = torch.empty(
                    self.out_features,
                    device=device,
                    dtype=dtype,
                )
                bias_tensor = torch.nn.Parameter(bias_tensor)
            self.register_parameter(f"bias{group_idx}", bias_tensor)

        # Initialize weights if needed
        if device.type != "meta":
            self.reset_parameters()

        # Whether to accumulate weight gradient into main_grad
        self._accumulate_into_main_grad: bool = accumulate_into_main_grad

    def num_quantizers(self, mode: str) -> int:
        if mode == "forward":
            return 2 * self.group_size
        if mode == "backward":
            return self.group_size
        return 0

    @property
    def has_bias(self) -> bool:
        return self.bias0 is not None

    @torch.no_grad
    def reset_parameters(self) -> None:
        """Initialize parameter buffers and values"""

        for group_idx in range(self.group_size):

            # Parameters
            weight = getattr(self, f"weight{group_idx}")
            bias = getattr(self, f"bias{group_idx}")

            # Parameter device
            device = weight.device
            if device.type == "meta":
                device = canonicalize_device(None)

            # Allocate buffers if needed
            if is_quantized_tensor(weight):
                weight = torch.empty(
                    weight.size(),
                    dtype=weight.dtype,
                    device=device,
                )
            elif not devices_match(weight.device, device):
                weight = torch.empty_like(weight, device=device)
            if bias is not None and not devices_match(bias.device, device):
                bias = torch.empty_like(bias, device=device)

            # Initialize values
            init_context = contextlib.nullcontext()
            if self._rng_state_tracker_function is not None:
                init_context = self._rng_state_tracker_function().fork()
            with init_context:
                torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            if bias is not None:
                bias.zero_()

            # Quantize weight if needed
            if self._with_quantized_weight:
                quantizer = self.get_quantizer("forward", 1)
                if quantizer is None:
                    raise RuntimeError(
                        "Tried to quantize weight with deferred initialization "
                        "due to meta device, but no quantizer was available. "
                        "This is most likely because the weight was initialized "
                        "within quantized_model_init, but the forward pass was not "
                        "performed within autocast."
                    )
                quantizer.set_usage(
                    rowwise=True,
                    columnwise=torch.is_grad_enabled(),
                )
                quantizer.internal = False
                with torch.no_grad():
                    weight = quantizer(weight)

            # Save updated parameters
            if not isinstance(weight, torch.nn.Parameter):
                weight = torch.nn.Parameter(weight)
            setattr(self, f"weight{group_idx}", weight)
            if bias is not None:
                if not isinstance(bias, torch.nn.Parameter):
                    bias = torch.nn.Parameter(bias)
                setattr(self, f"bias{group_idx}", bias)

    def pre_first_fuser_forward(self) -> None:
        super().pre_first_fuser_forward()

        # Initialize params if needed
        if any(param.device.type == "meta" for param in self.parameters()):
            self.reset_parameters()

        # Check that weights are consistent
        dtype = self.weight0.dtype
        device = self.weight0.device
        weight_requires_grad = self.weight0.requires_grad
        weight_tensor_type = type(self.weight0.data)
        for group_idx in range(self.group_size):
            weight = getattr(self, f"weight{group_idx}")
            if weight.dtype != dtype:
                raise RuntimeError(
                    f"Weight {group_idx} has invalid dtype "
                    f"(expected {dtype}, got {weight.dtype})."
                )
            if not devices_match(weight.device, device):
                raise RuntimeError(
                    f"Weight {group_idx} has invalid device "
                    f"(expected {device}, got {weight.device})."
                )
            if weight.requires_grad != weight_requires_grad:
                raise RuntimeError(
                    f"Weight {group_idx} has requires_grad={weight.requires_grad}, "
                    f"but expected requires_grad={weight_requires_grad}."
                )
            if type(weight.data) != weight_tensor_type:
                raise RuntimeError(
                    f"Weight {group_idx} has invalid tensor type "
                    f"(expected {weight_tensor_type.__name__}, "
                    f"got {type(weight.data).__name__})."
                )

        # Check that biases are consistent
        for group_idx in range(self.group_size):
            bias = getattr(self, f"bias{group_idx}")
            if self.has_bias:
                if bias is None:
                    raise RuntimeError(
                        f"Expected biases, but bias {group_idx} is uninitialized"
                    )
                if bias.dtype != dtype:
                    raise RuntimeError(
                        f"Bias {group_idx} has invalid dtype "
                        f"(expected {dtype}, got {bias.dtype})."
                    )
                if not devices_match(bias.device, device):
                    raise RuntimeError(
                        f"Bias {group_idx} has invalid device "
                        f"(expected {device}, got {bias.device})."
                    )
                if bias.requires_grad != weight_requires_grad:
                    raise RuntimeError(
                        f"Bias {group_idx} has requires_grad={bias.requires_grad}, "
                        f"but expected requires_grad={weight_requires_grad}."
                    )
            else:
                if bias is not None:
                    raise RuntimeError(
                        f"Expected no biases, but bias {group_idx} is initialized"
                    )

    def pre_fuser_forward(self, *, requires_grad: bool) -> None:
        super().pre_fuser_forward(requires_grad=requires_grad)
        if FP8GlobalStateManager.is_fp8_enabled():
            # Assume weights have consistent grad requirement
            weight_requires_grad = requires_grad and self.weight0.requires_grad

            # Configure quantizer usages
            # Note: We cache the quantized input for backward pass,
            # but discard the quantized weights.
            for group_idx in range(self.group_size):
                input_quantizer = self.get_quantizer("forward", 2 * group_idx)
                weight_quantizer = self.get_quantizer("forward", 2 * group_idx + 1)
                grad_output_quantizer = self.get_quantizer("backward", group_idx)
                input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
                weight_quantizer.set_usage(rowwise=True, columnwise=False)
                grad_output_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)

    def op_forward(self, *args, **kwargs):
        raise RuntimeError(
            "{self.__class__.__name__} operation has "
            f"{self.num_extra_inputs} extra tensor inputs "
            f"and {self.num_extra_outputs} extra tensor outputs. "
            "It overrides `fuser_forward` instead of `op_forward`."
        )

    def op_backward(self, *args, **kwargs):
        raise RuntimeError(
            "{self.__class__.__name__} operation has "
            f"{self.num_extra_inputs} extra tensor inputs "
            f"and {self.num_extra_outputs} extra tensor outputs. "
            "It overrides `fuser_backward` instead of `op_backward`."
        )

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:

        # Check which grads are required
        ctx = basic_op_ctxs[0]
        input_requires_grad = ctx.requires_grad
        weight_requires_grad = ctx.requires_grad and self.weight0.requires_grad

        # Quantizers
        input_quantizers = None
        weight_quantizers = None
        grad_output_quantizers = None
        with_quantized_compute = FP8GlobalStateManager.is_fp8_enabled()
        if with_quantized_compute:
            input_quantizers = []
            weight_quantizers = []
            grad_output_quantizers = []
            for group_idx in range(self.group_size):
                input_quantizers.append(self.get_quantizer("forward", 2 * group_idx))
                weight_quantizers.append(self.get_quantizer("forward", 2 * group_idx + 1))
                grad_output_quantizers.append(self.get_quantizer("backward", group_idx))

        # Get autocast dtype if needed
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = self.weight0.dtype

        # Extract split sizes from extra input
        # TODO Support splits on GPU
        split_sizes = basic_op_extra_inputs[0][0]
        split_sizes_int = [int(s) for s in split_sizes.tolist()]
        if len(split_sizes_int) != self.group_size:
            raise ValueError(
                f"Expected {self.group_size} splits, but got {len(split_sizes_int)}."
            )

        # Extract params
        weights = []
        biases = []
        for group_idx in range(self.group_size):
            weights.append(getattr(self, f"weight{group_idx}"))
            biases.append(getattr(self, f"bias{group_idx}"))

        # Perform GEMMs
        # TODO: Fused impl, quantization
        xs = torch.split(input_, split_sizes_int)
        ys = []
        for x, w, b in zip(xs, weights, biases):
            y = torch.nn.functional.linear(x, w, bias=b)
            ys.append(y)
        out = torch.cat(ys)

        # Save state for backward pass
        if ctx.requires_grad:
            ctx.save_for_backward(split_sizes, *xs, *weights)
            ctx.with_quantized_compute = with_quantized_compute
            ctx.input_quantizers = input_quantizers
            ctx.weight_quantizers = weight_quantizers
            ctx.grad_output_quantizers = grad_output_quantizers
            ctx.grad_input_quantizers = None
            ctx.dtype = dtype
            ctx.input_requires_grad = input_requires_grad
            ctx.weight_requires_grad = weight_requires_grad

        return out, [()]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        Iterable[Iterable[Optional[torch.Tensor]]],
        Iterable[Iterable[Optional[torch.Tensor]]],
    ]:
        group_size = self.group_size
        has_bias = self.has_bias

        # Saved tensors from forward pass
        ctx = basic_op_ctxs[0]
        saved_tensors = ctx.saved_tensors
        split_sizes, saved_tensors = saved_tensors[0], saved_tensors[1:]
        xs, saved_tensors = saved_tensors[:group_size], saved_tensors[group_size:]
        weights, saved_tensors = saved_tensors[:group_size], saved_tensors[group_size:]

        # Split grad output tensor
        # TODO Support splits on GPU
        split_sizes_int = [int(s) for s in split_sizes.tolist()]
        dys = torch.split(grad_output, split_sizes_int)

        # Megatron-LM wgrad fusion
        # Note: Get grad tensors from params so we can accumulate
        # directly into it.
        accumulate_into_main_grad = self._accumulate_into_main_grad
        grad_weights = [None] * group_size
        if ctx.weight_requires_grad and accumulate_into_main_grad:
            for group_idx in range(group_size):
                weight_param = getattr(self, f"weight{group_idx}")
                if hasattr(weight_param, "__fsdp_param__"):
                    weight_param.main_grad = weight_param.get_main_grad()
                accumulate_into_main_grad = not getattr(weight_param, "overwrite_main_grad", False)
                if not hasattr(weight_param, "main_grad"):
                    raise RuntimeError(
                        "GroupLinear op is configured with "
                        "accumulate_into_main_grad=True, "
                        "but weight parameter does not have main_grad attribute"
                    )
                grad_weights[group_idx] = weight_param.main_grad.detach()
        else:
            accumulate_into_main_grad = False

        # Compute grad biases
        # TODO: Fuse with quantization
        grad_biases = [None] * group_size
        if ctx.weight_requires_grad and has_bias:
            for group_idx in range(group_size):
                dy = dys[group_idx]
                grad_biases[group_idx] = dy.reshape(-1, dy.size(-1)).sum(0)

        # Perform GEMMs
        # TODO: Fused impl, quantization
        grad_input = None
        if ctx.input_requires_grad:
            dxs = []
            for group_idx in range(group_size):
                dy_shape = list(dys[group_idx].size())
                dx = torch.matmul(
                    dys[group_idx].reshape(-1, dy_shape[-1]),
                    weights[group_idx],
                )
                dxs.append(dx.reshape(dy_shape[:-1] + [dx.size(-1)]))
            grad_input = torch.cat(dxs)
        if ctx.weight_requires_grad:
            for group_idx in range(group_size):
                grad_weights[group_idx] = torch.matmul(
                    dys[group_idx].reshape(-1, dys[group_idx].size(-1)).T,
                    xs[group_idx].reshape(-1, xs[group_idx].size(-1)),
                    out=grad_weights[group_idx],
                )

        # Clear input tensors if possible
        clear_tensor_data(*xs)

        # Megatron-LM wgrad fusion
        # Note: Return dummy tensor for grad weight if needed.
        if accumulate_into_main_grad:
            grad_weights = [None] * group_size
            for group_idx in range(group_size):
                weight_param = getattr(self, f"weight{group_idx}")
                if hasattr(weight_param, "grad_added_to_main_grad"):
                    weight_param.grad_added_to_main_grad = True
                    grad_weights[group_idx] = get_dummy_wgrad(
                        list(weight_param.size()),
                        weight_param.dtype,
                        zero=getattr(weight_param, "zero_out_wgrad", False),
                    )

        grad_params = grad_weights + grad_biases if has_bias else grad_weights
        return grad_input, [grad_params], [(None,)]

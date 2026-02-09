# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for bias."""

from __future__ import annotations
from collections.abc import Callable, Iterable, Sequence
import contextlib
import math
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...cpp_extensions import general_grouped_gemm
from ...distributed import CudaRNGStatesTracker
from ...module.base import (
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
    get_dummy_wgrad,
)
from ...quantization import FP8GlobalStateManager, Recipe
from ...tensor import MXFP8Quantizer, MXFP8Tensor, Quantizer
from ...utils import (
    canonicalize_device,
    canonicalize_dtype,
    clear_tensor_data,
    devices_match,
    round_up_to_nearest_multiple,
)
from .._common import is_quantized_tensor, maybe_dequantize
from ..op import BasicOperation, OperationContext


class GroupedLinear(BasicOperation):
    r"""Apply multiple linear transformations: :math:``y_i = x_i W_i^T + b_i``

    This is equivalent to splitting the input tensor along its first
    dimension, applying a separate ``torch.nn.Linear`` to each split,
    and concatenating along the first dimension.

    Parameters
    ----------
    num_groups : int
        Number of linear transformations.
    in_features : int
        Inner dimension of input tensor.
    out_features : int
        Inner dimension of output tensor.
    bias : bool, default = ``True``
        Apply additive bias.
    device : torch.device, default = default CUDA device
        Tensor device.
    dtype : torch.dtype, default = default dtype
        Tensor datatype.
    rng_state_tracker_function : callable
        Function that returns ``CudaRNGStatesTracker``, which is used
        for model-parallel weight initialization.
    accumulate_into_main_grad : bool, default = ``False``
        Whether to directly accumulate weight gradients into the
        weight's ``main_grad`` attribute instead of relying on PyTorch
        autograd. The weight's ``main_grad`` must be set externally
        and there is no guarantee that `grad` will be set or be
        meaningful. This is primarily intented to integrate with
        Megatron-LM. This argument along with weight tensor having
        attribute ``overwrite_main_grad`` set to True will overwrite
        ``main_grad`` instead of accumulating.

    """

    # Operation expects input split sizes
    num_extra_inputs: int = 1

    def __init__(
        self,
        num_groups: int,
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
        self.num_groups: int = num_groups
        self.in_features: int = in_features
        self.out_features: int = out_features
        if self.num_groups <= 0:
            raise ValueError(f"Invalid number of groups ({self.num_groups})")
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
        for group_idx in range(self.num_groups):
            weight_tensor = torch.empty(
                self.out_features,
                self.in_features,
                device="meta",
                dtype=dtype,
            )
            self.register_parameter(
                f"weight{group_idx}",
                torch.nn.Parameter(weight_tensor),
            )

        # Register biases
        self.bias0: Optional[torch.nn.Parameter]
        for group_idx in range(self.num_groups):
            bias_tensor = None
            if bias:
                bias_tensor = torch.empty(
                    self.out_features,
                    device="meta",
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
            return 2 * self.num_groups
        if mode == "backward":
            return self.num_groups
        return 0

    @property
    def has_bias(self) -> bool:
        """Whether an additive bias is being applied"""
        return self.bias0 is not None

    def reset_parameters(self) -> None:
        """Initialize parameter buffers and values"""

        # Parameter device
        device = self.weight0.device
        if device.type == "meta":
            device = canonicalize_device(None)

        # Initialize weight values
        # Note: Allocate a single buffer in order to support grouped
        # GEMM kernels that expect a single weight buffer.
        packed_weights = torch.empty(
            self.num_groups,
            self.out_features,
            self.in_features,
            dtype=self.weight0.dtype,
            device=device,
        )
        weights = [packed_weights[idx] for idx in range(self.num_groups)]
        for weight in weights:
            init_context = contextlib.nullcontext()
            if self._rng_state_tracker_function is not None:
                init_context = self._rng_state_tracker_function().fork()
            with init_context:
                torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        # Quantize weights if needed
        if self._with_quantized_weight:

            # Configure quantizers
            quantizers = [
                self.get_quantizer("forward", 2 * idx + 1) for idx in range(self.num_groups)
            ]
            with_rowwise_usage = True
            with_columnwise_usage = torch.is_grad_enabled()
            for quantizer in quantizers:
                if quantizer is None:
                    raise RuntimeError(
                        "Tried to quantize weight with deferred initialization "
                        "due to meta device, but no quantizer was available. "
                        "This is most likely because the weight was initialized "
                        "within quantized_model_init, but the forward pass was not "
                        "performed within autocast."
                    )
                quantizer.set_usage(
                    rowwise=with_rowwise_usage,
                    columnwise=with_columnwise_usage,
                )
                quantizer.internal = False

            # Quantize weights
            weights = self._quantize_weights(weights, quantizers)

        # Register weights
        for group_idx, weight in enumerate(weights):
            if not isinstance(weight, torch.nn.Parameter):
                weight = torch.nn.Parameter(weight)
            setattr(self, f"weight{group_idx}", weight)

        # Initialize biases if needed
        if self.bias0 is not None:
            packed_biases = torch.zeros(
                self.num_groups,
                self.out_features,
                dtype=self.bias0.dtype,
                device=device,
            )
            for group_idx in range(self.num_groups):
                bias = torch.nn.Parameter(packed_biases[group_idx])
                setattr(self, f"bias{group_idx}", bias)

    def _quantize_weights(
        self,
        weights: Sequence[torch.Tensor],
        quantizers: Sequence[Quantizer],
    ) -> Sequence[torch.Tensor]:
        """Construct quantized weight tensors."""

        # Manually construct MXFP8 weights
        if isinstance(quantizers[0], MXFP8Quantizer):
            return self._quantize_weights_mxfp8(weights, quantizers)

        # Use quantizers to construct quantized weights
        with torch.no_grad():
            return [quantizer(weight) for quantizer, weight in zip(quantizers, weights)]

    def _quantize_weights_mxfp8(
        self,
        weights: Sequence[torch.Tensor],
        quantizers: Sequence[Quantizer],
    ) -> Sequence[MXFP8Tensor]:
        """Construct MXFP8 weight tensors.

        Instead of allocating separate buffers for each weight tensor,
        this function constructs large buffers and assigns subviews to
        each tensor. This is intended to support grouped GEMM kernels
        that expect packed buffers.

        """

        # Tensor dimensions
        num_groups = len(weights)
        out_features, in_features = weights[0].size()
        packed_shape = (num_groups, out_features, in_features)
        unpacked_shape = (out_features, in_features)

        # Tensor attributes
        device = weights[0].device
        dtype = weights[0].dtype
        requires_grad = torch.is_grad_enabled()
        with_rowwise_usage = quantizers[0].rowwise_usage
        with_columnwise_usage = quantizers[0].columnwise_usage

        # Construct packed buffers
        rowwise_data = [None] * num_groups
        rowwise_scales = [None] * num_groups
        columnwise_data = [None] * num_groups
        columnwise_scales = [None] * num_groups
        if with_rowwise_usage:
            scale_shape = (
                num_groups,
                round_up_to_nearest_multiple(out_features, 128),
                round_up_to_nearest_multiple(in_features // 32, 4),
            )
            packed_data = torch.empty(packed_shape, dtype=torch.uint8, device=device)
            packed_scales = torch.empty(scale_shape, dtype=torch.uint8, device=device)
            rowwise_data = [packed_data[idx] for idx in range(num_groups)]
            rowwise_scales = [packed_scales[idx] for idx in range(num_groups)]
        if with_columnwise_usage:
            scale_shape = (
                num_groups,
                round_up_to_nearest_multiple(out_features // 32, 4),
                round_up_to_nearest_multiple(in_features, 128),
            )
            packed_data = torch.empty(packed_shape, dtype=torch.uint8, device=device)
            packed_scales = torch.empty(scale_shape, dtype=torch.uint8, device=device)
            columnwise_data = [packed_data[idx] for idx in range(num_groups)]
            columnwise_scales = [packed_scales[idx] for idx in range(num_groups)]

        # Construct MXFP8 tensors and cast to MXFP8
        out = []
        with torch.no_grad():
            for group_idx in range(num_groups):
                weight = MXFP8Tensor(
                    shape=unpacked_shape,
                    dtype=dtype,
                    fp8_dtype=tex.DType.kFloat8E4M3,
                    rowwise_data=rowwise_data[group_idx],
                    rowwise_scale_inv=rowwise_scales[group_idx],
                    columnwise_data=columnwise_data[group_idx],
                    columnwise_scale_inv=columnwise_scales[group_idx],
                    quantizer=quantizers[group_idx],
                    requires_grad=requires_grad,
                    with_gemm_swizzled_scales=False,
                )
                weight.copy_(weights[group_idx])
                out.append(weight)

        return out

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
        for group_idx in range(self.num_groups):
            weight = getattr(self, f"weight{group_idx}")
            if weight.dtype != dtype:
                raise RuntimeError(
                    f"Weight {group_idx} has invalid dtype (expected {dtype}, got {weight.dtype})."
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
            if type(weight.data) != weight_tensor_type:  # pylint: disable=unidiomatic-typecheck
                raise RuntimeError(
                    f"Weight {group_idx} has invalid tensor type "
                    f"(expected {weight_tensor_type.__name__}, "
                    f"got {type(weight.data).__name__})."
                )

        # Check that biases are consistent
        for group_idx in range(self.num_groups):
            bias = getattr(self, f"bias{group_idx}")
            if self.has_bias:
                if bias is None:
                    raise RuntimeError(f"Expected biases, but bias {group_idx} is uninitialized")
                if bias.dtype != dtype:
                    raise RuntimeError(
                        f"Bias {group_idx} has invalid dtype (expected {dtype}, got {bias.dtype})."
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
                    raise RuntimeError(f"Expected no biases, but bias {group_idx} is initialized")

    def pre_fuser_forward(self, *, requires_grad: bool) -> None:
        super().pre_fuser_forward(requires_grad=requires_grad)
        if FP8GlobalStateManager.is_fp8_enabled():
            # Assume weights have consistent grad requirement
            weight_requires_grad = requires_grad and self.weight0.requires_grad

            # Configure quantizer usages
            # Note: We cache the quantized input for backward pass,
            # but discard the quantized weights.
            for group_idx in range(self.num_groups):
                input_quantizer = self.get_quantizer("forward", 2 * group_idx)
                weight_quantizer = self.get_quantizer("forward", 2 * group_idx + 1)
                grad_output_quantizer = self.get_quantizer("backward", group_idx)
                input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
                weight_quantizer.set_usage(rowwise=True, columnwise=False)
                grad_output_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)

    def reset_recipe_state(self, *, recipe: Optional[Recipe]) -> None:
        super().reset_recipe_state(recipe=recipe)

        for group_idx in range(self.num_groups):
            # Input/grad output quantizers use internal tensors
            input_quantizer = self.get_quantizer("forward", 2 * group_idx)
            grad_output_quantizer = self.get_quantizer("backward", group_idx)
            if input_quantizer is not None:
                input_quantizer.internal = True
            if grad_output_quantizer is not None:
                grad_output_quantizer.internal = True

            # Handle weight quantizer
            # Note: This function may be called in base class constructor,
            # before any basic linear attrs have been set.
            weight_quantizer = self.get_quantizer("forward", 2 * group_idx + 1)
            if weight_quantizer is None:
                pass
            elif is_quantized_tensor(getattr(self, f"weight{group_idx}", None)):
                # Make sure weight param has correct quantizer
                weight_quantizer.set_usage(rowwise=True, columnwise=torch.is_grad_enabled())
                weight_quantizer.internal = False
                getattr(self, f"weight{group_idx}").update_quantizer(weight_quantizer.copy())
            else:
                # Use internal tensors if quantized weights will not be
                # exposed externally
                weight_quantizer.internal = (
                    not FP8GlobalStateManager.with_fp8_parameters()
                    and not getattr(self, "_with_quantized_weight", False)
                )

            # Recipe-specific configuration
            # Note: This function may be called in base class constructor,
            # before any basic linear attrs have been set.
            if recipe is not None:
                if recipe.float8_current_scaling():
                    input_quantizer.force_pow_2_scales = recipe.fp8_quant_fwd_inp.power_2_scale
                    input_quantizer.amax_epsilon_scales = recipe.fp8_quant_fwd_inp.amax_epsilon
                    weight_quantizer.force_pow_2_scales = recipe.fp8_quant_fwd_weight.power_2_scale
                    weight_quantizer.amax_epsilon_scales = recipe.fp8_quant_fwd_weight.amax_epsilon
                    grad_output_quantizer.force_pow_2_scales = (
                        recipe.fp8_quant_bwd_grad.power_2_scale
                    )
                    grad_output_quantizer.amax_epsilon_scales = (
                        recipe.fp8_quant_bwd_grad.amax_epsilon
                    )

    def op_forward(self, *args, **kwargs):
        raise RuntimeError(
            f"{self.__class__.__name__} operation has "
            f"{self.num_extra_inputs} extra tensor inputs "
            f"and {self.num_extra_outputs} extra tensor outputs. "
            "It overrides `fuser_forward` instead of `op_forward`."
        )

    def op_backward(self, *args, **kwargs):
        raise RuntimeError(
            f"{self.__class__.__name__} operation has "
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
        num_groups = self.num_groups
        has_bias = self.has_bias
        device = self.weight0.device

        # Check which grads are required
        ctx = basic_op_ctxs[0]
        input_requires_grad = ctx.requires_grad
        weight_requires_grad = ctx.requires_grad and self.weight0.requires_grad

        # Quantizers
        input_quantizers = [None] * num_groups
        weight_quantizers = [None] * num_groups
        grad_output_quantizers = [None] * num_groups
        with_quantized_compute = FP8GlobalStateManager.is_fp8_enabled()
        if with_quantized_compute:
            for group_idx in range(num_groups):
                input_quantizers[group_idx] = self.get_quantizer("forward", 2 * group_idx)
                weight_quantizers[group_idx] = self.get_quantizer("forward", 2 * group_idx + 1)
                grad_output_quantizers[group_idx] = self.get_quantizer("backward", group_idx)

        # Get autocast dtype if needed
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = self.weight0.dtype

        # Extract split sizes from extra input
        split_sizes = basic_op_extra_inputs[0][0]
        split_sizes_int = [int(s) for s in split_sizes.tolist()]
        if len(split_sizes_int) != num_groups:
            raise ValueError(f"Expected {num_groups} splits, but got {len(split_sizes_int)}.")

        # Extract params
        weights = [getattr(self, f"weight{idx}") for idx in range(num_groups)]
        bs = None
        if has_bias:
            bs = [maybe_dequantize(getattr(self, f"bias{idx}"), dtype) for idx in range(num_groups)]

        # Convert weight dtype if needed
        ws = []
        for w, quantizer in zip(weights, weight_quantizers):
            if not with_quantized_compute:
                w = maybe_dequantize(w, dtype)
            elif with_quantized_compute and not is_quantized_tensor(w):
                quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                w = quantizer(w)
            ws.append(w)

        # Split input tensor and convert dtypes if needed
        x = maybe_dequantize(input_, dtype)
        xs = None
        if with_quantized_compute:
            for quantizer in input_quantizers:
                quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
            xs = tex.split_quantize(x, split_sizes_int, input_quantizers)
        else:
            xs = torch.split(x, split_sizes_int)

        # Allocate output tensor
        in_shape = list(input_.size())
        out_shape = in_shape[:-1] + [self.out_features]
        out = torch.empty(out_shape, dtype=dtype, device=device)

        # Perform GEMMs
        general_grouped_gemm(
            ws,
            xs,
            [out],
            [None] * num_groups,  # quantization_params
            dtype,
            m_splits=split_sizes_int,
            bias=bs,
            use_bias=has_bias,
            use_split_accumulator=_2X_ACC_FPROP,
            single_output=True,
        )

        # Prepare weight tensors for backward pass
        if not input_requires_grad:
            ws = [None] * num_groups
        elif with_quantized_compute:
            for w, weight_param in zip(ws, weights):
                if w is not weight_param:
                    w.update_usage(rowwise_usage=False, columnwise_usage=True)

        # Prepare input tensor for backward pass
        if not weight_requires_grad:
            xs = [None] * num_groups
        elif with_quantized_compute:
            for x in xs:
                x.update_usage(rowwise_usage=False, columnwise_usage=True)

        # Save state for backward pass
        if ctx.requires_grad:
            ctx.save_for_backward(split_sizes, *xs, *ws)
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
        num_groups = self.num_groups
        has_bias = self.has_bias
        device = self.weight0.device

        # Saved tensors from forward pass
        ctx = basic_op_ctxs[0]
        saved_tensors = ctx.saved_tensors
        split_sizes, saved_tensors = saved_tensors[0], saved_tensors[1:]
        xs, saved_tensors = saved_tensors[:num_groups], saved_tensors[num_groups:]
        ws, saved_tensors = saved_tensors[:num_groups], saved_tensors[num_groups:]

        # Split grad output tensor and convert dtypes if needed
        split_sizes_int = [int(s) for s in split_sizes.tolist()]
        dy = maybe_dequantize(grad_output, ctx.dtype)
        dys = None
        grad_biases = [None] * num_groups
        if ctx.with_quantized_compute:
            for quantizer in ctx.grad_output_quantizers:
                quantizer.set_usage(
                    rowwise=ctx.input_requires_grad,
                    columnwise=ctx.weight_requires_grad,
                )
            dys = tex.split_quantize(dy, split_sizes_int, ctx.grad_output_quantizers)
            if has_bias:
                grad_biases = [
                    dy.reshape(-1, dy.size(-1)).sum(dim=0)
                    for dy in torch.split(grad_output, split_sizes_int)
                ]
        else:
            dys = torch.split(dy, split_sizes_int)
            if has_bias:
                grad_biases = [dy.reshape(-1, dy.size(-1)).sum(dim=0) for dy in dys]

        # Initialize grad weight buffers
        accumulate_into_main_grad = self._accumulate_into_main_grad
        grad_weights = [None] * num_groups
        if ctx.weight_requires_grad:
            if accumulate_into_main_grad:
                # Megatron-LM wgrad fusion
                # Note: Get grad tensors from params so we can
                # accumulate directly into it.
                for group_idx in range(num_groups):
                    weight_param = getattr(self, f"weight{group_idx}")
                    if hasattr(weight_param, "__fsdp_param__"):
                        weight_param.main_grad = weight_param.get_main_grad()
                    grad_weights[group_idx] = weight_param.main_grad
                accumulate_into_main_grad = not getattr(self.weight0, "overwrite_main_grad", False)
            else:
                weight_shape = ws[0].size()
                for group_idx in range(num_groups):
                    grad_weights[group_idx] = torch.empty(
                        weight_shape,
                        dtype=ctx.dtype,
                        device=device,
                    )
        else:
            accumulate_into_main_grad = False

        # Perform dgrad GEMMs
        grad_input = None
        if ctx.input_requires_grad:
            out_shape = list(grad_output.size())
            in_shape = out_shape[:-1] + [self.in_features]
            grad_input = torch.empty(
                in_shape,
                dtype=ctx.dtype,
                device=device,
            )
            general_grouped_gemm(
                ws,
                dys,
                [grad_input],
                [None] * num_groups,  # quantization_params
                ctx.dtype,
                layout="NN",
                m_splits=split_sizes_int,
                use_split_accumulator=_2X_ACC_DGRAD,
                single_output=True,
            )

        # Perform wgrad GEMMs
        if ctx.weight_requires_grad:
            general_grouped_gemm(
                xs,
                dys,
                grad_weights,
                [None] * num_groups,  # quantization_params
                ctx.dtype,
                layout="NT",
                m_splits=split_sizes_int,
                use_split_accumulator=_2X_ACC_WGRAD,
                accumulate=accumulate_into_main_grad,
            )

        # Clear input tensors if possible
        clear_tensor_data(*xs)

        # Megatron-LM wgrad fusion
        # Note: Return dummy tensor for grad weight if needed.
        if accumulate_into_main_grad:
            grad_weights = [None] * num_groups
            for group_idx in range(num_groups):
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

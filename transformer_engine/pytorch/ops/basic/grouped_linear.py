# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for grouped linear layer."""

from __future__ import annotations
from collections.abc import Callable, Iterable, Sequence
import contextlib
import functools
import math
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...cpp_extensions import general_grouped_gemm, general_grouped_gemm_for_grouped_tensor
from ...distributed import CudaRNGStatesTracker
from ...module._common import WeightGradStore
from ...module.base import (
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ...quantization import FP8GlobalStateManager, Recipe
from ...quantized_tensor import QuantizedTensorStorage
from ...tensor import MXFP8Quantizer, MXFP8Tensor, Quantizer
from ...utils import (
    canonicalize_device,
    canonicalize_dtype,
    clear_tensor_data,
    devices_match,
    get_device_compute_capability,
    resolve_grouped_linear_single_param_flags,
    round_up_to_nearest_multiple,
)
from .._common import (
    get_accumulate_flag_in_param,
    get_dummy_wgrads_for_params,
    get_main_grad_from_param,
    is_quantized_tensor,
    maybe_dequantize,
    view_main_grad_as_grouped_buffer,
)
from ..op import BasicOperation, OperationContext
from ...tensor import GroupedTensor, GroupedTensorStorage
from ...triton.grouped_dbias_dscales import (
    compute_grouped_dbias,
    compute_grouped_dbias_dscales,
)


class GroupedLinear(BasicOperation):
    r"""Apply multiple linear transformations: :math:``y_i = x_i W_i^T + b_i``

    This feature is experimental and subject to change.

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
        meaningful. This is primarily intended to integrate with
        Megatron-LM. This argument along with weight tensor having
        attribute ``overwrite_main_grad`` set to True will overwrite
        ``main_grad`` instead of accumulating.
    single_grouped_weight : bool, default = ``False``
        Store all expert weights as one ``GroupedTensor`` parameter ``weight``.
        EXPERIMENTAL and subject to change. Gated by the
        ``NVTE_GROUPED_LINEAR_SINGLE_PARAM`` environment variable: if the env var
        is not set this argument is forced to ``False`` with a warning.
    delay_wgrad_compute : bool, default = ``False``
        Whether to delay weight gradient computation
    single_grouped_bias : bool, default = ``False``
        If ``True`` (and ``bias=True``), store all expert biases as one ``GroupedTensor``
        parameter named ``bias`` instead of ``bias0``..``bias{N-1}``.
        EXPERIMENTAL and subject to change. Gated by the
        ``NVTE_GROUPED_LINEAR_SINGLE_PARAM`` environment variable: if the env var
        is not set this argument is forced to ``False`` with a warning.
    scale_bias : bool, default = ``False``
        If ``True`` (and ``bias=True``), expects a probability tensor as an
        additional extra input and adds ``bias * scales`` instead of ``bias``
        in the forward pass. The scale tensor has shape
        ``(total_tokens,)`` and is split according to the split sizes.

    """

    # Operation expects input split sizes (and optionally scales tensor)
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
        single_grouped_weight: bool = False,
        single_grouped_bias: bool = False,
        delay_wgrad_compute: bool = False,
        scale_bias: bool = False,
    ) -> None:
        super().__init__()

        self._scale_bias: bool = scale_bias and bias
        if self._scale_bias:
            self.num_extra_inputs = 2

        self.wgrad_store = WeightGradStore(delay_wgrad_compute)
        self.wgrad_accumulation_and_reduce_hooks: list = []

        # Weight tensor dimensions
        self.num_groups: int = num_groups
        self.in_features: int = in_features
        self.out_features: int = out_features
        single_grouped_weight, single_grouped_bias = resolve_grouped_linear_single_param_flags(
            single_grouped_weight, single_grouped_bias
        )
        self.single_grouped_weight: bool = single_grouped_weight
        self.single_grouped_bias: bool = single_grouped_bias
        self.use_bias: bool = bias
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
        # TODO(ksivaman): Proper support for meta device.
        # We do not want to reset params later as it wipes off
        # main_grad and related attributes.
        self.weight0: torch.nn.Parameter
        for group_idx in range(self.num_groups):
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
        for group_idx in range(self.num_groups):
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

        self._apply_delay_wgrad_param_hooks()

    def _apply_delay_wgrad_param_hooks(self) -> None:
        """Set ``skip_backward_post_hook`` on weights when delaying wgrad (bias uses main backward)."""
        if not self.wgrad_store.delay_wgrad_compute():
            return
        if self.single_grouped_weight:
            self.weight.skip_backward_post_hook = True
        else:
            for group_idx in range(self.num_groups):
                getattr(self, f"weight{group_idx}").skip_backward_post_hook = True

    def register_wgrad_accumulation_and_reduce_hooks(
        self, wgrad_accumulation_and_reduce_hook: Callable
    ) -> None:
        """Register a hook to run after delayed wgrad computation completes.

        Mirrors ``TransformerEngineBaseModule.register_wgrad_accumulation_and_reduce_hooks``
        so that DDP can wire its ``param.grad = None`` / reduce-scatter callback here
        instead of directly on the AccumulateGrad node (which is bypassed when
        ``skip_backward_post_hook`` is set).
        """
        self.wgrad_accumulation_and_reduce_hooks.append(wgrad_accumulation_and_reduce_hook)

    def _trigger_wgrad_accumulation_and_reduce_hooks(self) -> None:
        """Call all registered wgrad accumulation and reduce hooks."""
        for hook in self.wgrad_accumulation_and_reduce_hooks:
            hook()

    def need_backward_dw(self) -> bool:
        """Return whether :meth:`backward_dw` must run to finish weight gradients."""
        return self.wgrad_store is not None and self.wgrad_store.delay_wgrad_compute()

    def backward_dw(self) -> None:
        """Execute delayed weight gradient grouped GEMMs (see ``delay_wgrad_compute``)."""
        if not self.need_backward_dw():
            return
        if self.wgrad_store.context is None or self.wgrad_store.context.empty():
            return
        _, tensor_list = self.wgrad_store.pop()
        activations = tensor_list[0]
        grad_weights = tensor_list[2]
        if isinstance(activations, list):
            clear_tensor_data(*activations)
        else:
            # Fused MXFP8 grouped MLP saves `GroupedTensor` activations for wgrad.
            clear_tensor_data(
                activations.rowwise_data,
                activations.columnwise_data,
                activations.scale_inv,
                activations.columnwise_scale_inv,
            )
        if self._accumulate_into_main_grad:
            self._trigger_wgrad_accumulation_and_reduce_hooks()
            return
        if self.single_grouped_weight:
            if isinstance(grad_weights, list):
                self.weight.grad = torch.stack(grad_weights, dim=0).to(self.weight.dtype)
            else:
                self.weight.grad = grad_weights.rowwise_data.view(
                    self.num_groups,
                    self.out_features,
                    self.in_features,
                ).to(self.weight.dtype)
        else:
            for group_idx in range(self.num_groups):
                w = getattr(self, f"weight{group_idx}")
                w.grad = grad_weights[group_idx].to(w.dtype)
        self._trigger_wgrad_accumulation_and_reduce_hooks()

    def _get_bias_tensors(self, dtype: torch.dtype) -> list[torch.Tensor]:
        """Retrieve per-group bias tensors in the given dtype."""
        if self.single_grouped_bias:
            bias_parts = self.bias.quantized_tensors
            if bias_parts is None:
                bias_parts = self.bias.split_into_quantized_tensors()
            return [maybe_dequantize(p.reshape(-1), dtype) for p in bias_parts]
        return [
            maybe_dequantize(getattr(self, f"bias{idx}"), dtype) for idx in range(self.num_groups)
        ]

    def num_quantizers(self, mode: str) -> int:
        if mode == "forward":
            return 2 * self.num_groups
        if mode == "backward":
            return self.num_groups
        return 0

    @property
    def has_bias(self) -> bool:
        """Whether an additive bias is being applied"""
        return self.use_bias

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
        packed_biases: Optional[torch.Tensor] = None
        if self.use_bias:
            if self.bias0 is not None:
                bias_dtype = self.bias0.dtype
            elif getattr(self, "bias", None) is not None:
                bias_dtype = self.bias.dtype
            elif getattr(self, "weight", None) is not None:
                bias_dtype = self.weight.dtype
            else:
                bias_dtype = self.weight0.dtype
            packed_biases = torch.zeros(
                self.num_groups,
                self.out_features,
                dtype=bias_dtype,
                device=device,
            )
            if not self.single_grouped_bias:
                for group_idx in range(self.num_groups):
                    bias = torch.nn.Parameter(packed_biases[group_idx])
                    setattr(self, f"bias{group_idx}", bias)
        else:
            for group_idx in range(self.num_groups):
                self.register_parameter(f"bias{group_idx}", None)

        if self.single_grouped_weight:
            self.make_grouped_weights()
        if self.use_bias and self.single_grouped_bias:
            assert packed_biases is not None
            self._make_grouped_biases_from_packed(packed_biases)
        self._apply_delay_wgrad_param_hooks()

    def make_grouped_weights(self) -> None:
        """
        Convert parameters into a GroupedTensor and re-register them as parameters.
        """

        weights = [getattr(self, f"weight{idx}") for idx in range(self.num_groups)]
        quantizer = self.get_quantizer("forward", 1)

        recipe = None if quantizer is None else quantizer._get_compatible_recipe()
        if recipe is not None and (recipe.delayed() or recipe.float8_current_scaling()):
            raise RuntimeError(
                "Delayed scaling or float8 current scaling is not supported with"
                " single_grouped_weight=True"
            )

        grouped_weights = GroupedTensor.make_grouped_tensor_with_shapes(
            num_tensors=self.num_groups,
            shapes=[(self.out_features, self.in_features)] * self.num_groups,
            quantizer=quantizer,
            dtype=self.weight0.dtype,
            device=self.weight0.device,
        )

        # Copy existing params into storage.
        with torch.no_grad():
            for i in range(self.num_groups):
                if self._with_quantized_weight:
                    grouped_weights.quantized_tensors[i].copy_from_storage(weights[i])
                else:
                    grouped_weights.quantized_tensors[i].copy_(weights[i])

        assert isinstance(grouped_weights, torch.Tensor) and (
            quantizer is None or not quantizer.internal
        ), "Found internal quantizer with `single_grouped_weight=True`."

        # Re-register as a single grouped weight parameter.
        self.register_parameter("weight", torch.nn.Parameter(grouped_weights))
        for group_idx in range(self.num_groups):
            self.register_parameter(f"weight{group_idx}", None)

        self._apply_delay_wgrad_param_hooks()

    def _make_grouped_biases_from_packed(self, packed_biases: torch.Tensor) -> None:
        """Replace per-group bias parameters with one ``GroupedTensor`` (``single_grouped_bias``)."""
        bias_data = packed_biases.detach().clone().contiguous()
        grouped_bias = GroupedTensor.make_grouped_tensor_from_rowwise_data(
            num_tensors=self.num_groups,
            tensor_shape=(self.out_features,),
            rowwise_data=bias_data,
            dtype=bias_data.dtype,
        )
        grouped_bias.requires_grad_(True)
        self.register_parameter("bias", torch.nn.Parameter(grouped_bias))
        for group_idx in range(self.num_groups):
            self.register_parameter(f"bias{group_idx}", None)

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

        # Check that all weight params are consistent
        if not self.single_grouped_weight:
            dtype = self.weight0.dtype
            device = self.weight0.device
            weight_requires_grad = self.weight0.requires_grad
            weight_tensor_type = type(self.weight0.data)
            for group_idx in range(self.num_groups):
                weight = getattr(self, f"weight{group_idx}")
                if weight.dtype != dtype:
                    raise RuntimeError(
                        f"Weight {group_idx} has invalid dtype (expected {dtype}, got"
                        f" {weight.dtype})."
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
        else:
            dtype = self.weight.dtype
            device = self.weight.device
            weight_requires_grad = self.weight.requires_grad
            weight_tensor_type = type(self.weight.data)

        # Check that biases are consistent
        if self.has_bias:
            if self.single_grouped_bias:
                bias = self.bias
                if bias.dtype != dtype:
                    raise RuntimeError(
                        f"Bias has invalid dtype (expected {dtype}, got {bias.dtype})."
                    )
                if not devices_match(bias.device, device):
                    raise RuntimeError(
                        f"Bias has invalid device (expected {device}, got {bias.device})."
                    )
                if bias.requires_grad != weight_requires_grad:
                    raise RuntimeError(
                        f"Bias has requires_grad={bias.requires_grad}, "
                        f"but expected requires_grad={weight_requires_grad}."
                    )
            else:
                for group_idx in range(self.num_groups):
                    bias = getattr(self, f"bias{group_idx}")
                    if bias is None:
                        raise RuntimeError(
                            f"Expected biases, but bias {group_idx} is uninitialized"
                        )
                    if bias.dtype != dtype:
                        raise RuntimeError(
                            f"Bias {group_idx} has invalid dtype (expected {dtype}, got"
                            f" {bias.dtype})."
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
            if self.single_grouped_bias:
                if getattr(self, "bias", None) is not None:
                    raise RuntimeError("Expected no biases, but grouped `bias` is registered")
            else:
                for group_idx in range(self.num_groups):
                    bias = getattr(self, f"bias{group_idx}")
                    if bias is not None:
                        raise RuntimeError(
                            f"Expected no biases, but bias {group_idx} is initialized"
                        )

    def pre_fuser_forward(self, *, requires_grad: bool) -> None:
        super().pre_fuser_forward(requires_grad=requires_grad)
        if FP8GlobalStateManager.is_fp8_enabled():
            # Assume weights have consistent grad requirement
            weight_requires_grad = (
                self.weight.requires_grad
                if self.single_grouped_weight
                else self.weight0.requires_grad
            )
            weight_requires_grad = requires_grad and weight_requires_grad

            # Configure quantizer usages
            for group_idx in range(self.num_groups):
                input_quantizer = self.get_quantizer("forward", 2 * group_idx)
                weight_quantizer = self.get_quantizer("forward", 2 * group_idx + 1)
                grad_output_quantizer = self.get_quantizer("backward", group_idx)
                input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
                weight_quantizer.set_usage(rowwise=True, columnwise=requires_grad)
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

            # Get weight tensor
            # Note: This function may be called in base class constructor,
            # before any grouped linear attrs have been set.
            weight = None
            weight_is_quantized = False
            if getattr(self, "single_grouped_weight", False):
                weight = getattr(self, "weight", None)
                weight_is_quantized = weight is not None and weight.quantizer is not None
            else:
                weight = getattr(self, f"weight{group_idx}", None)
                weight_is_quantized = is_quantized_tensor(weight)

            # Configure weight quantizer
            weight_quantizer = self.get_quantizer("forward", 2 * group_idx + 1)
            if weight_quantizer is not None:
                # Determine if quantized weight is exposed as parameter
                weight_quantizer.internal = not (
                    FP8GlobalStateManager.with_fp8_parameters()
                    or getattr(self, "_with_quantized_weight", False)
                    or weight_is_quantized
                )

            # Recipe-specific configuration
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

            # Update quantizer in quantized weight tensor
            if weight_quantizer is not None and weight_is_quantized:
                # Get quantizer from weight tensor
                weight_tensor_quantizer = (
                    weight.quantizer if self.single_grouped_weight else weight._quantizer
                )

                # Preserve existing usages in weight tensor. Even if a
                # usage is currently unnecessary, the weight tensor
                # may be used elsewhere.
                if weight_tensor_quantizer is not None:
                    weight_quantizer.set_usage(
                        rowwise=weight_tensor_quantizer.rowwise_usage,
                        columnwise=weight_tensor_quantizer.columnwise_usage,
                    )

                # Update weight tensor
                if self.single_grouped_weight:
                    if group_idx == 0:
                        weight.quantizer = weight_quantizer.copy()
                else:
                    weight.update_quantizer(weight_quantizer.copy())

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

    @staticmethod
    def _is_graph_safe_path_supported(
        *,
        with_quantized_compute: bool,
        input_quantizers: Sequence[Optional[Quantizer]],
        dtype: torch.dtype,
    ) -> bool:
        """Whether the graph-safe grouped-tensor flow can be used.

        * The graph-safe path dispatches to ``general_grouped_gemm_for_grouped_tensor``,
          which is backed by ``nvte_grouped_gemm_with_discrete_inputA`` in the common
          library. That kernel requires Blackwell (SM100) or newer with cuBLAS 13.3+.
        * Quantized compute is currently MXFP8-only; every other quantization
          recipe (fp8 delayed / current scaling, fp8 block scaling, NVFP4, ...)
          falls back to the legacy flow.
        * Unquantized compute supports BF16/FP16 only -- FP32 is excluded
          because the cublasLt grouped GEMM doesn't support it.
        """
        if get_device_compute_capability() < (10, 0):
            return False
        if with_quantized_compute:
            return all(isinstance(q, MXFP8Quantizer) for q in input_quantizers)
        return dtype in (torch.bfloat16, torch.float16)

    def _get_grouped_weight_for_gemm(
        self,
        weight_param: GroupedTensor,
        weight_quantizers: list[Optional[Quantizer]],
        columnwise_usage: bool,
        with_quantized_compute: bool,
        dtype: torch.dtype,
    ) -> GroupedTensor:
        """Prepare weights for ``general_grouped_gemm_for_grouped_tensor``.
        Supports MXFP8/BF16/FP16 compute paths.
        """
        num_groups = self.num_groups
        is_weight_quantized = weight_param.quantizer is not None
        if is_weight_quantized and with_quantized_compute:
            # GGEMM can use it as it is
            return weight_param
        if is_weight_quantized and not with_quantized_compute:
            # This use-case isnt optimized yet. Involves a per-group
            # dequantize loop and a torch.stack copy.
            weight_parts = weight_param.quantized_tensors
            if weight_parts is None:
                weight_parts = weight_param.split_into_quantized_tensors()
            dequantized = [maybe_dequantize(w, dtype) for w in weight_parts]
            weight_data = torch.stack(dequantized, dim=0).contiguous()
            return GroupedTensor(
                shape=(num_groups * self.out_features, self.in_features),
                dtype=dtype,
                num_tensors=num_groups,
                shapes=[(self.out_features, self.in_features)] * num_groups,
                quantizer=None,
                data=weight_data.reshape(-1),
            )
        if not with_quantized_compute:
            # Make sure that weight param is the correct dtype,
            # otherwise cast it to the correct dtype.
            if weight_param.rowwise_data.dtype == dtype:
                return weight_param
            weight_data = weight_param.rowwise_data.to(dtype=dtype)
            return GroupedTensor(
                shape=(num_groups * self.out_features, self.in_features),
                dtype=dtype,
                num_tensors=num_groups,
                shapes=[(self.out_features, self.in_features)] * num_groups,
                quantizer=None,
                data=weight_data.reshape(-1),
            )
        # Quantized compute path, use the fused group quantize kernel.
        weight_quantizer = weight_quantizers[0]
        weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
        return tex.group_quantize(
            weight_param.rowwise_data.view(weight_param.logical_shape),
            weight_quantizer,
            num_groups,
            None,
        )

    def _get_discrete_weights_for_gemm(
        self,
        weight_params: Optional[GroupedTensor] | list[torch.Tensor],
        weight_quantizers: list[Optional[Quantizer]],
        columnwise_usage: bool,
        with_quantized_compute: bool,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Prepare weights for ``general_grouped_gemm_for_grouped_tensor``.
        Returns a Python list, which dispatches the GEMM to ``discrete_in`` mode.
        """
        out: list[torch.Tensor] = []
        for w, quantizer in zip(weight_params, weight_quantizers):
            if not with_quantized_compute:
                w = maybe_dequantize(w, dtype)
            elif with_quantized_compute and not is_quantized_tensor(w):
                quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
                w = quantizer(w)
            out.append(w)
        return out

    def _get_weight_tensors(self) -> list[torch.nn.Parameter]:
        """Return the weight parameters in registration order.

        Length is 1 when ``single_grouped_weight=True`` (one
        ``GroupedTensor`` parameter), otherwise ``num_groups``.
        """
        if self.single_grouped_weight:
            return [self.weight]
        return [getattr(self, f"weight{idx}") for idx in range(self.num_groups)]

    def _get_grouped_bias_for_gemm(
        self,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Build a uniform GroupedTensor of per-group biases for the cublas
        grouped GEMM.

        Each group expects a (1, out_features) bias vector. Returns ``None``
        when no additive bias is configured.
        """
        if not self.has_bias:
            return None
        num_groups = self.num_groups

        if self.single_grouped_bias:
            # Already a contiguous (num_groups * out_features) buffer.
            bias_data = self.bias.rowwise_data
            if bias_data.dtype != dtype:
                bias_data = bias_data.to(dtype=dtype)
        else:
            bias_list = [
                maybe_dequantize(getattr(self, f"bias{idx}"), dtype) for idx in range(num_groups)
            ]
            bias_data = torch.stack(bias_list, dim=0).contiguous()

        return GroupedTensor(
            shape=(num_groups, self.out_features),
            dtype=dtype,
            num_tensors=num_groups,
            shapes=[(1, self.out_features)] * num_groups,
            quantizer=None,
            data=bias_data.reshape(-1),
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
        weight_param = self.weight if self.single_grouped_weight else self.weight0
        device = weight_param.device

        if self._accumulate_into_main_grad:
            if not hasattr(weight_param, "main_grad"):
                raise RuntimeError("MAIN GRAD NOT FOUND")
            if weight_param.main_grad is None:
                raise RuntimeError("MAIN GRAD IS NONE")

        # Check which grads are required
        ctx = basic_op_ctxs[0]
        input_requires_grad = ctx.requires_grad
        weight_requires_grad = ctx.requires_grad and weight_param.requires_grad

        # Quantizers
        input_quantizers = [None] * num_groups
        weight_quantizers = [None] * num_groups
        with_quantized_compute = FP8GlobalStateManager.is_fp8_enabled()
        if with_quantized_compute:
            for group_idx in range(num_groups):
                input_quantizers[group_idx] = self.get_quantizer("forward", 2 * group_idx)
                weight_quantizers[group_idx] = self.get_quantizer("forward", 2 * group_idx + 1)

        # Get autocast dtype if needed
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = weight_param.dtype

        # Extract split sizes from extra input. Keep on GPU for graph safety.
        split_sizes = basic_op_extra_inputs[0][0]
        if int(split_sizes.numel()) != num_groups:
            raise ValueError(f"Expected {num_groups} splits, but got {int(split_sizes.numel())}.")
        if split_sizes.dtype != torch.int64:
            split_sizes = split_sizes.to(dtype=torch.int64)
        if split_sizes.device != device:
            split_sizes = split_sizes.to(device=device)

        # Extract scales tensor for bias scaling
        scales = None
        if self._scale_bias:
            scales = basic_op_extra_inputs[0][1]

        # Dispatch: graph-safe GroupedTensor flow whenever it can be used.
        # See ``_is_graph_safe_path_supported`` for the gating rationale --
        # in short it requires Blackwell (SM100+) plus a supported dtype /
        # quantization recipe. Otherwise we fall back to the legacy
        # ``tex.split_quantize`` + ``general_grouped_gemm`` flow.
        use_grouped_tensor_path = self._is_graph_safe_path_supported(
            with_quantized_compute=with_quantized_compute,
            input_quantizers=input_quantizers,
            dtype=dtype,
        )

        if use_grouped_tensor_path:
            out, tensors_to_save = self._fuser_forward_grouped_tensor(
                input_=input_,
                split_sizes=split_sizes,
                scales=scales,
                with_quantized_compute=with_quantized_compute,
                input_quantizers=input_quantizers,
                weight_quantizers=weight_quantizers,
                dtype=dtype,
                input_requires_grad=input_requires_grad,
                weight_requires_grad=weight_requires_grad,
                device=device,
            )
        else:
            out, tensors_to_save = self._fuser_forward_split_quantize(
                input_=input_,
                split_sizes=split_sizes,
                scales=scales,
                with_quantized_compute=with_quantized_compute,
                input_quantizers=input_quantizers,
                weight_quantizers=weight_quantizers,
                dtype=dtype,
                input_requires_grad=input_requires_grad,
                weight_requires_grad=weight_requires_grad,
                device=device,
            )

        # Save tensors and autograd metadata on the basic-op context.
        self.fuser_forward_save_ctx(
            basic_op_ctxs=basic_op_ctxs,
            input_=input_,
            tensors_to_save=[tensors_to_save],
            requires_grad=[ctx.requires_grad],
            basic_op_extra_inputs=basic_op_extra_inputs,
            prev_op_grad_output_quantizer=prev_op_grad_output_quantizer,
            next_op_input_quantizer=next_op_input_quantizer,
            basic_op_kwargs=basic_op_kwargs,
            use_grouped_tensor_path=use_grouped_tensor_path,
        )

        return out, [()]

    def fuser_forward_save_ctx(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,  # pylint: disable=unused-argument
        tensors_to_save: list[
            tuple[Optional[torch.Tensor | QuantizedTensorStorage | GroupedTensorStorage], ...]
        ],
        *,
        requires_grad: list[bool],
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],  # pylint: disable=unused-argument
        prev_op_grad_output_quantizer: Optional[Quantizer],  # pylint: disable=unused-argument
        next_op_input_quantizer: Optional[Quantizer],  # pylint: disable=unused-argument
        basic_op_kwargs: list[dict[str, Any]],  # pylint: disable=unused-argument
        use_grouped_tensor_path: bool,
    ) -> None:
        """
        Save tensors and autograd metadata in context.
        """
        if not requires_grad[0]:
            return

        ctx = basic_op_ctxs[0]
        ctx.save_for_backward(*tensors_to_save[0])

        num_groups = self.num_groups
        weight_param = self.weight if self.single_grouped_weight else self.weight0

        with_quantized_compute = FP8GlobalStateManager.is_fp8_enabled()
        input_quantizers = [None] * num_groups
        weight_quantizers = [None] * num_groups
        grad_output_quantizers = [None] * num_groups
        if with_quantized_compute:
            for group_idx in range(num_groups):
                input_quantizers[group_idx] = self.get_quantizer("forward", 2 * group_idx)
                weight_quantizers[group_idx] = self.get_quantizer("forward", 2 * group_idx + 1)
                grad_output_quantizers[group_idx] = self.get_quantizer("backward", group_idx)

        ctx.use_grouped_tensor_path = use_grouped_tensor_path
        ctx.with_quantized_compute = with_quantized_compute
        ctx.input_quantizers = input_quantizers
        ctx.weight_quantizers = weight_quantizers
        ctx.grad_output_quantizers = grad_output_quantizers
        ctx.grad_input_quantizers = None
        # ``split_sizes`` and ``base_split_offsets`` are routed through
        # ``save_for_backward`` (see ``_fuser_forward_split_quantize`` and
        # ``_fuser_forward_grouped_tensor`` for the saved-tensor layout).
        if torch.is_autocast_enabled():
            ctx.dtype = torch.get_autocast_dtype("cuda")
        else:
            ctx.dtype = weight_param.dtype
        ctx.input_requires_grad = requires_grad[0]
        ctx.weight_requires_grad = requires_grad[0] and weight_param.requires_grad

    # ==================================================================
    # Legacy `tex.split_quantize` + `general_grouped_gemm` flow.
    # ``m_splits`` is needed on CPU here, so this flow is NOT cuda-graphable.
    # ==================================================================
    def _fuser_forward_split_quantize(
        self,
        *,
        input_: torch.Tensor,
        split_sizes: torch.Tensor,
        scales: Optional[torch.Tensor],
        with_quantized_compute: bool,
        input_quantizers: list[Optional[Quantizer]],
        weight_quantizers: list[Optional[Quantizer]],
        dtype: torch.dtype,
        input_requires_grad: bool,
        weight_requires_grad: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, tuple[Optional[torch.Tensor], ...]]:
        """Legacy ``tex.split_quantize`` + ``general_grouped_gemm`` flow."""
        num_groups = self.num_groups
        has_bias = self.has_bias

        # Need CPU split sizes for split_quantize / general_grouped_gemm.
        split_sizes_int = [int(s) for s in split_sizes.tolist()]

        # Extract params
        if self.single_grouped_weight:
            weights = self.weight.quantized_tensors
            if weights is None:
                weights = self.weight.split_into_quantized_tensors()
        else:
            weights = [getattr(self, f"weight{idx}") for idx in range(num_groups)]
        bs = None
        if has_bias:
            bs = self._get_bias_tensors(dtype)

        ws = self._get_discrete_weights_for_gemm(
            weights,
            weight_quantizers,
            columnwise_usage=input_requires_grad,
            with_quantized_compute=with_quantized_compute,
            dtype=dtype,
        )

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
        use_gemm_bias = has_bias and not self._scale_bias
        general_grouped_gemm(
            ws,
            xs,
            [out],
            [None] * num_groups,  # quantization_params
            dtype,
            m_splits=split_sizes_int,
            bias=bs if use_gemm_bias else None,
            use_bias=use_gemm_bias,
            use_split_accumulator=_2X_ACC_FPROP,
            single_output=True,
        )

        # Add bias * scales when scale_bias is enabled
        if self._scale_bias and has_bias:
            scales_splits = torch.split(scales, split_sizes_int)
            out_splits = torch.split(out, split_sizes_int)
            for i in range(num_groups):
                out_splits[i].add_(bs[i].unsqueeze(0) * scales_splits[i].unsqueeze(-1))

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

        # Build the tuple of tensors to save for backward. Layout:
        #   [split_sizes, base_split_offsets, split_points,
        #    (scales if scale_bias), *xs, *ws]
        # ``base_split_offsets`` and ``split_points`` are unused on the
        # split-quantize backward path but are included as ``None`` so the
        # saved-tensor layout matches the graph-safe
        # ``_fuser_forward_grouped_tensor`` path (and the fused MLP forward).
        saved: list[Optional[torch.Tensor]] = [split_sizes, None, None]
        if self._scale_bias:
            saved.append(scales)
        saved.extend(xs)
        saved.extend(ws)
        return out, tuple(saved)

    def _fuser_forward_grouped_tensor(
        self,
        *,
        input_: torch.Tensor,
        split_sizes: torch.Tensor,
        scales: Optional[torch.Tensor],
        with_quantized_compute: bool,
        input_quantizers: list[Optional[Quantizer]],
        weight_quantizers: list[Optional[Quantizer]],
        dtype: torch.dtype,
        input_requires_grad: bool,
        weight_requires_grad: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, tuple[Optional[torch.Tensor], ...]]:
        """Graph-safe GroupedTensor forward path (pure compute).
        Returns ``(output, tensors_to_save)``. ``split_sizes``,
        ``base_split_offsets`` and ``split_points`` are returned so that
        ``fuser_forward_save_ctx`` can call ``save_for_backward`` on them.
        """
        num_groups = self.num_groups
        has_bias = self.has_bias

        base_split_offsets = tex.splits_to_offsets(split_sizes, 1)
        split_points = base_split_offsets[1:].to(dtype=torch.int)

        # Flatten to 2D so the first dim is the total token count.
        original_shape = list(input_.size())
        x = maybe_dequantize(input_, dtype).reshape(-1, self.in_features)
        total_tokens = x.size(0)

        # Build the input GroupedTensor.
        if with_quantized_compute:
            input_quantizer = input_quantizers[0]
            input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
            input_quantizer.optimize_for_gemm = True
            grouped_x = tex.group_quantize(x, input_quantizer, num_groups, split_sizes)
        else:
            # No quantize: wrap the contiguous high-precision buffer.
            grouped_x = GroupedTensor(
                shape=(total_tokens, self.in_features),
                dtype=dtype,
                num_tensors=num_groups,
                quantizer=None,
                data=x.reshape(-1),
                first_dims=split_sizes,
                tensor_offsets=base_split_offsets * self.in_features,
            )

        # Build the weight GroupedTensor / list.
        if self.single_grouped_weight:
            # GroupedTensor
            grouped_weights = self._get_grouped_weight_for_gemm(
                self.weight,
                weight_quantizers,
                columnwise_usage=input_requires_grad,
                with_quantized_compute=with_quantized_compute,
                dtype=dtype,
            )
        else:
            # Discrete weights
            grouped_weights = self._get_discrete_weights_for_gemm(
                [getattr(self, f"weight{idx}") for idx in range(num_groups)],
                weight_quantizers,
                columnwise_usage=input_requires_grad,
                with_quantized_compute=with_quantized_compute,
                dtype=dtype,
            )

        # Allocate output buffer and wrap as a GroupedTensor view.
        out_shape = original_shape[:-1] + [self.out_features]
        out = torch.empty(out_shape, dtype=dtype, device=device)
        grouped_out = GroupedTensor(
            shape=(total_tokens, self.out_features),
            dtype=dtype,
            num_tensors=num_groups,
            quantizer=None,
            data=out.reshape(-1),
            first_dims=split_sizes,
            tensor_offsets=base_split_offsets * self.out_features,
        )

        # Bias: hand off to the grouped GEMM (graph-safe, fused). Plain bias
        # uses ``bias=``; scaled bias also passes per-token ``bias_scale=``.
        grouped_bias = None
        bias_scale: Optional[torch.Tensor] = None
        if has_bias:
            # Bias always needs to be passed as a GroupedTensor for the grouped GEMM.
            grouped_bias = self._get_grouped_bias_for_gemm(dtype)
            if self._scale_bias:
                bias_scale = scales.reshape(-1)
                if bias_scale.dtype != torch.float32:
                    bias_scale = bias_scale.to(dtype=torch.float32)

        # Forward grouped GEMM.
        general_grouped_gemm_for_grouped_tensor(
            grouped_weights,
            grouped_x,
            grouped_out,
            layout="TN",
            use_split_accumulator=_2X_ACC_FPROP,
            bias=grouped_bias,
            bias_scale=bias_scale,
        )

        if not input_requires_grad:
            grouped_weights = None if self.single_grouped_weight else [None] * num_groups

        if not weight_requires_grad:
            grouped_x = None

        # Build the tuple of tensors to save for backward. Layout:
        #   [split_sizes, base_split_offsets, split_points,
        #    (scales if _scale_bias), grouped_x, *weights]
        if grouped_x is not None:
            if with_quantized_compute:
                # only columnwise data is needed for wgrad
                grouped_x.rowwise_data = None
                grouped_x.scale_inv = None
        saved: list[Optional[torch.Tensor]] = [split_sizes, base_split_offsets, split_points]
        if self._scale_bias:
            saved.append(scales)
        saved.append(grouped_x)
        if self.single_grouped_weight:
            saved.append(grouped_weights)
        else:
            saved.extend(grouped_weights)
        return out, tuple(saved)

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
        ctx = basic_op_ctxs[0]
        # Dispatch to the path used in forward (saved as ``ctx.use_grouped_tensor_path``).
        if getattr(ctx, "use_grouped_tensor_path", False):
            return self._fuser_backward_grouped_tensor(
                ctx=ctx,
                grad_output=grad_output,
            )
        return self._fuser_backward_split_quantize(
            ctx=ctx,
            grad_output=grad_output,
        )

    def _fuser_backward_split_quantize(
        self,
        *,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        Iterable[Iterable[Optional[torch.Tensor]]],
        Iterable[Iterable[Optional[torch.Tensor]]],
    ]:
        num_groups = self.num_groups
        has_bias = self.has_bias
        weights = self._get_weight_tensors()
        device = weights[0].device

        # Saved tensors from forward pass. Layout:
        #   [split_sizes, base_split_offsets, split_points,
        #    (scales if _scale_bias), *xs, *ws]
        # ``base_split_offsets`` and ``split_points`` are unused on this path
        # but are present so the saved-tensor layout matches the graph-safe
        # path (and the fused MLP forward).
        saved_tensors = ctx.saved_tensors
        split_sizes = saved_tensors[0]
        saved_tensors = saved_tensors[3:]
        scales = None
        if self._scale_bias:
            scales, saved_tensors = saved_tensors[0], saved_tensors[1:]
        xs, saved_tensors = saved_tensors[:num_groups], saved_tensors[num_groups:]
        ws, saved_tensors = saved_tensors[:num_groups], saved_tensors[num_groups:]

        # Split grad output tensor and convert dtypes if needed
        split_sizes_int = [int(s) for s in split_sizes.tolist()]
        dy = maybe_dequantize(grad_output, ctx.dtype)
        dys = None
        grad_biases = [None] * num_groups
        grad_scales = None
        if ctx.with_quantized_compute:
            for quantizer in ctx.grad_output_quantizers:
                quantizer.set_usage(
                    rowwise=ctx.input_requires_grad,
                    columnwise=ctx.weight_requires_grad,
                )
            dys = tex.split_quantize(dy, split_sizes_int, ctx.grad_output_quantizers)
        else:
            dys = torch.split(dy, split_sizes_int)

        if has_bias:
            dy_2d = dy.reshape(-1, dy.size(-1))
            offsets = torch.zeros(num_groups + 1, dtype=torch.int64, device=device)
            offsets[1:] = split_sizes.cumsum(0)
            if self._scale_bias:
                bias_packed = torch.stack(self._get_bias_tensors(ctx.dtype))
                scales_f32 = scales.to(dtype=torch.float32)
                dbias_packed, grad_scales = compute_grouped_dbias_dscales(
                    dy_2d,
                    scales_f32,
                    bias_packed,
                    offsets=offsets,
                )
            else:
                dbias_packed = compute_grouped_dbias(dy_2d, offsets, num_groups)
            grad_biases = [dbias_packed[idx].to(dtype=ctx.dtype) for idx in range(num_groups)]

        # Initialize grad weight buffers.
        accumulate_into_main_grad = self._accumulate_into_main_grad
        grad_weights = [None] * num_groups
        final_weight_grads: list[Optional[torch.Tensor]] = (
            [None] if self.single_grouped_weight else [None] * num_groups
        )
        if ctx.weight_requires_grad:
            weight_shape = (self.out_features, self.in_features)
            grouped_shape = (num_groups, *weight_shape)
            if self.single_grouped_weight:
                if accumulate_into_main_grad:
                    # Megatron-LM wgrad fusion: GEMM accumulates into the
                    # parameter's ``main_grad`` directly.
                    main_grad = get_main_grad_from_param(weights[0], op_label="GroupedLinear")
                    main_grad = view_main_grad_as_grouped_buffer(
                        main_grad, num_groups, weight_shape, label="GroupedLinear weight"
                    )
                    final_weight_grads[0] = main_grad
                    grad_weights = [main_grad[idx] for idx in range(num_groups)]
                    accumulate_into_main_grad = get_accumulate_flag_in_param(weights[0])
                else:
                    final_weight_grads[0] = torch.empty(
                        grouped_shape, dtype=ctx.dtype, device=device
                    )
                    grad_weights = [final_weight_grads[0][idx] for idx in range(num_groups)]
            else:
                if accumulate_into_main_grad:
                    grad_weights = [
                        get_main_grad_from_param(w, op_label="GroupedLinear") for w in weights
                    ]
                    accumulate_into_main_grad = get_accumulate_flag_in_param(weights[0])
                else:
                    grad_weights = tex.bulk_allocate(
                        [weight_shape] * num_groups,
                        [ctx.dtype] * num_groups,
                        device,
                        [256] * num_groups,  # alignment
                    )
                final_weight_grads = list(grad_weights)

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
        delay_wgrad = (
            ctx.weight_requires_grad
            and self.wgrad_store is not None
            and self.wgrad_store.delay_wgrad_compute()
        )
        if ctx.weight_requires_grad:
            if delay_wgrad:
                grouped_gemm_wgrad = functools.partial(
                    general_grouped_gemm,
                    quantization_params=[None] * num_groups,
                    out_dtype=ctx.dtype,
                    layout="NT",
                    m_splits=split_sizes_int,
                    use_split_accumulator=_2X_ACC_WGRAD,
                    accumulate=accumulate_into_main_grad,
                )
                self.wgrad_store.put([xs, dys, grad_weights], grouped_gemm_wgrad)
            else:
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

        if not delay_wgrad:
            clear_tensor_data(*xs)

        # Megatron-LM wgrad fusion: regardless of overwrite vs. accumulate,
        # signal that ``main_grad`` already carries the wgrad and replace
        # ``.grad`` with a dummy so DDP/FSDP hooks won't add ``.grad`` into
        # ``main_grad`` again.
        if ctx.weight_requires_grad and self._accumulate_into_main_grad:
            final_weight_grads = get_dummy_wgrads_for_params(weights)
        elif ctx.weight_requires_grad and delay_wgrad:
            final_weight_grads = [None] if self.single_grouped_weight else [None] * num_groups

        if not has_bias:
            grad_params = list(final_weight_grads)
        elif self.single_grouped_bias:
            final_bias_grads = torch.stack(grad_biases, dim=0).to(ctx.dtype)
            grad_params = list(final_weight_grads) + [final_bias_grads]
        else:
            if self.single_grouped_weight:
                grad_params = list(grad_biases) + list(final_weight_grads)
            else:
                grad_params = list(final_weight_grads) + list(grad_biases)

        grad_extra = (None, grad_scales) if self._scale_bias else (None,)
        return grad_input, [grad_params], [grad_extra]

    # ==================================================================
    # Graph-safe backward: counterpart of `_fuser_forward_grouped_tensor`.
    # ==================================================================
    def _fuser_backward_grouped_tensor(
        self,
        *,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        Iterable[Iterable[Optional[torch.Tensor]]],
        Iterable[Iterable[Optional[torch.Tensor]]],
    ]:
        num_groups = self.num_groups
        has_bias = self.has_bias
        weights = self._get_weight_tensors()
        device = weights[0].device
        dtype = ctx.dtype

        with_quantized_compute = bool(getattr(ctx, "with_quantized_compute", False))

        # Saved tensors from forward pass
        # Layout: [split_sizes, base_split_offsets, split_points,
        #          (scales if _scale_bias), grouped_x, *weights]
        # ``split_points`` is unused on this path but is present so the
        # saved-tensor layout matches the fused MLP forward (which needs it
        # for the cuDNN grouped GEMM kernel).
        saved_tensors = ctx.saved_tensors
        split_sizes = saved_tensors[0]
        base_split_offsets = saved_tensors[1]
        saved_tensors = saved_tensors[3:]
        scales = None
        if self._scale_bias:
            scales, saved_tensors = saved_tensors[0], saved_tensors[1:]
        grouped_x, saved_tensors = saved_tensors[0], saved_tensors[1:]
        if self.single_grouped_weight:
            ws, saved_tensors = saved_tensors[0], saved_tensors[1:]
        else:
            ws, saved_tensors = saved_tensors[:num_groups], saved_tensors[num_groups:]

        # Flatten grad_output to 2D (total_tokens, out_features)
        # to figure out total tokens.
        dy_2d = grad_output.reshape(-1, self.out_features)
        total_tokens = dy_2d.size(0)

        # Build the grad_output GroupedTensor.
        # Optionally get dbias is fusion available with bgrad_group_quantize
        dbias_packed = None
        if with_quantized_compute:
            grad_output_quantizer = ctx.grad_output_quantizers[0]
            grad_output_quantizer.set_usage(
                rowwise=ctx.input_requires_grad, columnwise=ctx.weight_requires_grad
            )
            grad_output_quantizer.optimize_for_gemm = True

            if has_bias and not self._scale_bias:
                grouped_dy, dbias_packed = tex.bgrad_group_quantize(
                    dy_2d, grad_output_quantizer, num_groups, split_sizes
                )
            else:
                grouped_dy = tex.group_quantize(
                    dy_2d, grad_output_quantizer, num_groups, split_sizes
                )
        else:
            dy_2d = maybe_dequantize(dy_2d, dtype)
            # Wrap BF16/FP16 buffer as a GroupedTensor for grouped gemm
            grouped_dy = GroupedTensor(
                shape=(total_tokens, self.out_features),
                dtype=dtype,
                num_tensors=num_groups,
                quantizer=None,
                data=dy_2d.reshape(-1),
                first_dims=split_sizes,
                tensor_offsets=base_split_offsets * self.out_features,
            )

        # Bias Grads compute if not already computed in bgrad_group_quantize
        final_bias_grads: Optional[torch.Tensor] = None
        grad_scales: Optional[torch.Tensor] = None
        if has_bias:
            if self._scale_bias:
                bias_packed = torch.stack(self._get_bias_tensors(dtype))
                scales_f32 = scales.to(dtype=torch.float32)
                dbias_packed, grad_scales = compute_grouped_dbias_dscales(
                    dy_2d,
                    scales_f32,
                    bias_packed,
                    offsets=base_split_offsets,
                )
            elif dbias_packed is None:
                # BF16/FP16 path
                dbias_packed = compute_grouped_dbias(dy_2d, base_split_offsets, num_groups)
            if self.single_grouped_bias:
                final_bias_grads = [dbias_packed.to(dtype=dtype)]
            else:
                final_bias_grads = [dbias_packed[idx].to(dtype=dtype) for idx in range(num_groups)]

        # ---- dgrad GEMM ----------------------------------------------------
        grad_input = None
        if ctx.input_requires_grad:
            grad_input_shape = list(grad_output.size())[:-1] + [self.in_features]
            grad_input = torch.empty(grad_input_shape, dtype=dtype, device=device)
            grouped_grad_input = GroupedTensor(
                shape=(total_tokens, self.in_features),
                dtype=dtype,
                num_tensors=num_groups,
                quantizer=None,
                data=grad_input.reshape(-1),
                first_dims=split_sizes,
                tensor_offsets=base_split_offsets * self.in_features,
            )
            general_grouped_gemm_for_grouped_tensor(
                ws,
                grouped_dy,
                grouped_grad_input,
                layout="NN",
                use_split_accumulator=_2X_ACC_DGRAD,
            )

        # params init for wgrad GEMM
        accumulate_into_main_grad = False
        weight_shape = (self.out_features, self.in_features)
        wgrad_output: Any = None
        grouped_wgrad: Optional[GroupedTensor] = None
        final_weight_grads: list[Optional[torch.Tensor]] = (
            [None] if self.single_grouped_weight else [None] * num_groups
        )

        # Get the right wgrad buffers for grouped gemm.
        # Can be a GroupedTensor or list of tensors based on single_grouped_weight.
        if ctx.weight_requires_grad:
            if self.single_grouped_weight:
                if self._accumulate_into_main_grad:
                    # Main-grad fusion: GEMM writes directly into ``main_grad``.
                    # ``overwrite_main_grad`` only flips the GEMM's
                    # ``accumulate`` flag.
                    main_grad = get_main_grad_from_param(weights[0], op_label="GroupedLinear")
                    main_grad = view_main_grad_as_grouped_buffer(
                        main_grad, num_groups, weight_shape, label="GroupedLinear weight"
                    )
                    grouped_wgrad = GroupedTensor.make_grouped_tensor_from_rowwise_data(
                        num_tensors=num_groups,
                        tensor_shape=weight_shape,
                        rowwise_data=main_grad.view(-1),
                        dtype=main_grad.dtype,
                    )
                    accumulate_into_main_grad = get_accumulate_flag_in_param(weights[0])
                else:
                    grouped_wgrad = GroupedTensor.make_grouped_tensor_with_shapes(
                        num_tensors=num_groups,
                        shapes=[weight_shape] * num_groups,
                        quantizer=None,
                        device=device,
                        dtype=dtype,
                    )
                final_weight_grads[0] = grouped_wgrad.rowwise_data.view(num_groups, *weight_shape)
                wgrad_output = grouped_wgrad
            else:
                if self._accumulate_into_main_grad:
                    final_weight_grads = [
                        get_main_grad_from_param(w, op_label="GroupedLinear") for w in weights
                    ]
                    accumulate_into_main_grad = get_accumulate_flag_in_param(weights[0])
                else:
                    final_weight_grads = [
                        torch.empty(weight_shape, dtype=dtype, device=device)
                        for _ in range(num_groups)
                    ]
                wgrad_output = final_weight_grads

        # wgrad GEMM
        delay_wgrad = (
            ctx.weight_requires_grad
            and self.wgrad_store is not None
            and self.wgrad_store.delay_wgrad_compute()
        )
        if ctx.weight_requires_grad:
            wgrad_gemm = functools.partial(
                general_grouped_gemm_for_grouped_tensor,
                layout="NT",
                accumulate=accumulate_into_main_grad,
                use_split_accumulator=_2X_ACC_WGRAD,
            )
            if delay_wgrad:
                self.wgrad_store.put([grouped_x, grouped_dy, wgrad_output], wgrad_gemm)
            else:
                wgrad_gemm(grouped_x, grouped_dy, wgrad_output)

        # Megatron-LM wgrad fusion: regardless of overwrite vs. accumulate,
        # signal that ``main_grad`` already carries the wgrad and replace
        # ``.grad`` with a dummy so DDP/FSDP hooks won't add ``.grad`` into
        # ``main_grad`` again.
        if ctx.weight_requires_grad and self._accumulate_into_main_grad:
            final_weight_grads = get_dummy_wgrads_for_params(weights)
        elif ctx.weight_requires_grad and delay_wgrad:
            final_weight_grads = [None] if self.single_grouped_weight else [None] * num_groups

        # Assemble grad params in parameter registration order and return.
        if not has_bias:
            grad_params = final_weight_grads
        elif self.single_grouped_bias:
            grad_params = final_weight_grads + final_bias_grads
        else:
            if self.single_grouped_weight:
                grad_params = final_bias_grads + final_weight_grads
            else:
                grad_params = final_weight_grads + final_bias_grads

        grad_extra = (None, grad_scales) if self._scale_bias else (None,)
        return grad_input, [grad_params], [grad_extra]

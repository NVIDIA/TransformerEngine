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
from ...cpp_extensions import general_grouped_gemm_for_grouped_tensor
from ...distributed import CudaRNGStatesTracker
from ...module._common import WeightGradStore
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
from ...tensor import GroupedTensor
from ...tensor.storage.grouped_tensor_storage import GroupedTensorStorage
from ...triton.grouped_dbias_dscales import _compute_grouped_dbias_dscales


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
    delay_wgrad_compute : bool, default = ``False``
        Whether to delay weight gradient computation
    single_grouped_bias : bool, default = ``False``
        If ``True`` (and ``bias=True``), store all expert biases as one ``GroupedTensor``
        parameter named ``bias`` instead of ``bias0``..``bias{N-1}``.
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

        # Weight tensor dimensions
        self.num_groups: int = num_groups
        self.in_features: int = in_features
        self.out_features: int = out_features
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
            clear_tensor_data(
                activations.rowwise_data,
                activations.columnwise_data,
                activations.scale_inv,
                activations.columnwise_scale_inv,
            )
        if self._accumulate_into_main_grad:
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
                if self.single_grouped_weight:
                    self.weight.quantizer = weight_quantizer.copy()
                else:
                    getattr(self, f"weight{group_idx}").update_quantizer(weight_quantizer.copy())
            else:
                # Use internal tensors if quantized weights will not be
                # exposed externally
                weight_quantizer.internal = (
                    not FP8GlobalStateManager.with_fp8_parameters()
                    and not getattr(self, "_with_quantized_weight", False)
                    and not self.single_grouped_weight
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
            dtype = weight_param.dtype

        # Extract split sizes from extra input
        split_sizes = basic_op_extra_inputs[0][0]
        split_sizes_int = [int(s) for s in split_sizes.tolist()]
        if len(split_sizes_int) != num_groups:
            raise ValueError(f"Expected {num_groups} splits, but got {len(split_sizes_int)}.")

        # Extract scales tensor for bias scaling
        scales = None
        if self._scale_bias:
            scales = basic_op_extra_inputs[0][1]

        # Extract bias tensors (common for both paths)
        bs = None
        if has_bias:
            if self.single_grouped_bias:
                bias_parts = self.bias.quantized_tensors
                if bias_parts is None:
                    bias_parts = self.bias.split_into_quantized_tensors()
                bs = [maybe_dequantize(p.reshape(-1), dtype) for p in bias_parts]
            else:
                bs = [
                    maybe_dequantize(getattr(self, f"bias{idx}"), dtype)
                    for idx in range(num_groups)
                ]

        in_shape = list(input_.size())
        out_shape = in_shape[:-1] + [self.out_features]
        split_sizes_gpu = split_sizes.to(dtype=torch.int64, device=device)

        # Build GroupedTensor for bias if needed
        grouped_bias = None
        bias_scale = None
        if has_bias:
            bias_data = torch.cat([b.reshape(-1) for b in bs])
            grouped_bias = GroupedTensor(
                shape=(num_groups, self.out_features),
                dtype=dtype,
                num_tensors=num_groups,
                data=bias_data,
            )
            if self._scale_bias:
                bias_scale = scales

        if with_quantized_compute:
            # ---- Quantized path: group_quantize + grouped GEMM ----
            x = maybe_dequantize(input_, dtype)
            x_2d = x.reshape(-1, self.in_features)
            total_tokens = x_2d.size(0)

            input_quantizer = input_quantizers[0]
            input_quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
            input_quantizer.optimize_for_gemm = True
            grouped_x = tex.group_quantize(
                x_2d, input_quantizer, num_groups, split_sizes_gpu
            )

            # Prepare weights
            ws = None
            grouped_w = None
            if self.single_grouped_weight:
                grouped_w = self.weight
                if isinstance(grouped_w, GroupedTensor):
                    w_quantizer = weight_quantizers[0]
                    w_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                    grouped_w = tex.group_quantize(
                        grouped_w.rowwise_data.view(grouped_w.logical_shape),
                        w_quantizer,
                        num_groups,
                        None,
                    )
            else:
                weights = [getattr(self, f"weight{idx}") for idx in range(num_groups)]
                ws = []
                for w, quantizer in zip(weights, weight_quantizers):
                    if not is_quantized_tensor(w):
                        quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                        w = quantizer(w)
                    ws.append(w)

            # Create grouped output
            tensor_offsets_out = GroupedTensorStorage.make_tensor_offsets(
                split_sizes_gpu, self.out_features
            )
            out_data = torch.empty(
                total_tokens * self.out_features, dtype=dtype, device=device
            )
            grouped_out = GroupedTensor(
                shape=(total_tokens, self.out_features),
                dtype=dtype,
                num_tensors=num_groups,
                data=out_data,
                first_dims=split_sizes_gpu,
                tensor_offsets=tensor_offsets_out,
            )

            gemm_a = grouped_w if self.single_grouped_weight else ws
            general_grouped_gemm_for_grouped_tensor(
                gemm_a,
                grouped_x,
                grouped_out,
                layout="TN",
                use_split_accumulator=_2X_ACC_FPROP,
                bias=grouped_bias,
                bias_scale=bias_scale,
            )
            out = out_data.view(out_shape)

            # Save state for backward pass
            if ctx.requires_grad:
                saved = [split_sizes_gpu]
                if self._scale_bias:
                    saved.append(scales)
                if self.single_grouped_weight:
                    saved.append(grouped_w if input_requires_grad else None)
                else:
                    for w in ws:
                        if not input_requires_grad:
                            saved.append(None)
                        else:
                            w.update_usage(rowwise_usage=False, columnwise_usage=True)
                            saved.append(w)
                if weight_requires_grad:
                    x_tensor_offsets = GroupedTensorStorage.make_tensor_offsets(
                        split_sizes_gpu, self.in_features
                    )
                    saved.extend([
                        grouped_x.columnwise_data,
                        grouped_x.columnwise_scale_inv,
                        x_tensor_offsets,
                    ])
                else:
                    saved.extend([None, None, None])
                ctx.save_for_backward(*saved)
                ctx.with_quantized_compute = True
                ctx.input_quantizer = input_quantizer
                ctx.grad_output_quantizers = grad_output_quantizers
                ctx.dtype = dtype
                ctx.input_requires_grad = input_requires_grad
                ctx.weight_requires_grad = weight_requires_grad

        else:
            # ---- Non-quantized path: GroupedTensor + grouped GEMM ----
            x = maybe_dequantize(input_, dtype)
            x_2d = x.reshape(-1, self.in_features)
            total_tokens = x_2d.size(0)

            tensor_offsets_x = GroupedTensorStorage.make_tensor_offsets(
                split_sizes_gpu, self.in_features
            )
            grouped_x = GroupedTensor(
                shape=(total_tokens, self.in_features),
                dtype=dtype,
                num_tensors=num_groups,
                data=x_2d.reshape(-1),
                first_dims=split_sizes_gpu,
                tensor_offsets=tensor_offsets_x,
            )

            if self.single_grouped_weight:
                weights = self.weight.quantized_tensors
                if weights is None:
                    weights = self.weight.split_into_quantized_tensors()
            else:
                weights = [getattr(self, f"weight{idx}") for idx in range(num_groups)]
            ws = [maybe_dequantize(w, dtype) for w in weights]

            tensor_offsets_out = GroupedTensorStorage.make_tensor_offsets(
                split_sizes_gpu, self.out_features
            )
            out_data = torch.empty(
                total_tokens * self.out_features, dtype=dtype, device=device
            )
            grouped_out = GroupedTensor(
                shape=(total_tokens, self.out_features),
                dtype=dtype,
                num_tensors=num_groups,
                data=out_data,
                first_dims=split_sizes_gpu,
                tensor_offsets=tensor_offsets_out,
            )

            general_grouped_gemm_for_grouped_tensor(
                ws,
                grouped_x,
                grouped_out,
                layout="TN",
                use_split_accumulator=_2X_ACC_FPROP,
                bias=grouped_bias,
                bias_scale=bias_scale,
            )
            out = out_data.view(out_shape)

            if not input_requires_grad:
                ws = [None] * num_groups

            if ctx.requires_grad:
                saved = [split_sizes_gpu]
                if self._scale_bias:
                    saved.append(scales)
                saved.extend(ws)
                if weight_requires_grad:
                    saved.extend([grouped_x.rowwise_data, None, tensor_offsets_x])
                else:
                    saved.extend([None, None, None])
                ctx.save_for_backward(*saved)
                ctx.with_quantized_compute = False
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
        weight_param = self.weight if self.single_grouped_weight else self.weight0
        device = weight_param.device

        # Saved tensors from forward pass
        ctx = basic_op_ctxs[0]
        saved_tensors = ctx.saved_tensors
        split_sizes, saved_tensors = saved_tensors[0], saved_tensors[1:]
        scales = None
        if self._scale_bias:
            scales, saved_tensors = saved_tensors[0], saved_tensors[1:]

        # Restore saved weights and input data based on forward path
        ws = None
        grouped_w_saved = None
        x_saved_data = None
        x_saved_scale = None
        x_tensor_offsets = None
        if ctx.with_quantized_compute and self.single_grouped_weight:
            grouped_w_saved, saved_tensors = saved_tensors[0], saved_tensors[1:]
        else:
            ws = list(saved_tensors[:num_groups])
            saved_tensors = saved_tensors[num_groups:]
        x_saved_data, x_saved_scale, x_tensor_offsets = (
            saved_tensors[0],
            saved_tensors[1],
            saved_tensors[2],
        )

        # Split grad output tensor and convert dtypes if needed
        split_sizes_int = [int(s) for s in split_sizes.tolist()]
        dy = maybe_dequantize(grad_output, ctx.dtype)
        grouped_dy = None
        grad_biases = [None] * num_groups
        grad_scales = None
        dy_2d = dy.reshape(-1, self.out_features)
        split_sizes_gpu = split_sizes.to(dtype=torch.int64, device=device)

        if ctx.with_quantized_compute:
            grad_output_quantizer = ctx.grad_output_quantizers[0]
            grad_output_quantizer.set_usage(
                rowwise=ctx.input_requires_grad,
                columnwise=ctx.weight_requires_grad,
            )
            grad_output_quantizer.optimize_for_gemm = True
            grouped_dy = tex.group_quantize(
                dy_2d, grad_output_quantizer, num_groups, split_sizes_gpu
            )
        else:
            tensor_offsets_dy = GroupedTensorStorage.make_tensor_offsets(
                split_sizes_gpu, self.out_features
            )
            grouped_dy = GroupedTensor(
                shape=(dy_2d.size(0), self.out_features),
                dtype=ctx.dtype,
                num_tensors=num_groups,
                data=dy_2d.reshape(-1),
                first_dims=split_sizes_gpu,
                tensor_offsets=tensor_offsets_dy,
            )

        if has_bias and not self._scale_bias:
            dy_splits = list(torch.split(grad_output, split_sizes_int))
            grad_biases = [
                dy_s.reshape(-1, dy_s.size(-1)).sum(dim=0) for dy_s in dy_splits
            ]

        if self._scale_bias and has_bias:
            bias_packed = torch.stack(self._get_bias_tensors(ctx.dtype))
            scales_f32 = scales.to(dtype=torch.float32)
            offsets = torch.zeros(num_groups + 1, dtype=torch.int64, device=device)
            offsets[1:] = split_sizes.cumsum(0)
            dy_2d_bias = dy.reshape(-1, dy.size(-1))
            dbias_packed, grad_scales = _compute_grouped_dbias_dscales(
                dy_2d_bias,
                scales_f32,
                bias_packed,
                offsets=offsets,
            )
            grad_biases = [dbias_packed[idx] for idx in range(num_groups)]

        # Initialize grad weight buffers
        accumulate_into_main_grad = self._accumulate_into_main_grad
        weight_shape = (self.out_features, self.in_features)
        grad_weights = [None] * num_groups
        grouped_wgrad = None
        wgrad_output = None
        if ctx.weight_requires_grad:
            if accumulate_into_main_grad:
                if self.single_grouped_weight:
                    if hasattr(weight_param, "__fsdp_param__"):
                        weight_param.main_grad = weight_param.get_main_grad()
                    main_grad = weight_param.main_grad
                    grouped_shape = (num_groups, *weight_shape)
                    if isinstance(main_grad, GroupedTensor):
                        grouped_wgrad = main_grad
                    else:
                        if main_grad.shape != grouped_shape:
                            if main_grad.numel() != math.prod(grouped_shape):
                                raise RuntimeError(
                                    "GroupedLinear expected grouped weight main_grad "
                                    f"to have shape {grouped_shape} or matching "
                                    f"numel, but got shape {tuple(main_grad.shape)}"
                                )
                            main_grad = main_grad.reshape(grouped_shape)
                        grouped_wgrad = (
                            GroupedTensor.make_grouped_tensor_from_rowwise_data(
                                num_tensors=num_groups,
                                tensor_shape=weight_shape,
                                rowwise_data=main_grad,
                                dtype=main_grad.dtype,
                            )
                        )
                    accumulate_into_main_grad = not getattr(
                        weight_param, "overwrite_main_grad", False
                    )
                else:
                    for group_idx in range(num_groups):
                        wp = getattr(self, f"weight{group_idx}")
                        if hasattr(wp, "__fsdp_param__"):
                            wp.main_grad = wp.get_main_grad()
                        grad_weights[group_idx] = wp.main_grad
                    accumulate_into_main_grad = not getattr(
                        self.weight0, "overwrite_main_grad", False
                    )
            else:
                if self.single_grouped_weight:
                    grouped_wgrad = GroupedTensor.make_grouped_tensor_with_shapes(
                        num_tensors=num_groups,
                        shapes=[weight_shape] * num_groups,
                        quantizer=None,
                        device=device,
                        dtype=ctx.dtype,
                    )
                else:
                    for group_idx in range(num_groups):
                        grad_weights[group_idx] = torch.empty(
                            weight_shape,
                            dtype=ctx.dtype,
                            device=device,
                        )
        else:
            accumulate_into_main_grad = False

        if self.single_grouped_weight:
            wgrad_output = grouped_wgrad
        else:
            wgrad_output = grad_weights

        # Perform dgrad and wgrad GEMMs
        delay_wgrad = (
            ctx.weight_requires_grad
            and self.wgrad_store is not None
            and self.wgrad_store.delay_wgrad_compute()
        )

        total_tokens = dy_2d.size(0)

        # Dgrad
        grad_input = None
        if ctx.input_requires_grad:
            out_shape = list(grad_output.size())
            in_shape = out_shape[:-1] + [self.in_features]
            tensor_offsets_gi = GroupedTensorStorage.make_tensor_offsets(
                split_sizes_gpu, self.in_features
            )
            gi_data = torch.empty(
                total_tokens * self.in_features, dtype=ctx.dtype, device=device
            )
            grouped_grad_input = GroupedTensor(
                shape=(total_tokens, self.in_features),
                dtype=ctx.dtype,
                num_tensors=num_groups,
                data=gi_data,
                first_dims=split_sizes_gpu,
                tensor_offsets=tensor_offsets_gi,
            )
            gemm_a = grouped_w_saved if (
                ctx.with_quantized_compute and self.single_grouped_weight
            ) else ws
            general_grouped_gemm_for_grouped_tensor(
                gemm_a,
                grouped_dy,
                grouped_grad_input,
                layout="NN",
                use_split_accumulator=_2X_ACC_DGRAD,
            )
            grad_input = gi_data.view(in_shape)

        # Wgrad
        if ctx.weight_requires_grad:
            if ctx.with_quantized_compute:
                grouped_x = GroupedTensor(
                    shape=(total_tokens, self.in_features),
                    dtype=ctx.dtype,
                    num_tensors=num_groups,
                    quantizer=ctx.input_quantizer,
                    columnwise_data=x_saved_data,
                    columnwise_scale_inv=x_saved_scale,
                    first_dims=split_sizes_gpu,
                    tensor_offsets=x_tensor_offsets,
                    with_gemm_swizzled_scales=True,
                )
            else:
                grouped_x = GroupedTensor(
                    shape=(total_tokens, self.in_features),
                    dtype=ctx.dtype,
                    num_tensors=num_groups,
                    data=x_saved_data,
                    first_dims=split_sizes_gpu,
                    tensor_offsets=x_tensor_offsets,
                )

            gemm_fn = functools.partial(
                general_grouped_gemm_for_grouped_tensor,
                layout="NT",
                accumulate=accumulate_into_main_grad,
                use_split_accumulator=_2X_ACC_WGRAD,
            )
            if delay_wgrad:
                self.wgrad_store.put(
                    [grouped_x, grouped_dy, wgrad_output], gemm_fn
                )
            else:
                gemm_fn(grouped_x, grouped_dy, wgrad_output)

        if not delay_wgrad and ctx.weight_requires_grad:
            clear_tensor_data(x_saved_data, x_saved_scale)

        # Extract grad_weights from grouped wgrad
        if ctx.weight_requires_grad and self.single_grouped_weight:
            if grouped_wgrad is not None and not delay_wgrad:
                grad_weights = [
                    grouped_wgrad.rowwise_data.view(num_groups, *weight_shape)
                ]
            else:
                grad_weights = [None]

        # Megatron-LM wgrad fusion
        if accumulate_into_main_grad:
            if self.single_grouped_weight:
                grad_weights = [None]
            else:
                grad_weights = [None] * num_groups
            if self.single_grouped_weight:
                if hasattr(weight_param, "grad_added_to_main_grad"):
                    weight_param.grad_added_to_main_grad = True
                    grad_weight = get_dummy_wgrad(
                        list(weight_param.size()),
                        weight_param.dtype,
                        zero=getattr(weight_param, "zero_out_wgrad", False),
                    )
                else:
                    grad_weight = None
                if has_bias:
                    if self.single_grouped_bias:
                        final_bias_grads = torch.stack(grad_biases, dim=0).to(ctx.dtype)
                        grad_params = [grad_weight, final_bias_grads]
                    else:
                        grad_params = grad_biases + [grad_weight]
                else:
                    grad_params = [grad_weight]
                grad_extra = (None, grad_scales) if self._scale_bias else (None,)
                return grad_input, [grad_params], [grad_extra]
            for group_idx in range(num_groups):
                wp = getattr(self, f"weight{group_idx}")
                if hasattr(wp, "grad_added_to_main_grad"):
                    wp.grad_added_to_main_grad = True
                    grad_weights[group_idx] = get_dummy_wgrad(
                        list(wp.size()),
                        wp.dtype,
                        zero=getattr(wp, "zero_out_wgrad", False),
                    )

        if self.single_grouped_weight:
            grad_weight = None
            if ctx.weight_requires_grad:
                if delay_wgrad:
                    grad_weight = None
                else:
                    grad_weight = grad_weights[0] if grad_weights else None
            final_weight_grads = [grad_weight]
        else:
            if delay_wgrad and ctx.weight_requires_grad:
                final_weight_grads = [None] * num_groups
            else:
                final_weight_grads = grad_weights

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

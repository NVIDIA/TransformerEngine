# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GroupedLinear API"""

from dataclasses import dataclass
from typing import Any, Union, Optional, Callable, Tuple, List
from itertools import chain
import os
import warnings
import weakref

import functools
import torch

import transformer_engine_torch as tex

from transformer_engine.common.recipe import Recipe
from transformer_engine.pytorch.tensor.grouped_tensor import (
    GroupedTensor,
    GroupedTensorStorage,
)
from .base import (
    get_dummy_wgrad,
    quantize_weight,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
    _attach_high_precision_init_val,
    _clear_high_precision_init_val,
    _get_high_precision_init_val,
)
from ._common import can_reconstruct_wgrad_input_from_original, WeightGradStore
from . import _split_quantization
from ..quantization import FP8GlobalStateManager, QuantizerRole
from ..utils import (
    divide,
    cast_if_needed,
    clear_tensor_data,
    get_device_compute_capability,
    init_method_constant,
    mark_grouped_tensor,
    requires_grad,
    resolve_grouped_linear_single_param_flags,
    get_nvtx_range_context,
)
from ..distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    is_fp8_activation_recompute_enabled,
    in_fp8_activation_recompute_phase,
)
from ..distributed_weight import (
    is_distributed_weight,
    materialize_weight_for_forward,
    materialize_weight_for_backward,
    finalize_weight_grads,
)
from ..cpp_extensions import (
    general_grouped_gemm,
    general_grouped_gemm_for_grouped_tensor,
)
from ..constants import GemmParallelModes, dist_group_type
from ..jit import no_torch_dynamo
from ..cpu_offload import is_cpu_offload_enabled, mark_not_offload, start_offload
from ..triton.grouped_dbias_dscales import compute_grouped_dbias

from ..tensor import (
    Float8BlockQuantizer,
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
    HybridQuantizer,
    IdentityQuantizer,
    MXFP8Quantizer,
)
from ..quantized_tensor import (
    QuantizedTensorStorage,
    Quantizer,
    prepare_for_saving,
    restore_from_func_ctx,
)
from ...debug.pytorch.debug_quantization import DebugQuantizer
from ...debug.pytorch.debug_state import TEDebugState

__all__ = ["GroupedLinear", "is_module_grouped_tensor_path_supported"]


def is_module_grouped_tensor_path_supported(
    recipe: Optional[Recipe],
    dtype: torch.dtype,
) -> bool:
    """Whether the module grouped-tensor path supports this recipe and dtype.

    The grouped-tensor path dispatches to ``general_grouped_gemm_for_grouped_tensor``
    and does not inspect split values because they may reside in a CUDA tensor.
    Inspecting them on the host would add synchronization and break CUDA Graph safety.

    Supported Compute Capability (CC) and precisions:

    * Hopper (CC 9.0): BF16/FP16, FP8 per-tensor current scaling, and FP8
      block scaling.
    * Blackwell (CC 10.x and 11.0): BF16/FP16, FP8 per-tensor current scaling,
      MXFP8, and NVFP4 with RHT.
    * Custom recipes are unsupported because they may assign different
      quantizers to input, weight, and grad-output roles. This predicate
      currently supports only built-in recipes with known uniform layouts.
    * FP8 delayed scaling is unsupported because the required grouped
      quantization kernels are unavailable.
    * FP8 block scaling is unsupported by this path on Blackwell because it
      does not implement the legacy path's MXFP8-broadcast emulation.
    * Grouped GEMM requires cuBLASLt 13.3+, with 13.4+ required on Hopper,
      13.5+ required for FP8 per-tensor current scaling on Hopper, and 13.6+
      required for FP8 block scaling on Hopper.
    * FP32 is unsupported by the cuBLASLt grouped GEMM.

    Runtime-only restrictions such as debug mode, CPU offloading, calibration,
    output quantization, and backend selection are checked separately by
    ``GroupedLinear``.
    """
    if dtype not in (torch.bfloat16, torch.float16):
        return False

    device_capability = get_device_compute_capability()
    if not (9, 0) <= device_capability <= (11, 0):
        return False
    cublaslt_version = tex.get_cublasLt_version()
    if cublaslt_version < 130300:
        return False
    if device_capability < (10, 0) and cublaslt_version < 130400:
        return False

    if recipe is None:
        return True
    if recipe.custom():
        return False
    if recipe.backward_override is not None:
        return False
    if recipe.float8_current_scaling():
        return device_capability >= (10, 0) or cublaslt_version >= 130500
    if recipe.float8_block_scaling():
        # cuBLASLt 13.6 fixes Hopper grouped GEMM algo selection for block-scaled FP8.
        return device_capability < (10, 0) and cublaslt_version >= 130600
    if recipe.mxfp8():
        return device_capability >= (10, 0)
    if recipe.nvfp4():
        return (
            device_capability >= (10, 0)
            and not recipe.disable_rht
            and not recipe.row_scaled_activation
        )
    return False


@dataclass(slots=True)
class GroupedLinearFwdArgs:
    """Forward configuration and operands."""

    inp: torch.Tensor
    weights: List[Union[torch.Tensor, QuantizedTensorStorage]]
    biases: List[torch.Tensor]

    weight_workspaces: List[Union[torch.Tensor, QuantizedTensorStorage]]
    out: Optional[torch.Tensor]
    dgrad_out: Optional[torch.Tensor]
    skip_fp8_weight_update: Optional[torch.Tensor]
    m_splits_tensor: Optional[torch.Tensor]

    input_requires_grad: bool
    weights_requires_grad: bool

    input_quantizers: List[Quantizer]
    weight_quantizers: List[Quantizer]
    output_quantizers: List[Quantizer]
    grad_input_quantizers: List[Quantizer]
    grad_weight_quantizers: List[Quantizer]
    grad_output_quantizers: List[Quantizer]

    m_splits: Optional[List[int]]
    num_gemms: int

    activation_dtype: torch.dtype
    fp8: bool
    fp8_calibration: bool
    save_original_input: bool
    backward_override: Optional[str]
    fprop_use_split_accumulator: bool
    dgrad_use_split_accumulator: bool
    wgrad_use_split_accumulator: bool
    debug: bool

    is_first_microbatch: Optional[bool]
    cache_weight: bool

    use_grouped_tensor_path: bool
    single_grouped_weight: bool
    single_grouped_bias: bool

    use_bias: bool
    fuse_wgrad_accumulation: bool
    wgrad_store: Optional[Any]
    cpu_offloading: bool
    is_grad_enabled: bool


@dataclass(slots=True)
class GroupedLinearBwdArgs:
    """Backward configuration and saved operands."""

    grad_output: Optional[torch.Tensor] = None
    inputmats: List[Union[torch.Tensor, QuantizedTensorStorage]] = None
    weights_fp8: List[Union[torch.Tensor, QuantizedTensorStorage]] = None
    saved_weights: List[Union[torch.Tensor, QuantizedTensorStorage]] = None
    biases: List[torch.Tensor] = None
    dgrad_out: Optional[torch.Tensor] = None

    input_quantizers: List[Quantizer] = None
    weight_quantizers: List[Quantizer] = None
    grad_input_quantizers: List[Quantizer] = None
    grad_weight_quantizers: List[Quantizer] = None
    grad_output_quantizers: List[Quantizer] = None

    m_splits: Optional[List[int]] = None
    num_gemms: int = 0
    weights_shape_1: int = 0

    use_bias: bool = False
    requires_dgrad: bool = False
    weights_requires_grad: bool = False

    activation_dtype: Optional[torch.dtype] = None
    fp8: bool = False
    backward_override: Optional[str] = None
    dgrad_use_split_accumulator: bool = _2X_ACC_DGRAD
    wgrad_use_split_accumulator: bool = _2X_ACC_WGRAD
    save_original_input: bool = False
    debug: bool = False

    is_first_microbatch: Optional[bool] = None
    fuse_wgrad_accumulation: bool = False
    wgrad_store: Optional[Any] = None
    origin_weight_refs: Optional[Any] = None
    origin_weights_overwrite_main_grad: bool = False
    main_grad_funcs: Optional[Any] = None

    reduce_and_update_bwd_fp8_tensors: bool = False

    cpu_offloading: bool = False

    def setup_saved_tensors(self, ctx: torch.autograd.function.FunctionCtx) -> None:
        """Pull saved tensors from ``ctx`` into the fields backward consumes."""
        saved = restore_from_func_ctx(ctx)
        n = self.num_gemms
        self.inputmats = list(saved[:n])
        self.weights_fp8 = list(saved[n : 2 * n])
        self.saved_weights = list(saved[2 * n : 3 * n])
        self.biases = list(saved[3 * n : 4 * n])


def _grouped_linear_forward_impl(
    args: GroupedLinearFwdArgs,
) -> Tuple[Any, ...]:
    """Compute the legacy forward and collect tensors needed by backward."""
    inp = args.inp
    weights = list(args.weights)
    biases = list(args.biases)
    num_gemms = args.num_gemms
    m_splits = list(args.m_splits)
    input_quantizers = args.input_quantizers
    weight_quantizers = args.weight_quantizers
    output_quantizers = args.output_quantizers
    activation_dtype = args.activation_dtype
    fp8 = args.fp8
    debug = args.debug
    use_bias = args.use_bias
    is_grad_enabled = args.is_grad_enabled
    save_original_input = args.save_original_input
    backward_override = args.backward_override
    cpu_offloading = args.cpu_offloading
    device = inp.device
    weight_requires_grad = args.weights_requires_grad

    is_dist_weight = is_distributed_weight(weights[0])
    if is_dist_weight:
        weights = materialize_weight_for_forward(weights)

    # Configure quantizers
    if input_quantizers[0] is not None:
        for input_quantizer in input_quantizers:
            input_quantizer.set_usage(
                rowwise=True,
                columnwise=(
                    is_grad_enabled
                    and weight_requires_grad
                    and not save_original_input
                    and backward_override is None
                ),
            )
        columnwise_usage = is_grad_enabled and args.input_requires_grad
        if backward_override is not None:
            columnwise_usage = False
        if not columnwise_usage:
            columnwise_usage = (
                is_fp8_activation_recompute_enabled() and not in_fp8_activation_recompute_phase()
            )
        # No need to set the quantizer states if weight is already quantized
        # for debug mode we create quantizer every iteration, thus we need to set the quantizer states
        if weight_quantizers[0] is not None and (
            not isinstance(weights[0], QuantizedTensorStorage) or debug
        ):
            for weight_quantizer in weight_quantizers:
                weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
        elif isinstance(weights[0], QuantizedTensorStorage):
            # If weights are already quantized, no need to set quantizer states
            weight_quantizers = [weight._quantizer for weight in weights]
    if output_quantizers[0] is not None:
        for output_quantizer in output_quantizers:
            output_quantizer.set_usage(rowwise=True, columnwise=False)

    # Initialize input tensors
    in_features = weights[0].size(-1)
    if inp.size(-1) != in_features:
        raise ValueError(
            f"Input tensor (shape={tuple(inp.size())}) is not compatible with "
            f"weight tensor (shape={tuple(weights[0].size())})"
        )

    inp_view = inp.reshape(-1, in_features)
    inputmats, _ = _split_quantization._split_quantize(
        inp_view,
        m_splits,
        input_quantizers,
        activation_dtype,
        with_quantized_output=fp8 or debug,
        disable_bulk_allocation=cpu_offloading,
    )

    if cpu_offloading:
        start_offload(*inputmats)

    # Initialize weights
    weights_fp8: list
    new_workspaces = [None] * num_gemms
    if fp8 or debug:
        weights_fp8 = []
        update_ws = args.is_first_microbatch is None or args.is_first_microbatch
        for i in range(num_gemms):
            weight_fp8, new_workspaces[i] = quantize_weight(
                tensor=weights[i],
                quantizer=weight_quantizers[i],
                workspace=args.weight_workspaces[i] if args.weight_workspaces else None,
                update_workspace=update_ws,
                skip_update_flag=args.skip_fp8_weight_update,
                workspace_dtype=activation_dtype,
                cache=args.cache_weight and not is_dist_weight,
            )
            weights_fp8.append(weight_fp8)
    else:
        weights_fp8 = [cast_if_needed(weight, activation_dtype) for weight in weights]

    # Initialize biases
    bias_dtype = activation_dtype
    if fp8 and activation_dtype == torch.float32:
        bias_dtype = torch.bfloat16  # FP8 GEMM only supports BF16/FP16 bias
    biases = [cast_if_needed(bias, bias_dtype) for bias in biases] if use_bias else biases
    # Initialize output tensor
    out = _GroupedLinear._validate_or_alloc_output(
        args.out,
        sum(m_splits),
        weights_fp8[0].size(0),
        activation_dtype,
        device,
    )

    # Perform GEMM
    general_grouped_gemm(
        weights_fp8,
        inputmats,
        [out],
        output_quantizers,
        activation_dtype,
        single_output=True,
        m_splits=m_splits,
        bias=biases,
        use_bias=use_bias,
        use_split_accumulator=args.fprop_use_split_accumulator,
    )

    if args.fp8_calibration:
        for i in range(num_gemms):
            input_quantizers[i].calibrate(inputmats[i])
            weight_quantizers[i].calibrate(weights[i])

    if cpu_offloading:
        mark_not_offload(*weights_fp8, *weights)

    tensors_to_save = None
    if is_grad_enabled:
        # TODO: update after #1638 is merged. # pylint: disable=fixme
        if weight_requires_grad:
            if save_original_input:
                inputmats = [None] * num_gemms
                inputmats[0] = inp
            else:
                for inputmat in inputmats:
                    if isinstance(inputmat, QuantizedTensorStorage):
                        if backward_override is not None:
                            # In dequantized mode we should dequantize directly from
                            # fprop quantized layouts without retargeting usage.
                            inputmat.update_usage(rowwise_usage=True, columnwise_usage=False)
                        else:
                            inputmat.update_usage(rowwise_usage=False, columnwise_usage=True)
        else:
            inputmats = [None] * num_gemms

        # Original weights are only needed by high_precision dgrad. The weakrefs
        # used for fused wgrad accumulation serve a different purpose: restoring
        # Python parameter attributes without keeping the parameter alive here.
        saved_weights = (
            weights
            if backward_override == "high_precision" and args.input_requires_grad
            else [None] * num_gemms
        )
        if is_dist_weight:
            # GTP: gathered workspace is transient (re-gathered in backward), don't save it.
            weights_fp8 = [None] * num_gemms
            saved_weights = args.weights
        tensors_to_save = (*inputmats, *weights_fp8, *saved_weights, *biases)

    return out.view(-1, *inp.shape[1:-1], out.shape[-1]), new_workspaces, tensors_to_save


def _grouped_linear_setup_ctx(
    bwd_args: GroupedLinearBwdArgs,
    fwd_args: GroupedLinearFwdArgs,
) -> None:
    """Populate backward arguments from the forward configuration."""
    num_gemms = fwd_args.num_gemms
    weights = fwd_args.weights
    weight_quantizers = fwd_args.weight_quantizers
    if isinstance(weights[0], QuantizedTensorStorage) and not fwd_args.debug:
        weight_quantizers = [weight._quantizer for weight in weights]

    bwd_args.input_quantizers = fwd_args.input_quantizers
    bwd_args.weight_quantizers = weight_quantizers
    bwd_args.grad_input_quantizers = fwd_args.grad_input_quantizers
    bwd_args.grad_weight_quantizers = fwd_args.grad_weight_quantizers
    bwd_args.grad_output_quantizers = fwd_args.grad_output_quantizers

    bwd_args.m_splits = fwd_args.m_splits
    bwd_args.num_gemms = num_gemms
    bwd_args.weights_shape_1 = weights[0].shape[1]

    bwd_args.use_bias = fwd_args.use_bias
    bwd_args.requires_dgrad = fwd_args.input_requires_grad
    bwd_args.weights_requires_grad = fwd_args.weights_requires_grad

    bwd_args.activation_dtype = fwd_args.activation_dtype
    bwd_args.fp8 = fwd_args.fp8
    bwd_args.backward_override = fwd_args.backward_override
    bwd_args.dgrad_use_split_accumulator = fwd_args.dgrad_use_split_accumulator
    bwd_args.wgrad_use_split_accumulator = fwd_args.wgrad_use_split_accumulator
    bwd_args.save_original_input = fwd_args.save_original_input
    bwd_args.debug = fwd_args.debug

    bwd_args.is_first_microbatch = fwd_args.is_first_microbatch
    bwd_args.fuse_wgrad_accumulation = fwd_args.fuse_wgrad_accumulation
    bwd_args.wgrad_store = fwd_args.wgrad_store
    bwd_args.cpu_offloading = fwd_args.cpu_offloading
    bwd_args.dgrad_out = fwd_args.dgrad_out

    if fwd_args.fuse_wgrad_accumulation and fwd_args.weights_requires_grad:
        # Keep weakrefs to weights to preserve attributes like main_grad
        # when we need to modify the weight python objects
        bwd_args.origin_weight_refs = [weakref.ref(w) for w in weights]
        bwd_args.origin_weights_overwrite_main_grad = getattr(
            weights[0], "overwrite_main_grad", False
        )
        # MCore FSDP creates main_grad lazily before backward
        if hasattr(weights[0], "__fsdp_param__"):
            bwd_args.main_grad_funcs = [weights[i].get_main_grad for i in range(num_gemms)]
        elif is_distributed_weight(weights[0]):
            bwd_args.main_grad_funcs = [weights[i].grad_buffer for i in range(num_gemms)]
        else:
            bwd_args.main_grad_funcs = [lambda j=i: weights[j].main_grad for i in range(num_gemms)]

    if fwd_args.backward_override is not None:
        bwd_args.fp8 = False
        bwd_args.debug = False
        bwd_args.grad_input_quantizers = [None] * num_gemms
        bwd_args.grad_weight_quantizers = [None] * num_gemms
        bwd_args.grad_output_quantizers = [None] * num_gemms


def _finish_wgrad(weight, main_grad, wgrad, fuse_wgrad_accumulation):
    """Return the gradient expected by autograd or MCore's main-grad hooks."""
    if not fuse_wgrad_accumulation:
        return wgrad
    if not hasattr(weight, "grad_added_to_main_grad"):
        return None
    weight.grad_added_to_main_grad = True
    return get_dummy_wgrad(
        list(main_grad.shape), weight.dtype, zero=getattr(weight, "zero_out_wgrad", False)
    )


def _grouped_linear_backward_impl(
    args: GroupedLinearBwdArgs,
) -> Tuple[Optional[torch.Tensor], List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
    """Backward implementation for the grouped linear layer.

    Caller must have populated ``args.grad_output`` and run
    ``args.setup_saved_tensors(ctx)`` before invocation. Returns
    ``(dgrad, wgrad_list, grad_biases)``.
    """
    grad_output = args.grad_output
    num_gemms = args.num_gemms
    m_splits = list(args.m_splits)
    inputmats = list(args.inputmats)
    weights = list(args.weights_fp8)
    saved_weights = list(args.saved_weights)
    biases = list(args.biases)
    device = grad_output.device
    in_features = args.weights_shape_1
    dgrad = None

    # Restore from weakrefs to get original weight python objects
    # (preserves attributes like main_grad, grad_added_to_main_grad, etc.)
    # Only needed when fuse_wgrad_accumulation is enabled.
    origin_weights = [None] * num_gemms
    main_grads = [None] * num_gemms
    is_dist_weight = is_distributed_weight(saved_weights[0])
    if is_dist_weight:
        origin_weights = saved_weights
        if args.fuse_wgrad_accumulation and args.weights_requires_grad:
            main_grads = [main_grad_func() for main_grad_func in args.main_grad_funcs]
    elif args.fuse_wgrad_accumulation and args.weights_requires_grad:
        origin_weight_refs = args.origin_weight_refs
        args.origin_weight_refs = None
        origin_weights = [ref() if ref is not None else None for ref in origin_weight_refs]
        assert all(
            w is not None for w in origin_weights
        ), "weight was removed while fuse_wgrad_accumulation=True"
        main_grads = [main_grad_func() for main_grad_func in args.main_grad_funcs]
        for origin_weight, main_grad in zip(origin_weights, main_grads):
            if main_grad is not None:
                origin_weight.main_grad = main_grad

    # Preprocess grad output
    grad_output_view = grad_output.contiguous().view(-1, grad_output.shape[-1])
    out_features = grad_output_view.shape[-1]
    grad_output_reference = args.grad_output_quantizers[0]
    if args.fp8 and isinstance(grad_output_reference, HybridQuantizer):
        # Usage is a runtime decision, not part of generation validation.
        # Apply it uniformly so dispatch can read the first parent without
        # rescanning every expert.
        for grad_output_quantizer in args.grad_output_quantizers:
            grad_output_quantizer.set_usage(
                rowwise=args.requires_dgrad,
                columnwise=args.weights_requires_grad,
            )
    grad_output, grad_biases = _split_quantization._split_quantize(
        grad_output_view,
        m_splits,
        args.grad_output_quantizers,
        args.activation_dtype,
        with_quantized_output=args.fp8 or args.debug,
        compute_dbias=(args.fp8 or args.debug) and (args.use_bias or args.debug),
        disable_bulk_allocation=args.cpu_offloading,
    )
    if grad_biases is None:
        grad_biases = [None] * num_gemms

    if is_dist_weight:
        accumulate_wgrad_into_param_main_grad = False
    elif args.is_first_microbatch is not None:
        accumulate_wgrad_into_param_main_grad = (
            args.fuse_wgrad_accumulation and not args.is_first_microbatch
        )
    else:
        accumulate_wgrad_into_param_main_grad = args.fuse_wgrad_accumulation

    if is_dist_weight:
        weights = materialize_weight_for_backward(origin_weights)

    if args.requires_dgrad:
        dgrad = _GroupedLinear._validate_or_alloc_output(
            args.dgrad_out,
            sum(m_splits),
            in_features,
            args.activation_dtype,
            device,
        )
        weights_for_dgrad = weights
        if args.backward_override == "dequantized":
            weights_for_dgrad = [
                _GroupedLinear._maybe_dequantize(weight, args.activation_dtype)
                for weight in weights
            ]
        elif args.backward_override == "high_precision":
            weights_for_dgrad = [
                _GroupedLinear._maybe_dequantize(weight, args.activation_dtype)
                for weight in saved_weights
            ]
        elif is_dist_weight and args.fp8:
            weights_for_dgrad = []
            for idx, weight in enumerate(weights):
                if not isinstance(weight, QuantizedTensorStorage):
                    quantizer = args.weight_quantizers[idx]
                    quantizer.set_usage(rowwise=True, columnwise=True)
                    weight = quantizer(weight)
                weights_for_dgrad.append(weight)
        # Make sure weights are available in column-wise format
        # for dgrad computation.
        for weight in weights_for_dgrad:
            if isinstance(weight, QuantizedTensorStorage):
                weight.update_usage(columnwise_usage=True)
        general_grouped_gemm(
            weights_for_dgrad,
            grad_output,
            [dgrad],
            args.grad_input_quantizers,
            args.activation_dtype,
            single_output=True,
            layout="NN",
            m_splits=m_splits,
            grad=True,
            use_split_accumulator=args.dgrad_use_split_accumulator,
        )

    if args.weights_requires_grad:
        if (
            is_dist_weight
            and args.wgrad_store is not None
            and args.wgrad_store.delay_wgrad_compute()
        ):
            raise RuntimeError(
                "distributed-weight GroupedLinear requires delay_wgrad_compute=False."
            )
        if args.fuse_wgrad_accumulation:
            wgrad_list = main_grads
        else:
            wgrad_packed = torch.empty(
                num_gemms,
                out_features,
                in_features,
                dtype=args.activation_dtype,
                device=device,
            )
            wgrad_list = [wgrad_packed[i] for i in range(num_gemms)]
            if is_dist_weight:
                # Gathered weights are no longer needed after dgrad GEMM.
                del weights

        if args.save_original_input:
            inp = inputmats[0]
            inp_view = inp.reshape(-1, in_features)
            if args.input_quantizers[0] is not None:
                for input_quantizer in args.input_quantizers:
                    if isinstance(
                        input_quantizer,
                        (Float8Quantizer, Float8CurrentScalingQuantizer),
                    ):
                        input_quantizer.set_usage(rowwise=True, columnwise=True)
                    else:
                        input_quantizer.set_usage(rowwise=False, columnwise=True)
            inputmats, _ = _split_quantization._split_quantize(
                inp_view,
                m_splits,
                with_quantized_output=args.fp8 or args.debug,
                quantizers=args.input_quantizers,
                activation_dtype=args.activation_dtype,
                disable_bulk_allocation=args.cpu_offloading,
            )
        elif args.backward_override == "dequantized":
            inputmats = [
                _GroupedLinear._maybe_dequantize(inputmat, args.activation_dtype)
                for inputmat in inputmats
            ]
        grouped_gemm_wgrad = functools.partial(
            general_grouped_gemm,
            quantization_params=args.grad_weight_quantizers,
            out_dtype=args.activation_dtype,
            layout="NT",
            grad=True,
            m_splits=m_splits,
            use_bias=args.use_bias if grad_biases[0] is None else None,
            bias=biases,
            use_split_accumulator=args.wgrad_use_split_accumulator,
            accumulate=(
                accumulate_wgrad_into_param_main_grad
                if not is_dist_weight and not args.origin_weights_overwrite_main_grad
                else False
            ),
        )
        # WGRAD
        if args.wgrad_store is not None and args.wgrad_store.delay_wgrad_compute():
            args.wgrad_store.put([inputmats, grad_output, wgrad_list], grouped_gemm_wgrad)
        else:
            _, grad_biases_, _ = grouped_gemm_wgrad(inputmats, grad_output, wgrad_list)

            for i in range(num_gemms):
                if grad_biases[i] is None:
                    grad_biases[i] = grad_biases_[i]
            del grad_biases_

            clear_tensor_data(*inputmats)

        if is_dist_weight:
            wgrad_list = finalize_weight_grads(origin_weights, wgrad_list)
        else:
            wgrad_list = [
                _finish_wgrad(weight, main_grad, wgrad, args.fuse_wgrad_accumulation)
                for weight, main_grad, wgrad in zip(origin_weights, main_grads, wgrad_list)
            ]
    else:
        wgrad_list = [None] * num_gemms

    if not args.use_bias or (
        args.wgrad_store is not None and args.wgrad_store.delay_wgrad_compute() and not args.fp8
    ):
        grad_biases = [None] * num_gemms

    dgrad_out = None
    if args.requires_dgrad:
        # Input shape rederived from grad_output (out.shape == (*inp.shape[:-1], out_features)).
        dgrad_out = dgrad.view(*args.grad_output.shape[:-1], in_features)
    return (dgrad_out, wgrad_list, list(grad_biases))


@dataclass(slots=True)
class GroupedLinearFusedBwdArgs:
    """Backward configuration and saved grouped operands."""

    grad_output: Optional[torch.Tensor] = None
    inputmat: Any = None
    weights_fp8: List[Union[torch.Tensor, QuantizedTensorStorage]] = None
    m_splits_tensor: Optional[torch.Tensor] = None
    base_split_offsets: Optional[torch.Tensor] = None
    input_tensor_offsets: Optional[torch.Tensor] = None
    output_tensor_offsets: Optional[torch.Tensor] = None
    dgrad_out: Optional[torch.Tensor] = None
    input_quantizers: List[Quantizer] = None
    weight_quantizers: List[Quantizer] = None
    grad_output_quantizers: List[Quantizer] = None
    is_dist_weight: bool = False
    num_gemms: int = 0
    in_features: int = 0
    out_features: int = 0
    activation_dtype: Optional[torch.dtype] = None
    fp8: bool = False
    use_bias: bool = False
    requires_dgrad: bool = False
    weights_requires_grad: bool = False
    save_original_input: bool = False
    single_grouped_weight: bool = False
    single_grouped_bias: bool = False
    is_first_microbatch: Optional[bool] = None
    dgrad_use_split_accumulator: bool = _2X_ACC_DGRAD
    wgrad_use_split_accumulator: bool = _2X_ACC_WGRAD
    fuse_wgrad_accumulation: bool = False
    origin_weight_refs: Optional[Any] = None
    origin_weights_overwrite_main_grad: bool = False
    main_grad_funcs: Optional[Any] = None
    wgrad_store: Optional[WeightGradStore] = None
    reduce_and_update_bwd_fp8_tensors: bool = False

    def setup_saved_tensors(self, ctx: torch.autograd.function.FunctionCtx) -> None:
        """Restore the grouped operands and split metadata."""
        saved = restore_from_func_ctx(ctx)
        n = 1 if self.single_grouped_weight else self.num_gemms
        self.inputmat = saved[0]
        self.weights_fp8 = list(saved[1 : 1 + n])
        (
            self.m_splits_tensor,
            self.base_split_offsets,
            self.input_tensor_offsets,
            self.output_tensor_offsets,
        ) = saved[1 + n :]


def _grouped_linear_fused_forward(args: GroupedLinearFwdArgs) -> Tuple[Any, ...]:
    """Compute the grouped forward and collect tensors needed by backward."""
    inp = args.inp
    use_bias = args.use_bias
    is_first_microbatch = args.is_first_microbatch
    fp8 = args.fp8
    input_quantizers = args.input_quantizers
    weight_quantizers = args.weight_quantizers
    activation_dtype = args.activation_dtype
    is_grad_enabled = args.is_grad_enabled
    weight_workspaces = args.weight_workspaces
    cache_weight = args.cache_weight
    skip_fp8_weight_update = args.skip_fp8_weight_update
    save_original_input = args.save_original_input
    single_grouped_weight = args.single_grouped_weight
    single_grouped_bias = args.single_grouped_bias
    weights = args.weights
    is_dist_weight = is_distributed_weight(weights[0])
    if is_dist_weight:
        weights = materialize_weight_for_forward(weights)
    biases = args.biases
    out = args.out
    m_splits = args.m_splits_tensor
    num_gemms = len(m_splits)
    device = inp.device
    in_features = weights[0].size(-1)
    out_features = weights[0].size(-2)
    weight_requires_grad = args.weights_requires_grad
    save_original_input = save_original_input and weight_requires_grad

    split_sizes, (
        base_split_offsets,
        input_tensor_offsets,
        output_tensor_offsets,
    ) = tex.splits_to_offsets_multi(
        m_splits,
        device,
        strides=[1, in_features, out_features],
        include_leading_zero=[True, True, True],
        dtypes=[torch.int64, torch.int64, torch.int64],
        bulk_allocate=True,
    )

    inp_view = inp.reshape(-1, in_features)
    x = cast_if_needed(inp_view, activation_dtype)
    if fp8:
        input_quantizer = input_quantizers[0]
        input_quantizer.set_usage(
            rowwise=True,
            columnwise=(is_grad_enabled and weight_requires_grad and not save_original_input),
        )
        input_quantizer.optimize_for_gemm = True
        grouped_x = tex.group_quantize(
            x,
            input_quantizer,
            num_gemms,
            split_sizes,
            tensor_offsets=input_tensor_offsets,
        )
    else:
        grouped_x = _GroupedLinear._make_grouped_tensor(
            x,
            num_gemms=num_gemms,
            split_sizes=split_sizes,
            tensor_offsets=input_tensor_offsets,
            last_dim=in_features,
            dtype=activation_dtype,
        )

    columnwise_usage = is_grad_enabled and args.input_requires_grad
    weights_for_gemm, new_workspaces = _GroupedLinear._prepare_weights_for_grouped_tensor_gemm(
        weights,
        weight_quantizers,
        weight_workspaces,
        num_gemms=num_gemms,
        single_grouped_weight=single_grouped_weight,
        with_quantized_compute=fp8,
        columnwise_usage=columnwise_usage,
        activation_dtype=activation_dtype,
        is_first_microbatch=is_first_microbatch,
        skip_fp8_weight_update=skip_fp8_weight_update,
        cache_weight=cache_weight and not is_dist_weight,
    )

    out = _GroupedLinear._validate_or_alloc_output(
        out,
        x.size(0),
        out_features,
        activation_dtype,
        device,
    )
    grouped_out = _GroupedLinear._make_grouped_tensor(
        out,
        num_gemms=num_gemms,
        split_sizes=split_sizes,
        tensor_offsets=output_tensor_offsets,
        last_dim=out_features,
        dtype=activation_dtype,
    )

    grouped_bias = None
    if use_bias:
        grouped_bias = _GroupedLinear._prepare_bias_for_grouped_tensor_gemm(
            biases,
            single_grouped_bias=single_grouped_bias,
            num_gemms=num_gemms,
            out_features=out_features,
            dtype=activation_dtype,
        )

    general_grouped_gemm_for_grouped_tensor(
        weights_for_gemm,
        grouped_x,
        grouped_out,
        layout="TN",
        bias=grouped_bias,
        use_split_accumulator=args.fprop_use_split_accumulator,
    )

    tensors_to_save = None
    if is_grad_enabled:
        input_to_save = grouped_x
        if weight_requires_grad:
            if save_original_input:
                # Save the high-precision input and reconstruct the grouped columnwise
                # operand in backward instead of retaining a second quantized copy.
                input_to_save = inp
            elif fp8 and grouped_x.columnwise_data is not None:
                # Wgrad only consumes the columnwise representation.
                grouped_x.rowwise_data = None
                grouped_x.scale_inv = None
        else:
            input_to_save = None

        weights_to_save = [weights_for_gemm] if single_grouped_weight else weights_for_gemm
        if not args.input_requires_grad:
            weights_to_save = [None] * len(weights_to_save)

        if is_dist_weight:
            weights_to_save = list(args.weights)

        # Megatron-LM paged stashing uses this marker to identify the dynamic activation
        # buffers among the tensors saved by the GroupedLinear autograd function. The
        # operation-fuser grouped MLP applies the same marker to its saved activations.
        mark_grouped_tensor(input_to_save)
        tensors_to_save = (
            input_to_save,
            *weights_to_save,
            split_sizes,
            base_split_offsets,
            input_tensor_offsets,
            output_tensor_offsets,
        )
    return out.view(-1, *inp.shape[1:-1], out.shape[-1]), new_workspaces, tensors_to_save


def _grouped_linear_fused_setup(
    bwd_args: GroupedLinearFusedBwdArgs,
    fwd_args: GroupedLinearFwdArgs,
) -> None:
    """Populate grouped backward arguments from the forward configuration."""
    bwd_args.input_quantizers = fwd_args.input_quantizers
    bwd_args.weight_quantizers = fwd_args.weight_quantizers
    bwd_args.is_dist_weight = is_distributed_weight(fwd_args.weights[0])
    bwd_args.grad_output_quantizers = fwd_args.grad_output_quantizers
    bwd_args.m_splits_tensor = fwd_args.m_splits_tensor
    bwd_args.num_gemms = fwd_args.num_gemms
    bwd_args.activation_dtype = fwd_args.activation_dtype
    bwd_args.fp8 = fwd_args.fp8
    bwd_args.use_bias = fwd_args.use_bias
    bwd_args.single_grouped_weight = fwd_args.single_grouped_weight
    bwd_args.single_grouped_bias = fwd_args.single_grouped_bias
    bwd_args.is_first_microbatch = fwd_args.is_first_microbatch
    bwd_args.dgrad_use_split_accumulator = fwd_args.dgrad_use_split_accumulator
    bwd_args.wgrad_use_split_accumulator = fwd_args.wgrad_use_split_accumulator
    bwd_args.fuse_wgrad_accumulation = fwd_args.fuse_wgrad_accumulation
    bwd_args.wgrad_store = fwd_args.wgrad_store
    bwd_args.dgrad_out = fwd_args.dgrad_out
    bwd_args.in_features = fwd_args.weights[0].shape[-1]
    bwd_args.out_features = fwd_args.weights[0].shape[-2]
    bwd_args.requires_dgrad = fwd_args.input_requires_grad
    bwd_args.weights_requires_grad = fwd_args.weights_requires_grad
    bwd_args.save_original_input = fwd_args.save_original_input and fwd_args.weights_requires_grad
    if fwd_args.fuse_wgrad_accumulation and fwd_args.weights_requires_grad:
        weights = fwd_args.weights
        bwd_args.origin_weight_refs = [weakref.ref(w) for w in weights]
        bwd_args.origin_weights_overwrite_main_grad = getattr(
            weights[0], "overwrite_main_grad", False
        )
        if hasattr(weights[0], "__fsdp_param__"):
            bwd_args.main_grad_funcs = [weight.get_main_grad for weight in weights]
        elif bwd_args.is_dist_weight:
            bwd_args.main_grad_funcs = [weight.grad_buffer for weight in weights]
        else:
            bwd_args.main_grad_funcs = [
                lambda j=i: weights[j].main_grad for i in range(len(weights))
            ]


def _grouped_linear_fused_backward(
    args: GroupedLinearFusedBwdArgs,
) -> Tuple[Optional[torch.Tensor], List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
    """Compute gradients with the grouped-tensor kernels."""
    grad_output = args.grad_output
    N = args.num_gemms
    saved_input = args.inputmat
    weight_tensors = args.weights_fp8
    weights_for_gemm = weight_tensors[0] if args.single_grouped_weight else weight_tensors
    split_sizes = args.m_splits_tensor
    base_split_offsets = args.base_split_offsets
    input_tensor_offsets = args.input_tensor_offsets
    output_tensor_offsets = args.output_tensor_offsets
    if args.save_original_input:
        x = cast_if_needed(
            saved_input.reshape(-1, args.in_features),
            args.activation_dtype,
        )
        if args.fp8:
            input_quantizer = args.input_quantizers[0]
            input_quantizer.set_usage(rowwise=False, columnwise=True)
            input_quantizer.optimize_for_gemm = True
            grouped_x = tex.group_quantize(x, input_quantizer, N, split_sizes)
        else:
            grouped_x = _GroupedLinear._make_grouped_tensor(
                x,
                num_gemms=N,
                split_sizes=split_sizes,
                tensor_offsets=input_tensor_offsets,
                last_dim=args.in_features,
                dtype=args.activation_dtype,
            )
    else:
        grouped_x = saved_input

    num_weight_args = 1 if args.single_grouped_weight else N
    origin_weights = [None] * num_weight_args
    main_grads = [None] * num_weight_args
    is_dist_weight = args.is_dist_weight
    if is_dist_weight:
        origin_weights = list(weight_tensors)
        if args.fuse_wgrad_accumulation and args.weights_requires_grad:
            main_grads = [main_grad_func() for main_grad_func in args.main_grad_funcs]
        weight_tensors = materialize_weight_for_backward(origin_weights)
        weights_for_gemm = None
        if args.requires_dgrad:
            weights_for_gemm, _ = _GroupedLinear._prepare_weights_for_grouped_tensor_gemm(
                weight_tensors,
                args.weight_quantizers,
                None,
                num_gemms=N,
                single_grouped_weight=args.single_grouped_weight,
                with_quantized_compute=args.fp8,
                columnwise_usage=True,
                activation_dtype=args.activation_dtype,
                is_first_microbatch=None,
                skip_fp8_weight_update=None,
                cache_weight=False,
            )
    elif args.fuse_wgrad_accumulation and args.weights_requires_grad:
        origin_weight_refs = args.origin_weight_refs
        args.origin_weight_refs = None
        origin_weights = [ref() if ref is not None else None for ref in origin_weight_refs]
        assert all(
            w is not None for w in origin_weights
        ), "weight was removed while fuse_wgrad_accumulation=True"
        main_grads = [main_grad_func() for main_grad_func in args.main_grad_funcs]
        for origin_weight, main_grad in zip(origin_weights, main_grads):
            if main_grad is not None:
                origin_weight.main_grad = main_grad

    grad_output_view = grad_output.contiguous().view(-1, grad_output.shape[-1])
    dy_2d = cast_if_needed(grad_output_view, args.activation_dtype)
    dbias_packed = None
    if args.fp8:
        grad_output_quantizer = args.grad_output_quantizers[0]
        grad_output_quantizer.set_usage(
            rowwise=args.requires_dgrad,
            columnwise=args.weights_requires_grad,
        )
        grad_output_quantizer.optimize_for_gemm = True
        # The grouped FP8 block-scaling bgrad kernel computes dbias in the rowwise
        # pass, so the fusion needs rowwise output (i.e. dgrad required).
        fuse_bgrad = isinstance(grad_output_quantizer, MXFP8Quantizer) or (
            isinstance(grad_output_quantizer, Float8BlockQuantizer) and args.requires_dgrad
        )
        if args.use_bias and fuse_bgrad:
            grouped_dy, dbias_packed = tex.bgrad_group_quantize(
                dy_2d,
                grad_output_quantizer,
                N,
                split_sizes,
                tensor_offsets=output_tensor_offsets,
            )
        else:
            grouped_dy = tex.group_quantize(
                dy_2d,
                grad_output_quantizer,
                N,
                split_sizes,
                tensor_offsets=output_tensor_offsets,
            )
    else:
        grouped_dy = _GroupedLinear._make_grouped_tensor(
            dy_2d,
            num_gemms=N,
            split_sizes=split_sizes,
            tensor_offsets=output_tensor_offsets,
            last_dim=args.out_features,
            dtype=args.activation_dtype,
        )

    if args.use_bias:
        if dbias_packed is None:
            dbias_packed = compute_grouped_dbias(dy_2d, base_split_offsets, N)
        if args.single_grouped_bias:
            grad_bias_args = [dbias_packed.to(dtype=args.activation_dtype)]
        else:
            grad_bias_args = [dbias_packed[i].to(dtype=args.activation_dtype) for i in range(N)]
    else:
        num_bias_args = 1 if args.single_grouped_bias else N
        grad_bias_args = [None] * num_bias_args

    dgrad = None
    if args.requires_dgrad:
        for weight in weight_tensors:
            if isinstance(weight, QuantizedTensorStorage):
                weight.update_usage(columnwise_usage=True)
        dgrad = _GroupedLinear._validate_or_alloc_output(
            args.dgrad_out,
            dy_2d.size(0),
            args.in_features,
            args.activation_dtype,
            grad_output.device,
        )
        grouped_dgrad = _GroupedLinear._make_grouped_tensor(
            dgrad,
            num_gemms=N,
            split_sizes=split_sizes,
            tensor_offsets=input_tensor_offsets,
            last_dim=args.in_features,
            dtype=args.activation_dtype,
        )
        general_grouped_gemm_for_grouped_tensor(
            weights_for_gemm,
            grouped_dy,
            grouped_dgrad,
            layout="NN",
            use_split_accumulator=args.dgrad_use_split_accumulator,
        )

    if is_dist_weight:
        accumulate_wgrad_into_param_main_grad = False
    elif args.is_first_microbatch is not None:
        accumulate_wgrad_into_param_main_grad = (
            args.fuse_wgrad_accumulation and not args.is_first_microbatch
        )
    else:
        accumulate_wgrad_into_param_main_grad = args.fuse_wgrad_accumulation

    if args.weights_requires_grad:
        if (
            is_dist_weight
            and args.wgrad_store is not None
            and args.wgrad_store.delay_wgrad_compute()
        ):
            raise RuntimeError(
                "distributed-weight GroupedLinear requires delay_wgrad_compute=False."
            )
        if args.fuse_wgrad_accumulation:
            if args.single_grouped_weight:
                main_grad = main_grads[0]
                grouped_wgrad = GroupedTensor.make_grouped_tensor_from_rowwise_data(
                    num_tensors=N,
                    tensor_shape=(args.out_features, args.in_features),
                    rowwise_data=main_grad.view(-1),
                    dtype=main_grad.dtype,
                )
                wgrad_output = grouped_wgrad
                wgrad_list = [main_grad]
            else:
                wgrad_output = main_grads
                wgrad_list = main_grads
        else:
            if args.single_grouped_weight:
                grouped_wgrad = GroupedTensor.make_grouped_tensor_with_shapes(
                    num_tensors=N,
                    shapes=[(args.out_features, args.in_features)] * N,
                    quantizer=None,
                    device=grad_output.device,
                    dtype=args.activation_dtype,
                )
                wgrad_output = grouped_wgrad
                wgrad_list = [
                    grouped_wgrad.rowwise_data.view(N, args.out_features, args.in_features)
                ]
            else:
                wgrad_packed = torch.empty(
                    N,
                    args.out_features,
                    args.in_features,
                    dtype=args.activation_dtype,
                    device=grad_output.device,
                )
                wgrad_output = [wgrad_packed[i] for i in range(N)]
                wgrad_list = wgrad_output

        accumulate = (
            accumulate_wgrad_into_param_main_grad
            if not getattr(args, "origin_weights_overwrite_main_grad", False)
            else False
        )

        def grouped_gemm_wgrad(inputmats, grad_output_mats, grad_weights):
            general_grouped_gemm_for_grouped_tensor(
                inputmats,
                grad_output_mats,
                grad_weights,
                layout="NT",
                use_split_accumulator=args.wgrad_use_split_accumulator,
                accumulate=accumulate,
            )
            return None, [None] * N, None

        if args.wgrad_store is not None and args.wgrad_store.delay_wgrad_compute():
            args.wgrad_store.put([grouped_x, grouped_dy, wgrad_output], grouped_gemm_wgrad)
        else:
            grouped_gemm_wgrad(grouped_x, grouped_dy, wgrad_output)

        if is_dist_weight:
            wgrad_list = finalize_weight_grads(origin_weights, wgrad_list)
        else:
            wgrad_list = [
                _finish_wgrad(weight, main_grad, wgrad, args.fuse_wgrad_accumulation)
                for weight, main_grad, wgrad in zip(origin_weights, main_grads, wgrad_list)
            ]
    else:
        wgrad_list = [None] * num_weight_args

    if dgrad is not None:
        dgrad = dgrad.view(*grad_output.shape[:-1], args.in_features)
    return dgrad, wgrad_list, grad_bias_args


@no_torch_dynamo()
def _grouped_linear_eager(
    inp: torch.Tensor,
    m_splits: torch.Tensor,
    fwd_args: GroupedLinearFwdArgs,
    weights_and_biases: Tuple[torch.Tensor, ...],
    is_grad_enabled: bool,
) -> Tuple[torch.Tensor, list]:
    """Run ``_GroupedLinear`` eagerly, bypassing Dynamo."""
    if fwd_args.m_splits is None and not fwd_args.use_grouped_tensor_path:
        fwd_args.m_splits = tuple(m_splits.tolist())
    if is_grad_enabled:
        return _GroupedLinear.apply(inp, fwd_args, *weights_and_biases)
    return _GroupedLinear.forward(None, inp, fwd_args, *weights_and_biases)


class _GroupedLinear(torch.autograd.Function):
    """GroupedLinear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def _maybe_dequantize(
        tensor: Union[torch.Tensor, QuantizedTensorStorage],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize quantized tensors or cast regular tensors to ``dtype``."""
        if isinstance(tensor, QuantizedTensorStorage):
            return tensor.dequantize(dtype=dtype)
        return cast_if_needed(tensor, dtype)

    @staticmethod
    def _make_grouped_tensor(
        data: torch.Tensor,
        *,
        num_gemms: int,
        split_sizes: torch.Tensor,
        tensor_offsets: torch.Tensor,
        last_dim: int,
        dtype: torch.dtype,
    ) -> GroupedTensorStorage:
        """Wrap a packed 2D buffer as a varying-first-dimension GroupedTensorStorage."""
        return GroupedTensorStorage(
            shape=(data.size(0), last_dim),
            dtype=dtype,
            num_tensors=num_gemms,
            quantizer=None,
            data=data.reshape(-1),
            first_dims=split_sizes,
            tensor_offsets=tensor_offsets,
        )

    @staticmethod
    def _make_grouped_bias(
        biases: Tuple[torch.Tensor, ...],
        *,
        num_gemms: int,
        out_features: int,
        dtype: torch.dtype,
    ) -> GroupedTensorStorage:
        """Pack per-GEMM biases into the grouped GEMM bias format."""
        bias_data = torch.stack(
            [_GroupedLinear._maybe_dequantize(bias, dtype) for bias in biases],
            dim=0,
        ).contiguous()
        return GroupedTensorStorage(
            shape=(num_gemms, out_features),
            dtype=dtype,
            num_tensors=num_gemms,
            shapes=[(1, out_features)] * num_gemms,
            quantizer=None,
            data=bias_data.reshape(-1),
        )

    @staticmethod
    def _prepare_weights_for_grouped_tensor_gemm(
        weights: Tuple[torch.Tensor, ...],
        weight_quantizers: List[Optional[Quantizer]],
        weight_workspaces: List[Optional[QuantizedTensorStorage]],
        *,
        num_gemms: int,
        single_grouped_weight: bool,
        with_quantized_compute: bool,
        columnwise_usage: bool,
        activation_dtype: torch.dtype,
        is_first_microbatch: Optional[bool],
        skip_fp8_weight_update: Optional[torch.Tensor],
        cache_weight: bool,
    ) -> Tuple[
        Union[GroupedTensorStorage, List[torch.Tensor]],
        List[Optional[QuantizedTensorStorage]],
    ]:
        """Prepare a grouped parameter or discrete weights for GroupedTensor GEMM."""
        if single_grouped_weight:
            weight = weights[0]
            if not isinstance(weight, GroupedTensorStorage):
                raise TypeError(
                    "single_grouped_weight requires the weight parameter to be a GroupedTensor."
                )

            new_workspaces: List[Optional[QuantizedTensorStorage]] = [None]
            if weight.quantizer is not None:
                if not with_quantized_compute:
                    raise RuntimeError(
                        "Quantized single grouped weights require quantized grouped GEMM compute."
                    )
                return weight, new_workspaces

            if not with_quantized_compute:
                if weight.rowwise_data is None:
                    raise RuntimeError("Single grouped weight has no rowwise storage.")
                if weight.rowwise_data.dtype == activation_dtype:
                    return weight, new_workspaces
                data = weight.rowwise_data.to(dtype=activation_dtype)
                return (
                    GroupedTensorStorage(
                        shape=weight.logical_shape,
                        dtype=activation_dtype,
                        num_tensors=num_gemms,
                        shapes=weight.tensor_shapes,
                        quantizer=None,
                        data=data,
                    ),
                    new_workspaces,
                )

            weight_quantizer = weight_quantizers[0]
            if weight_quantizer is None:
                raise RuntimeError("Quantized grouped compute requires a weight quantizer.")
            weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
            # forward() already applied _enable_weight_preswizzle(); preserve that decision
            # because not every quantizer and weight shape supports fused quantize-swizzle.

            workspace = weight_workspaces[0] if weight_workspaces else None

            if workspace is not None and (
                workspace.quantizer is not weight_quantizer
                or (columnwise_usage and workspace.columnwise_data is None)
            ):
                workspace = None

            if weight.rowwise_data is None:
                raise RuntimeError("Single grouped weight has no rowwise storage to quantize.")
            source = weight.rowwise_data.view(weight.logical_shape)
            update_workspace = is_first_microbatch is None or is_first_microbatch
            if workspace is None:
                if cache_weight:
                    # Match quantize_weight(): persistent workspaces must be Tensor subclasses
                    # so autograd can save them without decomposing their storage metadata.
                    saved_internal = weight_quantizer.internal
                    weight_quantizer.internal = False
                grouped_weight = tex.group_quantize(
                    source,
                    weight_quantizer,
                    num_gemms,
                    None,
                )
                if cache_weight:
                    weight_quantizer.internal = saved_internal
            elif skip_fp8_weight_update is not None or update_workspace:
                grouped_weight = tex.group_quantize(
                    source,
                    weight_quantizer,
                    num_gemms,
                    None,
                    noop_flag=skip_fp8_weight_update,
                    output=workspace,
                )
            else:
                grouped_weight = workspace

            if cache_weight:
                new_workspaces[0] = grouped_weight
            return grouped_weight, new_workspaces

        weights_for_gemm: List[torch.Tensor] = []
        new_workspaces: List[Optional[QuantizedTensorStorage]] = [None] * len(weights)
        if not with_quantized_compute:
            return (
                [_GroupedLinear._maybe_dequantize(weight, activation_dtype) for weight in weights],
                new_workspaces,
            )

        update_ws = is_first_microbatch is None or is_first_microbatch
        for idx, weight in enumerate(weights):
            weight_quantizer = weight_quantizers[idx]
            weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
            weight_fp8, new_workspaces[idx] = quantize_weight(
                tensor=weight,
                quantizer=weight_quantizer,
                workspace=weight_workspaces[idx] if weight_workspaces else None,
                update_workspace=update_ws,
                skip_update_flag=skip_fp8_weight_update,
                workspace_dtype=activation_dtype,
                cache=cache_weight,
            )
            weights_for_gemm.append(weight_fp8)
        return weights_for_gemm, new_workspaces

    @staticmethod
    def _validate_or_alloc_output(
        buffer: Optional[torch.Tensor],
        rows: int,
        cols: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Validate and return the caller's output buffer, or allocate one if it is None.

        The buffer must be a 2D, contiguous, non-grad tensor matching the required shape,
        dtype, and device. Validation reads host-side metadata only, with no device sync.
        """
        if buffer is None:
            return torch.empty((rows, cols), dtype=dtype, device=device)
        expected_shape = (rows, cols)
        if buffer.shape != expected_shape:
            raise ValueError(
                f"Output buffer shape {tuple(buffer.shape)} must match required {expected_shape}."
            )
        if buffer.dtype != dtype:
            raise ValueError(f"Output buffer dtype {buffer.dtype} does not match required {dtype}.")
        if buffer.device != device:
            raise ValueError(
                f"Output buffer device {buffer.device} does not match required {device}."
            )
        if not buffer.is_contiguous():
            raise ValueError("Output buffer must be contiguous.")
        if buffer.requires_grad:
            raise ValueError("Output buffer must not require gradient.")
        return buffer

    @staticmethod
    def _prepare_bias_for_grouped_tensor_gemm(
        biases: Tuple[torch.Tensor, ...],
        *,
        single_grouped_bias: bool,
        num_gemms: int,
        out_features: int,
        dtype: torch.dtype,
    ) -> GroupedTensorStorage:
        """Prepare grouped or discrete bias storage for grouped GEMM."""
        if not single_grouped_bias:
            return _GroupedLinear._make_grouped_bias(
                biases,
                num_gemms=num_gemms,
                out_features=out_features,
                dtype=dtype,
            )

        bias = biases[0]
        if not isinstance(bias, GroupedTensorStorage):
            raise TypeError("single_grouped_bias requires a GroupedTensor parameter.")
        bias_data = bias.rowwise_data
        if bias_data.dtype != dtype:
            bias_data = bias_data.to(dtype=dtype)

        # The parameter exposes a packed [num_gemms, out_features] tensor, but its grouped
        # members are 1D vectors. The grouped bias-add kernel consumes those same bytes as
        # num_gemms row matrices with shape [1, out_features].
        return GroupedTensorStorage(
            shape=(num_gemms, out_features),
            dtype=dtype,
            num_tensors=num_gemms,
            shapes=[(1, out_features)] * num_gemms,
            quantizer=None,
            data=bias_data.reshape(-1),
        )

    @staticmethod
    def _forward_grouped_tensor(ctx, fwd_args: GroupedLinearFwdArgs) -> Tuple[torch.Tensor, list]:
        """Run the grouped forward and persist its autograd state."""
        out, new_workspaces, saved = _grouped_linear_fused_forward(fwd_args)
        if ctx is not None:
            bwd_args = GroupedLinearFusedBwdArgs()
            _grouped_linear_fused_setup(bwd_args, fwd_args)
            tensors_to_save, tensor_objects = prepare_for_saving(*saved)
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects
            ctx.use_grouped_tensor_path = True
            ctx.inp_shape = fwd_args.inp.shape
            ctx.backward_objects = bwd_args
            if fwd_args.fp8 and requires_grad(
                fwd_args.inp, fwd_args.weights[0], fwd_args.biases[0]
            ):
                bwd_args.reduce_and_update_bwd_fp8_tensors = (
                    FP8GlobalStateManager.is_first_fp8_module()
                )
        return out, new_workspaces

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        fwd_args: GroupedLinearFwdArgs,
        *weights_and_biases,
    ) -> Tuple[torch.Tensor, list]:
        """Forward pass: compute grouped linear output and set up autograd context.

        ``inp`` and the weights / biases are positional Tensor arguments so
        autograd tracks them; they are immediately re-attached to ``fwd_args``
        so every downstream helper can be invoked with a single argument.
        """
        num_gemms = fwd_args.num_gemms
        fwd_args.inp = inp
        num_weight_args = 1 if fwd_args.single_grouped_weight else num_gemms
        fwd_args.weights = list(weights_and_biases[:num_weight_args])
        fwd_args.biases = list(weights_and_biases[num_weight_args:])

        if fwd_args.use_grouped_tensor_path:
            return _GroupedLinear._forward_grouped_tensor(ctx, fwd_args)

        out, new_workspaces, saved = _grouped_linear_forward_impl(fwd_args)
        if ctx is not None:
            ctx.use_grouped_tensor_path = False
            bwd_args = GroupedLinearBwdArgs()
            _grouped_linear_setup_ctx(bwd_args, fwd_args)
            tensors_to_save, tensor_objects = prepare_for_saving(*saved)
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects
            ctx.backward_objects = bwd_args
            if fwd_args.fp8 and requires_grad(inp, fwd_args.weights[0], fwd_args.biases[0]):
                bwd_args.reduce_and_update_bwd_fp8_tensors = (
                    FP8GlobalStateManager.is_first_fp8_module()
                )
            if fwd_args.backward_override is not None:
                bwd_args.reduce_and_update_bwd_fp8_tensors = False

        return out, new_workspaces

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor, _grad_workspaces
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring
        with get_nvtx_range_context("_GroupedLinear_backward"):
            bwd_args = ctx.backward_objects
            bwd_args.grad_output = grad_output
            bwd_args.setup_saved_tensors(ctx)
            if ctx.use_grouped_tensor_path:
                dgrad, wgrad_list, grad_biases = _grouped_linear_fused_backward(bwd_args)
                if dgrad is not None:
                    dgrad = dgrad.view(ctx.inp_shape)
            else:
                dgrad, wgrad_list, grad_biases = _grouped_linear_backward_impl(bwd_args)
            reduce_and_update_bwd_fp8_tensors = bwd_args.reduce_and_update_bwd_fp8_tensors
            # Release saved state after either backward path.
            ctx.backward_objects = None
            del bwd_args

        if reduce_and_update_bwd_fp8_tensors:
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
        return (dgrad, None, *wgrad_list, *grad_biases)


class GroupedLinear(TransformerEngineBaseModule):
    """Applies linear transformations to the incoming data list
       :math:`y_i = x_iA_i^T + b_i` in a grouped way.

    Parameters
    ----------
    num_gemms : int
                number of GEMMs to be performed simutaneously.
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    bias : bool, default = True
          if set to ``False``, the layer will not learn an additive bias.
    init_method : Callable, default = None
                 used for initializing weights in the following way: ``init_method(weight)``.
                 When set to ``None``, defaults to ``torch.nn.init.normal_(mean=0.0, std=0.023)``.
    get_rng_state_tracker : Callable, default = None
                 used to get the random number generator state tracker for initializing weights.
    rng_tracker_name : str, default = None
                 the param passed to get_rng_state_tracker to get the specific rng tracker.
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will be allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = False
                             if set to ``True``, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional ``main_grad`` attribute (used instead of the
                             regular ``grad``) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in. This argument along with
                             weight tensor having attribute 'overwrite_main_grad' set to True
                             will overwrite ``main_grad`` instead of accumulating.
    return_bias : bool, default = False
                 when set to ``True``, this module will not apply the additive bias itself, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations. A single grouped
                 bias is returned as its packed ``GroupedTensor`` parameter; discrete biases
                 are returned as a list of per-GEMM tensors.
    params_dtype : torch.dtype, default = torch.get_default_dtype()
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    delay_wgrad_compute : bool, default = False
                         Whether to delay weight gradient computation
    save_original_input : bool, default = False
                       If set to ``True``, always saves the original input tensor rather than the
                       cast tensor. In some scenarios, the input tensor is used by multiple modules,
                       and saving the original input tensor may reduce the memory usage.
                       Requires input quantizers that can safely reproduce their results from the
                       original input. Cannot work with FP8 DelayedScaling recipe.
    single_grouped_weight : bool, default = False
                       If set to ``True``, grouped weights are stored as a single grouped parameter
                       instead of one parameter per GEMM.
                       EXPERIMENTAL and subject to change. Gated by the
                       ``NVTE_GROUPED_LINEAR_SINGLE_PARAM`` environment variable: if the env var
                       is not set this argument is forced to ``False`` with a warning.
    single_grouped_bias : bool, default = False
                       If set to ``True``, grouped biases are stored as a single grouped bias
                       instead of one bias per GEMM.
                       EXPERIMENTAL and subject to change. Gated by the
                       ``NVTE_GROUPED_LINEAR_SINGLE_PARAM`` environment variable: if the env var
                       is not set this argument is forced to ``False`` with a warning.
    use_grouped_tensor : bool or None, default = None
                       Prefer the native GroupedTensor grouped GEMM path. Discrete parameters
                       fall back to split-quantize when the path is unsupported. Single grouped
                       parameters require the native path and raise instead of falling back.
                       The native path requires CUDA ``m_splits``. ``None`` preserves the deprecated
                       ``NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM`` environment-variable
                       selection for compatibility. New callers should pass a boolean explicitly.

    Notes
    -----
    GroupedLinear doesn't really handle the TP communications inside. The ``tp_size`` and
    ``parallel_mode`` are used to determine the shapes of weights and biases.
    The TP communication should be handled in the dispatch and combine stages of MoE models.
    """

    def __init__(
        self,
        num_gemms: int,
        in_features: int,
        out_features: int,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        rng_tracker_name: Optional[str] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        device: Union[torch.device, str] = "cuda",
        ub_overlap_rs: bool = False,
        ub_overlap_ag: bool = False,
        ub_name: Optional[str] = None,
        delay_wgrad_compute: bool = False,
        save_original_input: bool = False,
        single_grouped_weight: bool = False,
        single_grouped_bias: bool = False,
        name: Optional[str] = None,
        use_grouped_tensor: Optional[bool] = None,
    ) -> None:
        super().__init__(name)

        self.params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.ub_overlap_rs = ub_overlap_rs
        self.ub_overlap_ag = ub_overlap_ag
        self.ub_name = ub_name
        self.save_original_input = save_original_input
        if use_grouped_tensor is None:
            use_grouped_tensor_env = os.getenv("NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM")
            if use_grouped_tensor_env is not None:
                warnings.warn(
                    "NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM is deprecated and will be "
                    "removed in a future release. Pass use_grouped_tensor=True or "
                    "use_grouped_tensor=False to GroupedLinear instead.",
                    FutureWarning,
                    stacklevel=2,
                )
            else:
                use_grouped_tensor_env = "0"
            use_grouped_tensor = bool(int(use_grouped_tensor_env))
        if not isinstance(use_grouped_tensor, bool):
            raise TypeError(
                f"use_grouped_tensor must be a bool or None, got {type(use_grouped_tensor)}."
            )
        self.use_grouped_tensor = use_grouped_tensor
        single_grouped_weight, single_grouped_bias = resolve_grouped_linear_single_param_flags(
            single_grouped_weight, single_grouped_bias
        )
        self.single_grouped_weight = single_grouped_weight
        self.single_grouped_bias = single_grouped_bias
        if self.use_bias and self.single_grouped_weight and not self.single_grouped_bias:
            warnings.warn(
                "GroupedLinear has single_grouped_weight=True and bias=True, but "
                "single_grouped_bias=False. This requires packing the per-GEMM biases on every "
                "forward; enable single_grouped_bias to keep both parameters grouped.",
                UserWarning,
                stacklevel=2,
            )
        if ub_overlap_rs or ub_overlap_ag:
            raise ValueError("GroupedLinear doesn't support Userbuffer overlap.")
        self.init_method = init_method
        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name

        self.wgrad_store = WeightGradStore(delay_wgrad_compute)

        self._offsets = {
            "input": 0,
            "weight": 1,
            "output": 2,
            "grad_output": 0,
            "grad_input": 1,
        }
        self._num_fp8_tensors_per_gemm = {
            "fwd": 3,
            "bwd": 2,
        }
        self._custom_quantizer_cache = {}
        self._uses_custom_recipe = False

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        if self.tp_size > 1 and bias:
            raise ValueError(
                "GroupedLinear doesn't support bias when TP > 1. "
                "Because the TP communication is handled outside of this module."
            )

        self.parallel_mode = parallel_mode
        if self.parallel_mode not in GemmParallelModes:
            raise ValueError(
                f"parallel_mode {parallel_mode!r} not supported."
                f" Supported modes: {GemmParallelModes}"
            )

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        for i in range(self.num_gemms):
            # Construct weight parameter
            self.register_parameter(
                f"weight{i}",
                torch.nn.Parameter(
                    torch.empty(
                        self.out_features,
                        self.in_features,
                        device=device,
                        dtype=self.params_dtype,
                    ),
                ),
                init_fn=init_method,
                get_rng_state_tracker=get_rng_state_tracker,
                fp8_meta_index=self._offsets["weight"] + i * self._num_fp8_tensors_per_gemm["fwd"],
            )

            # Construct bias parameters if needed
            if self.use_bias:
                self.register_parameter(
                    f"bias{i}",
                    torch.nn.Parameter(
                        torch.empty(
                            self.out_features,
                            device=device,
                            dtype=self.params_dtype,
                        ),
                    ),
                    init_fn=init_method_constant(0.0),
                )
            else:
                bias = torch.Tensor().to(dtype=self.params_dtype, device=device)
                setattr(self, f"bias{i}", bias)

        if self.primary_weights_in_fp8:
            self.init_fp8_metadata(num_gemms=self.num_gemms)

        is_meta = torch.device(device).type == "meta"
        self.reset_parameters(defer_init=is_meta)

        if self.wgrad_store.delay_wgrad_compute():
            for name, param in self.named_parameters():
                if name in ("weight", "bias"):
                    param.skip_backward_post_hook = True
                    continue
                for i in range(self.num_gemms):
                    if name in (f"weight{i}", f"bias{i}"):
                        param.skip_backward_post_hook = True

    def set_meta_tensor(self, fwd: bool, recipe: Recipe) -> None:
        """Init scales and amaxes for fwd | bwd."""
        super().set_meta_tensor(fwd, recipe)

        # Recipe-specific quantizer configuration
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        if recipe.float8_current_scaling():
            self._customize_quantizers_float8_current_scaling(fwd, recipe)

        self._uses_custom_recipe = recipe.custom()
        self._validate_custom_recipe_quantizers(fwd, recipe)

    def _validate_custom_recipe_quantizers(self, fwd: bool, recipe: Recipe) -> None:
        """Validate one CustomRecipe quantizer generation."""
        if not recipe.custom():
            return

        # A CustomRecipe factory may return a different quantizer for every expert,
        # while grouped execution selects its implementation from expert 0. Validate
        # every newly constructed list once and keep the steady-state path O(1).
        meta_key = "scaling_fwd" if fwd else "scaling_bwd"
        generation = self.quantizers.get(meta_key)
        if generation is None:
            return
        if self._custom_quantizer_cache.get(meta_key) is generation:
            return
        if not fwd and not torch.is_grad_enabled():
            return

        if fwd:
            stride = self._num_fp8_tensors_per_gemm["fwd"]
            input_quantizers = tuple(
                generation[self._offsets["input"] + i * stride] for i in range(self.num_gemms)
            )
            weight_quantizers = tuple(
                generation[self._offsets["weight"] + i * stride] for i in range(self.num_gemms)
            )
            _split_quantization.validate_grouped_quantizer_list(
                input_quantizers, operand_name="input"
            )
            _split_quantization.validate_grouped_quantizer_list(
                weight_quantizers, operand_name="weight"
            )
        else:
            stride = self._num_fp8_tensors_per_gemm["bwd"]
            grad_output_quantizers = tuple(
                generation[self._offsets["grad_output"] + i * stride] for i in range(self.num_gemms)
            )
            _split_quantization.validate_grouped_quantizer_list(
                grad_output_quantizers,
                operand_name="grad_output",
            )

        # Cache only a fully validated generation so failures are retried.
        self._custom_quantizer_cache[meta_key] = generation

    def get_quantizer_roles(
        self,
        *,
        fwd: bool,
        num_quantizers: int,
    ) -> Optional[List[QuantizerRole]]:
        """QuantizerRole list for quantizers used by ``GroupedLinear``.

        For grouped GEMMs we repeat the same pattern for each GEMM in
        order.  The output (fwd) and grad-input (bwd) slots default to
        ``None`` (unknown consumer).  Set :attr:`output_quantizer_role` /
        :attr:`grad_input_quantizer_role` to provide consumer identity.
        """
        name = self.name or ""
        if fwd:
            base = [
                QuantizerRole(module_type="grouped_linear", tensor_type="input", name=name),
                QuantizerRole(module_type="grouped_linear", tensor_type="weight", name=name),
                self._output_quantizer_role,
            ]
        else:
            base = [
                QuantizerRole(module_type="grouped_linear", tensor_type="grad_output", name=name),
                self._grad_input_quantizer_role,
            ]
        return [base[i % len(base)] for i in range(num_quantizers)]

    def make_grouped_weights(self, defer_init=False) -> None:
        """
        Convert parameters into a GroupedTensor and re-register them as parameters.
        """

        if defer_init:
            return

        weight_quantizers = self._get_weight_quantizers()
        # TODO(#3158): Support Identity/Hybrid single grouped weights.
        unsupported_quantizers = tuple(
            type(quantizer).__name__
            for quantizer in weight_quantizers
            if isinstance(quantizer, (IdentityQuantizer, HybridQuantizer))
        )
        if unsupported_quantizers:
            quantizer_names = ", ".join(dict.fromkeys(unsupported_quantizers))
            raise NotImplementedError(
                "GroupedLinear(single_grouped_weight=True) does not support "
                f"{quantizer_names} weight quantizers yet. Set "
                "single_grouped_weight=False or unset "
                "NVTE_GROUPED_LINEAR_SINGLE_PARAM. See #3158."
            )

        recipe = (
            weight_quantizers[0]._get_compatible_recipe()
            if weight_quantizers and weight_quantizers[0] is not None
            else None
        )
        if recipe is not None and recipe.delayed():
            self.set_tensor_parallel_attributes(defer_init=defer_init)
            return

        weights = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]

        # TE preserves the original BF16/FP16 initialization on each quantized
        # parameter so distributed optimizers can construct lossless FP32 masters.
        # Packing the parameters must transfer those values to the new registered
        # grouped parameter; otherwise its master is initialized by dequantizing
        # MXFP8 and starts from a different value than the discrete-weight layout.
        high_precision_init_vals = [_get_high_precision_init_val(weight) for weight in weights]
        if any(value is not None for value in high_precision_init_vals) and not all(
            value is not None for value in high_precision_init_vals
        ):
            raise RuntimeError(
                "Grouped weights have inconsistent high-precision initialization state"
            )

        # Create the weight storage.
        grouped_weights = GroupedTensor.make_grouped_tensor_with_shapes(
            num_tensors=self.num_gemms,
            shapes=[(self.out_features, self.in_features)] * self.num_gemms,
            quantizer=weight_quantizers[0],
            dtype=self.params_dtype,
            device=weights[0].device,
        )

        # Copy existing params into storage.
        with torch.no_grad():
            for i in range(self.num_gemms):
                if self.primary_weights_in_fp8:
                    grouped_weights.quantized_tensors[i].copy_from_storage(weights[i])
                else:
                    grouped_weights.quantized_tensors[i].copy_(weights[i])

        # Re-register as a single grouped weight parameter.
        if not (
            isinstance(grouped_weights, torch.Tensor)
            and (weight_quantizers[0] is None or not weight_quantizers[0].internal)
        ):
            raise RuntimeError("Found internal quantizer with `single_grouped_weight=True`.")
        grouped_parameter = torch.nn.Parameter(grouped_weights)
        if all(value is not None for value in high_precision_init_vals):
            _attach_high_precision_init_val(
                grouped_parameter,
                torch.stack(high_precision_init_vals, dim=0),
            )
            for weight in weights:
                _clear_high_precision_init_val(weight)

        self.register_parameter(
            "weight",
            grouped_parameter,
            init_fn=self.init_method,
            get_rng_state_tracker=self.get_rng_state_tracker,
            fp8_meta_index=self._offsets["weight"],
        )
        for i in range(self.num_gemms):
            self.register_parameter(f"weight{i}", None)

        if self.use_bias and self.single_grouped_bias:
            self._make_grouped_biases()

        self.set_tensor_parallel_attributes(defer_init=defer_init)

    def _make_grouped_biases(self) -> None:
        """Pack per-GEMM biases into one ``GroupedTensor`` (``single_grouped_bias``)."""
        biases = [getattr(self, f"bias{i}") for i in range(self.num_gemms)]
        packed = torch.stack([b.detach().clone() for b in biases], dim=0).contiguous()
        grouped_bias = GroupedTensor.make_grouped_tensor_from_rowwise_data(
            num_tensors=self.num_gemms,
            tensor_shape=(self.out_features,),
            rowwise_data=packed,
            dtype=packed.dtype,
        )
        grouped_bias.requires_grad_(True)
        self.register_parameter("bias", torch.nn.Parameter(grouped_bias))
        for i in range(self.num_gemms):
            self.register_parameter(f"bias{i}", None)

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)
        # Grouped tensor weights / biases are opt-in features.
        if self.single_grouped_weight:
            self.make_grouped_weights(defer_init=defer_init)
        elif self.single_grouped_bias:
            self._make_grouped_biases()

    def set_tensor_parallel_attributes(self, defer_init=False) -> None:
        """Set attributes needed for TP"""

        if not defer_init:
            # Set parallelism attributes for linear weights
            grouped_weight = getattr(self, "weight", None)
            if grouped_weight is not None:
                set_tensor_model_parallel_attributes(
                    tensor=grouped_weight,
                    is_parallel=True,
                    dim=1 if self.parallel_mode == "row" else 0,
                    stride=1,
                )
            else:
                for i in range(self.num_gemms):
                    set_tensor_model_parallel_attributes(
                        tensor=getattr(self, f"weight{i}"),
                        is_parallel=True,
                        dim=1 if self.parallel_mode == "row" else 0,
                        stride=1,
                    )

            # Set parallelism attributes for linear biases
            if self.use_bias:
                grouped_bias = getattr(self, "bias", None)
                if grouped_bias is not None:
                    if self.parallel_mode == "row":
                        setattr(grouped_bias, "sequence_parallel", self.sequence_parallel)
                    elif self.parallel_mode == "column":
                        set_tensor_model_parallel_attributes(grouped_bias, True, 0, 1)
                else:
                    for i in range(self.num_gemms):
                        if self.parallel_mode == "row":
                            setattr(
                                getattr(self, f"bias{i}"),
                                "sequence_parallel",
                                self.sequence_parallel,
                            )
                        elif self.parallel_mode == "column":
                            set_tensor_model_parallel_attributes(
                                getattr(self, f"bias{i}"), True, 0, 1
                            )

    def _remap_grouped_weight_state_dict_keys(self, state_dict, prefix: str) -> None:
        """Remap weight keys between single and per-GEMM checkpoint formats."""
        grouped_weight_key = f"{prefix}weight"
        per_gemm_weight_keys = [f"{prefix}weight{i}" for i in range(self.num_gemms)]
        has_grouped_weight = grouped_weight_key in state_dict
        has_per_gemm_weights = all(key in state_dict for key in per_gemm_weight_keys)

        if self.single_grouped_weight:
            # Backward compatibility: checkpoints saved without single_grouped_weight
            # store one weight tensor per GEMM (weight0..weightN). Convert them into a
            # single stacked grouped weight expected by this module configuration.
            if not has_grouped_weight and has_per_gemm_weights:
                per_gemm_weights = [state_dict.pop(key) for key in per_gemm_weight_keys]
                per_gemm_weights = [
                    (weight.dequantize() if isinstance(weight, QuantizedTensorStorage) else weight)
                    for weight in per_gemm_weights
                ]
                state_dict[grouped_weight_key] = torch.stack(per_gemm_weights, dim=0)
            elif has_grouped_weight:
                # Drop any redundant per-GEMM keys to avoid strict-load unexpected-key errors.
                for key in per_gemm_weight_keys:
                    state_dict.pop(key, None)
        else:
            # Forward compatibility: checkpoints saved with single_grouped_weight
            # store one grouped `weight`. Convert it back to weight0..weightN.
            if not has_per_gemm_weights and has_grouped_weight:
                grouped_weight = state_dict.pop(grouped_weight_key)
                if hasattr(grouped_weight, "split_into_quantized_tensors"):
                    grouped_members = grouped_weight.quantized_tensors
                    if grouped_members is None:
                        grouped_members = grouped_weight.split_into_quantized_tensors()
                    per_gemm_weights = [
                        (
                            weight.dequantize()
                            if isinstance(weight, QuantizedTensorStorage)
                            else weight
                        )
                        for weight in grouped_members
                    ]
                else:
                    grouped_weight = (
                        grouped_weight.dequantize()
                        if isinstance(grouped_weight, QuantizedTensorStorage)
                        else grouped_weight
                    )
                    per_gemm_weights = list(grouped_weight.unbind(dim=0))
                for i, weight in enumerate(per_gemm_weights):
                    state_dict[f"{prefix}weight{i}"] = weight
            elif has_per_gemm_weights:
                # Drop any redundant grouped key to avoid strict-load unexpected-key errors.
                state_dict.pop(grouped_weight_key, None)

    def _remap_grouped_bias_state_dict_keys(self, state_dict, prefix: str) -> None:
        """Remap bias keys between single grouped and per-GEMM checkpoint formats."""
        if not self.use_bias:
            return
        grouped_bias_key = f"{prefix}bias"
        per_gemm_bias_keys = [f"{prefix}bias{i}" for i in range(self.num_gemms)]
        has_grouped_bias = grouped_bias_key in state_dict
        has_per_gemm_biases = all(key in state_dict for key in per_gemm_bias_keys)

        if self.single_grouped_bias:
            if not has_grouped_bias and has_per_gemm_biases:
                per_gemm = [state_dict.pop(key) for key in per_gemm_bias_keys]
                state_dict[grouped_bias_key] = torch.stack(per_gemm, dim=0)
            elif has_grouped_bias:
                for key in per_gemm_bias_keys:
                    state_dict.pop(key, None)
                val = state_dict[grouped_bias_key]
                if isinstance(val, torch.Tensor) and val.dim() == 3 and val.shape[1] == 1:
                    state_dict[grouped_bias_key] = val.squeeze(1)
        else:
            if not has_per_gemm_biases and has_grouped_bias:
                gb = state_dict.pop(grouped_bias_key)
                if hasattr(gb, "split_into_quantized_tensors"):
                    members = gb.quantized_tensors
                    if members is None:
                        members = gb.split_into_quantized_tensors()
                    per_gemm = [m.reshape(-1) if m.dim() > 1 else m for m in members]
                else:
                    per_gemm = list(gb.unbind(0))
                for i, b in enumerate(per_gemm):
                    state_dict[f"{prefix}bias{i}"] = b.reshape(-1) if b.dim() > 1 else b
            elif has_per_gemm_biases:
                state_dict.pop(grouped_bias_key, None)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load state dict with grouped-weight format compatibility."""
        state_dict_copy = state_dict.copy()
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            state_dict_copy._metadata = metadata
        self._remap_grouped_weight_state_dict_keys(state_dict_copy, prefix="")
        self._remap_grouped_bias_state_dict_keys(state_dict_copy, prefix="")
        return super().load_state_dict(state_dict_copy, strict=strict, assign=assign)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Load state, including compatibility across grouped-weight checkpoint formats."""
        self._remap_grouped_weight_state_dict_keys(state_dict, prefix)
        self._remap_grouped_bias_state_dict_keys(state_dict, prefix)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        m_splits: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
        out: Optional[torch.Tensor] = None,
        dgrad_out: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply the linear transformation to the input.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        m_splits : torch.Tensor
                 Split sizes for the input tensor.
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. Between the microbatches of the same minibatch
                             the model weights are not updated. Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
                             * it also allows skipping gradient accumulation during the
                               first microbatch (since it is the first gradient being
                               produced)
        out : torch.Tensor, default = None
             Optional preallocated buffer for the forward output; the returned tensor
             aliases it with no copy. Must be a 2D, contiguous, non-grad tensor of shape
             [num_tokens, out_features] in the activation dtype. Only the first
             sum(m_splits) rows are written; any padded trailing rows are left unchanged.
             Can be given independently of dgrad_out. If the buffer is reused across
             iterations, pass ``buffer.detach()`` so autograd does not set its
             ``requires_grad`` (which would trip the non-grad check on the next call).
        dgrad_out : torch.Tensor, default = None
             Optional preallocated buffer for the backward input gradient, of shape
             [num_tokens, in_features] with the same constraints as out. Receives the
             final gradient only when inp has a single consumer in the autograd graph;
             otherwise autograd accumulates into a new tensor.
        """
        debug = self.is_debug_iter()
        is_grad_enabled = torch.is_grad_enabled()
        num_gemms = self.num_gemms

        # Make sure splits are in expected format
        if (self.single_grouped_weight or self.single_grouped_bias) and not self.use_grouped_tensor:
            raise RuntimeError(
                "single_grouped_weight and single_grouped_bias require "
                "use_grouped_tensor=True; the split-quantize path only supports discrete "
                "parameters."
            )
        if not isinstance(m_splits, torch.Tensor):
            # Convert list of ints to tensor for backward compatibility
            m_splits = torch.tensor(m_splits, dtype=torch.int64, device="cpu")
        elif m_splits.dtype != torch.int64:
            m_splits = m_splits.to(dtype=torch.int64)
        if m_splits.size() != (num_gemms,):
            raise ValueError(
                f"Shape of splits tensor ({tuple(m_splits.size())}) "
                f"does not match number of GEMMs ({num_gemms})."
            )

        if FP8GlobalStateManager.fp8_graph_capturing():
            skip_fp8_weight_update = (
                FP8GlobalStateManager.quantization_state.skip_fp8_weight_update_tensor
            )
        else:
            skip_fp8_weight_update = None
        if skip_fp8_weight_update is not None:
            is_first_microbatch = False

        # Preprocess input tensor
        if isinstance(inp, QuantizedTensorStorage):
            raise TypeError("GroupedLinear doesn't support input tensor in FP8.")
        inp = self.prepare_forward(inp, num_gemms=self.num_gemms)

        try:
            weight_tensors = self._get_weight_tensors()
            bias_tensors = self._get_bias_tensors()
            use_grouped_bias = self.use_bias and self.single_grouped_bias

            quantizers = self._get_quantizers() if not debug else self._get_debug_quantizers()

            if debug:
                if self.no_debug_features_active(list(chain(*quantizers))):
                    debug = False
                    quantizers = self._get_quantizers()
            if debug and (self.single_grouped_weight or self.single_grouped_bias):
                raise RuntimeError(
                    "TE debug features do not support single grouped parameters. DebugQuantizer "
                    "uses the split-quantize path, which only supports discrete parameters. "
                    "Disable single_grouped_weight and single_grouped_bias, or disable TE debug "
                    "features for this GroupedLinear."
                )

            (
                input_quantizers,
                weight_quantizers,
                output_quantizers,
                grad_input_quantizers,
                grad_weight_quantizers,
                grad_output_quantizers,
            ) = quantizers
            if not debug and weight_quantizers[0] is not None:
                # Experts share shape and recipe settings: compute once and broadcast.
                optimize_for_gemm = self._enable_weight_preswizzle(
                    weight_quantizers[0], weight_tensors[0]
                )
                for q in weight_quantizers:
                    q.optimize_for_gemm = optimize_for_gemm

            cache_weight = is_first_microbatch is not None
            if self.single_grouped_weight:
                weight_workspaces = [self._fp8_workspaces.get("weight")] if cache_weight else [None]
            else:
                weight_workspaces = (
                    [self._fp8_workspaces.get(f"weight{i}") for i in range(num_gemms)]
                    if cache_weight
                    else [None] * num_gemms
                )

            weight_requires_grad = weight_tensors[0].requires_grad
            fprop_use_split_accumulator = _2X_ACC_FPROP
            dgrad_use_split_accumulator = _2X_ACC_DGRAD
            wgrad_use_split_accumulator = _2X_ACC_WGRAD
            if self.fp8:
                _recipe = FP8GlobalStateManager.get_fp8_recipe()
                backward_override = _recipe.backward_override
                if hasattr(_recipe, "fp8_gemm_fprop"):
                    fprop_use_split_accumulator = _recipe.fp8_gemm_fprop.use_split_accumulator
                if hasattr(_recipe, "fp8_gemm_dgrad"):
                    dgrad_use_split_accumulator = _recipe.fp8_gemm_dgrad.use_split_accumulator
                if hasattr(_recipe, "fp8_gemm_wgrad"):
                    wgrad_use_split_accumulator = _recipe.fp8_gemm_wgrad.use_split_accumulator
            else:
                _recipe = None
                backward_override = None

            save_original_input = self.save_original_input
            if backward_override == "high_precision":
                save_original_input = True
            elif backward_override == "dequantized":
                save_original_input = False
            backward_needs_input = is_grad_enabled and weight_requires_grad
            if backward_override is None and save_original_input and backward_needs_input:
                if _recipe is not None and _recipe.delayed():
                    raise ValueError(
                        "DelayedScaling recipe is not supported with save_original_input"
                    )

                # Megatron-Core may enable this automatically to reuse an activation
                # already retained by an upstream operation. Built-in recipes guarantee
                # group homogeneity, while CustomRecipe generations are validated once,
                # so runtime safety can be determined from expert 0.
                if _recipe is not None:
                    input_quantizer = input_quantizers[0]
                    if isinstance(input_quantizer, Float8Quantizer):
                        warnings.warn(
                            "save_original_input is incompatible with delayed-scaling quantizers "
                            "(Float8Quantizer). Disabling save_original_input for this module.",
                            stacklevel=2,
                        )
                        save_original_input = False
                    elif not can_reconstruct_wgrad_input_from_original(input_quantizer):
                        warnings.warn(
                            "Ignoring save_original_input=True because the input quantizer cannot "
                            "safely reconstruct the backward operand from the original input "
                            f"({input_quantizer}).",
                            stacklevel=2,
                        )
                        save_original_input = False

            cpu_offloading = is_cpu_offload_enabled()
            use_grouped_tensor_path = False
            if self.use_grouped_tensor and not (
                self.fp8_calibration
                or debug
                or cpu_offloading
                or any(q is not None for q in output_quantizers)
            ):
                if (
                    self.fp8
                    and _recipe.float8_block_scaling()
                    and (10, 0) <= get_device_compute_capability() <= (11, 0)
                ):
                    raise RuntimeError(
                        "use_grouped_tensor=True does not support the FP8 block-scaling recipe on"
                        " Blackwell GPUs: the native grouped FP8 block-scaling path is Hopper-only."
                        " Set use_grouped_tensor=False, or unset"
                        " NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM if it enabled this path, to"
                        " use the MXFP8-emulated path on Blackwell."
                    )
                use_grouped_tensor_path = is_module_grouped_tensor_path_supported(
                    _recipe,
                    self.activation_dtype,
                )
            if (
                self.use_grouped_tensor
                and not use_grouped_tensor_path
                and (self.single_grouped_weight or use_grouped_bias)
            ):
                raise RuntimeError(
                    "Single grouped parameters require the native grouped-tensor path, but the"
                    " active device, cuBLASLt version, quantization recipe, or GroupedLinear"
                    " feature configuration does not support it. Disable single_grouped_weight and"
                    " single_grouped_bias to allow the split-quantize fallback."
                )
            if use_grouped_tensor_path:
                if m_splits.device.type != "cuda":
                    raise ValueError(
                        "The native grouped_tensor path requires CUDA m_splits. Pass a CUDA int64 "
                        "tensor, or set use_grouped_tensor=False."
                    )
            wgrad_store = (
                self.wgrad_store
                if self.wgrad_store is not None and self.wgrad_store.delay_wgrad_compute()
                else None
            )

            fwd_args = GroupedLinearFwdArgs(
                # tensors
                inp=inp,
                weights=list(weight_tensors),
                biases=list(bias_tensors),
                weight_workspaces=weight_workspaces,
                out=out,
                dgrad_out=dgrad_out,
                skip_fp8_weight_update=skip_fp8_weight_update,
                m_splits_tensor=m_splits,
                # requires_grad flags
                input_requires_grad=inp.requires_grad,
                weights_requires_grad=weight_requires_grad,
                # quantizers
                input_quantizers=input_quantizers,
                weight_quantizers=weight_quantizers,
                output_quantizers=output_quantizers,
                grad_input_quantizers=grad_input_quantizers,
                grad_weight_quantizers=grad_weight_quantizers,
                grad_output_quantizers=grad_output_quantizers,
                # split geometry
                m_splits=None,
                num_gemms=num_gemms,
                # numerical / dtype config
                activation_dtype=self.activation_dtype,
                fp8=self.fp8,
                fp8_calibration=self.fp8_calibration,
                save_original_input=save_original_input,
                backward_override=backward_override,
                fprop_use_split_accumulator=fprop_use_split_accumulator,
                dgrad_use_split_accumulator=dgrad_use_split_accumulator,
                wgrad_use_split_accumulator=wgrad_use_split_accumulator,
                debug=debug,
                # weight-workspace caching
                is_first_microbatch=is_first_microbatch,
                cache_weight=cache_weight,
                # fused GroupedTensor path
                use_grouped_tensor_path=use_grouped_tensor_path,
                single_grouped_weight=self.single_grouped_weight,
                single_grouped_bias=use_grouped_bias,
                # misc
                use_bias=self.apply_bias,
                fuse_wgrad_accumulation=self.fuse_wgrad_accumulation,
                wgrad_store=wgrad_store,
                cpu_offloading=cpu_offloading,
                is_grad_enabled=is_grad_enabled,
            )

            out, new_workspaces = _grouped_linear_eager(
                inp,
                m_splits,
                fwd_args,
                (*weight_tensors, *bias_tensors),
                is_grad_enabled,
            )

            if cache_weight:
                for i, ws in enumerate(new_workspaces):
                    if ws is not None:
                        if isinstance(ws, torch.Tensor):
                            ws = ws.detach()
                        key = "weight" if self.single_grouped_weight else f"weight{i}"
                        self._fp8_workspaces[key] = ws

        finally:
            self.end_forward()

        if self.return_bias:
            if use_grouped_bias:
                return out, bias_tensors[0]
            return out, [cast_if_needed(b, self.activation_dtype) for b in bias_tensors]
        return out

    def backward_dw(self):
        """
        Execute the delayed weight gradient computation.
        This method is called after the main backward pass to compute weight gradients.
        """
        if not self.need_backward_dw():
            return
        if self.wgrad_store.context is None or self.wgrad_store.context.empty():
            return
        with get_nvtx_range_context("_GroupedLinear_wgrad"):
            (_, grad_biases_, _), tensor_list = self.wgrad_store.pop()
            wgrad_output = tensor_list[2]
            weight_params = self._get_weight_tensors()
            if not self.fuse_wgrad_accumulation:
                if self.single_grouped_weight:
                    weight_params[0].grad = wgrad_output.rowwise_data.view(
                        self.num_gemms, self.out_features, self.in_features
                    ).to(weight_params[0].dtype)
                else:
                    for i in range(self.num_gemms):
                        weight_params[i].grad = wgrad_output[i].to(weight_params[i].dtype)
            has_grad_biases = [
                grad_bias is not None and grad_bias.numel() != 0 for grad_bias in grad_biases_
            ]
            if self.use_bias and any(has_grad_biases):
                if self.use_grouped_tensor:
                    raise RuntimeError(
                        "GroupedLinear(use_grouped_tensor=True) fell back to the split-quantize "
                        "path, which produced per-expert bias gradients during delayed wgrad. "
                        "This implicit fallback is unsupported with delay_wgrad_compute=True. "
                        "Use a configuration supported by the grouped-tensor path, or set "
                        "use_grouped_tensor=False to select the legacy path explicitly."
                    )
                bias_params = [getattr(self, f"bias{i}") for i in range(self.num_gemms)]
                for i in range(self.num_gemms):
                    if has_grad_biases[i] and bias_params[i].grad is None:
                        bias_params[i].grad = grad_biases_[i].to(bias_params[i].dtype)
            del grad_biases_
            del wgrad_output
            del tensor_list
            self._trigger_wgrad_accumulation_and_reduce_hooks()

    def _customize_quantizers_float8_current_scaling(self, fwd: bool, recipe: Recipe) -> None:
        """Customize quantizers based on current scaling recipe + linear."""

        if self.tp_size > 1:
            raise ValueError(
                "GroupedLinear doesn't support TP > 1 with Float8 current scaling. "
                "Because the TP communication is handled outside of this module."
            )

        if fwd:
            for i in range(self.num_gemms):
                # set configs about amax epsilon and power_2_scale
                self.quantizers["scaling_fwd"][
                    self._offsets["input"] + i * self._num_fp8_tensors_per_gemm["fwd"]
                ].force_pow_2_scales = recipe.fp8_quant_fwd_inp.power_2_scale
                self.quantizers["scaling_fwd"][
                    self._offsets["input"] + i * self._num_fp8_tensors_per_gemm["fwd"]
                ].amax_epsilon = recipe.fp8_quant_fwd_inp.amax_epsilon
                # also set weight quantizer with same amax_epsilon & power_2_scale
                self.quantizers["scaling_fwd"][
                    self._offsets["weight"] + i * self._num_fp8_tensors_per_gemm["fwd"]
                ].force_pow_2_scales = recipe.fp8_quant_fwd_weight.power_2_scale
                self.quantizers["scaling_fwd"][
                    self._offsets["weight"] + i * self._num_fp8_tensors_per_gemm["fwd"]
                ].amax_epsilon = recipe.fp8_quant_fwd_weight.amax_epsilon
        else:
            for i in range(self.num_gemms):
                # set grad_output_quantizer with amax epsilon and power_2_scale
                self.quantizers["scaling_bwd"][
                    self._offsets["input"] + i * self._num_fp8_tensors_per_gemm["bwd"]
                ].force_pow_2_scales = recipe.fp8_quant_bwd_grad.power_2_scale
                self.quantizers["scaling_bwd"][
                    self._offsets["input"] + i * self._num_fp8_tensors_per_gemm["bwd"]
                ].amax_epsilon = recipe.fp8_quant_bwd_grad.amax_epsilon

    def _get_weight_tensors(self) -> List[Union[torch.Tensor, QuantizedTensorStorage]]:
        """Get the weight tensors of the module."""
        grouped_weight = getattr(self, "weight", None)
        if grouped_weight is not None:
            weight_tensors = [grouped_weight]
        else:
            weight_tensors = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
        if not self.fp8 and any(isinstance(w, QuantizedTensorStorage) for w in weight_tensors):
            warnings.warn(
                "You are using quantized weights without quantized compute. "
                "Please make sure this is intentional."
            )
            weight_tensors = [
                w.dequantize() if isinstance(w, QuantizedTensorStorage) else w
                for w in weight_tensors
            ]
        return weight_tensors

    def _get_bias_tensors(self) -> List[torch.Tensor]:
        """Get bias parameters in their registered grouped or per-GEMM layout.

        A single grouped bias remains one packed GroupedTensor. When return_bias=True,
        an upper-level framework such as MCore must apply that packed bias accordingly;
        Discrete bias parameters retain the existing list-of-per-GEMM contract.

        Example with 2 experts, 128 output features:

            single grouped bias:
              GroupedTensor shape = [2, 128] -> [grouped_bias]

            discrete biases:
              bias0 [128] + bias1 [128] -> [bias0, bias1]
        """
        grouped_bias = getattr(self, "bias", None)
        if grouped_bias is not None:
            return [grouped_bias]
        return [getattr(self, f"bias{i}") for i in range(self.num_gemms)]

    def _get_weight_quantizers(self) -> List[Quantizer]:
        """Get the weight quantizers of the module."""
        if not self.fp8 and not self.fp8_calibration and not self.primary_weights_in_fp8:
            return [None] * self.num_gemms
        weight_quantizers = [
            self.quantizers["scaling_fwd"][
                self._offsets["weight"] + i * self._num_fp8_tensors_per_gemm["fwd"]
            ]
            for i in range(self.num_gemms)
        ]
        for i in range(self.num_gemms):
            weight_quantizers[i].internal = not self.primary_weights_in_fp8
        return weight_quantizers

    def _get_quantizers(self):
        if self.fp8 and self._uses_custom_recipe:
            # Validation normally runs while installing metadata. Retry here because
            # a failed generation remains installed when the caller catches the error.
            recipe = FP8GlobalStateManager.get_fp8_recipe()
            self._validate_custom_recipe_quantizers(True, recipe)
            self._validate_custom_recipe_quantizers(False, recipe)

        weight_quantizers = self._get_weight_quantizers()
        input_quantizers, output_quantizers = (
            [None] * self.num_gemms,
            [None] * self.num_gemms,
        )
        grad_input_quantizers, grad_weight_quantizers, grad_output_quantizers = (
            [None] * self.num_gemms,
            [None] * self.num_gemms,
            [None] * self.num_gemms,
        )
        if self.fp8:
            input_quantizers = [
                self.quantizers["scaling_fwd"][
                    self._offsets["input"] + i * self._num_fp8_tensors_per_gemm["fwd"]
                ]
                for i in range(self.num_gemms)
            ]
            for i in range(self.num_gemms):
                input_quantizers[i].internal = True
                input_quantizers[i].optimize_for_gemm = True
            if torch.is_grad_enabled():
                grad_output_quantizers = [
                    self.quantizers["scaling_bwd"][
                        self._offsets["input"] + i * self._num_fp8_tensors_per_gemm["bwd"]
                    ]
                    for i in range(self.num_gemms)
                ]
                for i in range(self.num_gemms):
                    grad_output_quantizers[i].internal = True
                    grad_output_quantizers[i].optimize_for_gemm = True
        return (
            input_quantizers,
            weight_quantizers,
            output_quantizers,
            grad_input_quantizers,
            grad_weight_quantizers,
            grad_output_quantizers,
        )

    def _get_debug_quantizers(self):
        original_quantizers = self._get_quantizers()
        if not TEDebugState.debug_enabled:
            raise RuntimeError("TEDebugState.debug_enabled must be True to get debug quantizers")

        names = ["activation", "weight", "output", "dgrad", "wgrad", "gradient"]
        return tuple(
            [
                DebugQuantizer(self.name + f".gemm_{q_id}", name, q, self.tp_group, self.tp_size)
                for q_id, q in enumerate(qs)
            ]
            for name, qs in zip(names, original_quantizers)
        )

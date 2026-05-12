# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
from functools import reduce
from operator import mul as multiply_op
import warnings
import weakref

import torch

import transformer_engine_torch as tex

from transformer_engine.common.recipe import Recipe
from transformer_engine.pytorch.torch_version import torch_version

from .base import (
    fill_userbuffers_buffer_for_all_gather,
    get_dummy_wgrad,
    get_ub,
    quantize_weight,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ._common import noop_cat, WeightGradStore
from ..quantization import FP8GlobalStateManager, QuantizerRole
from ..utils import (
    cast_if_needed,
    clear_tensor_data,
    divide,
    init_method_constant,
    needs_quantized_gemm,
    assert_dim_for_fp8_exec,
    nvtx_range_pop,
    nvtx_range_push,
    get_nvtx_range_context,
)
from ..distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    allreduce,
    symmetric_all_reduce,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
    is_fp8_activation_recompute_enabled,
    in_fp8_activation_recompute_phase,
    _fsdp_scatter_tensors,
    _fsdp_gather_tensors,
)
from ..cpp_extensions import (
    general_gemm,
)
from ..constants import FP8BwdTensorIdx, FP8FwdTensorIdx, GemmParallelModes, dist_group_type
from ..jit import no_torch_dynamo
from ..graph import is_graph_capturing
from ..quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorStorage,
    Quantizer,
    prepare_for_saving,
    restore_from_func_ctx,
)
from ..tensor.float8_tensor import Float8CurrentScalingQuantizer, Float8Quantizer
from ..tensor.mxfp8_tensor import MXFP8Quantizer
from ..tensor.utils import clear_columnwise_cache, is_custom
from ..export import is_in_onnx_export_mode, assert_warmed_up
from ..cpu_offload import (
    is_cpu_offload_enabled,
    start_offload,
    mark_not_offload,
    mark_activation_offload,
)
from ...debug.pytorch.debug_state import TEDebugState

__all__ = ["Linear"]


TensorOrQuantized = Union[torch.Tensor, QuantizedTensorStorage]


@dataclass(slots=True)
class LinearFwdArgs:
    """Single-argument bag for the forward path of :class:`_Linear`."""

    # --- Differentiable tensors (also passed positionally to autograd) ---
    weight: TensorOrQuantized
    inp: torch.Tensor
    bias: Optional[torch.Tensor]

    # --- Non-differentiable cached tensors ---
    weight_workspace: Optional[torch.Tensor]

    # --- requires_grad flags (cached so backward does not re-query) ---
    input_requires_grad: bool
    weight_requires_grad: bool
    bias_requires_grad: bool

    # --- Quantizers ---
    input_quantizer: Optional[Quantizer]
    weight_quantizer: Optional[Quantizer]
    output_quantizer: Optional[Quantizer]
    grad_input_quantizer: Optional[Quantizer]
    grad_weight_quantizer: Optional[Quantizer]
    grad_output_quantizer: Optional[Quantizer]

    # --- Numerical / dtype config ---
    activation_dtype: torch.dtype
    fp8: bool
    fp8_calibration: bool
    fp8_output: bool
    save_original_input: bool
    backward_override: Optional[str]
    custom: bool
    debug: bool

    # --- Weight-workspace caching ---
    is_first_microbatch: Optional[bool]
    cache_weight: bool
    skip_fp8_weight_update: Optional[bool]

    # --- Tensor / sequence parallelism ---
    parallel_mode: Optional[str]
    tp_group: Optional[Any]
    tp_size: int
    tensor_parallel: bool
    sequence_parallel: bool
    symmetric_ar_type: Optional[str]
    backward_input_needs_gather: bool

    # --- Userbuffers (comm + GEMM overlap) ---
    ub_name: Optional[str]
    ub_overlap_ag_fprop: bool
    ub_overlap_rs_fprop: bool
    ub_overlap_ag_dgrad: bool
    ub_overlap_rs_dgrad: bool
    ub_bulk_dgrad: bool
    ub_bulk_wgrad: bool

    # --- FSDP ---
    fsdp_group: Optional[Any]
    is_fsdp2: bool

    # --- Weight-grad scheduling ---
    fuse_wgrad_accumulation: bool
    wgrad_store: Optional[Any]

    # --- Misc ---
    cpu_offloading: bool
    is_grad_enabled: bool


@dataclass(slots=True)
class LinearBwdArgs:
    """Single-argument bag for the backward path of :class:`_Linear`."""

    # --- Saved / restored tensors (populated at backward entry) ---
    grad_output: Optional[torch.Tensor] = None
    inputmat: Optional[TensorOrQuantized] = None
    weight_fp8: Optional[TensorOrQuantized] = None
    saved_weight: Optional[TensorOrQuantized] = None
    bias: Optional[torch.Tensor] = None

    # --- Quantizers ---
    input_quantizer: Optional[Quantizer] = None
    weight_quantizer: Optional[Quantizer] = None
    grad_input_quantizer: Optional[Quantizer] = None
    grad_weight_quantizer: Optional[Quantizer] = None
    grad_output_quantizer: Optional[Quantizer] = None

    # --- Differentiability summary ---
    use_bias: bool = False
    requires_dgrad: bool = False
    requires_wgrad: bool = False
    inp_shape: Optional[torch.Size] = None

    # --- Numerical / dtype config ---
    activation_dtype: Optional[torch.dtype] = None
    fp8: bool = False
    fp8_recipe: Optional[Recipe] = None
    backward_override: Optional[str] = None
    is_weight_param_quantized: bool = False
    custom: bool = False
    debug: bool = False

    # --- Tensor / sequence parallelism ---
    parallel_mode: Optional[str] = None
    tp_group: Optional[Any] = None
    tp_size: int = 1
    tensor_parallel: bool = False
    sequence_parallel: bool = False
    backward_input_needs_gather: bool = False

    # --- Userbuffers (comm + GEMM overlap) ---
    ub_name: Optional[str] = None
    ub_overlap_ag: bool = False
    ub_overlap_rs_dgrad: bool = False
    ub_bulk_dgrad: bool = False
    ub_bulk_wgrad: bool = False

    # --- FSDP ---
    fsdp_group: Optional[Any] = None
    fsdp_shapes: Any = None
    is_fsdp2: bool = False

    # --- Weight-grad scheduling / accumulation ---
    is_first_microbatch: Optional[bool] = None
    fuse_wgrad_accumulation: bool = False
    wgrad_store: Optional[Any] = None
    origin_weight_ref: Optional[Any] = None
    origin_weight_overwrites_main_grad: bool = False
    main_grad_func: Optional[Callable[[], torch.Tensor]] = None

    # --- FP8 reduce-and-update bookkeeping ---
    reduce_and_update_bwd_fp8_tensors: bool = False

    # --- Misc ---
    cpu_offloading: bool = False
    owns_input: bool = False

    # --- Per-backward scratch state (populated inside _linear_backward) ---
    ub_obj_gradout: Optional[Any] = None

    def setup_saved_tensors(self, ctx: torch.autograd.function.FunctionCtx) -> None:
        """Pull saved tensors from ``ctx`` into the fields backward consumes."""
        (
            self.inputmat,
            self.weight_fp8,
            self.saved_weight,
            self.bias,
        ) = restore_from_func_ctx(
            ctx
        )  # pylint: disable=unbalanced-tuple-unpacking


def _check_fp8_reduce_and_update():
    """Check if this is the first FP8 module (for backward reduce-and-update)."""
    qstate = FP8GlobalStateManager.quantization_state
    _first_fp8_module = qstate.is_first_fp8_module
    result = FP8GlobalStateManager.is_first_fp8_module()
    if in_fp8_activation_recompute_phase():
        qstate.is_first_fp8_module = _first_fp8_module
    return result


def _linear_forward_impl(
    args: LinearFwdArgs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple], None, Optional[Dict]]:
    """Forward implementation for the linear layer.

    Returns ``(out, new_weight_workspace, tensors_to_save_from_forward, None,
    ctx_attrs)``. ``new_weight_workspace`` is the freshly produced FP8 weight
    workspace (returned alongside ``out`` so the caller can refresh its
    cache). The last three are ``None`` when gradients are disabled.
    """

    weight = args.weight
    inp = args.inp
    bias = args.bias
    input_quantizer = args.input_quantizer
    weight_quantizer = args.weight_quantizer
    output_quantizer = args.output_quantizer
    is_first_microbatch = args.is_first_microbatch
    fp8 = args.fp8
    cpu_offloading = args.cpu_offloading
    tp_group = args.tp_group
    sequence_parallel = args.sequence_parallel
    activation_dtype = args.activation_dtype
    parallel_mode = args.parallel_mode
    is_grad_enabled = args.is_grad_enabled
    ub_overlap_rs_fprop = args.ub_overlap_rs_fprop
    ub_overlap_ag_fprop = args.ub_overlap_ag_fprop
    ub_name = args.ub_name
    fsdp_group = args.fsdp_group
    symmetric_ar_type = args.symmetric_ar_type
    save_original_input = args.save_original_input
    debug = args.debug
    backward_override = args.backward_override
    is_fsdp2 = args.is_fsdp2
    if backward_override == "high_precision":
        save_original_input = True

    # NVTX label for profiling
    nvtx_label = "transformer_engine._Linear.forward"
    if ub_name is not None:
        nvtx_label = f"{nvtx_label}.{ub_name}"

    # Make sure input dimensions are compatible
    out_features, in_features = weight.shape
    assert inp.shape[-1] == in_features, "GEMM not possible"

    # Configure tensor-parallel communication
    tp_world_size = get_distributed_world_size(tp_group)
    backward_needs_input = is_grad_enabled and weight.requires_grad
    with_input_all_gather_nccl = (
        parallel_mode == "column" and sequence_parallel and not ub_overlap_ag_fprop
    )

    # Configure Userbuffers communication (comm+GEMM overlap)
    ub_obj = None
    ub_type = None
    if ub_overlap_rs_fprop:
        ub_obj = get_ub(ub_name + "_fprop", fp8)
        ub_type = tex.CommOverlapType.RS
    elif ub_overlap_ag_fprop:
        ub_obj = get_ub(ub_name + "_fprop", fp8)
        ub_type = tex.CommOverlapType.AG

    # ------------------------------------------------------
    # Prepare input tensor
    # Note: Cast to expected dtype and perform tensor-parallel communication
    # ------------------------------------------------------
    nvtx_range_push(f"{nvtx_label}.input_cast_comm")
    inputmat = inp  # Input tensor to save for backward (maybe sharded)
    inputmat_total = None  # Input tensor to pass to GEMM (gathered)
    own_quantized_input = False
    if fp8:
        assert_dim_for_fp8_exec(inputmat, weight)
        if save_original_input:
            assert not isinstance(
                input_quantizer, Float8Quantizer
            ), "DelayedScaling recipe is not supported with save_original_input"

    if with_input_all_gather_nccl or ub_overlap_ag_fprop:  # All-gather input tensor

        # Cast local input tensor if needed
        if fp8 or debug:
            if input_quantizer is None:
                raise ValueError("Missing quantizer for input tensor")
            if not isinstance(inputmat, QuantizedTensorStorage) and not args.custom:
                own_quantized_input = True
                input_quantizer.set_usage(
                    rowwise=True,
                    columnwise=backward_needs_input and backward_override is None,
                )
                if isinstance(input_quantizer, (Float8Quantizer, Float8CurrentScalingQuantizer)):
                    # All-gather is not supported with FP8 column-wise data
                    input_quantizer.set_usage(columnwise=False)
                if save_original_input:
                    # No need for column-wise data since this
                    # tensor will not be cached for backward pass
                    input_quantizer.set_usage(columnwise=False)
                    own_quantized_input = False
                inputmat = input_quantizer(inputmat)
        else:
            inputmat = cast_if_needed(inp, activation_dtype)  # Cast for AMP

        # Initialize gathered input tensor
        quantizer = None
        if fp8 or debug:
            quantizer = input_quantizer
            quantizer.set_usage(rowwise=True, columnwise=False)
        if with_input_all_gather_nccl:  # Perform NCCL all-gather
            inputmat_total, _ = gather_along_first_dim(
                inputmat,
                tp_group,
                quantizer=quantizer,
            )
        elif ub_overlap_ag_fprop:  # Initialize Userbuffers all-gather
            inputmat_total, _ = fill_userbuffers_buffer_for_all_gather(
                ub_obj,
                inputmat,
                quantizer,
                tp_group,
            )

    else:  # Do not all-gather input tensor
        if fp8 or debug:
            if isinstance(inputmat, QuantizedTensorStorage):
                inputmat.update_usage(rowwise_usage=True)
            else:
                if input_quantizer is None:
                    raise ValueError("Missing quantizer for input tensor")
                input_quantizer.set_usage(
                    rowwise=True,
                    columnwise=(
                        backward_needs_input
                        and not save_original_input
                        and backward_override is None
                    ),
                )
                inputmat = input_quantizer(inputmat)
                own_quantized_input = True
        else:
            inputmat = cast_if_needed(inp, activation_dtype)  # Cast for AMP
        inputmat_total = inputmat

    if is_cpu_offload_enabled():
        start_offload(inputmat)
    nvtx_range_pop(f"{nvtx_label}.input_cast_comm")
    # ------------------------------------------------------
    # Input tensor is ready for GEMM...
    # ------------------------------------------------------

    # ------------------------------------------------------
    # Prepare weight tensor
    # ------------------------------------------------------
    new_weight_workspace = None
    weightmat = weight
    if fp8 or debug:
        # Configure quantizer
        # No need to set the quantizer states if weight is already quantized
        # for debug mode we create quantizer every iteration, thus we need to set the quantizer states
        if weight_quantizer is not None and (not isinstance(weight, QuantizedTensor) or debug):
            columnwise_usage = is_grad_enabled and inp.requires_grad and not is_fsdp2
            if backward_override is not None:
                columnwise_usage = False
            if not columnwise_usage:
                columnwise_usage = (
                    is_fp8_activation_recompute_enabled()
                    and not in_fp8_activation_recompute_phase()
                )
            weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
        elif isinstance(weight, QuantizedTensor):
            weight_quantizer = weight._quantizer
        # Get quantized weight
        update_ws = is_first_microbatch is None or is_first_microbatch
        weightmat, new_weight_workspace = quantize_weight(
            tensor=weight,
            quantizer=weight_quantizer,
            workspace=args.weight_workspace,
            update_workspace=update_ws,
            skip_update_flag=args.skip_fp8_weight_update,
            fsdp_group=fsdp_group,
            workspace_dtype=activation_dtype,
            cache=args.cache_weight,
        )
        weightmat.update_usage(rowwise_usage=True)

    else:
        weightmat = cast_if_needed(weightmat, activation_dtype)  # Cast for AMP
    # ------------------------------------------------------
    # Weight tensor is ready for GEMM...
    # ------------------------------------------------------

    # Cast bias to expected dtype
    bias_dtype = activation_dtype
    if needs_quantized_gemm(inputmat_total) and activation_dtype == torch.float32:
        # cuBLAS does not support FP8 GEMM with FP32 bias, so we cast to BF16
        bias_dtype = torch.bfloat16
    bias = cast_if_needed(bias, bias_dtype) if bias is not None else bias

    # Calibrate quantizers if needed
    if not fp8 and args.fp8_calibration:
        if input_quantizer is not None:
            input_quantizer.calibrate(inputmat_total)
        if weight_quantizer is not None:
            weight_quantizer.calibrate(weight)

    # Choose whether to use GEMM kernel with split accumulator
    use_split_accumulator = _2X_ACC_FPROP
    if fp8:
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        if hasattr(recipe, "fp8_gemm_fprop"):
            use_split_accumulator = recipe.fp8_gemm_fprop.use_split_accumulator

    # Configure output quantizer
    if output_quantizer is not None:
        output_quantizer.set_usage(rowwise=True, columnwise=False)

    # Output buffer for Userbuffers reduce-scatter
    reduce_scatter_out = None
    if ub_overlap_rs_fprop:
        out_shape = list(inp.shape)
        out_shape[0] //= tp_world_size
        out_shape[-1] = out_features
        reduce_scatter_out = torch.empty(out_shape, dtype=activation_dtype, device=inp.device)

    # ------------------------------------------------------
    # Forward GEMM
    # Note: y = x * w^T
    # ------------------------------------------------------
    nvtx_range_push(f"{nvtx_label}.gemm")
    gemm_out, *_, reduce_scatter_out = general_gemm(
        weightmat,
        inputmat_total,
        quantization_params=output_quantizer,
        out_dtype=activation_dtype,
        bias=bias,
        use_split_accumulator=use_split_accumulator,
        ub=ub_obj,
        ub_type=ub_type,
        extra_output=reduce_scatter_out,
    )
    nvtx_range_pop(f"{nvtx_label}.gemm")
    # ------------------------------------------------------
    # Finished forward GEMM...
    # ------------------------------------------------------

    # Deallocate GEMM input tensor if no longer needed
    # TODO(yuzhongw, tmoon): Figure out why inputmat_total is not automatically
    # deallocated by GC. Manually deallocating is a temporary hack.
    if with_input_all_gather_nccl:
        clear_tensor_data(inputmat_total)
        inputmat_total = None

    # ------------------------------------------------------
    # Prepare output tensor
    # Note: Perform tensor-parallel communication
    # ------------------------------------------------------
    out = None
    if ub_overlap_rs_fprop:
        out = reduce_scatter_out
    elif parallel_mode == "row" and args.tp_size > 1:
        nvtx_range_push(f"{nvtx_label}.row_parallel_comm")
        out = gemm_out
        if sequence_parallel:
            out, _ = reduce_scatter_along_first_dim(out, tp_group)
        elif args.tensor_parallel:
            if symmetric_ar_type is not None:
                out, _ = symmetric_all_reduce(out, tp_group, all_reduce_type=symmetric_ar_type)
            else:
                out, _ = allreduce(out, tp_group)
        nvtx_range_pop(f"{nvtx_label}.row_parallel_comm")
    else:
        out = gemm_out
    # ------------------------------------------------------
    # Output tensor is ready to return...
    # ------------------------------------------------------

    # Prepare backward state
    tensors_to_save_from_forward = None
    ctx_attrs = None

    if is_grad_enabled:
        if save_original_input:
            inputmat = inp

        # Discard unneeded data in input tensor
        if (
            backward_needs_input
            and own_quantized_input
            and isinstance(inputmat, QuantizedTensorStorage)
        ):
            if backward_override is not None:
                inputmat.update_usage(rowwise_usage=True, columnwise_usage=False)
            elif (
                args.backward_input_needs_gather
                and weight_quantizer.supports_only_rowwise_all_gather()
            ):
                # All-gather is not supported with FP8 column-wise data
                inputmat.update_usage(rowwise_usage=True, columnwise_usage=False)
            else:
                # Discard row-wise data since it is not needed in backward pass
                inputmat.update_usage(rowwise_usage=False, columnwise_usage=True)

        # Cached input tensor
        saved_inputmat = None
        if backward_needs_input:
            saved_inputmat = inputmat

        if cpu_offloading and saved_inputmat is not None:
            mark_activation_offload(saved_inputmat)

        # Scatter intermediate/activation tensors saved for the backward pass
        # NOTE: FSDP sharding is not valid for models initialized with primary Fp8 weights
        nvtx_range_push(f"{nvtx_label}.fsdp_scatter")
        fsdp_shapes = _fsdp_scatter_tensors(
            fsdp_group,
            saved_inputmat,
            weightmat if fp8 and not isinstance(weight, QuantizedTensorStorage) else None,
        )
        nvtx_range_pop(f"{nvtx_label}.fsdp_scatter")

        if cpu_offloading:
            mark_not_offload(weight, weightmat, bias)

        # TODO(ksivamani): Check memory usage
        # FSDP2: Don't save FP8 workspace for non-quantized weights.
        # Backward will re-quantize from the FSDP2 all-gathered weight.
        # (Issue #2681)
        wt_save = weightmat
        if is_fsdp2 and weightmat is not weight:
            wt_save = None

        # Dedup save slots that alias forward inputs; ``_linear_setup_ctx``
        # rebuilds the refs from ``inp`` / ``weight`` / ``bias``.
        # Needed for torch.compile to work correctly.
        saved_tensor_aliases = (
            "inp" if saved_inputmat is inp else None,
            "weight" if wt_save is weight else None,
            "weight",  # ``saved_weight`` slot is always the weight parameter
            "bias" if bias is not None else None,
        )
        tensors_to_save_from_forward = (
            None if saved_tensor_aliases[0] is not None else saved_inputmat,
            None if saved_tensor_aliases[1] is not None else wt_save,
            None,
            None if saved_tensor_aliases[3] is not None else bias,
        )

        ctx_attrs = {
            "fsdp_shapes": fsdp_shapes,
            "saved_tensor_aliases": saved_tensor_aliases,
        }

    return out, new_weight_workspace, tensors_to_save_from_forward, None, ctx_attrs


def _linear_setup_ctx(
    bwd_args: LinearBwdArgs,
    fwd_args: LinearFwdArgs,
    out: torch.Tensor,
    ctx_attrs: Dict,
    tensors_to_save_from_forward: Tuple[Any, ...],
) -> Tuple[Any, ...]:
    """Populate ``bwd_args`` from forward state.

    Returns the merged list of tensors that should be passed through
    ``prepare_for_saving`` by the caller (``_Linear.forward``). Keeping the
    ``prepare_for_saving`` call out of here lets callers stitch in extra
    tensors (e.g. the original ``weight`` parameter so backward can reuse it
    for FSDP2 re-quantization) without having to mutate the structured
    metadata returned by ``prepare_for_saving``.
    """
    del out  # No-op; kept for symmetry with the compile-time helper signature.

    inp = fwd_args.inp
    weight = fwd_args.weight
    bias = fwd_args.bias

    backward_override = fwd_args.backward_override
    fp8 = fwd_args.fp8
    fuse_wgrad_accumulation = fwd_args.fuse_wgrad_accumulation

    # Quantizers
    bwd_args.input_quantizer = fwd_args.input_quantizer
    bwd_args.weight_quantizer = (
        weight._quantizer if isinstance(weight, QuantizedTensor) else fwd_args.weight_quantizer
    )
    bwd_args.grad_input_quantizer = fwd_args.grad_input_quantizer
    bwd_args.grad_weight_quantizer = fwd_args.grad_weight_quantizer
    bwd_args.grad_output_quantizer = fwd_args.grad_output_quantizer

    # Differentiability summary
    bwd_args.use_bias = bias is not None
    bwd_args.requires_dgrad = fwd_args.input_requires_grad
    bwd_args.requires_wgrad = fwd_args.weight_requires_grad
    bwd_args.inp_shape = inp.shape

    # Numerical / dtype config
    bwd_args.activation_dtype = fwd_args.activation_dtype
    bwd_args.fp8 = fp8
    bwd_args.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
    bwd_args.backward_override = backward_override
    bwd_args.is_weight_param_quantized = isinstance(weight, QuantizedTensorStorage)
    bwd_args.custom = fwd_args.custom
    bwd_args.debug = fwd_args.debug

    # Tensor / sequence parallelism
    bwd_args.parallel_mode = fwd_args.parallel_mode
    bwd_args.tp_group = fwd_args.tp_group
    bwd_args.tp_size = fwd_args.tp_size
    bwd_args.tensor_parallel = fwd_args.tensor_parallel
    bwd_args.sequence_parallel = fwd_args.sequence_parallel
    bwd_args.backward_input_needs_gather = fwd_args.backward_input_needs_gather

    # Userbuffers
    bwd_args.ub_name = fwd_args.ub_name
    bwd_args.ub_overlap_ag = fwd_args.ub_overlap_ag_dgrad
    bwd_args.ub_overlap_rs_dgrad = fwd_args.ub_overlap_rs_dgrad
    bwd_args.ub_bulk_dgrad = fwd_args.ub_bulk_dgrad
    bwd_args.ub_bulk_wgrad = fwd_args.ub_bulk_wgrad

    # FSDP
    bwd_args.fsdp_group = fwd_args.fsdp_group
    bwd_args.fsdp_shapes = ctx_attrs["fsdp_shapes"]
    bwd_args.is_fsdp2 = fwd_args.is_fsdp2

    # Weight-grad scheduling / accumulation
    bwd_args.is_first_microbatch = fwd_args.is_first_microbatch
    bwd_args.fuse_wgrad_accumulation = fuse_wgrad_accumulation
    bwd_args.wgrad_store = fwd_args.wgrad_store
    if fuse_wgrad_accumulation and fwd_args.weight_requires_grad:
        bwd_args.origin_weight_ref = weakref.ref(weight)
        bwd_args.origin_weight_overwrites_main_grad = getattr(weight, "overwrite_main_grad", False)
        if hasattr(weight, "__fsdp_param__"):
            bwd_args.main_grad_func = weight.get_main_grad
        else:
            bwd_args.main_grad_func = lambda: weight.main_grad

    # Misc
    bwd_args.cpu_offloading = fwd_args.cpu_offloading

    if backward_override is not None:
        bwd_args.fp8 = False
        bwd_args.debug = False
        bwd_args.ub_overlap_ag = False
        bwd_args.ub_overlap_rs_dgrad = False
        bwd_args.ub_bulk_dgrad = False
        bwd_args.ub_bulk_wgrad = False
        bwd_args.grad_input_quantizer = None
        bwd_args.grad_weight_quantizer = None
        bwd_args.grad_output_quantizer = None

    saved_inputmat, wt_save, saved_weight, saved_bias = tensors_to_save_from_forward
    inputmat_alias, wt_save_alias, saved_weight_alias, bias_alias = ctx_attrs[
        "saved_tensor_aliases"
    ]
    bwd_args.owns_input = inputmat_alias != "inp"
    if inputmat_alias == "inp":
        saved_inputmat = inp
    if wt_save_alias == "weight":
        wt_save = weight
    if saved_weight_alias == "weight":
        saved_weight = weight
    if bias_alias == "bias":
        saved_bias = bias
    return (saved_inputmat, wt_save, saved_weight, saved_bias)


def _linear_backward(args: LinearBwdArgs) -> Tuple[Union[torch.Tensor, None], ...]:
    """Backward implementation for the linear layer.

    Caller must have populated ``args.grad_output`` and run
    ``args.setup_saved_tensors(ctx)`` before invocation.
    """
    bwd_args = args
    grad_output = args.grad_output
    assert grad_output is not None
    inputmat = args.inputmat
    weight_fp8 = args.weight_fp8
    saved_weight = args.saved_weight
    bias = args.bias
    input_quantizer = args.input_quantizer
    weight_quantizer = args.weight_quantizer
    grad_input_quantizer = args.grad_input_quantizer
    grad_weight_quantizer = args.grad_weight_quantizer
    grad_output_quantizer = args.grad_output_quantizer

    # NVTX label for profiling
    nvtx_label = "transformer_engine._Linear.backward"
    if bwd_args.ub_name is not None:
        nvtx_label = f"{nvtx_label}.{bwd_args.ub_name}"

    with get_nvtx_range_context("_Linear_backward"):
        origin_weight_python_object = None
        origin_weight_overwrites_main_grad = bwd_args.origin_weight_overwrites_main_grad
        main_grad = None
        if bwd_args.fuse_wgrad_accumulation and bwd_args.requires_wgrad:
            origin_weight_ref = bwd_args.origin_weight_ref
            bwd_args.origin_weight_ref = None
            origin_weight_python_object = (
                origin_weight_ref() if origin_weight_ref is not None else None
            )
            assert (
                origin_weight_python_object is not None
            ), "weight was removed while fuse_wgrad_accumulation=True"
            main_grad = bwd_args.main_grad_func()
            origin_weight_python_object.main_grad = main_grad

        # Gather intermediate/activation tensors if needed
        # NOTE: weight_fp8 = weight when bwd_args.fp8 == False and torch.disttributed.FSDP already
        #       shards/unshards the base weights so we don't do it ourselves
        nvtx_range_push(f"{nvtx_label}.fsdp_gather")
        _fsdp_gather_tensors(
            bwd_args.fsdp_group,
            bwd_args.fsdp_shapes,
            inputmat,
            weight_fp8,
        )
        nvtx_range_pop(f"{nvtx_label}.fsdp_gather")

        # Configure Userbuffers communication (comm+GEMM overlap)
        bwd_args.ub_obj_gradout = None
        ub_obj_dgrad = None
        ub_obj_wgrad = None
        ub_type_dgrad = None
        ub_type_wgrad = None
        dgrad_shape = [
            reduce(multiply_op, bwd_args.inp_shape[:-1]),
            bwd_args.inp_shape[-1],
        ]
        if bwd_args.ub_overlap_ag:
            # Overlap grad_output all-gather with dgrad compute
            bwd_args.ub_obj_gradout = get_ub(bwd_args.ub_name + "_dgrad", bwd_args.fp8)
            ub_obj_dgrad = bwd_args.ub_obj_gradout
            ub_type_dgrad = tex.CommOverlapType.AG
        elif bwd_args.ub_overlap_rs_dgrad:
            # Overlap dgrad reduce-scatter with dgrad compute
            bwd_args.ub_obj_gradout = get_ub(bwd_args.ub_name + "_dgrad", bwd_args.fp8)
            ub_obj_dgrad = bwd_args.ub_obj_gradout
            ub_type_dgrad = tex.CommOverlapType.RS
        else:
            if bwd_args.ub_bulk_dgrad:
                # Overlap inputmat all-gather with dgrad compute
                bwd_args.ub_obj_gradout = get_ub(bwd_args.ub_name + "_dgrad", bwd_args.fp8)
                ub_obj_dgrad = bwd_args.ub_obj_gradout
                ub_type_dgrad = tex.CommOverlapType.AG
            if bwd_args.ub_bulk_wgrad:
                # Overlap dgrad reduce-scatter with wgrad compute
                ub_obj_wgrad = get_ub(bwd_args.ub_name + "_wgrad", bwd_args.fp8)
                ub_type_wgrad = tex.CommOverlapType.RS

        # --------------------------------------------------
        # Prepare grad output tensor
        # Note: Cast to expected dtype and perform tensor-parallel communication
        # --------------------------------------------------

        # Unmodified grad output tensor
        grad_output_arg = grad_output

        # Configure quantizer for grad output tensor
        # Note: dgrad GEMM requires row-wise usage, wgrad GEMM
        # requires column-wise usage
        if grad_output_quantizer is not None:
            quantizer = grad_output_quantizer
            quantizer.set_usage(rowwise=True, columnwise=True)
            if bwd_args.ub_overlap_ag:
                # Userbuffers only supports communication for one
                # tensor usage at a time. Configure quantizer with
                # usage for only dgrad GEMM.
                quantizer.set_usage(columnwise=False)

        # Adjust the quantization direction approach depending
        # on whether wgrad calculations will be performed.
        # NOTE: If requires_dgrad is False, disabling `rowwise` quantization and keeping `columnwise` quantization
        #       results in `Assertion failed: output_tensor->has_data(). Quantizing in only the columnwise direction not supported yet!`
        # NOTE: For `bias is True`, selected quantize kernel errors with
        #       `cast_kernels.cuh:1322 in function fp8_quantize_arch_l_100: Not implemented scaling mode or fusion: NVTE_DELAYED_TENSOR_SCALING or IS_DBIAS=true on GPU with compute capability < 10.0.`
        if (
            not bwd_args.use_bias
            and not bwd_args.requires_wgrad
            and grad_output_quantizer is not None
        ):
            grad_output_quantizer.set_usage(columnwise=False)

        # Prepare grad output tensor.
        # ``grad_output_preprocess`` accesses a small set of attributes
        # (sequence_parallel, fp8, backward_override, debug, ub_overlap_ag,
        # tp_group, ub_obj_gradout, use_bias). ``LinearBwdArgs`` exposes the
        # same names so we can pass it directly.
        nvtx_range_push(f"{nvtx_label}.grad_output_preprocess")
        (
            grad_output,
            grad_bias,
        ) = TransformerEngineBaseModule.grad_output_preprocess(
            bwd_args,
            grad_output,
            bwd_args.parallel_mode == "row",
            grad_output_quantizer,
        )
        nvtx_range_pop(f"{nvtx_label}.grad_output_preprocess")

        # --------------------------------------------------
        # Grad output tensor is ready for computing grad input...
        # --------------------------------------------------

        # --------------------------------------------------
        # Prepare input tensor
        # Note: Input tensor is needed for wgrad GEMM.
        # Tensor-parallel communication is overlapped with dgrad
        # GEMM.
        # --------------------------------------------------
        inputmat_total = None
        inputmat_total_work = None
        if bwd_args.requires_wgrad:
            if bwd_args.fp8 or bwd_args.debug:
                if isinstance(inputmat, QuantizedTensorStorage):
                    # Input tensor is already quantized
                    pass
                elif bwd_args.debug or bwd_args.custom:
                    # Debug quantizer will be applied immediately before wgrad GEMM
                    pass
                else:
                    # Quantize input tensor
                    quantizer = input_quantizer
                    if quantizer.supports_only_rowwise_all_gather():
                        # All-gather is not supported with FP8 column-wise data
                        quantizer.set_usage(
                            rowwise=True,
                            columnwise=not bwd_args.backward_input_needs_gather,
                        )
                    else:
                        quantizer.set_usage(rowwise=False, columnwise=True)
                    inputmat = quantizer(inputmat)
            else:
                if isinstance(inputmat, QuantizedTensorStorage):
                    inputmat = inputmat.dequantize(dtype=bwd_args.activation_dtype)
                else:
                    inputmat = cast_if_needed(inputmat, bwd_args.activation_dtype)
        if bwd_args.backward_input_needs_gather:
            quantizer = None
            if bwd_args.fp8 or bwd_args.debug:
                quantizer = input_quantizer
                if quantizer.supports_only_rowwise_all_gather():
                    # If data is in FP8, we compute FP8 transposes manually
                    quantizer.set_usage(rowwise=True, columnwise=False)
                else:
                    # wgrad GEMM requires input with column-wise usage
                    quantizer.set_usage(rowwise=False, columnwise=True)
            if bwd_args.ub_bulk_dgrad:
                inputmat_total, _ = fill_userbuffers_buffer_for_all_gather(
                    ub_obj_dgrad,
                    inputmat,
                    quantizer,
                    bwd_args.tp_group,
                )
            else:
                nvtx_range_push(f"{nvtx_label}.column_parallel_comm_input")
                inputmat_total, inputmat_total_work = gather_along_first_dim(
                    inputmat,
                    bwd_args.tp_group,
                    async_op=True,
                    quantizer=quantizer,
                )
                nvtx_range_pop(f"{nvtx_label}.column_parallel_comm_input")
        else:
            inputmat_total = inputmat
        # --------------------------------------------------
        # Input tensor is ready for computing grad weight...
        # --------------------------------------------------

        # --------------------------------------------------
        # Compute grad input tensor
        # --------------------------------------------------

        dgrad = None
        dgrad_work = None
        if bwd_args.requires_dgrad:

            # FSDP2: Re-create workspace from all-gathered weight when
            # workspace was not saved. (Issue #2681)
            # Use saved_weight (the original weight parameter) since
            # weight_fp8 is only set when workspace was saved.
            if weight_fp8 is None:
                if isinstance(saved_weight, QuantizedTensorStorage):
                    # saved weight is already set to right usages by
                    # fsdp2 quantized-tensor hooks when workspace was not saved.
                    weight_fp8 = saved_weight
                elif bwd_args.weight_quantizer is not None:
                    bwd_args.weight_quantizer.set_usage(rowwise=True, columnwise=True)
                    weight_fp8 = bwd_args.weight_quantizer(saved_weight)

            # Make sure required data is available
            if isinstance(grad_output, QuantizedTensorStorage):
                grad_output.update_usage(rowwise_usage=True)
            if (
                bwd_args.fp8
                and weight_quantizer is not None
                and isinstance(weight_fp8, QuantizedTensorStorage)
            ):
                weight_fp8.update_usage(columnwise_usage=True)

            # Choose whether to use GEMM kernel with split accumulator
            use_split_accumulator = _2X_ACC_DGRAD
            if bwd_args.fp8:
                recipe = bwd_args.fp8_recipe
                if hasattr(recipe, "fp8_gemm_dgrad"):
                    use_split_accumulator = recipe.fp8_gemm_dgrad.use_split_accumulator

            # Update grad input quantizer
            if grad_input_quantizer is not None:
                grad_input_quantizer.set_usage(rowwise=True, columnwise=False)

            # Output buffers for Userbuffers reduce-scatter
            gemm_out = None
            reduce_scatter_out = None
            if bwd_args.ub_overlap_rs_dgrad:
                reduce_scatter_out = torch.empty(
                    dgrad_shape,
                    dtype=bwd_args.activation_dtype,
                    device=grad_output_arg.device,
                )
            elif bwd_args.ub_bulk_wgrad:
                gemm_out = ub_obj_wgrad.get_buffer(local_chunk=False)

            # dgrad GEMM
            # Note: dx = dy * w

            nvtx_range_push(f"{nvtx_label}.dgrad_gemm")
            weight_for_dgrad = weight_fp8
            if bwd_args.backward_override == "dequantized":
                if isinstance(weight_for_dgrad, QuantizedTensorStorage):
                    weight_for_dgrad = weight_for_dgrad.dequantize(dtype=bwd_args.activation_dtype)
                else:
                    weight_for_dgrad = cast_if_needed(weight_for_dgrad, bwd_args.activation_dtype)
            elif bwd_args.backward_override == "high_precision":
                weight_for_dgrad = saved_weight
                if isinstance(weight_for_dgrad, QuantizedTensorStorage):
                    weight_for_dgrad = weight_for_dgrad.dequantize(dtype=bwd_args.activation_dtype)
            gemm_out, *_, reduce_scatter_out = general_gemm(
                weight_for_dgrad,
                grad_output,
                layout="NN",
                grad=True,
                quantization_params=grad_input_quantizer,
                out=gemm_out,
                out_dtype=bwd_args.activation_dtype,
                use_split_accumulator=use_split_accumulator,
                ub=ub_obj_dgrad,
                ub_type=ub_type_dgrad,
                extra_output=reduce_scatter_out,
                bulk_overlap=bwd_args.ub_bulk_dgrad,
            )
            nvtx_range_pop(f"{nvtx_label}.dgrad_gemm")

            # FSDP2 only handles deallocation all-gathered weights that it allocates.
            # Columnwise data is derived from rowwise data after allgather for fp8
            # and 2d block-scaled weights in TE managed memory. So we need to clear
            # it here.
            # (Issues #2681, #2717)
            if bwd_args.is_fsdp2 and isinstance(weight_fp8, QuantizedTensorStorage):
                clear_columnwise_cache(weight_fp8)

            # Prepare grad input tensor
            # Note: Perform tensor-parallel communication
            if bwd_args.ub_overlap_rs_dgrad:
                dgrad = reduce_scatter_out
            elif bwd_args.ub_bulk_wgrad:
                dgrad = ub_obj_wgrad.get_buffer(local_chunk=True)
            elif bwd_args.parallel_mode == "column" and bwd_args.tp_size > 1:
                nvtx_range_push(f"{nvtx_label}.column_parallel_comm_dgrad")
                dgrad = gemm_out
                if bwd_args.sequence_parallel:
                    dgrad, dgrad_work = reduce_scatter_along_first_dim(
                        dgrad,
                        bwd_args.tp_group,
                        async_op=True,
                    )
                else:
                    dgrad, dgrad_work = allreduce(dgrad, bwd_args.tp_group, async_op=True)
                nvtx_range_pop(f"{nvtx_label}.column_parallel_comm_dgrad")
            else:
                dgrad = gemm_out

        # --------------------------------------------------
        # Grad input tensor has been computed...
        # --------------------------------------------------

        # --------------------------------------------------
        # Compute grad weight
        # --------------------------------------------------

        wgrad = None
        if bwd_args.requires_wgrad:

            # Prepare input tensor
            # Note: Synchronize tensor-parallel communication and
            # make sure required data is available
            if inputmat_total_work is not None:
                inputmat_total_work.wait()
                inputmat_total_work = None
            if bwd_args.fp8 or bwd_args.debug:
                if isinstance(inputmat_total, QuantizedTensorStorage):
                    inputmat_total.update_usage(columnwise_usage=True)
                else:
                    input_quantizer.set_usage(rowwise=False, columnwise=True)
                    inputmat_total = input_quantizer(inputmat_total)

            # Prepare grad output tensor
            # Note: Synchronize tensor-parallel communication and
            # make sure required data is available
            if bwd_args.ub_overlap_ag and isinstance(grad_output_quantizer, MXFP8Quantizer):
                # UB does not support pipelined overlapping grad output
                # all-gather with wgrad GEMM. Also, we can't
                # convert row-scaled MXFP8 to column-scaled, so we
                # can't reuse the grad output that was gathered
                # for the dgrad GEMM. We work around by explicitly
                # overlapping the AG operation with the dgrad GEMM.

                # Get the communication stream from the dgrad GEMM to use for the AG
                dgrad_send_stream, dgrad_recv_stream = ub_obj_dgrad.get_communication_stream()

                # This object is separate from the ub_obj_wgrad object which is passed to the GEMM
                ub_obj_overlap_wgrad = get_ub(bwd_args.ub_name + "_wgrad", bwd_args.fp8)

                grad_output_quantizer.set_usage(rowwise=False, columnwise=True)

                # We use the send stream to copy into the userbuffers.
                # This is the same stream that we will use to access the data in the AG,
                # so we dont need to add any syncs yet.
                with torch.cuda.stream(dgrad_send_stream):
                    grad_output, _ = fill_userbuffers_buffer_for_all_gather(
                        ub_obj_overlap_wgrad,
                        grad_output_arg,
                        grad_output_quantizer,
                        bwd_args.tp_group,
                    )

                # Allgather grad_outputs[0] using the dgrad streams so we can overlap with the fc2_dgrad gemm
                tex.bulk_overlap_ag_with_external_gemm(
                    ub_obj_overlap_wgrad, dgrad_send_stream, dgrad_recv_stream
                )

            if bwd_args.fp8 or bwd_args.debug:
                if isinstance(grad_output, QuantizedTensorStorage):
                    grad_output.update_usage(columnwise_usage=True)
                else:
                    grad_output_quantizer.set_usage(rowwise=False, columnwise=True)
                    grad_output = grad_output_quantizer(grad_output)

            # Figure out whether to use split accumulator
            use_split_accumulator = _2X_ACC_WGRAD
            if bwd_args.fp8:
                recipe = bwd_args.fp8_recipe
                if hasattr(recipe, "fp8_gemm_wgrad"):
                    use_split_accumulator = recipe.fp8_gemm_wgrad.use_split_accumulator

            # Figure out whether to output wgrad GEMM directly into main grad
            if bwd_args.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    bwd_args.fuse_wgrad_accumulation and not bwd_args.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = bwd_args.fuse_wgrad_accumulation

            # Output buffer for overlapping FP8 grad input
            # reduce-scatter with wgrad GEMM
            reduce_scatter_out = None
            if bwd_args.ub_bulk_wgrad and ub_obj_wgrad.is_fp8_ubuf():
                reduce_scatter_out = torch.empty(
                    dgrad_shape,
                    dtype=bwd_args.activation_dtype,
                    device=grad_output_arg.device,
                )

            # Arguments to include in wgrad GEMM closure
            wgrad_gemm_kwargs = {
                "out_dtype": (
                    main_grad.dtype
                    if bwd_args.fuse_wgrad_accumulation
                    else bwd_args.activation_dtype
                ),
                "quantization_params": grad_weight_quantizer,
                "accumulate": (
                    accumulate_wgrad_into_param_main_grad
                    if not origin_weight_overwrites_main_grad
                    else False
                ),
                "layout": "NT",
                "out": main_grad if bwd_args.fuse_wgrad_accumulation else None,
                "bias": (bias if (grad_bias is None and not bwd_args.fp8) else None),
                "use_split_accumulator": use_split_accumulator,
                "grad": True,
                "ub": ub_obj_wgrad,
                "ub_type": ub_type_wgrad,
                "extra_output": reduce_scatter_out,
                "bulk_overlap": bwd_args.ub_bulk_wgrad,
            }

            def wgrad_gemm(
                x: torch.Tensor,
                dy: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                """Perform wgrad GEMM: dw = dy^T * x

                May be fused with bgrad computation.

                May be called outside of this function to enable
                some advanced communication/compute overlapping.

                """
                nvtx_range_push(f"{nvtx_label}.wgrad_gemm")
                dw, db, *_ = general_gemm(x, dy, **wgrad_gemm_kwargs)
                nvtx_range_pop(f"{nvtx_label}.wgrad_gemm")
                return dw, db

            # Choose whether to call wgrad GEMM now or delay
            if bwd_args.wgrad_store is not None and bwd_args.wgrad_store.delay_wgrad_compute():
                if (
                    wgrad_gemm_kwargs["ub"] is not None
                    or wgrad_gemm_kwargs["ub_type"] is not None
                    or wgrad_gemm_kwargs["extra_output"] is not None
                    or wgrad_gemm_kwargs["bulk_overlap"]
                ):
                    raise NotImplementedError(
                        "Delayed weight grad computation is not supported "
                        "with Userbuffers (tensor-parallel communication overlapping)"
                    )
                bwd_args.wgrad_store.put([inputmat_total, grad_output], wgrad_gemm)
            else:

                # Call wgrad GEMM now
                wgrad, grad_bias_ = wgrad_gemm(inputmat_total, grad_output)

                # Update grad bias if needed
                if grad_bias is None:
                    grad_bias = grad_bias_
                del grad_bias_

                # Deallocate tensors if permitted
                if bwd_args.owns_input:
                    # Input tensor is internal
                    clear_tensor_data(inputmat_total)
                elif bwd_args.backward_input_needs_gather:
                    # Gathered input tensor is internal
                    clear_tensor_data(inputmat_total)
                if bwd_args.parallel_mode == "row" and bwd_args.sequence_parallel:
                    # Gathered grad output tensor is internal
                    clear_tensor_data(grad_output)

            # Update grad input if overlapping reduce-scatter with wgrad GEMM
            if bwd_args.ub_bulk_wgrad:
                if ub_obj_wgrad.is_fp8_ubuf():
                    dgrad = reduce_scatter_out
                else:
                    dgrad = ub_obj_wgrad.get_buffer(local_chunk=True).clone()

        # --------------------------------------------------
        # Grad weight has been computed...
        # --------------------------------------------------

        # Don't return grad bias if not needed
        if not bwd_args.use_bias:
            grad_bias = None

        # Make sure all tensor-parallel communication is finished
        if inputmat_total_work is not None:
            inputmat_total_work.wait()
            inputmat_total_work = None
        if dgrad_work is not None:
            dgrad_work.wait()
            dgrad_work = None

    if bwd_args.requires_wgrad:
        # Handle custom DDP from mcore.
        if bwd_args.fuse_wgrad_accumulation and hasattr(
            origin_weight_python_object, "grad_added_to_main_grad"
        ):
            origin_weight_python_object.grad_added_to_main_grad = True
            if getattr(origin_weight_python_object, "zero_out_wgrad", False):
                wgrad = get_dummy_wgrad(
                    list(main_grad.shape),
                    origin_weight_python_object.dtype,
                    zero=True,
                )
            else:
                wgrad = get_dummy_wgrad(
                    list(main_grad.shape),
                    origin_weight_python_object.dtype,
                )
        elif bwd_args.fuse_wgrad_accumulation:
            wgrad = None
    else:
        wgrad = None

    # Scatter fp8 weight buffers
    if bwd_args.fp8 and not bwd_args.is_weight_param_quantized:
        _fsdp_scatter_tensors(bwd_args.fsdp_group, weight_fp8)
    return (
        wgrad,
        dgrad.view(bwd_args.inp_shape) if bwd_args.requires_dgrad else None,
        grad_bias,
    )


class _Linear(torch.autograd.Function):
    """Linear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        inp: torch.Tensor,
        bias: Optional[torch.Tensor],
        fwd_args: LinearFwdArgs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass: compute linear output and set up autograd context.

        ``weight``, ``inp`` and ``bias`` are positional Tensor arguments so
        autograd tracks them; they are immediately re-attached to ``fwd_args``
        so every downstream helper can be invoked with a single argument.

        ``weight_workspace`` is intentionally NOT a positional input: it is a
        non-differentiable cached tensor passed in via
        ``fwd_args.weight_workspace`` and the freshly produced workspace is
        returned as a separate output so the module can refresh its cache.
        """
        fwd_args.weight = weight
        fwd_args.inp = inp
        fwd_args.bias = bias
        (
            out,
            new_weight_workspace,
            tensors_to_save_from_forward,
            _,
            ctx_attrs,
        ) = _linear_forward_impl(fwd_args)
        if ctx is not None:
            bwd_args = LinearBwdArgs()
            tensors_to_save_from_setup = _linear_setup_ctx(
                bwd_args,
                fwd_args,
                out,
                ctx_attrs,
                tensors_to_save_from_forward,
            )
            tensors_to_save, tensor_objects = prepare_for_saving(*tensors_to_save_from_setup)
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects
            ctx.backward_objects = bwd_args
            if fwd_args.fp8 and (
                fwd_args.input_requires_grad
                or fwd_args.weight_requires_grad
                or fwd_args.bias_requires_grad
            ):
                bwd_args.reduce_and_update_bwd_fp8_tensors = _check_fp8_reduce_and_update()
            if fwd_args.backward_override is not None:
                bwd_args.reduce_and_update_bwd_fp8_tensors = False

        return out, new_weight_workspace

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
        _grad_weight_workspace,
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """Backward pass: compute gradients and reduce FP8 scaling factors."""
        bwd_args: LinearBwdArgs = ctx.backward_objects
        bwd_args.grad_output = grad_output
        bwd_args.setup_saved_tensors(ctx)
        nvtx_label = "transformer_engine._Linear.backward"
        if bwd_args.ub_name is not None:
            nvtx_label = f"{nvtx_label}.{bwd_args.ub_name}"
        result = _linear_backward(bwd_args) + (None,)  # fwd_args grad slot
        reduce_and_update_bwd_fp8_tensors = bwd_args.reduce_and_update_bwd_fp8_tensors
        # Drop all references held by bwd_args (saved tensors, quantizers, weakrefs,
        # main_grad closure) so they don't outlive backward via ctx under retain_graph.
        ctx.backward_objects = None
        del bwd_args
        if reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            nvtx_range_push(f"{nvtx_label}.reduce_and_update_fp8_tensors")
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
            nvtx_range_pop(f"{nvtx_label}.reduce_and_update_fp8_tensors")
        return result


class Linear(TransformerEngineBaseModule):
    """Applies a linear transformation to the incoming data :math:`y = xA^T + b`

    On NVIDIA GPUs it is a drop-in replacement for ``torch.nn.Linear``.

    Parameters
    ----------
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
    parameters_split : Optional[Union[Tuple[str, ...], Dict[str, int]]], default = None
                      Configuration for splitting the weight and bias tensors along dim 0 into
                      multiple PyTorch parameters. If a list or tuple of strings is provided,
                      they are used to make the names of equally-sized parameters. If a dict
                      (preferably an OrderedDict) is provided, the keys are used as names and
                      values as split sizes along dim 0. The resulting parameters will have
                      names that end in ``_weight`` or ``_bias``, so trailing underscores are
                      stripped from any provided names.
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will be allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.
    name : str, default = None
        name of the module, currently used for debugging purposes.

    Parallelism parameters
    ----------------------
    sequence_parallel : bool, default = False
                       if set to ``True``, uses sequence parallelism.
    tp_group : ProcessGroup, default = None
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             ``set_tensor_parallel_group(tp_group)`` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.
    parallel_mode : {None, 'column', 'row'}, default = None
                   used to decide whether this Linear layer is Column Parallel Linear or Row
                   Parallel Linear as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
                   When set to ``None``, no communication is performed.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
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
                 the bias addition can be fused to subsequent operations.
    params_dtype : torch.dtype, default = torch.get_default_dtype()
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    delay_wgrad_compute : bool, default = False
                         Whether or not to delay weight gradient computation. If set to ``True``,
                         it's the user's responsibility to call ``module.backward_dw`` to compute
                         weight gradients.
    symmetric_ar_type : {None, 'multimem_all_reduce', 'two_shot', 'one_shot'}, default = None
                   Type of symmetric memory all-reduce to use during the forward pass.
                   This can help in latency bound communication situations.
                   Requires PyTorch version 2.7.0 or higher. When set to ``None``, standard all-reduce
                   is used.
    save_original_input : bool, default = False
                       If set to ``True``, always saves the original input tensor rather than the
                       cast tensor. In some scenarios, the input tensor is used by multiple modules,
                       and saving the original input tensor may reduce the memory usage.
                       Cannot work with FP8 DelayedScaling recipe.
    """

    def __init__(
        self,
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
        parameters_split: Optional[Union[Tuple[str, ...], Dict[str, int]]] = None,
        device: Union[torch.device, str] = "cuda",
        ub_overlap_ag: bool = False,
        ub_overlap_rs: bool = False,
        ub_overlap_rs_dgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_bulk_wgrad: bool = False,
        ub_name: Optional[str] = None,
        delay_wgrad_compute: bool = False,
        symmetric_ar_type: Optional[str] = None,
        save_original_input: bool = False,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name
        self.symmetric_ar_type = symmetric_ar_type
        self.save_original_input = save_original_input

        self.wgrad_store = WeightGradStore(delay_wgrad_compute, ub_bulk_wgrad)

        if device == "meta":
            assert parameters_split is None, "Cannot split module parameters on 'meta' device."
        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        self.parallel_mode = parallel_mode
        assert (
            self.parallel_mode in GemmParallelModes
        ), f"parallel_mode {parallel_mode} not supported"

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        # Column parallel TP overlap options
        self.ub_overlap_ag_fprop = (
            self.parallel_mode == "column" and self.sequence_parallel and ub_overlap_ag
        )
        self.ub_overlap_rs_dgrad = (
            self.parallel_mode == "column" and self.sequence_parallel and ub_overlap_rs_dgrad
        )
        self.ub_bulk_dgrad = (
            self.parallel_mode == "column"
            and self.sequence_parallel
            and ub_bulk_dgrad
            and not self.ub_overlap_rs_dgrad
        )
        self.ub_bulk_wgrad = (
            self.parallel_mode == "column"
            and self.sequence_parallel
            and ub_bulk_wgrad
            and not self.ub_overlap_rs_dgrad
        )

        # Row parallel TP overlap options
        self.ub_overlap_rs_fprop = (
            self.parallel_mode == "row" and self.sequence_parallel and ub_overlap_rs
        )
        self.ub_overlap_ag_dgrad = (
            self.parallel_mode == "row" and self.sequence_parallel and ub_overlap_ag
        )

        if any(
            [
                self.ub_overlap_rs_fprop,
                self.ub_overlap_ag_dgrad,
                self.ub_overlap_ag_fprop,
                self.ub_overlap_rs_dgrad,
                self.ub_bulk_dgrad,
                self.ub_bulk_wgrad,
            ]
        ):
            assert ub_name is not None, f"Comm+GEMM overlap layer '{ub_name}' is not initialized."
        self.ub_name = ub_name

        if self.symmetric_ar_type is not None:
            assert torch_version() >= (
                2,
                7,
                0,
            ), "Torch version must be at least 2.7 to use symmetric memory"

        # Initialize params in FP8
        with_fp8_params = FP8GlobalStateManager.with_fp8_parameters()

        # Contiguous buffers for params
        weight_tensor = torch.empty(
            self.out_features,
            self.in_features,
            device=device,
            dtype=params_dtype,
        )
        bias_tensor = None
        if self.use_bias:
            bias_tensor = torch.empty(
                self.out_features,
                device=device,
                dtype=params_dtype,
            )

        # Configure parameter splits
        self.weight_names = []
        self.bias_names = []
        self.parameter_split_sizes = []
        if parameters_split is None:
            # Split into a single parameter by default
            self.weight_names = ["weight"]
            self.bias_names = ["bias"]
            self.parameter_split_sizes = [out_features]
        elif not parameters_split:
            raise ValueError("Cannot split weight buffer into 0 parameters")
        elif isinstance(parameters_split, dict):
            # Split parameters with provided sizes
            for name, split_size in parameters_split.items():
                self.weight_names.append(f"{name.rstrip('_')}_weight")
                self.bias_names.append(f"{name.rstrip('_')}_bias")
                self.parameter_split_sizes.append(split_size)
        elif all(isinstance(name, str) for name in parameters_split):
            # Split parameters evenly
            split_size = out_features // len(parameters_split)
            for name in parameters_split:
                self.weight_names.append(f"{name.rstrip('_')}_weight")
                self.bias_names.append(f"{name.rstrip('_')}_bias")
                self.parameter_split_sizes.append(split_size)
        else:
            raise TypeError("Invalid configuration for parameters split")

        # Make sure parameter splits are valid
        if sum(self.parameter_split_sizes) != out_features:
            raise ValueError(
                f"Trying to split weight buffer ({out_features=}) "
                f"with split sizes {self.parameter_split_sizes}"
            )

        # Adjust parameter splits for tensor-parallel distribution
        if self.parallel_mode == "column":
            for i, size in enumerate(self.parameter_split_sizes):
                if size % self.tp_size != 0:
                    raise RuntimeError(
                        f"Attempting to distribute a parameter with out_features={size} "
                        f"between {self.tp_size} tensor-parallel processes"
                    )
                self.parameter_split_sizes[i] = size // self.tp_size

        # Construct weight parameters
        # Note: Register weights together so that they are adjacent to
        # each other in Linear.parameters(). This makes it more likely
        # that they will stay contiguous if the weights are
        # manipulated externally, e.g. by FSDP.
        offset = 0
        for i, split_size in enumerate(self.parameter_split_sizes):
            split_start = offset
            offset += split_size
            split_end = offset

            # Check if parameters are subviews of buffers
            is_subview = (split_start, split_end) != (0, self.out_features)
            if is_subview and with_fp8_params:
                raise RuntimeError(
                    "Splitting QuantizedTensor into multiple params is not supported"
                )

            # Construct weight parameter
            self.register_parameter(
                self.weight_names[i],
                torch.nn.Parameter(weight_tensor[split_start:split_end]),
                init_fn=init_method,
                get_rng_state_tracker=get_rng_state_tracker,
                fp8_meta_index=FP8FwdTensorIdx.GEMM1_WEIGHT,
            )

        # Construct bias parameters if needed
        if self.use_bias:
            offset = 0
            for i, split_size in enumerate(self.parameter_split_sizes):
                split_start = offset
                offset += split_size
                split_end = offset
                self.register_parameter(
                    self.bias_names[i],
                    torch.nn.Parameter(bias_tensor[split_start:split_end]),
                    init_fn=init_method_constant(0.0),
                )
        else:
            for name in self.bias_names:
                bias = torch.Tensor().to(dtype=params_dtype, device=device)
                setattr(self, name, bias)

        if with_fp8_params:
            self.init_fp8_metadata()

        self.reset_parameters(defer_init=device == "meta")

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.parallel_mode == "row" and self.apply_bias:
            self.gemm_bias_unfused_add = True
        else:
            self.gemm_bias_unfused_add = False

        if self.wgrad_store.delay_wgrad_compute():
            for name, param in self.named_parameters():
                if name in self.weight_names or name in self.bias_names:
                    param.skip_backward_post_hook = True

    def get_quantizer_roles(
        self,
        *,
        fwd: bool,
        num_quantizers: int,
    ) -> Optional[List[QuantizerRole]]:
        """QuantizerRole list for quantizers used by ``Linear``.

        The output (fwd) and grad-input (bwd) slots default to ``None``
        (unknown consumer).  Set :attr:`output_quantizer_role` /
        :attr:`grad_input_quantizer_role` to provide consumer identity.
        """
        name = self.name or ""
        if fwd:
            base = [
                QuantizerRole(module_type="linear", tensor_type="input", name=name),
                QuantizerRole(module_type="linear", tensor_type="weight", name=name),
                self._output_quantizer_role,
            ]
        else:
            base = [
                QuantizerRole(module_type="linear", tensor_type="grad_output", name=name),
                self._grad_input_quantizer_role,
            ]
        return [base[i % len(base)] for i in range(num_quantizers)]

    def set_meta_tensor(self, fwd: bool, recipe: Recipe) -> None:
        """Init scales and amaxes for fwd | bwd."""
        super().set_meta_tensor(fwd, recipe)

        # Recipe-specific quantizer configuration
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        if recipe.float8_current_scaling():
            self._customize_quantizers_float8_current_scaling(fwd, recipe)
        elif recipe.nvfp4():
            self._customize_quantizers_nvfp4(fwd, recipe)

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
            # Set parallelism attributes for linear weights
            for weight in self.weight_names:
                set_tensor_model_parallel_attributes(
                    tensor=getattr(self, weight),
                    is_parallel=True,
                    dim=1 if self.parallel_mode == "row" else 0,
                    stride=1,
                )

            # Set parallelism attributes for linear biases
            if self.use_bias:
                for bias in self.bias_names:
                    if self.parallel_mode == "row":
                        setattr(getattr(self, bias), "sequence_parallel", self.sequence_parallel)
                    elif self.parallel_mode == "column":
                        set_tensor_model_parallel_attributes(getattr(self, bias), True, 0, 1)

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
        fp8_output: Optional[bool] = False,
        fp8_grad: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply the linear transformation to the input.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
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
        """
        is_grad_enabled = torch.is_grad_enabled()

        if is_in_onnx_export_mode():
            return self.onnx_forward(inp, fp8_output, is_grad_enabled)

        debug = self.is_debug_iter()

        if FP8GlobalStateManager.fp8_graph_capturing():
            skip_fp8_weight_update = (
                FP8GlobalStateManager.quantization_state.skip_fp8_weight_update_tensor
            )
        else:
            skip_fp8_weight_update = None
        if skip_fp8_weight_update is not None:
            is_first_microbatch = False

        if self.ub_overlap_rs_fprop:
            if get_ub(
                self.ub_name + "_fprop", FP8GlobalStateManager.is_fp8_enabled()
            ).is_fp8_ubuf():
                fp8_output = True
        if self.ub_overlap_rs_dgrad:
            if get_ub(
                self.ub_name + "_dgrad", FP8GlobalStateManager.is_fp8_enabled()
            ).is_fp8_ubuf():
                fp8_grad = True

        inp = self.prepare_forward(inp, allow_non_contiguous=isinstance(inp, QuantizedTensor))
        try:
            weight_tensor, bias_tensor = self._get_weight_and_bias_tensors()

            quantizers = (
                self._get_quantizers(fp8_output, fp8_grad, is_grad_enabled)
                if not debug
                else self._get_debug_quantizers(fp8_output, fp8_grad, is_grad_enabled)
            )
            if debug:
                if self.no_debug_features_active(quantizers):
                    debug = False
                    quantizers = self._get_quantizers(fp8_output, fp8_grad, is_grad_enabled)

            (
                input_quantizer,
                weight_quantizer,
                output_quantizer,
                grad_input_quantizer,
                grad_weight_quantizer,
                grad_output_quantizer,
            ) = quantizers

            if is_grad_enabled:
                linear_fn = _Linear.apply
                autograd_ctx = []
            else:
                linear_fn = _Linear.forward
                autograd_ctx = [None]

            cache_name = None if (is_first_microbatch is None or self.is_fsdp2) else "weight"
            weight_workspace = (
                self._fp8_workspaces.get(cache_name) if cache_name is not None else None
            )

            if self.fp8:
                backward_override = FP8GlobalStateManager.get_fp8_recipe().backward_override
            else:
                backward_override = None
            custom = is_custom(input_quantizer) or is_custom(weight_quantizer)
            backward_input_needs_gather = (
                weight_tensor.requires_grad
                and self.parallel_mode == "column"
                and self.sequence_parallel
            )

            if debug:
                ub_overlap_rs_fprop = False
                ub_overlap_ag_dgrad = False
                ub_overlap_ag_fprop = False
                ub_overlap_rs_dgrad = False
                ub_bulk_dgrad = False
                ub_bulk_wgrad = False
            else:
                ub_overlap_rs_fprop = self.ub_overlap_rs_fprop
                ub_overlap_ag_dgrad = self.ub_overlap_ag_dgrad
                ub_overlap_ag_fprop = self.ub_overlap_ag_fprop
                ub_overlap_rs_dgrad = self.ub_overlap_rs_dgrad
                ub_bulk_dgrad = self.ub_bulk_dgrad
                ub_bulk_wgrad = self.ub_bulk_wgrad

            linear_bias_tensor = (
                bias_tensor if (self.apply_bias and not self.gemm_bias_unfused_add) else None
            )
            wgrad_store = self.wgrad_store if self.wgrad_store.delay_wgrad_compute() else None
            fwd_args = LinearFwdArgs(
                # tensors
                weight=weight_tensor,
                inp=inp,
                bias=linear_bias_tensor,
                weight_workspace=weight_workspace,
                # requires_grad flags
                input_requires_grad=inp.requires_grad,
                weight_requires_grad=weight_tensor.requires_grad,
                bias_requires_grad=(
                    linear_bias_tensor.requires_grad if linear_bias_tensor is not None else False
                ),
                # quantizers
                input_quantizer=input_quantizer,
                weight_quantizer=weight_quantizer,
                output_quantizer=output_quantizer,
                grad_input_quantizer=grad_input_quantizer,
                grad_weight_quantizer=grad_weight_quantizer,
                grad_output_quantizer=grad_output_quantizer,
                # numerical / dtype config
                activation_dtype=self.activation_dtype,
                fp8=self.fp8,
                fp8_calibration=self.fp8_calibration,
                fp8_output=fp8_output,
                save_original_input=self.save_original_input,
                backward_override=backward_override,
                custom=custom,
                debug=debug,
                # weight-workspace caching
                is_first_microbatch=is_first_microbatch,
                cache_weight=cache_name is not None,
                skip_fp8_weight_update=skip_fp8_weight_update,
                # tensor / sequence parallelism
                parallel_mode=self.parallel_mode,
                tp_group=self.tp_group,
                tp_size=self.tp_size,
                tensor_parallel=self.tp_size > 1,
                sequence_parallel=self.sequence_parallel,
                symmetric_ar_type=self.symmetric_ar_type,
                backward_input_needs_gather=backward_input_needs_gather,
                # userbuffers
                ub_name=self.ub_name,
                ub_overlap_ag_fprop=ub_overlap_ag_fprop,
                ub_overlap_rs_fprop=ub_overlap_rs_fprop,
                ub_overlap_ag_dgrad=ub_overlap_ag_dgrad,
                ub_overlap_rs_dgrad=ub_overlap_rs_dgrad,
                ub_bulk_dgrad=ub_bulk_dgrad,
                ub_bulk_wgrad=ub_bulk_wgrad,
                # FSDP
                fsdp_group=self.fsdp_group,
                is_fsdp2=self.is_fsdp2,
                # weight-grad scheduling
                fuse_wgrad_accumulation=self.fuse_wgrad_accumulation,
                wgrad_store=wgrad_store,
                # misc
                cpu_offloading=is_cpu_offload_enabled(),
                is_grad_enabled=is_grad_enabled,
            )
            out, new_weight_workspace = linear_fn(
                *autograd_ctx,
                weight_tensor,
                inp,
                linear_bias_tensor,
                fwd_args,
            )

            if new_weight_workspace is not None and cache_name is not None:
                if isinstance(new_weight_workspace, torch.Tensor):
                    new_weight_workspace = new_weight_workspace.detach()
                self._fp8_workspaces[cache_name] = new_weight_workspace

        finally:
            self.end_forward()
        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        return out

    def _get_quantizers(self, fp8_output, fp8_grad, is_grad_enabled):
        if not self.fp8:
            return [None] * 6

        self._warn_missing_output_quantizer_role(fp8_output, fp8_grad)

        grad_input_quantizer = None
        grad_weight_quantizer = None
        grad_output_quantizer = None
        output_quantizer = None
        input_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_INPUT]
        input_quantizer.internal = True
        if not (self.parallel_mode == "column" and self.sequence_parallel):
            input_quantizer.optimize_for_gemm = True
        (weight_quantizer,) = self._get_weight_quantizers()
        if fp8_output:
            output_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_OUTPUT]
        if is_grad_enabled:
            grad_output_quantizer = self.quantizers["scaling_bwd"][FP8BwdTensorIdx.GRAD_OUTPUT1]
            grad_output_quantizer.internal = True
            if not (self.parallel_mode == "row" and self.sequence_parallel):
                grad_output_quantizer.optimize_for_gemm = True
            if fp8_grad:
                grad_input_quantizer = self.quantizers["scaling_bwd"][FP8BwdTensorIdx.GRAD_INPUT1]
        return (
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            grad_input_quantizer,
            grad_weight_quantizer,
            grad_output_quantizer,
        )

    def _get_debug_quantizers(self, fp8_output, fp8_grad, is_grad_enabled):
        original_quantizers = self._get_quantizers(fp8_output, fp8_grad, is_grad_enabled)
        assert TEDebugState.debug_enabled
        from ...debug.pytorch.debug_quantization import DebugQuantizer

        names = ["activation", "weight", "output", "dgrad", "wgrad", "gradient"]
        return tuple(
            DebugQuantizer(self.name, name, q, self.tp_group, self.tp_size)
            for name, q in zip(names, original_quantizers)
        )

    def _get_weight_tensors(self) -> List[Union[torch.Tensor, QuantizedTensorStorage]]:
        """Get the weight tensors of the module."""
        unfused_weights = [getattr(self, name) for name in self.weight_names]
        if any(isinstance(w, QuantizedTensor) for w in unfused_weights):
            if self.fp8:
                if len(unfused_weights) != 1:
                    raise RuntimeError(
                        "Splitting QuantizedTensor into multiple params is not supported"
                    )
            else:
                warnings.warn(
                    "You are using quantized weights without quantized compute. "
                    "Please make sure this is intentional."
                )
                unfused_weights = [w.dequantize() for w in unfused_weights]
        return unfused_weights

    def _get_weight_and_bias_tensors(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Get concatenated weight and bias tensors
        unfused_weights = self._get_weight_tensors()
        weight_tensor = noop_cat(unfused_weights)
        if self.use_bias:
            bias_tensor = noop_cat([getattr(self, name) for name in self.bias_names])
        else:
            bias_tensor = None
        return weight_tensor, bias_tensor

    def onnx_forward(
        self,
        inp: torch.Tensor,
        fp8_output: bool,
        is_grad_enabled: bool,
    ) -> torch.Tensor:
        """
        ONNX-compatible version of the forward function that provides numerical equivalence
        while only using operations that have defined ONNX symbolic translations.
        This simplified implementation is designed specifically for inference scenarios.
        """
        from ..export import onnx_gemm

        assert_warmed_up(self)
        assert not TEDebugState.debug_enabled, "Debug mode is not supported in ONNX export."
        weight_tensor, bias_tensor = self._get_weight_and_bias_tensors()
        (
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            *_,
        ) = self._get_quantizers(fp8_output, False, is_grad_enabled)
        inp_dtype = inp.dtype

        if input_quantizer is not None:
            inp_q = input_quantizer.onnx_quantize(inp)
            inp = input_quantizer.onnx_dequantize(inp_q)
            inp = inp.to(inp_dtype)

        if weight_quantizer is not None:
            weight_q = weight_quantizer.onnx_quantize(weight_tensor)
            weight_tensor = weight_quantizer.onnx_dequantize(weight_q)
        if bias_tensor is not None:
            bias_tensor = bias_tensor.to(inp_dtype)
        weight_tensor = weight_tensor.to(inp_dtype)

        if self.apply_bias:
            output = onnx_gemm(weight_tensor, inp, bias_tensor)
        else:
            output = onnx_gemm(weight_tensor, inp, None)

        if output_quantizer is not None:
            raise NotImplementedError("ONNX export of quantized output is not supported")

        if self.return_bias:
            return output, bias_tensor

        return output

    def _customize_quantizers_float8_current_scaling(self, fwd: bool, recipe: Recipe) -> None:
        """Customize quantizers based on current scaling recipe + linear."""
        assert (
            recipe.float8_current_scaling()
        ), "current scaling recipe quantizer customization here"
        if fwd:
            # set configs about amax epsilon and power_2_scale
            self.quantizers["scaling_fwd"][
                FP8FwdTensorIdx.GEMM1_INPUT
            ].force_pow_2_scales = recipe.fp8_quant_fwd_inp.power_2_scale
            self.quantizers["scaling_fwd"][
                FP8FwdTensorIdx.GEMM1_INPUT
            ].amax_epsilon = recipe.fp8_quant_fwd_inp.amax_epsilon
            # also set weight quantizer with same amax_epsilon & power_2_scale
            self.quantizers["scaling_fwd"][
                FP8FwdTensorIdx.GEMM1_WEIGHT
            ].force_pow_2_scales = recipe.fp8_quant_fwd_weight.power_2_scale
            self.quantizers["scaling_fwd"][
                FP8FwdTensorIdx.GEMM1_WEIGHT
            ].amax_epsilon = recipe.fp8_quant_fwd_weight.amax_epsilon
            # paralle related
            if self.sequence_parallel and self.parallel_mode == "column":
                # customize input_quantizer with amax reduction TP group
                self.quantizers["scaling_fwd"][
                    FP8FwdTensorIdx.GEMM1_INPUT
                ].with_amax_reduction = True
                self.quantizers["scaling_fwd"][
                    FP8FwdTensorIdx.GEMM1_INPUT
                ].amax_reduction_group = self.tp_group
        else:
            # set grad_output_quantizer with amax epsilon and power_2_scale
            self.quantizers["scaling_bwd"][
                FP8BwdTensorIdx.GRAD_OUTPUT1
            ].force_pow_2_scales = recipe.fp8_quant_bwd_grad.power_2_scale
            self.quantizers["scaling_bwd"][
                FP8BwdTensorIdx.GRAD_OUTPUT1
            ].amax_epsilon = recipe.fp8_quant_bwd_grad.amax_epsilon
            # parallel related
            if self.sequence_parallel and self.parallel_mode == "row":
                # customize grad_output_quantizer with amax reduction TP group
                self.quantizers["scaling_bwd"][
                    FP8BwdTensorIdx.GRAD_OUTPUT1
                ].with_amax_reduction = True
                self.quantizers["scaling_bwd"][
                    FP8BwdTensorIdx.GRAD_OUTPUT1
                ].amax_reduction_group = self.tp_group

    def _customize_quantizers_nvfp4(self, fwd: bool, recipe: Recipe) -> None:
        """Customize quantizers based on current scaling recipe + linear."""
        assert recipe.nvfp4(), "Incorrect recipe."
        if fwd:
            if self.sequence_parallel and self.parallel_mode == "column":
                # customize input_quantizer with amax reduction TP group
                self.quantizers["scaling_fwd"][
                    FP8FwdTensorIdx.GEMM1_INPUT
                ].with_amax_reduction = True
                self.quantizers["scaling_fwd"][
                    FP8FwdTensorIdx.GEMM1_INPUT
                ].amax_reduction_group = self.tp_group
        else:
            if self.sequence_parallel and self.parallel_mode == "row":
                # customize grad_output_quantizer with amax reduction TP group
                self.quantizers["scaling_bwd"][
                    FP8BwdTensorIdx.GRAD_OUTPUT1
                ].with_amax_reduction = True
                self.quantizers["scaling_bwd"][
                    FP8BwdTensorIdx.GRAD_OUTPUT1
                ].amax_reduction_group = self.tp_group

    def _get_weight_quantizers(self) -> List[Quantizer]:
        """Get the weight quantizers of the module."""
        if not self.fp8 and not self.fp8_calibration:
            return [None]
        weight_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_WEIGHT]
        weight_quantizer.internal = True
        return [weight_quantizer]

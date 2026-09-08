# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LayerNormLinear API"""
import os
import warnings
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
from functools import reduce
from operator import mul as multiply_op

import torch
from torch.nn import init

import transformer_engine_torch as tex

from transformer_engine.common.recipe import Recipe
from transformer_engine.pytorch.torch_version import torch_version
from transformer_engine.pytorch.tensor.utils import clear_columnwise_cache, is_custom
from .base import (
    fill_userbuffers_buffer_for_all_gather,
    get_ub,
    get_ub_is_fp8,
    is_ub_initialized,
    using_cublasmp_backend,
    quantize_weight,
    TransformerEngineBaseModule,
    get_dummy_wgrad,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ..quantization import FP8GlobalStateManager, QuantizerRole
from ..utils import (
    assert_dim_for_fp8_exec,
    cast_if_needed,
    clear_tensor_data,
    divide,
    get_default_init_method,
    init_method_constant,
    nvtx_range_pop,
    nvtx_range_push,
    needs_quantized_gemm,
    get_nvtx_range_context,
)
from ..distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    allreduce,
    symmetric_all_reduce,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
    _fsdp_scatter_tensors,
    _fsdp_gather_tensors,
)
from ..distributed_weight import (
    is_distributed_weight,
    materialize_weight_for_forward,
    materialize_weight_for_backward,
    finalize_weight_grads,
)
from ..constants import FP8BwdTensorIdx, FP8FwdTensorIdx, GemmParallelModes, dist_group_type
from ..jit import no_torch_dynamo
from ..graph import is_graph_capturing
from ._common import (
    apply_normalization,
    check_fp8_reduce_and_update,
    noop_cat,
    set_quantizer_amax_reduction_group,
    set_quantizer_usage_for_wgrad_all_gather,
    WeightGradStore,
)
from ..quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorStorage,
    Quantizer,
    prepare_for_saving,
    restore_from_func_ctx,
)
from ..dynamo import TensorOrQuantized
from ...debug.pytorch.debug_state import TEDebugState
from ..tensor.mxfp8_tensor import MXFP8Quantizer
from ..tensor.hybrid_tensor import HybridQuantizer
from ..tensor.identity_tensor import IdentityQuantizer
from ..cpu_offload import (
    is_cpu_offload_enabled,
    start_offload,
    mark_not_offload,
    mark_activation_offload,
)
from ..tensor.storage.float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage
from ..tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
from ..export import is_in_onnx_export_mode, assert_warmed_up

from ..cpp_extensions import (
    general_gemm,
)

__all__ = ["LayerNormLinear"]


@dataclass(slots=True)
class LayerNormLinearFwdArgs:
    """Single-argument bag for the forward path of :class:`_LayerNormLinear`."""

    # --- Differentiable tensors (also passed positionally to autograd) ---
    inp: torch.Tensor
    ln_weight: torch.Tensor
    ln_bias: Optional[torch.Tensor]
    weight: TensorOrQuantized
    bias: Optional[torch.Tensor]

    # --- Non-differentiable cached tensors ---
    weight_workspace: Optional[TensorOrQuantized]

    # --- requires_grad flags (cached so backward does not re-query) ---
    input_requires_grad: bool
    ln_weight_requires_grad: bool
    ln_bias_requires_grad: bool
    weight_requires_grad: bool
    bias_requires_grad: bool

    # --- Quantizers ---
    input_quantizer: Optional[Quantizer]
    weight_quantizer: Optional[Quantizer]
    output_quantizer: Optional[Quantizer]
    grad_input_quantizer: Optional[Quantizer]
    grad_weight_quantizer: Optional[Quantizer]
    grad_output_quantizer: Optional[Quantizer]

    # --- Normalization ---
    eps: float
    normalization: str
    zero_centered_gamma: bool
    fwd_ln_sm_margin: int
    bwd_ln_sm_margin: int
    return_layernorm_output: bool
    return_layernorm_output_gathered: bool

    # --- Numerical / dtype config ---
    activation_dtype: torch.dtype
    fp8: bool
    fp8_calibration: bool
    backward_override: Optional[str]
    dgrad_use_split_accumulator: bool
    wgrad_use_split_accumulator: bool
    debug: bool

    # --- Weight-workspace caching ---
    is_first_microbatch: Optional[bool]
    cache_weight: bool
    skip_fp8_weight_update: Optional[torch.Tensor]

    # --- Tensor / sequence parallelism ---
    parallel_mode: Optional[str]
    tp_group: Optional[dist_group_type]
    tp_size: int
    tensor_parallel: bool
    sequence_parallel: bool
    symmetric_ar_type: Optional[str]

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

    def any_requires_grad(self) -> bool:
        """Whether any differentiable input requires a gradient."""
        return any(
            (
                self.input_requires_grad,
                self.ln_weight_requires_grad,
                self.ln_bias_requires_grad,
                self.weight_requires_grad,
                self.bias_requires_grad,
            )
        )


@dataclass(slots=True)
class LayerNormLinearBwdArgs:
    """Single-argument bag for the backward path of :class:`_LayerNormLinear`."""

    # --- Incoming gradients (populated at backward entry) ---
    grad_output: Optional[torch.Tensor] = None
    grad_ln_out: Optional[torch.Tensor] = None

    # --- Saved / restored tensors (populated at backward entry) ---
    inputmat: Optional[torch.Tensor] = None
    weight_fp8: Optional[TensorOrQuantized] = None
    saved_weight: Optional[TensorOrQuantized] = None
    bias: Optional[torch.Tensor] = None
    ln_weight: Optional[torch.Tensor] = None
    ln_out: Optional[TensorOrQuantized] = None
    mu: Optional[torch.Tensor] = None
    rsigma: Optional[torch.Tensor] = None

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
    ln_out_needs_gather: bool = False
    inp_shape: Optional[torch.Size] = None

    # --- Normalization ---
    normalization: str = "LayerNorm"
    zero_centered_gamma: bool = False
    bwd_ln_sm_margin: int = 0
    return_layernorm_output: bool = False
    return_layernorm_output_gathered: bool = False

    # --- Numerical / dtype config ---
    activation_dtype: Optional[torch.dtype] = None
    fp8: bool = False
    dgrad_use_split_accumulator: bool = _2X_ACC_DGRAD
    wgrad_use_split_accumulator: bool = _2X_ACC_WGRAD
    backward_override: Optional[str] = None
    is_weight_param_quantized: bool = False
    debug: bool = False

    # --- Tensor / sequence parallelism ---
    parallel_mode: Optional[str] = None
    tp_group: Optional[dist_group_type] = None
    tp_size: int = 1
    tensor_parallel: bool = False
    sequence_parallel: bool = False

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

    # --- Per-backward scratch state (populated inside the backward impl) ---
    ub_obj_gradout: Optional[Any] = None

    def setup_saved_tensors(self, ctx: torch.autograd.function.FunctionCtx) -> None:
        """Pull saved tensors from ``ctx`` into the fields backward consumes."""
        (
            self.inputmat,
            self.weight_fp8,
            self.saved_weight,
            self.bias,
            self.ln_weight,
            self.ln_out,
            self.mu,
            self.rsigma,
        ) = restore_from_func_ctx(
            ctx
        )  # pylint: disable=unbalanced-tuple-unpacking


def _layernorm_linear_forward_impl(
    args: LayerNormLinearFwdArgs,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[Tuple],
    Optional[Dict],
]:
    """Forward implementation for the layernorm-linear layer.

    Returns ``(out, ln_out_return, new_weight_workspace,
    tensors_to_save_from_forward, ctx_attrs)``. ``new_weight_workspace`` is
    the freshly produced FP8 weight workspace (returned alongside ``out`` so
    the caller can refresh its cache). The last two are ``None`` when
    gradients are disabled.
    """
    inp = args.inp
    ln_weight = args.ln_weight
    ln_bias = args.ln_bias
    weight = args.weight
    bias = args.bias
    input_quantizer = args.input_quantizer
    weight_quantizer = args.weight_quantizer
    output_quantizer = args.output_quantizer
    is_first_microbatch = args.is_first_microbatch
    fp8 = args.fp8
    debug = args.debug
    cpu_offloading = args.cpu_offloading
    tp_group = args.tp_group
    tp_size = args.tp_size
    sequence_parallel = args.sequence_parallel
    activation_dtype = args.activation_dtype
    parallel_mode = args.parallel_mode
    is_grad_enabled = args.is_grad_enabled
    return_layernorm_output = args.return_layernorm_output
    return_layernorm_output_gathered = args.return_layernorm_output_gathered
    backward_override = args.backward_override
    ub_name = args.ub_name
    ub_overlap_ag_fprop = args.ub_overlap_ag_fprop
    ub_overlap_rs_fprop = args.ub_overlap_rs_fprop
    fsdp_group = args.fsdp_group
    is_fsdp2 = args.is_fsdp2
    weight_requires_grad = args.weight_requires_grad

    # NVTX label for profiling
    nvtx_label = "transformer_engine._LayerNormLinear.forward"
    if ub_name is not None:
        nvtx_label = f"{nvtx_label}.{ub_name}"

    with_input_all_gather = parallel_mode == "column" and sequence_parallel

    # Make sure input dimensions are compatible
    out_features, in_features = weight.shape
    inp_shape = inp.shape
    assert inp_shape[-1] == in_features, "GEMM not possible"
    inp = inp.view((-1, in_features))
    inputmat = inp
    if fp8:
        assert_dim_for_fp8_exec(inputmat, weight)

    # Cast for native AMP
    nvtx_range_push(f"{nvtx_label}.norm_input_cast")
    inputmat = cast_if_needed(inputmat, activation_dtype)
    ln_weight_cast = cast_if_needed(ln_weight, activation_dtype)
    if ln_bias is not None:
        ln_bias = cast_if_needed(ln_bias, activation_dtype)
    nvtx_range_pop(f"{nvtx_label}.norm_input_cast")

    if is_cpu_offload_enabled():
        start_offload(inputmat)

    tp_world_size = get_distributed_world_size(tp_group)

    backward_needs_input = is_grad_enabled and weight_requires_grad

    # Configure Userbuffers communication (comm+GEMM overlap)
    ub_obj = None
    ub_type = None
    ub_overlap_ag_fprop = ub_overlap_ag_fprop and is_grad_enabled and not return_layernorm_output
    if ub_overlap_rs_fprop:
        ub_obj = get_ub(ub_name + "_fprop", fp8)
        ub_type = tex.CommOverlapType.RS
    elif ub_overlap_ag_fprop:
        ub_obj = get_ub(ub_name + "_fprop", fp8)
        ub_type = tex.CommOverlapType.AG

    # Configure quantizer for norm output
    if fp8:
        if input_quantizer is None:
            raise ValueError("Missing quantizer for input tensor")
        input_quantizer.set_usage(
            rowwise=True,
            columnwise=backward_needs_input and backward_override is None,
        )
        if with_input_all_gather and input_quantizer.supports_only_rowwise_all_gather():
            # All-gather is not supported with FP8 column-wise data
            input_quantizer.set_usage(columnwise=False)
        # Amax reduction group for the input quantizer (column-parallel sequence parallel)
        set_quantizer_amax_reduction_group(
            input_quantizer,
            tp_group if (sequence_parallel and parallel_mode == "column") else None,
        )

    # Avoid quantized norm kernel if norm output will be returned
    # or if a gather of ln_out must be in high precision.
    custom = is_custom(input_quantizer)
    hybrid = isinstance(input_quantizer, HybridQuantizer)
    identity = isinstance(input_quantizer, IdentityQuantizer)
    with_quantized_norm = (
        fp8
        and not debug
        and not return_layernorm_output
        and not return_layernorm_output_gathered
        and backward_override is None
        and not custom  # TODO(negvet): and not FP8GlobalStateManager.get_fp8_recipe().custom()
        and not hybrid
        and not identity
    )

    # Apply normalization
    nvtx_range_push(f"{nvtx_label}.norm")
    ln_out, mu, rsigma = apply_normalization(
        inputmat,
        None,  # ln_out
        ln_weight_cast,
        ln_bias,
        args.eps,
        input_quantizer if with_quantized_norm else None,
        inputmat.dtype,
        args.normalization,
        args.fwd_ln_sm_margin,
        args.zero_centered_gamma,
    )
    nvtx_range_pop(f"{nvtx_label}.norm")

    # Store unquantized layer norm output if we need to return it
    ln_out_return = None
    if return_layernorm_output or return_layernorm_output_gathered:
        ln_out_return = ln_out
    ln_out_hp = ln_out if backward_override == "high_precision" else None

    # ------------------------------------------------------
    # Prepare GEMM input tensor
    # Note: Cast to expected dtype and perform tensor-parallel communication
    # ------------------------------------------------------
    nvtx_range_push(f"{nvtx_label}.gemm_input_cast_comm")
    ln_out_total = None
    if with_input_all_gather:
        if return_layernorm_output_gathered:
            # Perform all-gather in high precision if gathered
            # norm output will be returned
            ln_out_total, _ = gather_along_first_dim(ln_out, tp_group)
            ln_out_return = ln_out_total
            if fp8 or debug:
                ln_out = input_quantizer(ln_out)
                input_quantizer.set_usage(rowwise=True, columnwise=False)
                ln_out_total = input_quantizer(ln_out_total)
        else:
            quantizer = None
            if fp8 or debug:
                quantizer = input_quantizer
                # custom recipe doesn't need to support quantized AG
                if not with_quantized_norm and not custom:
                    ln_out = quantizer(ln_out)
                quantizer.set_usage(rowwise=True, columnwise=False)
            if ub_overlap_ag_fprop:  # Initialize Userbuffers all-gather
                ln_out_total, _ = fill_userbuffers_buffer_for_all_gather(
                    ub_obj,
                    ln_out,
                    quantizer,
                    tp_group,
                )
            else:  # Perform NCCL all-gather
                ln_out_total, _ = gather_along_first_dim(
                    ln_out,
                    tp_group,
                    quantizer=quantizer,
                )
    else:
        if (fp8 or debug) and not with_quantized_norm:
            ln_out = input_quantizer(ln_out)
        ln_out_total = ln_out
    nvtx_range_pop(f"{nvtx_label}.gemm_input_cast_comm")
    # ------------------------------------------------------
    # GEMM input tensor is ready...
    # ------------------------------------------------------

    # ------------------------------------------------------
    # Prepare weight tensor
    # ------------------------------------------------------
    is_dist_weight = is_distributed_weight(args.weight)
    if is_dist_weight:
        weight = materialize_weight_for_forward(weight)[0]
        out_features = weight.shape[0]
    new_weight_workspace = None
    weightmat = weight
    is_weight_param_quantized = False
    if fp8 or debug:
        is_weight_param_quantized = isinstance(weight, QuantizedTensorStorage)

        # Configure quantizer
        # If weight is already quantized, weight._quantizer is its true quantizer.
        # for debug mode we create quantizer every iteration, thus we need to set the quantizer states
        if is_weight_param_quantized and not debug:
            weight_quantizer = weight._quantizer
        elif weight_quantizer is not None:
            # FSDP2: Skip columnwise/transpose creation during forward
            # to avoid accumulating caches across layers. Backward's
            # FSDP2 all-gather will recreate them. (Issue #2681)
            weight_quantizer.set_usage(
                rowwise=True,
                columnwise=is_grad_enabled and not is_fsdp2 and backward_override is None,
            )

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
    if needs_quantized_gemm(ln_out_total) and activation_dtype == torch.float32:
        # cuBLAS does not support FP8 GEMM with FP32 bias, so we cast to BF16
        bias_dtype = torch.bfloat16
    bias_cast = cast_if_needed(bias, bias_dtype) if bias is not None else bias

    # Calibrate quantizers if needed
    if not fp8 and args.fp8_calibration:
        if input_quantizer is not None:
            input_quantizer.calibrate(ln_out_total)
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
        out_shape = list(inp_shape)
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
        ln_out_total,
        quantization_params=output_quantizer,
        out_dtype=activation_dtype,
        bias=bias_cast,
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
    if not weight_requires_grad and not return_layernorm_output:
        clear_tensor_data(ln_out, ln_out_total)
        ln_out = ln_out_total = None
    elif with_input_all_gather and not return_layernorm_output_gathered:
        # ln_out_total aliases ln_out for the cuBLASMp backend; skip the
        # deallocation to avoid corrupting the backward-saved tensor.
        if ln_out_total is not ln_out:
            clear_tensor_data(ln_out_total)
        ln_out_total = None

    # ------------------------------------------------------
    # Prepare output tensor
    # Note: Perform tensor-parallel communication
    # ------------------------------------------------------
    out = None
    if ub_overlap_rs_fprop:
        # cuBLASMp writes the reduce-scattered output directly into the
        # GEMM output tensor; Userbuffers writes it into the extra-output buffer.
        out = gemm_out if ub_obj is not None and ub_obj.with_cublasmp() else reduce_scatter_out
    elif parallel_mode == "row" and tp_size > 1:
        nvtx_range_push(f"{nvtx_label}.row_parallel_comm")
        out = gemm_out
        if sequence_parallel:
            out, _ = reduce_scatter_along_first_dim(out, tp_group)
        elif args.tensor_parallel:
            if args.symmetric_ar_type is not None:
                out, _ = symmetric_all_reduce(out, tp_group, all_reduce_type=args.symmetric_ar_type)
            else:
                out, _ = allreduce(out, tp_group)
        nvtx_range_pop(f"{nvtx_label}.row_parallel_comm")
    else:
        out = gemm_out
    out = out.view(-1, *inp_shape[1:-1], out_features)
    # ------------------------------------------------------
    # Output tensor is ready to return...
    # ------------------------------------------------------

    # Prepare backward state
    tensors_to_save_from_forward = None
    ctx_attrs = None

    if is_grad_enabled:
        ln_out_to_save = ln_out
        if backward_override == "high_precision":
            ln_out_to_save = ln_out_hp
        ln_out_needs_gather = (
            weight_requires_grad and parallel_mode == "column" and sequence_parallel
        )

        # Input with column-wise usage is needed for wgrad GEMM.
        if backward_needs_input and backward_override is None:
            if isinstance(ln_out, QuantizedTensorStorage):
                # For sequence parallel in vanilla FP8, rowwise data is
                # to gather the input. For MXFP8, columnwise only data
                # can be allgathered.
                if (
                    isinstance(ln_out, (MXFP8TensorStorage, Float8BlockwiseQTensorStorage))
                    or not ln_out_needs_gather
                ):
                    ln_out.update_usage(rowwise_usage=False)

        if cpu_offloading:
            mark_activation_offload(inputmat, mu, rsigma, ln_out_to_save)

        # Scatter intermediate/activation tensors saved for the backward pass
        # NOTE: weight_fp8 = weight when fp8 == False and torch.disttributed.FSDP already
        #       shards/unshards the base weights so we don't do it ourselves
        nvtx_range_push(f"{nvtx_label}.fsdp_scatter")
        fsdp_shapes = _fsdp_scatter_tensors(
            fsdp_group,
            mu,
            rsigma,
            weightmat if fp8 and not is_weight_param_quantized else None,
            ln_out_to_save if weight_requires_grad else None,
        )
        nvtx_range_pop(f"{nvtx_label}.fsdp_scatter")

        if cpu_offloading:
            mark_not_offload(
                weightmat,
                weight,
                bias_cast,
                ln_weight_cast,
                ln_bias,
            )

        # FSDP2: Don't save FP8 workspace for non-quantized weights.
        # Backward will re-quantize from FSDP2 all-gathered weight.
        # (Issue #2681)
        wt_save = weightmat
        if is_fsdp2 and weightmat is not weight:
            wt_save = None
        # Distributed weight (e.g. GTP): don't save the gathered quantized workspace;
        # backward re-gathers from the saved (sharded) weight and re-quantizes.
        if is_dist_weight:
            wt_save = None

        # Dedup save slots that alias forward inputs or other outputs;
        # ``_layernorm_linear_setup_ctx`` rebuilds the refs.
        if wt_save is None:
            wt_alias = None
        elif wt_save is weight:
            wt_alias = "weight"
        elif new_weight_workspace is not None and wt_save is new_weight_workspace:
            wt_alias = "new_weight_workspace"
        elif args.weight_workspace is not None and wt_save is args.weight_workspace:
            wt_alias = "weight_workspace"
        else:
            wt_alias = None
        saved_tensor_aliases = (
            "inp" if inputmat is inp else None,
            wt_alias,
            "weight",  # ``saved_weight`` slot is always the weight parameter
            "bias" if bias_cast is not None and bias_cast is bias else None,
            "ln_weight" if ln_weight_cast is ln_weight else None,
            (
                "ln_out"
                if return_layernorm_output
                and ln_out_to_save is not None
                and ln_out_to_save is ln_out_return
                else None
            ),
            None,
            None,
        )
        tensors_to_save_from_forward = (
            None if saved_tensor_aliases[0] is not None else inputmat,
            None if saved_tensor_aliases[1] is not None else wt_save,
            None,
            None if saved_tensor_aliases[3] is not None else bias_cast,
            None if saved_tensor_aliases[4] is not None else ln_weight_cast,
            None if saved_tensor_aliases[5] is not None else ln_out_to_save,
            mu,
            rsigma,
        )

        ctx_attrs = {
            "fsdp_shapes": fsdp_shapes,
            "saved_tensor_aliases": saved_tensor_aliases,
            "is_weight_param_quantized": is_weight_param_quantized,
            "ln_out_needs_gather": ln_out_needs_gather,
        }

    ln_out_for_return = None
    if return_layernorm_output:
        if return_layernorm_output_gathered:
            shape = list(inp_shape)
            shape[0] *= tp_size if with_input_all_gather else 1
            ln_out_for_return = ln_out_return.view(shape)
        else:
            ln_out_for_return = ln_out_return.view(inp_shape)
    return out, ln_out_for_return, new_weight_workspace, tensors_to_save_from_forward, ctx_attrs


def _layernorm_linear_setup_ctx(
    bwd_args: LayerNormLinearBwdArgs,
    fwd_args: LayerNormLinearFwdArgs,
    fwd_outputs: Tuple[Any, ...],
    ctx_attrs: Dict,
    tensors_to_save_from_forward: Tuple[Any, ...],
) -> Tuple[Any, ...]:
    """Populate ``bwd_args`` from forward state.

    Returns the tensors that should be passed through ``prepare_for_saving``
    by the caller. ``fwd_outputs`` is ``(out, ln_out_return,
    new_weight_workspace)``; the last two rebuild the deduped save slots.
    """
    inp = fwd_args.inp
    weight = fwd_args.weight
    bias = fwd_args.bias
    ln_weight = fwd_args.ln_weight

    backward_override = fwd_args.backward_override
    fp8 = fwd_args.fp8
    debug = fwd_args.debug
    fuse_wgrad_accumulation = fwd_args.fuse_wgrad_accumulation
    is_weight_param_quantized = ctx_attrs["is_weight_param_quantized"]

    # Quantizers
    bwd_args.input_quantizer = fwd_args.input_quantizer
    bwd_args.weight_quantizer = (
        weight._quantizer
        if (is_weight_param_quantized and not debug and isinstance(weight, QuantizedTensorStorage))
        else fwd_args.weight_quantizer
    )
    bwd_args.grad_input_quantizer = fwd_args.grad_input_quantizer
    bwd_args.grad_weight_quantizer = fwd_args.grad_weight_quantizer
    bwd_args.grad_output_quantizer = fwd_args.grad_output_quantizer

    # Differentiability summary
    bwd_args.use_bias = bias is not None
    bwd_args.requires_dgrad = fwd_args.input_requires_grad
    bwd_args.requires_wgrad = fwd_args.weight_requires_grad
    bwd_args.ln_out_needs_gather = ctx_attrs["ln_out_needs_gather"]
    bwd_args.inp_shape = inp.shape

    # Normalization
    bwd_args.normalization = fwd_args.normalization
    bwd_args.zero_centered_gamma = fwd_args.zero_centered_gamma
    bwd_args.bwd_ln_sm_margin = fwd_args.bwd_ln_sm_margin
    bwd_args.return_layernorm_output = fwd_args.return_layernorm_output
    bwd_args.return_layernorm_output_gathered = fwd_args.return_layernorm_output_gathered

    # Numerical / dtype config
    bwd_args.activation_dtype = fwd_args.activation_dtype
    bwd_args.fp8 = fp8
    bwd_args.dgrad_use_split_accumulator = fwd_args.dgrad_use_split_accumulator
    bwd_args.wgrad_use_split_accumulator = fwd_args.wgrad_use_split_accumulator
    bwd_args.backward_override = backward_override
    bwd_args.is_weight_param_quantized = is_weight_param_quantized
    bwd_args.debug = debug

    # Tensor / sequence parallelism
    bwd_args.parallel_mode = fwd_args.parallel_mode
    bwd_args.tp_group = fwd_args.tp_group
    bwd_args.tp_size = fwd_args.tp_size
    bwd_args.tensor_parallel = fwd_args.tensor_parallel
    bwd_args.sequence_parallel = fwd_args.sequence_parallel

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
        # Keep weakref to weight to preserve attributes like main_grad
        # when we need to modify the weight python object
        bwd_args.origin_weight_ref = weakref.ref(weight)
        bwd_args.origin_weight_overwrites_main_grad = getattr(weight, "overwrite_main_grad", False)
        # MCore FSDP creates main_grad lazily before backward, so don't touch it here
        if hasattr(weight, "__fsdp_param__"):
            bwd_args.main_grad_func = weight.get_main_grad
        elif is_distributed_weight(weight):
            bwd_args.main_grad_func = weight.grad_buffer
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

    (
        saved_inputmat,
        wt_save,
        saved_weight,
        saved_bias,
        saved_ln_weight,
        saved_ln_out,
        mu,
        rsigma,
    ) = tensors_to_save_from_forward
    (
        inputmat_alias,
        wt_save_alias,
        saved_weight_alias,
        bias_alias,
        ln_weight_alias,
        ln_out_alias,
        _,
        _,
    ) = ctx_attrs["saved_tensor_aliases"]
    in_features = inp.shape[-1]
    if inputmat_alias == "inp":
        saved_inputmat = inp.view((-1, in_features))
    if wt_save_alias == "weight":
        wt_save = weight
    elif wt_save_alias == "new_weight_workspace":
        wt_save = fwd_outputs[2]
    elif wt_save_alias == "weight_workspace":
        wt_save = fwd_args.weight_workspace
    if saved_weight_alias == "weight":
        saved_weight = weight
    if bias_alias == "bias":
        saved_bias = bias
    if ln_weight_alias == "ln_weight":
        saved_ln_weight = ln_weight
    if ln_out_alias == "ln_out":
        saved_ln_out = fwd_outputs[1].view((-1, in_features))
    if fwd_args.cpu_offloading:
        # Rebuilt views don't carry the offload marks set on the forward tensors
        mark_activation_offload(
            saved_inputmat if inputmat_alias == "inp" else None,
            saved_ln_out if ln_out_alias == "ln_out" else None,
        )
    return (
        saved_inputmat,
        wt_save,
        saved_weight,
        saved_bias,
        saved_ln_weight,
        saved_ln_out,
        mu,
        rsigma,
    )


def _layernorm_linear_backward_impl(
    args: LayerNormLinearBwdArgs,
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Backward implementation for the layernorm-linear layer.

    Caller must have populated ``args.grad_output`` / ``args.grad_ln_out`` and
    run ``args.setup_saved_tensors(ctx)`` before invocation. Returns
    ``(dgrad, dgamma, dbeta, wgrad, grad_bias)``.
    """
    grad_output = args.grad_output
    assert grad_output is not None

    # NVTX label for profiling
    nvtx_label = "transformer_engine._LayerNormLinear.backward"
    if args.ub_name is not None:
        nvtx_label = f"{nvtx_label}.{args.ub_name}"

    with get_nvtx_range_context("_LayerNormLinear_backward"):
        inputmat = args.inputmat
        weight = args.weight_fp8
        saved_weight = args.saved_weight
        bias = args.bias
        ln_weight = args.ln_weight
        ln_out = args.ln_out
        mu = args.mu
        rsigma = args.rsigma

        is_dist_weight = is_distributed_weight(saved_weight)
        if is_dist_weight:
            weight = materialize_weight_for_backward(saved_weight)[0]
        # Restore from weakref to get original weight python object
        # (preserves attributes like main_grad, grad_added_to_main_grad, etc.)
        # Only needed when fuse_wgrad_accumulation is enabled.
        origin_weight = None
        origin_weight_overwrites_main_grad = args.origin_weight_overwrites_main_grad
        main_grad = None
        if args.fuse_wgrad_accumulation and args.requires_wgrad:
            origin_weight_ref = args.origin_weight_ref
            args.origin_weight_ref = None
            origin_weight = origin_weight_ref() if origin_weight_ref is not None else None
            assert (
                origin_weight is not None
            ), "weight was removed while fuse_wgrad_accumulation=True"
            # Since main_grad can be modified inplace, it should not be a part of saved_tensors
            main_grad = args.main_grad_func() if weight is not None else None
            if main_grad is not None and not is_dist_weight:
                origin_weight.main_grad = main_grad

        # Gather intermediate/activation tensors if needed
        # NOTE: weight_fp8 = weight when fp8 == False and torch.disttributed.FSDP already
        #       shards/unshards the base weights so we don't do it ourselves
        nvtx_range_push(f"{nvtx_label}.fsdp_gather")
        _fsdp_gather_tensors(
            args.fsdp_group,
            args.fsdp_shapes,
            mu,
            rsigma,
            weight if args.fp8 and not args.is_weight_param_quantized else None,
            ln_out,
        )
        nvtx_range_pop(f"{nvtx_label}.fsdp_gather")

        # Configure Userbuffers communication (comm+GEMM overlap)
        args.ub_obj_gradout = None
        ub_obj_dgrad = None
        ub_obj_wgrad = None
        ub_type_dgrad = None
        ub_type_wgrad = None
        dgrad_shape = [reduce(multiply_op, args.inp_shape[:-1]), args.inp_shape[-1]]
        if args.ub_overlap_ag:
            # Overlap grad_output all-gather with dgrad compute
            args.ub_obj_gradout = get_ub(args.ub_name + "_dgrad", args.fp8)
            ub_obj_dgrad = args.ub_obj_gradout
            ub_type_dgrad = tex.CommOverlapType.AG
        elif args.ub_overlap_rs_dgrad:
            # Overlap dgrad reduce-scatter with dgrad compute
            args.ub_obj_gradout = get_ub(args.ub_name + "_dgrad", args.fp8)
            ub_obj_dgrad = args.ub_obj_gradout
            ub_type_dgrad = tex.CommOverlapType.RS
        else:
            if args.ub_bulk_dgrad:
                # Overlap inputmat all-gather with dgrad compute
                args.ub_obj_gradout = get_ub(args.ub_name + "_dgrad", args.fp8)
                ub_obj_dgrad = args.ub_obj_gradout
                ub_type_dgrad = tex.CommOverlapType.AG
            if args.ub_bulk_wgrad:
                # Overlap dgrad reduce-scatter with wgrad compute
                ub_obj_wgrad = get_ub(args.ub_name + "_wgrad", args.fp8)
                ub_type_wgrad = tex.CommOverlapType.RS

        # --------------------------------------------------
        # Prepare grad output tensor
        # Note: Cast to expected dtype and perform tensor-parallel communication
        # --------------------------------------------------

        # Configure quantizer for grad output tensor
        # Note: dgrad GEMM requires row-wise usage, wgrad GEMM
        # requires column-wise usage
        if args.grad_output_quantizer is not None:
            quantizer = args.grad_output_quantizer
            quantizer.set_usage(rowwise=True, columnwise=True)
            if args.ub_overlap_ag:
                # Userbuffers only supports communication for one
                # tensor usage at a time. Configure quantizer with
                # usage for only dgrad GEMM.
                quantizer.set_usage(columnwise=False)
            # Amax reduction group for grad output (row-parallel sequence parallel)
            set_quantizer_amax_reduction_group(
                quantizer,
                (
                    args.tp_group
                    if (args.sequence_parallel and args.parallel_mode == "row")
                    else None
                ),
            )

        # Prepare grad output tensor
        # Note: Cast to expected dtype and perform tensor-parallel communication
        nvtx_range_push(f"{nvtx_label}.grad_output_preprocess")
        (
            grad_output,
            grad_bias,
        ) = TransformerEngineBaseModule.grad_output_preprocess(
            args,
            grad_output,
            args.parallel_mode == "row",
            args.grad_output_quantizer,
        )
        nvtx_range_pop(f"{nvtx_label}.grad_output_preprocess")

        # --------------------------------------------------
        # Grad output tensor is ready for computing grad input...
        # --------------------------------------------------

        # --------------------------------------------------
        # Prepare GEMM input tensor
        # Note: Input tensor is needed for wgrad GEMM.
        # Tensor-parallel communication is overlapped with dgrad
        # GEMM.
        # --------------------------------------------------
        ln_out_total = None
        ln_out_total_work = None
        if args.backward_override == "dequantized":
            if isinstance(ln_out, QuantizedTensorStorage):
                ln_out = ln_out.dequantize(dtype=args.activation_dtype)
            else:
                ln_out = cast_if_needed(ln_out, args.activation_dtype)
        if args.ln_out_needs_gather:
            quantizer = None
            if args.input_quantizer is not None and args.fp8:
                quantizer = args.input_quantizer
                set_quantizer_usage_for_wgrad_all_gather(quantizer)
            if args.ub_bulk_dgrad:
                ln_out_total, _ = fill_userbuffers_buffer_for_all_gather(
                    ub_obj_dgrad,
                    ln_out,
                    quantizer,
                    args.tp_group,
                )
            else:
                nvtx_range_push(f"{nvtx_label}.column_parallel_comm_input")
                ln_out_total, ln_out_total_work = gather_along_first_dim(
                    ln_out,
                    args.tp_group,
                    async_op=True,
                    quantizer=quantizer,
                )
                nvtx_range_pop(f"{nvtx_label}.column_parallel_comm_input")
        else:
            ln_out_total = ln_out
        # --------------------------------------------------
        # Input tensor is ready for computing grad weight...
        # --------------------------------------------------

        # --------------------------------------------------
        # Compute grad input tensor
        # Note: Gradient w.r.t. GEMM input (i.e. norm output).
        # --------------------------------------------------

        # FSDP2: Re-create workspace from all-gathered weight when
        # workspace was not saved. (Issue #2681)
        # Use saved_weight (the original weight parameter) since
        # origin_weight is only set when fuse_wgrad_accumulation=True.
        if weight is None:
            if isinstance(saved_weight, QuantizedTensorStorage):
                # saved weight is already set to right usages by
                # fsdp2 quantized-tensor hooks when workspace was not saved.
                weight = saved_weight
            elif args.weight_quantizer is not None:
                args.weight_quantizer.set_usage(rowwise=True, columnwise=True)
                weight = args.weight_quantizer(saved_weight)

        # Make sure required data is available
        if isinstance(grad_output, QuantizedTensorStorage):
            grad_output.update_usage(rowwise_usage=True)
        if (
            args.fp8
            and args.weight_quantizer is not None
            and isinstance(weight, QuantizedTensorStorage)
        ):
            weight.update_usage(columnwise_usage=True)

        # Choose whether to use GEMM kernel with split accumulator
        use_split_accumulator = args.dgrad_use_split_accumulator

        # Update grad input quantizer
        if args.grad_input_quantizer is not None:
            args.grad_input_quantizer.set_usage(rowwise=True, columnwise=False)

        # Output buffers for Userbuffers reduce-scatter
        gemm_out = None
        reduce_scatter_out = None
        if args.ub_overlap_rs_dgrad:
            reduce_scatter_out = torch.empty(
                dgrad_shape, dtype=args.activation_dtype, device=args.grad_output.device
            )
        elif args.ub_bulk_wgrad:
            gemm_out = ub_obj_wgrad.get_buffer(local_chunk=False)

        # dgrad GEMM
        # Note: dx = dy * w
        nvtx_range_push(f"{nvtx_label}.dgrad_gemm")
        weight_for_dgrad = weight
        if args.backward_override == "dequantized":
            if isinstance(weight_for_dgrad, QuantizedTensorStorage):
                weight_for_dgrad = weight_for_dgrad.dequantize(dtype=args.activation_dtype)
            else:
                weight_for_dgrad = cast_if_needed(weight_for_dgrad, args.activation_dtype)
        elif args.backward_override == "high_precision":
            weight_for_dgrad = saved_weight
            if isinstance(weight_for_dgrad, QuantizedTensorStorage):
                weight_for_dgrad = weight_for_dgrad.dequantize(dtype=args.activation_dtype)
        gemm_out, *_, reduce_scatter_out = general_gemm(
            weight_for_dgrad,
            grad_output,
            layout="NN",
            grad=True,
            quantization_params=args.grad_input_quantizer,
            out=gemm_out,
            out_dtype=args.activation_dtype,
            use_split_accumulator=use_split_accumulator,
            ub=ub_obj_dgrad,
            ub_type=ub_type_dgrad,
            extra_output=reduce_scatter_out,
            bulk_overlap=args.ub_bulk_dgrad,
        )
        nvtx_range_pop(f"{nvtx_label}.dgrad_gemm")

        # FSDP2 only handles deallocation all-gathered weights that it allocates.
        # Columnwise data is derived from rowwise data after allgather for fp8
        # and 2d block-scaled weights in TE managed memory. So we need to clear
        # it here.
        # (Issues #2681, #2717)
        if args.is_fsdp2 and isinstance(weight, QuantizedTensorStorage):
            clear_columnwise_cache(weight)

        # Prepare grad input tensor
        # Note: Perform tensor-parallel communication
        dgrad = None
        dgrad_work = None
        if args.ub_overlap_rs_dgrad:
            # cuBLASMp writes the reduce-scattered dgrad directly into the
            # GEMM output tensor; Userbuffers uses the extra-output buffer.
            dgrad = (
                gemm_out
                if ub_obj_dgrad is not None and ub_obj_dgrad.with_cublasmp()
                else reduce_scatter_out
            )
        elif args.ub_bulk_wgrad:
            dgrad = ub_obj_wgrad.get_buffer(local_chunk=True)
        elif args.parallel_mode == "column" and args.tp_size > 1:
            nvtx_range_push(f"{nvtx_label}.column_parallel_comm_dgrad")
            dgrad = gemm_out
            if args.sequence_parallel:
                dgrad, dgrad_work = reduce_scatter_along_first_dim(
                    dgrad,
                    args.tp_group,
                    async_op=True,
                )
            else:
                dgrad, dgrad_work = allreduce(dgrad, args.tp_group, async_op=True)
            nvtx_range_pop(f"{nvtx_label}.column_parallel_comm_dgrad")
        else:
            dgrad = gemm_out

        # --------------------------------------------------
        # Grad input tensor has been computed...
        # --------------------------------------------------

        # cuBLASMp's AG+GEMM consumes the gathered grad_output inline and
        # does not preserve it for wgrad. Userbuffers leaves the gathered
        # tensor in its persistent buffer; cuBLASMp does not, so we gather
        # here. Route through the same FP8-aware all-gather as the
        # non-overlap path in
        # ``TransformerEngineBaseModule.grad_output_preprocess`` by passing
        # the grad_output quantizer. Columnwise data needed for wgrad is
        # produced by ``update_usage(columnwise_usage=True)`` further below.
        if (
            args.requires_wgrad
            and args.ub_overlap_ag
            and args.ub_obj_gradout is not None
            and args.ub_obj_gradout.with_cublasmp()
        ):
            if args.grad_output_quantizer is not None:
                set_quantizer_usage_for_wgrad_all_gather(args.grad_output_quantizer)
            grad_output, _ = gather_along_first_dim(
                grad_output,
                args.tp_group,
                quantizer=args.grad_output_quantizer,
            )

        # --------------------------------------------------
        # Compute grad weight
        # --------------------------------------------------

        wgrad = None
        if args.requires_wgrad:
            # Prepare grad output tensor
            # Note: Synchronize tensor-parallel communication and
            # make sure required data is available
            if args.ub_overlap_ag and isinstance(args.grad_output_quantizer, MXFP8Quantizer):
                # UB does not support pipelined overlapping grad output
                # all-gather with wgrad GEMM. Also, we can't
                # convert row-scaled MXFP8 to column-scaled, so we
                # can't reuse the grad output that was gathered
                # for the dgrad GEMM. We work around by explicitly
                # overlapping the AG operation with the dgrad GEMM.

                # Get the communication stream from the dgrad GEMM to use for the AG
                dgrad_send_stream, dgrad_recv_stream = ub_obj_dgrad.get_communication_stream()

                # This object is separate from the ub_obj_wgrad object which is passed to the GEMM
                ub_obj_overlap_wgrad = get_ub(args.ub_name + "_wgrad", args.fp8)

                args.grad_output_quantizer.set_usage(rowwise=False, columnwise=True)

                # We use the send stream to copy into the userbuffers.
                # This is the same stream that we will use to access the data in the AG,
                # so we dont need to add any syncs yet.
                with torch.cuda.stream(dgrad_send_stream):
                    grad_output, _ = fill_userbuffers_buffer_for_all_gather(
                        ub_obj_overlap_wgrad,
                        args.grad_output,
                        args.grad_output_quantizer,
                        args.tp_group,
                    )

                # Allgather grad_outputs[0] using the dgrad streams so we can overlap with the fc2_dgrad gemm
                tex.bulk_overlap_ag_with_external_gemm(
                    ub_obj_overlap_wgrad, dgrad_send_stream, dgrad_recv_stream
                )

            # Prepare input tensor
            # Note: Synchronize tensor-parallel communication and
            # make sure required data is available
            if ln_out_total_work is not None:
                ln_out_total_work.wait()
                ln_out_total_work = None
            if args.fp8 or args.debug:
                if isinstance(ln_out_total, QuantizedTensorStorage):
                    ln_out_total.update_usage(columnwise_usage=True)
                else:
                    args.input_quantizer.set_usage(rowwise=False, columnwise=True)
                    ln_out_total = args.input_quantizer(ln_out_total)

            if args.fp8 or args.debug:
                if isinstance(grad_output, QuantizedTensorStorage):
                    grad_output.update_usage(columnwise_usage=True)
                else:
                    args.grad_output_quantizer.set_usage(rowwise=False, columnwise=True)
                    grad_output = args.grad_output_quantizer(grad_output)

            # Figure out whether to use split accumulator
            use_split_accumulator = args.wgrad_use_split_accumulator

            # Figure out whether to output wgrad GEMM directly into main grad
            if is_dist_weight:
                # Distributed weight (e.g. GTP): accumulation happens downstream in finalize.
                accumulate_wgrad_into_param_main_grad = False
            elif args.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    args.fuse_wgrad_accumulation and not args.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = args.fuse_wgrad_accumulation

            # Output buffer for overlapping FP8 grad input
            # reduce-scatter with wgrad GEMM
            reduce_scatter_out = None
            if args.ub_bulk_wgrad and ub_obj_wgrad.is_fp8_ubuf():
                reduce_scatter_out = torch.empty(
                    dgrad_shape, dtype=args.activation_dtype, device=args.grad_output.device
                )

            # Arguments to include in wgrad GEMM closure
            wgrad_gemm_kwargs = {
                "out_dtype": (
                    main_grad.dtype if args.fuse_wgrad_accumulation else args.activation_dtype
                ),
                "quantization_params": args.grad_weight_quantizer,
                "accumulate": (
                    accumulate_wgrad_into_param_main_grad
                    if not origin_weight_overwrites_main_grad
                    else False
                ),
                "layout": "NT",
                "out": main_grad if args.fuse_wgrad_accumulation else None,
                "bias": (bias if (grad_bias is None and not args.fp8) else None),
                "use_split_accumulator": use_split_accumulator,
                "grad": True,
                "ub": ub_obj_wgrad,
                "ub_type": ub_type_wgrad,
                "extra_output": reduce_scatter_out,
                "bulk_overlap": args.ub_bulk_wgrad,
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
            if args.wgrad_store is not None and args.wgrad_store.delay_wgrad_compute():
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
                args.wgrad_store.put([ln_out_total, grad_output], wgrad_gemm)
            else:

                # Call wgrad GEMM now
                wgrad, grad_bias_ = wgrad_gemm(ln_out_total, grad_output)

                if is_dist_weight:
                    wgrad = finalize_weight_grads(saved_weight, [wgrad])[0]

                # Update grad bias if needed
                if grad_bias is None:
                    grad_bias = grad_bias_
                del grad_bias_

                # Deallocate input tensors if permitted
                if not args.return_layernorm_output and not args.return_layernorm_output_gathered:
                    # Input tensors have not been exposed externally
                    clear_tensor_data(ln_out)
                elif args.ln_out_needs_gather and args.return_layernorm_output_gathered:
                    # Non-gathered input has not been exposed externally
                    clear_tensor_data(ln_out)
                if args.ln_out_needs_gather:
                    # Gathered input is internal
                    clear_tensor_data(ln_out_total)
                if args.sequence_parallel and (
                    args.parallel_mode == "row" or (args.parallel_mode == "column" and args.fp8)
                ):
                    # Gathered (row-SP) or quantized (column-SP FP8) grad_output is internal
                    clear_tensor_data(grad_output)

            # Update grad input if overlapping reduce-scatter with wgrad GEMM
            if args.ub_bulk_wgrad:
                if ub_obj_wgrad.is_fp8_ubuf():
                    dgrad = reduce_scatter_out
                else:
                    dgrad = ub_obj_wgrad.get_buffer(local_chunk=True).clone()

        # --------------------------------------------------
        # Grad weight has been computed...
        # --------------------------------------------------

        # Don't return grad bias if not needed
        if not args.use_bias:
            grad_bias = None

        # Synchronize tensor parallel communication
        if ln_out_total_work is not None:
            ln_out_total_work.wait()
            ln_out_total_work = None
        if dgrad_work is not None:
            dgrad_work.wait()
            dgrad_work = None

        # Residual gradient
        dgrad = dgrad.view(inputmat.shape)
        if args.return_layernorm_output and not args.return_layernorm_output_gathered:
            dgrad = dgrad + args.grad_ln_out.view_as(dgrad)

        # Norm gradient
        dgamma = None
        dbeta = None
        nvtx_range_push(f"{nvtx_label}.norm")
        if args.normalization == "LayerNorm":
            dgrad, dgamma, dbeta = tex.layernorm_bwd(
                dgrad,
                inputmat,
                mu,
                rsigma,
                ln_weight,
                args.bwd_ln_sm_margin,
                args.zero_centered_gamma,
            )
            dgrad = dgrad.reshape(inputmat.size())
        elif args.normalization == "RMSNorm":
            dgrad, dgamma = tex.rmsnorm_bwd(
                dgrad,
                inputmat,
                rsigma,
                ln_weight,
                args.bwd_ln_sm_margin,
                args.zero_centered_gamma,
            )
            dgrad = dgrad.reshape(inputmat.size())
            dbeta = None
        nvtx_range_pop(f"{nvtx_label}.norm")
        clear_tensor_data(mu)
        clear_tensor_data(rsigma)

    if args.requires_wgrad:
        # Handle custom DDP from mcore.
        if args.fuse_wgrad_accumulation and hasattr(origin_weight, "grad_added_to_main_grad"):
            origin_weight.grad_added_to_main_grad = True
            if getattr(origin_weight, "zero_out_wgrad", False):
                wgrad = get_dummy_wgrad(
                    list(main_grad.shape),
                    origin_weight.dtype,
                    zero=True,
                )
            else:
                wgrad = get_dummy_wgrad(
                    list(main_grad.shape),
                    origin_weight.dtype,
                )
        elif args.fuse_wgrad_accumulation:
            wgrad = None
    else:
        wgrad = None

    return (
        dgrad.view(args.inp_shape) if args.requires_dgrad else None,
        dgamma,
        dbeta,
        wgrad,
        grad_bias,
    )


class _LayerNormLinear(torch.autograd.Function):
    """LayerNormLinear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: Optional[torch.Tensor],
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        fwd_args: LayerNormLinearFwdArgs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Forward pass: compute the output and set up the autograd context.

        The tensors are positional so autograd tracks them; they are
        re-attached to ``fwd_args`` so every downstream helper takes a single
        argument. ``weight_workspace`` is a non-differentiable cached tensor
        passed in via ``fwd_args`` and the freshly produced workspace is
        returned as a separate output so the module can refresh its cache.
        """
        fwd_args.inp = inp
        fwd_args.ln_weight = ln_weight
        fwd_args.ln_bias = ln_bias
        fwd_args.weight = weight
        fwd_args.bias = bias
        (
            out,
            ln_out_return,
            new_weight_workspace,
            tensors_to_save_from_forward,
            ctx_attrs,
        ) = _layernorm_linear_forward_impl(fwd_args)
        if ctx is not None:
            bwd_args = LayerNormLinearBwdArgs()
            tensors_to_save_from_setup = _layernorm_linear_setup_ctx(
                bwd_args,
                fwd_args,
                (out, ln_out_return, new_weight_workspace),
                ctx_attrs,
                tensors_to_save_from_forward,
            )
            tensors_to_save, tensor_objects = prepare_for_saving(*tensors_to_save_from_setup)
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects
            ctx.backward_objects = bwd_args
            if fwd_args.fp8 and fwd_args.any_requires_grad():
                bwd_args.reduce_and_update_bwd_fp8_tensors = check_fp8_reduce_and_update()
            if fwd_args.backward_override is not None:
                bwd_args.reduce_and_update_bwd_fp8_tensors = False

        return out, ln_out_return, new_weight_workspace

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
        grad_ln_out: Optional[torch.Tensor],
        _grad_weight_workspace,
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """Backward pass: compute gradients and reduce FP8 scaling factors."""
        bwd_args: LayerNormLinearBwdArgs = ctx.backward_objects
        bwd_args.grad_output = grad_output
        bwd_args.grad_ln_out = grad_ln_out
        bwd_args.setup_saved_tensors(ctx)
        nvtx_label = "transformer_engine._LayerNormLinear.backward"
        if bwd_args.ub_name is not None:
            nvtx_label = f"{nvtx_label}.{bwd_args.ub_name}"
        dgrad, dgamma, dbeta, wgrad, grad_bias = _layernorm_linear_backward_impl(bwd_args)
        reduce_and_update_bwd_fp8_tensors = bwd_args.reduce_and_update_bwd_fp8_tensors
        # Drop all references held by bwd_args (saved tensors, quantizers, weakrefs,
        # main_grad closure) so they don't outlive backward via ctx under retain_graph.
        ctx.backward_objects = None
        del bwd_args
        if reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            nvtx_range_push(f"{nvtx_label}.reduce_and_update_fp8_tensors")
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
            nvtx_range_pop(f"{nvtx_label}.reduce_and_update_fp8_tensors")
        return (
            dgrad,
            dgamma,
            dbeta,
            wgrad,
            grad_bias,
            None,  # fwd_args
        )


class LayerNormLinear(TransformerEngineBaseModule):
    r"""
    Applies layer normalization followed by linear transformation to the incoming data.

    Parameters
    ----------
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    eps : float, default = 1e-5
         a value added to the denominator of layer normalization for numerical stability.
    bias : bool, default = True
          if set to ``False``, the layer will not learn an additive bias.
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    init_method : Callable, default = None
                 used for initializing weights in the following way: ``init_method(weight)``.
                 When set to ``None``, defaults to ``torch.nn.init.normal_(mean=0.0, std=0.023)``.
    return_layernorm_output : bool, default = False
                             if set to ``True``, output of layernorm is returned from the forward
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module is
                             taken post layernorm.
    return_layernorm_output_gathered : bool, default = False
                             if set to ``True``, output of layernorm is returned after the all
                             gather operation. Ignored if return_layernorm_output is False.
                             Example use case: with sequence parallel, input to residual connection
                             for transformer module (e.g. LoRA) will need to be gathered.
                             Returning layernorm output gathered will prevent a redundant gather.
    parameters_split : Optional[Union[Tuple[str, ...], Dict[str, int]]], default = None
                      Configuration for splitting the weight and bias tensors along dim 0 into
                      multiple PyTorch parameters. If a list or tuple of strings is provided,
                      they are used to make the names of equally-sized parameters. If a dict
                      (preferably an OrderedDict) is provided, the keys are used as names and
                      values as split sizes along dim 0. The resulting parameters will have
                      names that end in ``_weight`` or ``_bias``, so trailing underscores are
                      stripped from any provided names.
    zero_centered_gamma : bool, default = 'False'
                         if set to ``'True'``, gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
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
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        normalization: str = "LayerNorm",
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        return_layernorm_output: bool = False,
        return_layernorm_output_gathered: bool = False,
        parameters_split: Optional[Union[Tuple[str, ...], Dict[str, int]]] = None,
        zero_centered_gamma: bool = False,
        device: Union[torch.device, str] = "cuda",
        ub_overlap_ag: bool = False,
        ub_overlap_rs: bool = False,
        ub_overlap_rs_dgrad: bool = False,
        ub_bulk_wgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_name: Optional[str] = None,
        delay_wgrad_compute: bool = False,
        symmetric_ar_type: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.normalization = normalization
        assert normalization in ["LayerNorm", "RMSNorm"], "Unsupported normalization type!"
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = self.use_bias and not return_bias
        self.return_layernorm_output = return_layernorm_output
        self.return_layernorm_output_gathered = (
            return_layernorm_output_gathered if return_layernorm_output else False
        )
        self.zero_centered_gamma = zero_centered_gamma
        self.symmetric_ar_type = symmetric_ar_type

        self.wgrad_store = WeightGradStore(delay_wgrad_compute, ub_bulk_wgrad)

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
        if self.parallel_mode == "row":
            raise NotImplementedError(
                "Normalization does not support tensor-parallel distribution."
            )

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

        if init_method is None:
            init_method = get_default_init_method()

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        # Column-parallel overlaps
        self.ub_overlap_ag_fprop = (
            ub_overlap_ag and self.sequence_parallel and self.parallel_mode == "column"
        )
        self.ub_overlap_rs_dgrad = (
            ub_overlap_rs_dgrad and self.sequence_parallel and self.parallel_mode == "column"
        )
        # Bulk overlaps require the Userbuffers backend; the cuBLASMp backend
        # falls back to async NCCL ops via torch.distributed.
        self.ub_bulk_wgrad = (
            ub_bulk_wgrad
            and self.sequence_parallel
            and self.parallel_mode == "column"
            and not self.ub_overlap_rs_dgrad
        )
        self.ub_bulk_dgrad = (
            ub_bulk_dgrad
            and self.sequence_parallel
            and self.parallel_mode == "column"
            and not self.ub_overlap_rs_dgrad
        )

        # Row-parallel overlaps
        self.ub_overlap_rs_fprop = (
            ub_overlap_rs and self.sequence_parallel and self.parallel_mode == "row"
        )
        self.ub_overlap_ag_dgrad = (
            ub_overlap_ag and self.sequence_parallel and self.parallel_mode == "row"
        )
        if any(
            [
                self.ub_overlap_ag_fprop,
                self.ub_overlap_rs_dgrad,
                self.ub_bulk_dgrad,
                self.ub_bulk_wgrad,
                self.ub_overlap_rs_fprop,
                self.ub_overlap_ag_dgrad,
            ]
        ):
            assert is_ub_initialized(), "initialize_ub() must be called before layer construction."
            assert ub_name is not None, "Userbuffer name [string] is not set."
        self.ub_name = ub_name

        if using_cublasmp_backend():
            if self.ub_bulk_dgrad:
                warnings.warn(
                    f"cuBLASMp backend does not support bulk overlaps for '{self.ub_name}_dgrad' "
                    f"and '{self.ub_name}_wgrad' GEMMs. Falling back on DGRAD+RS overlap for "
                    f"'{self.ub_name}_dgrad' GEMM with no bulk overlap for '{self.ub_name}_wgrad' "
                    "GEMM. In order to enable bulk overlaps for these GEMMs, set "
                    "`with_cublasmp=False` when calling `initialize_ub()`."
                )
            self.ub_overlap_rs_dgrad = self.ub_overlap_rs_dgrad or self.ub_bulk_dgrad
            self.ub_bulk_dgrad = False
            self.ub_bulk_wgrad = False

        if self.symmetric_ar_type is not None:
            assert torch_version() >= (
                2,
                7,
                0,
            ), "Torch version must be at least 2.7 to use symmetric memory"

        self.eps = eps
        layer_norm_weight = torch.nn.Parameter(
            torch.empty(self.in_features, device=device, dtype=params_dtype)
        )
        self.register_parameter(
            "layer_norm_weight",
            layer_norm_weight,
            init_fn=init_method_constant(float(not self.zero_centered_gamma)),
        )
        if self.normalization != "RMSNorm":
            layer_norm_bias = torch.nn.Parameter(
                torch.empty(self.in_features, device=device, dtype=params_dtype)
            )
            self.register_parameter(
                "layer_norm_bias", layer_norm_bias, init_fn=init_method_constant(0.0)
            )
        else:
            self.layer_norm_bias = None

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
        # each other in LayerNormLinear.parameters(). This makes it
        # more likely that they will stay contiguous if the weights
        # are manipulated externally, e.g. by FSDP.
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

        # These many SMs are subtracted from the total SM count when calling forward
        # and backward LayerNorm C APIs. These envvars can be used to prevent the LN
        # kernels from using all SMs in the device. This is useful for cases such as
        # communication overlap with LN.
        self.fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
        self.bwd_ln_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))
        self.inf_ln_sm_margin = int(os.getenv("NVTE_INF_LAYERNORM_SM_MARGIN", "0"))

        if self.wgrad_store.delay_wgrad_compute():
            for name, param in self.named_parameters():
                if name in self.weight_names or name in self.bias_names:
                    param.skip_backward_post_hook = True

    def set_meta_tensor(self, fwd: bool, recipe: Recipe) -> None:
        """Init scales and amaxes for fwd | bwd."""
        super().set_meta_tensor(fwd, recipe)

        # Recipe-specific quantizer configuration
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        if recipe.float8_current_scaling():
            self._customize_quantizers_float8_current_scaling(fwd, recipe)

    def get_quantizer_roles(
        self,
        *,
        fwd: bool,
        num_quantizers: int,
    ) -> Optional[List[QuantizerRole]]:
        """QuantizerRole list for quantizers used by ``LayerNormLinear``.

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

    def reset_layer_norm_parameters(self) -> None:
        """Init LN params"""
        warnings.warn(
            "This method will be deprecated in an upcoming release. "
            "Update your code to use LayerNormLinear.reset_parameters() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.zero_centered_gamma:
            init.ones_(self.layer_norm_weight)
        else:
            init.zeros_(self.layer_norm_weight)
        if self.layer_norm_bias is not None:
            init.zeros_(self.layer_norm_bias)

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
            # Set parallelism attributes for layer norm parameters
            setattr(self.layer_norm_weight, "sequence_parallel", self.sequence_parallel)
            if self.normalization != "RMSNorm":
                setattr(self.layer_norm_bias, "sequence_parallel", self.sequence_parallel)

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
        Apply layer normalization to the input followed by a linear transformation.

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
            if get_ub_is_fp8(self.ub_name + "_fprop", FP8GlobalStateManager.is_fp8_enabled()):
                fp8_output = True
        if self.ub_overlap_rs_dgrad:
            if get_ub_is_fp8(self.ub_name + "_dgrad", FP8GlobalStateManager.is_fp8_enabled()):
                fp8_grad = True

        inp = self.prepare_forward(
            inp, allow_non_contiguous=False  # removed .contiguous from inside the layer
        )

        try:
            # Get concatenated weight and bias tensors
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
            if weight_quantizer is not None and not debug:
                weight_quantizer.optimize_for_gemm = self._enable_weight_preswizzle(
                    weight_quantizer, weight_tensor
                )

            cache_name = None if (is_first_microbatch is None or self.is_fsdp2) else "weight"
            weight_workspace = (
                self._fp8_workspaces.get(cache_name) if cache_name is not None else None
            )

            dgrad_use_split_accumulator = _2X_ACC_DGRAD
            wgrad_use_split_accumulator = _2X_ACC_WGRAD
            if self.fp8:
                _recipe = FP8GlobalStateManager.get_fp8_recipe()
                backward_override = _recipe.backward_override
                if hasattr(_recipe, "fp8_gemm_dgrad"):
                    dgrad_use_split_accumulator = _recipe.fp8_gemm_dgrad.use_split_accumulator
                if hasattr(_recipe, "fp8_gemm_wgrad"):
                    wgrad_use_split_accumulator = _recipe.fp8_gemm_wgrad.use_split_accumulator
            else:
                backward_override = None

            if debug:  # turn off userbuffers in debug mode
                ub_overlap_ag_fprop = False
                ub_overlap_rs_fprop = False
                ub_overlap_ag_dgrad = False
                ub_overlap_rs_dgrad = False
                ub_bulk_wgrad = False
                ub_bulk_dgrad = False
            else:
                ub_overlap_ag_fprop = self.ub_overlap_ag_fprop
                ub_overlap_rs_fprop = self.ub_overlap_rs_fprop
                ub_overlap_ag_dgrad = self.ub_overlap_ag_dgrad
                ub_overlap_rs_dgrad = self.ub_overlap_rs_dgrad
                ub_bulk_wgrad = self.ub_bulk_wgrad
                ub_bulk_dgrad = self.ub_bulk_dgrad

            linear_bias_tensor = (
                bias_tensor if (self.apply_bias and not self.gemm_bias_unfused_add) else None
            )
            wgrad_store = self.wgrad_store if self.wgrad_store.delay_wgrad_compute() else None

            fwd_args = LayerNormLinearFwdArgs(
                # tensors
                inp=inp,
                ln_weight=self.layer_norm_weight,
                ln_bias=self.layer_norm_bias,
                weight=weight_tensor,
                bias=linear_bias_tensor,
                weight_workspace=weight_workspace,
                # requires_grad flags
                input_requires_grad=inp.requires_grad,
                ln_weight_requires_grad=self.layer_norm_weight.requires_grad,
                ln_bias_requires_grad=(
                    self.layer_norm_bias.requires_grad
                    if self.layer_norm_bias is not None
                    else False
                ),
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
                # normalization
                eps=self.eps,
                normalization=self.normalization,
                zero_centered_gamma=self.zero_centered_gamma,
                fwd_ln_sm_margin=(
                    self.fwd_ln_sm_margin if is_grad_enabled else self.inf_ln_sm_margin
                ),
                bwd_ln_sm_margin=self.bwd_ln_sm_margin,
                return_layernorm_output=self.return_layernorm_output,
                return_layernorm_output_gathered=self.return_layernorm_output_gathered,
                # numerical / dtype config
                activation_dtype=self.activation_dtype,
                fp8=self.fp8,
                fp8_calibration=self.fp8_calibration,
                backward_override=backward_override,
                dgrad_use_split_accumulator=dgrad_use_split_accumulator,
                wgrad_use_split_accumulator=wgrad_use_split_accumulator,
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

            if is_grad_enabled:
                out, ln_out, new_weight_workspace = _LayerNormLinear.apply(
                    inp,
                    self.layer_norm_weight,
                    self.layer_norm_bias,
                    weight_tensor,
                    linear_bias_tensor,
                    fwd_args,
                )
            else:
                out, ln_out, new_weight_workspace = _LayerNormLinear.forward(
                    None,
                    inp,
                    self.layer_norm_weight,
                    self.layer_norm_bias,
                    weight_tensor,
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
            if self.return_layernorm_output:
                return out, cast_if_needed(bias_tensor, self.activation_dtype), ln_out
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        if self.return_layernorm_output:
            return out, ln_out
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

    def _get_weight_and_bias_tensors(self):
        # Get concatenated weight and bias tensors
        unfused_weights = self._get_weight_tensors()

        weight_tensor = noop_cat(unfused_weights)
        if self.use_bias:
            bias_tensor = noop_cat([getattr(self, name) for name in self.bias_names])
        else:
            bias_tensor = getattr(self, self.bias_names[0])  # Unused
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
        from ..export import onnx_layernorm, onnx_gemm

        assert not TEDebugState.debug_enabled, "Debug mode is not supported in ONNX export"
        assert_warmed_up(self)
        (
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            *_,
        ) = self._get_quantizers(fp8_output, False, is_grad_enabled)
        inp_dtype = inp.dtype

        weight_tensor, bias_tensor = self._get_weight_and_bias_tensors()
        ln_out, ln_out_return = onnx_layernorm(
            inp,
            self.layer_norm_weight,
            self.layer_norm_bias,
            self.eps,
            self.normalization,
            self.zero_centered_gamma,
            inp_dtype,
            self.return_layernorm_output,
            input_quantizer,
        )

        if weight_quantizer is not None:
            weight_tensor_quantized = weight_quantizer.onnx_quantize(weight_tensor)
            weight_tensor = weight_quantizer.onnx_dequantize(weight_tensor_quantized)
        weight_tensor = weight_tensor.to(inp_dtype)

        if bias_tensor is not None:
            bias_tensor = bias_tensor.to(inp_dtype)

        output = onnx_gemm(weight_tensor, ln_out, bias_tensor if self.apply_bias else None)

        if output_quantizer is not None:
            raise NotImplementedError("ONNX export of quantized output is not supported")
        if self.return_layernorm_output and self.return_bias:
            return output, bias_tensor.to(inp_dtype), ln_out_return
        if self.return_layernorm_output:
            return output, ln_out_return
        if self.return_bias:
            return output, bias_tensor.to(inp_dtype)
        return output

    def _customize_quantizers_float8_current_scaling(self, fwd: bool, recipe: Recipe) -> None:
        """Customize quantizers based on current scaling recipe + layernorm_linear."""
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
        else:
            # set grad_output_quantizer with amax epsilon and power_2_scale (no amax reduction here)
            self.quantizers["scaling_bwd"][
                FP8BwdTensorIdx.GRAD_OUTPUT1
            ].force_pow_2_scales = recipe.fp8_quant_bwd_grad.power_2_scale
            self.quantizers["scaling_bwd"][
                FP8BwdTensorIdx.GRAD_OUTPUT1
            ].amax_epsilon = recipe.fp8_quant_bwd_grad.amax_epsilon

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

    def _get_weight_quantizers(self) -> List[Quantizer]:
        """Get the weight quantizers of the module."""
        if not self.fp8 and not self.fp8_calibration:
            return [None]
        weight_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_WEIGHT]
        weight_quantizer.internal = True
        return [weight_quantizer]

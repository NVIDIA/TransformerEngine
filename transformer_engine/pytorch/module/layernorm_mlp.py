# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LayerNormMLP API"""
import os
import warnings
from dataclasses import dataclass, replace as dataclass_replace
import weakref
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, List
from functools import reduce
from operator import mul as multiply_op

import torch
from torch.nn.parameter import Parameter
from torch.nn import init

import transformer_engine_torch as tex

from transformer_engine.common.recipe import Recipe
from transformer_engine.pytorch.torch_version import torch_version
from transformer_engine.pytorch.tensor.utils import clear_columnwise_cache, is_custom
from .base import (
    fill_userbuffers_buffer_for_all_gather,
    _ub_communicators,
    get_ub,
    get_ub_is_fp8,
    is_ub_initialized,
    using_cublasmp_backend,
    quantize_weight,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ..quantization import FP8GlobalStateManager, QuantizerRole
from ..jit import (
    bias_gelu_fused,
    bgrad_dgelu_fused,
    set_jit_fusion_options,
    warmup_jit_bias_gelu_all_dtypes,
)
from ..utils import (
    divide,
    get_default_init_method,
    init_method_constant,
    cast_if_needed,
    assert_dim_for_fp8_exec,
    clear_tensor_data,
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
    use_reentrant_activation_recompute,
    _fsdp_scatter_tensors,
    _get_cuda_rng_state,
    _set_cuda_rng_state,
)
from ..constants import FP8BwdTensorIdx, FP8FwdTensorIdx, dist_group_type
from ..jit import no_torch_dynamo
from ..graph import is_graph_capturing
from ..tensor.float8_tensor import Float8Tensor
from ..tensor.mxfp8_tensor import MXFP8Quantizer
from ..tensor.nvfp4_tensor import NVFP4Quantizer
from ..tensor.float8_blockwise_tensor import Float8BlockQuantizer
from ..tensor.hybrid_tensor import HybridQuantizer
from ..tensor.identity_tensor import IdentityQuantizer
from ._common import (
    apply_normalization,
    check_fp8_reduce_and_update,
    set_quantizer_amax_reduction_group,
    set_quantizer_usage_for_wgrad_all_gather,
    WeightGradStore,
)
from ..cpu_offload import (
    is_cpu_offload_enabled,
    start_offload,
    mark_not_offload,
    mark_activation_offload,
)
from ..quantized_tensor import (
    QuantizedTensorStorage,
    Quantizer,
    prepare_for_saving,
    restore_from_func_ctx,
)
from ..dynamo import TensorOrQuantized
from ..cpp_extensions import (
    general_gemm,
)
from ..export import is_in_onnx_export_mode, assert_warmed_up
from ...debug.pytorch.debug_state import TEDebugState

__all__ = ["LayerNormMLP"]


def _get_act_func_supported_list(recipe: Optional[Recipe] = None):
    if recipe is None:
        # bf16 (recipe is None):
        return {
            "gelu": (tex.gelu, tex.dgelu, None),
            "geglu": (tex.geglu, tex.dgeglu, None),
            "glu": (tex.glu, tex.dglu, None),
            "qgelu": (tex.qgelu, tex.dqgelu, None),
            "qgeglu": (tex.qgeglu, tex.dqgeglu, None),
            "relu": (tex.relu, tex.drelu, None),
            "reglu": (tex.reglu, tex.dreglu, None),
            "srelu": (tex.srelu, tex.dsrelu, None),
            "sreglu": (tex.sreglu, tex.dsreglu, None),
            "silu": (tex.silu, tex.dsilu, None),
            "swiglu": (tex.swiglu, tex.dswiglu, None),
            "clamped_swiglu": (tex.clamped_swiglu, tex.clamped_dswiglu, None),
        }
    if recipe.delayed() or recipe.mxfp8():
        # Delayed scaling, fusion supported list: [tex.dbias_dgelu, tex.dbias_drelu, tex.dbias_dqgelu, tex.dbias_dsrelu]
        # MXFP8: [tex.dbias_dgelu, tex.dbias_drelu, tex.dbias_dqgelu, tex.dbias_dsrelu]
        return {
            "gelu": (tex.gelu, tex.dgelu, tex.dbias_dgelu),
            "geglu": (tex.geglu, tex.dgeglu, None),
            "glu": (tex.glu, tex.dglu, None),
            "qgelu": (tex.qgelu, tex.dqgelu, tex.dbias_dqgelu),
            "qgeglu": (tex.qgeglu, tex.dqgeglu, None),
            "relu": (tex.relu, tex.drelu, tex.dbias_drelu),
            "reglu": (tex.reglu, tex.dreglu, None),
            "srelu": (tex.srelu, tex.dsrelu, tex.dbias_dsrelu),
            "sreglu": (tex.sreglu, tex.dsreglu, None),
            "silu": (tex.silu, tex.dsilu, tex.dbias_dsilu),
            "swiglu": (tex.swiglu, tex.dswiglu, None),
            "clamped_swiglu": (tex.clamped_swiglu, tex.clamped_dswiglu, None),
        }
    # no activation fusion written yet
    # Per-tensor current scaling or fp8 blockwise scaling or custom quantization: []
    # TODO(ksivaman): Fuse nvfp4 act once kernel is available.
    if (
        recipe.float8_current_scaling()
        or recipe.float8_block_scaling()
        or recipe.nvfp4()
        or recipe.custom()
    ):
        return {
            "gelu": (tex.gelu, tex.dgelu, None),
            "geglu": (tex.geglu, tex.dgeglu, None),
            "glu": (tex.glu, tex.dglu, None),
            "qgelu": (tex.qgelu, tex.dqgelu, None),
            "qgeglu": (tex.qgeglu, tex.dqgeglu, None),
            "relu": (tex.relu, tex.drelu, None),
            "reglu": (tex.reglu, tex.dreglu, None),
            "srelu": (tex.srelu, tex.dsrelu, None),
            "sreglu": (tex.sreglu, tex.dsreglu, None),
            "silu": (tex.silu, tex.dsilu, None),
            "swiglu": (tex.swiglu, tex.dswiglu, None),
            "clamped_swiglu": (tex.clamped_swiglu, tex.clamped_dswiglu, None),
        }
    raise NotImplementedError(f"Unhandled recipe type {recipe}")


def _act_func(activation: str, recipe: Optional[Recipe] = None):
    # based on each quantization mode, we have different kernel fusion supported:
    # bf16 (recipe is None): [tex.dbias_dgelu, tex.dbias_drelu, tex.dbias_dqgelu, tex.dbias_dsrelu]
    # Delayed scaling, fusion supported list: [tex.dbias_dgelu, tex.dbias_drelu, tex.dbias_dqgelu, tex.dbias_dsrelu]
    # MXFP8: [tex.dbias_dgelu, tex.dbias_drelu, tex.dbias_dqgelu, tex.dbias_dsrelu]
    # Per-tensor current scaling or fp8 blockwise scaling: []
    funcs = _get_act_func_supported_list(recipe)
    if activation not in funcs:
        raise NotImplementedError("Activation type " + activation + " is not supported!")
    return funcs[activation]


@dataclass(slots=True)
class LayerNormMLPFwdArgs:
    """Single-argument bag for the forward path of :class:`_LayerNormMLP`."""

    # --- Differentiable tensors (also passed positionally to autograd) ---
    inp: torch.Tensor
    ln_weight: torch.Tensor
    ln_bias: Optional[torch.Tensor]
    fc1_weight: TensorOrQuantized
    fc1_bias: Optional[torch.Tensor]
    fc2_weight: TensorOrQuantized
    fc2_bias: Optional[torch.Tensor]

    # --- Non-differentiable cached tensors ---
    fc1_weight_workspace: Optional[TensorOrQuantized]
    fc2_weight_workspace: Optional[TensorOrQuantized]

    # --- requires_grad flags (cached so backward does not re-query) ---
    input_requires_grad: bool
    ln_weight_requires_grad: bool
    ln_bias_requires_grad: bool
    fc1_weight_requires_grad: bool
    fc1_bias_requires_grad: bool
    fc2_weight_requires_grad: bool
    fc2_bias_requires_grad: bool

    # --- Quantizers ---
    fc1_input_quantizer: Optional[Quantizer]
    fc1_weight_quantizer: Optional[Quantizer]
    fc1_output_quantizer: Optional[Quantizer]
    fc1_grad_input_quantizer: Optional[Quantizer]
    fc1_grad_weight_quantizer: Optional[Quantizer]
    fc1_grad_output_quantizer: Optional[Quantizer]
    fc2_input_quantizer: Optional[Quantizer]
    fc2_weight_quantizer: Optional[Quantizer]
    fc2_output_quantizer: Optional[Quantizer]
    fc2_grad_input_quantizer: Optional[Quantizer]
    fc2_grad_weight_quantizer: Optional[Quantizer]
    fc2_grad_output_quantizer: Optional[Quantizer]

    # --- Normalization ---
    eps: float
    normalization: str
    zero_centered_gamma: bool
    fwd_ln_sm_margin: int
    bwd_ln_sm_margin: int
    return_layernorm_output: bool
    return_layernorm_output_gathered: bool

    # --- Activation ---
    activation: str
    activation_params: Optional[Dict[str, Any]]
    bias_gelu_fusion: bool
    gemm_gelu_fusion: bool

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
    set_parallel_mode: bool
    tp_group: Optional[dist_group_type]
    tp_size: int
    tensor_parallel: bool
    sequence_parallel: bool
    symmetric_ar_type: Optional[str]

    # --- Userbuffers (comm + GEMM overlap) ---
    ub_overlap_ag: bool
    ub_overlap_rs: bool
    ub_overlap_rs_dgrad: bool
    ub_bulk_dgrad: bool
    ub_bulk_wgrad: bool

    # --- FSDP ---
    fsdp_group: Optional[Any]
    is_fsdp2: bool

    # --- Weight-grad scheduling ---
    fuse_wgrad_accumulation: bool
    wgrad_store: Optional[Any]

    # --- Activation checkpointing (recompute in backward) ---
    checkpoint: bool
    fp8_meta: Optional[Any]
    recompute_for_bwd: bool

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
                self.fc1_weight_requires_grad,
                self.fc1_bias_requires_grad,
                self.fc2_weight_requires_grad,
                self.fc2_bias_requires_grad,
            )
        )


@dataclass(slots=True)
class LayerNormMLPBwdArgs:
    """Single-argument bag for the backward path of :class:`_LayerNormMLP`."""

    # --- Incoming gradients (populated at backward entry) ---
    grad_output: Optional[torch.Tensor] = None
    grad_ln_out: Optional[torch.Tensor] = None

    # --- Saved / restored tensors (populated at backward entry) ---
    inputmat: Optional[torch.Tensor] = None
    ln_weight: Optional[torch.Tensor] = None
    ln_out: Optional[TensorOrQuantized] = None
    fc1_weight_fp8: Optional[TensorOrQuantized] = None
    fc1_weight: Optional[TensorOrQuantized] = None
    fc1_bias: Optional[torch.Tensor] = None
    fc1_out: Optional[TensorOrQuantized] = None
    fc1_out_without_bias: Optional[torch.Tensor] = None
    act_out: Optional[TensorOrQuantized] = None
    fc2_weight_fp8: Optional[TensorOrQuantized] = None
    fc2_weight: Optional[TensorOrQuantized] = None
    fc2_bias: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    rsigma: Optional[torch.Tensor] = None

    # --- Activation checkpointing (forward inputs saved, recomputed in backward) ---
    checkpoint: bool = False
    checkpoint_fwd_args: Optional[Any] = None
    autocast_state: Optional[Any] = None
    cpu_rng_state: Optional[Any] = None
    cuda_rng_state: Optional[Any] = None

    # --- Quantizers ---
    fc1_input_quantizer: Optional[Quantizer] = None
    fc1_weight_quantizer: Optional[Quantizer] = None
    fc1_grad_input_quantizer: Optional[Quantizer] = None
    fc1_grad_weight_quantizer: Optional[Quantizer] = None
    fc1_grad_output_quantizer: Optional[Quantizer] = None
    fc2_input_quantizer: Optional[Quantizer] = None
    fc2_weight_quantizer: Optional[Quantizer] = None
    fc2_grad_input_quantizer: Optional[Quantizer] = None
    fc2_grad_weight_quantizer: Optional[Quantizer] = None
    fc2_grad_output_quantizer: Optional[Quantizer] = None

    # --- Differentiability summary ---
    use_bias: bool = False
    requires_dgrad: bool = False
    fc1_weight_requires_grad: bool = False
    fc1_bias_requires_grad: bool = False
    fc2_weight_requires_grad: bool = False
    inp_shape: Optional[torch.Size] = None

    # --- Normalization ---
    normalization: str = "LayerNorm"
    zero_centered_gamma: bool = False
    bwd_ln_sm_margin: int = 0
    return_layernorm_output: bool = False
    return_layernorm_output_gathered: bool = False

    # --- Activation ---
    activation: str = "gelu"
    activation_params: Optional[Dict[str, Any]] = None
    bias_gelu_fusion: bool = False

    # --- Numerical / dtype config ---
    activation_dtype: Optional[torch.dtype] = None
    fp8: bool = False
    fp8_recipe: Optional[Any] = None
    dgrad_use_split_accumulator: bool = _2X_ACC_DGRAD
    wgrad_use_split_accumulator: bool = _2X_ACC_WGRAD
    backward_override: Optional[str] = None
    debug: bool = False

    # --- Tensor / sequence parallelism ---
    set_parallel_mode: bool = False
    tp_group: Optional[dist_group_type] = None
    tp_size: int = 1
    tensor_parallel: bool = False
    sequence_parallel: bool = False

    # --- Userbuffers (comm + GEMM overlap) ---
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
    fc1_weight_ref: Optional[Any] = None
    fc2_weight_ref: Optional[Any] = None
    fc1_weight_overwrites_main_grad: bool = False
    fc2_weight_overwrites_main_grad: bool = False
    fc1_main_grad_func: Optional[Callable[[], torch.Tensor]] = None
    fc2_main_grad_func: Optional[Callable[[], torch.Tensor]] = None

    # --- FP8 reduce-and-update bookkeeping ---
    reduce_and_update_bwd_fp8_tensors: bool = False

    # --- Misc ---
    cpu_offloading: bool = False

    # --- Per-backward scratch state (populated inside the backward impl) ---
    ub_obj_gradout: Optional[Any] = None

    def setup_saved_tensors(self, ctx: torch.autograd.function.FunctionCtx) -> None:
        """Pull saved tensors from ``ctx`` into the fields backward consumes."""
        self.set_saved_tensors(restore_from_func_ctx(ctx))

    def set_saved_tensors(self, tensors: Sequence[Any]) -> None:
        """Bind the (restored or recomputed) saved tensors to their fields."""
        (
            self.inputmat,
            self.ln_weight,
            self.ln_out,
            self.fc1_weight_fp8,
            self.fc1_weight,
            self.fc1_bias,
            self.fc1_out,
            self.fc1_out_without_bias,
            self.act_out,
            self.fc2_weight_fp8,
            self.fc2_weight,
            self.fc2_bias,
            self.mu,
            self.rsigma,
        ) = tensors


_CHECKPOINT_SAVED_ALIASES = (
    "inp",
    "ln_weight",
    "ln_bias",
    "fc1_weight",
    "fc1_bias",
    "fc2_weight",
    "fc2_bias",
)


def _layernorm_mlp_forward_impl(
    args: LayerNormMLPFwdArgs,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[Tuple],
    Optional[Dict],
]:
    """Forward implementation for the layernorm-MLP layer.

    Returns ``(fc2_out, ln_out_return, new_fc1_weight_workspace,
    new_fc2_weight_workspace, tensors_to_save_from_forward, ctx_attrs)``. The
    new workspaces are the freshly produced FP8 weight workspaces (returned so
    the caller can refresh its cache). The last two are ``None`` when
    gradients are disabled.

    With ``args.checkpoint`` the forward saves only its inputs and the
    backward recomputes the activations by calling this again with
    ``args.recompute_for_bwd``; that call returns ``None`` user outputs and
    the recomputed saved tensors.
    """
    inp = args.inp
    ln_weight = args.ln_weight
    ln_bias = args.ln_bias
    fc1_weight = args.fc1_weight
    fc1_bias = args.fc1_bias
    fc2_weight = args.fc2_weight
    fc2_bias = args.fc2_bias
    fc1_input_quantizer = args.fc1_input_quantizer
    fc1_weight_quantizer = args.fc1_weight_quantizer
    fc1_output_quantizer = args.fc1_output_quantizer
    fc2_input_quantizer = args.fc2_input_quantizer
    fc2_weight_quantizer = args.fc2_weight_quantizer
    fc2_output_quantizer = args.fc2_output_quantizer
    is_first_microbatch = args.is_first_microbatch
    fp8 = args.fp8
    fp8_calibration = args.fp8_calibration
    debug = args.debug
    cpu_offloading = args.cpu_offloading
    tp_group = args.tp_group
    tp_size = args.tp_size
    sequence_parallel = args.sequence_parallel
    tensor_parallel = args.tensor_parallel
    set_parallel_mode = args.set_parallel_mode
    activation_dtype = args.activation_dtype
    is_grad_enabled = args.is_grad_enabled
    return_layernorm_output = args.return_layernorm_output
    return_layernorm_output_gathered = args.return_layernorm_output_gathered
    activation = args.activation
    bias_gelu_fusion = args.bias_gelu_fusion
    gemm_gelu_fusion = args.gemm_gelu_fusion
    ub_overlap_ag = args.ub_overlap_ag
    ub_overlap_rs = args.ub_overlap_rs
    fsdp_group = args.fsdp_group
    is_fsdp2 = args.is_fsdp2
    checkpoint = args.checkpoint
    recompute_for_bwd = args.recompute_for_bwd
    fc1_weight_requires_grad = args.fc1_weight_requires_grad
    fc2_weight_requires_grad = args.fc2_weight_requires_grad

    assert args.backward_override is None, (
        "NVTE_BACKWARD_OVERRIDE=high_precision/dequantized is not implemented in LayerNormMLP."
        " Replace LayerNormMLP with LayerNormLinear + Linear to enable"
        " high_precision/dequantized backward."
    )

    ctx_attrs = None
    # if grad is enabled and this is not the bwd stage, we must save this so bwd knows which path to take
    if is_grad_enabled and not recompute_for_bwd:
        ctx_attrs = {"checkpoint": checkpoint}
        if checkpoint:
            # save the state of autocast and quantizers for recomputation
            ctx_attrs["autocast_state"] = FP8GlobalStateManager.get_autocast_state()
            if (
                fp8
                and FP8GlobalStateManager.get_fp8_recipe().__class__.__name__ == "DelayedScaling"
            ):  # only applicable for delayed scaling
                FP8GlobalStateManager.copy_forward_fp8_meta_tensors_for_recompute(
                    args.fp8_meta
                )  # to restore quantizers during recomputation
            # save the rng states
            ctx_attrs["cpu_rng_state"] = torch.get_rng_state()
            ctx_attrs["cuda_rng_state"] = _get_cuda_rng_state()

    # whether to save activations regularly, or save inputs for recomputation in bwd
    save_for_checkpoint = checkpoint and is_grad_enabled and not recompute_for_bwd

    # whether we are in the forward stage, or recomputing in the bwd stage (false if not checkpointing)
    is_recomputation = checkpoint and is_grad_enabled and recompute_for_bwd

    tensors_to_save_from_forward = None
    # save the initial state for recomputation by bwd
    if save_for_checkpoint:
        tensors_to_save_from_forward = (None,) * len(_CHECKPOINT_SAVED_ALIASES)
        ctx_attrs["saved_tensor_aliases"] = _CHECKPOINT_SAVED_ALIASES

    # Make sure input dimensions are compatible
    in_features, inp_shape = ln_weight.numel(), inp.shape
    assert inp_shape[-1] == in_features, "GEMM not possible"
    inp = inp.view((-1, in_features))
    inputmat = inp
    if fp8:
        assert_dim_for_fp8_exec(inputmat, fc1_weight, fc2_weight)

    activation_func = _act_func(
        activation, FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
    )[0]

    # Cast for native AMP
    inputmat = cast_if_needed(inputmat, activation_dtype)
    ln_weight_cast = cast_if_needed(ln_weight, activation_dtype)
    if ln_bias is not None:
        ln_bias = cast_if_needed(ln_bias, activation_dtype)
    if is_cpu_offload_enabled():
        start_offload(inputmat)

    tp_world_size = get_distributed_world_size(tp_group)

    # bwd needs fc1 input when grad is enabled, fc1 needs grad, and either
    # 1) no checkpointing
    # or 2) doing the recomputation with checkpointing
    backwards_needs_fc1_input = fc1_weight_requires_grad and (
        (is_grad_enabled and not checkpoint) or is_recomputation
    )

    device = inp.device

    # Configure Userbuffers communication (comm+GEMM overlap)
    ub_overlap_ag = ub_overlap_ag and is_grad_enabled and not return_layernorm_output_gathered
    ub_overlap_rs = ub_overlap_rs and is_grad_enabled

    # Choose whether to use GEMM kernel with split accumulator
    use_split_accumulator = _2X_ACC_FPROP
    if fp8:
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        if hasattr(recipe, "fp8_gemm_fprop"):
            use_split_accumulator = recipe.fp8_gemm_fprop.use_split_accumulator

    # Configure quantizer for norm output
    if fp8:
        if fc1_input_quantizer is None:
            raise ValueError("Missing quantizer for FC1 input tensor")
        fc1_input_quantizer.set_usage(rowwise=True, columnwise=backwards_needs_fc1_input)
        if sequence_parallel and fc1_input_quantizer.supports_only_rowwise_all_gather():
            # All-gather is not supported with FP8 column-wise data
            fc1_input_quantizer.set_usage(columnwise=False)
        # Amax reduction group for the FC1 input quantizer (column-parallel sequence parallel)
        set_quantizer_amax_reduction_group(
            fc1_input_quantizer,
            tp_group if (sequence_parallel and set_parallel_mode) else None,
        )

    # for fp8 DelayedScaling: layernorm output = FP8
    #                   only output of the linear is returned
    # for return_layernorm_output: layernorm output = High precision, then cast to FP8
    #                              high precision layernorm output and output of the linear are returned
    # for debug: : layernorm output = High precision to enable processing of this norm

    custom = is_custom(fc1_input_quantizer)
    hybrid = isinstance(fc1_input_quantizer, HybridQuantizer)
    identity = isinstance(fc1_input_quantizer, IdentityQuantizer)
    with_quantized_norm = (
        fp8
        and not debug
        and not return_layernorm_output
        and not return_layernorm_output_gathered
        and not custom
        and not hybrid
        and not identity
    )

    # Apply normalization
    ln_out, mu, rsigma = apply_normalization(
        inputmat,
        None,  # ln_out
        ln_weight_cast,
        ln_bias,
        args.eps,
        fc1_input_quantizer if with_quantized_norm else None,
        inputmat.dtype,
        args.normalization,
        args.fwd_ln_sm_margin,
        args.zero_centered_gamma,
    )
    ln_out_return = None

    # do not return layernorm output unless 1) no checkpointing or 2) checkpointing but not recomputing
    if (return_layernorm_output or return_layernorm_output_gathered) and not is_recomputation:
        ln_out_return = ln_out

    # Prepare GEMM input
    # Note: Cast to expected dtype and perform tensor-parallel communication
    ln_out_total = None
    ub_obj_lnout = None
    if sequence_parallel:

        # do not return ln output if checkpointing and in recomputation, not necessary
        if return_layernorm_output_gathered and not is_recomputation:
            # Perform all-gather in high precision if gathered
            # norm output will be returned
            ln_out_total, _ = gather_along_first_dim(ln_out, tp_group)
            ln_out_return = ln_out_total
            if fp8 or debug:
                ln_out = fc1_input_quantizer(ln_out)
                fc1_input_quantizer.set_usage(rowwise=True, columnwise=False)
                ln_out_total = fc1_input_quantizer(ln_out_total)
        else:
            quantizer = None
            if fp8 or debug:
                quantizer = fc1_input_quantizer
                # custom recipe doesn't need to support quantized AG
                if not with_quantized_norm and not custom:
                    ln_out = fc1_input_quantizer(ln_out)
                fc1_input_quantizer.set_usage(rowwise=True, columnwise=False)
            if ub_overlap_ag:
                # Copy into Userbuffers buffer
                ub_obj_lnout = get_ub("fc1_fprop", fp8)
                ln_out_total, _ = fill_userbuffers_buffer_for_all_gather(
                    ub_obj_lnout,
                    ln_out,
                    quantizer,
                    tp_group,
                )
            else:
                # All-gather with NCCL
                ln_out_total, _ = gather_along_first_dim(
                    ln_out,
                    tp_group,
                    quantizer=quantizer,
                )
    else:
        if (fp8 or debug) and not with_quantized_norm:
            ln_out = fc1_input_quantizer(ln_out)
        ln_out_total = ln_out

    # Cast weights to expected dtype
    new_fc1_weight_workspace = None
    new_fc2_weight_workspace = None
    fc1_weight_final = fc1_weight
    fc2_weight_final = fc2_weight
    # FSDP2: Skip columnwise/transpose creation during forward (not
    # recompute) to avoid accumulating FP8 caches across layers.
    # Backward's FSDP2 all-gather will recreate them. (Issue #2681)
    fsdp2_skip_columnwise = is_fsdp2 and not is_recomputation
    if fp8 or debug:
        update_ws = is_first_microbatch is None or is_first_microbatch
        # If weight is already quantized, weight._quantizer is its true quantizer.
        # for debug mode we create quantizer every iteration, thus we need to set the quantizer states
        if isinstance(fc1_weight, QuantizedTensorStorage) and not debug:
            fc1_weight_quantizer = fc1_weight._quantizer
        elif fc1_weight_quantizer is not None:
            fc1_weight_quantizer.set_usage(
                rowwise=True,
                columnwise=is_grad_enabled and not fsdp2_skip_columnwise,
            )

        if isinstance(fc2_weight, QuantizedTensorStorage) and not debug:
            fc2_weight_quantizer = fc2_weight._quantizer
        elif fc2_weight_quantizer is not None:
            fc2_weight_quantizer.set_usage(
                rowwise=True,
                columnwise=is_grad_enabled and not fsdp2_skip_columnwise,
            )

        fc1_weight_final, new_fc1_weight_workspace = quantize_weight(
            tensor=fc1_weight,
            quantizer=fc1_weight_quantizer,
            workspace=args.fc1_weight_workspace,
            update_workspace=update_ws,
            skip_update_flag=args.skip_fp8_weight_update,
            fsdp_group=fsdp_group,
            workspace_dtype=activation_dtype,
            cache=args.cache_weight,
        )
        fc2_weight_final, new_fc2_weight_workspace = quantize_weight(
            tensor=fc2_weight,
            quantizer=fc2_weight_quantizer,
            workspace=args.fc2_weight_workspace,
            update_workspace=update_ws,
            skip_update_flag=args.skip_fp8_weight_update,
            fsdp_group=fsdp_group,
            workspace_dtype=activation_dtype,
            cache=args.cache_weight,
        )
        fc1_weight_final.update_usage(rowwise_usage=True)
        fc2_weight_final.update_usage(rowwise_usage=True)
    else:
        fc1_weight_final = cast_if_needed(fc1_weight_final, activation_dtype)
        fc2_weight_final = cast_if_needed(fc2_weight_final, activation_dtype)

    # Cast biases to expected dtype
    bias_dtype = activation_dtype
    if needs_quantized_gemm(ln_out_total) and activation_dtype == torch.float32:
        # cuBLAS does not support FP8 GEMM with FP32 bias, so we cast to BF16
        bias_dtype = torch.bfloat16
    fc1_bias_cast = fc1_bias
    fc2_bias_cast = fc2_bias
    if fc1_bias is not None:
        fc1_bias_cast = cast_if_needed(fc1_bias, bias_dtype)
    if fc2_bias is not None:
        fc2_bias_cast = cast_if_needed(fc2_bias, bias_dtype)

    # Calibrate quantizers if needed
    if not fp8 and fp8_calibration:
        if fc1_input_quantizer is not None:
            fc1_input_quantizer.calibrate(ln_out_total)
        if fc1_weight_quantizer is not None:
            fc1_weight_quantizer.calibrate(fc1_weight)

    # ------------------------------------------------------
    # FC1 GEMM
    # ------------------------------------------------------

    # There are 2 fusions possible:
    # - gemm_gelu_fusion - default for full precision, optional for fp8 - need to turn on gemm_gelu_fusion,
    # - bias_gelu_fusion - only for full precision.
    # If both gemm_gelu_fusion and bias_gelu_fusion are enabled, only bias_gelu_fusion will be performer
    if activation != "gelu":
        # blockwise scaled gemms don't support gemm_gelu_fusion in fwd.
        gemm_gelu_fusion = bias_gelu_fusion = False
    else:
        if fp8:
            assert not bias_gelu_fusion, "Bias gelu fusion is supported only for full precision"
        else:
            gemm_gelu_fusion = True
        if gemm_gelu_fusion and bias_gelu_fusion:
            gemm_gelu_fusion = False
    if debug:
        gemm_gelu_fusion = False
    fc1_outputs = general_gemm(
        fc1_weight_final,
        ln_out_total,
        quantization_params=(
            fc2_input_quantizer
            if gemm_gelu_fusion
            else fc1_output_quantizer  # fused gelu output is in fp8
        ),
        out_dtype=activation_dtype,
        bias=(
            fc1_bias_cast if not bias_gelu_fusion else None
        ),  # otherwise bias is added later (fused with gelu)
        gelu=gemm_gelu_fusion,
        use_split_accumulator=use_split_accumulator,
        ub=ub_obj_lnout,
        ub_type=tex.CommOverlapType.AG if ub_overlap_ag else None,
    )

    # ------------------------------------------------------
    # Finished FC1 GEMM...
    # ------------------------------------------------------

    # Deallocate FC1 GEMM input tensor if no longer needed
    # first part of if statement means that we only clear ln_out_total if
    # 1) checkpointing and not recomputing (in the forward stage, not bwd recompute stage)
    # 2) not checkpointing and grad disabled
    # The `is not ln_out` guard avoids clearing the bwd-saved tensor when
    # ln_out_total aliases ln_out (cuBLASMp AG-fprop path).
    if (
        ((checkpoint and not is_recomputation) or not is_grad_enabled)
        and ln_out_total is not ln_out_return
        and ln_out_total is not ln_out
    ):
        clear_tensor_data(ln_out_total)

    # ACTIVATION - sometimes activation is fused with the GEMM above.

    fc1_out_without_bias = None
    act_params = args.activation_params or {}

    if bias_gelu_fusion:
        fc1_out = None
        fc1_out_without_bias, *_ = fc1_outputs
        act_out = bias_gelu_fused(fc1_out_without_bias, fc1_bias_cast)
    elif gemm_gelu_fusion:
        act_out, _, fc1_out, _ = fc1_outputs
    elif debug:
        fc1_out, *_ = fc1_outputs
        act_out = activation_func(fc1_out, None, **act_params)
        act_out = fc2_input_quantizer(act_out)
    else:
        fc1_out, *_ = fc1_outputs
        if fp8:
            recipe = FP8GlobalStateManager.get_fp8_recipe()
            if recipe.float8_block_scaling():
                # tex.quantize does not support GELU fusion for blockwise
                act_out = activation_func(fc1_out, None, **act_params)
                act_out = tex.quantize(act_out, fc2_input_quantizer)
            elif recipe.custom():
                # tex.quantize does not support custom quantizers
                act_out = activation_func(fc1_out, None, **act_params)
                act_out = fc2_input_quantizer(act_out)
            else:
                act_out = activation_func(fc1_out, fc2_input_quantizer, **act_params)
        else:
            if fp8_calibration:
                act_out = activation_func(fc1_out, None, **act_params)
            else:
                act_out = activation_func(fc1_out, fc2_input_quantizer, **act_params)

    if not fp8 and fp8_calibration:
        if fc2_input_quantizer is not None:
            fc2_input_quantizer.calibrate(act_out)

    # we want to skip fc2 computation if we are checkpointing and recomputing,
    # otherwise we compute fc2
    fc2_out = None
    if not (is_recomputation and checkpoint):

        # if we get to this point, we know this is not bwd recomputation
        # so we must be in the fwd
        # now is_grad_enabled can be true or false
        # if false, can safely delete
        # if true, we can only delete if checkpoint is true, since we will recompute anyways,
        # otherwise, checkpoint is false, so cant delete
        if checkpoint or not is_grad_enabled:  # we can safely get rid of these if this is the case
            clear_tensor_data(fc1_out)

        if not fp8 and fp8_calibration:

            if fc2_weight_quantizer is not None:
                fc2_weight_quantizer.calibrate(fc2_weight)

        # Configure Userbuffers reduce-scatter if needed
        ub_obj_fc2out = None
        reduce_scatter_out = None
        if ub_overlap_rs:
            ub_obj_fc2out = get_ub("fc2_fprop", fp8)
            dim_size = list(act_out.size())
            dim_size[0] //= tp_world_size
            dim_size[-1] = fc2_weight.size(0)
            reduce_scatter_out = torch.empty(dim_size, dtype=activation_dtype, device=device)

        # ------------------------------------------------------
        # FC2 GEMM
        # ------------------------------------------------------
        gemm_out, *_, reduce_scatter_out = general_gemm(
            fc2_weight_final,
            act_out,
            out_dtype=activation_dtype,
            bias=fc2_bias_cast,
            quantization_params=fc2_output_quantizer,
            use_split_accumulator=use_split_accumulator,
            ub=ub_obj_fc2out,
            ub_type=tex.CommOverlapType.RS if ub_overlap_rs else None,
            extra_output=reduce_scatter_out,
        )
        # ------------------------------------------------------
        # Finished FC2 GEMM...
        # ------------------------------------------------------

        # Deallocate tensors if no longer needed, again, can safely deallocate
        if checkpoint or not is_grad_enabled:  # same logic as last clear_tensor_data block
            clear_tensor_data(act_out, fc1_out_without_bias, fc1_out)

        # Prepare output tensor
        # Note: Perform tensor-parallel communication if needed
        if ub_overlap_rs:
            # cuBLASMp writes the reduce-scattered output directly into the
            # GEMM output tensor; Userbuffers writes it into the extra-output buffer.
            fc2_out = (
                gemm_out
                if ub_obj_fc2out is not None and ub_obj_fc2out.with_cublasmp()
                else reduce_scatter_out
            )
        elif set_parallel_mode and sequence_parallel:
            fc2_out, _ = reduce_scatter_along_first_dim(gemm_out, tp_group)
        elif set_parallel_mode and tensor_parallel:
            if args.symmetric_ar_type is not None:
                fc2_out, _ = symmetric_all_reduce(
                    gemm_out, tp_group, all_reduce_type=args.symmetric_ar_type
                )
            else:
                fc2_out, _ = allreduce(gemm_out, tp_group)
        else:
            fc2_out = gemm_out
        fc2_out = fc2_out.view(-1, *inp_shape[1:-1], fc2_out.shape[-1])

    # now saving stuff for bwd:
    # if we are using checkpointing, this information will be saved in the bwd recomputation stage, so can skip it in fwd
    # if we are not checkpointing, then we must save this if grad is enabled
    if is_grad_enabled and not save_for_checkpoint:
        if ctx_attrs is None:
            ctx_attrs = {}

        if not fc1_weight_requires_grad:
            if not return_layernorm_output:
                clear_tensor_data(ln_out)
            ln_out = None
        if not fc2_weight_requires_grad:
            clear_tensor_data(act_out)
            act_out = None

        fsdp_shapes = None
        if not checkpoint:  # regular path, no selective activation checkpointing

            if cpu_offloading:
                mark_activation_offload(
                    inputmat, mu, rsigma, ln_out, fc1_out, fc1_out_without_bias, act_out
                )

            # Scatter intermediate/activation tensors saved for the backward pass
            # NOTE: weight_fp8 = weight when fp8 == False and torch.disttributed.FSDP already
            #       shards/unshards the base weights so we don't do it ourselves
            fsdp_shapes = (
                _fsdp_scatter_tensors(  # again, ony relevant if we have activations to save
                    fsdp_group,
                    mu,
                    rsigma,
                    ln_out,
                    fc1_out_without_bias if bias_gelu_fusion else fc1_out,
                    act_out,
                    (
                        fc1_weight_final
                        if fp8 and not isinstance(fc1_weight, Float8Tensor)
                        else None
                    ),
                    (
                        fc2_weight_final
                        if fp8 and not isinstance(fc2_weight, Float8Tensor)
                        else None
                    ),
                )
            )

            if cpu_offloading:
                mark_not_offload(
                    ln_weight_cast,
                    ln_bias,
                    fc1_weight_final,
                    fc1_weight,
                    fc1_bias_cast,
                    fc2_weight_final,
                    fc2_weight,
                    fc2_bias_cast,
                )

        # FSDP2: Don't save FP8 workspace copies for non-quantized
        # weights. Backward will re-quantize from the FSDP2
        # all-gathered weight parameter. (Issue #2681)
        fc1_wt_save = fc1_weight_final
        fc2_wt_save = fc2_weight_final
        if fsdp2_skip_columnwise:
            if fc1_weight_final is not fc1_weight:
                fc1_wt_save = None
            if fc2_weight_final is not fc2_weight:
                fc2_wt_save = None

        # Dedup save slots that alias forward inputs or other outputs;
        # ``_layernorm_mlp_setup_ctx`` rebuilds the refs.
        def _wt_alias(wt_save, weight, new_workspace, workspace, prefix):
            if wt_save is None:
                return None
            if wt_save is weight:
                return prefix + "_weight"
            if new_workspace is not None and wt_save is new_workspace:
                return "new_" + prefix + "_weight_workspace"
            if workspace is not None and wt_save is workspace:
                return prefix + "_weight_workspace"
            return None

        saved_tensor_aliases = (
            "inp" if inputmat is inp else None,
            "ln_weight" if ln_weight_cast is ln_weight else None,
            (
                "ln_out"
                if return_layernorm_output
                and ln_out is not None
                and ln_out_return is not None
                and ln_out is ln_out_return
                else None
            ),
            _wt_alias(
                fc1_wt_save,
                fc1_weight,
                new_fc1_weight_workspace,
                args.fc1_weight_workspace,
                "fc1",
            ),
            "fc1_weight",
            "fc1_bias" if fc1_bias_cast is not None and fc1_bias_cast is fc1_bias else None,
            None,
            None,
            None,
            _wt_alias(
                fc2_wt_save,
                fc2_weight,
                new_fc2_weight_workspace,
                args.fc2_weight_workspace,
                "fc2",
            ),
            "fc2_weight",
            "fc2_bias" if fc2_bias_cast is not None and fc2_bias_cast is fc2_bias else None,
            None,
            None,
        )
        saved = (
            inputmat,
            ln_weight_cast,
            ln_out,
            fc1_wt_save,
            fc1_weight,
            fc1_bias_cast,
            fc1_out,
            fc1_out_without_bias,
            act_out,
            fc2_wt_save,
            fc2_weight,
            fc2_bias_cast,
            mu,
            rsigma,
        )
        tensors_to_save_from_forward = tuple(
            None if alias is not None else tensor
            for alias, tensor in zip(saved_tensor_aliases, saved)
        )
        ctx_attrs["saved_tensor_aliases"] = saved_tensor_aliases
        ctx_attrs["fsdp_shapes"] = fsdp_shapes
        ctx_attrs["is_recomputation"] = is_recomputation

        if is_recomputation:  # return the recomputed tensors
            return None, None, None, None, tensors_to_save_from_forward, ctx_attrs

    # we only get to this point if we are not recomputing for bwd, since that would have returned in the block above
    ln_out_for_return = None
    if return_layernorm_output:
        if return_layernorm_output_gathered:
            shape = list(inp_shape)
            shape[0] *= tp_size if (sequence_parallel and set_parallel_mode) else 1
            ln_out_for_return = ln_out_return.view(shape)
        else:
            ln_out_for_return = ln_out_return.view(inp_shape)
    return (
        fc2_out,
        ln_out_for_return,
        new_fc1_weight_workspace,
        new_fc2_weight_workspace,
        tensors_to_save_from_forward,
        ctx_attrs,
    )


def _layernorm_mlp_setup_ctx(
    bwd_args: LayerNormMLPBwdArgs,
    fwd_args: LayerNormMLPFwdArgs,
    fwd_outputs: Tuple[Any, ...],
    ctx_attrs: Dict,
    tensors_to_save_from_forward: Tuple[Any, ...],
) -> Tuple[Any, ...]:
    """Populate ``bwd_args`` from forward state.

    Returns the tensors that should be passed through ``prepare_for_saving``
    by the caller. ``fwd_outputs`` is ``(fc2_out, ln_out_return,
    new_fc1_weight_workspace, new_fc2_weight_workspace)``; the last three
    rebuild the deduped save slots.

    With ``ctx_attrs["checkpoint"]`` and no recomputation yet, only the
    forward inputs are saved; the rest of ``bwd_args`` is filled when the
    backward recomputes the forward.
    """
    checkpoint = ctx_attrs.get("checkpoint", False)
    is_recomputation = ctx_attrs.get("is_recomputation", False)
    if checkpoint and not is_recomputation:
        bwd_args.checkpoint = True
        bwd_args.autocast_state = ctx_attrs["autocast_state"]
        bwd_args.cpu_rng_state = ctx_attrs["cpu_rng_state"]
        bwd_args.cuda_rng_state = ctx_attrs["cuda_rng_state"]
        # The saved inputs are re-bound to the args when the backward recomputes.
        bwd_args.checkpoint_fwd_args = dataclass_replace(
            fwd_args,
            inp=None,
            ln_weight=None,
            ln_bias=None,
            fc1_weight=None,
            fc1_bias=None,
            fc2_weight=None,
            fc2_bias=None,
            fc1_weight_workspace=None,
            fc2_weight_workspace=None,
            cache_weight=False,
            recompute_for_bwd=True,
        )
        return tuple(getattr(fwd_args, name) for name in ctx_attrs["saved_tensor_aliases"])

    inp = fwd_args.inp
    fc1_weight = fwd_args.fc1_weight
    fc2_weight = fwd_args.fc2_weight
    fp8 = fwd_args.fp8
    debug = fwd_args.debug
    fuse_wgrad_accumulation = fwd_args.fuse_wgrad_accumulation
    fc1_weight_requires_grad = fwd_args.fc1_weight_requires_grad
    fc2_weight_requires_grad = fwd_args.fc2_weight_requires_grad

    # Quantizers
    bwd_args.fc1_input_quantizer = fwd_args.fc1_input_quantizer
    bwd_args.fc2_input_quantizer = fwd_args.fc2_input_quantizer
    bwd_args.fc1_weight_quantizer = (
        fc1_weight._quantizer
        if (fp8 or debug) and isinstance(fc1_weight, QuantizedTensorStorage) and not debug
        else fwd_args.fc1_weight_quantizer
    )
    bwd_args.fc2_weight_quantizer = (
        fc2_weight._quantizer
        if (fp8 or debug) and isinstance(fc2_weight, QuantizedTensorStorage) and not debug
        else fwd_args.fc2_weight_quantizer
    )
    bwd_args.fc1_grad_input_quantizer = fwd_args.fc1_grad_input_quantizer
    bwd_args.fc1_grad_weight_quantizer = fwd_args.fc1_grad_weight_quantizer
    bwd_args.fc1_grad_output_quantizer = fwd_args.fc1_grad_output_quantizer
    bwd_args.fc2_grad_input_quantizer = fwd_args.fc2_grad_input_quantizer
    bwd_args.fc2_grad_weight_quantizer = fwd_args.fc2_grad_weight_quantizer
    bwd_args.fc2_grad_output_quantizer = fwd_args.fc2_grad_output_quantizer

    # Differentiability summary
    bwd_args.use_bias = fwd_args.fc2_bias is not None
    bwd_args.requires_dgrad = (
        fwd_args.input_requires_grad
        or fwd_args.ln_weight_requires_grad
        or fwd_args.ln_bias_requires_grad
    )
    bwd_args.fc1_weight_requires_grad = fc1_weight_requires_grad
    bwd_args.fc1_bias_requires_grad = fwd_args.fc1_bias_requires_grad
    bwd_args.fc2_weight_requires_grad = fc2_weight_requires_grad
    bwd_args.inp_shape = inp.shape

    # Normalization
    bwd_args.normalization = fwd_args.normalization
    bwd_args.zero_centered_gamma = fwd_args.zero_centered_gamma
    bwd_args.bwd_ln_sm_margin = fwd_args.bwd_ln_sm_margin
    bwd_args.return_layernorm_output = fwd_args.return_layernorm_output
    bwd_args.return_layernorm_output_gathered = (
        fwd_args.return_layernorm_output_gathered and fwd_args.sequence_parallel
    )

    # Activation
    bwd_args.activation = fwd_args.activation
    bwd_args.activation_params = fwd_args.activation_params
    bwd_args.bias_gelu_fusion = fwd_args.bias_gelu_fusion

    # Numerical / dtype config
    bwd_args.activation_dtype = fwd_args.activation_dtype
    bwd_args.fp8 = fp8
    bwd_args.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
    bwd_args.dgrad_use_split_accumulator = fwd_args.dgrad_use_split_accumulator
    bwd_args.wgrad_use_split_accumulator = fwd_args.wgrad_use_split_accumulator
    bwd_args.backward_override = fwd_args.backward_override
    bwd_args.debug = debug

    # Tensor / sequence parallelism
    bwd_args.set_parallel_mode = fwd_args.set_parallel_mode
    bwd_args.tp_group = fwd_args.tp_group
    bwd_args.tp_size = fwd_args.tp_size
    bwd_args.tensor_parallel = fwd_args.tensor_parallel
    bwd_args.sequence_parallel = fwd_args.sequence_parallel

    # Userbuffers
    bwd_args.ub_overlap_ag = (
        fwd_args.ub_overlap_ag
        and fwd_args.is_grad_enabled
        and not fwd_args.return_layernorm_output_gathered
    )
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
    if fuse_wgrad_accumulation:
        # Keep weakrefs to weights to preserve attributes like main_grad
        # when we need to modify the weight python objects
        bwd_args.fc1_weight_ref = weakref.ref(fc1_weight) if fc1_weight_requires_grad else None
        bwd_args.fc2_weight_ref = weakref.ref(fc2_weight) if fc2_weight_requires_grad else None
        bwd_args.fc1_weight_overwrites_main_grad = getattr(fc1_weight, "overwrite_main_grad", False)
        bwd_args.fc2_weight_overwrites_main_grad = getattr(fc2_weight, "overwrite_main_grad", False)
        # MCore FSDP creates main_grad lazily before backward, so don't touch it here
        if hasattr(fc1_weight, "__fsdp_param__") and hasattr(fc2_weight, "__fsdp_param__"):
            bwd_args.fc1_main_grad_func = (
                fc1_weight.get_main_grad if fc1_weight_requires_grad else lambda: None
            )
            bwd_args.fc2_main_grad_func = (
                fc2_weight.get_main_grad if fc2_weight_requires_grad else lambda: None
            )
        else:
            bwd_args.fc1_main_grad_func = lambda: fc1_weight.main_grad
            bwd_args.fc2_main_grad_func = lambda: fc2_weight.main_grad

    # Misc
    bwd_args.cpu_offloading = fwd_args.cpu_offloading

    saved = list(tensors_to_save_from_forward)
    aliases = ctx_attrs["saved_tensor_aliases"]
    in_features = inp.shape[-1]
    for i, alias in enumerate(aliases):
        if alias is None:
            continue
        if alias == "inp":
            saved[i] = inp.view((-1, in_features))
        elif alias == "ln_out":
            saved[i] = fwd_outputs[1].view((-1, in_features))
        elif alias == "new_fc1_weight_workspace":
            saved[i] = fwd_outputs[2]
        elif alias == "new_fc2_weight_workspace":
            saved[i] = fwd_outputs[3]
        else:
            saved[i] = getattr(fwd_args, alias)
    if fwd_args.cpu_offloading:
        # Rebuilt views don't carry the offload marks set on the forward tensors
        mark_activation_offload(
            *(saved[i] for i, alias in enumerate(aliases) if alias in ("inp", "ln_out"))
        )
    return tuple(saved)


def _layernorm_mlp_recompute(
    bwd_args: LayerNormMLPBwdArgs, ctx: torch.autograd.function.FunctionCtx
) -> None:
    """Bind the saved tensors to ``bwd_args``, recomputing the forward when
    the module ran with activation checkpointing."""
    tensors = restore_from_func_ctx(ctx)
    if not bwd_args.checkpoint:
        bwd_args.set_saved_tensors(tensors)
        return

    # backward is not in autocast context, so we set the state here
    # we also have to set the quantizer states to what they were before the forward pass (only relevant for DelayedScaling recipe)
    fwd_args: LayerNormMLPFwdArgs = bwd_args.checkpoint_fwd_args
    bwd_args.checkpoint_fwd_args = None
    final_autocast_state = FP8GlobalStateManager.get_autocast_state()
    FP8GlobalStateManager.set_autocast_state(bwd_args.autocast_state)
    if (
        fwd_args.fp8
        and FP8GlobalStateManager.get_fp8_recipe().__class__.__name__ == "DelayedScaling"
    ):  # only applicable for delayed scaling
        FP8GlobalStateManager.get_old_fp8_meta_tensors_for_recompute(
            fwd_args.fp8_meta
        )  # set old quantizer state

    # get current rng state
    final_cpu_rng_state = torch.get_rng_state()
    final_cuda_rng_state = _get_cuda_rng_state()

    # set rng state for fwd
    torch.set_rng_state(bwd_args.cpu_rng_state)
    _set_cuda_rng_state(bwd_args.cuda_rng_state)

    for name, tensor in zip(_CHECKPOINT_SAVED_ALIASES, tensors):
        setattr(fwd_args, name, tensor)
    (
        _,
        _,
        new_fc1_weight_workspace,
        new_fc2_weight_workspace,
        tensors_to_save_from_forward,
        ctx_attrs,
    ) = _layernorm_mlp_forward_impl(fwd_args)
    recomputed = _layernorm_mlp_setup_ctx(
        bwd_args,
        fwd_args,
        (None, None, new_fc1_weight_workspace, new_fc2_weight_workspace),
        ctx_attrs,
        tensors_to_save_from_forward,
    )
    bwd_args.set_saved_tensors(recomputed)
    if fwd_args.fp8 and fwd_args.any_requires_grad():
        bwd_args.reduce_and_update_bwd_fp8_tensors = check_fp8_reduce_and_update(
            restore_first_module=True
        )

    FP8GlobalStateManager.set_autocast_state(final_autocast_state)
    if (
        fwd_args.fp8
        and FP8GlobalStateManager.get_fp8_recipe().__class__.__name__ == "DelayedScaling"
    ):
        FP8GlobalStateManager.restore_fp8_meta_tensors(fwd_args.fp8_meta)  # restore quantizers

    # set rng state for fwd
    torch.set_rng_state(final_cpu_rng_state)
    _set_cuda_rng_state(final_cuda_rng_state)


def _layernorm_mlp_backward_impl(
    args: LayerNormMLPBwdArgs,
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Backward implementation for the layernorm-MLP layer.

    Caller must have populated ``args.grad_output`` / ``args.grad_ln_out`` and
    the saved-tensor fields before invocation. Returns ``(dgrad, dgamma,
    dbeta, fc1_wgrad, fc1_bias_grad, fc2_wgrad, fc2_bias_grad)``.
    """
    with get_nvtx_range_context("_LayerNormMLP_backward"):
        inputmat = args.inputmat
        ln_weight = args.ln_weight
        ln_out = args.ln_out
        fc1_weight = args.fc1_weight_fp8
        origin_fc1_weight = args.fc1_weight
        fc1_bias = args.fc1_bias
        fc1_out = args.fc1_out
        fc1_out_without_bias = args.fc1_out_without_bias
        act_out = args.act_out
        fc2_weight = args.fc2_weight_fp8
        origin_fc2_weight = args.fc2_weight
        fc2_bias = args.fc2_bias
        mu = args.mu
        rsigma = args.rsigma
        grad_output_arg = args.grad_output

        # Restore origin weights from weakrefs
        # Only needed when fuse_wgrad_accumulation is enabled.
        fc1_weight_python_object = None
        fc2_weight_python_object = None
        fc1_weight_main_grad = None
        fc2_weight_main_grad = None
        if args.fuse_wgrad_accumulation:
            fc1_weight_ref = args.fc1_weight_ref
            fc2_weight_ref = args.fc2_weight_ref
            args.fc1_weight_ref = None
            args.fc2_weight_ref = None
            fc1_weight_python_object = fc1_weight_ref() if fc1_weight_ref is not None else None
            fc2_weight_python_object = fc2_weight_ref() if fc2_weight_ref is not None else None
            if args.fc1_weight_requires_grad:
                assert (
                    fc1_weight_python_object is not None
                ), "fc1_weight was removed while fuse_wgrad_accumulation=True"
                fc1_weight_main_grad = args.fc1_main_grad_func()
                fc1_weight_python_object.main_grad = fc1_weight_main_grad
            if args.fc2_weight_requires_grad:
                assert (
                    fc2_weight_python_object is not None
                ), "fc2_weight was removed while fuse_wgrad_accumulation=True"
                fc2_weight_main_grad = args.fc2_main_grad_func()
                fc2_weight_python_object.main_grad = fc2_weight_main_grad

        # TODO: Fix this  # pylint: disable=fixme
        # Gather saved autograd context tensors when running with FSDP
        # NOTE: weight_fp8 = weight when fp8 == False and torch.disttributed.FSDP already
        #       shards/unshards the base weights so we don't do it ourselves
        # _fsdp_gather_tensors(
        #    args.fsdp_group,
        #    args.fsdp_shapes,
        #    mu,
        #    rsigma,
        #    ln_out,
        #    fc1_out_without_bias if bias_gelu_nvfusion else fc1_out,,
        #    gelu_out,
        #    fc1_weight_fp8 if args.fp8 and not isinstance(fc1_weight, Float8Tensor) else None,
        #    fc2_weight_fp8 if args.fp8 and not isinstance(fc2_weight, Float8Tensor) else None,
        # )

        # Choose whether to use GEMM kernel with split accumulator
        dgrad_use_split_accumulator = args.dgrad_use_split_accumulator
        wgrad_use_split_accumulator = args.wgrad_use_split_accumulator

        # No need to do bulk DGRAD/WGRAD overlap if WGRAD is not required
        ub_bulk_dgrad = args.fc1_weight_requires_grad and args.ub_bulk_dgrad
        ub_bulk_wgrad = args.fc1_weight_requires_grad and args.ub_bulk_wgrad

        # Configure quantizer for FC2 grad output tensor
        # Note: dgrad GEMM requires row-wise usage, wgrad GEMM
        # requires column-wise usage
        if args.fc2_grad_output_quantizer is not None:
            quantizer = args.fc2_grad_output_quantizer
            quantizer.set_usage(rowwise=True, columnwise=True)
            if args.ub_overlap_ag:
                # Userbuffers only supports communication for one
                # tensor usage at a time. Configure quantizer with
                # usage for only dgrad GEMM.
                quantizer.set_usage(columnwise=False)
            # Amax reduction group for FC2 grad output (row-parallel sequence parallel)
            set_quantizer_amax_reduction_group(
                quantizer,
                args.tp_group if (args.sequence_parallel and args.set_parallel_mode) else None,
            )

        # Prepare FC2 grad output tensor
        # Note: Cast to expected dtype and perform tensor-parallel communication
        ub_obj_fc2_dgrad = None
        if args.ub_overlap_ag:
            ub_obj_fc2_dgrad = get_ub("fc2_dgrad", args.fp8)
        args.ub_obj_gradout = ub_obj_fc2_dgrad
        (
            grad_output,
            fc2_bias_grad,
        ) = TransformerEngineBaseModule.grad_output_preprocess(
            args, grad_output_arg, True, args.fc2_grad_output_quantizer
        )

        # Launch tensor-parallel communication for FC1 GEMM input
        ln_out_total = None
        ln_out_total_work = None
        ub_obj_fc1_dgrad = None
        if args.fc1_weight_requires_grad and args.tensor_parallel and args.sequence_parallel:
            quantizer = None
            if args.fp8 or args.debug:
                quantizer = args.fc1_input_quantizer
                set_quantizer_usage_for_wgrad_all_gather(quantizer)
            if ub_bulk_dgrad:
                ub_obj_fc1_dgrad = get_ub("fc1_dgrad", args.fp8)
                ln_out_total, _ = fill_userbuffers_buffer_for_all_gather(
                    ub_obj_fc1_dgrad,
                    ln_out,
                    quantizer,
                    args.tp_group,
                )
            else:
                ln_out_total, ln_out_total_work = gather_along_first_dim(
                    ln_out,
                    args.tp_group,
                    async_op=True,
                    quantizer=quantizer,
                )
        else:
            ln_out_total = ln_out

        # Check whether to output wgrad GEMM directly into main grad
        if args.is_first_microbatch is not None:
            accumulate_wgrad_into_param_main_grad = (
                args.fuse_wgrad_accumulation and not args.is_first_microbatch
            )
        else:
            accumulate_wgrad_into_param_main_grad = args.fuse_wgrad_accumulation

        # --------------------------------------------------
        # FC2 DGRAD
        # --------------------------------------------------

        # There are 6 possible fusion paths
        # 1 high-precision bias_gelu_fusion: gemm, FC1_bias + gelu,
        # 2 high-precision fc2_dgrad_gemm_gelu_fusion: gemm + gelu, FC1_bias + quantize
        # 3 fp8 activation+bias+quantize fusion: gemm, activation + FC1_bias + quantize
        # 4 fp8 bias+quantize fusion: gemm, activation, FC1_bias + quantize
        # 5 high-precision unfused: gemm, activation, FC1_bias + FC1_gemm
        # 6 fp8 unfused: gemm, activation, FC1_bias + FC1_gemm
        fc2_dgrad_gemm_gelu_fusion = (
            not args.fp8
            and (args.activation == "gelu")
            and (not args.bias_gelu_fusion)
            and (not args.debug)
        )

        # FSDP2: Re-create workspace from all-gathered weight when
        # workspace was not saved to avoid forward memory
        # accumulation. (Issue #2681)
        if fc2_weight is None:
            if isinstance(origin_fc2_weight, QuantizedTensorStorage):
                fc2_weight = origin_fc2_weight
            elif args.fc2_weight_quantizer is not None:
                args.fc2_weight_quantizer.set_usage(rowwise=True, columnwise=True)
                fc2_weight = args.fc2_weight_quantizer(origin_fc2_weight)

        # Make sure required data is available
        if isinstance(grad_output, QuantizedTensorStorage):
            grad_output.update_usage(rowwise_usage=True)
        if args.fc2_weight_quantizer is not None and isinstance(fc2_weight, QuantizedTensorStorage):
            fc2_weight.update_usage(columnwise_usage=True)

        # Perform GEMM
        gemm_output, *_ = general_gemm(
            fc2_weight,
            grad_output,
            layout="NN",
            grad=True,
            quantization_params=(
                args.fc1_grad_input_quantizer if fc2_dgrad_gemm_gelu_fusion or args.debug else None
            ),  # high precision to activation
            out_dtype=args.activation_dtype,
            gelu=fc2_dgrad_gemm_gelu_fusion,
            gelu_in=fc1_out if fc2_dgrad_gemm_gelu_fusion else None,
            use_split_accumulator=dgrad_use_split_accumulator,
            ub=ub_obj_fc2_dgrad,
            ub_type=tex.CommOverlapType.AG if args.ub_overlap_ag else None,
        )

        # FSDP2: Clear columnwise/transpose caches after FC2 dgrad GEMM
        # to prevent them from persisting on the all-gathered buffer.
        # Uses is_fsdp2 (not fsdp2_skip_columnwise) so cleanup runs
        # even when backward follows gradient-checkpoint recomputation.
        # (Issues #2681, #2717)
        if args.is_fsdp2 and isinstance(fc2_weight, QuantizedTensorStorage):
            clear_columnwise_cache(fc2_weight)

        # Prepare input grad tensor
        dact = None
        fc2_dgrad = None
        if fc2_dgrad_gemm_gelu_fusion:
            dact = gemm_output
        else:
            fc2_dgrad = gemm_output

        # --------------------------------------------------
        # Finished FC2 DGRAD...
        # --------------------------------------------------

        # cuBLASMp's AG+GEMM consumes the gathered grad_output inline and
        # does not preserve it for fc2_wgrad. Userbuffers leaves the
        # gathered tensor in its persistent buffer; cuBLASMp does not, so
        # we gather here. Route through the same FP8-aware all-gather as
        # the non-overlap path in
        # ``TransformerEngineBaseModule.grad_output_preprocess`` by passing
        # the grad_output quantizer. Columnwise data needed for fc2_wgrad
        # is produced by ``update_usage(columnwise_usage=True)`` further
        # below.
        if (
            args.fc2_weight_requires_grad
            and args.ub_overlap_ag
            and args.ub_obj_gradout is not None
            and args.ub_obj_gradout.with_cublasmp()
        ):
            if args.fc2_grad_output_quantizer is not None:
                set_quantizer_usage_for_wgrad_all_gather(args.fc2_grad_output_quantizer)
            grad_output, _ = gather_along_first_dim(
                grad_output,
                args.tp_group,
                quantizer=args.fc2_grad_output_quantizer,
            )

        # --------------------------------------------------
        # FC2 WGRAD
        # --------------------------------------------------

        fc2_wgrad = None
        if args.fc2_weight_requires_grad:
            # Prepare grad output tensor
            # Note: Synchronize tensor-parallel communication and
            # make sure required data is available
            if args.ub_overlap_ag and isinstance(args.fc2_grad_output_quantizer, MXFP8Quantizer):
                # UB does not support pipelined overlapping grad output
                # all-gather with wgrad GEMM. Also, we can't
                # convert row-scaled MXFP8 to column-scaled, so we
                # can't reuse the grad output that was gathered
                # for the dgrad GEMM. We work around by explicitly
                # overlapping the AG operation with the dgrad GEMM.

                # Get the communication stream from the dgrad GEMM to use for the AG
                dgrad_send_stream, dgrad_recv_stream = ub_obj_fc2_dgrad.get_communication_stream()

                ub_obj_fc2_wgrad = get_ub("fc2_wgrad", args.fp8)

                args.fc2_grad_output_quantizer.set_usage(rowwise=False, columnwise=True)

                # We use the send stream to copy into the userbuffers.
                # This is the same stream that we will use to access the data in the AG,
                # so we dont need to add any syncs yet.
                with torch.cuda.stream(dgrad_send_stream):
                    grad_output, _ = fill_userbuffers_buffer_for_all_gather(
                        ub_obj_fc2_wgrad,
                        grad_output_arg,
                        args.fc2_grad_output_quantizer,
                        args.tp_group,
                    )

                # Allgather grad_outputs[0] using the dgrad streams so we can overlap with the fc2_dgrad gemm
                tex.bulk_overlap_ag_with_external_gemm(
                    ub_obj_fc2_wgrad, dgrad_send_stream, dgrad_recv_stream
                )

            # Prepare input tensor
            # Note: Synchronize tensor-parallel communication and
            # make sure required data is available
            if args.fp8 or args.debug:
                if isinstance(act_out, QuantizedTensorStorage):
                    act_out.update_usage(columnwise_usage=True)
                else:
                    args.fc2_input_quantizer.set_usage(rowwise=False, columnwise=True)
                    act_out = args.fc2_input_quantizer(act_out)

            if args.fp8 or args.debug:
                if isinstance(grad_output, QuantizedTensorStorage):
                    grad_output.update_usage(columnwise_usage=True)
                else:
                    args.fc2_grad_output_quantizer.set_usage(rowwise=False, columnwise=True)
                    grad_output = args.fc2_grad_output_quantizer(grad_output)

            # Whether to set grad arg in general_gemm
            grad_arg = True
            if args.fp8 and args.fp8_recipe.float8_block_scaling():
                grad_arg = False

            # Arguments to include in wgrad GEMM closure
            fc2_wgrad_gemm_kwargs = {
                "out_dtype": (
                    fc2_weight_main_grad.dtype
                    if args.fuse_wgrad_accumulation
                    else args.activation_dtype
                ),
                "quantization_params": args.fc2_grad_weight_quantizer,  # wgrad in high precision
                "accumulate": (
                    accumulate_wgrad_into_param_main_grad
                    if not args.fc2_weight_overwrites_main_grad
                    else False
                ),
                "layout": "NT",
                "out": fc2_weight_main_grad if args.fuse_wgrad_accumulation else None,
                "bias": fc2_bias if fc2_bias is not None and fc2_bias_grad is None else None,
                "use_split_accumulator": wgrad_use_split_accumulator,
                "grad": grad_arg,
            }

            def fc2_wgrad_gemm(
                x: torch.Tensor,
                dy: torch.Tensor,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                """Perform FC2 WGRAD GEMM

                May be called outside of this function to enable
                some advanced communication/compute overlapping.

                """
                dw, db, *_ = general_gemm(x, dy, **fc2_wgrad_gemm_kwargs)
                return dw, db

            # Choose whether to call wgrad GEMM now or delay
            if args.wgrad_store is not None and args.wgrad_store.delay_wgrad_compute():
                args.wgrad_store.put([act_out, grad_output], fc2_wgrad_gemm)
            else:

                # Call wgrad GEMM now
                fc2_wgrad, fc2_bias_grad_ = fc2_wgrad_gemm(act_out, grad_output)

                # Update grad bias if needed
                if fc2_bias_grad is None:
                    if args.fp8 and args.fp8_recipe.float8_block_scaling() and fc2_bias is not None:
                        # BGRAD not fused with GEMM for float8 blockwise gemm.
                        fc2_bias_grad_ = act_out.view(-1, act_out.shape[-1]).sum(dim=0)
                    fc2_bias_grad = fc2_bias_grad_
                del fc2_bias_grad_

        # Deallocate input tensor if permitted
        if args.wgrad_store is not None and not args.wgrad_store.delay_wgrad_compute():
            clear_tensor_data(act_out)

        # --------------------------------------------------
        # Finished FC2 WGRAD...
        # --------------------------------------------------

        # bias computation
        act_params = args.activation_params or {}
        fc1_bias_grad = None
        fuse_gemm_and_bias_fc1_wgrad = False
        if args.fc1_grad_output_quantizer is not None:
            args.fc1_grad_output_quantizer.set_usage(rowwise=True, columnwise=True)
        if args.bias_gelu_fusion:
            # Fusion: gemm, bias + gelu
            assert args.activation == "gelu"
            assert not args.fp8
            fc1_bias_grad, dact = bgrad_dgelu_fused(fc2_dgrad, fc1_out_without_bias, fc1_bias)
            if args.fc1_grad_output_quantizer is not None:
                dact = args.fc1_grad_output_quantizer(dact)
        elif args.debug:
            dact_func = _act_func(args.activation)[1]
            dact = dact_func(fc2_dgrad, fc1_out.to(args.activation_dtype), None, **act_params)
            fc1_bias_grad = dact.sum(dim=0)
            dact = args.fc1_grad_output_quantizer(dact)
        elif (
            _act_func(args.activation, args.fp8_recipe if args.fp8 else None)[2] is not None
            and args.fp8
        ):
            # Fusion: gemm, bias + gelu + quantize
            dbias_dact_quantize_func = _act_func(
                args.activation, args.fp8_recipe if args.fp8 else None
            )[2]
            fc1_bias_grad, dact = dbias_dact_quantize_func(
                fc2_dgrad,
                fc1_out.to(args.activation_dtype),
                args.fc1_grad_output_quantizer,
                **act_params,
            )  # quantize bgrad gelu fused
        else:
            # Fusion: gemm + gelu,
            if not fc2_dgrad_gemm_gelu_fusion:
                activation_func_bwd = _act_func(
                    args.activation, args.fp8_recipe if args.fp8 else None
                )[1]
                dact = activation_func_bwd(
                    fc2_dgrad, fc1_out.to(args.activation_dtype), None, **act_params
                )  # activation in high precision

            if args.fp8:
                # TODO float8 blockwise current scaling (as well as custom quantizers) has no bgrad fusion for now
                if (
                    isinstance(
                        args.fc1_grad_output_quantizer,
                        (Float8BlockQuantizer, IdentityQuantizer),
                    )
                    or args.fp8_recipe.custom()
                ):
                    fc1_bias_grad = dact.view(-1, dact.shape[-1]).sum(dim=0)
                    dact = args.fc1_grad_output_quantizer(dact)
                else:
                    fc1_bias_grad, dact = tex.bgrad_quantize(dact, args.fc1_grad_output_quantizer)
            else:
                fuse_gemm_and_bias_fc1_wgrad = (
                    True  # fc1_bias_grad is computed later, fused with wgrad gemm for the FC1
                )
                # it may  not be calculated in case wgrad is not required.
                if fc1_bias is not None:
                    if not args.fc1_weight_requires_grad and args.fc1_bias_requires_grad:
                        fc1_bias_grad = dact.sum(dim=0)

        # Overwrite data. Deleting the tensor does not release underlying memory.
        clear_tensor_data(fc1_out, fc1_out_without_bias)

        # Set UB algo and UB obj for fc1_dgrad/wgrad bulk/pipelined overlap
        ub_obj_fc1_dgrad = None
        ub_obj_fc1_wgrad = None
        ub_type_fc1_dgrad = None
        ub_type_fc1_wgrad = None
        fc1_dgrad_shape = [reduce(multiply_op, inputmat.shape[:-1]), inputmat.shape[-1]]
        if args.ub_overlap_rs_dgrad:
            # Overlap DGRAD+RS
            ub_obj_fc1_dgrad = get_ub("fc1_dgrad", args.fp8)
            ub_type_fc1_dgrad = tex.CommOverlapType.RS
        else:
            if ub_bulk_dgrad:
                # Overlap ln_out all-gather with DGRAD compute
                ub_obj_fc1_dgrad = get_ub("fc1_dgrad", args.fp8)
                ub_type_fc1_dgrad = tex.CommOverlapType.AG
            if ub_bulk_wgrad:
                # Overlap FC1 DGRAD reduce-scatter with WGRAD compute
                ub_obj_fc1_wgrad = get_ub("fc1_wgrad", args.fp8)
                ub_type_fc1_wgrad = tex.CommOverlapType.RS

        # --------------------------------------------------
        # FC1 DGRAD
        # --------------------------------------------------

        # FSDP2: Re-create workspace from all-gathered weight when
        # workspace was not saved. (Issue #2681)
        if fc1_weight is None:
            if isinstance(origin_fc1_weight, QuantizedTensorStorage):
                fc1_weight = origin_fc1_weight
            elif args.fc1_weight_quantizer is not None:
                args.fc1_weight_quantizer.set_usage(rowwise=True, columnwise=True)
                fc1_weight = args.fc1_weight_quantizer(origin_fc1_weight)

        # Make sure required data is available
        if args.fc1_weight_quantizer is not None and isinstance(fc1_weight, QuantizedTensorStorage):
            fc1_weight.update_usage(columnwise_usage=True)

        # Output buffers for Userbuffers reduce-scatter
        gemm_out = None
        reduce_scatter_out = None
        if args.ub_overlap_rs_dgrad:
            reduce_scatter_out = torch.empty(
                fc1_dgrad_shape, dtype=args.activation_dtype, device="cuda"
            )
        if ub_bulk_wgrad:
            gemm_out = ub_obj_fc1_wgrad.get_buffer(local_chunk=False)

        # dgrad GEMM
        gemm_out, *_, reduce_scatter_out = general_gemm(
            fc1_weight,
            dact,
            out=gemm_out,
            out_dtype=args.activation_dtype,
            quantization_params=args.fc1_grad_input_quantizer,
            layout="NN",
            grad=True,
            use_split_accumulator=dgrad_use_split_accumulator,
            ub=ub_obj_fc1_dgrad,
            ub_type=ub_type_fc1_dgrad,
            extra_output=reduce_scatter_out,
            bulk_overlap=ub_bulk_dgrad,
        )

        # FSDP2: Clear columnwise/transpose caches after FC1 dgrad GEMM
        # to prevent them from persisting on the all-gathered buffer.
        # Uses is_fsdp2 (not fsdp2_skip_columnwise) so cleanup runs
        # even when backward follows gradient-checkpoint recomputation.
        # (Issues #2681, #2717)
        if args.is_fsdp2 and isinstance(fc1_weight, QuantizedTensorStorage):
            clear_columnwise_cache(fc1_weight)

        # Prepare grad input tensor
        # Note: Perform tensor-parallel communication
        fc1_dgrad = None
        fc1_dgrad_work = None
        if args.ub_overlap_rs_dgrad:
            # cuBLASMp writes the reduce-scattered dgrad directly into the
            # GEMM output tensor; Userbuffers uses the extra-output buffer.
            fc1_dgrad = (
                gemm_out
                if ub_obj_fc1_dgrad is not None and ub_obj_fc1_dgrad.with_cublasmp()
                else reduce_scatter_out
            )
        elif ub_bulk_wgrad:
            fc1_dgrad = ub_obj_fc1_wgrad.get_buffer(local_chunk=True)
        elif args.set_parallel_mode and not ub_bulk_wgrad:
            fc1_dgrad = gemm_out
            if args.sequence_parallel:
                if args.return_layernorm_output and args.return_layernorm_output_gathered:
                    fc1_dgrad = fc1_dgrad + args.grad_ln_out.view_as(fc1_dgrad)
                fc1_dgrad, fc1_dgrad_work = reduce_scatter_along_first_dim(
                    fc1_dgrad,
                    args.tp_group,
                    async_op=True,
                )
            elif args.tensor_parallel:
                fc1_dgrad, fc1_dgrad_work = allreduce(fc1_dgrad, args.tp_group, async_op=True)
        else:
            fc1_dgrad = gemm_out

        # --------------------------------------------------
        # Finished FC1 DGRAD...
        # --------------------------------------------------

        # --------------------------------------------------
        # FC1 WGRAD
        # --------------------------------------------------
        fc1_wgrad = None
        if args.fc1_weight_requires_grad:

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
                    args.fc1_input_quantizer.set_usage(rowwise=False, columnwise=True)
                    ln_out_total = args.fc1_input_quantizer(ln_out_total)

            # Prepare grad output tensor
            # Note: Synchronize tensor-parallel communication and
            # make sure required data is available
            if args.fp8 or args.debug:
                if isinstance(dact, QuantizedTensorStorage):
                    dact.update_usage(columnwise_usage=True)
                else:
                    args.fc1_grad_output_quantizer.set_usage(rowwise=False, columnwise=True)
                    dact = args.fc1_grad_output_quantizer(dact)

            # Output buffer for overlapping grad input
            # reduce-scatter with wgrad GEMM
            reduce_scatter_out = None
            if ub_bulk_wgrad and ub_obj_fc1_wgrad.is_fp8_ubuf():
                reduce_scatter_out = torch.empty(
                    fc1_dgrad_shape, dtype=args.activation_dtype, device="cuda"
                )

            # Arguments to include in wgrad GEMM closure
            fc1_wgrad_gemm_kwargs = {
                "out_dtype": (
                    fc1_weight_main_grad.dtype
                    if args.fuse_wgrad_accumulation
                    else args.activation_dtype
                ),
                "quantization_params": args.fc1_grad_weight_quantizer,
                "accumulate": (
                    accumulate_wgrad_into_param_main_grad
                    if not args.fc1_weight_overwrites_main_grad
                    else False
                ),
                "layout": "NT",
                "out": fc1_weight_main_grad if args.fuse_wgrad_accumulation else None,
                "bias": fc1_bias if fuse_gemm_and_bias_fc1_wgrad else None,
                "use_split_accumulator": wgrad_use_split_accumulator,
                "grad": fuse_gemm_and_bias_fc1_wgrad,
                "ub": ub_obj_fc1_wgrad,
                "ub_type": ub_type_fc1_wgrad,
                "extra_output": reduce_scatter_out,
                "bulk_overlap": ub_bulk_wgrad,
            }

            def fc1_wgrad_gemm(
                x: torch.Tensor,
                dy: torch.Tensor,
                _is_delayed: bool = True,
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                """Perform FC1 WGRAD GEMM

                May be called outside of this function to enable
                some advanced communication/compute overlapping.

                """
                dw, db, *_ = general_gemm(x, dy, **fc1_wgrad_gemm_kwargs)
                return dw, db

            # Choose whether to call wgrad GEMM now or delay
            if args.wgrad_store is not None and args.wgrad_store.delay_wgrad_compute():
                if (
                    fc1_wgrad_gemm_kwargs["ub"] is not None
                    or fc1_wgrad_gemm_kwargs["ub_type"] is not None
                    or fc1_wgrad_gemm_kwargs["extra_output"] is not None
                    or fc1_wgrad_gemm_kwargs["bulk_overlap"]
                ):
                    raise NotImplementedError(
                        "Delayed weight grad computation is not supported "
                        "with Userbuffers (tensor-parallel communication overlapping)"
                    )
                args.wgrad_store.put([ln_out_total, dact], fc1_wgrad_gemm)
                if fuse_gemm_and_bias_fc1_wgrad:
                    fc1_bias_grad = None
            else:

                # Call wgrad GEMM now
                fc1_wgrad_outputs = fc1_wgrad_gemm(ln_out_total, dact)
                if fuse_gemm_and_bias_fc1_wgrad:
                    fc1_wgrad, fc1_bias_grad = fc1_wgrad_outputs
                else:
                    fc1_wgrad, _ = fc1_wgrad_outputs

                # Deallocate tensors if permitted
                clear_tensor_data(dact)
                if not args.return_layernorm_output_gathered:
                    clear_tensor_data(ln_out_total)

            # Update grad input if overlapping reduce-scatter with wgrad GEMM
            if ub_bulk_wgrad:
                if ub_obj_fc1_wgrad.is_fp8_ubuf():
                    fc1_dgrad = reduce_scatter_out
                else:
                    fc1_dgrad = ub_obj_fc1_wgrad.get_buffer(local_chunk=True).clone()

        # --------------------------------------------------
        # Finished FC1 WGRAD...
        # --------------------------------------------------

        # Make sure all tensor-parallel communication is finished
        if ln_out_total_work is not None:
            ln_out_total_work.wait()
            ln_out_total_work = None
        if fc1_dgrad_work is not None:
            fc1_dgrad_work.wait()
            fc1_dgrad_work = None

        # Residual gradient
        dgrad = fc1_dgrad.view(inputmat.shape)
        if args.return_layernorm_output and not args.return_layernorm_output_gathered:
            dgrad = dgrad + args.grad_ln_out.view_as(dgrad)

        # Norm gradient
        dgamma = None
        dbeta = None
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
        elif args.normalization == "RMSNorm":
            dgrad, dgamma = tex.rmsnorm_bwd(
                dgrad,
                inputmat,
                rsigma,
                ln_weight,
                args.bwd_ln_sm_margin,
                args.zero_centered_gamma,
            )
            dbeta = None
    clear_tensor_data(mu, rsigma)

    if args.fc1_weight_requires_grad:
        # Handle custom DDP from mcore.
        if args.fuse_wgrad_accumulation and hasattr(
            fc1_weight_python_object, "grad_added_to_main_grad"
        ):
            fc1_weight_python_object.grad_added_to_main_grad = True
            if getattr(fc1_weight_python_object, "zero_out_wgrad", False):
                fc1_wgrad = torch.zeros(
                    fc1_weight_main_grad.shape,
                    dtype=fc1_weight_python_object.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
            else:
                fc1_wgrad = torch.empty(
                    fc1_weight_main_grad.shape,
                    dtype=fc1_weight_python_object.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
        elif args.fuse_wgrad_accumulation:
            fc1_wgrad = None
    else:
        fc1_wgrad = None

    if args.fc2_weight_requires_grad:
        # Handle custom DDP from mcore.
        if args.fuse_wgrad_accumulation and hasattr(
            fc2_weight_python_object, "grad_added_to_main_grad"
        ):
            fc2_weight_python_object.grad_added_to_main_grad = True
            if getattr(fc2_weight_python_object, "zero_out_wgrad", False):
                fc2_wgrad = torch.zeros(
                    fc2_weight_main_grad.shape,
                    dtype=fc2_weight_python_object.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
            else:
                fc2_wgrad = torch.empty(
                    fc2_weight_main_grad.shape,
                    dtype=fc2_weight_python_object.dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
        elif args.fuse_wgrad_accumulation:
            fc2_wgrad = None
    else:
        fc2_wgrad = None

    # FIX THIS
    # Scatter Fp8 tranposed-weight buffers
    # if args.fp8:
    #    _fsdp_scatter_tensors(
    #        args.fsdp_group,
    #        fc1_weight_fp8 if not isinstance(fc1_weight, Float8Tensor) else None,
    #        fc2_weight_fp8 if not isinstance(fc2_weight, Float8Tensor) else None,
    #    )
    return (
        dgrad.view(args.inp_shape) if args.requires_dgrad else None,
        dgamma,
        dbeta,
        fc1_wgrad,
        fc1_bias_grad if fc1_bias is not None else None,
        fc2_wgrad,  # pylint: disable=possibly-used-before-assignment
        fc2_bias_grad,
    )


class _LayerNormMLP(torch.autograd.Function):
    """LayerNormMLP semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: Optional[torch.Tensor],
        fc1_weight: torch.Tensor,
        fc1_bias: Optional[torch.Tensor],
        fc2_weight: torch.Tensor,
        fc2_bias: Optional[torch.Tensor],
        fwd_args: LayerNormMLPFwdArgs,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """Forward pass: compute the output and set up the autograd context.

        The tensors are positional so autograd tracks them; they are
        re-attached to ``fwd_args`` so every downstream helper takes a single
        argument. The weight workspaces are non-differentiable cached tensors
        passed in via ``fwd_args`` and the freshly produced workspaces are
        returned as separate outputs so the module can refresh its cache.
        """
        fwd_args.inp = inp
        fwd_args.ln_weight = ln_weight
        fwd_args.ln_bias = ln_bias
        fwd_args.fc1_weight = fc1_weight
        fwd_args.fc1_bias = fc1_bias
        fwd_args.fc2_weight = fc2_weight
        fwd_args.fc2_bias = fc2_bias
        (
            out,
            ln_out_return,
            new_fc1_weight_workspace,
            new_fc2_weight_workspace,
            tensors_to_save_from_forward,
            ctx_attrs,
        ) = _layernorm_mlp_forward_impl(fwd_args)
        if ctx is not None:
            bwd_args = LayerNormMLPBwdArgs()
            tensors_to_save_from_setup = _layernorm_mlp_setup_ctx(
                bwd_args,
                fwd_args,
                (out, ln_out_return, new_fc1_weight_workspace, new_fc2_weight_workspace),
                ctx_attrs,
                tensors_to_save_from_forward,
            )
            tensors_to_save, tensor_objects = prepare_for_saving(*tensors_to_save_from_setup)
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects
            ctx.backward_objects = bwd_args
            if not bwd_args.checkpoint and fwd_args.fp8 and fwd_args.any_requires_grad():
                bwd_args.reduce_and_update_bwd_fp8_tensors = check_fp8_reduce_and_update()

        return out, ln_out_return, new_fc1_weight_workspace, new_fc2_weight_workspace

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
        grad_ln_out: Optional[torch.Tensor],
        _grad_fc1_weight_workspace,
        _grad_fc2_weight_workspace,
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """Backward pass: compute gradients and reduce FP8 scaling factors."""
        bwd_args: LayerNormMLPBwdArgs = ctx.backward_objects
        bwd_args.grad_output = grad_output
        bwd_args.grad_ln_out = grad_ln_out
        with get_nvtx_range_context("_LayerNormMLP_backward"):
            _layernorm_mlp_recompute(bwd_args, ctx)
        (
            dgrad,
            dgamma,
            dbeta,
            fc1_wgrad,
            fc1_bias_grad,
            fc2_wgrad,
            fc2_bias_grad,
        ) = _layernorm_mlp_backward_impl(bwd_args)
        reduce_and_update_bwd_fp8_tensors = bwd_args.reduce_and_update_bwd_fp8_tensors
        # Drop all references held by bwd_args (saved tensors, quantizers, weakrefs,
        # main_grad closures) so they don't outlive backward via ctx under retain_graph.
        ctx.backward_objects = None
        del bwd_args
        if reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
        return (
            dgrad,
            dgamma,
            dbeta,
            fc1_wgrad,
            fc1_bias_grad,
            fc2_wgrad,
            fc2_bias_grad,
            None,  # fwd_args
        )


class LayerNormMLP(TransformerEngineBaseModule):
    r"""
    Applies layer normalization on the input followed by the MLP module, consisting of
    2 successive linear transformations, separated by the activation function.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    ffn_hidden_size : int
                     intermediate size to which input samples are projected.
    eps : float, default = 1e-5
         a value added to the denominator of layer normalization for numerical stability.
    bias : bool, default = True
          if set to ``False``, the FC1 and FC2 layers will not learn an additive bias.
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    activation : str, default = 'gelu'
          activation function used.
          Options: ``'gelu'``, ``'geglu'``, ``'glu'``, ``'qgelu'``, ``'qgeglu'``, ``'relu'``, ``'reglu'``, ``'srelu'``, ``'sreglu'``,
          ``'silu'``, ``'swiglu'``, and ``'clamped_swiglu'``.
    activation_params : dict, default = None
                        Additional parameters for the activation function.
                        At the moment, only used for ``'clamped_swiglu'`` activation which
                        supports ``'limit'``, ``'alpha'``, and ``'glu_linear_offset'`` parameters.
    init_method : Callable, default = None
                 used for initializing FC1 weights in the following way: ``init_method(weight)``.
                 When set to ``None``, defaults to ``torch.nn.init.normal_(mean=0.0, std=0.023)``.
    output_layer_init_method : Callable, default = None
                              used for initializing FC2 weights in the following way:
                              ``output_layer_init_method(weight)``. When set to ``None``, defaults to
                              ``torch.nn.init.normal_(mean=0.0, std=0.023)``.
    return_layernorm_output : bool, default = False
                             if set to ``True``, output of layernorm is returned from the :meth:`forward` method
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module
                             is taken post layernorm.
    return_layernorm_output_gathered : bool, default = False
                             if set to ``True``, output of layernorm is returned after the all
                             gather operation. Ignored if ``return_layernorm_output`` is False.
                             Example use case: with sequence parallel, input to residual connection
                             for transformer module (e.g. LoRA) will need to be gathered.
                             Returning layernorm output gathered will prevent a redundant gather.
    zero_centered_gamma : bool, default = False
                         if set to ``True``, gamma parameter in LayerNorm is initialized to 0 and
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
    set_parallel_mode : bool, default = False
                      if set to ``True``, FC1 is used as Column Parallel and FC2 is used as Row
                      Parallel as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
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

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = False
                             if set to ``True``, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional ``main_grad`` attribute (used instead of the
                             regular ``grad``) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in. This argument along with
                             weight tensor having attribute ``'overwrite_main_grad'`` set to True
                             will overwrite ``main_grad`` instead of accumulating.
    return_bias : bool, default = False
                 when set to ``True``, this module will not apply the additive bias for FC2, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    params_dtype : torch.dtype, default = torch.get_default_dtype()
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    seq_length : int
               sequence length of input samples. Needed for JIT Warmup, a technique where jit fused
               functions are warmed up before training to ensure same kernels are used for forward
               propogation and activation recompute phase.
    micro_batch_size : int
                     batch size per training step. Needed for JIT Warmup, a technique where jit
                     fused functions are warmed up before training to ensure same kernels are
                     used for forward propogation and activation recompute phase.
    delay_wgrad_compute : bool, default = False
                         Whether or not to delay weight gradient computation. If set to ``True``,
                         it's the user's responsibility to call :meth:`backward_dw` to compute
                         weight gradients.
    symmetric_ar_type : {None, 'multimem_all_reduce', 'two_shot', 'one_shot'}, default = None
                   Type of symmetric memory all-reduce to use during the forward pass.
                   This can help in latency bound communication situations.
                   Requires PyTorch version 2.7.0 or higher. When set to ``None``, standard all-reduce
                   is used.
    checkpoint : bool, default = False
                whether to use selective activation checkpointing, where activations are not saved for bwd,
                and instead are recomputed (skipping fc2, as it is not needed for backward). Trades compute
                for memory. default is false, in which activations are saved in fwd. not supported for onnx forward
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        return_bias: bool = False,
        get_rng_state_tracker: Optional[Callable] = None,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        normalization: str = "LayerNorm",
        activation: str = "gelu",
        activation_params: Optional[dict] = None,
        output_layer_init_method: Optional[Callable] = None,
        fuse_wgrad_accumulation: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        return_layernorm_output: bool = False,
        return_layernorm_output_gathered: bool = False,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        set_parallel_mode: bool = False,
        zero_centered_gamma: bool = False,
        device: Union[torch.device, str] = "cuda",
        ub_overlap_ag: bool = False,
        name: Optional[str] = None,
        ub_overlap_rs: bool = False,
        ub_overlap_rs_dgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_bulk_wgrad: bool = False,
        delay_wgrad_compute: bool = False,
        symmetric_ar_type: Optional[str] = None,
        checkpoint: bool = False,
    ) -> None:
        super().__init__(name)

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.normalization = normalization
        assert normalization in ["LayerNorm", "RMSNorm"], "Unsupported normalization type!"
        self.use_bias = bias
        self.activation = activation
        self.activation_params = activation_params
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.return_layernorm_output = return_layernorm_output
        self.return_layernorm_output_gathered = return_layernorm_output_gathered
        self.bias_gelu_nvfusion = (
            bool(int(os.getenv("NVTE_BIAS_GELU_NVFUSION", "1"))) and self.activation == "gelu"
        )
        self.set_parallel_mode = set_parallel_mode
        self.zero_centered_gamma = zero_centered_gamma
        self.symmetric_ar_type = symmetric_ar_type
        self.checkpoint = checkpoint

        # GEMM-GELU fusion is currently only supported with split GEMM-AG overlap
        self.gemm_gelu_fusion = (
            bool(int(os.getenv("NVTE_GEMM_GELU_FUSION", "0")))
            and self.activation == "gelu"
            and all(
                ("fc1_fprop", use_fp8) not in _ub_communicators
                or not get_ub("fc1_fprop", use_fp8).is_atomic_gemm()
                for use_fp8 in [False, True]
            )
        )

        self.wgrad_store = WeightGradStore(delay_wgrad_compute, ub_bulk_wgrad)

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        if init_method is None:
            init_method = get_default_init_method()
        if output_layer_init_method is None:
            output_layer_init_method = get_default_init_method()

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel
        self.size_per_partition = divide(ffn_hidden_size, self.tp_size)

        self.ub_overlap_ag = ub_overlap_ag and self.sequence_parallel
        self.ub_overlap_rs = ub_overlap_rs and self.sequence_parallel
        self.ub_overlap_rs_dgrad = ub_overlap_rs_dgrad and self.sequence_parallel
        self.ub_bulk_wgrad = (
            ub_bulk_wgrad and self.sequence_parallel and not self.ub_overlap_rs_dgrad
        )
        self.ub_bulk_dgrad = (
            ub_bulk_dgrad and self.sequence_parallel and not self.ub_overlap_rs_dgrad
        )

        if any(
            [
                self.ub_overlap_ag,
                self.ub_overlap_rs,
                self.ub_overlap_rs_dgrad,
                self.ub_bulk_dgrad,
                self.ub_bulk_wgrad,
            ]
        ):
            assert is_ub_initialized(), "initialize_ub() must be called before layer construction."

        if using_cublasmp_backend():
            if self.ub_bulk_dgrad:
                warnings.warn(
                    "cuBLASMp backend does not support bulk overlaps for 'fc1_dgrad' and "
                    "'fc1_wgrad' GEMMs. Falling back on DGRAD+RS overlap for 'fc1_dgrad' GEMM with "
                    "no bulk overlap for 'fc1_wgrad' GEMM. In order to enable bulk overlaps for "
                    "these GEMMs, set `with_cublasmp=False` when calling `initialize_ub()`."
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

        # Initialize params in FP8
        with_fp8_params = FP8GlobalStateManager.with_fp8_parameters()

        # LN init
        self.eps = eps
        layer_norm_weight = Parameter(torch.empty(hidden_size, device=device, dtype=params_dtype))
        self.register_parameter(
            "layer_norm_weight",
            layer_norm_weight,
            init_fn=init_method_constant(float(not self.zero_centered_gamma)),
        )
        if self.normalization != "RMSNorm":
            layer_norm_bias = Parameter(torch.empty(hidden_size, device=device, dtype=params_dtype))
            self.register_parameter(
                "layer_norm_bias", layer_norm_bias, init_fn=init_method_constant(0.0)
            )
        else:
            self.layer_norm_bias = None

        # FC1 init
        if self.activation in [
            "geglu",
            "glu",
            "qgeglu",
            "reglu",
            "sreglu",
            "swiglu",
            "clamped_swiglu",
        ]:
            fc1_output_features = 2 * self.size_per_partition
        else:
            fc1_output_features = self.size_per_partition

        fc1_weight = Parameter(
            torch.empty(fc1_output_features, hidden_size, device=device, dtype=params_dtype)
        )
        self.register_parameter(
            "fc1_weight",
            fc1_weight,
            init_fn=init_method,
            get_rng_state_tracker=get_rng_state_tracker,
            fp8_meta_index=FP8FwdTensorIdx.GEMM1_WEIGHT,
        )

        if self.use_bias:
            fc1_bias = Parameter(
                torch.empty(fc1_output_features, device=device, dtype=params_dtype)
            )
            self.register_parameter("fc1_bias", fc1_bias, init_fn=init_method_constant(0.0))
        else:
            self.fc1_bias = torch.Tensor().to(dtype=params_dtype, device=device)

        # FC2 init
        fc2_weight = Parameter(
            torch.empty(hidden_size, self.size_per_partition, device=device, dtype=params_dtype)
        )
        self.register_parameter(
            "fc2_weight",
            fc2_weight,
            init_fn=output_layer_init_method,
            get_rng_state_tracker=get_rng_state_tracker,
            fp8_meta_index=FP8FwdTensorIdx.GEMM2_WEIGHT,
        )

        if self.use_bias:
            fc2_bias = Parameter(torch.empty(hidden_size, device=device, dtype=params_dtype))
            self.register_parameter("fc2_bias", fc2_bias, init_fn=init_method_constant(0.0))
        else:
            self.fc2_bias = torch.Tensor().to(dtype=params_dtype, device=device)

        if with_fp8_params:
            self.init_fp8_metadata(num_gemms=2)

        self.reset_parameters(defer_init=device == "meta")

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.set_parallel_mode and self.apply_bias:
            self.gemm_bias_unfused_add = True
        else:
            self.gemm_bias_unfused_add = False

        if self.bias_gelu_nvfusion:
            set_jit_fusion_options()
            if seq_length and micro_batch_size:
                warmup_jit_bias_gelu_all_dtypes(
                    self.size_per_partition, seq_length, micro_batch_size
                )
        if self.wgrad_store.delay_wgrad_compute():
            for name, param in self.named_parameters():
                if name in ["fc1_weight", "fc2_weight", "fc1_bias", "fc2_bias"]:
                    param.skip_backward_post_hook = True

        # These many SMs are subtracted from the total SM count when calling forward
        # and backward LayerNorm C APIs. These envvars can be used to prevent the LN
        # kernels from using all SMs in the device. This is useful for cases such as
        # communication overlap with LN.
        self.fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
        self.bwd_ln_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))
        self.inf_ln_sm_margin = int(os.getenv("NVTE_INF_LAYERNORM_SM_MARGIN", "0"))

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
        """QuantizerRole list for quantizers used by ``LayerNormMLP``.

        Each internal GEMM (fc1, fc2) gets a distinct name suffix so that
        custom-recipe factories can target them individually.

        The module's final output (fc2 fwd) and final grad (fc1 bwd)
        slots default to ``None`` (unknown consumer).  Set
        :attr:`output_quantizer_role` / :attr:`grad_input_quantizer_role`
        to provide consumer identity.  Internal boundaries use fixed
        roles with known consumer identity.
        """
        base_name = self.name or ""
        fc1_name = f"{base_name}.fc1" if base_name else "fc1"
        fc2_name = f"{base_name}.fc2" if base_name else "fc2"
        # Roles use the *consumer's* identity: internal boundary tensors are
        # labeled with the downstream module that will consume them.
        #
        # Forward:  fc1_input -> fc1 GEMM -> [act] -> fc2_input -> fc2 GEMM -> output
        # Backward: grad_input <- fc1 GEMM <- [act'] <- fc2 GEMM <- grad_output
        if fwd:
            base = [
                QuantizerRole(module_type="linear", tensor_type="input", name=fc1_name),
                QuantizerRole(module_type="linear", tensor_type="weight", name=fc1_name),
                # fc1 output — consumed by fc2 (via activation), so labeled as fc2 input
                QuantizerRole(module_type="linear", tensor_type="input", name=fc2_name),
                QuantizerRole(module_type="linear", tensor_type="input", name=fc2_name),
                QuantizerRole(module_type="linear", tensor_type="weight", name=fc2_name),
                # fc2 output — boundary, consumer unknown
                self._output_quantizer_role,
            ]
        else:
            base = [
                QuantizerRole(module_type="linear", tensor_type="grad_output", name=fc1_name),
                # fc1 grad_input — boundary, consumer unknown
                self._grad_input_quantizer_role,
                QuantizerRole(module_type="linear", tensor_type="grad_output", name=fc2_name),
                # fc2 grad_input — consumed by fc1 (via activation'), so labeled as fc1 grad_output
                QuantizerRole(module_type="linear", tensor_type="grad_output", name=fc1_name),
            ]
        return [base[i % len(base)] for i in range(num_quantizers)]

    def reset_layer_norm_parameters(self) -> None:
        """Init LN params"""
        warnings.warn(
            "This method will be deprecated in an upcoming release. "
            "Update your code to use LayerNormMLP.reset_parameters() instead.",
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
            # Set parallel attributes for layer norm parameters
            setattr(self.layer_norm_weight, "sequence_parallel", self.sequence_parallel)
            if self.normalization != "RMSNorm":
                setattr(self.layer_norm_bias, "sequence_parallel", self.sequence_parallel)

            # Set parallel attributes for linear parameters
            set_tensor_model_parallel_attributes(self.fc1_weight, True, 0, 1)
            set_tensor_model_parallel_attributes(self.fc2_weight, True, 1, 1)
            if self.use_bias:
                set_tensor_model_parallel_attributes(self.fc1_bias, True, 0, 1)
                if self.set_parallel_mode:
                    setattr(self.fc2_bias, "sequence_parallel", self.sequence_parallel)

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply layer normalization to the input followed by a feedforward network (MLP Block).

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
            return self.onnx_forward(inp, is_grad_enabled)

        debug = self.is_debug_iter()

        if FP8GlobalStateManager.fp8_graph_capturing():
            skip_fp8_weight_update = (
                FP8GlobalStateManager.quantization_state.skip_fp8_weight_update_tensor
            )
        else:
            skip_fp8_weight_update = None
        if skip_fp8_weight_update is not None:
            is_first_microbatch = False

        fp8_output = False
        if self.ub_overlap_rs:
            if get_ub_is_fp8("fc2_fprop", FP8GlobalStateManager.is_fp8_enabled()):
                fp8_output = True

        inp = self.prepare_forward(inp, num_gemms=2)

        try:
            quantizers = (
                self._get_quantizers(fp8_output, is_grad_enabled)
                if not debug
                else self._get_debug_quantizers(fp8_output, is_grad_enabled)
            )
            if debug:
                if self.no_debug_features_active(quantizers):
                    debug = False
                    quantizers = self._get_quantizers(fp8_output, is_grad_enabled)

            # Get quantizers
            (
                fc1_input_quantizer,
                fc1_weight_quantizer,
                fc1_output_quantizer,
                fc1_grad_input_quantizer,
                fc1_grad_weight_quantizer,
                fc1_grad_output_quantizer,
                fc2_input_quantizer,
                fc2_weight_quantizer,
                fc2_output_quantizer,
                fc2_grad_input_quantizer,
                fc2_grad_weight_quantizer,
                fc2_grad_output_quantizer,
            ) = quantizers

            # Get weight tensors
            fc1_weight, fc2_weight = self._get_weight_tensors()
            fc1_bias = self.fc1_bias if self.use_bias else None
            fc2_bias = self.fc2_bias if self.use_bias else None
            if not debug:
                if fc1_weight_quantizer is not None:
                    fc1_weight_quantizer.optimize_for_gemm = self._enable_weight_preswizzle(
                        fc1_weight_quantizer, fc1_weight
                    )
                if fc2_weight_quantizer is not None:
                    fc2_weight_quantizer.optimize_for_gemm = self._enable_weight_preswizzle(
                        fc2_weight_quantizer, fc2_weight
                    )
            if not self.fp8:
                if isinstance(fc1_weight, Float8Tensor):
                    fc1_weight = fc1_weight.dequantize()
                if isinstance(fc2_weight, Float8Tensor):
                    fc2_weight = fc2_weight.dequantize()

            # Disable bias_gelu_nvfusion for determinism checkpointing in non-reentrant mode
            if self.bias_gelu_nvfusion and not use_reentrant_activation_recompute():
                self.fast_setattr("bias_gelu_nvfusion", False)

            cache_name_fc1 = (
                None if (is_first_microbatch is None or self.is_fsdp2) else "fc1_weight"
            )
            cache_name_fc2 = (
                None if (is_first_microbatch is None or self.is_fsdp2) else "fc2_weight"
            )
            fc1_weight_workspace = (
                self._fp8_workspaces.get(cache_name_fc1) if cache_name_fc1 is not None else None
            )
            fc2_weight_workspace = (
                self._fp8_workspaces.get(cache_name_fc2) if cache_name_fc2 is not None else None
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
                ub_overlap_ag = False
                ub_overlap_rs = False
                ub_overlap_rs_dgrad = False
                ub_bulk_wgrad = False
                ub_bulk_dgrad = False
            else:
                ub_overlap_ag = self.ub_overlap_ag
                ub_overlap_rs = self.ub_overlap_rs
                ub_overlap_rs_dgrad = self.ub_overlap_rs_dgrad
                ub_bulk_wgrad = self.ub_bulk_wgrad
                ub_bulk_dgrad = self.ub_bulk_dgrad

            fc2_bias_tensor = (
                fc2_bias if self.apply_bias and not self.gemm_bias_unfused_add else None
            )
            fwd_args = LayerNormMLPFwdArgs(
                # tensors
                inp=inp,
                ln_weight=self.layer_norm_weight,
                ln_bias=self.layer_norm_bias,
                fc1_weight=fc1_weight,
                fc1_bias=fc1_bias,
                fc2_weight=fc2_weight,
                fc2_bias=fc2_bias_tensor,
                fc1_weight_workspace=fc1_weight_workspace,
                fc2_weight_workspace=fc2_weight_workspace,
                # requires_grad flags
                input_requires_grad=inp.requires_grad,
                ln_weight_requires_grad=self.layer_norm_weight.requires_grad,
                ln_bias_requires_grad=(
                    self.layer_norm_bias.requires_grad
                    if self.layer_norm_bias is not None
                    else False
                ),
                fc1_weight_requires_grad=fc1_weight.requires_grad,
                fc1_bias_requires_grad=fc1_bias.requires_grad if fc1_bias is not None else False,
                fc2_weight_requires_grad=fc2_weight.requires_grad,
                fc2_bias_requires_grad=(
                    fc2_bias_tensor.requires_grad if fc2_bias_tensor is not None else False
                ),
                # quantizers
                fc1_input_quantizer=fc1_input_quantizer,
                fc1_weight_quantizer=fc1_weight_quantizer,
                fc1_output_quantizer=fc1_output_quantizer,
                fc1_grad_input_quantizer=fc1_grad_input_quantizer,
                fc1_grad_weight_quantizer=fc1_grad_weight_quantizer,
                fc1_grad_output_quantizer=fc1_grad_output_quantizer,
                fc2_input_quantizer=fc2_input_quantizer,
                fc2_weight_quantizer=fc2_weight_quantizer,
                fc2_output_quantizer=fc2_output_quantizer,
                fc2_grad_input_quantizer=fc2_grad_input_quantizer,
                fc2_grad_weight_quantizer=fc2_grad_weight_quantizer,
                fc2_grad_output_quantizer=fc2_grad_output_quantizer,
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
                # activation
                activation=self.activation,
                activation_params=self.activation_params,
                bias_gelu_fusion=self.bias_gelu_nvfusion and not self.fp8 and not debug,
                gemm_gelu_fusion=self.gemm_gelu_fusion and not debug,
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
                cache_weight=cache_name_fc1 is not None,
                skip_fp8_weight_update=skip_fp8_weight_update,
                # tensor / sequence parallelism
                set_parallel_mode=self.set_parallel_mode,
                tp_group=self.tp_group,
                tp_size=self.tp_size,
                tensor_parallel=self.tp_size > 1,
                sequence_parallel=self.sequence_parallel,
                symmetric_ar_type=self.symmetric_ar_type,
                # userbuffers
                ub_overlap_ag=ub_overlap_ag,
                ub_overlap_rs=ub_overlap_rs,
                ub_overlap_rs_dgrad=ub_overlap_rs_dgrad,
                ub_bulk_dgrad=ub_bulk_dgrad,
                ub_bulk_wgrad=ub_bulk_wgrad,
                # FSDP
                fsdp_group=self.fsdp_group,
                is_fsdp2=self.is_fsdp2,
                # weight-grad scheduling
                fuse_wgrad_accumulation=self.fuse_wgrad_accumulation,
                wgrad_store=self.wgrad_store,
                # activation checkpointing
                checkpoint=self.checkpoint,
                fp8_meta=self.fp8_meta if self.checkpoint else None,
                recompute_for_bwd=False,
                # misc
                cpu_offloading=is_cpu_offload_enabled(),
                is_grad_enabled=is_grad_enabled,
            )

            if is_grad_enabled:
                out, ln_out, new_fc1_ws, new_fc2_ws = _LayerNormMLP.apply(
                    inp,
                    self.layer_norm_weight,
                    self.layer_norm_bias,
                    fc1_weight,
                    fc1_bias,
                    fc2_weight,
                    fc2_bias_tensor,
                    fwd_args,
                )
            else:
                out, ln_out, new_fc1_ws, new_fc2_ws = _LayerNormMLP.forward(
                    None,
                    inp,
                    self.layer_norm_weight,
                    self.layer_norm_bias,
                    fc1_weight,
                    fc1_bias,
                    fc2_weight,
                    fc2_bias_tensor,
                    fwd_args,
                )

            if new_fc1_ws is not None and cache_name_fc1 is not None:
                if isinstance(new_fc1_ws, torch.Tensor):
                    new_fc1_ws = new_fc1_ws.detach()
                self._fp8_workspaces[cache_name_fc1] = new_fc1_ws
            if new_fc2_ws is not None and cache_name_fc2 is not None:
                if isinstance(new_fc2_ws, torch.Tensor):
                    new_fc2_ws = new_fc2_ws.detach()
                self._fp8_workspaces[cache_name_fc2] = new_fc2_ws

        finally:
            self.end_forward()

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(fc2_bias, self.activation_dtype)

        if self.return_bias:
            if self.return_layernorm_output:
                return out, cast_if_needed(fc2_bias, self.activation_dtype), ln_out
            return out, cast_if_needed(fc2_bias, self.activation_dtype)
        if self.return_layernorm_output:
            return out, ln_out
        return out

    def _get_quantizers(self, fp8_output, is_grad_enabled):
        if self.fp8:
            self._warn_missing_output_quantizer_role(fp8_output, False)

        (
            fc1_input_quantizer,
            fc1_output_quantizer,
            fc1_grad_input_quantizer,
            fc1_grad_weight_quantizer,
            fc1_grad_output_quantizer,
            fc2_input_quantizer,
            fc2_output_quantizer,
            fc2_grad_input_quantizer,
            fc2_grad_weight_quantizer,
            fc2_grad_output_quantizer,
        ) = [None] * 10
        fc1_weight_quantizer, fc2_weight_quantizer = self._get_weight_quantizers()
        if self.fp8 or self.fp8_calibration:
            fc1_input_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_INPUT]
            fc1_input_quantizer.internal = True
            if not self.sequence_parallel:
                fc1_input_quantizer.optimize_for_gemm = True
            fc2_input_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM2_INPUT]
            fc2_input_quantizer.set_usage(
                rowwise=True,
                columnwise=isinstance(
                    fc2_input_quantizer,
                    (MXFP8Quantizer, Float8BlockQuantizer, NVFP4Quantizer, HybridQuantizer),
                ),
            )
            fc2_input_quantizer.internal = True
            fc2_input_quantizer.optimize_for_gemm = True
            if fp8_output:
                fc2_output_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM2_OUTPUT]
            if is_grad_enabled:
                fc2_grad_output_quantizer = self.quantizers["scaling_bwd"][
                    FP8BwdTensorIdx.GRAD_OUTPUT2
                ]
                fc2_grad_output_quantizer.internal = True
                if not self.sequence_parallel:
                    fc2_grad_output_quantizer.optimize_for_gemm = True
                fc1_grad_output_quantizer = self.quantizers["scaling_bwd"][
                    FP8BwdTensorIdx.GRAD_OUTPUT1
                ]
                fc1_grad_output_quantizer.internal = True
                fc1_grad_output_quantizer.optimize_for_gemm = True

        return (
            fc1_input_quantizer,
            fc1_weight_quantizer,
            fc1_output_quantizer,
            fc1_grad_input_quantizer,
            fc1_grad_weight_quantizer,
            fc1_grad_output_quantizer,
            fc2_input_quantizer,
            fc2_weight_quantizer,
            fc2_output_quantizer,
            fc2_grad_input_quantizer,
            fc2_grad_weight_quantizer,
            fc2_grad_output_quantizer,
        )

    def onnx_forward(
        self, inp: torch.Tensor, is_grad_enabled: bool
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        ONNX-compatible version of the :meth:`forward` method that provides numerical equivalence
        while only using operations that have defined ONNX symbolic translations.
        This simplified implementation is designed specifically for inference scenarios.
        """
        from ..export import onnx_layernorm, onnx_gemm

        assert not TEDebugState.debug_enabled, "Debug mode is not supported in ONNX export"
        assert_warmed_up(self)

        # Get quantizers
        (
            fc1_input_quantizer,
            fc1_weight_quantizer,
            _,
            _,
            _,
            _,
            fc2_input_quantizer,
            fc2_weight_quantizer,
            fc2_output_quantizer,
            _,
            _,
            _,
        ) = self._get_quantizers(False, is_grad_enabled)

        inp_dtype = inp.dtype

        fc1_weight, fc2_weight = self._get_weight_tensors()
        fc1_bias = self.fc1_bias if self.use_bias else None
        fc2_bias = self.fc2_bias if self.use_bias else None

        # layernorm + fp8 cast
        ln_out, ln_out_return = onnx_layernorm(
            inp,
            self.layer_norm_weight,
            self.layer_norm_bias,
            self.eps,
            self.normalization,
            self.zero_centered_gamma,
            inp_dtype,
            self.return_layernorm_output,
            fc1_input_quantizer,
        )

        if fc1_weight_quantizer is not None:
            fc1_weight_q = fc1_weight_quantizer.onnx_quantize(fc1_weight)
            fc1_weight = fc1_weight_quantizer.onnx_dequantize(fc1_weight_q)
        fc1_weight = fc1_weight.to(inp_dtype)

        fc1_out = onnx_gemm(fc1_weight, ln_out, fc1_bias)

        fc1_out = fc1_out.to(torch.float32)  # activation is computed in fp32
        act_params = self.activation_params or {}
        clamped_swiglu_limit = act_params.get("limit", 7.0)
        clamped_swiglu_alpha = act_params.get("alpha", 1.702)
        clamped_swiglu_offset = act_params.get("glu_linear_offset", 1.0)

        def _clamped_swiglu(x, limit, alpha, offset):
            x_glu, x_linear = x.chunk(2, dim=-1)
            x_glu = x_glu.clamp(min=None, max=limit)
            x_linear = x_linear.clamp(min=-limit, max=limit)
            out_glu = x_glu * torch.sigmoid(alpha * x_glu)
            y = out_glu * (x_linear + offset)
            return y

        activation_map = {
            "gelu": lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
            "geglu": lambda x: torch.nn.functional.gelu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1],
            "glu": lambda x: torch.sigmoid(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1],
            "qgelu": lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
            "qgeglu": lambda x: torch.nn.functional.gelu(x.chunk(2, -1)[0], approximate="tanh")
            * x.chunk(2, -1)[1],
            "relu": torch.nn.functional.relu,
            "reglu": lambda x: torch.nn.functional.relu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1],
            "srelu": lambda x: torch.nn.functional.relu(x) ** 2,
            "sreglu": lambda x: torch.nn.functional.relu(x.chunk(2, -1)[0]) ** 2
            * x.chunk(2, -1)[1],
            "silu": torch.nn.functional.silu,
            "swiglu": lambda x: torch.nn.functional.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1],
            "clamped_swiglu": lambda x: _clamped_swiglu(
                x, clamped_swiglu_limit, clamped_swiglu_alpha, clamped_swiglu_offset
            ),
        }
        if self.activation not in activation_map:
            raise ValueError(f"Unsupported activation in onnx export: {self.activation}")
        act_out = activation_map[self.activation](fc1_out)
        if fc2_weight_quantizer is not None:
            fc2_weight_q = fc2_weight_quantizer.onnx_quantize(fc2_weight)
            fc2_weight = fc2_weight_quantizer.onnx_dequantize(fc2_weight_q)
        fc2_weight = fc2_weight.to(inp_dtype)

        if fc2_input_quantizer is not None:
            act_out_q = fc2_input_quantizer.onnx_quantize(act_out)
            act_out = fc2_input_quantizer.onnx_dequantize(act_out_q)
        act_out = act_out.to(inp_dtype)

        fc2_out = onnx_gemm(fc2_weight, act_out, fc2_bias)

        if fc2_output_quantizer is not None:
            raise NotImplementedError("ONNX export of quantized output is not supported")

        if self.return_layernorm_output:
            if self.return_bias:
                return fc2_out, fc2_bias.to(inp_dtype), ln_out_return
            return fc2_out, ln_out_return
        if self.return_bias:
            return fc2_out, fc2_bias.to(inp_dtype)
        return fc2_out

    def _get_debug_quantizers(self, fp8_output, is_grad_enabled):
        from ...debug.pytorch.debug_quantization import DebugQuantizer

        base_quantizers = list(self._get_quantizers(fp8_output, is_grad_enabled))
        assert TEDebugState.debug_enabled

        def make_debug(prefix, offset):
            labels = ["activation", "weight", "output", "dgrad", "wgrad", "gradient"]
            return [
                DebugQuantizer(
                    f"{self.name}.{prefix}",
                    label,
                    None if label in ("dgrad", "wgrad") else base_quantizers[i + offset],
                    self.tp_group,
                    self.tp_size,
                )
                for i, label in enumerate(labels)
            ]

        return tuple(make_debug("fc1", 0) + make_debug("fc2", 6))

    def _customize_quantizers_float8_current_scaling(self, fwd: bool, recipe: Recipe) -> None:
        """Customize quantizers based on current scaling recipe + layernorm_mlp."""
        assert (
            recipe.float8_current_scaling()
        ), "current scaling recipe quantizer customization here"
        if fwd:
            # fc1_input_quantizer: set configs about amax epsilon and power_2_scale
            self.quantizers["scaling_fwd"][
                FP8FwdTensorIdx.GEMM1_INPUT
            ].force_pow_2_scales = recipe.fp8_quant_fwd_inp.power_2_scale
            self.quantizers["scaling_fwd"][
                FP8FwdTensorIdx.GEMM1_INPUT
            ].amax_epsilon = recipe.fp8_quant_fwd_inp.amax_epsilon
            # fc2_input_quantizer
            self.quantizers["scaling_fwd"][
                FP8FwdTensorIdx.GEMM2_INPUT
            ].force_pow_2_scales = recipe.fp8_quant_fwd_inp.power_2_scale
            self.quantizers["scaling_fwd"][
                FP8FwdTensorIdx.GEMM2_INPUT
            ].amax_epsilon = recipe.fp8_quant_fwd_inp.amax_epsilon
            # fc1_weight_quantizer: also set numerical configs about weight
            self.quantizers["scaling_fwd"][
                FP8FwdTensorIdx.GEMM1_WEIGHT
            ].force_pow_2_scales = recipe.fp8_quant_fwd_weight.power_2_scale
            self.quantizers["scaling_fwd"][
                FP8FwdTensorIdx.GEMM1_WEIGHT
            ].amax_epsilon = recipe.fp8_quant_fwd_weight.amax_epsilon
            # fc2_weight_quantizer
            self.quantizers["scaling_fwd"][
                FP8FwdTensorIdx.GEMM2_WEIGHT
            ].force_pow_2_scales = recipe.fp8_quant_fwd_weight.power_2_scale
            self.quantizers["scaling_fwd"][
                FP8FwdTensorIdx.GEMM2_WEIGHT
            ].amax_epsilon = recipe.fp8_quant_fwd_weight.amax_epsilon
        else:
            # fc2_grad_output_quantizer: set configs about amax epsilon and power_2_scale for fc2_grad_output_quantizer
            self.quantizers["scaling_bwd"][
                FP8BwdTensorIdx.GRAD_OUTPUT2
            ].force_pow_2_scales = recipe.fp8_quant_bwd_grad.power_2_scale
            self.quantizers["scaling_bwd"][
                FP8BwdTensorIdx.GRAD_OUTPUT2
            ].amax_epsilon = recipe.fp8_quant_bwd_grad.amax_epsilon
            # fc1_grad_output_quantizer: also set numerical configs for fc1_grad_output_quantizer
            self.quantizers["scaling_bwd"][
                FP8BwdTensorIdx.GRAD_OUTPUT1
            ].force_pow_2_scales = recipe.fp8_quant_bwd_grad.power_2_scale
            self.quantizers["scaling_bwd"][
                FP8BwdTensorIdx.GRAD_OUTPUT1
            ].amax_epsilon = recipe.fp8_quant_bwd_grad.amax_epsilon

    def _get_weight_tensors(self) -> List[Union[torch.Tensor, QuantizedTensorStorage]]:
        """Get the weight tensors of the module."""
        return [self.fc1_weight, self.fc2_weight]

    def _get_weight_quantizers(self) -> List[Quantizer]:
        """Get the weight quantizers of the module."""
        if not self.fp8 and not self.fp8_calibration:
            return [None, None]
        fc1_weight_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_WEIGHT]
        fc1_weight_quantizer.internal = True
        fc2_weight_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM2_WEIGHT]
        fc2_weight_quantizer.internal = True
        return [fc1_weight_quantizer, fc2_weight_quantizer]

    def backward_dw(self):
        """
        Execute the delayed weight gradient computation.
        This method is called after the main backward pass to compute weight gradients.
        """
        if not self.need_backward_dw():
            return
        with get_nvtx_range_context("_LayerNormMLP_wgrad"):
            (fc2_wgrad, fc2_bias_grad_, *_), tensor_list_fc2 = self.wgrad_store.pop()
            if self.use_bias and self.fc1_bias.grad is None:
                (fc1_wgrad, fc1_bias_grad, *_), _ = self.wgrad_store.pop()
            else:
                (fc1_wgrad, *_), _ = self.wgrad_store.pop()
                fc1_bias_grad = None
            if self.use_bias:
                if self.fc2_bias.grad is None:
                    if (
                        self.fp8
                        and FP8GlobalStateManager.get_fp8_recipe().float8_block_scaling()
                        and self.apply_bias
                        and not self.gemm_bias_unfused_add
                    ):
                        act_out = tensor_list_fc2[0]
                        # BGRAD not fused with GEMM for float8 blockwise gemm.
                        fc2_bias_grad_ = act_out.view(-1, act_out.shape[-1]).sum(dim=0)
                    self.fc2_bias.grad = fc2_bias_grad_.to(self.fc2_bias.dtype)
                if self.fc1_bias.grad is None:
                    self.fc1_bias.grad = fc1_bias_grad.to(self.fc1_bias.dtype)
            if not self.fuse_wgrad_accumulation:
                self.fc2_weight.grad = fc2_wgrad.to(self.fc2_weight.dtype)
                self.fc1_weight.grad = fc1_wgrad.to(self.fc1_weight.dtype)
            del fc2_bias_grad_
            del fc2_wgrad
            del fc1_wgrad
            del fc1_bias_grad
            self._trigger_wgrad_accumulation_and_reduce_hooks()

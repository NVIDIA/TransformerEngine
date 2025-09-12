# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LayerNormMLP API"""
import os
import warnings
from typing import Callable, Optional, Tuple, Union, List
from functools import reduce
from operator import mul as multiply_op

import torch
from torch.nn.parameter import Parameter
from torch.nn import init

import transformer_engine_torch as tex

from transformer_engine.common.recipe import Recipe
from transformer_engine.pytorch import torch_version
from .base import (
    fill_userbuffers_buffer_for_all_gather,
    get_workspace,
    _ub_communicators,
    get_ub,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ..fp8 import FP8GlobalStateManager
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
    requires_grad,
    needs_quantized_gemm,
)
from ..distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    allreduce,
    symmetric_all_reduce,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
    use_reentrant_activation_recompute,
    in_fp8_activation_recompute_phase,
    _fsdp_scatter_tensors,
)
from ..constants import dist_group_type
from ..jit import no_torch_dynamo
from ..graph import is_graph_capturing
from ..tensor.float8_tensor import (
    Float8CurrentScalingQuantizer,
    Float8Quantizer,
    Float8Tensor,
)
from ..tensor.mxfp8_tensor import MXFP8Quantizer
from ..tensor.float8_blockwise_tensor import Float8BlockQuantizer
from ._common import apply_normalization, WeightGradStore
from ..cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ..tensor.quantized_tensor import (
    QuantizedTensorBase,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)
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
            "qgelu": (tex.qgelu, tex.dqgelu, None),
            "qgeglu": (tex.qgeglu, tex.dqgeglu, None),
            "relu": (tex.relu, tex.drelu, None),
            "reglu": (tex.reglu, tex.dreglu, None),
            "srelu": (tex.srelu, tex.dsrelu, None),
            "sreglu": (tex.sreglu, tex.dsreglu, None),
            "silu": (tex.silu, tex.dsilu, None),
            "swiglu": (tex.swiglu, tex.dswiglu, None),
        }
    if recipe.delayed() or recipe.mxfp8():
        # Delayed scaling, fusion supported list: [tex.dbias_dgelu, tex.dbias_drelu, tex.dbias_dqgelu, tex.dbias_dsrelu]
        # MXFP8: [tex.dbias_dgelu, tex.dbias_drelu, tex.dbias_dqgelu, tex.dbias_dsrelu]
        return {
            "gelu": (tex.gelu, tex.dgelu, tex.dbias_dgelu),
            "geglu": (tex.geglu, tex.dgeglu, None),
            "qgelu": (tex.qgelu, tex.dqgelu, tex.dbias_dqgelu),
            "qgeglu": (tex.qgeglu, tex.dqgeglu, None),
            "relu": (tex.relu, tex.drelu, tex.dbias_drelu),
            "reglu": (tex.reglu, tex.dreglu, None),
            "srelu": (tex.srelu, tex.dsrelu, tex.dbias_dsrelu),
            "sreglu": (tex.sreglu, tex.dsreglu, None),
            "silu": (tex.silu, tex.dsilu, tex.dbias_dsilu),
            "swiglu": (tex.swiglu, tex.dswiglu, None),
        }
    # no activation fusion written yet
    # Per-tensor current scaling or fp8 blockwise scaling: []
    if recipe.float8_current_scaling() or recipe.float8_block_scaling():
        return {
            "gelu": (tex.gelu, tex.dgelu, None),
            "geglu": (tex.geglu, tex.dgeglu, None),
            "qgelu": (tex.qgelu, tex.dqgelu, None),
            "qgeglu": (tex.qgeglu, tex.dqgeglu, None),
            "relu": (tex.relu, tex.drelu, None),
            "reglu": (tex.reglu, tex.dreglu, None),
            "srelu": (tex.srelu, tex.dsrelu, None),
            "sreglu": (tex.sreglu, tex.dsreglu, None),
            "silu": (tex.silu, tex.dsilu, None),
            "swiglu": (tex.swiglu, tex.dswiglu, None),
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


class _LayerNormMLP(torch.autograd.Function):
    """LayerNormMLP semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_bias: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_bias: torch.Tensor,
        eps: float,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        wgrad_store: WeightGradStore,
        fuse_wgrad_accumulation: bool,
        fc1_input_quantizer: Optional[Quantizer],
        fc1_weight_quantizer: Optional[Quantizer],
        fc1_output_quantizer: Optional[Quantizer],
        fc1_grad_input_quantizer: Optional[Quantizer],
        fc1_grad_weight_quantizer: Optional[Quantizer],
        fc1_grad_output_quantizer: Optional[Quantizer],
        fc2_input_quantizer: Optional[Quantizer],
        fc2_weight_quantizer: Optional[Quantizer],
        fc2_output_quantizer: Optional[Quantizer],
        fc2_grad_input_quantizer: Optional[Quantizer],
        fc2_grad_weight_quantizer: Optional[Quantizer],
        fc2_grad_output_quantizer: Optional[Quantizer],
        cpu_offloading: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        return_layernorm_output: bool,
        return_layernorm_output_gathered: bool,
        bias_gelu_fusion: bool,
        set_parallel_mode: bool,
        is_grad_enabled: bool,
        fwd_ln_sm_margin: int,
        bwd_ln_sm_margin: int,
        zero_centered_gamma: bool,
        activation: str,
        normalization: str,
        ub_overlap_ag: bool,
        ub_overlap_rs: bool,
        ub_overlap_rs_dgrad: bool,
        ub_bulk_wgrad: bool,
        ub_bulk_dgrad: bool,
        gemm_gelu_fusion: bool,
        fsdp_group: Union[dist_group_type, None],
        module: torch.nn.Module,
        skip_fp8_weight_update: bool,
        symmetric_ar_type: str,
        debug: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        # pylint: disable=missing-function-docstring

        # Make sure input dimensions are compatible
        in_features, inp_shape = ln_weight.numel(), inp.shape
        assert inp_shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view((-1, in_features))
        if fp8:
            assert_dim_for_fp8_exec(inputmat, fc1_weight, fc2_weight)

        activation_func = _act_func(
            activation, FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
        )[0]

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        ln_weight = cast_if_needed(ln_weight, activation_dtype)
        if ln_bias is not None:
            ln_bias = cast_if_needed(ln_bias, activation_dtype)

        tp_world_size = get_distributed_world_size(tp_group)
        backwards_needs_fc1_input = is_grad_enabled and fc1_weight.requires_grad
        device = inp.device

        # Configure Userbuffers communication (comm+GEMM overlap)
        if debug:  # turn off userbuffers in debug mode
            ub_overlap_ag = False
            ub_overlap_rs = False
            ub_overlap_rs_dgrad = False
            ub_bulk_wgrad = False
            ub_bulk_dgrad = False
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

        # for fp8 DelayedScaling: layernorm output = FP8
        #                   only output of the linear is returned
        # for return_layernorm_output: layernorm output = High precision, then cast to FP8
        #                              high precision layernorm output and output of the linear are returned
        # for debug: : layernorm output = High precision to enable processing of this norm

        with_quantized_norm = (
            fp8
            and not debug
            and not return_layernorm_output
            and not return_layernorm_output_gathered
        )

        # Apply normalization
        ln_out, mu, rsigma = apply_normalization(
            inputmat,
            None,  # ln_out
            ln_weight,
            ln_bias,
            eps,
            fc1_input_quantizer if with_quantized_norm else None,
            inputmat.dtype,
            normalization,
            fwd_ln_sm_margin,
            zero_centered_gamma,
        )
        ln_out_return = None
        if return_layernorm_output or return_layernorm_output_gathered:
            ln_out_return = ln_out

        # Prepare GEMM input
        # Note: Cast to expected dtype and perform tensor-parallel communication
        ln_out_total = None
        ub_obj_lnout = None
        if sequence_parallel:
            if return_layernorm_output_gathered:
                # Perform all-gather in high precision if gathered
                # norm output will be returned
                ln_out_total, _ = gather_along_first_dim(ln_out, tp_group)
                ln_out_return = ln_out_total
                if fp8 or debug:
                    ln_out = fc1_input_quantizer(ln_out)
                    fc1_input_quantizer.set_usage(rowwise=True, columnwise=False)
                    if isinstance(fc1_input_quantizer, Float8BlockQuantizer):
                        fc1_input_quantizer.all_gather_usage = False
                    ln_out_total = fc1_input_quantizer(ln_out_total)
            else:
                quantizer = None
                if fp8 or debug:
                    quantizer = fc1_input_quantizer
                    if not with_quantized_norm:
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
        fc1_weight_final = fc1_weight
        fc2_weight_final = fc2_weight
        if fp8 or debug:
            # If weights are not quantized, we call get_weight_workspace,
            # which handles weight caching etc.
            # FP8 cast to workspace buffer
            update_workspace = is_first_microbatch is None or is_first_microbatch
            fc1_weight_quantizer.set_usage(rowwise=True, columnwise=is_grad_enabled)
            fc2_weight_quantizer.set_usage(rowwise=True, columnwise=is_grad_enabled)
            fc1_weight_final = module.get_weight_workspace(
                tensor=fc1_weight,
                quantizer=fc1_weight_quantizer,
                cache_name=(None if is_first_microbatch is None else "fc1_weight"),
                update_workspace=update_workspace,
                skip_update_flag=skip_fp8_weight_update,
                fsdp_group=fsdp_group,
                workspace_dtype=activation_dtype,
            )
            fc2_weight_final = module.get_weight_workspace(
                tensor=fc2_weight,
                quantizer=fc2_weight_quantizer,
                cache_name=(None if is_first_microbatch is None else "fc2_weight"),
                update_workspace=update_workspace,
                skip_update_flag=skip_fp8_weight_update,
                fsdp_group=fsdp_group,
                workspace_dtype=activation_dtype,
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
        if fc1_bias is not None:
            fc1_bias = cast_if_needed(fc1_bias, bias_dtype)
        if fc2_bias is not None:
            fc2_bias = cast_if_needed(fc2_bias, bias_dtype)

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
            get_workspace(),
            quantization_params=(
                fc2_input_quantizer
                if gemm_gelu_fusion
                else fc1_output_quantizer  # fused gelu output is in fp8
            ),
            out_dtype=activation_dtype,
            bias=(
                fc1_bias if not bias_gelu_fusion else None
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
        if not is_grad_enabled and (ln_out_total is not ln_out_return):
            clear_tensor_data(ln_out_total)

        # ACTIVATION - sometimes activation is fused with the GEMM above.

        fc1_out_without_bias = None

        if bias_gelu_fusion:
            fc1_out = None
            fc1_out_without_bias, *_ = fc1_outputs
            act_out = bias_gelu_fused(fc1_out_without_bias, fc1_bias)
        elif gemm_gelu_fusion:
            act_out, _, fc1_out, _ = fc1_outputs
        elif debug:
            fc1_out, *_ = fc1_outputs
            act_out = activation_func(fc1_out, None)
            act_out = fc2_input_quantizer(act_out)
        else:
            fc1_out, *_ = fc1_outputs
            if fp8 and FP8GlobalStateManager.get_fp8_recipe().float8_block_scaling():
                # tex.quantize does not support GELU fusion for blockwise.
                act_out = activation_func(fc1_out, None)
                act_out = tex.quantize(act_out, fc2_input_quantizer)
            else:
                act_out = activation_func(fc1_out, fc2_input_quantizer)

        if not is_grad_enabled:
            clear_tensor_data(fc1_out)

        if fp8_calibration:
            fc2_input_quantizer.calibrate(act_out)
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
            get_workspace(),
            out_dtype=activation_dtype,
            bias=fc2_bias,
            quantization_params=fc2_output_quantizer,
            use_split_accumulator=use_split_accumulator,
            ub=ub_obj_fc2out,
            ub_type=tex.CommOverlapType.RS if ub_overlap_rs else None,
            extra_output=reduce_scatter_out,
        )
        # ------------------------------------------------------
        # Finished FC2 GEMM...
        # ------------------------------------------------------

        # Deallocate tensors if no longer needed
        if not is_grad_enabled:
            clear_tensor_data(act_out, fc1_out_without_bias, fc1_out)

        # Prepare output tensor
        # Note: Perform tensor-parallel communication if needed
        fc2_out = None
        if ub_overlap_rs:
            fc2_out = reduce_scatter_out
        elif set_parallel_mode and sequence_parallel:
            fc2_out, _ = reduce_scatter_along_first_dim(gemm_out, tp_group)
        elif set_parallel_mode and tensor_parallel:
            if symmetric_ar_type is not None:
                fc2_out, _ = symmetric_all_reduce(
                    gemm_out, tp_group, all_reduce_type=symmetric_ar_type
                )
            else:
                fc2_out, _ = allreduce(gemm_out, tp_group)
        else:
            fc2_out = gemm_out
        fc2_out = fc2_out.view(-1, *inp_shape[1:-1], fc2_out.shape[-1])

        # Cache state for backward pass
        if is_grad_enabled:

            # Weight with column-wise usage is needed for dgrad GEMM.
            if isinstance(fc1_weight_final, QuantizedTensorBase):
                fc1_weight_final.update_usage(columnwise_usage=True)
            if isinstance(fc2_weight_final, QuantizedTensorBase):
                fc2_weight_final.update_usage(columnwise_usage=True)

            if cpu_offloading:
                mark_activation_offload(
                    inputmat, mu, rsigma, ln_out, fc1_out, fc1_out_without_bias, act_out
                )

            # Scatter intermediate/activation tensors saved for the backward pass
            # NOTE: weight_fp8 = weight when ctx.fp8 == False and torch.disttributed.FSDP already
            #       shards/unshards the base weights so we don't do it ourselves
            ctx.fsdp_group = fsdp_group
            ctx.fsdp_shapes = _fsdp_scatter_tensors(
                fsdp_group,
                mu,
                rsigma,
                ln_out,
                fc1_out_without_bias if bias_gelu_fusion else fc1_out,
                act_out,
                fc1_weight_final if fp8 and not isinstance(fc1_weight, Float8Tensor) else None,
                fc2_weight_final if fp8 and not isinstance(fc2_weight, Float8Tensor) else None,
            )

            ctx.fc1_weight_quantizer = fc1_weight_quantizer
            ctx.fc2_weight_quantizer = fc2_weight_quantizer
            if not fc1_weight.requires_grad:
                if not return_layernorm_output:
                    clear_tensor_data(ln_out)
                ln_out = None
            if not fc2_weight.requires_grad:
                clear_tensor_data(act_out)
                act_out = None
            tensors_to_save, tensor_objects = prepare_for_saving(
                inputmat,
                ln_weight,
                ln_out,
                fc1_weight_final,
                fc1_weight,
                fc1_bias,
                fc1_out,
                fc1_out_without_bias,
                act_out,
                fc2_weight_final,
                fc2_weight,
                fc2_bias,
                mu,
                rsigma,
            )

            if fuse_wgrad_accumulation:
                # This check is needed to ensure that main_grad is not created
                # during the forward pass when using MCore FSDP as it creates
                # the main_grad buffer lazily before backprop
                if hasattr(fc1_weight, "__fsdp_param__") and hasattr(fc2_weight, "__fsdp_param__"):
                    # MCore FSDP creates main_grad lazily before backward
                    ctx.fc1_main_grad_func = (
                        fc1_weight.get_main_grad if fc1_weight.requires_grad else lambda: None
                    )
                    ctx.fc2_main_grad_func = (
                        fc2_weight.get_main_grad if fc2_weight.requires_grad else lambda: None
                    )
                else:
                    ctx.fc1_main_grad_func = lambda: fc1_weight.main_grad
                    ctx.fc2_main_grad_func = lambda: fc2_weight.main_grad

            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects

            ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
            ctx.fc1_grad_input_quantizer = fc1_grad_input_quantizer
            ctx.fc1_grad_weight_quantizer = fc1_grad_weight_quantizer
            ctx.fc1_grad_output_quantizer = fc1_grad_output_quantizer
            ctx.fc2_grad_input_quantizer = fc2_grad_input_quantizer
            ctx.fc2_grad_weight_quantizer = fc2_grad_weight_quantizer
            ctx.fc2_grad_output_quantizer = fc2_grad_output_quantizer
            ctx.fc1_input_quantizer = fc1_input_quantizer
            ctx.fc2_input_quantizer = fc2_input_quantizer

            ctx.fc1_weight_requires_grad = fc1_weight.requires_grad
            ctx.fc2_weight_requires_grad = fc2_weight.requires_grad
            ctx.fc1_weight = fc1_weight
            ctx.fc2_weight = fc2_weight

            ctx.device = device
            ctx.activation_dtype = activation_dtype
            ctx.activation = activation
            ctx.fp8 = fp8
            ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = fc2_bias is not None
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp_shape
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.bias_gelu_fusion = bias_gelu_fusion
            ctx.return_layernorm_output = return_layernorm_output
            ctx.return_layernorm_output_gathered = (
                return_layernorm_output_gathered and sequence_parallel
            )
            ctx.set_parallel_mode = set_parallel_mode
            ctx.bwd_ln_sm_margin = bwd_ln_sm_margin
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.ub_bulk_wgrad = ub_bulk_wgrad
            ctx.ub_bulk_dgrad = ub_bulk_dgrad
            ctx.ub_overlap_rs_dgrad = ub_overlap_rs_dgrad
            ctx.ub_overlap_ag = ub_overlap_ag
            ctx.debug = debug

            ctx.requires_dgrad = (
                inp.requires_grad or ln_weight.requires_grad or ln_bias.requires_grad
            )
            ctx.normalization = normalization
            ctx.reduce_and_update_bwd_fp8_tensors = False
            if ctx.fp8 and requires_grad(
                inp, ln_weight, ln_bias, fc1_weight, fc2_weight, fc1_bias, fc2_bias
            ):
                _first_fp8_module = FP8GlobalStateManager.IS_FIRST_FP8_MODULE
                ctx.reduce_and_update_bwd_fp8_tensors = FP8GlobalStateManager.is_first_fp8_module()
                if in_fp8_activation_recompute_phase():
                    FP8GlobalStateManager.IS_FIRST_FP8_MODULE = _first_fp8_module

            ctx.wgrad_store = wgrad_store

        if return_layernorm_output:
            if return_layernorm_output_gathered:
                shape = list(inp_shape)
                shape[0] *= tp_size if (sequence_parallel and set_parallel_mode) else 1
                return fc2_out, ln_out_return.view(shape)
            return fc2_out, ln_out_return.view(inp_shape)
        return fc2_out

    @staticmethod
    def backward(
        ctx, *grad_outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring
        with torch.cuda.nvtx.range("_LayerNormMLP_backward"):
            saved_tensors = ctx.saved_tensors
            (  # pylint: disable=unbalanced-tuple-unpacking
                inputmat,
                ln_weight,
                ln_out,
                fc1_weight,
                origin_fc1_weight,
                fc1_bias,
                fc1_out,
                fc1_out_without_bias,
                act_out,
                fc2_weight,
                origin_fc2_weight,
                fc2_bias,
                mu,
                rsigma,
            ) = restore_from_saved(ctx.tensor_objects, saved_tensors)
            # Delete the references to tensor objects once they've been consumed
            # by the `restore_from_saved` method to construct back the actual tensors.
            ctx.tensor_objects = None

            # Since main_grad can be modified inplace, it should not be a part of saved_tensors
            fc1_weight_main_grad = (
                ctx.fc1_main_grad_func()
                if fc1_weight is not None
                and ctx.fuse_wgrad_accumulation
                and ctx.fc1_weight_requires_grad
                else None
            )
            fc2_weight_main_grad = (
                ctx.fc2_main_grad_func()
                if origin_fc2_weight is not None
                and ctx.fuse_wgrad_accumulation
                and ctx.fc2_weight_requires_grad
                else None
            )

            # For CPU offloading, we offloaded weight and weight.main_grad to different tensors,
            # we need to connect them into one.
            if ctx.fuse_wgrad_accumulation:
                origin_fc1_weight.main_grad = fc1_weight_main_grad
                origin_fc2_weight.main_grad = fc2_weight_main_grad

            # TODO: Fix this  # pylint: disable=fixme
            # Gather saved autograd context tensors when running with FSDP
            # NOTE: weight_fp8 = weight when ctx.fp8 == False and torch.disttributed.FSDP already
            #       shards/unshards the base weights so we don't do it ourselves
            # _fsdp_gather_tensors(
            #    ctx.fsdp_group,
            #    ctx.fsdp_shapes,
            #    mu,
            #    rsigma,
            #    ln_out,
            #    fc1_out_without_bias if bias_gelu_nvfusion else fc1_out,,
            #    gelu_out,
            #    fc1_weight_fp8 if ctx.fp8 and not isinstance(fc1_weight, Float8Tensor) else None,
            #    fc2_weight_fp8 if ctx.fp8 and not isinstance(fc2_weight, Float8Tensor) else None,
            # )

            # Choose whether to use GEMM kernel with split accumulator
            dgrad_use_split_accumulator = _2X_ACC_DGRAD
            wgrad_use_split_accumulator = _2X_ACC_WGRAD
            if ctx.fp8:
                recipe = ctx.fp8_recipe
                if hasattr(recipe, "fp8_gemm_dgrad"):
                    dgrad_use_split_accumulator = recipe.fp8_gemm_dgrad.use_split_accumulator
                if hasattr(recipe, "fp8_gemm_wgrad"):
                    wgrad_use_split_accumulator = recipe.fp8_gemm_wgrad.use_split_accumulator

            # No need to do bulk DGRAD/WGRAD overlap if WGRAD is not required
            ctx.ub_bulk_dgrad = ctx.fc1_weight_requires_grad and ctx.ub_bulk_dgrad
            ctx.ub_bulk_wgrad = ctx.fc1_weight_requires_grad and ctx.ub_bulk_wgrad

            # Configure quantizer for FC2 grad output tensor
            # Note: dgrad GEMM requires row-wise usage, wgrad GEMM
            # requires column-wise usage
            if ctx.fc2_grad_output_quantizer is not None:
                quantizer = ctx.fc2_grad_output_quantizer
                quantizer.set_usage(rowwise=True, columnwise=True)
                if ctx.ub_overlap_ag:
                    # Userbuffers only supports communication for one
                    # tensor usage at a time. Configure quantizer with
                    # usage for only dgrad GEMM.
                    quantizer.set_usage(columnwise=False)

            # Prepare FC2 grad output tensor
            # Note: Cast to expected dtype and perform tensor-parallel communication
            ub_obj_fc2_dgrad = None
            if ctx.ub_overlap_ag:
                ub_obj_fc2_dgrad = get_ub("fc2_dgrad", ctx.fp8)
            ctx.ub_obj_gradout = ub_obj_fc2_dgrad
            (
                grad_output,
                fc2_bias_grad,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx, grad_outputs[0], True, ctx.fc2_grad_output_quantizer
            )

            # Launch tensor-parallel communication for FC1 GEMM input
            ln_out_total = None
            ln_out_total_work = None
            ub_obj_fc1_dgrad = None
            if ctx.fc1_weight_requires_grad and ctx.tensor_parallel and ctx.sequence_parallel:
                quantizer = None
                if ctx.fp8 or ctx.debug:
                    quantizer = ctx.fc1_input_quantizer
                    if isinstance(quantizer, (Float8Quantizer, Float8CurrentScalingQuantizer)):
                        # If data is in FP8, we compute FP8 transposes manually
                        quantizer.set_usage(rowwise=True, columnwise=False)
                    else:
                        # wgrad GEMM requires input with column-wise usage
                        quantizer.set_usage(rowwise=False, columnwise=True)
                if ctx.ub_bulk_dgrad:
                    ub_obj_fc1_dgrad = get_ub("fc1_dgrad", ctx.fp8)
                    ln_out_total, _ = fill_userbuffers_buffer_for_all_gather(
                        ub_obj_fc1_dgrad,
                        ln_out,
                        quantizer,
                        ctx.tp_group,
                    )
                else:
                    ln_out_total, ln_out_total_work = gather_along_first_dim(
                        ln_out,
                        ctx.tp_group,
                        async_op=True,
                        quantizer=quantizer,
                    )
            else:
                ln_out_total = ln_out

            # Check whether to output wgrad GEMM directly into main grad
            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

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
                not ctx.fp8
                and (ctx.activation == "gelu")
                and (not ctx.bias_gelu_fusion)
                and (not ctx.debug)
            )

            # Make sure required data is available
            if isinstance(grad_output, QuantizedTensorBase):
                grad_output.update_usage(rowwise_usage=True)
            if ctx.fc2_weight_quantizer is not None and isinstance(
                ctx.fc2_weight, QuantizedTensorBase
            ):
                ctx.fc2_weight.update_usage(columnwise_usage=True)

            # Perform GEMM
            gemm_output, *_ = general_gemm(
                fc2_weight,
                grad_output,
                get_workspace(),
                layout="NN",
                grad=True,
                quantization_params=(
                    ctx.fc1_grad_input_quantizer
                    if fc2_dgrad_gemm_gelu_fusion or ctx.debug
                    else None
                ),  # high precision to activation
                out_dtype=ctx.activation_dtype,
                gelu=fc2_dgrad_gemm_gelu_fusion,
                gelu_in=fc1_out if fc2_dgrad_gemm_gelu_fusion else None,
                use_split_accumulator=dgrad_use_split_accumulator,
                ub=ub_obj_fc2_dgrad,
                ub_type=tex.CommOverlapType.AG if ctx.ub_overlap_ag else None,
            )

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

            # --------------------------------------------------
            # FC2 WGRAD
            # --------------------------------------------------

            fc2_wgrad = None
            if ctx.fc2_weight_requires_grad:
                # Prepare grad output tensor
                # Note: Synchronize tensor-parallel communication and
                # make sure required data is available
                if ctx.ub_overlap_ag and isinstance(ctx.fc2_grad_output_quantizer, MXFP8Quantizer):
                    # UB does not support pipelined overlapping grad output
                    # all-gather with wgrad GEMM. Also, we can't
                    # convert row-scaled MXFP8 to column-scaled, so we
                    # can't reuse the grad output that was gathered
                    # for the dgrad GEMM. We work around by explicitly
                    # overlapping the AG operation with the dgrad GEMM.

                    # Get the communication stream from the dgrad GEMM to use for the AG
                    dgrad_send_stream, dgrad_recv_stream = (
                        ub_obj_fc2_dgrad.get_communication_stream()
                    )

                    ub_obj_fc2_wgrad = get_ub("fc2_wgrad", ctx.fp8)

                    ctx.fc2_grad_output_quantizer.set_usage(rowwise=False, columnwise=True)

                    # We use the send stream to copy into the userbuffers.
                    # This is the same stream that we will use to access the data in the AG,
                    # so we dont need to add any syncs yet.
                    with torch.cuda.stream(dgrad_send_stream):
                        grad_output, _ = fill_userbuffers_buffer_for_all_gather(
                            ub_obj_fc2_wgrad,
                            grad_outputs[0],
                            ctx.fc2_grad_output_quantizer,
                            ctx.tp_group,
                        )

                    # Allgather grad_outputs[0] using the dgrad streams so we can overlap with the fc2_dgrad gemm
                    tex.bulk_overlap_ag_with_external_gemm(
                        ub_obj_fc2_wgrad, dgrad_send_stream, dgrad_recv_stream
                    )

                # Prepare input tensor
                # Note: Synchronize tensor-parallel communication and
                # make sure required data is available
                if ctx.fp8 or ctx.debug:
                    if isinstance(act_out, QuantizedTensorBase):
                        act_out.update_usage(columnwise_usage=True)
                    else:
                        ctx.fc2_input_quantizer.set_usage(rowwise=False, columnwise=True)
                        act_out = ctx.fc2_input_quantizer(act_out)

                if ctx.fp8 or ctx.debug:
                    if isinstance(grad_output, QuantizedTensorBase):
                        grad_output.update_usage(columnwise_usage=True)
                    else:
                        ctx.fc2_grad_output_quantizer.set_usage(rowwise=False, columnwise=True)
                        grad_output = ctx.fc2_grad_output_quantizer(grad_output)

                # Whether to set grad arg in general_gemm
                grad_arg = True
                if ctx.fp8 and ctx.fp8_recipe.float8_block_scaling():
                    grad_arg = False

                # Arguments to include in wgrad GEMM closure
                fc2_wgrad_gemm_kwargs = {
                    "workspace": get_workspace(),
                    "out_dtype": (
                        origin_fc2_weight.main_grad.dtype
                        if ctx.fuse_wgrad_accumulation
                        else ctx.activation_dtype
                    ),
                    "quantization_params": ctx.fc2_grad_weight_quantizer,  # wgrad in high precision
                    "accumulate": accumulate_wgrad_into_param_main_grad,
                    "layout": "NT",
                    "out": origin_fc2_weight.main_grad if ctx.fuse_wgrad_accumulation else None,
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
                if ctx.wgrad_store is not None and ctx.wgrad_store.delay_wgrad_compute():
                    ctx.wgrad_store.put([act_out, grad_output], fc2_wgrad_gemm)
                else:

                    # Call wgrad GEMM now
                    fc2_wgrad, fc2_bias_grad_ = fc2_wgrad_gemm(act_out, grad_output)

                    # Update grad bias if needed
                    if fc2_bias_grad is None:
                        if (
                            ctx.fp8
                            and ctx.fp8_recipe.float8_block_scaling()
                            and fc2_bias is not None
                        ):
                            # BGRAD not fused with GEMM for float8 blockwise gemm.
                            fc2_bias_grad_ = act_out.view(-1, act_out.shape[-1]).sum(dim=0)
                        fc2_bias_grad = fc2_bias_grad_
                    del fc2_bias_grad_

            # Deallocate input tensor if permitted
            if ctx.wgrad_store is not None and not ctx.wgrad_store.delay_wgrad_compute():
                clear_tensor_data(act_out)

            # --------------------------------------------------
            # Finished FC2 WGRAD...
            # --------------------------------------------------

            # bias computation
            fc1_bias_grad = None
            fuse_gemm_and_bias_fc1_wgrad = False
            if ctx.fc1_grad_output_quantizer is not None:
                ctx.fc1_grad_output_quantizer.set_usage(rowwise=True, columnwise=True)
            if ctx.bias_gelu_fusion:
                # Fusion: gemm, bias + gelu
                assert ctx.activation == "gelu"
                assert not ctx.fp8
                fc1_bias_grad, dact = bgrad_dgelu_fused(fc2_dgrad, fc1_out_without_bias, fc1_bias)
                if ctx.fc1_grad_output_quantizer is not None:
                    dact = ctx.fc1_grad_output_quantizer(dact)
            elif ctx.debug:
                dact_func = _act_func(ctx.activation)[1]
                dact = dact_func(fc2_dgrad, fc1_out.to(ctx.activation_dtype), None)
                fc1_bias_grad = dact.sum(dim=0)
                dact = ctx.fc1_grad_output_quantizer(dact)
            elif (
                _act_func(ctx.activation, ctx.fp8_recipe if ctx.fp8 else None)[2] is not None
                and ctx.fp8
            ):
                # Fusion: gemm, bias + gelu + quantize
                dbias_dact_quantize_func = _act_func(
                    ctx.activation, ctx.fp8_recipe if ctx.fp8 else None
                )[2]
                fc1_bias_grad, dact = dbias_dact_quantize_func(
                    fc2_dgrad, fc1_out.to(ctx.activation_dtype), ctx.fc1_grad_output_quantizer
                )  # quantize bgrad gelu fused
            else:
                # Fusion: gemm + gelu,
                if not fc2_dgrad_gemm_gelu_fusion:
                    activation_func_bwd = _act_func(
                        ctx.activation, ctx.fp8_recipe if ctx.fp8 else None
                    )[1]
                    dact = activation_func_bwd(
                        fc2_dgrad, fc1_out.to(ctx.activation_dtype), None
                    )  # activation in high precision

                if ctx.fp8:
                    # TODO float8 blockwise current scaling has no bgrad fusion for now
                    if isinstance(ctx.fc1_grad_output_quantizer, Float8BlockQuantizer):
                        fc1_bias_grad = dact.view(-1, dact.shape[-1]).sum(dim=0)
                        dact = ctx.fc1_grad_output_quantizer(dact)
                    else:
                        fc1_bias_grad, dact = tex.bgrad_quantize(
                            dact, ctx.fc1_grad_output_quantizer
                        )
                else:
                    fuse_gemm_and_bias_fc1_wgrad = (
                        True  # fc1_bias_grad is computed later, fused with wgrad gemm for the FC1
                    )
                    # it may  not be calculated in case wgrad is not required.
                    if fc1_bias is not None:
                        if not ctx.fc1_weight_requires_grad and fc1_bias.requires_grad:
                            fc1_bias_grad = dact.sum(dim=0)

            # Overwrite data. Deleting the tensor does not release underlying memory.
            clear_tensor_data(fc1_out, fc1_out_without_bias)

            # Set UB algo and UB obj for fc1_dgrad/wgrad bulk/pipelined overlap
            ub_obj_fc1_dgrad = None
            ub_obj_fc1_wgrad = None
            ub_type_fc1_dgrad = None
            ub_type_fc1_wgrad = None
            fc1_dgrad_shape = [reduce(multiply_op, inputmat.shape[:-1]), inputmat.shape[-1]]
            if ctx.ub_overlap_rs_dgrad:
                # Overlap DGRAD+RS
                ub_obj_fc1_dgrad = get_ub("fc1_dgrad", ctx.fp8)
                ub_type_fc1_dgrad = tex.CommOverlapType.RS
            else:
                if ctx.ub_bulk_dgrad:
                    # Overlap ln_out all-gather with DGRAD compute
                    ub_obj_fc1_dgrad = get_ub("fc1_dgrad", ctx.fp8)
                    ub_type_fc1_dgrad = tex.CommOverlapType.AG
                if ctx.ub_bulk_wgrad:
                    # Overlap FC1 DGRAD reduce-scatter with WGRAD compute
                    ub_obj_fc1_wgrad = get_ub("fc1_wgrad", ctx.fp8)
                    ub_type_fc1_wgrad = tex.CommOverlapType.RS

            # --------------------------------------------------
            # FC1 DGRAD
            # --------------------------------------------------

            # Make sure required data is available
            if ctx.fc1_weight_quantizer is not None and isinstance(
                ctx.fc1_weight_quantizer, QuantizedTensorBase
            ):
                ctx.fc1_weight.update_usage(columnwise_usage=True)

            # Output buffers for Userbuffers reduce-scatter
            gemm_out = None
            reduce_scatter_out = None
            if ctx.ub_overlap_rs_dgrad:
                reduce_scatter_out = torch.empty(
                    fc1_dgrad_shape, dtype=ctx.activation_dtype, device="cuda"
                )
            if ctx.ub_bulk_wgrad:
                gemm_out = ub_obj_fc1_wgrad.get_buffer(local_chunk=False)

            # dgrad GEMM
            gemm_out, *_, reduce_scatter_out = general_gemm(
                fc1_weight,
                dact,
                get_workspace(),
                out=gemm_out,
                out_dtype=ctx.activation_dtype,
                quantization_params=ctx.fc1_grad_input_quantizer,
                layout="NN",
                grad=True,
                use_split_accumulator=dgrad_use_split_accumulator,
                ub=ub_obj_fc1_dgrad,
                ub_type=ub_type_fc1_dgrad,
                extra_output=reduce_scatter_out,
                bulk_overlap=ctx.ub_bulk_dgrad,
            )

            # Prepare grad input tensor
            # Note: Perform tensor-parallel communication
            fc1_dgrad = None
            fc1_dgrad_work = None
            if ctx.ub_overlap_rs_dgrad:
                fc1_dgrad = reduce_scatter_out
            elif ctx.ub_bulk_wgrad:
                fc1_dgrad = ub_obj_fc1_wgrad.get_buffer(local_chunk=True)
            elif ctx.set_parallel_mode and not ctx.ub_bulk_wgrad:
                fc1_dgrad = gemm_out
                if ctx.sequence_parallel:
                    if ctx.return_layernorm_output and ctx.return_layernorm_output_gathered:
                        fc1_dgrad = fc1_dgrad + grad_outputs[1].view_as(fc1_dgrad)
                    fc1_dgrad, fc1_dgrad_work = reduce_scatter_along_first_dim(
                        fc1_dgrad,
                        ctx.tp_group,
                        async_op=True,
                    )
                elif ctx.tensor_parallel:
                    fc1_dgrad, fc1_dgrad_work = allreduce(fc1_dgrad, ctx.tp_group, async_op=True)
            else:
                fc1_dgrad = gemm_out

            # --------------------------------------------------
            # Finished FC1 DGRAD...
            # --------------------------------------------------

            # --------------------------------------------------
            # FC1 WGRAD
            # --------------------------------------------------
            fc1_wgrad = None
            if ctx.fc1_weight_requires_grad:

                # Prepare input tensor
                # Note: Synchronize tensor-parallel communication and
                # make sure required data is available
                if ln_out_total_work is not None:
                    ln_out_total_work.wait()
                    ln_out_total_work = None
                if ctx.fp8 or ctx.debug:
                    if isinstance(ln_out_total, QuantizedTensorBase):
                        ln_out_total.update_usage(columnwise_usage=True)
                    else:
                        ctx.fc1_input_quantizer.set_usage(rowwise=False, columnwise=True)
                        ln_out_total = ctx.fc1_input_quantizer(ln_out_total)

                # Prepare grad output tensor
                # Note: Synchronize tensor-parallel communication and
                # make sure required data is available
                if ctx.fp8 or ctx.debug:
                    if isinstance(dact, QuantizedTensorBase):
                        dact.update_usage(columnwise_usage=True)
                    else:
                        ctx.fc1_grad_output_quantizer.set_usage(rowwise=False, columnwise=True)
                        dact = ctx.fc1_grad_output_quantizer(dact)

                # Output buffer for overlapping grad input
                # reduce-scatter with wgrad GEMM
                reduce_scatter_out = None
                if ctx.ub_bulk_wgrad and ub_obj_fc1_wgrad.is_fp8_ubuf():
                    reduce_scatter_out = torch.empty(
                        fc1_dgrad_shape, dtype=ctx.activation_dtype, device="cuda"
                    )

                # Arguments to include in wgrad GEMM closure
                fc1_wgrad_gemm_kwargs = {
                    "workspace": get_workspace(),
                    "out_dtype": (
                        origin_fc1_weight.main_grad.dtype
                        if ctx.fuse_wgrad_accumulation
                        else ctx.activation_dtype
                    ),
                    "quantization_params": ctx.fc1_grad_weight_quantizer,
                    "accumulate": accumulate_wgrad_into_param_main_grad,
                    "layout": "NT",
                    "out": origin_fc1_weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                    "bias": fc1_bias if fuse_gemm_and_bias_fc1_wgrad else None,
                    "use_split_accumulator": wgrad_use_split_accumulator,
                    "grad": fuse_gemm_and_bias_fc1_wgrad,
                    "ub": ub_obj_fc1_wgrad,
                    "ub_type": ub_type_fc1_wgrad,
                    "extra_output": reduce_scatter_out,
                    "bulk_overlap": ctx.ub_bulk_wgrad,
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
                if ctx.wgrad_store is not None and ctx.wgrad_store.delay_wgrad_compute():
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
                    ctx.wgrad_store.put([ln_out_total, dact], fc1_wgrad_gemm)
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
                    if not ctx.return_layernorm_output_gathered:
                        clear_tensor_data(ln_out_total)

                # Update grad input if overlapping reduce-scatter with wgrad GEMM
                if ctx.ub_bulk_wgrad:
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
            if ctx.return_layernorm_output and not ctx.return_layernorm_output_gathered:
                dgrad = dgrad + grad_outputs[1].view_as(dgrad)

            # Norm gradient
            dgamma = None
            dbeta = None
            if ctx.normalization == "LayerNorm":
                dgrad, dgamma, dbeta = tex.layernorm_bwd(
                    dgrad,
                    inputmat,
                    mu,
                    rsigma,
                    ln_weight,
                    ctx.bwd_ln_sm_margin,
                    ctx.zero_centered_gamma,
                )
            elif ctx.normalization == "RMSNorm":
                dgrad, dgamma = tex.rmsnorm_bwd(
                    dgrad,
                    inputmat,
                    rsigma,
                    ln_weight,
                    ctx.bwd_ln_sm_margin,
                    ctx.zero_centered_gamma,
                )
                dbeta = None
        clear_tensor_data(mu, rsigma)

        if ctx.fc1_weight_requires_grad:
            # Handle custom DDP from mcore.
            if ctx.fuse_wgrad_accumulation and hasattr(fc1_weight, "grad_added_to_main_grad"):
                origin_fc1_weight.grad_added_to_main_grad = True
                if getattr(origin_fc1_weight, "zero_out_wgrad", False):
                    fc1_wgrad = torch.zeros(
                        origin_fc1_weight.main_grad.shape,
                        dtype=origin_fc1_weight.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    fc1_wgrad = torch.empty(
                        origin_fc1_weight.main_grad.shape,
                        dtype=origin_fc1_weight.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
            elif ctx.fuse_wgrad_accumulation:
                fc1_wgrad = None
        else:
            fc1_wgrad = None

        if ctx.fc2_weight_requires_grad:
            # Handle custom DDP from mcore.
            if ctx.fuse_wgrad_accumulation and hasattr(
                origin_fc2_weight, "grad_added_to_main_grad"
            ):
                origin_fc2_weight.grad_added_to_main_grad = True
                if getattr(origin_fc2_weight, "zero_out_wgrad", False):
                    fc2_wgrad = torch.zeros(
                        origin_fc2_weight.main_grad.shape,
                        dtype=origin_fc2_weight.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    fc2_wgrad = torch.empty(
                        origin_fc2_weight.main_grad.shape,
                        dtype=origin_fc2_weight.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
            elif ctx.fuse_wgrad_accumulation:
                fc2_wgrad = None
        else:
            fc2_wgrad = None

        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        # FIX THIS
        # Scatter Fp8 tranposed-weight buffers
        # if ctx.fp8:
        #    _fsdp_scatter_tensors(
        #        ctx.fsdp_group,
        #        fc1_weight_fp8 if not isinstance(fc1_weight, Float8Tensor) else None,
        #        fc2_weight_fp8 if not isinstance(fc2_weight, Float8Tensor) else None,
        #    )
        return (
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            dgamma,
            dbeta,
            fc1_wgrad,
            fc1_bias_grad if fc1_bias is not None else None,
            fc2_wgrad,  # pylint: disable=possibly-used-before-assignment
            fc2_bias_grad,
            None,  # eps
            None,  # is_first_microbatch
            None,  # fp8
            None,  # fp8_calibration
            None,  # wgrad_store
            None,  # fuse_wgrad_accumulation
            None,  # fc1_input_quantizer,
            None,  # fc1_weight_quantizer,
            None,  # fc1_output_quantizer,
            None,  # fc1_grad_input_quantizer,
            None,  # fc1_grad_weight_quantizer,
            None,  # fc1_grad_output_quantizer,
            None,  # fc2_input_quantizer,
            None,  # fc2_weight_quantizer,
            None,  # fc2_output_quantizer,
            None,  # fc2_grad_input_quantizer,
            None,  # fc2_grad_weight_quantizer,
            None,  # fc2_grad_output_quantizer,
            None,  # cpu_offloading
            None,  # tp_group
            None,  # tp_size
            None,  # sequence_parallel
            None,  # tensor_parallel
            None,  # activation_dtype
            None,  # return_layernorm_output
            None,  # return_layernorm_output_gathered
            None,  # bias_gelu_fusion
            None,  # set_parallel_mode
            None,  # is_grad_enabled
            None,  # fwd_ln_sm_margin
            None,  # bwd_ln_sm_margin
            None,  # zero_centered_gamma
            None,  # activation
            None,  # normalization
            None,  # ub_overlap_ag
            None,  # ub_overlap_rs
            None,  # ub_overlap_rs_dgrad
            None,  # ub_bulk_dgrad
            None,  # ub_bulk_wgrad
            None,  # gemm_gelu_fusion
            None,  # fsdp_group
            None,  # module
            None,  # skip_fp8_weight_update
            None,  # symmetric_ar_type
            None,  # debug
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
    bias : bool, default = `True`
          if set to `False`, the FC1 and FC2 layers will not learn an additive bias.
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    activation : str, default = 'gelu'
          activation function used.
          Options: 'gelu', 'geglu', 'qgelu', 'qgeglu', 'relu', 'reglu', 'srelu', 'sreglu',
                   'silu', and 'swiglu'.
    init_method : Callable, default = `None`
                 used for initializing FC1 weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    output_layer_init_method : Callable, default = `None`
                              used for initializing FC2 weights in the following way:
                              `output_layer_init_method(weight)`. When set to `None`, defaults to
                              `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    return_layernorm_output : bool, default = `False`
                             if set to `True`, output of layernorm is returned from the forward
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module
                             is taken post layernorm.
    return_layernorm_output_gathered : bool, default = `False`
                             if set to `True`, output of layernorm is returned after the all
                             gather operation. Ignored if return_layernorm_output is False.
                             Example use case: with sequence parallel, input to residual connection
                             for transformer module (e.g. LoRA) will need to be gathered.
                             Returning layernorm output gathered will prevent a redundant gather.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will be allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.
    name: str, default = `None`
        name of the module, currently used for debugging purposes.

    Parallelism parameters
    ----------------------
    set_parallel_mode : bool, default = `False`
                      if set to `True`, FC1 is used as Column Parallel and FC2 is used as Row
                      Parallel as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             `set_tensor_parallel_group(tp_group)` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias for FC2, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    seq_length: int
               sequence length of input samples. Needed for JIT Warmup, a technique where jit fused
               functions are warmed up before training to ensure same kernels are used for forward
               propogation and activation recompute phase.
    micro_batch_size: int
                     batch size per training step. Needed for JIT Warmup, a technique where jit
                     fused functions are warmed up before training to ensure same kernels are
                     used for forward propogation and activation recompute phase.
    delay_wgrad_compute : bool, default = `False`
                         Whether or not to delay weight gradient computation. If set to `True`,
                         it's the user's responsibility to call `module.backward_dw` to compute
                         weight gradients.
    symmetric_ar_type : {None, 'multimem_all_reduce', 'two_shot', 'one_shot'}, default = None
                   Type of symmetric memory all-reduce to use during the forward pass.
                   This can help in latency bound communication situations.
                   Requires PyTorch version 2.7.0 or higher. When set to None, standard all-reduce
                   is used.
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
        name: str = None,
        ub_overlap_rs: bool = False,
        ub_overlap_rs_dgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_bulk_wgrad: bool = False,
        delay_wgrad_compute: bool = False,
        symmetric_ar_type: Optional[str] = None,
    ) -> None:
        super().__init__()

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.normalization = normalization
        assert normalization in ["LayerNorm", "RMSNorm"], "Unsupported normalization type!"
        self.use_bias = bias
        self.activation = activation
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
        self.name = name

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
        if self.activation in ["geglu", "qgeglu", "reglu", "sreglu", "swiglu"]:
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
            fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT,
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
            fp8_meta_index=tex.FP8FwdTensors.GEMM2_WEIGHT,
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

        # customize quantizers based on each recipe & layer configs
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        if recipe.float8_current_scaling():
            self._customize_quantizers_float8_current_scaling(fwd, recipe)
        elif recipe.float8_block_scaling():
            self._customize_quantizers_float8_blockwise_scaling(fwd, recipe)
        # elif for other recipes (mxfp8, etc.)

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
        if is_in_onnx_export_mode():
            return self.onnx_forward(inp)

        debug = self.is_debug_iter()

        if FP8GlobalStateManager.fp8_graph_capturing():
            skip_fp8_weight_update = FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
        else:
            skip_fp8_weight_update = None
        if skip_fp8_weight_update is not None:
            is_first_microbatch = False

        fp8_output = False
        if self.ub_overlap_rs:
            if get_ub("fc2_fprop", FP8GlobalStateManager.is_fp8_enabled()).is_fp8_ubuf():
                fp8_output = True

        with torch.cuda.device(
            getattr(self, list(self.named_parameters())[0][0]).device
        ), self.prepare_forward(inp, num_gemms=2) as inp:

            quantizers = (
                self._get_quantizers(fp8_output)
                if not debug
                else self._get_debug_quantizers(fp8_output)
            )
            if debug:
                if self.no_debug_features_active(quantizers):
                    debug = False
                    quantizers = self._get_quantizers(fp8_output)

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
            if not self.fp8:
                if isinstance(fc1_weight, Float8Tensor):
                    fc1_weight = fc1_weight.dequantize()
                if isinstance(fc2_weight, Float8Tensor):
                    fc2_weight = fc2_weight.dequantize()

            # Disable bias_gelu_nvfusion for determinism checkpointing in non-reentrant mode
            if self.bias_gelu_nvfusion and not use_reentrant_activation_recompute():
                self.bias_gelu_nvfusion = False

            if torch.is_grad_enabled():
                fwd_fn = _LayerNormMLP.apply
                args = []
            else:
                fwd_fn = _LayerNormMLP.forward
                args = [None]
            args += (
                inp,
                self.layer_norm_weight,
                self.layer_norm_bias,
                fc1_weight,
                fc1_bias,
                fc2_weight,
                fc2_bias if self.apply_bias and not self.gemm_bias_unfused_add else None,
                self.eps,
                is_first_microbatch,
                self.fp8,
                self.fp8_calibration,
                self.wgrad_store,
                self.fuse_wgrad_accumulation,
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
                is_cpu_offload_enabled(),
                self.tp_group,
                self.tp_size,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.return_layernorm_output,
                self.return_layernorm_output_gathered,
                self.bias_gelu_nvfusion and not self.fp8 and not debug,
                self.set_parallel_mode,
                torch.is_grad_enabled(),
                self.fwd_ln_sm_margin if torch.is_grad_enabled() else self.inf_ln_sm_margin,
                self.bwd_ln_sm_margin,
                self.zero_centered_gamma,
                self.activation,
                self.normalization,
                self.ub_overlap_ag,
                self.ub_overlap_rs,
                self.ub_overlap_rs_dgrad,
                self.ub_bulk_dgrad,
                self.ub_bulk_wgrad,
                self.gemm_gelu_fusion and not debug,
                self.fsdp_group,
                self,
                skip_fp8_weight_update,
                self.symmetric_ar_type,
                debug,
            )
            out = fwd_fn(*args)

        if self.return_layernorm_output:
            out, ln_out = out

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(fc2_bias, self.activation_dtype)

        if self.return_bias:
            if self.return_layernorm_output:
                return out, cast_if_needed(fc2_bias, self.activation_dtype), ln_out
            return out, cast_if_needed(fc2_bias, self.activation_dtype)
        if self.return_layernorm_output:
            return out, ln_out
        return out

    def _get_quantizers(self, fp8_output):
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
        if self.fp8:
            fc1_input_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_INPUT]
            fc1_input_quantizer.internal = True
            fc2_input_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM2_INPUT]
            fc2_input_quantizer.set_usage(
                rowwise=True,
                columnwise=isinstance(fc2_input_quantizer, (MXFP8Quantizer, Float8BlockQuantizer)),
            )
            fc1_input_quantizer.internal = True
            if fp8_output:
                fc2_output_quantizer = self.quantizers["scaling_fwd"][
                    tex.FP8FwdTensors.GEMM2_OUTPUT
                ]
            if torch.is_grad_enabled():
                fc2_grad_output_quantizer = self.quantizers["scaling_bwd"][
                    tex.FP8BwdTensors.GRAD_OUTPUT2
                ]
                fc2_grad_output_quantizer.internal = True
                fc1_grad_output_quantizer = self.quantizers["scaling_bwd"][
                    tex.FP8BwdTensors.GRAD_OUTPUT1
                ]
                fc1_grad_output_quantizer.internal = True

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

    def onnx_forward(self, inp: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        ONNX-compatible version of the forward function that provides numerical equivalence
        while only using operations that have defined ONNX symbolic translations.
        This simplified implementation is designed specifically for inference scenarios.
        """
        from ..export import onnx_layernorm, onnx_gemm

        assert not TEDebugState.debug_enabled, "Debug mode is not supported in ONNX export"
        assert_warmed_up(self)
        (
            fc1_input_quantizer,
            fc1_weight_quantizer,
            fc2_input_quantizer,
            fc2_weight_quantizer,
            output_quantizer,
            *_,
        ) = self._get_quantizers(False)
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

        activation_map = {
            "gelu": lambda x: torch.nn.functional.gelu(x, approximate="tanh"),
            "geglu": lambda x: torch.nn.functional.gelu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1],
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

        if output_quantizer is not None:
            raise NotImplementedError("ONNX export of quantized output is not supported")

        if self.return_layernorm_output:
            if self.return_bias:
                return fc2_out, fc2_bias.to(inp_dtype), ln_out_return
            return fc2_out, ln_out_return
        if self.return_bias:
            return fc2_out, fc2_bias.to(inp_dtype)
        return fc2_out

    def _get_debug_quantizers(self, fp8_output):
        from ...debug.pytorch.debug_quantization import DebugQuantizer

        base_quantizers = list(self._get_quantizers(fp8_output))
        assert TEDebugState.debug_enabled

        def make_debug(prefix, offset):
            labels = ["activation", "weight", "output", "dgrad", "wgrad", "gradient"]
            return [
                DebugQuantizer(
                    f"{self.name}.{prefix}",
                    label,
                    None if label in ("dgrad", "wgrad") else base_quantizers[i + offset],
                    self.tp_group,
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
                tex.FP8FwdTensors.GEMM1_INPUT
            ].force_pow_2_scales = recipe.fp8_quant_fwd_inp.power_2_scale
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM1_INPUT
            ].amax_epsilon = recipe.fp8_quant_fwd_inp.amax_epsilon
            # fc2_input_quantizer
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM2_INPUT
            ].force_pow_2_scales = recipe.fp8_quant_fwd_inp.power_2_scale
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM2_INPUT
            ].amax_epsilon = recipe.fp8_quant_fwd_inp.amax_epsilon
            # fc1_weight_quantizer: also set numerical configs about weight
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM1_WEIGHT
            ].force_pow_2_scales = recipe.fp8_quant_fwd_weight.power_2_scale
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM1_WEIGHT
            ].amax_epsilon = recipe.fp8_quant_fwd_weight.amax_epsilon
            # fc2_weight_quantizer
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM2_WEIGHT
            ].force_pow_2_scales = recipe.fp8_quant_fwd_weight.power_2_scale
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM2_WEIGHT
            ].amax_epsilon = recipe.fp8_quant_fwd_weight.amax_epsilon
            # parallel related
            if self.sequence_parallel and self.set_parallel_mode:
                # fc1_input_quantizer: customize input_quantizer with amax reduction TP group, column parallel + sequence parallel here
                self.quantizers["scaling_fwd"][
                    tex.FP8FwdTensors.GEMM1_INPUT
                ].with_amax_reduction = True
                self.quantizers["scaling_fwd"][
                    tex.FP8FwdTensors.GEMM1_INPUT
                ].amax_reduction_group = self.tp_group
        else:
            # fc2_grad_output_quantizer: set configs about amax epsilon and power_2_scale for fc2_grad_output_quantizer
            self.quantizers["scaling_bwd"][
                tex.FP8BwdTensors.GRAD_OUTPUT2
            ].force_pow_2_scales = recipe.fp8_quant_bwd_grad.power_2_scale
            self.quantizers["scaling_bwd"][
                tex.FP8BwdTensors.GRAD_OUTPUT2
            ].amax_epsilon = recipe.fp8_quant_bwd_grad.amax_epsilon
            # fc1_grad_output_quantizer: also set numerical configs for fc1_grad_output_quantizer
            self.quantizers["scaling_bwd"][
                tex.FP8BwdTensors.GRAD_OUTPUT1
            ].force_pow_2_scales = recipe.fp8_quant_bwd_grad.power_2_scale
            self.quantizers["scaling_bwd"][
                tex.FP8BwdTensors.GRAD_OUTPUT1
            ].amax_epsilon = recipe.fp8_quant_bwd_grad.amax_epsilon
            if self.sequence_parallel and self.set_parallel_mode:
                # fc2_grad_output_quantizer: customize grad_output_quantizer with amax reduction TP group, row parallel + sequence parallel here
                self.quantizers["scaling_bwd"][
                    tex.FP8BwdTensors.GRAD_OUTPUT2
                ].with_amax_reduction = True
                self.quantizers["scaling_bwd"][
                    tex.FP8BwdTensors.GRAD_OUTPUT2
                ].amax_reduction_group = self.tp_group

    def _get_weight_tensors(self) -> List[Union[torch.Tensor, QuantizedTensorBase]]:
        """Get the weight tensors of the module."""
        return [self.fc1_weight, self.fc2_weight]

    def _get_weight_quantizers(self) -> List[Quantizer]:
        """Get the weight quantizers of the module."""
        if not self.fp8:
            return [None, None]
        fc1_weight_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_WEIGHT]
        fc1_weight_quantizer.internal = True
        fc2_weight_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM2_WEIGHT]
        fc2_weight_quantizer.internal = True
        return [fc1_weight_quantizer, fc2_weight_quantizer]

    def _customize_quantizers_float8_blockwise_scaling(self, fwd: bool, recipe: Recipe) -> None:
        """Customize quantizers based on blockwise scaling recipe + layernorm_mlp."""
        assert (
            recipe.float8_block_scaling()
        ), "blockwise scaling recipe quantizer customization here"
        if fwd:
            if self.sequence_parallel and self.set_parallel_mode:
                self.quantizers["scaling_fwd"][
                    tex.FP8FwdTensors.GEMM1_INPUT
                ].all_gather_usage = True
        else:
            if self.sequence_parallel and self.set_parallel_mode:
                self.quantizers["scaling_bwd"][
                    tex.FP8BwdTensors.GRAD_OUTPUT2
                ].all_gather_usage = True

    def backward_dw(self):
        """
        Execute the delayed weight gradient computation.
        This method is called after the main backward pass to compute weight gradients.
        """
        if self.wgrad_store is None or not self.wgrad_store.delay_wgrad_compute():
            return
        with torch.cuda.nvtx.range("_LayerNormMLP_wgrad"):
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
            for wgrad_accumulation_and_reduce_hook in self.wgrad_accumulation_and_reduce_hooks:
                wgrad_accumulation_and_reduce_hook()

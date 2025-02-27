# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LayerNormMLP API"""
import os
import warnings
from typing import Callable, Optional, Tuple, Union
from functools import reduce
from operator import mul as multiply_op

import torch
from torch.nn.parameter import Parameter
from torch.nn import init

import transformer_engine_torch as tex

from .base import (
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
    non_tn_fp8_gemm_supported,
)
from ..distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    allreduce,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
    use_reentrant_activation_recompute,
    in_fp8_activation_recompute_phase,
    _fsdp_scatter_tensors,
)

from ..constants import dist_group_type
from ..jit import no_torch_dynamo
from ..graph import is_graph_capturing
from ..tensor.float8_tensor import Float8Tensor
from ..tensor.mxfp8_tensor import MXFP8Quantizer
from ._common import apply_normalization, _fix_gathered_fp8_transpose
from ..cpu_offload import is_cpu_offload_enabled, set_offloading_param

from ..tensor.quantized_tensor import (
    QuantizedTensor,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)
from ..cpp_extensions import (
    general_gemm,
)

__all__ = ["LayerNormMLP"]


def _act_func(activation: str):
    funcs = {
        "gelu": (tex.gelu, tex.dgelu, tex.dbias_dgelu),
        "relu": (tex.relu, tex.drelu, tex.dbias_drelu),
        "geglu": (tex.geglu, tex.dgeglu, None),
        "reglu": (tex.reglu, tex.dreglu, None),
        "swiglu": (tex.swiglu, tex.dswiglu, None),
        "qgelu": (tex.qgelu, tex.dqgelu, tex.dbias_dqgelu),
        "qgeglu": (tex.qgeglu, tex.dqgeglu, None),
        "srelu": (tex.srelu, tex.dsrelu, tex.dbias_dsrelu),
    }
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
        use_fc1_bias: bool,
        fc2_weight: torch.Tensor,
        fc2_bias: torch.Tensor,
        use_fc2_bias: bool,
        eps: float,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        fuse_wgrad_accumulation: bool,
        fc1_input_quantizer: Optional[Quantizer],
        fc1_weight_quantizer: Optional[Quantizer],
        fc2_input_quantizer: Optional[Quantizer],
        fc2_weight_quantizer: Optional[Quantizer],
        output_quantizer: Optional[Quantizer],
        grad_fc2_output_quantizer: Optional[Quantizer],
        grad_fc1_output_quantizer: Optional[Quantizer],
        grad_input_quantizer: Optional[Quantizer],
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
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        # pylint: disable=missing-function-docstring

        in_features, inp_shape = ln_weight.numel(), inp.shape
        # Make sure input dimensions are compatible
        assert inp_shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view((-1, in_features))
        if fp8:
            assert_dim_for_fp8_exec(inputmat, fc1_weight, fc2_weight)
            if (
                any([ub_overlap_ag, ub_overlap_rs])
                and not FP8GlobalStateManager.get_fp8_recipe().delayed()
            ):
                raise NotImplementedError(
                    "Comm+GEMM overlap is only supported with FP8 delayed scaling"
                )

        activation_func = _act_func(activation)[0]
        device = inp.device

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        ln_weight = cast_if_needed(ln_weight, activation_dtype)
        if ln_bias is not None:
            ln_bias = cast_if_needed(ln_bias, activation_dtype)

        # for standard fp8: layernorm output = FP8
        #                   only output of the linear is returned
        # for return_layernorm_output: layernorm output = High precision, then cast to FP8
        #                              high precision layernorm output and output of the linear are returned
        with_quantized_norm = fp8 and not return_layernorm_output

        tp_world_size = get_distributed_world_size(tp_group)
        ub_overlap_ag = ub_overlap_ag and is_grad_enabled and not return_layernorm_output
        ub_overlap_rs = ub_overlap_rs and is_grad_enabled
        with_input_all_gather_nccl = sequence_parallel and not ub_overlap_ag
        backwards_needs_fc1_input = is_grad_enabled and fc1_weight.requires_grad

        # Configure quantizer for normalization output
        if fp8 and fc1_input_quantizer is None:
            raise ValueError("Missing quantizer for input tensor")
        if with_quantized_norm:
            if with_input_all_gather_nccl:
                fc1_input_quantizer.set_usage(rowwise=True, columnwise=False)
                if isinstance(fc1_input_quantizer, MXFP8Quantizer):
                    with_quantized_norm = False
            else:
                fc1_input_quantizer.set_usage(
                    rowwise=True,
                    columnwise=backwards_needs_fc1_input,
                )

        ub_obj_lnout = None
        ln_out = None
        if ub_overlap_ag:
            ub_obj_lnout = get_ub("fc1_fprop")
            ln_out = ub_obj_lnout.get_buffer(fc1_input_quantizer, local_chunk=True)
        elif not with_quantized_norm:
            ln_out = torch.empty_like(
                inputmat, dtype=inputmat.dtype, memory_format=torch.contiguous_format, device="cuda"
            )

        # Apply normalization
        ln_out, mu, rsigma = apply_normalization(
            inputmat,
            ln_out,
            ln_weight,
            ln_bias,
            eps,
            fc1_input_quantizer if with_quantized_norm else None,
            inp.dtype,
            normalization,
            fwd_ln_sm_margin,
            zero_centered_gamma,
        )

        # Prepare GEMM input
        # Note: Cast to expected dtype and perform tensor-parallel communication
        ln_out_gathered = False
        with_quantized_all_gather = fp8
        if with_input_all_gather_nccl:
            if return_layernorm_output and return_layernorm_output_gathered:
                with_quantized_all_gather = False
            if fp8:
                fc1_input_quantizer.set_usage(rowwise=True, columnwise=False)
            ln_out_total, _ = gather_along_first_dim(
                ln_out,
                tp_group,
                quantizer=(fc1_input_quantizer if with_quantized_all_gather else None),
            )
            ln_out_gathered = True
        else:
            with_quantized_all_gather = False
            if ub_overlap_ag:
                ln_out_total = ub_obj_lnout.get_buffer(fc1_input_quantizer, False)
            else:
                ln_out_total = ln_out

        # If residual connection is after LN, we need `ln_out`
        # tensor in higher precision, this comes at the cost
        # of an extra fp8 cast.
        ln_out_return = None
        if return_layernorm_output:
            ln_out_return = ln_out_total if return_layernorm_output_gathered else ln_out
            if fp8 and not with_quantized_all_gather:
                ln_out_total = fc1_input_quantizer(ln_out_total)
                if ln_out_gathered:
                    rank = torch.distributed.get_rank(tp_group)
                    slice_start = rank * ln_out.size(0)
                    slice_end = (rank + 1) * ln_out.size(0)
                    ln_out = ln_out_total[
                        slice_start:slice_end, ...
                    ]  # TODO(pgadzinski) - check this  # pylint: disable=fixme
                else:
                    ln_out = ln_out_total

        # Cast weights to expected dtype
        fc1_weight_final = fc1_weight
        fc2_weight_final = fc2_weight
        if not fp8:
            fc1_weight_final = cast_if_needed(fc1_weight_final, activation_dtype)
            fc2_weight_final = cast_if_needed(fc2_weight_final, activation_dtype)
        else:
            # If weights are not quantized, we call get_weight_workspace,
            # which handles weight caching etc.
            if not isinstance(fc1_weight, QuantizedTensor):
                # FP8 cast to workspace buffer
                update_workspace = is_first_microbatch is None or is_first_microbatch
                fc1_weight_final = module.get_weight_workspace(
                    tensor=fc1_weight,
                    quantizer=fc1_weight_quantizer,
                    cache_name=(None if is_first_microbatch is None else "fc1_weight"),
                    update_workspace=update_workspace,
                    skip_update_flag=skip_fp8_weight_update,
                    fsdp_group=fsdp_group,
                )
            if not isinstance(fc2_weight, QuantizedTensor):
                fc2_weight_quantizer.set_usage(rowwise=True, columnwise=True)
                fc2_weight_final = module.get_weight_workspace(
                    tensor=fc2_weight,
                    quantizer=fc2_weight_quantizer,
                    cache_name=(None if is_first_microbatch is None else "fc2_weight"),
                    update_workspace=update_workspace,
                    skip_update_flag=skip_fp8_weight_update,
                    fsdp_group=fsdp_group,
                )

        # Cast biases to expected dtype
        bias_dtype = activation_dtype
        if fp8 and activation_dtype == torch.float32:
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

        # FC1 GEMM

        # There are 2 fussions possible:
        # - gemm_gelu_fusion - default for full precision, optional for fp8 - need to turn on gemm_gelu_fusion,
        # - bias_gelu_fusion - only for full precision.
        # If both gemm_gelu_fusion and bias_gelu_fusion are enabled, only bias_gelu_fusion will be performer
        if activation != "gelu":
            gemm_gelu_fusion = bias_gelu_fusion = False
        else:
            if fp8:
                assert not bias_gelu_fusion, "Bias gelu fusion is supported only for full precision"
            else:
                gemm_gelu_fusion = True
            if gemm_gelu_fusion and bias_gelu_fusion:
                gemm_gelu_fusion = False

        fc1_outputs = general_gemm(
            fc1_weight_final,
            ln_out_total,
            get_workspace(),
            quantization_params=(
                fc2_input_quantizer if gemm_gelu_fusion else None  # fused gelu output is in fp8
            ),
            out_dtype=activation_dtype,
            bias=(
                fc1_bias if not bias_gelu_fusion else None
            ),  # otherwise bias is added later (fused with gelu)
            gelu=gemm_gelu_fusion,
            accumulate=_2X_ACC_FPROP,
            ub=ub_obj_lnout,
            ub_type=tex.CommOverlapType.AG if ub_overlap_ag else None,
        )
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
        else:
            fc1_out, *_ = fc1_outputs
            act_out = activation_func(fc1_out, fc2_input_quantizer)

        if not is_grad_enabled:
            clear_tensor_data(fc1_out)

        if fp8_calibration:
            fc2_input_quantizer.calibrate(act_out)
            fc2_weight_quantizer.calibrate(fc2_weight)

        ub_obj_fc2out = None
        rs_out = None
        fc2_out = None
        if ub_overlap_rs:
            ub_obj_fc2out = get_ub("fc2_fprop")
            dim_size = list(act_out.size())
            dim_size[0] = dim_size[0] // tp_world_size
            dim_size[1] = fc2_weight.size(0)
            rs_out = torch.empty(dim_size, dtype=activation_dtype, device=device)
            fc2_out = ub_obj_fc2out.get_buffer(output_quantizer)
        else:
            dim_size = list(act_out.size())
            dim_size[1] = fc2_weight.size(0)
            fc2_out = torch.empty(dim_size, dtype=activation_dtype, device=device)

        # FC2 GEMM
        _ = general_gemm(
            fc2_weight_final,
            act_out,
            get_workspace(),
            out_dtype=activation_dtype,
            bias=fc2_bias,
            quantization_params=output_quantizer,
            out=fc2_out,
            use_split_accumulator=_2X_ACC_FPROP,
            ub=ub_obj_fc2out,
            ub_type=tex.CommOverlapType.RS if ub_overlap_rs else None,
            extra_output=rs_out,
        )
        if not is_grad_enabled:
            clear_tensor_data(act_out, fc1_out_without_bias, fc1_out)

        if is_grad_enabled:
            if cpu_offloading:
                if fp8 and fc1_weight_final is not None:
                    set_offloading_param(fc1_weight_final, "weight_offloading", True)
                if fp8 and fc2_weight_final is not None:
                    set_offloading_param(fc2_weight_final, "weight_offloading", True)
                set_offloading_param(ln_weight, "weight_offloading", True)
                set_offloading_param(fc1_weight, "weight_offloading", True)
                set_offloading_param(fc2_weight, "weight_offloading", True)
                set_offloading_param(fc1_bias, "weight_offloading", True)

                set_offloading_param(inputmat, "activation_offloading", True)
                set_offloading_param(mu, "activation_offloading", True)
                set_offloading_param(rsigma, "activation_offloading", True)
                set_offloading_param(mu, "activation_offloading", True)
                set_offloading_param(ln_out, "activation_offloading", True)
                set_offloading_param(fc1_out, "activation_offloading", True)
                set_offloading_param(fc1_out_without_bias, "activation_offloading", True)
                set_offloading_param(act_out, "activation_offloading", True)

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
                ln_out.clone() if ub_overlap_ag else ln_out,  # avoid saving a UB buffer
                fc1_weight_final,
                fc1_bias,
                fc1_out,
                fc1_out_without_bias,
                act_out,
                fc2_weight_final,
                fc2_bias,
                mu,
                rsigma,
            )

            if fuse_wgrad_accumulation:
                ctx.fc1_main_grad = fc1_weight.main_grad if fc1_weight.requires_grad else None
                ctx.fc2_main_grad = fc2_weight.main_grad if fc2_weight.requires_grad else None

            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects

            ctx.grad_fc1_output_quantizer = grad_fc1_output_quantizer
            ctx.grad_fc2_output_quantizer = grad_fc2_output_quantizer
            ctx.grad_input_quantizer = grad_input_quantizer
            ctx.fc2_input_quantizer = fc2_input_quantizer
            ctx.fc1_input_quantizer = fc1_input_quantizer

            ctx.fc1_weight_requires_grad = fc1_weight.requires_grad
            ctx.fc2_weight_requires_grad = fc2_weight.requires_grad
            ctx.fc1_weight = fc1_weight
            ctx.fc2_weight = fc2_weight

            ctx.device = device
            ctx.activation_dtype = activation_dtype
            ctx.activation = activation
            ctx.fp8 = fp8
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_fc1_bias = use_fc1_bias
            ctx.use_fc2_bias = use_fc2_bias
            ctx.use_bias = ctx.use_fc1_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp_shape
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.bias_gelu_fusion = bias_gelu_fusion
            ctx.return_layernorm_output = return_layernorm_output
            ctx.return_layernorm_output_gathered = (
                return_layernorm_output_gathered and ln_out_gathered
            )
            ctx.set_parallel_mode = set_parallel_mode
            ctx.bwd_ln_sm_margin = bwd_ln_sm_margin
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.ub_bulk_wgrad = ub_bulk_wgrad
            ctx.ub_bulk_dgrad = ub_bulk_dgrad
            ctx.ub_overlap_rs_dgrad = ub_overlap_rs_dgrad
            ctx.ub_overlap_ag = ub_overlap_ag

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

        # Row Parallel Linear
        if ub_overlap_rs:
            fc2_out = rs_out
        elif set_parallel_mode and sequence_parallel:
            fc2_out, _ = reduce_scatter_along_first_dim(fc2_out, tp_group)
        elif set_parallel_mode and tensor_parallel:
            fc2_out, _ = allreduce(fc2_out, tp_group)

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        fc2_out = fc2_out.view(-1, *inp_shape[1:-1], fc2_out.shape[-1])

        if return_layernorm_output:
            if return_layernorm_output_gathered:
                shape = list(inp_shape)
                shape[0] *= tp_size
                return fc2_out, ln_out_return.view(shape)
            return fc2_out, ln_out_return.view_as(inp)
        return fc2_out

    @staticmethod
    def backward(
        ctx, *grad_outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring
        with torch.cuda.nvtx.range("_LayerNormMLP_backward"):
            if (
                ctx.fp8
                and any(
                    [
                        ctx.ub_overlap_ag,
                        ctx.ub_overlap_rs_dgrad,
                        ctx.ub_bulk_dgrad,
                        ctx.ub_bulk_wgrad,
                    ]
                )
                and not FP8GlobalStateManager.get_fp8_recipe().delayed()
            ):
                raise NotImplementedError(
                    "Comm+GEMM overlap is only supported with FP8 delayed scaling"
                )

            saved_tensors = ctx.saved_tensors
            (  # pylint: disable=unbalanced-tuple-unpacking
                inputmat,
                ln_weight,
                ln_out,
                fc1_weight,
                fc1_bias,
                fc1_out,
                fc1_out_without_bias,
                act_out,
                fc2_weight,
                fc2_bias,
                mu,
                rsigma,
            ) = restore_from_saved(ctx.tensor_objects, saved_tensors)
            # Delete the references to tensor objects once they've been consumed
            # by the `restore_from_saved` method to construct back the actual tensors.
            ctx.tensor_objects = None

            # Since main_grad can be modified inplace, it should not be a part of saved_tensors
            fc1_weight_main_grad = (
                ctx.fc1_main_grad
                if fc1_weight is not None
                and ctx.fuse_wgrad_accumulation
                and ctx.fc1_weight_requires_grad
                else None
            )
            fc2_weight_main_grad = (
                ctx.fc2_main_grad
                if fc2_weight is not None
                and ctx.fuse_wgrad_accumulation
                and ctx.fc2_weight_requires_grad
                else None
            )

            # For CPU offloading, we offloaded weight and weight.main_grad to different tensors,
            # we need to connect them into one.
            if ctx.fuse_wgrad_accumulation:
                fc1_weight.main_grad = fc1_weight_main_grad
                fc2_weight.main_grad = fc2_weight_main_grad

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

            # No need to do bulk DGRAD/WGRAD overlap if WGRAD is not required
            ctx.ub_bulk_dgrad = ctx.fc1_weight_requires_grad and ctx.ub_bulk_dgrad
            ctx.ub_bulk_wgrad = ctx.fc1_weight_requires_grad and ctx.ub_bulk_wgrad

            # Prepare grad output tensor
            # Note: Cast to expected dtype and perform tensor-parallel communication
            if ctx.grad_fc2_output_quantizer is not None:
                ctx.grad_fc2_output_quantizer.set_usage(
                    rowwise=True,
                    columnwise=True,
                )

            ub_obj_fc2_dgrad = None
            if ctx.ub_overlap_ag:
                ub_obj_fc2_dgrad = get_ub("fc2_dgrad")
            ctx.ub_obj_gradout = ub_obj_fc2_dgrad
            (
                grad_output,
                fc2_bias_grad,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx, grad_outputs[0], True, ctx.grad_fc2_output_quantizer
            )

            # Prepare FC1 GEMM input
            # Note: Perform tensor-parallel communication if needed
            ln_out_total = None
            ln_out_total_work = None
            if (
                ctx.fc1_weight_requires_grad
                and ctx.tensor_parallel
                and ctx.sequence_parallel
                and not ctx.ub_bulk_dgrad
            ):
                quantizer = None
                if ctx.fp8:
                    quantizer = ctx.fc1_input_quantizer
                    quantizer.set_usage(rowwise=True, columnwise=True)
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
            # There are 5 possible fusion paths
            # 1 high-precision bias_gelu_fusion: gemm, FC1_bias + gelu,
            # 2 high-precision fc2_dgrad_gemm_gelu_fusion: gemm + gelu, FC1_bias + quantize
            # 3 fp8 activation+bias+quantize fusion: gemm, activation + FC1_bias + quantize
            # 4 fp8 bias+quantize fusion: gemm, activation, FC1_bias + quantize
            # 5 high-precision unfused: gemm, activation, FC1_bias + FC1_gemm
            fc2_dgrad_gemm_gelu_fusion = (
                not ctx.fp8 and (ctx.activation == "gelu") and (not ctx.bias_gelu_fusion)
            )

            # FC2 DGRAD; Unconditional
            gemm_output, *_ = general_gemm(
                fc2_weight,
                grad_output,
                get_workspace(),
                layout="NN",
                grad=True,
                quantization_params=(
                    ctx.grad_fc1_output_quantizer if fc2_dgrad_gemm_gelu_fusion else None
                ),  # high precision to activation
                out_dtype=ctx.activation_dtype,
                gelu=fc2_dgrad_gemm_gelu_fusion,
                gelu_in=fc1_out if fc2_dgrad_gemm_gelu_fusion else None,
                use_split_accumulator=_2X_ACC_DGRAD,
                ub=ub_obj_fc2_dgrad,
                ub_type=tex.CommOverlapType.AG if ctx.ub_overlap_ag else None,
            )
            if fc2_dgrad_gemm_gelu_fusion:
                dact = gemm_output
                fc2_dgrad = None
            else:
                fc2_dgrad = gemm_output

            # FC2 WGRAD
            if ctx.fc2_weight_requires_grad:
                if isinstance(act_out, QuantizedTensor):
                    act_out.update_usage(rowwise_usage=True, columnwise_usage=True)

                if isinstance(grad_output, QuantizedTensor):
                    grad_output.update_usage(rowwise_usage=True, columnwise_usage=True)

                fc2_wgrad, fc2_bias_grad_, *_ = general_gemm(
                    act_out,
                    grad_output,
                    get_workspace(),
                    out_dtype=ctx.activation_dtype,
                    quantization_params=None,  # wgrad in high precision
                    layout="NT",
                    grad=True,
                    bias=fc2_bias if fc2_bias is not None and fc2_bias_grad is None else None,
                    accumulate=accumulate_wgrad_into_param_main_grad,
                    use_split_accumulator=_2X_ACC_WGRAD,
                    out=fc2_weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                )
                if fc2_bias_grad is None:
                    fc2_bias_grad = fc2_bias_grad_
                del fc2_bias_grad_
            clear_tensor_data(act_out)

            # bias computation
            fc1_bias_grad = None
            fuse_gemm_and_bias_fc1_wgrad = False
            if ctx.grad_fc1_output_quantizer is not None:
                ctx.grad_fc1_output_quantizer.set_usage(rowwise=True, columnwise=True)
            if ctx.bias_gelu_fusion:
                # Fusion: gemm, bias + gelu
                assert ctx.activation == "gelu"
                assert not ctx.fp8
                fc1_bias_grad, dact = bgrad_dgelu_fused(fc2_dgrad, fc1_out_without_bias, fc1_bias)
                if ctx.grad_fc1_output_quantizer is not None:
                    dact = ctx.grad_fc1_output_quantizer(dact)
            elif _act_func(ctx.activation)[2] is not None and ctx.fp8:
                # Fusion: gemm, bias + gelu + quantize
                dbias_dact_quantize_func = _act_func(ctx.activation)[2]
                fc1_bias_grad, dact = dbias_dact_quantize_func(
                    fc2_dgrad, fc1_out.to(ctx.activation_dtype), ctx.grad_fc1_output_quantizer
                )  # quantize bgrad gelu fused
            else:
                # Fusion: gemm + gelu,
                if not fc2_dgrad_gemm_gelu_fusion:
                    activation_func_bwd = _act_func(ctx.activation)[1]
                    dact = activation_func_bwd(
                        fc2_dgrad, fc1_out.to(ctx.activation_dtype), None
                    )  # activation in high precision

                if ctx.fp8:
                    fc1_bias_grad, dact = tex.bgrad_quantize(dact, ctx.grad_fc1_output_quantizer)
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
            fc1_dgrad_shape = [reduce(multiply_op, inputmat.shape[:-1]), inputmat.shape[-1]]
            fc1_dgrad_rs_out = None
            fc1_dgrad_bulk = None
            if ctx.ub_overlap_rs_dgrad:
                # Overlap DGRAD+RS
                ub_obj_fc1_dgrad = get_ub("fc1_dgrad")
                ub_type_fc1_dgrad = tex.CommOverlapType.RS
                fc1_dgrad_rs_out = torch.empty(
                    fc1_dgrad_shape, dtype=ctx.activation_dtype, device="cuda"
                )

            else:
                if ctx.ub_bulk_dgrad:
                    # Overlap ln_out all-gather with DGRAD compute
                    # NOTE: Copying into communication buffer will always prefer rowwise data,
                    #       and will copy columnwise data if rowwise does not exist. In that case,
                    #       the all-gather will apply to the leading dimension of the transpose,
                    #       which then needs to be interleaved correctly before WGRAD.
                    ub_obj_fc1_dgrad = get_ub("fc1_dgrad")
                    ub_type_fc1_dgrad = tex.CommOverlapType.AG
                    ub_obj_fc1_dgrad.copy_into_buffer(
                        ln_out, ctx.fc1_input_quantizer, local_chunk=True
                    )

                if ctx.ub_bulk_wgrad:
                    # Overlap FC1 DGRAD reduce-scatter with WGRAD compute
                    ub_obj_fc1_wgrad = get_ub("fc1_wgrad")
                    fc1_dgrad_bulk = ub_obj_fc1_wgrad.get_buffer(None)

            # FC1 DGRAD: Unconditional
            fc1_dgrad, *_, fc1_dgrad_rs_out = general_gemm(
                fc1_weight,
                dact,
                get_workspace(),
                out=fc1_dgrad_bulk,
                out_dtype=ctx.activation_dtype,
                layout="NN",
                grad=True,
                ub=ub_obj_fc1_dgrad,
                ub_type=ub_type_fc1_dgrad,
                extra_output=fc1_dgrad_rs_out,
                bulk_overlap=ctx.ub_bulk_dgrad,
            )

            # Overlap dgrad-RS/AR with wgrad
            fc1_dgrad_work = None
            if ctx.ub_overlap_rs_dgrad:
                fc1_dgrad = fc1_dgrad_rs_out
            elif ctx.set_parallel_mode and not ctx.ub_bulk_wgrad:
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

            # FC1 WGRAD
            fc1_wgrad = None
            if ctx.fc1_weight_requires_grad:
                if ctx.ub_bulk_dgrad:
                    ln_out_total = ub_obj_fc1_dgrad.get_buffer(ctx.fc1_input_quantizer)
                    if ctx.fp8:
                        if ln_out._data is None:
                            # All-gather executed on columnwise data and result is in rowwise data,
                            # so we need to fix the interleaving before WGRAD.
                            ln_out_total = _fix_gathered_fp8_transpose(ln_out_total, ctx.tp_size)
                        elif not non_tn_fp8_gemm_supported():
                            # FP8 GEMM on Hopper only supports TN layout so the gathered input must
                            # have a valid transpose.
                            ln_out_total._create_transpose()

                else:
                    if ln_out_total_work is not None:
                        # Synchronize tensor-parallel communication
                        ln_out_total_work.wait()
                        ln_out_total_work = None

                # Make sure GEMM inputs have expected data
                if isinstance(ln_out_total, QuantizedTensor):
                    ln_out_total.update_usage(rowwise_usage=True, columnwise_usage=True)
                if isinstance(dact, QuantizedTensor):
                    dact.update_usage(rowwise_usage=True, columnwise_usage=True)

                if ctx.ub_bulk_wgrad and ub_obj_fc1_wgrad.is_fp8_ubuf():
                    fc1_dgrad_rs_out = torch.empty(
                        fc1_dgrad_shape, dtype=ctx.activation_dtype, device="cuda"
                    )

                fc1_wgrad_outputs = general_gemm(
                    ln_out_total,
                    dact,
                    get_workspace(),
                    out_dtype=ctx.activation_dtype,
                    layout="NT",
                    grad=fuse_gemm_and_bias_fc1_wgrad,
                    bias=fc1_bias if fuse_gemm_and_bias_fc1_wgrad else None,
                    accumulate=accumulate_wgrad_into_param_main_grad,
                    out=fc1_weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                    ub=ub_obj_fc1_wgrad,
                    ub_type=tex.CommOverlapType.RS if ctx.ub_bulk_wgrad else None,
                    extra_output=fc1_dgrad_rs_out,
                    bulk_overlap=ctx.ub_bulk_wgrad,
                )

                clear_tensor_data(ln_out_total, dact)

                if fuse_gemm_and_bias_fc1_wgrad:
                    fc1_wgrad, fc1_bias_grad, *_ = fc1_wgrad_outputs
                else:
                    fc1_wgrad, *_ = fc1_wgrad_outputs

                if ctx.ub_bulk_wgrad:
                    if ub_obj_fc1_wgrad.is_fp8_ubuf():
                        fc1_dgrad = fc1_dgrad_rs_out
                    else:
                        fc1_dgrad = ub_obj_fc1_wgrad.get_buffer(None, local_chunk=True)

            # Synchronize tensor parallel communication
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
                fc1_weight.grad_added_to_main_grad = True
                if getattr(fc1_weight, "zero_out_wgrad", False):
                    fc1_wgrad = torch.zeros(
                        fc1_weight.main_grad.shape,
                        dtype=fc1_weight.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    fc1_wgrad = None
            elif ctx.fuse_wgrad_accumulation:
                fc1_wgrad = None
        else:
            fc1_wgrad = None

        if ctx.fc2_weight_requires_grad:
            # Handle custom DDP from mcore.
            if ctx.fuse_wgrad_accumulation and hasattr(fc2_weight, "grad_added_to_main_grad"):
                fc2_weight.grad_added_to_main_grad = True
                if getattr(fc2_weight, "zero_out_wgrad", False):
                    fc2_wgrad = torch.zeros(
                        fc2_weight.main_grad.shape,
                        dtype=fc2_weight.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    fc2_wgrad = None
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
            fc1_bias_grad if ctx.use_fc1_bias else None,
            None,  # use_fc1_bias
            fc2_wgrad,  # pylint: disable=possibly-used-before-assignment
            fc2_bias_grad if ctx.use_fc2_bias else None,
            None,  # use_fc2_bias
            None,  # eps
            None,  # is_first_microbatch
            None,  # fp8
            None,  # fp8_calibration
            None,  # fuse_wgrad_accumulation
            None,  # fc1_input_quantizer
            None,  # fc1_weight_quantizer
            None,  # fc2_input_quantizer
            None,  # fc2_weight_quantizer
            None,  # output_quantizer
            None,  # grad_fc2_output_quantizer
            None,  # grad_fc1_output_quantizer
            None,  # grad_input_quantizer
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
        )


class LayerNormMLP(TransformerEngineBaseModule):
    r"""
    Applies layer normalization on the input followed by the MLP module, consisting of
    2 successive linear transformations, separated by the GeLU activation.

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
          Options: 'gelu', 'geglu', 'relu', 'reglu', 'squared_relu', 'swiglu', 'qgelu', 'srelu'.
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
        ub_overlap_rs: bool = False,
        ub_overlap_rs_dgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_bulk_wgrad: bool = False,
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

        # GEMM-GELU fusion is currently only supported with split GEMM-AG overlap
        self.gemm_gelu_fusion = (
            bool(int(os.getenv("NVTE_GEMM_GELU_FUSION", "0")))
            and self.activation == "gelu"
            and ((_ub_communicators is None) or (not get_ub("fc1_fprop").is_atomic_gemm()))
        )

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
        if self.activation in ["reglu", "geglu", "qgeglu", "swiglu"]:
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

        # These many SMs are subtracted from the total SM count when calling forward
        # and backward LayerNorm C APIs. These envvars can be used to prevent the LN
        # kernels from using all SMs in the device. This is useful for cases such as
        # communication overlap with LN.
        self.fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
        self.bwd_ln_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))
        self.inf_ln_sm_margin = int(os.getenv("NVTE_INF_LAYERNORM_SM_MARGIN", "0"))

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
        self, inp: torch.Tensor, is_first_microbatch: Optional[bool] = None
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

        if FP8GlobalStateManager.fp8_graph_capturing():
            skip_fp8_weight_update = FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
        else:
            skip_fp8_weight_update = None
        if skip_fp8_weight_update is not None:
            is_first_microbatch = False

        with self.prepare_forward(inp, num_gemms=2) as inp:
            # Get quantizers
            (
                fc1_input_quantizer,
                fc1_weight_quantizer,
                fc2_input_quantizer,
                fc2_weight_quantizer,
                output_quantizer,
                grad_fc1_output_quantizer,
                grad_fc2_output_quantizer,
                grad_input_quantizer,
            ) = self._get_quantizers()

            # Get weight tensors
            fc1_weight = self.fc1_weight
            fc1_bias = self.fc1_bias if self.use_bias else None
            fc2_weight = self.fc2_weight
            fc2_bias = self.fc2_bias if self.use_bias else None
            if not self.fp8:
                if isinstance(fc1_weight, Float8Tensor):
                    fc1_weight = fc1_weight.from_float8()
                if isinstance(fc2_weight, Float8Tensor):
                    fc2_weight = fc2_weight.from_float8()

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
                self.use_bias,
                fc2_weight,
                fc2_bias,
                self.apply_bias and not self.gemm_bias_unfused_add,
                self.eps,
                is_first_microbatch,
                self.fp8,
                self.fp8_calibration,
                self.fuse_wgrad_accumulation,
                fc1_input_quantizer,
                fc1_weight_quantizer,
                fc2_input_quantizer,
                fc2_weight_quantizer,
                output_quantizer,
                grad_input_quantizer,
                grad_fc1_output_quantizer,
                grad_fc2_output_quantizer,
                is_cpu_offload_enabled(),
                self.tp_group,
                self.tp_size,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.return_layernorm_output,
                self.return_layernorm_output_gathered,
                self.bias_gelu_nvfusion and not self.fp8,
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
                self.gemm_gelu_fusion,
                self.fsdp_group,
                self,
                skip_fp8_weight_update,
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

    def _get_quantizers(self):
        (
            fc1_input_quantizer,
            fc1_weight_quantizer,
            fc2_input_quantizer,
            fc2_weight_quantizer,
            output_quantizer,
            grad_fc1_output_quantizer,
            grad_fc2_output_quantizer,
            grad_input_quantizer,
        ) = [None] * 8
        if self.fp8:
            fc1_input_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_INPUT]
            fc1_input_quantizer.internal = False  # temporary
            fc1_weight_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_WEIGHT]
            fc1_weight_quantizer.internal = True
            fc2_input_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM2_INPUT]
            fc2_input_quantizer.set_usage(
                rowwise=True, columnwise=isinstance(fc2_input_quantizer, MXFP8Quantizer)
            )
            fc2_weight_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM2_WEIGHT]
            fc2_weight_quantizer.internal = True
            if torch.is_grad_enabled():
                grad_fc2_output_quantizer = self.quantizers["scaling_bwd"][
                    tex.FP8BwdTensors.GRAD_OUTPUT1
                ]
                grad_fc2_output_quantizer.internal = True
                grad_fc1_output_quantizer = self.quantizers["scaling_bwd"][
                    tex.FP8BwdTensors.GRAD_INPUT1
                ]
                grad_fc1_output_quantizer.internal = True
                grad_input_quantizer = self.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_INPUT2]
                grad_input_quantizer.internal = True

        return (
            fc1_input_quantizer,
            fc1_weight_quantizer,
            fc2_input_quantizer,
            fc2_weight_quantizer,
            output_quantizer,
            grad_fc1_output_quantizer,
            grad_fc2_output_quantizer,
            grad_input_quantizer,
        )

# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LayerNormMLP API"""
import os
from typing import Union, Optional, Callable, Tuple, List, Dict, Any

import torch
from torch.nn.parameter import Parameter
from torch.nn import init

from .base import (
    get_workspace,
    _prepare_backward,
    get_ub,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ..fp8 import get_fp8_te_dtype
from ..jit import (
    bias_gelu_fused,
    bgrad_dgelu_fused,
    set_jit_fusion_options,
    warmup_jit_bias_gelu_all_dtypes,
)
from ..utils import (
    divide,
    get_default_init_method,
    cast_if_needed,
    assert_dim_for_fp8_exec,
)
from ..distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    allreduce,
    initialize_affine_weight_gpu,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
)

from .. import cpp_extensions as tex

from ..constants import dist_group_type, TE_DType
from ..jit import no_torch_dynamo

from ._common import _apply_normalization


__all__ = ["LayerNormMLP"]


def _act_func(activation: str):
    funcs = {
            'gelu': (tex.gelu, tex.dgelu),
            'relu': (tex.relu, tex.drelu),
            'geglu': (tex.geglu, tex.dgeglu),
            'reglu': (tex.reglu, tex.dreglu),
            'swiglu': (tex.swiglu, tex.dswiglu),
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
        fc1_weight_fp8: Union[torch.Tensor, None],
        fc1_weight_t_fp8: Union[torch.Tensor, None],
        fc1_bias: torch.Tensor,
        use_fc1_bias: bool,
        fc2_weight: torch.Tensor,
        fc2_weight_fp8: Union[torch.Tensor, None],
        fc2_weight_t_fp8: Union[torch.Tensor, None],
        fc2_bias: torch.Tensor,
        use_fc2_bias: bool,
        eps: float,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        fp8_meta: Dict[str, Any],
        fuse_wgrad_accumulation: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        return_layernorm_output: bool,
        bias_gelu_nvfusion: bool,
        set_parallel_mode: bool,
        is_grad_enabled: bool,
        fwd_ln_sm_margin: int,
        bwd_ln_sm_margin: int,
        zero_centered_gamma: bool,
        ub_bulk_wgrad: bool,
        ub_bulk_dgrad: bool,
        ub_split_rs: bool,
        ub_split_ag: bool,
        activation: str,
        normalization: str,
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        # Make sure input dimensions are compatible
        in_features = ln_weight.numel()
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view((-1, in_features))
        if fp8:
            assert_dim_for_fp8_exec(inputmat)
            assert_dim_for_fp8_exec(fc1_weight)
            assert_dim_for_fp8_exec(fc2_weight)

        update_fp8_weights = is_first_microbatch is None or is_first_microbatch

        activation_func = _act_func(activation)[0]

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        ln_weight = cast_if_needed(ln_weight, activation_dtype)
        if ln_bias is not None:
            ln_bias = cast_if_needed(ln_bias, activation_dtype)

        if ub_split_ag:
            tp_world_size = get_distributed_world_size(tp_group)
            if tp_world_size == 1 or (not is_grad_enabled) or return_layernorm_output:
                ub_split_ag = False
        if ub_split_ag:
            ub_obj_lnout = get_ub("fc1_fprop")
            ln_out = ub_obj_lnout.get_ubuf_output(0)
        else:
            ln_out_dtype = torch.uint8 if (fp8 and not return_layernorm_output) else inputmat.dtype
            ln_out = torch.empty_like(inputmat, dtype=ln_out_dtype)
        if ub_split_rs:
            tp_world_size = get_distributed_world_size(tp_group)
            if tp_world_size == 1:
                ub_split_rs = False

        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

        ln_out, mu, rsigma = _apply_normalization(inputmat,
                                                  ln_out,
                                                  ln_weight,
                                                  ln_bias,
                                                  eps,
                                                  fp8 and not return_layernorm_output,
                                                  fp8_meta,
                                                  normalization,
                                                  fwd_ln_sm_margin,
                                                  zero_centered_gamma,
                                                  is_grad_enabled)
        # If residual connection is after LN, we need `ln_out`
        # tensor in higher precision, this comes at the cost
        # of an extra fp8 cast.
        if return_layernorm_output:
            ln_out_return = ln_out
            if fp8:
                ln_out = tex.cast_to_fp8(
                    ln_out,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_forward,
                )
        # Column Parallel Linear
        if ub_split_ag:
            ln_out_total = ub_obj_lnout.get_ubuf_output(1)
            ln_out = torch.empty_like(ln_out)
        elif set_parallel_mode and sequence_parallel:
            ln_out_total, _ = gather_along_first_dim(ln_out, tp_group)
        else:
            ln_out_total = ln_out

        if fp8:
            bias_dtype = (
                torch.bfloat16
                if activation_dtype == torch.float32
                else activation_dtype
            )
            fc1_bias = cast_if_needed(fc1_bias, bias_dtype) if use_fc1_bias else fc1_bias
            fc2_bias = cast_if_needed(fc2_bias, bias_dtype) if use_fc2_bias else fc2_bias

            if update_fp8_weights:
                if is_grad_enabled:
                    tex.fp8_cast_transpose_fused(
                        fc1_weight,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                        cast_out=fc1_weight_fp8,
                        transpose_out=fc1_weight_t_fp8,
                    )

                    tex.fp8_cast_transpose_fused(
                        fc2_weight,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM2_WEIGHT,
                        fp8_dtype_forward,
                        cast_out=fc2_weight_fp8,
                        transpose_out=fc2_weight_t_fp8,
                    )
                else:
                    fc1_weight_t_fp8 = None
                    fc1_weight_fp8 = tex.cast_to_fp8(
                        fc1_weight,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                    )
                    fc2_weight_t_fp8 = None
                    fc2_weight_fp8 = tex.cast_to_fp8(
                        fc2_weight,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM2_WEIGHT,
                        fp8_dtype_forward,
                    )

            fc1_out = tex.fp8_gemm(
                fc1_weight_fp8,
                fp8_meta["scaling_fwd"].scale_inv,
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                fp8_dtype_forward,
                ln_out_total,
                fp8_meta["scaling_fwd"].scale_inv,
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                activation_dtype,
                get_workspace(),
                bias=fc1_bias,
                use_bias=use_fc1_bias,
                use_split_accumulator=_2X_ACC_FPROP,
                ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG if ub_split_ag else None,
                ub=ub_obj_lnout if ub_split_ag else None,
                extra_output_tensor=ln_out if ub_split_ag else None,
            )

            gelu_out = activation_func(
                fc1_out,
                fp8_meta["scaling_fwd"],
                tex.FP8FwdTensors.GEMM2_INPUT,
                fp8_dtype_forward,
            )

            if ub_split_rs:
                ub_obj_fc2out = get_ub("fc2_fprop")
                fc2_out = ub_obj_fc2out.get_ubuf_output(1)
                dim_size = list(gelu_out.size())
                dim_size[0] = dim_size[0] // tp_world_size
                dim_size[1] = fc2_weight.size(0)
                rs_out = torch.empty(dim_size, dtype=activation_dtype, device=gelu_out.device)
            else:
                dim_size = list(gelu_out.size())
                dim_size[1] = fc2_weight.size(0)
                fc2_out = torch.empty(dim_size, dtype=activation_dtype, device=gelu_out.device)

            _ = tex.fp8_gemm(
                fc2_weight_fp8,
                fp8_meta["scaling_fwd"].scale_inv,
                tex.FP8FwdTensors.GEMM2_WEIGHT,
                fp8_dtype_forward,
                gelu_out,
                fp8_meta["scaling_fwd"].scale_inv,
                tex.FP8FwdTensors.GEMM2_INPUT,
                fp8_dtype_forward,
                activation_dtype,
                get_workspace(),
                bias=fc2_bias,
                use_bias=use_fc2_bias,
                use_split_accumulator=_2X_ACC_FPROP,
                out=fc2_out,
                ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS if ub_split_rs else None,
                ub=ub_obj_fc2out if ub_split_rs else None,
                extra_output_tensor=rs_out if ub_split_rs else None,
            )
        else:
            # Cast for native AMP
            fc1_weight = cast_if_needed(fc1_weight, activation_dtype)
            fc2_weight = cast_if_needed(fc2_weight, activation_dtype)
            fc1_bias = (
                cast_if_needed(fc1_bias, activation_dtype) if use_fc1_bias else fc1_bias
            )
            fc2_bias = (
                cast_if_needed(fc2_bias, activation_dtype) if use_fc2_bias else fc2_bias
            )

            if fp8_calibration:
                # amax of fc1 input
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM1_INPUT] = \
                    torch.amax(ln_out_total).float()
                # amax of fc1 weight
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM1_WEIGHT] = \
                    torch.amax(fc1_weight).float()

            fc1_outputs = tex.gemm(
                fc1_weight,
                ln_out_total,
                activation_dtype,
                get_workspace(),
                bias=fc1_bias,
                use_bias=(not bias_gelu_nvfusion) and use_fc1_bias,
                gelu=not bias_gelu_nvfusion and (activation == 'gelu'),
                ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG if ub_split_ag else None,
                ub=ub_obj_lnout if ub_split_ag else None,
                extra_output_tensor=ln_out if ub_split_ag else None,
            )

            if bias_gelu_nvfusion:
                fc1_out, _, _ = fc1_outputs
                gelu_out = bias_gelu_fused(fc1_out, fc1_bias)
            else:
                if activation == 'gelu':
                    gelu_out, _, fc1_out = fc1_outputs
                else:
                    fc1_out, _, _ = fc1_outputs
                    gelu_out = activation_func(fc1_out,
                                               None,
                                               tex.FP8FwdTensors.GEMM2_INPUT,
                                               TE_DType[fc1_out.dtype])

            if fp8_calibration:
                # amax of fc2 input
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM2_INPUT] = \
                    torch.amax(gelu_out).float()
                # amax of fc2 weight
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM2_WEIGHT] = \
                    torch.amax(fc2_weight).float()

            if ub_split_rs:
                ub_obj_fc2out = get_ub("fc2_fprop")
                fc2_out = ub_obj_fc2out.get_ubuf_output(1)
                dim_size = list(gelu_out.size())
                dim_size[0] = dim_size[0] // tp_world_size
                dim_size[1] = fc2_weight.size(0)
                rs_out = torch.empty(dim_size, dtype=activation_dtype, device=gelu_out.device)
            else:
                dim_size = list(gelu_out.size())
                dim_size[1] = fc2_weight.size(0)
                fc2_out = torch.empty(dim_size, dtype=activation_dtype, device=gelu_out.device)
            _, _, _ = tex.gemm(
                fc2_weight,
                gelu_out,
                activation_dtype,
                get_workspace(),
                bias=fc2_bias,
                use_bias=use_fc2_bias,
                out=fc2_out,
                ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS if ub_split_rs else None,
                ub=ub_obj_fc2out if ub_split_rs else None,
                extra_output_tensor=rs_out if ub_split_rs else None,
            )

        if is_grad_enabled:
            ctx.save_for_backward(
                inputmat,
                ln_weight,
                mu,
                rsigma,
                ln_out,
                fc1_out,
                gelu_out,
                fc1_weight,
                fc1_weight_t_fp8,
                fc2_weight,
                fc2_weight_t_fp8,
                fc1_bias,
                fp8_meta["scaling_fwd"].scale_inv.clone() if fp8 else None,
            )
            ctx.activation_dtype = activation_dtype
            ctx.activation = activation
            ctx.fp8 = fp8
            ctx.fp8_meta = fp8_meta
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_fc1_bias = use_fc1_bias
            ctx.use_fc2_bias = use_fc2_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp.shape
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.bias_gelu_nvfusion = bias_gelu_nvfusion
            ctx.return_layernorm_output = return_layernorm_output
            ctx.set_parallel_mode = set_parallel_mode
            ctx.bwd_ln_sm_margin = bwd_ln_sm_margin
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.ub_bulk_wgrad = ub_bulk_wgrad
            ctx.ub_bulk_dgrad = ub_bulk_dgrad
            ctx.ub_split_ag = ub_split_ag
            ctx.requires_dgrad = inp.requires_grad
            ctx.normalization = normalization

        # Row Parallel Linear
        if ub_split_rs:
            fc2_out = rs_out
        elif set_parallel_mode and sequence_parallel:
            fc2_out, _ = reduce_scatter_along_first_dim(fc2_out, tp_group)
        elif set_parallel_mode and tensor_parallel:
            fc2_out, _ = allreduce(fc2_out, tp_group)

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        fc2_out = fc2_out.view(-1, *inp.shape[1:-1], fc2_out.shape[-1])

        if return_layernorm_output:
            return fc2_out, ln_out_return.view_as(inp)
        return fc2_out


    @staticmethod
    def backward(
        ctx, *grad_outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        with _prepare_backward(
            ctx.fp8, ctx.fp8_meta, ctx.tp_group, ctx.tp_size, name="_LayerNormMLP"
        ):
            (
                inputmat,
                ln_weight,
                mu,
                rsigma,
                ln_out,
                fc1_out,
                gelu_out,
                fc1_weight,
                fc1_weight_t_fp8,
                fc2_weight,
                fc2_weight_t_fp8,
                fc1_bias,
                fwd_scale_inverses,
            ) = ctx.saved_tensors

            activation_func = _act_func(ctx.activation)[1]

            if ctx.ub_bulk_dgrad:
                tp_world_size = get_distributed_world_size(ctx.tp_group)
                if tp_world_size == 1:
                    ctx.ub_bulk_dgrad = False
            if ctx.ub_bulk_dgrad:
                dim_size = list(ln_out.size())
                dim_size[0] = dim_size[0] * tp_world_size
                ub_obj_lnout = get_ub("fc1_dgrad")
                ub_obj_lnout.copy_input_to_ubuf(ln_out, 1)
            if ctx.ub_split_ag:
                tp_world_size = get_distributed_world_size(ctx.tp_group)
                if tp_world_size == 1:
                    ctx.ub_split_ag = False
            if ctx.ub_split_ag:
                dim_size = list(grad_outputs[0].size())
                dim_size[0] = dim_size[0] * tp_world_size
                ctx.ub_obj_gradout = get_ub("fc2_dgrad")

            ctx.use_bias = ctx.use_fc2_bias # For grad_output_preprocess
            (
                grad_output,
                grad_output_c,
                grad_output_t,
                fc2_bias_grad,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx, grad_outputs[0], True
            )
            if ctx.ub_bulk_wgrad:
                tp_world_size = get_distributed_world_size(ctx.tp_group)
                if tp_world_size == 1:
                    ctx.ub_bulk_wgrad = False
            # Column Parallel Linear
            # Overlap input AG with dgrad
            if (fc1_weight.requires_grad
                and (not ctx.ub_bulk_dgrad)
                and ctx.set_parallel_mode
                and ctx.sequence_parallel):
                ln_out_total, handle = gather_along_first_dim(
                    ln_out, ctx.tp_group, async_op=True
                )
            else:
                ln_out_total = ln_out
                handle = None

            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

            if ctx.fp8:
                fp8_dtype_forward = get_fp8_te_dtype(
                    ctx.fp8_meta["recipe"], fprop_tensor=True
                )
                fp8_dtype_backward = get_fp8_te_dtype(
                    ctx.fp8_meta["recipe"], fprop_tensor=False
                )

                # FC2 DGRAD; Unconditional
                fc2_dgrad = tex.fp8_gemm(
                    fc2_weight_t_fp8,
                    fwd_scale_inverses,
                    tex.FP8FwdTensors.GEMM2_WEIGHT,
                    fp8_dtype_forward,
                    grad_output_c,
                    ctx.fp8_meta["scaling_bwd"].scale_inv,
                    tex.FP8BwdTensors.GRAD_OUTPUT1,
                    fp8_dtype_backward,
                    ctx.activation_dtype,
                    get_workspace(),
                    use_split_accumulator=_2X_ACC_DGRAD,
                    ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG if ctx.ub_split_ag else None,
                    ub=ctx.ub_obj_gradout if ctx.ub_split_ag else None,
                )
                if ctx.ub_split_ag:
                    grad_output_t = tex.fp8_transpose(grad_output_c, fp8_dtype_backward)
                # FC2 WGRAD
                if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                    if fc2_weight.requires_grad:
                        gelu_out_t = tex.fp8_transpose(gelu_out, fp8_dtype_forward)
                        fc2_wgrad = tex.fp8_gemm(
                            gelu_out_t,
                            fwd_scale_inverses,
                            tex.FP8FwdTensors.GEMM2_INPUT,
                            fp8_dtype_forward,
                            grad_output_t,
                            ctx.fp8_meta["scaling_bwd"].scale_inv,
                            tex.FP8BwdTensors.GRAD_OUTPUT1,
                            fp8_dtype_backward,
                            ctx.activation_dtype,
                            get_workspace(),
                            accumulate=accumulate_wgrad_into_param_main_grad,
                            out=fc2_weight.main_grad
                            if ctx.fuse_wgrad_accumulation
                            else None,
                            use_split_accumulator=_2X_ACC_WGRAD,
                        )

                    if ctx.activation == 'gelu':
                        fc1_bias_grad, dgelu, dgelu_t = tex.fp8_cast_transpose_bgrad_dgelu_fused(
                            fc2_dgrad,
                            fc1_out,
                            ctx.fp8_meta["scaling_bwd"],
                            tex.FP8BwdTensors.GRAD_OUTPUT2,
                            fp8_dtype_backward,
                        )
                    else:
                        dgelu = activation_func(fc2_dgrad, fc1_out,
                                                TE_DType[fc2_dgrad.dtype])
                        fc1_bias_grad, dgelu, dgelu_t = tex.fp8_cast_transpose_bgrad_fused(
                            dgelu,
                            ctx.fp8_meta["scaling_bwd"],
                            tex.FP8BwdTensors.GRAD_OUTPUT2,
                            fp8_dtype_backward,
                        )
                else:
                    if fc2_weight.requires_grad:
                        gelu_out_c = tex.cast_from_fp8(
                            gelu_out,
                            ctx.fp8_meta["scaling_fwd"],
                            tex.FP8FwdTensors.GEMM2_INPUT,
                            fp8_dtype_forward,
                            TE_DType[ctx.activation_dtype],
                        )
                        fc2_wgrad, _, _ = tex.gemm(
                            gelu_out_c,
                            grad_output,
                            ctx.activation_dtype,
                            get_workspace(),
                            layout="NT",
                            grad=True,
                            use_bias=False,
                            accumulate=accumulate_wgrad_into_param_main_grad,
                            out=fc2_weight.main_grad
                            if ctx.fuse_wgrad_accumulation
                            else None,
                        )

                    if ctx.activation == 'gelu':
                        fc1_bias_grad, dgelu_no_fp8 = bgrad_dgelu_fused(
                            fc2_dgrad, fc1_out, fc1_bias
                        )
                    else:
                        dgelu_no_fp8 = activation_func(fc2_dgrad,
                                                       fc1_out,
                                                       TE_DType[fc2_dgrad.dtype])
                        fc1_bias_grad = dgelu_no_fp8.sum(dim=0)

                    dgelu = tex.cast_to_fp8(
                        dgelu_no_fp8,
                        ctx.fp8_meta["scaling_bwd"],
                        tex.FP8BwdTensors.GRAD_OUTPUT2,
                        fp8_dtype_backward,
                    )
                    dgelu_t = None

                fc1_dgrad_size = list(dgelu.size())
                fc1_dgrad_size[1] = fc1_weight.size(1)
                if ctx.ub_bulk_wgrad: # allocate dgrad output
                    ub_obj_dgrad = get_ub("fc1_wgrad")
                    fc1_dgrad = ub_obj_dgrad.get_ubuf_output(1) # AllGather output
                else:
                    fc1_dgrad = torch.empty(
                        fc1_dgrad_size, dtype=ctx.activation_dtype, device=fc1_weight.device
                    )
                # FC1 DGRAD: Unconditional
                _ = tex.fp8_gemm(
                    fc1_weight_t_fp8,
                    fwd_scale_inverses,
                    tex.FP8FwdTensors.GEMM1_WEIGHT,
                    fp8_dtype_forward,
                    dgelu,
                    ctx.fp8_meta["scaling_bwd"].scale_inv,
                    tex.FP8BwdTensors.GRAD_OUTPUT2,
                    fp8_dtype_backward,
                    ctx.activation_dtype,
                    get_workspace(),
                    out=fc1_dgrad,
                    use_split_accumulator=_2X_ACC_DGRAD,
                    ub_algo=tex.UbufOverlapAlgo.BULK_OVERLAP_AG if ctx.ub_bulk_dgrad else None,
                    ub=ub_obj_lnout if ctx.ub_bulk_dgrad else None
                )
            else:
                # FC2 DGRAD; Unconditional
                fc2_dgrad, _, _ = tex.gemm(
                    fc2_weight,
                    grad_output,
                    ctx.activation_dtype,
                    get_workspace(),
                    layout="NN",
                    gelu=(not ctx.bias_gelu_nvfusion) and (ctx.activation == 'gelu'),
                    grad=True,
                    gelu_input=fc1_out,
                    ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG if ctx.ub_split_ag else None,
                    ub=ctx.ub_obj_gradout if ctx.ub_split_ag else None,
                )

                # FC2 WGRAD
                if fc2_weight.requires_grad:
                    fc2_wgrad, fc2_bias_grad, _ = tex.gemm(
                        gelu_out,
                        grad_output,
                        ctx.activation_dtype,
                        get_workspace(),
                        layout="NT",
                        grad=True,
                        use_bias=ctx.use_fc2_bias,
                        accumulate=accumulate_wgrad_into_param_main_grad,
                        out=fc2_weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                    )

                if ctx.bias_gelu_nvfusion and ctx.activation == 'gelu':
                    fc1_bias_grad, dgelu = bgrad_dgelu_fused(fc2_dgrad, fc1_out, fc1_bias)
                else:
                    if ctx.activation == 'gelu':
                        dgelu = fc2_dgrad
                    else:
                        dgelu = activation_func(fc2_dgrad,
                                                fc1_out,
                                                TE_DType[fc2_dgrad.dtype])

                    # For non-fp8 execution, FC1 bias gradient is fused with FC1 wgrad GEMM
                    # and will not be calculated in case wgrad is not required.
                    if not fc1_weight.requires_grad:
                        fc1_bias_grad = dgelu.sum(dim=0)

                fc1_dgrad_size = list(dgelu.size())
                fc1_dgrad_size[1] = fc1_weight.size(1)
                if ctx.ub_bulk_wgrad: # allocate dgrad output
                    ub_obj_dgrad = get_ub("fc1_wgrad")
                    fc1_dgrad = ub_obj_dgrad.get_ubuf_output(1) # AllGather output
                else:
                    fc1_dgrad = torch.empty(
                        fc1_dgrad_size, dtype=ctx.activation_dtype, device=fc1_weight.device
                    )
                # FC1 DGRAD: Unconditional
                _, _, _ = tex.gemm(
                    fc1_weight,
                    dgelu,
                    ctx.activation_dtype,
                    get_workspace(),
                    out=fc1_dgrad,
                    layout="NN",
                    grad=True,
                    ub_algo=tex.UbufOverlapAlgo.BULK_OVERLAP_AG if ctx.ub_bulk_dgrad else None,
                    ub=ub_obj_lnout if ctx.ub_bulk_dgrad else None
                )

            if ctx.ub_bulk_dgrad:
                ln_out_total = ub_obj_lnout.get_ubuf_output(1)
            # Overlap dgrad-RS/AR with wgrad
            if ctx.set_parallel_mode and ctx.sequence_parallel:
                if not ctx.ub_bulk_dgrad and handle is not None:
                    handle.wait()
                if not ctx.ub_bulk_wgrad:
                    fc1_dgrad, handle = reduce_scatter_along_first_dim(
                        fc1_dgrad, ctx.tp_group, async_op=True
                    )
            elif ctx.set_parallel_mode and ctx.tensor_parallel:
                fc1_dgrad, handle = allreduce(fc1_dgrad, ctx.tp_group, async_op=True)

            if fc1_weight.requires_grad:
                if ctx.fp8:
                    # FC1 WGRAD
                    if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                        ln_out_total_t = tex.fp8_transpose(ln_out_total, fp8_dtype_forward)
                        fc1_wgrad = tex.fp8_gemm(
                            ln_out_total_t,
                            fwd_scale_inverses,
                            tex.FP8FwdTensors.GEMM1_INPUT,
                            fp8_dtype_forward,
                            dgelu_t,
                            ctx.fp8_meta["scaling_bwd"].scale_inv,
                            tex.FP8BwdTensors.GRAD_OUTPUT2,
                            fp8_dtype_backward,
                            ctx.activation_dtype,
                            get_workspace(),
                            accumulate=accumulate_wgrad_into_param_main_grad,
                            out=fc1_weight.main_grad
                            if ctx.fuse_wgrad_accumulation
                            else None,
                            use_split_accumulator=_2X_ACC_WGRAD,
                            ub_algo=tex.UbufOverlapAlgo.BULK_OVERLAP_RS
                            if ctx.ub_bulk_wgrad else None,
                            ub=ub_obj_dgrad if ctx.ub_bulk_wgrad else None,
                        )
                    else:
                        ln_out_total_c = tex.cast_from_fp8(
                            ln_out_total,
                            ctx.fp8_meta["scaling_fwd"],
                            tex.FP8FwdTensors.GEMM1_INPUT,
                            fp8_dtype_forward,
                            TE_DType[ctx.activation_dtype],
                        )
                        fc1_wgrad, _, _ = tex.gemm(
                            ln_out_total_c,
                            dgelu_no_fp8,
                            ctx.activation_dtype,
                            get_workspace(),
                            layout="NT",
                            grad=True,
                            accumulate=accumulate_wgrad_into_param_main_grad,
                            out=fc1_weight.main_grad
                            if ctx.fuse_wgrad_accumulation
                            else None,
                            ub_algo=tex.UbufOverlapAlgo.BULK_OVERLAP_RS
                            if ctx.ub_bulk_wgrad else None,
                            ub=ub_obj_dgrad if ctx.ub_bulk_wgrad else None,
                        )
                else:
                    # FC1 WGRAD
                    fc1_wgrad_outputs = tex.gemm(
                        ln_out_total,
                        dgelu,
                        ctx.activation_dtype,
                        get_workspace(),
                        layout="NT",
                        grad=True,
                        use_bias=not ctx.bias_gelu_nvfusion,
                        accumulate=accumulate_wgrad_into_param_main_grad,
                        out=fc1_weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                        ub_algo=tex.UbufOverlapAlgo.BULK_OVERLAP_RS if ctx.ub_bulk_wgrad else None,
                        ub=ub_obj_dgrad if ctx.ub_bulk_wgrad else None
                    )

                    if ctx.bias_gelu_nvfusion:
                        fc1_wgrad, _, _ = fc1_wgrad_outputs
                    else:
                        fc1_wgrad, fc1_bias_grad, _ = fc1_wgrad_outputs

            # Column Parallel Linear
            if ctx.ub_bulk_wgrad:
                fc1_dgrad = ub_obj_dgrad.get_ubuf_output(0) # Reduce-scatter output
            elif ctx.set_parallel_mode and ctx.tensor_parallel and handle is not None:
                handle.wait()

            # LayerNorm gradient
            d_ln_out = fc1_dgrad.view(inputmat.shape)

            # Residual gradient
            if ctx.return_layernorm_output:
                d_ln_out = d_ln_out + grad_outputs[1].view_as(d_ln_out)

            if ctx.normalization == "LayerNorm":
                dxmat, dgamma, dbeta = tex.layernorm_bwd(
                    d_ln_out, inputmat, mu, rsigma, ln_weight,
                    ctx.bwd_ln_sm_margin, ctx.zero_centered_gamma
                )
            elif ctx.normalization == "RMSNorm":
                dxmat, dgamma = tex.rmsnorm_bwd(
                    d_ln_out, inputmat, rsigma, ln_weight,
                    ctx.bwd_ln_sm_margin, ctx.zero_centered_gamma
                )
                dbeta = None

        return (
            dxmat.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            dgamma,
            dbeta,
            fc1_wgrad if fc1_weight.requires_grad else None,
            None,
            None,
            fc1_bias_grad if ctx.use_fc1_bias else None,
            None,
            fc2_wgrad if fc2_weight.requires_grad else None,
            None,
            None,
            fc2_bias_grad if ctx.use_fc2_bias else None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
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
          Options: 'gelu', 'geglu', 'relu', 'reglu', 'squared_relu', 'swiglu'.
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
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will allocated. It is the user's
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
        normalization: str = 'LayerNorm',
        activation : str = "gelu",
        output_layer_init_method: Optional[Callable] = None,
        fuse_wgrad_accumulation: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        return_layernorm_output: bool = False,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        set_parallel_mode: bool = False,
        zero_centered_gamma: bool = False,
        ub_bulk_wgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_split_rs: bool = False,
        ub_split_ag: bool = False,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.normalization = normalization
        assert normalization in ['LayerNorm', 'RMSNorm'], "Unsupported normalization type!"
        self.use_bias = bias
        self.activation = activation
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.return_layernorm_output = return_layernorm_output
        self.bias_gelu_nvfusion = (bool(int(os.getenv("NVTE_BIAS_GELU_NVFUSION", "1"))) and
                                   self.activation == 'gelu')
        self.set_parallel_mode = set_parallel_mode
        self.zero_centered_gamma = zero_centered_gamma
        self.ub_bulk_wgrad = ub_bulk_wgrad
        self.ub_bulk_dgrad = ub_bulk_dgrad
        self.ub_split_rs = ub_split_rs
        self.ub_split_ag = ub_split_ag

        if ub_bulk_wgrad or ub_bulk_dgrad or ub_split_rs or ub_split_ag:
            assert (
                tex.userbuf_comm_available()
            ), "Userbuffer communication backend not available."

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

        # LN init
        self.eps = eps
        self.layer_norm_weight = Parameter(
            torch.empty(hidden_size, device=device, dtype=params_dtype)
        )
        setattr(self.layer_norm_weight, "sequence_parallel", self.sequence_parallel)
        if self.normalization != "RMSNorm":
            self.layer_norm_bias = Parameter(
                torch.empty(hidden_size, device=device, dtype=params_dtype)
            )
            setattr(self.layer_norm_bias, "sequence_parallel", self.sequence_parallel)
        else:
            self.layer_norm_bias = None
        self.reset_layer_norm_parameters()

        if self.activation in ['reglu', 'geglu', 'swiglu']:
            fc1_output_features = 2 * self.size_per_partition
        else:
            fc1_output_features = self.size_per_partition
        # FC1 init
        self.fc1_weight = Parameter(
            torch.empty(fc1_output_features, hidden_size, device=device, dtype=params_dtype)
        )
        self.fp8_weight_shapes.append(self.fc1_weight.shape)

        initialize_affine_weight_gpu(
            self.fc1_weight,
            init_method,
            get_rng_state_tracker,
            partition_dim=0,
            stride=1,
        )

        if self.use_bias:
            self.fc1_bias = Parameter(
                torch.empty(fc1_output_features, device=device, dtype=params_dtype)
            )
            set_tensor_model_parallel_attributes(self.fc1_bias, True, 0, 1)
        else:
            self.fc1_bias = torch.Tensor().to(dtype=params_dtype, device=device)

        with torch.no_grad():
            self.fc1_bias.zero_()

        # FC2 init
        self.fc2_weight = Parameter(
            torch.empty(hidden_size, self.size_per_partition, device=device, dtype=params_dtype)
        )
        self.fp8_weight_shapes.append(self.fc2_weight.shape)

        initialize_affine_weight_gpu(
            self.fc2_weight,
            output_layer_init_method,
            get_rng_state_tracker,
            partition_dim=1,
            stride=1,
        )

        if self.use_bias:
            self.fc2_bias = Parameter(
                torch.empty(hidden_size, device=device, dtype=params_dtype)
            )
        else:
            self.fc2_bias = torch.Tensor().to(dtype=params_dtype, device=device)

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.set_parallel_mode and self.apply_bias:
            self.gemm_bias_unfused_add = True
        else:
            self.gemm_bias_unfused_add = False

        with torch.no_grad():
            self.fc2_bias.zero_()

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

    def reset_layer_norm_parameters(self) -> None:
        """Init LN params"""
        if not self.zero_centered_gamma:
            init.ones_(self.layer_norm_weight)
        else:
            init.zeros_(self.layer_norm_weight)
        if self.layer_norm_bias is not None:
            init.zeros_(self.layer_norm_bias)

    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        """
        Fetch the fp8 weight tensor placeholders if they exist (when
        `is_first_microbatch` is not `None`) or return empty fp8 weight
        tensors (if `is_first_microbatch is None`)
        """
        if not self.fp8:
            return [None, None, None, None]

        if is_first_microbatch is None:
            # Return empty weight placeholders for each fwd/bwd pass
            fp8_weight_tensors = self.get_fp8_weights_empty_tensors(
                is_first_microbatch
            )
        else:
            # These persistent weight placeholders should've been created in
            # `set_fp8_weights` method
            fp8_weight_tensors = [self.weight1_fp8, self.weight1_t_fp8,
                                  self.weight2_fp8, self.weight2_t_fp8]

        return fp8_weight_tensors

    @no_torch_dynamo
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

        with self.prepare_forward(inp, is_first_microbatch, num_gemms=2) as inp:
            # Fetch the fp8 weights placeholders (for linear/gemm)
            weight1_fp8, weight1_t_fp8, weight2_fp8, weight2_t_fp8 = \
                self.get_fp8_weights_scratchpad(
                        is_first_microbatch
                )

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
                self.fc1_weight,
                weight1_fp8,
                weight1_t_fp8,
                self.fc1_bias,
                self.use_bias,
                self.fc2_weight,
                weight2_fp8,
                weight2_t_fp8,
                self.fc2_bias,
                self.apply_bias and not self.gemm_bias_unfused_add,
                self.eps,
                is_first_microbatch,
                self.fp8,
                self.fp8_calibration,
                self.fp8_meta,
                self.fuse_wgrad_accumulation,
                self.tp_group,
                self.tp_size,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.return_layernorm_output,
                self.bias_gelu_nvfusion,
                self.set_parallel_mode,
                torch.is_grad_enabled(),
                self.fwd_ln_sm_margin,
                self.bwd_ln_sm_margin,
                self.zero_centered_gamma,
                self.ub_bulk_wgrad,
                self.ub_bulk_dgrad,
                self.ub_split_rs,
                self.ub_split_ag,
                self.activation,
                self.normalization,
            )
            out = fwd_fn(*args)

        if self.return_layernorm_output:
            out, ln_out = out

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(self.fc2_bias, self.activation_dtype)

        if self.return_bias:
            if self.return_layernorm_output:
                return out, cast_if_needed(self.fc2_bias, self.activation_dtype), ln_out
            return out, cast_if_needed(self.fc2_bias, self.activation_dtype)
        if self.return_layernorm_output:
            return out, ln_out
        return out

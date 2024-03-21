# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LayerNormLinear API"""
import os
import warnings
from typing import Union, Optional, Callable, Tuple, List, Dict, Any

import torch
from torch.nn import init

from .. import cpp_extensions as tex

from .base import (
    get_workspace,
    _prepare_backward,
    get_ub,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ..fp8 import get_fp8_te_dtype, FP8GlobalStateManager
from ..utils import (
    divide,
    get_default_init_method,
    init_method_constant,
    cast_if_needed,
    assert_dim_for_fp8_exec,
    clear_tensor_data,
)
from ..distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    allreduce,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
    is_fp8_activation_recompute_enabled,
    in_fp8_activation_recompute_phase,
    _fsdp_scatter_tensors,
    _fsdp_gather_tensors,
)
from ..constants import GemmParallelModes, dist_group_type, TE_DType
from ..jit import no_torch_dynamo
from ._common import _apply_normalization, _noop_cat
from ..float8_tensor import Float8Tensor

__all__ = ["LayerNormLinear"]


class _LayerNormLinear(torch.autograd.Function):
    """LayerNormLinear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: Union[torch.Tensor, None],
        weight: torch.Tensor,
        weight_fp8: Union[torch.Tensor, None],
        weight_t_fp8: Union[torch.Tensor, None],
        bias: torch.Tensor,
        use_bias: bool,
        eps: float,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        fp8_meta: Dict[str, Any],
        fuse_wgrad_accumulation: bool,
        cpu_offloading: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        parallel_mode: Union[str, None],
        return_layernorm_output: bool,
        return_layernorm_output_gathered: bool,
        is_grad_enabled: bool,
        fwd_ln_sm_margin: int,
        bwd_ln_sm_margin: int,
        zero_centered_gamma: bool,
        normalization: str,
        primary_weights_in_fp8: bool,
        ub_bulk_wgrad: bool,
        ub_bulk_dgrad: bool,
        ub_split_ag: bool,
        ub_atomic_gemm_ag: bool,
        ub_name: str,
        fsdp_group: Union[dist_group_type, None],
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        # Make sure input dimensions are compatible
        in_features = ln_weight.numel()
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view((-1, in_features))
        if fp8:
            assert_dim_for_fp8_exec(inputmat)
            assert_dim_for_fp8_exec(weight)

        update_fp8_weights = is_first_microbatch is None or is_first_microbatch

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        ln_weight = cast_if_needed(ln_weight, activation_dtype)
        if ln_bias is not None:
            ln_bias = cast_if_needed(ln_bias, activation_dtype)

        if ub_split_ag or ub_atomic_gemm_ag:
            tp_world_size = get_distributed_world_size(tp_group)
            if tp_world_size == 1 or (not is_grad_enabled) or return_layernorm_output:
                ub_split_ag = False
                ub_atomic_gemm_ag = False
        if ub_split_ag or ub_atomic_gemm_ag:
            dim_size = list(inputmat.size())
            dim_size[0] = dim_size[0] * tp_world_size
            ub_obj_lnout = get_ub(ub_name+"_fprop")
            ln_out = ub_obj_lnout.get_ubuf_output(0)
        else:
            ln_out_dtype = torch.uint8 if (fp8 and not return_layernorm_output) else inputmat.dtype
            ln_out = torch.empty_like(inputmat, dtype=ln_out_dtype)
        if ub_atomic_gemm_ag:
            assert fp8, "AtomicGemm overlap supported only for FP8 GEMM."

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

        # Column Parallel Linear
        ln_out_gathered = False
        if ub_split_ag or ub_atomic_gemm_ag:
            ln_out_total = ub_obj_lnout.get_ubuf_output(1)
            ln_out = torch.empty_like(ln_out)
        elif parallel_mode == "column" and sequence_parallel:
            ln_out_gathered = True
            ln_out_total, _ = gather_along_first_dim(ln_out, tp_group)
        else:
            ln_out_total = ln_out

        # If residual connection is after LN, we need `ln_out_return`
        # tensor in higher precision, this comes at the cost
        # of an extra fp8 cast.
        if return_layernorm_output:
            ln_out_return = ln_out_total if return_layernorm_output_gathered else ln_out
            if fp8:
                ln_out = tex.cast_to_fp8(
                    ln_out,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_forward,
                )

        if fp8:
            bias_dtype = (
                torch.bfloat16
                if activation_dtype == torch.float32
                else activation_dtype
            )
            bias = cast_if_needed(bias, bias_dtype) if use_bias else bias

            if primary_weights_in_fp8:
                # Weight is already in FP8
                weight.reset_fp8_meta_scale_inv()
                weight_fp8 = weight
                weight_t_fp8 = None
            elif update_fp8_weights:
                # Gather Fp8 weight buffers if needed
                if fsdp_group is not None and weight_fp8._data.shape != weight.data.shape:
                    _fsdp_gather_tensors(fsdp_group, [weight.data.shape], weight_fp8)
                # Need to cast weights to FP8
                weight_fp8 = Float8Tensor(
                    data=weight_fp8._data,
                    fp8_meta=fp8_meta,
                    fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT,
                )
                if (is_grad_enabled
                    or (is_fp8_activation_recompute_enabled()
                        and not in_fp8_activation_recompute_phase())):
                    # Gather Fp8 transposed-weight buffers if needed
                    weight_t_fp8_shape = tuple(reversed(weight_t_fp8._data.shape))
                    if (fsdp_group is not None
                        and weight_t_fp8._data.shape != weight_t_fp8_shape):
                        _fsdp_gather_tensors(fsdp_group,
                                             [weight_t_fp8_shape],
                                             weight_t_fp8)
                    tex.fp8_cast_transpose_fused(
                        weight,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                        cast_out=weight_fp8._data,
                        transpose_out=weight_t_fp8._data,
                    )
                else:
                    tex.cast_to_fp8(
                        weight,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                        out=weight_fp8._data,
                    )
                    weight_t_fp8 = None

            ub_algo = tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG if ub_split_ag else None
            ub_algo = tex.UbufOverlapAlgo.ATOMIC_GEMM_AG if ub_atomic_gemm_ag else ub_algo
            out, _ = tex.fp8_gemm(
                weight_fp8._data,
                fp8_meta["scaling_fwd"].scale_inv,
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                fp8_dtype_forward,
                ln_out_total,
                fp8_meta["scaling_fwd"].scale_inv,
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                activation_dtype,
                get_workspace(),
                bias=bias,
                use_bias=use_bias,
                use_split_accumulator=_2X_ACC_FPROP,
                ub_algo=ub_algo,
                ub=ub_obj_lnout if (ub_split_ag or ub_atomic_gemm_ag) else None,
                extra_output_tensor=ln_out if (ub_split_ag or ub_atomic_gemm_ag) else None,
            )
        else:
            # Cast for native AMP
            weight = cast_if_needed(weight, activation_dtype)
            bias = cast_if_needed(bias, activation_dtype) if use_bias else bias

            if fp8_calibration:
                # amax of input
                amin, amax = ln_out_total.aminmax()
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM1_INPUT] = \
                    torch.max(-amin, amax).float()
                # amax of weight
                amin, amax = weight.aminmax()
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM1_WEIGHT] = \
                    torch.max(-amin, amax).float()

            out, _, _ = tex.gemm(
                weight,
                ln_out_total,
                activation_dtype,
                get_workspace(),
                bias=bias,
                use_bias=use_bias,
                ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG if ub_split_ag else None,
                ub=ub_obj_lnout if ub_split_ag else None,
                extra_output_tensor=ln_out if ub_split_ag else None,
            )

        if is_grad_enabled:
            if cpu_offloading:
                if fuse_wgrad_accumulation:
                    weight.main_grad.weight_offloading = True
                if fp8 and weight_t_fp8 is not None:
                    weight_t_fp8.weight_offloading = True
                ln_weight.weight_offloading = True
                weight.weight_offloading = True

                inputmat.activation_offloading = True
                if normalization == "LayerNorm":
                    mu.activation_offloading = True
                rsigma.activation_offloading = True
                ln_out.activation_offloading = True

            # Scatter intermediate/activation tensors saved for the backward pass
            ctx.fsdp_group = fsdp_group
            ctx.fsdp_shapes = _fsdp_scatter_tensors(
                fsdp_group,
                mu,
                rsigma,
                weight_t_fp8,
                ln_out
            )

            ctx.save_for_backward(
                inputmat,
                ln_weight,
                mu,
                rsigma,
                weight,
                weight.main_grad if cpu_offloading and fuse_wgrad_accumulation else None,
                weight_t_fp8,
                ln_out,
                fp8_meta["scaling_fwd"].scale_inv.clone() if fp8 else None
            )

            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fp8_meta = fp8_meta
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = use_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp.shape
            ctx.parallel_mode = parallel_mode
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.return_layernorm_output = return_layernorm_output
            ctx.return_layernorm_output_gathered = return_layernorm_output_gathered \
                                                   and ln_out_gathered
            ctx.bwd_ln_sm_margin = bwd_ln_sm_margin
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.ub_bulk_wgrad = ub_bulk_wgrad
            ctx.ub_bulk_dgrad = ub_bulk_dgrad
            ctx.ub_name = ub_name
            ctx.requires_dgrad = inp.requires_grad
            ctx.normalization = normalization

        # Row Parallel Linear
        if parallel_mode == "row" and sequence_parallel:
            out, _ = reduce_scatter_along_first_dim(out, tp_group)
        elif parallel_mode == "row" and tensor_parallel:
            out, _ = allreduce(out, tp_group)

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        out = out.view(-1, *inp.shape[1:-1], out.shape[-1])

        # Scatter Fp8 weight buffers
        _fsdp_scatter_tensors(fsdp_group, weight_fp8, weight_fp8)

        if return_layernorm_output:
            if return_layernorm_output_gathered:
                shape = list(inp.shape)
                shape[0] *= tp_size
                return out, ln_out_return.view(shape)
            return out, ln_out_return.view_as(inp)
        return out


    @staticmethod
    def backward(
        ctx, *grad_outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        with _prepare_backward(
            ctx.fp8, ctx.fp8_meta, ctx.tp_group, ctx.tp_size, name="_LayerNormLinear"
        ):
            (
                inputmat,
                ln_weight,
                mu,
                rsigma,
                weight,
                main_grad,
                weight_t_fp8,
                ln_out,
                fwd_scale_inverses,
            ) = ctx.saved_tensors

            # Gather intermediate/activation tensors if needed
            _fsdp_gather_tensors(
                ctx.fsdp_group,
                ctx.fsdp_shapes,
                mu,
                rsigma,
                weight_t_fp8,
                ln_out
            )

            if ctx.cpu_offloading and ctx.fuse_wgrad_accumulation:
                weight = torch.nn.Parameter(weight, False)
                weight.main_grad = main_grad

            # Primary weights are in FP8.
            if ctx.fp8 and weight_t_fp8 is None:
                weight_t_fp8 = weight.transpose(
                    update_cache="reuse_only" if ctx.is_first_microbatch is None else "lazy",
                )

            if ctx.ub_bulk_dgrad:
                tp_world_size = get_distributed_world_size(ctx.tp_group)
                if tp_world_size == 1:
                    ctx.ub_bulk_dgrad = False
            if ctx.ub_bulk_dgrad:
                dim_size = list(ln_out.size())
                dim_size[0] = dim_size[0] * tp_world_size
                ub_obj_lnout = get_ub(ctx.ub_name+"_dgrad")
                ub_obj_lnout.copy_input_to_ubuf(ln_out, 1)
            (
                grad_output,
                grad_output_c,
                grad_output_t,
                grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx, grad_outputs[0], ctx.parallel_mode == "row"
            )

            if ctx.ub_bulk_wgrad:
                tp_world_size = get_distributed_world_size(ctx.tp_group)
                if tp_world_size == 1 or not weight.requires_grad:
                    ctx.ub_bulk_wgrad = False

            # Column Parallel Linear
            # Overlap input AG with dgrad
            if (weight.requires_grad
                and (not ctx.ub_bulk_dgrad)
                and ctx.parallel_mode == "column"
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


            dgrad_size = list(grad_output.size())
            dgrad_size[1] = weight.size(1)
            if ctx.ub_bulk_wgrad: # allocate dgrad output
                ub_obj_dgrad = get_ub(ctx.ub_name+"_wgrad")
                dgrad = ub_obj_dgrad.get_ubuf_output(1) # AllGather output
            else:
                dgrad = torch.empty(dgrad_size, dtype=ctx.activation_dtype, device=weight.device)

            if ctx.fp8:
                fp8_dtype_forward = get_fp8_te_dtype(
                    ctx.fp8_meta["recipe"], fprop_tensor=True
                )
                fp8_dtype_backward = get_fp8_te_dtype(
                    ctx.fp8_meta["recipe"], fprop_tensor=False
                )
                out_index, meta_tensor, out_te_type, out_type = (
                    None, None, None, ctx.activation_dtype)
                if ctx.ub_bulk_wgrad and ub_obj_dgrad.is_fp8_ubuf():
                    out_index = tex.FP8BwdTensors.GRAD_INPUT1
                    meta_tensor = ctx.fp8_meta["scaling_bwd"]
                    out_te_type = fp8_dtype_backward
                    out_type = torch.uint8
                    ub_obj_dgrad.set_ubuf_scale_inv(meta_tensor.scale_inv[out_index])

                # DGRAD: Evaluated unconditionally to feed into Linear backward
                _ = tex.fp8_gemm(
                    weight_t_fp8._data,
                    fwd_scale_inverses,
                    tex.FP8FwdTensors.GEMM1_WEIGHT,
                    fp8_dtype_forward,
                    grad_output_c,
                    ctx.fp8_meta["scaling_bwd"].scale_inv,
                    tex.FP8BwdTensors.GRAD_OUTPUT1,
                    fp8_dtype_backward,
                    out_type,
                    get_workspace(),
                    out=dgrad,
                    use_split_accumulator=_2X_ACC_DGRAD,
                    ub_algo=tex.UbufOverlapAlgo.BULK_OVERLAP_AG if ctx.ub_bulk_dgrad else None,
                    ub=ub_obj_lnout if ctx.ub_bulk_dgrad else None,
                    out_index=out_index,
                    fp8_meta_tensor = meta_tensor,
                    D_dtype = out_te_type,
                )
                clear_tensor_data(grad_output_c)
            else:
                # DGRAD: Evaluated unconditionally to feed into Linear backward
                _, _, _ = tex.gemm(
                    weight,
                    grad_output,
                    ctx.activation_dtype,
                    get_workspace(),
                    out=dgrad,
                    layout="NN",
                    grad=True,
                    ub_algo=tex.UbufOverlapAlgo.BULK_OVERLAP_AG if ctx.ub_bulk_dgrad else None,
                    ub=ub_obj_lnout if ctx.ub_bulk_dgrad else None
                )
            if ctx.ub_bulk_dgrad:
                ln_out_total = ub_obj_lnout.get_ubuf_output(1)

            # Overlap dgrad-RS/AR with wgrad
            if ctx.parallel_mode == "column" and ctx.sequence_parallel:
                if not ctx.ub_bulk_dgrad and handle is not None:
                    handle.wait()
                if not ctx.ub_bulk_wgrad:
                    if ctx.return_layernorm_output and ctx.return_layernorm_output_gathered:
                        dgrad = dgrad + grad_outputs[1].view_as(dgrad)
                    dgrad, handle = reduce_scatter_along_first_dim(
                        dgrad, ctx.tp_group, async_op=True
                    )
            elif ctx.parallel_mode == "column" and ctx.tensor_parallel:
                dgrad, handle = allreduce(dgrad, ctx.tp_group, async_op=True)

            if weight.requires_grad:
                if ctx.fp8:
                    # WGRAD
                    extra_output_tensor = None
                    if ctx.ub_bulk_wgrad:
                        if ub_obj_dgrad.is_fp8_ubuf():
                            dim_size = list(ub_obj_dgrad.get_ubuf_output(0).size()) # RS output
                            extra_output_tensor = torch.empty(
                                dim_size, dtype=ctx.activation_dtype, device=dgrad.device)
                            dgrad = extra_output_tensor
                        else:
                            dgrad = ub_obj_dgrad.get_ubuf_output(0)
                    if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                        ln_out_total_t = tex.fp8_transpose(ln_out_total, fp8_dtype_forward)
                        wgrad, _ = tex.fp8_gemm(
                            ln_out_total_t,
                            fwd_scale_inverses,
                            tex.FP8FwdTensors.GEMM1_INPUT,
                            fp8_dtype_forward,
                            grad_output_t,
                            ctx.fp8_meta["scaling_bwd"].scale_inv,
                            tex.FP8BwdTensors.GRAD_OUTPUT1,
                            fp8_dtype_backward,
                            ctx.activation_dtype,
                            get_workspace(),
                            accumulate=accumulate_wgrad_into_param_main_grad,
                            out=weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                            use_split_accumulator=_2X_ACC_WGRAD,
                            ub_algo=tex.UbufOverlapAlgo.BULK_OVERLAP_RS
                            if ctx.ub_bulk_wgrad else None,
                            ub=ub_obj_dgrad if ctx.ub_bulk_wgrad else None,
                            extra_output_tensor=extra_output_tensor
                        )
                        clear_tensor_data(ln_out_total_t, grad_output_t)
                    else:
                        ln_out_total_c = torch.ops.tex_ts.cast_from_fp8_ts(
                            ln_out_total,
                            fwd_scale_inverses,
                            tex.FP8FwdTensors.GEMM1_INPUT,
                            fp8_dtype_forward,
                            TE_DType[ctx.activation_dtype],
                        )
                        wgrad, _, _ = tex.gemm(
                            ln_out_total_c,
                            grad_output,
                            ctx.activation_dtype,
                            get_workspace(),
                            layout="NT",
                            grad=True,
                            accumulate=accumulate_wgrad_into_param_main_grad,
                            out=weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                            ub_algo=tex.UbufOverlapAlgo.BULK_OVERLAP_RS
                            if ctx.ub_bulk_wgrad else None,
                            ub=ub_obj_dgrad if ctx.ub_bulk_wgrad else None,
                            extra_output_tensor=extra_output_tensor
                        )
                        clear_tensor_data(ln_out_total_c)
                else:
                    # WGRAD
                    wgrad, grad_bias, _ = tex.gemm(
                        ln_out_total,
                        grad_output,
                        ctx.activation_dtype,
                        get_workspace(),
                        layout="NT",
                        grad=True,
                        use_bias=ctx.use_bias,
                        accumulate=accumulate_wgrad_into_param_main_grad,
                        out=weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                        ub_algo=tex.UbufOverlapAlgo.BULK_OVERLAP_RS if ctx.ub_bulk_wgrad else None,
                        ub=ub_obj_dgrad if ctx.ub_bulk_wgrad else None
                    )
                    clear_tensor_data(ln_out_total)
                    if ctx.ub_bulk_wgrad:
                        dgrad = ub_obj_dgrad.get_ubuf_output(0) # Reduce-scatter output

            # Column Parallel Linear
            if ((not ctx.ub_bulk_wgrad)
                and ctx.parallel_mode == "column"
                and ctx.tensor_parallel
                and handle is not None):
                handle.wait()

            # LayerNorm gradient
            dgrad = dgrad.view(inputmat.shape)

            # Residual gradient
            if ctx.return_layernorm_output and not ctx.return_layernorm_output_gathered:
                dgrad = dgrad + grad_outputs[1].view_as(dgrad)

            if ctx.normalization == "LayerNorm":
                dgrad, dgamma, dbeta = tex.layernorm_bwd(
                    dgrad, inputmat, mu, rsigma, ln_weight,
                    ctx.bwd_ln_sm_margin, ctx.zero_centered_gamma
                )
            elif ctx.normalization == "RMSNorm":
                dgrad, dgamma = tex.rmsnorm_bwd(
                    dgrad, inputmat, rsigma, ln_weight,
                    ctx.bwd_ln_sm_margin, ctx.zero_centered_gamma
                )
                dbeta = None
            clear_tensor_data(mu)
            clear_tensor_data(rsigma)

            if not ctx.use_bias:
                grad_bias = None

        if weight.requires_grad:
            # Handle custom DDP from mcore.
            if ctx.fuse_wgrad_accumulation and hasattr(weight, 'grad_added_to_main_grad'):
                weight.grad_added_to_main_grad = True
                if getattr(weight, 'zero_out_wgrad', False):
                    wgrad = torch.zeros(weight.main_grad.shape,
                                        dtype=weight.dtype,
                                        device=torch.cuda.current_device(),
                                        requires_grad=False
                                       )
                else:
                    wgrad = torch.empty(weight.main_grad.shape,
                                        dtype=weight.dtype,
                                        device=torch.cuda.current_device(),
                                        requires_grad=False
                                       )
            elif ctx.fuse_wgrad_accumulation:
                wgrad = None
        else:
            wgrad = None

        # Scatter fp8 transposed-weight buffers
        _fsdp_scatter_tensors(ctx.fsdp_group, weight_t_fp8)

        return (
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            dgamma,
            dbeta,
            wgrad,
            None,
            None,
            grad_bias,
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
            None,
            None,
            None,
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
    bias : bool, default = `True`
          if set to `False`, the layer will not learn an additive bias.
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    init_method : Callable, default = `None`
                 used for initializing weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    return_layernorm_output : bool, default = `False`
                             if set to `True`, output of layernorm is returned from the forward
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module is
                             taken post layernorm.
    return_layernorm_output_gathered : bool, default = `False`
                             if set to `True`, output of layernorm is returned after the all
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
                      names that end in `_weight` or `_bias`, so trailing underscores are
                      stripped from any provided names.
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
    parallel_mode : {None, 'Column', 'Row'}, default = `None`
                   used to decide whether this Linear layer is Column Parallel Linear or Row
                   Parallel Linear as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
                   When set to `None`, no communication is performed.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias itself, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
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
        normalization: str = 'LayerNorm',
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        return_layernorm_output: bool = False,
        return_layernorm_output_gathered: bool = False,
        parameters_split: Optional[Union[Tuple[str, ...], Dict[str, int]]] = None,
        zero_centered_gamma: bool = False,
        device: Union[torch.device, str] = "cuda",
        ub_bulk_wgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_split_ag: bool = False,
        ub_atomic_gemm_ag: bool = False,
        ub_name: Optional[str] = None,
    ) -> None:
        super().__init__()

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.normalization = normalization
        assert normalization in ['LayerNorm', 'RMSNorm'], "Unsupported normalization type!"
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = self.use_bias and not return_bias
        self.return_layernorm_output = return_layernorm_output
        self.return_layernorm_output_gathered = return_layernorm_output_gathered
        self.zero_centered_gamma = zero_centered_gamma
        self.primary_weights_in_fp8 = FP8GlobalStateManager.with_fp8_parameters()
        self.ub_bulk_wgrad = ub_bulk_wgrad
        self.ub_bulk_dgrad = ub_bulk_dgrad
        self.ub_split_ag = ub_split_ag
        self.ub_atomic_gemm_ag = ub_atomic_gemm_ag
        if any([ub_bulk_wgrad, ub_bulk_dgrad, ub_split_ag]):
            assert ub_name is not None, "Userbuffer name [string] is not set."
        self.ub_name = ub_name


        if ub_bulk_wgrad or ub_bulk_dgrad or ub_split_ag or ub_atomic_gemm_ag:
            assert (
                tex.userbuf_comm_available()
            ), "Userbuffer communication backend not available."

        if ub_atomic_gemm_ag:
            warnings.warn(
                "Atomic gemm uses a beta API from cublas and is not tested for all use cases."
            )

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

        if init_method is None:
            init_method = get_default_init_method()

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        self.eps = eps
        layer_norm_weight = torch.nn.Parameter(
            torch.empty(in_features, device=device, dtype=params_dtype)
        )
        self.register_parameter('layer_norm_weight', layer_norm_weight,
                                init_fn=init_method_constant(float(not self.zero_centered_gamma)))
        if self.normalization != "RMSNorm":
            layer_norm_bias = torch.nn.Parameter(
                torch.empty(in_features, device=device, dtype=params_dtype)
            )
            self.register_parameter('layer_norm_bias', layer_norm_bias,
                                    init_fn=init_method_constant(0.0))
        else:
            self.layer_norm_bias = None

        self.weight_tensor = torch.empty(
            self.out_features, self.in_features,
            device=device, dtype=params_dtype)

        if self.use_bias:
            self.bias_tensor = torch.empty(
                self.out_features,
                device=device,
                dtype=params_dtype)
        else:
            self.bias_tensor = torch.Tensor().to(dtype=params_dtype, device=device)

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

        # Construct parameters from weight and bias buffers
        offset = 0
        for i, split_size in enumerate(self.parameter_split_sizes):
            split_start = offset
            offset += split_size
            split_end = offset

            # Check if parameters are subviews of buffers
            is_subview = (split_start, split_end) != (0, self.out_features)
            if is_subview and self.primary_weights_in_fp8:
                raise RuntimeError(
                    "Splitting Float8Tensor into multiple params "
                    "is not supported"
                )

            # Construct weight parameter
            weight = self.weight_tensor
            if is_subview:
                weight = weight[split_start:split_end]
            weight = torch.nn.Parameter(weight)
            self.register_parameter(self.weight_names[i], weight,
                                    init_fn=init_method,
                                    get_rng_state_tracker=get_rng_state_tracker,
                                    fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT)

            # Construct bias parameter if needed
            if self.use_bias:
                bias = self.bias_tensor
                if is_subview:
                    bias = bias[split_start:split_end]
                bias = torch.nn.Parameter(bias)
                self.register_parameter(self.bias_names[i], bias,
                                        init_fn=init_method_constant(0.0))
            else:
                bias = torch.Tensor().to(dtype=params_dtype, device=device)
                setattr(self, self.bias_names[i], bias)

            # Concatenated tensors are not needed if not splitting
            # into multiple parameters
            if not is_subview:
                del self.weight_tensor
                del self.bias_tensor

        if self.primary_weights_in_fp8:
            self.init_fp8_metadata()
            self.fp8_meta["update_amax_and_scale_fwd"] = True

        self.reset_parameters(defer_init=(device == 'meta'))

        self.fp8_weight_shapes.append(torch.Size((self.out_features, self.in_features)))

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

    def reset_layer_norm_parameters(self) -> None:
        """Init LN params"""
        warnings.warn(
            ("This method will be deprecated in an upcoming release. "
             "Update your code to use LayerNormLinear.reset_parameters() instead."),
            DeprecationWarning,
            stacklevel=2
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


    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        """
        Fetch the fp8 weight tensor placeholders if they exist (when
        `is_first_microbatch` is not `None`) or return empty fp8 weight
        tensors (if `is_first_microbatch is None`)
        """
        if not self.fp8 or self.primary_weights_in_fp8:
            return [None, None]

        if is_first_microbatch is None:
            # Return empty weight placeholders for each fwd/bwd pass
            fp8_weight_tensors = self.get_fp8_weights_empty_tensors(
                is_first_microbatch
            )
        else:
            # These persistent weight placeholders should've been created in
            # `set_fp8_weights` method
            fp8_weight_tensors = [self.weight1_fp8, self.weight1_t_fp8]

        return fp8_weight_tensors

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
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

        with self.prepare_forward(inp, is_first_microbatch) as inp:
            assert self.fp8 or not self.primary_weights_in_fp8, \
                   "Need to run inside fp8_autocast region when weights are stored in FP8."

            # Get concatenated weight and bias tensors
            if len(self.parameter_split_sizes) == 1:
                weight_tensor = getattr(self, self.weight_names[0])
                bias_tensor = getattr(self, self.bias_names[0])
            elif torch.is_grad_enabled():
                weight_tensor = _noop_cat(
                    [getattr(self, name) for name in self.weight_names],
                    self.weight_tensor,
                )
                if self.use_bias:
                    bias_tensor = _noop_cat(
                        [getattr(self, name) for name in self.bias_names],
                        self.bias_tensor,
                    )
                else:
                    bias_tensor = getattr(self, self.bias_names[0])  # Unused
            else:
                weight_tensor = self.weight_tensor
                bias_tensor = self.bias_tensor

            # Fetch the fp8 weights placeholders (for linear/gemm)
            weight1_fp8, weight1_t_fp8 = self.get_fp8_weights_scratchpad(
                is_first_microbatch
            )

            from ..cpu_offload import CPUOffloadEnabled

            if torch.is_grad_enabled():
                fwd_fn = _LayerNormLinear.apply
                args = []
            else:
                fwd_fn = _LayerNormLinear.forward
                args = [None]
            args += (
                inp,
                self.layer_norm_weight,
                self.layer_norm_bias,
                weight_tensor,
                weight1_fp8,
                weight1_t_fp8,
                bias_tensor,
                self.apply_bias and not self.gemm_bias_unfused_add,
                self.eps,
                is_first_microbatch,
                self.fp8,
                self.fp8_calibration,
                self.fp8_meta,
                self.fuse_wgrad_accumulation,
                CPUOffloadEnabled,
                self.tp_group,
                self.tp_size,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.parallel_mode,
                self.return_layernorm_output,
                self.return_layernorm_output_gathered,
                torch.is_grad_enabled(),
                self.fwd_ln_sm_margin,
                self.bwd_ln_sm_margin,
                self.zero_centered_gamma,
                self.normalization,
                self.primary_weights_in_fp8,
                self.ub_bulk_wgrad,
                self.ub_bulk_dgrad,
                self.ub_split_ag,
                self.ub_atomic_gemm_ag,
                self.ub_name,
                self.fsdp_group,
            )
            out = fwd_fn(*args)

        if self.return_layernorm_output:
            out, ln_out = out

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            if self.return_layernorm_output:
                return out, cast_if_needed(bias_tensor, self.activation_dtype), ln_out
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        if self.return_layernorm_output:
            return out, ln_out
        return out

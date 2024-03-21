# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
import warnings
from typing import Union, Optional, Callable, Tuple, List, Dict, Any

import torch

import transformer_engine_extensions as tex

from .base import (
    get_workspace,
    _prepare_backward,
    get_ub,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ._common import _noop_cat
from ..fp8 import get_fp8_te_dtype, FP8GlobalStateManager
from ..utils import (
    divide,
    cast_if_needed,
    assert_dim_for_fp8_exec,
    clear_tensor_data,
    init_method_constant,
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
from ..cpp_extensions import (
    fp8_gemm,
    gemm,
    fp8_cast_transpose_fused,
    cast_to_fp8,
)
from ..constants import GemmParallelModes, dist_group_type
from ..jit import no_torch_dynamo

from ..float8_tensor import Float8Tensor

__all__ = ["Linear"]


class _Linear(torch.autograd.Function):
    """Linear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        weight: Union[Float8Tensor, torch.Tensor],
        weight_fp8: Union[Float8Tensor, None],
        weight_t_fp8: Union[Float8Tensor, None],
        inp: torch.Tensor,
        bias: torch.Tensor,
        use_bias: bool,
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
        is_grad_enabled: bool,
        primary_weights_in_fp8: bool,
        ub_split_rs: bool,
        ub_split_ag: bool,
        ub_atomic_gemm_rs: bool,
        ub_atomic_gemm_ag: bool,
        ub_name: str,
        fsdp_group: Union[dist_group_type, None],
    ) -> torch.Tensor:
        # Make sure input dimensions are compatible
        in_features = weight.shape[-1]
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view((-1, in_features))
        if fp8:
            assert_dim_for_fp8_exec(inputmat)
            assert_dim_for_fp8_exec(weight)

        update_fp8_weights = is_first_microbatch is None or is_first_microbatch

        if ub_split_rs or ub_atomic_gemm_rs:
            tp_world_size = get_distributed_world_size(tp_group)
            if tp_world_size == 1:
                ub_split_rs = False
                ub_atomic_gemm_rs = False
        if ub_atomic_gemm_rs or ub_atomic_gemm_ag:
            assert fp8, "AtomicGemm overlap supported only for FP8 GEMM."

        # Cast input to expected dtype
        inputmat = cast_if_needed(inputmat, activation_dtype)
        inputmat_t = None
        inputmat_no_fp8 = inputmat
        if fp8:
            fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
            if (
                not fp8_meta["recipe"].override_linear_precision.wgrad
                and is_grad_enabled
                and weight.requires_grad
                and not sequence_parallel
            ):
                # FP8 input for forward, FP8 input transpose for backward wgrad
                inputmat, inputmat_t = fp8_cast_transpose_fused(
                    inputmat,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_forward,
                )
            else:
                # FP8 input for forward
                inputmat = cast_to_fp8(
                    inputmat,
                    fp8_meta["scaling_fwd"],
                    tex.FP8FwdTensors.GEMM1_INPUT,
                    fp8_dtype_forward,
                )

        # Column Parallel Linear
        if parallel_mode == "column" and sequence_parallel:
            inputmat_total, _ = gather_along_first_dim(inputmat, tp_group)
        else:
            inputmat_total = inputmat

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
                    fp8_cast_transpose_fused(
                        weight,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                        cast_out=weight_fp8._data,
                        transpose_out=weight_t_fp8._data,
                    )
                else:
                    cast_to_fp8(
                        weight,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                        out=weight_fp8._data,
                    )
                    weight_t_fp8 = None

            proj_out_index, meta_tensor, proj_out_tetype, proj_out_pttype = (
                None, None, None, activation_dtype)
            if ub_split_rs or ub_atomic_gemm_rs:
                ub_obj_projout = get_ub(ub_name+"_fprop")
                out = ub_obj_projout.get_ubuf_output(1)
                dim_size = list(inputmat_total.size())
                dim_size[0] = dim_size[0] // tp_world_size
                dim_size[1] = weight.size(0)
                rs_out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat_total.device)

                if ub_obj_projout.is_fp8_ubuf():
                    proj_out_index = tex.FP8FwdTensors.GEMM1_OUTPUT
                    meta_tensor = fp8_meta["scaling_fwd"]
                    proj_out_tetype = fp8_dtype_forward
                    proj_out_pttype = torch.uint8
                    ub_obj_projout.set_ubuf_scale_inv(meta_tensor.scale_inv[proj_out_index])
            else:
                dim_size = list(inputmat_total.size())
                dim_size[1] = weight.size(0)
                out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat_total.device)

            ub_algo=tex.UbufOverlapAlgo.ATOMIC_GEMM_RS if ub_atomic_gemm_rs else None
            ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS if ub_split_rs else ub_algo
            _ = fp8_gemm(
                weight_fp8._data,
                fp8_meta["scaling_fwd"].scale_inv,
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                fp8_dtype_forward,
                inputmat_total,
                fp8_meta["scaling_fwd"].scale_inv,
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                proj_out_pttype,
                get_workspace(),
                bias=bias,
                use_bias=use_bias,
                use_split_accumulator=_2X_ACC_FPROP,
                out=out,
                ub_algo=ub_algo,
                ub=ub_obj_projout if (ub_split_rs or ub_atomic_gemm_rs) else None,
                extra_output_tensor=rs_out if (ub_split_rs or ub_atomic_gemm_rs) else None,
                out_index=proj_out_index,
                fp8_meta_tensor = meta_tensor,
                D_dtype = proj_out_tetype,
            )
        else:
            # Cast for native AMP
            weight = cast_if_needed(weight, activation_dtype)
            bias = cast_if_needed(bias, activation_dtype) if use_bias else bias

            if fp8_calibration:
                # amax of input
                amin, amax = inputmat_total.aminmax()
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM1_INPUT] = \
                    torch.max(-amin, amax).float()
                # amax of weight
                amin, amax = weight.aminmax()
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM1_WEIGHT] = \
                    torch.max(-amin, amax).float()

            if ub_split_rs:
                ub_obj_projout = get_ub("proj_fprop")
                out = ub_obj_projout.get_ubuf_output(1)
                dim_size = list(inputmat_total.size())
                dim_size[0] = dim_size[0] // tp_world_size
                dim_size[1] = weight.size(0)
                rs_out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat_total.device)
            else:
                dim_size = list(inputmat_total.size())
                dim_size[1] = weight.size(0)
                out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat_total.device)

            _ = gemm(
                weight,
                inputmat_total,
                activation_dtype,
                get_workspace(),
                bias=bias,
                use_bias=use_bias,
                out=out,
                ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS if ub_split_rs else None,
                ub=ub_obj_projout if ub_split_rs else None,
                extra_output_tensor=rs_out if ub_split_rs else None,
            )

        if is_grad_enabled:
            saved_inputmat = None
            saved_inputmat_t = None
            if weight.requires_grad:
                if fp8 and not fp8_meta["recipe"].override_linear_precision.wgrad:
                    if inputmat_t is None:
                        saved_inputmat = inputmat
                    else:
                        saved_inputmat_t = inputmat_t
                        if cpu_offloading:
                            saved_inputmat_t.activation_offloading = True
                else:
                    saved_inputmat = inputmat_no_fp8

                if cpu_offloading:
                    if fuse_wgrad_accumulation:
                        weight.main_grad.weight_offloading = True
                    if fp8 and weight_t_fp8 is not None:
                        weight_t_fp8.weight_offloading = True
                    weight.weight_offloading = True

                    if saved_inputmat is not None:
                        saved_inputmat.activation_offloading = True

            # Scatter intermediate/activation tensors saved for the backward pass
            ctx.fsdp_group = fsdp_group
            ctx.fsdp_shapes = _fsdp_scatter_tensors(
                fsdp_group,
                saved_inputmat,     # None if fp8 == False
                saved_inputmat_t,   # None if fp8 == False AND not is_grad_enabled
                weight_t_fp8 if fp8 else None,
            )

            ctx.save_for_backward(
                saved_inputmat,
                saved_inputmat_t,
                weight,
                weight.main_grad if cpu_offloading and fuse_wgrad_accumulation else None,
                weight_t_fp8 if fp8 else None,
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
            ctx.ub_split_ag = ub_split_ag
            ctx.ub_atomic_gemm_ag = ub_atomic_gemm_ag
            ctx.ub_name = ub_name
            ctx.tp_size = tp_size
            ctx.requires_dgrad = inp.requires_grad

        # Row Parallel Linear
        if ub_split_rs or ub_atomic_gemm_rs:
            out = rs_out
        elif parallel_mode == "row" and sequence_parallel:
            out, _ = reduce_scatter_along_first_dim(out, tp_group)
        elif parallel_mode == "row" and tensor_parallel:
            out, _ = allreduce(out, tp_group)

        # Scatter Fp8 weight buffers
        _fsdp_scatter_tensors(fsdp_group, weight_fp8)

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])


    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        with _prepare_backward(
            ctx.fp8, ctx.fp8_meta, ctx.tp_group, ctx.tp_size, name="_Linear"
        ):
            (
                inputmat,
                inputmat_t,
                weight,
                main_grad,
                weight_t_fp8,
                fwd_scale_inverses,
            ) = ctx.saved_tensors

            # Gather intermediate/activation tensors if needed
            _fsdp_gather_tensors(ctx.fsdp_group,
                                 ctx.fsdp_shapes,
                                 inputmat,
                                 inputmat_t,
                                 weight_t_fp8)

            if ctx.cpu_offloading and ctx.fuse_wgrad_accumulation:
                weight = torch.nn.Parameter(weight, False)
                weight.main_grad = main_grad

            # Primary weights are in FP8.
            if ctx.fp8 and weight_t_fp8 is None:
                weight_t_fp8 = weight.transpose(
                    update_cache="reuse_only" if ctx.is_first_microbatch is None else "lazy",
                )

            if ctx.ub_split_ag or ctx.ub_atomic_gemm_ag:
                tp_world_size = get_distributed_world_size(ctx.tp_group)
                if tp_world_size == 1:
                    ctx.ub_split_ag = False
                    ctx.ub_atomic_gemm_ag = False
            if ctx.ub_split_ag or ctx.ub_atomic_gemm_ag:
                dim_size = list(grad_output.size())
                dim_size[0] = dim_size[0] * tp_world_size
                ctx.ub_obj_gradout = get_ub(ctx.ub_name+"_dgrad")
            (
                grad_output,
                grad_output_c,
                grad_output_t,
                grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx, grad_output, ctx.parallel_mode == "row"
            )

            # Column Parallel Linear
            # Overlap input AG with dgrad
            inputmat_total = None
            inputmat_t_total = None
            handle = None
            if weight.requires_grad and ctx.parallel_mode == "column" and ctx.sequence_parallel:
                inputmat_total, handle = gather_along_first_dim(
                    inputmat, ctx.tp_group, async_op=ctx.requires_dgrad
                )
            else:
                inputmat_total = inputmat
                inputmat_t_total = inputmat_t

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

            ub_algo = tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG if ctx.ub_split_ag else None
            ub_algo = tex.UbufOverlapAlgo.ATOMIC_GEMM_AG if ctx.ub_atomic_gemm_ag else ub_algo
            if ctx.requires_dgrad:
                if ctx.fp8:
                    dgrad, _ = fp8_gemm(
                        weight_t_fp8._data,
                        fwd_scale_inverses,
                        tex.FP8FwdTensors.GEMM1_WEIGHT,
                        fp8_dtype_forward,
                        grad_output_c,
                        ctx.fp8_meta["scaling_bwd"].scale_inv,
                        tex.FP8BwdTensors.GRAD_OUTPUT1,
                        fp8_dtype_backward,
                        ctx.activation_dtype,
                        get_workspace(),
                        use_split_accumulator=_2X_ACC_DGRAD,
                        ub_algo=ub_algo,
                        ub=ctx.ub_obj_gradout if ctx.ub_split_ag or ctx.ub_atomic_gemm_ag else None,
                    )
                else:
                    dgrad, _, _ = gemm(
                        weight,
                        grad_output,
                        ctx.activation_dtype,
                        get_workspace(),
                        layout="NN",
                        grad=True,
                        ub_algo=tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG if ctx.ub_split_ag else None,
                        ub=ctx.ub_obj_gradout if ctx.ub_split_ag else None,
                    )

                # Overlap dgrad-RS/AR with wgrad
                if ctx.parallel_mode == "column" and ctx.sequence_parallel:
                    if handle is not None:
                        handle.wait()
                    dgrad, handle = reduce_scatter_along_first_dim(
                        dgrad, ctx.tp_group, async_op=True
                    )
                elif ctx.parallel_mode == "column" and ctx.tensor_parallel:
                    dgrad, handle = allreduce(dgrad, ctx.tp_group, async_op=True)

            if weight.requires_grad:
                if ctx.fp8:
                    # WGRAD
                    if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                        if ctx.ub_split_ag or ctx.ub_atomic_gemm_ag:
                            grad_output_t = tex.fp8_transpose(grad_output_c, fp8_dtype_backward)
                        if inputmat_t_total is None:
                            inputmat_t_total = tex.fp8_transpose(inputmat_total, fp8_dtype_backward)
                        wgrad, _ = fp8_gemm(
                            inputmat_t_total,
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
                        )
                    else:
                        wgrad, _, _ = gemm(
                            inputmat_total,
                            grad_output,
                            ctx.activation_dtype,
                            get_workspace(),
                            layout="NT",
                            grad=True,
                            accumulate=accumulate_wgrad_into_param_main_grad,
                            out=weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                        )
                else:
                    # WGRAD
                    wgrad, grad_bias, _ = gemm(
                        inputmat_total,
                        grad_output,
                        ctx.activation_dtype,
                        get_workspace(),
                        layout="NT",
                        grad=True,
                        use_bias=ctx.use_bias,
                        accumulate=accumulate_wgrad_into_param_main_grad,
                        out=weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                    )

                # Deallocate input tensor
                clear_tensor_data(inputmat_total)
                clear_tensor_data(inputmat_t_total)

            # Column Parallel Linear
            if ctx.parallel_mode == "column" and ctx.tensor_parallel and handle is not None:
                handle.wait()

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
            wgrad,
            None,
            None,
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
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
        )


class Linear(TransformerEngineBaseModule):
    """Applies a linear transformation to the incoming data :math:`y = xA^T + b`

    On NVIDIA GPUs it is a drop-in replacement for `torch.nn.Linear`.

    Parameters
    ----------
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    bias : bool, default = `True`
          if set to `False`, the layer will not learn an additive bias.
    init_method : Callable, default = `None`
                 used for initializing weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    parameters_split : Optional[Union[Tuple[str, ...], Dict[str, int]]], default = None
                      Configuration for splitting the weight and bias tensors along dim 0 into
                      multiple PyTorch parameters. If a list or tuple of strings is provided,
                      they are used to make the names of equally-sized parameters. If a dict
                      (preferably an OrderedDict) is provided, the keys are used as names and
                      values as split sizes along dim 0. The resulting parameters will have
                      names that end in `_weight` or `_bias`, so trailing underscores are
                      stripped from any provided names.
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
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        parameters_split: Optional[Union[Tuple[str, ...], Dict[str, int]]] = None,
        device: Union[torch.device, str] = "cuda",
        ub_split_rs: bool = False,
        ub_split_ag: bool = False,
        ub_atomic_gemm_rs: bool = False,
        ub_atomic_gemm_ag: bool = False,
        ub_name: Optional[str] = None,
    ) -> None:
        super().__init__()

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.primary_weights_in_fp8 = FP8GlobalStateManager.with_fp8_parameters()
        self.ub_split_rs = ub_split_rs
        self.ub_split_ag = ub_split_ag
        self.ub_atomic_gemm_rs = ub_atomic_gemm_rs
        self.ub_atomic_gemm_ag = ub_atomic_gemm_ag
        if any([ub_atomic_gemm_rs, ub_atomic_gemm_ag]):
            assert ub_name is not None, "Userbuffer name [string] is not set."
        self.ub_name = ub_name
        self.get_rng_state_tracker = get_rng_state_tracker
        if device == 'meta':
            assert parameters_split is None, ("Cannot split module parameters "
                                              "on 'meta' device.")

        if ub_split_rs or ub_split_ag or ub_atomic_gemm_rs:
            assert (
                tex.userbuf_comm_available()
            ), "Userbuffer communication backend not available."

        if ub_atomic_gemm_rs or ub_atomic_gemm_ag:
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

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        self.weight_tensor = torch.empty(
            self.out_features, self.in_features,
            device=device, dtype=params_dtype)

        if self.use_bias:
            self.bias_tensor = torch.empty(self.out_features, device=device, dtype=params_dtype)
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

    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[Float8Tensor]:
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
                linear_fn = _Linear.apply
                args = []
            else:
                linear_fn = _Linear.forward
                args = [None]
            args += (
                weight_tensor,
                weight1_fp8,
                weight1_t_fp8,
                inp,
                bias_tensor,
                self.apply_bias and not self.gemm_bias_unfused_add,
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
                torch.is_grad_enabled(),
                self.primary_weights_in_fp8,
                self.ub_split_rs,
                self.ub_split_ag,
                self.ub_atomic_gemm_rs,
                self.ub_atomic_gemm_ag,
                self.ub_name,
                self.fsdp_group,
            )
            out = linear_fn(*args)

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        return out

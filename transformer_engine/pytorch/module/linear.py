# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
import sys
from functools import reduce
from operator import mul as multiply_op
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

import transformer_engine_torch as tex

from .base import (
    get_workspace,
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
    requires_grad,
)
from ..distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    allreduce,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
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
from ..constants import GemmParallelModes, dist_group_type, TE_DType
from ..jit import no_torch_dynamo
from ..graph import is_graph_capturing
from ..float8_tensor import Float8Tensor
from ..export import is_in_onnx_export_mode
from ..tensor import QuantizedTensor
from ..cpu_offload import is_cpu_offload_enabled

__all__ = ["Linear"]


class _Linear(torch.autograd.Function):
    """Linear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        weight: Union[Float8Tensor, torch.Tensor],
        weight_fp8: Optional[Float8Tensor],
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
        ub_overlap_rs_fprop: bool,
        ub_overlap_ag_dgrad: bool,
        ub_overlap_ag_fprop: bool,
        ub_overlap_rs_dgrad: bool,
        ub_bulk_dgrad: bool,
        ub_bulk_wgrad: bool,
        ub_name: str,
        fp8_output: bool,
        fsdp_group: Union[dist_group_type, None],
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        is_input_fp8 = isinstance(inp, Float8Tensor)

        # Make sure input dimensions are compatible
        out_features, in_features = weight.shape
        inp_shape = inp.shape
        assert inp_shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view(-1, in_features)
        if fp8:
            assert_dim_for_fp8_exec(inputmat)
            assert_dim_for_fp8_exec(weight)

        tp_world_size = get_distributed_world_size(tp_group)
        ub_overlap_ag_fprop = False if tp_world_size == 1 else ub_overlap_ag_fprop
        ub_overlap_rs_fprop = False if tp_world_size == 1 else ub_overlap_rs_fprop

        # Cast input to expected dtype
        inputmat = cast_if_needed(inputmat, activation_dtype)
        inputmat_t = None
        inputmat_no_fp8 = inputmat
        inputmat_scale_inv = None

        if fp8:
            fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
            if isinstance(inputmat, Float8Tensor):
                inputmat_scale_inv = inputmat._scale_inv
            else:
                inputmat_scale_inv = torch.empty([1], dtype=torch.float32, device=inputmat.device)
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
                        scale_inv=inputmat_scale_inv,
                    )
                else:
                    # FP8 input for forward
                    inputmat = cast_to_fp8(
                        inputmat,
                        fp8_meta["scaling_fwd"],
                        tex.FP8FwdTensors.GEMM1_INPUT,
                        fp8_dtype_forward,
                        scale_inv=inputmat_scale_inv,
                    )

            # Hack for ONNX export
            # Note: ONNX models are represented as a graph of tensor
            # operations, so the in-place scale-inv update doesn't fit
            # very well. We work around this by making it look like
            # the scale-inv tensor is initialized with a copy.
            # Note: ONNX export expects FP8 scales can be represented
            # with constant ops. However, copying into a buffer
            # involves an expand op for array broadcasting. We work
            # around this by filling the buffer instead.
            if is_in_onnx_export_mode():
                inputmat_scale_inv.fill_(inputmat_scale_inv.item())

        # Column Parallel Linear
        if parallel_mode == "column" and sequence_parallel and not ub_overlap_ag_fprop:
            inputmat_total, _ = gather_along_first_dim(inputmat, tp_group)
        else:
            inputmat_total = inputmat

        if fp8:
            bias_dtype = torch.bfloat16 if activation_dtype == torch.float32 else activation_dtype
            bias = cast_if_needed(bias, bias_dtype) if use_bias else bias

            # Use FP8 weights
            if weight_fp8 is None:
                weight_fp8 = weight

            assert isinstance(weight_fp8, Float8Tensor)

            if fp8_output:
                out_index, meta_tensor, out_tedtype, out_pttype = (
                    tex.FP8FwdTensors.GEMM1_OUTPUT,
                    fp8_meta["scaling_fwd"],
                    fp8_dtype_forward,
                    torch.uint8,
                )
            else:
                out_index, meta_tensor, out_tedtype, out_pttype = (
                    None,
                    None,
                    None,
                    activation_dtype,
                )

            ub_obj = None
            ub_algo = None
            rs_out = None
            inputmat_data = (
                inputmat_total._data if isinstance(inputmat_total, Float8Tensor) else inputmat_total
            )
            if ub_overlap_rs_fprop:
                ub_obj = get_ub(ub_name + "_fprop")
                out = ub_obj.get_ubuf_output(1)
                dim_size = list(inputmat_total.size())
                dim_size[0] = dim_size[0] // tp_world_size
                dim_size[1] = out_features
                rs_out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat_total.device)
                if ub_obj.is_p2p_overlap():
                    if ub_obj.is_atomic_gemm():
                        ub_algo = tex.CommOverlapAlgo.ATOMIC_GEMM_RS_P2P
                    else:
                        ub_algo = tex.CommOverlapAlgo.SPLIT_PIPELINED_RS_P2P
                else:
                    if ub_obj.is_atomic_gemm():
                        ub_algo = tex.CommOverlapAlgo.ATOMIC_GEMM_RS
                    else:
                        ub_algo = tex.CommOverlapAlgo.SPLIT_PIPELINED_RS
                if ub_obj.is_fp8_ubuf():
                    out_index = tex.FP8FwdTensors.GEMM1_OUTPUT
                    meta_tensor = fp8_meta["scaling_fwd"]
                    out_tedtype = fp8_dtype_forward
                    out_pttype = torch.uint8
                    ub_obj.set_ubuf_scale_inv(meta_tensor.scale_inv[out_index])

            elif ub_overlap_ag_fprop:
                ub_obj = get_ub(ub_name + "_fprop")
                assert ub_obj.is_fp8_ubuf(), "AG overlap with FP8 GEMM requires FP8 buffer."
                ub_obj.copy_input_to_ubuf(inputmat_data, True)
                ub_obj.set_ubuf_scale_inv(inputmat_scale_inv)
                if ub_obj.is_atomic_gemm():
                    ub_algo = tex.CommOverlapAlgo.ATOMIC_GEMM_AG_P2P
                else:
                    ub_algo = tex.CommOverlapAlgo.SPLIT_PIPELINED_AG_P2P
                out_tedtype = TE_DType[activation_dtype]
                out_pttype = activation_dtype
                dim_size = list(inputmat_total.size())
                dim_size[0] *= tp_size
                dim_size[1] = out_features
                out = torch.empty(dim_size, dtype=out_pttype, device=inputmat_total.device)

            else:
                dim_size = list(inputmat_total.size())
                dim_size[1] = out_features
                out = torch.empty(dim_size, dtype=out_pttype, device=inputmat_total.device)

            _ = fp8_gemm(
                weight_fp8._data,
                weight_fp8._scale_inv,
                0,
                weight_fp8._fp8_dtype,
                inputmat_data,
                inputmat_scale_inv,
                0,
                fp8_dtype_forward,
                out_pttype,
                get_workspace(),
                bias=bias,
                use_bias=use_bias,
                use_split_accumulator=_2X_ACC_FPROP,
                out=out,
                ub_algo=ub_algo,
                ub=ub_obj,
                extra_output_tensor=rs_out,
                out_index=out_index,
                fp8_meta_tensor=meta_tensor,
                D_dtype=out_tedtype,
            )
            if fp8_output:
                out = Float8Tensor(
                    data=out,
                    fp8_meta=fp8_meta,
                    fp8_meta_forward=True,
                    fp8_meta_index=tex.FP8FwdTensors.GEMM1_OUTPUT,
                    fp8_dtype=fp8_dtype_forward,
                    dtype=activation_dtype,
                )
        else:
            # Cast for native AMP
            weight = cast_if_needed(weight, activation_dtype)
            bias = cast_if_needed(bias, activation_dtype) if use_bias else bias

            if fp8_calibration:
                # amax of input
                amin, amax = inputmat_total.aminmax()
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM1_INPUT] = torch.max(
                    -amin, amax
                ).float()
                # amax of weight
                amin, amax = weight.aminmax()
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM1_WEIGHT] = torch.max(
                    -amin, amax
                ).float()

            ub_obj = None
            ub_algo = None
            rs_out = None
            if ub_overlap_rs_fprop:
                ub_obj = get_ub(ub_name + "_fprop")
                out = ub_obj.get_ubuf_output(1)
                dim_size = list(inputmat_total.size())
                dim_size[0] = dim_size[0] // tp_world_size
                dim_size[1] = out_features
                rs_out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat_total.device)
                if ub_obj.is_p2p_overlap():
                    ub_algo = tex.CommOverlapAlgo.SPLIT_PIPELINED_RS_P2P
                else:
                    ub_algo = tex.CommOverlapAlgo.SPLIT_PIPELINED_RS

            elif ub_overlap_ag_fprop:
                ub_obj = get_ub(ub_name + "_fprop")
                ub_obj.copy_input_to_ubuf(inputmat_total, True)
                dim_size = list(inputmat_total.size())
                dim_size[0] *= tp_size  # all-gathered sequence length
                dim_size[1] = out_features
                out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat_total.device)
                ub_algo = tex.CommOverlapAlgo.SPLIT_PIPELINED_AG_P2P

            else:
                dim_size = list(inputmat_total.size())
                dim_size[1] = out_features
                out = torch.empty(dim_size, dtype=activation_dtype, device=inputmat_total.device)

            _ = gemm(
                weight,
                inputmat_total,
                activation_dtype,
                get_workspace(),
                bias=bias,
                use_bias=use_bias,
                out=out,
                ub_algo=ub_algo,
                ub=ub_obj,
                extra_output_tensor=rs_out,
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
                    if fp8 and weight_fp8 is not None:
                        weight_fp8.weight_offloading = True
                    weight.weight_offloading = True

                    if saved_inputmat is not None:
                        saved_inputmat.activation_offloading = True

            # Scatter intermediate/activation tensors saved for the backward pass
            # NOTE: FSDP sharding is not valid for models initialized with primary Fp8 weights
            ctx.fsdp_group = fsdp_group
            ctx.fsdp_shapes = _fsdp_scatter_tensors(
                fsdp_group,
                saved_inputmat,  # None if fp8 == False
                saved_inputmat_t,  # None if fp8 == False AND not is_grad_enabled
                weight_fp8 if fp8 and not isinstance(weight, Float8Tensor) else None,
            )

            ctx.save_for_backward(
                saved_inputmat,
                saved_inputmat_t,
                inputmat_scale_inv,
                weight,
                weight_fp8,
                weight.main_grad if cpu_offloading and fuse_wgrad_accumulation else None,
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
            ctx.inp_shape = inp_shape
            ctx.parallel_mode = parallel_mode
            ctx.tp_group = tp_group
            ctx.ub_overlap_ag = ub_overlap_ag_dgrad
            ctx.ub_overlap_rs_dgrad = ub_overlap_rs_dgrad
            ctx.ub_bulk_dgrad = ub_bulk_dgrad
            ctx.ub_bulk_wgrad = ub_bulk_wgrad
            ctx.ub_name = ub_name
            ctx.tp_size = tp_size
            ctx.requires_dgrad = inp.requires_grad
            ctx.is_input_fp8 = is_input_fp8
            ctx.reduce_and_update_bwd_fp8_tensors = False
            if ctx.fp8 and requires_grad(inp, weight, bias):
                _first_fp8_module = FP8GlobalStateManager.IS_FIRST_FP8_MODULE
                ctx.reduce_and_update_bwd_fp8_tensors = FP8GlobalStateManager.is_first_fp8_module()
                if in_fp8_activation_recompute_phase():
                    FP8GlobalStateManager.IS_FIRST_FP8_MODULE = _first_fp8_module

        # Row Parallel Linear
        if parallel_mode == "row":
            if ub_overlap_rs_fprop:
                out = rs_out
            elif sequence_parallel:
                out, _ = reduce_scatter_along_first_dim(out, tp_group)
            elif tensor_parallel:
                out, _ = allreduce(out, tp_group)

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        return out.view(-1, *inp_shape[1:-1], out_features)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring
        if isinstance(grad_output, Float8Tensor):
            ctx.fp8_meta["scaling_bwd"].scale_inv[
                tex.FP8BwdTensors.GRAD_OUTPUT1
            ] = grad_output._scale_inv

        with torch.cuda.nvtx.range("_Linear_backward"):
            (
                inputmat,
                inputmat_t,
                inputmat_scale_inv,
                weight,
                weight_fp8,
                main_grad,
            ) = ctx.saved_tensors

            # Gather intermediate/activation tensors if needed
            # NOTE: weight_fp8 = weight when ctx.fp8 == False and torch.disttributed.FSDP already
            #       shards/unshards the base weights so we don't do it ourselves
            _fsdp_gather_tensors(
                ctx.fsdp_group,
                ctx.fsdp_shapes,
                inputmat,
                inputmat_t,
                weight_fp8 if ctx.fp8 and not isinstance(weight, Float8Tensor) else None,
            )

            if ctx.cpu_offloading and ctx.fuse_wgrad_accumulation:
                weight = torch.nn.Parameter(weight, weight.requires_grad)
                weight.main_grad = main_grad

            tp_world_size = get_distributed_world_size(ctx.tp_group)
            ctx.ub_overlap_ag = False if tp_world_size == 1 else ctx.ub_overlap_ag
            ctx.ub_overlap_rs_dgrad = False if tp_world_size == 1 else ctx.ub_overlap_rs_dgrad
            ctx.ub_bulk_dgrad = False if tp_world_size == 1 else ctx.ub_bulk_dgrad
            ctx.ub_bulk_wgrad = False if tp_world_size == 1 else ctx.ub_bulk_wgrad

            ctx.ub_obj_gradout = None
            ub_obj_wgrad = None
            ub_algo_wgrad = None
            ub_algo_dgrad = None
            rs_out = None
            dgrad = None
            dgrad_shape = [reduce(multiply_op, ctx.inp_shape[:-1]), ctx.inp_shape[-1]]
            if ctx.ub_overlap_ag:
                # Overlap grad_output all-gather with dgrad compute
                ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad")
                if ctx.ub_obj_gradout.is_atomic_gemm():
                    ub_algo_dgrad = tex.CommOverlapAlgo.ATOMIC_GEMM_AG_P2P
                else:
                    ub_algo_dgrad = tex.CommOverlapAlgo.SPLIT_PIPELINED_AG_P2P
                dgrad = torch.empty(
                    dgrad_shape, dtype=ctx.activation_dtype, device=grad_output.device
                )

            elif ctx.ub_overlap_rs_dgrad:
                # Overlap dgrad reduce-scatter with dgrad compute
                ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad")
                dgrad = ctx.ub_obj_gradout.get_ubuf_output(1)
                if ctx.ub_obj_gradout.is_p2p_overlap():
                    if ctx.ub_obj_gradout.is_atomic_gemm():
                        ub_algo_dgrad = tex.CommOverlapAlgo.ATOMIC_GEMM_RS_P2P
                    else:
                        ub_algo_dgrad = tex.CommOverlapAlgo.SPLIT_PIPELINED_RS_P2P
                else:
                    if ctx.ub_obj_gradout.is_atomic_gemm():
                        ub_algo_dgrad = tex.CommOverlapAlgo.ATOMIC_GEMM_RS
                    else:
                        ub_algo_dgrad = tex.CommOverlapAlgo.SPLIT_PIPELINED_RS
                rs_out = torch.empty(
                    dgrad_shape, dtype=ctx.activation_dtype, device=grad_output.device
                )
                ctx.ub_bulk_dgrad = False
                ctx.ub_bulk_wgrad = False

            else:
                if ctx.ub_bulk_dgrad:
                    # Overlap inputmat all-gather with dgrad compute
                    ub_algo_dgrad = tex.CommOverlapAlgo.BULK_OVERLAP_AG
                    ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad")
                    inputmat_data = (
                        inputmat._data if isinstance(inputmat, Float8Tensor) else inputmat
                    )
                    ctx.ub_obj_gradout.copy_input_to_ubuf(inputmat_data, True)
                    inputmat_ubuf = ctx.ub_obj_gradout.get_ubuf_output(1)
                    if isinstance(inputmat, Float8Tensor):
                        inputmat._data = inputmat_ubuf
                    else:
                        inputmat = inputmat_ubuf

                if ctx.ub_bulk_wgrad:
                    # Overlap dgrad reduce-scatter with wgrad compute
                    ub_algo_wgrad = tex.CommOverlapAlgo.BULK_OVERLAP_RS
                    ub_obj_wgrad = get_ub(ctx.ub_name + "_wgrad")
                    dgrad = ub_obj_wgrad.get_ubuf_output(1)

            if dgrad is None:
                if ctx.parallel_mode == "column" and ctx.sequence_parallel:
                    dgrad_shape[0] = dgrad_shape[0] * tp_world_size
                dgrad = torch.empty(
                    dgrad_shape, dtype=ctx.activation_dtype, device=grad_output.device
                )

            (
                grad_output,
                grad_output_c,
                grad_output_t,
                grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx, grad_output, ctx.parallel_mode == "row"
            )

            # Overlap inputmat AG with dgrad via NCCL async comms (no TP overlap via Userbuffers)
            inputmat_total = None
            inputmat_t_total = None
            inputmat_gather_handle = None
            if (
                weight.requires_grad
                and ctx.parallel_mode == "column"
                and ctx.sequence_parallel
                and not ctx.ub_bulk_dgrad
            ):
                inputmat_total, inputmat_gather_handle = gather_along_first_dim(
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
                fp8_dtype_forward = get_fp8_te_dtype(ctx.fp8_meta["recipe"], fprop_tensor=True)
                fp8_dtype_backward = get_fp8_te_dtype(ctx.fp8_meta["recipe"], fprop_tensor=False)

            if ctx.requires_dgrad:
                if ctx.fp8:
                    if ctx.is_input_fp8 or (
                        ctx.ub_overlap_rs_dgrad and ctx.ub_obj_gradout.is_fp8_ubuf()
                    ):
                        out_index, meta_tensor, output_te_dtype, output_dtype = (
                            tex.FP8BwdTensors.GRAD_INPUT1,
                            ctx.fp8_meta["scaling_bwd"],
                            fp8_dtype_backward,
                            torch.uint8,
                        )
                        if ctx.ub_overlap_rs_dgrad and ctx.ub_obj_gradout.is_fp8_ubuf():
                            ctx.ub_obj_gradout.set_ubuf_scale_inv(meta_tensor.scale_inv[out_index])
                    else:
                        out_index, meta_tensor, output_te_dtype, output_dtype = (
                            None,
                            None,
                            None,
                            ctx.activation_dtype,
                        )
                    _ = fp8_gemm(
                        weight_fp8.transpose_2d(),
                        weight_fp8._scale_inv,
                        0,
                        weight_fp8._fp8_dtype,
                        grad_output_c,
                        ctx.fp8_meta["scaling_bwd"].scale_inv,
                        tex.FP8BwdTensors.GRAD_OUTPUT1,
                        fp8_dtype_backward,
                        output_dtype,
                        get_workspace(),
                        use_split_accumulator=_2X_ACC_DGRAD,
                        ub_algo=ub_algo_dgrad,
                        ub=ctx.ub_obj_gradout,
                        out=dgrad,
                        out_index=out_index,
                        fp8_meta_tensor=meta_tensor,
                        D_dtype=output_te_dtype,
                        extra_output_tensor=rs_out,
                    )

                    if ctx.ub_overlap_rs_dgrad:
                        dgrad = rs_out

                    if output_dtype == torch.uint8:
                        dgrad = Float8Tensor(
                            data=dgrad,
                            fp8_meta=ctx.fp8_meta,
                            fp8_meta_forward=False,
                            fp8_meta_index=tex.FP8BwdTensors.GRAD_INPUT1,
                            fp8_dtype=fp8_dtype_backward,
                            dtype=ctx.activation_dtype,
                        )
                else:
                    _ = gemm(
                        weight,
                        grad_output,
                        ctx.activation_dtype,
                        get_workspace(),
                        layout="NN",
                        grad=True,
                        ub_algo=ub_algo_dgrad,
                        ub=ctx.ub_obj_gradout,
                        out=dgrad,
                        extra_output_tensor=rs_out,
                    )

                    if ctx.ub_overlap_rs_dgrad:
                        dgrad = rs_out

            if inputmat_gather_handle is not None:
                inputmat_gather_handle.wait()

            # Overlap dgrad RS/AR with wgrad via NCCL async comms (no TP overlap via Userbuffers)
            dgrad_reduce_handle = None
            if ctx.requires_dgrad and ctx.parallel_mode == "column":
                if ctx.sequence_parallel and not (ctx.ub_overlap_rs_dgrad or ctx.ub_bulk_wgrad):
                    dgrad, dgrad_reduce_handle = reduce_scatter_along_first_dim(
                        dgrad, ctx.tp_group, async_op=True
                    )
                elif ctx.tensor_parallel and not ctx.sequence_parallel:
                    dgrad, dgrad_reduce_handle = allreduce(dgrad, ctx.tp_group, async_op=True)

            wgrad = None
            if weight.requires_grad:
                if ctx.fp8:
                    # WGRAD
                    if not ctx.fp8_meta["recipe"].override_linear_precision.wgrad:
                        if ctx.ub_overlap_ag:
                            if isinstance(grad_output_c, Float8Tensor):
                                grad_output_t = grad_output_c.transpose_2d()
                            else:
                                grad_output_t = tex.fp8_transpose(grad_output_c, fp8_dtype_backward)
                        if inputmat_t_total is None:
                            if isinstance(inputmat_total, Float8Tensor):
                                inputmat_t_total = inputmat_total.transpose_2d()
                            else:
                                inputmat_t_total = tex.fp8_transpose(
                                    inputmat_total, fp8_dtype_backward
                                )
                        wgrad, _ = fp8_gemm(
                            (
                                inputmat_t_total._data
                                if isinstance(inputmat_t_total, Float8Tensor)
                                else inputmat_t_total
                            ),
                            inputmat_scale_inv,
                            0,
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
                            ub=ub_obj_wgrad,
                            ub_algo=ub_algo_wgrad,
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
                            ub=ub_obj_wgrad,
                            ub_algo=ub_algo_wgrad,
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
                        ub=ub_obj_wgrad,
                        ub_algo=ub_algo_wgrad,
                    )

                if ctx.ub_bulk_wgrad:
                    dgrad = ub_obj_wgrad.get_ubuf_output(0)

                # Deallocate input tensor
                clear_tensor_data(inputmat_total)
                clear_tensor_data(inputmat_t_total)

            # Wait for dgrad reduce-scatter or all-reduce
            if dgrad_reduce_handle is not None:
                dgrad_reduce_handle.wait()

            if not ctx.use_bias:
                grad_bias = None

        if weight.requires_grad:
            # Handle custom DDP from mcore.
            if ctx.fuse_wgrad_accumulation and hasattr(weight, "grad_added_to_main_grad"):
                weight.grad_added_to_main_grad = True
                if getattr(weight, "zero_out_wgrad", False):
                    wgrad = torch.zeros(
                        weight.main_grad.shape,
                        dtype=weight.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    wgrad = torch.empty(
                        weight.main_grad.shape,
                        dtype=weight.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
            elif ctx.fuse_wgrad_accumulation:
                wgrad = None
        else:
            wgrad = None

        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        # Scatter fp8 weight buffers
        if ctx.fp8 and not isinstance(weight, Float8Tensor):
            _fsdp_scatter_tensors(ctx.fsdp_group, weight_fp8)

        return (
            wgrad,
            None,  # weight_fp8
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            grad_bias,
            None,  # use_bias
            None,  # is_first_microbatch
            None,  # fp8
            None,  # fp8_calibration
            None,  # fp8_meta
            None,  # fuse_wgrad_accumulation
            None,  # cpu_offloading
            None,  # tp_group
            None,  # tp_size
            None,  # sequence_parallel
            None,  # tensor_parallel
            None,  # activation_dtype
            None,  # parallel_mode
            None,  # is_grad_enabled
            None,  # ub_overlap_rs_fprop
            None,  # ub_overlap_ag_dgrad
            None,  # ub_overlap_ag_fprop
            None,  # ub_overlap_rs_dgrad
            None,  # ub_bulk_dgrad
            None,  # ub_bulk_wgrad
            None,  # ub_name
            None,  # fp8_output
            None,  # fsdp_group
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
    get_rng_state_tracker : Callable, default = `None`
                 used to get the random number generator state tracker for initializing weights.
    rng_tracker_name : str, default = `None`
                 the param passed to get_rng_state_tracker to get the specific rng tracker.
    parameters_split : Optional[Union[Tuple[str, ...], Dict[str, int]]], default = None
                      Configuration for splitting the weight and bias tensors along dim 0 into
                      multiple PyTorch parameters. If a list or tuple of strings is provided,
                      they are used to make the names of equally-sized parameters. If a dict
                      (preferably an OrderedDict) is provided, the keys are used as names and
                      values as split sizes along dim 0. The resulting parameters will have
                      names that end in `_weight` or `_bias`, so trailing underscores are
                      stripped from any provided names.
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will be allocated. It is the user's
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
    parallel_mode : {None, 'column', 'row'}, default = `None`
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
        ub_bulk_dgrad: bool = False,
        ub_bulk_wgrad: bool = False,
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
        self.ub_overlap_ag_fprop = parallel_mode == "column" and sequence_parallel and ub_overlap_ag
        self.ub_overlap_rs_dgrad = parallel_mode == "column" and sequence_parallel and ub_overlap_rs
        self.ub_bulk_dgrad = parallel_mode == "column" and sequence_parallel and ub_bulk_dgrad
        self.ub_bulk_wgrad = parallel_mode == "column" and sequence_parallel and ub_bulk_wgrad
        if self.ub_overlap_rs_dgrad:
            self.ub_bulk_dgrad = False
            self.ub_bulk_wgrad = False

        # Row parallel TP overlap options
        self.ub_overlap_rs_fprop = parallel_mode == "row" and sequence_parallel and ub_overlap_rs
        self.ub_overlap_ag_dgrad = parallel_mode == "row" and sequence_parallel and ub_overlap_ag

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
            assert ub_name is not None, f"Userbuffer name [string] is not set."
        self.ub_name = ub_name

        assert not (self.ub_overlap_rs_fprop and self.ub_overlap_ag_fprop), (
            "Failed to infer TP overlap options! "
            + "Forward pass cannot do AG+GEMM and GEMM+RS at the same time."
        )
        assert not (self.ub_overlap_ag_dgrad and self.ub_overlap_rs_dgrad), (
            "Failed to infer TP overlap options! "
            + "Backward pass cannot do AG+DGRAD and DGRAD+RS at the same time."
        )
        assert not self.ub_overlap_rs_dgrad and (self.ub_bulk_dgrad or self.ub_bulk_wgrad), (
            "Failed to infer TP overlap options! "
            + "Backward pass cannot do DGRAD+RS and bulk overlaps at the same time."
        )

        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name

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
                raise RuntimeError("Splitting Float8Tensor into multiple params is not supported")

            # Construct weight parameter
            self.register_parameter(
                self.weight_names[i],
                torch.nn.Parameter(weight_tensor[split_start:split_end]),
                init_fn=init_method,
                get_rng_state_tracker=get_rng_state_tracker,
                fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT,
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
        if FP8GlobalStateManager.fp8_graph_capturing():
            skip_fp8_weight_update = FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
        else:
            skip_fp8_weight_update = None
        if skip_fp8_weight_update is not None:
            is_first_microbatch = False

        with self.prepare_forward(
            inp,
            is_first_microbatch,
            allow_non_contiguous=isinstance(inp, QuantizedTensor),
        ) as inp:

            # Get concatenated weight and bias tensors
            unfused_weights = [getattr(self, name) for name in self.weight_names]
            if any(isinstance(w, QuantizedTensor) for w in unfused_weights):
                if self.fp8:
                    if len(unfused_weights) != 1:
                        raise RuntimeError(
                            "Splitting QuantizedTensor into multiple params is not supported"
                        )
                else:
                    unfused_weights = [w.dequantize() for w in unfused_weights]
            weight_tensor = _noop_cat(unfused_weights)
            if self.use_bias:
                bias_tensor = _noop_cat([getattr(self, name) for name in self.bias_names])
            else:
                bias_tensor = getattr(self, self.bias_names[0])  # Unused

            # Initialize FP8 weights if needed
            weight_fp8 = None
            if self.fp8:
                if isinstance(weight_tensor, Float8Tensor):
                    # Make sure transpose cache is valid, if present
                    # Note: Transpose cache may have been invalidated
                    # externally, e.g. by optimizer.
                    if weight_tensor._transpose is not None:
                        weight_tensor.transpose_2d(
                            fill_cache=True,
                            noop_flag=skip_fp8_weight_update,
                        )
                else:
                    # FP8 cast to workspace buffer
                    update_workspace = is_first_microbatch is None or is_first_microbatch
                    weight_fp8 = self.get_fp8_workspace(
                        tensor=weight_tensor,
                        fp8_meta_forward=True,
                        fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT,
                        cache_name=(None if is_first_microbatch is None else "weight"),
                        update_workspace=update_workspace,
                        skip_update_flag=skip_fp8_weight_update,
                        fsdp_group=self.fsdp_group,
                    )

            if torch.is_grad_enabled():
                linear_fn = _Linear.apply
                args = []
            else:
                linear_fn = _Linear.forward
                args = [None]
            args += (
                weight_tensor,
                weight_fp8,
                inp,
                bias_tensor,
                self.apply_bias and not self.gemm_bias_unfused_add,
                is_first_microbatch,
                self.fp8,
                self.fp8_calibration,
                self.fp8_meta,
                self.fuse_wgrad_accumulation,
                is_cpu_offload_enabled(),
                self.tp_group,
                self.tp_size,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.parallel_mode,
                torch.is_grad_enabled(),
                self.ub_overlap_rs_fprop,
                self.ub_overlap_ag_dgrad,
                self.ub_overlap_ag_fprop,
                self.ub_overlap_rs_dgrad,
                self.ub_bulk_dgrad,
                self.ub_bulk_wgrad,
                self.ub_name,
                fp8_output,
                self.fsdp_group,
            )
            out = linear_fn(*args)

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        return out

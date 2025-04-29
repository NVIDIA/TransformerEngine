# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
from typing import Callable, Dict, Optional, Tuple, Union
from functools import reduce
from operator import mul as multiply_op
import warnings

import functools
import torch

import transformer_engine_torch as tex

from transformer_engine.common.recipe import Recipe
from transformer_engine.pytorch import torch_version
from .base import (
    get_workspace,
    get_ub,
    TransformerEngineBaseModule,
    get_dummy_wgrad,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ._common import noop_cat, _fix_gathered_fp8_transpose, WeightGradStore
from ..fp8 import FP8GlobalStateManager
from ..utils import (
    cast_if_needed,
    clear_tensor_data,
    divide,
    init_method_constant,
    requires_grad,
    needs_quantized_gemm,
    is_non_tn_fp8_gemm_supported,
    assert_dim_for_fp8_exec,
    nvtx_range_pop,
    nvtx_range_push,
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
from ..constants import GemmParallelModes, dist_group_type
from ..jit import no_torch_dynamo
from ..graph import is_graph_capturing
from ..tensor.quantized_tensor import (
    QuantizedTensor,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)
from ..tensor.float8_tensor import Float8CurrentScalingQuantizer, Float8Quantizer
from ..tensor.mxfp8_tensor import MXFP8Quantizer
from ..tensor._internal.mxfp8_tensor_base import MXFP8TensorBase
from ..tensor.float8_blockwise_tensor import Float8BlockQuantizer
from ..cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...debug.pytorch.debug_state import TEDebugState
from ...debug.pytorch.utils import any_feature_enabled

__all__ = ["Linear"]


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
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        wgrad_store: WeightGradStore,
        input_quantizer: Optional[Quantizer],
        weight_quantizer: Optional[Quantizer],
        output_quantizer: Optional[Quantizer],
        grad_input_quantizer: Optional[Quantizer],
        grad_weight_quantizer: Optional[Quantizer],
        grad_output_quantizer: Optional[Quantizer],
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
        fp8_output: bool,  # pylint: disable=unused-argument
        fsdp_group: Union[dist_group_type, None],
        module: torch.nn.Module,
        skip_fp8_weight_update: bool,
        symmetric_ar_type: str,
        debug: Optional[bool] = False,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring

        # NVTX label for profiling
        nvtx_label = "transformer_engine._Linear.forward"
        if ub_name is not None:
            nvtx_label = f"{nvtx_label}.{ub_name}"

        # Make sure input dimensions are compatible
        out_features, in_features = weight.shape
        assert inp.shape[-1] == in_features, "GEMM not possible"

        tp_world_size = get_distributed_world_size(tp_group)
        backward_needs_input = is_grad_enabled and weight.requires_grad

        # Prepare input tensor
        # Note: Cast to expected dtype and perform tensor-parallel communication
        nvtx_range_push(f"{nvtx_label}.input_cast_comm")

        inputmat = inp
        inputmat_total = None
        with_input_all_gather_nccl = (
            parallel_mode == "column" and sequence_parallel and not ub_overlap_ag_fprop
        )
        own_quantized_input = False
        # TODO(kwyss): Support FP8 allgather for FP8 block quantization.
        force_hp_input_gather = (
            fp8 and with_input_all_gather_nccl and isinstance(input_quantizer, Float8BlockQuantizer)
        )  # Perform TP communication in high precision.
        if fp8:
            assert_dim_for_fp8_exec(inputmat, weight)
            if any([ub_overlap_ag_fprop, ub_overlap_rs_fprop]) and not (
                FP8GlobalStateManager.get_fp8_recipe().float8_per_tensor_scaling()
            ):
                raise NotImplementedError(
                    "Comm+GEMM overlap is only supported with FP8 delayed scaling or per-tensor"
                    " current scaling"
                )
        if fp8 or debug:
            if input_quantizer is None:
                raise ValueError("Missing quantizer for input tensor")
            if with_input_all_gather_nccl:
                if force_hp_input_gather:
                    input_quantizer.set_usage(rowwise=True, columnwise=False)
                    inputmat_total, _ = gather_along_first_dim(
                        inputmat, tp_group, quantizer=input_quantizer
                    )
                else:
                    if not isinstance(inputmat, QuantizedTensor):
                        columnwise_usage = backward_needs_input and isinstance(
                            input_quantizer, MXFP8Quantizer
                        )
                        # force_hp_input_gather should enforce this
                        assert not isinstance(input_quantizer, Float8BlockQuantizer)
                        input_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
                        inputmat = input_quantizer(inputmat)
                        own_quantized_input = True
                    input_quantizer.set_usage(rowwise=True, columnwise=False)
                    inputmat_total, _ = gather_along_first_dim(
                        inputmat,
                        tp_group,
                        quantizer=input_quantizer,
                    )
            else:
                if (
                    FP8GlobalStateManager.get_fp8_recipe().float8_per_tensor_scaling()
                    and ub_bulk_dgrad
                ):
                    # reduce duplicated transpose in `_fix_gathered_fp8_transpose`
                    input_quantizer.set_usage(rowwise=True, columnwise=False)
                else:
                    input_quantizer.set_usage(
                        rowwise=True,
                        columnwise=backward_needs_input,
                    )
                if not isinstance(inputmat, QuantizedTensor):
                    inputmat = input_quantizer(inputmat)
                    own_quantized_input = True
                elif backward_needs_input:
                    inputmat.update_usage(rowwise_usage=True, columnwise_usage=True)
                inputmat_total = inputmat
        else:
            inputmat = cast_if_needed(inp, activation_dtype)
            if with_input_all_gather_nccl:
                inputmat_total, _ = gather_along_first_dim(inputmat, tp_group)
            else:
                inputmat_total = inputmat
        nvtx_range_pop(f"{nvtx_label}.input_cast_comm")

        # Cast weight to expected dtype
        weightmat = weight

        if fp8 or debug:
            # Configure quantizer
            if weight_quantizer is not None:
                columnwise_usage = is_grad_enabled and inp.requires_grad
                if not columnwise_usage:
                    columnwise_usage = (
                        is_fp8_activation_recompute_enabled()
                        and not in_fp8_activation_recompute_phase()
                    )
                weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
            # FP8 cast to workspace buffer
            update_workspace = is_first_microbatch is None or is_first_microbatch
            weightmat = module.get_weight_workspace(
                tensor=weight,
                quantizer=weight_quantizer,
                cache_name=(None if is_first_microbatch is None else "weight"),
                update_workspace=update_workspace,
                skip_update_flag=skip_fp8_weight_update,
                fsdp_group=fsdp_group,
                workspace_dtype=activation_dtype,
            )
        else:
            weightmat = cast_if_needed(weightmat, activation_dtype)

        # Cast bias to expected dtype
        bias_dtype = activation_dtype
        if needs_quantized_gemm(inputmat_total) and activation_dtype == torch.float32:
            bias_dtype = torch.bfloat16
        bias = cast_if_needed(bias, bias_dtype) if bias is not None else bias

        # Configure output quantizer
        if output_quantizer is not None:
            output_quantizer.set_usage(rowwise=True, columnwise=False)

        # Calibrate quantizers if needed
        if not fp8 and fp8_calibration:
            if input_quantizer is not None:
                input_quantizer.calibrate(inputmat_total)
            if weight_quantizer is not None:
                weight_quantizer.calibrate(weight)

        ub_obj = None
        ub_type = None
        rs_out = None
        out_dtype = activation_dtype
        if ub_overlap_rs_fprop:
            ub_obj = get_ub(ub_name + "_fprop")
            ub_type = tex.CommOverlapType.RS
            out_shape = [reduce(multiply_op, inp.shape[:-1]) // tp_world_size, out_features]
            rs_out = torch.empty(out_shape, dtype=activation_dtype, device=inputmat_total.device)

        elif ub_overlap_ag_fprop:
            ub_obj = get_ub(ub_name + "_fprop")
            ub_type = tex.CommOverlapType.AG
            if fp8:
                assert ub_obj.is_fp8_ubuf(), "AG overlap with FP8 GEMM inputs requires FP8 buffer."
            ub_obj.copy_into_buffer(inputmat_total, input_quantizer, local_chunk=True)
            inputmat_total = ub_obj.get_buffer(input_quantizer)

        nvtx_range_push(f"{nvtx_label}.gemm")
        fprop_gemm_use_split_accumulator = _2X_ACC_FPROP
        if fp8:
            recipe = FP8GlobalStateManager.get_fp8_recipe()
            if hasattr(recipe, "fp8_gemm_fprop"):
                fprop_gemm_use_split_accumulator = recipe.fp8_gemm_fprop.use_split_accumulator

        out, *_, rs_out = general_gemm(
            weightmat,
            inputmat_total,
            get_workspace(),
            quantization_params=output_quantizer,
            out_dtype=out_dtype,
            bias=bias,
            use_split_accumulator=fprop_gemm_use_split_accumulator,
            ub=ub_obj,
            ub_type=ub_type,
            extra_output=rs_out,
        )
        nvtx_range_pop(f"{nvtx_label}.gemm")

        if is_grad_enabled:
            ctx.weight_quantizer = weight_quantizer
            saved_inputmat = None

            ctx.backward_input_needs_gather = (
                weight.requires_grad and parallel_mode == "column" and sequence_parallel
            )

            if backward_needs_input:
                if own_quantized_input and isinstance(inputmat, QuantizedTensor):
                    # For sequence parallel in vanilla FP8, rowwise data is
                    # to gather the input. For MXFP8, columnwise only data
                    # can be allgathered.
                    if isinstance(inputmat, MXFP8TensorBase) or not ctx.backward_input_needs_gather:
                        inputmat.update_usage(rowwise_usage=False, columnwise_usage=True)
                if force_hp_input_gather:
                    assert not isinstance(inputmat, QuantizedTensor)
                saved_inputmat = inputmat

            # Weight with column-wise usage is needed for dgrad GEMM.
            if inp.requires_grad:
                if isinstance(weightmat, QuantizedTensor):
                    weightmat.update_usage(columnwise_usage=True)

            if cpu_offloading and saved_inputmat is not None:
                mark_activation_offload(saved_inputmat)

            # Scatter intermediate/activation tensors saved for the backward pass
            # NOTE: FSDP sharding is not valid for models initialized with primary Fp8 weights
            nvtx_range_push(f"{nvtx_label}.fsdp_scatter")
            ctx.fsdp_group = fsdp_group
            ctx.fsdp_shapes = _fsdp_scatter_tensors(
                fsdp_group,
                saved_inputmat,
                weightmat if fp8 and not isinstance(weight, QuantizedTensor) else None,
            )
            nvtx_range_pop(f"{nvtx_label}.fsdp_scatter")

            if cpu_offloading:
                ctx.grad_added_to_main_grad = hasattr(weight, "grad_added_to_main_grad")

                if ctx.grad_added_to_main_grad:
                    # If you are passing torch.nn.Parameter through the Torch hooks, you will
                    # get back torch.Tensor. Torch rips off the Parameter wrapper.
                    # You need to preserve the weight object to have all the attributes user
                    # sets for the weights. Because of this, it is not recommended to offload
                    # weights if weights are externally touched outside this module
                    ctx.weight_object = weight

            # TODO(ksivamani): Check memory usage
            tensors_to_save, tensor_objects = prepare_for_saving(
                saved_inputmat,
                weightmat,
                weight,
                bias,
            )
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects

            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
            ctx.force_hp_input_gather = force_hp_input_gather
            ctx.input_quantizer = input_quantizer
            ctx.grad_input_quantizer = grad_input_quantizer
            ctx.grad_weight_quantizer = grad_weight_quantizer
            ctx.grad_output_quantizer = grad_output_quantizer
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            if fuse_wgrad_accumulation and weight.requires_grad:
                ctx.main_grad = weight.main_grad

            ctx.debug = debug
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = bias is not None
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp.shape
            ctx.parallel_mode = parallel_mode
            ctx.tp_group = tp_group
            ctx.ub_overlap_ag = ub_overlap_ag_dgrad
            ctx.ub_overlap_rs_dgrad = ub_overlap_rs_dgrad
            ctx.ub_bulk_dgrad = ub_bulk_dgrad
            ctx.ub_bulk_wgrad = ub_bulk_wgrad
            ctx.ub_name = ub_name
            ctx.tp_size = tp_size
            ctx.requires_dgrad = inp.requires_grad
            ctx.requires_wgrad = weight.requires_grad
            ctx.reduce_and_update_bwd_fp8_tensors = False

            ctx.owns_input = saved_inputmat is not inp
            if ctx.fp8 and requires_grad(inp, weight, bias):
                _first_fp8_module = FP8GlobalStateManager.IS_FIRST_FP8_MODULE
                ctx.reduce_and_update_bwd_fp8_tensors = FP8GlobalStateManager.is_first_fp8_module()
                if in_fp8_activation_recompute_phase():
                    FP8GlobalStateManager.IS_FIRST_FP8_MODULE = _first_fp8_module
            ctx.wgrad_store = wgrad_store

        # Row Parallel Linear
        if ub_overlap_rs_fprop:
            out = rs_out
        elif parallel_mode == "row":
            nvtx_range_push(f"{nvtx_label}.row_parallel_comm")
            if sequence_parallel:
                out, _ = reduce_scatter_along_first_dim(out, tp_group)
            elif tensor_parallel:
                if symmetric_ar_type is not None:
                    out, _ = symmetric_all_reduce(out, tp_group, all_reduce_type=symmetric_ar_type)
                else:
                    out, _ = allreduce(out, tp_group)
            nvtx_range_pop(f"{nvtx_label}.row_parallel_comm")

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring

        # NVTX label for profiling
        nvtx_label = "transformer_engine._Linear.backward"
        if ctx.ub_name is not None:
            nvtx_label = f"{nvtx_label}.{ctx.ub_name}"

        with torch.cuda.nvtx.range("_Linear_backward"):
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
                and (ctx.fp8_recipe is not None)
            ):
                if not ctx.fp8_recipe.float8_per_tensor_scaling():
                    raise NotImplementedError(
                        "Comm+GEMM overlap is only supported with FP8 delayed scaling or per-tensor"
                        " current scaling"
                    )

            saved_tensors = ctx.saved_tensors
            inputmat, weight_fp8, weight, bias = (  # pylint: disable=unbalanced-tuple-unpacking
                restore_from_saved(ctx.tensor_objects, saved_tensors)
            )
            # Delete the references to tensor objects once they've been consumed
            # by the `restore_from_saved` method to construct back the actual tensors.
            ctx.tensor_objects = None

            # Since main_grad can be modified inplace, it should not be a part of saved_tensors
            main_grad = (
                ctx.main_grad
                if weight is not None and ctx.fuse_wgrad_accumulation and ctx.requires_wgrad
                else None
            )

            if ctx.cpu_offloading:
                if ctx.grad_added_to_main_grad:
                    weight = ctx.weight_object
                if ctx.requires_wgrad and ctx.fuse_wgrad_accumulation:
                    weight.main_grad = main_grad

            # Gather intermediate/activation tensors if needed
            # NOTE: weight_fp8 = weight when ctx.fp8 == False and torch.disttributed.FSDP already
            #       shards/unshards the base weights so we don't do it ourselves
            nvtx_range_push(f"{nvtx_label}.fsdp_gather")
            _fsdp_gather_tensors(
                ctx.fsdp_group,
                ctx.fsdp_shapes,
                inputmat,
                weight_fp8,
            )
            nvtx_range_pop(f"{nvtx_label}.fsdp_gather")

            ctx.ub_obj_gradout = None
            ub_obj_dgrad = None
            ub_obj_wgrad = None
            ub_type_dgrad = None
            ub_type_wgrad = None
            dgrad_shape = [reduce(multiply_op, ctx.inp_shape[:-1]), ctx.inp_shape[-1]]
            rs_out = None
            dgrad_bulk = None
            if ctx.ub_overlap_ag:
                # Overlap grad_output all-gather with dgrad compute
                ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad")
                ub_obj_dgrad = ctx.ub_obj_gradout
                ub_type_dgrad = tex.CommOverlapType.AG

            elif ctx.ub_overlap_rs_dgrad:
                # Overlap dgrad reduce-scatter with dgrad compute
                ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad")
                ub_obj_dgrad = ctx.ub_obj_gradout
                ub_type_dgrad = tex.CommOverlapType.RS
                rs_out = torch.empty(
                    dgrad_shape, dtype=ctx.activation_dtype, device=grad_output.device
                )

            else:
                if ctx.ub_bulk_dgrad:
                    # Overlap inputmat all-gather with dgrad compute
                    # NOTE: Copying into communication buffer will always prefer rowwise data,
                    #       and will copy columnwise data if rowwise does not exist. In that case,
                    #       the all-gather will apply to the leading dimension of the transpose,
                    #       which then needs to be interleaved correctly before WGRAD.
                    ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad")
                    ub_obj_dgrad = ctx.ub_obj_gradout
                    ub_type_dgrad = tex.CommOverlapType.AG
                    ub_obj_dgrad.copy_into_buffer(inputmat, ctx.input_quantizer, local_chunk=True)

                if ctx.ub_bulk_wgrad:
                    # Overlap dgrad reduce-scatter with wgrad compute
                    ub_obj_wgrad = get_ub(ctx.ub_name + "_wgrad")
                    ub_type_wgrad = tex.CommOverlapType.RS
                    ub_obj_wgrad.set_buffer_params(ctx.grad_input_quantizer)
                    dgrad_bulk = ub_obj_wgrad.get_buffer(ctx.grad_input_quantizer)

            # Configure quantizer for grad output tensor
            # Note: dgrad GEMM requires row-wise usage, wgrad GEMM
            # requires column-wise usage
            if ctx.grad_output_quantizer is not None:
                rowwise_usage = True
                columnwise_usage = True
                if ctx.ub_overlap_ag and isinstance(
                    ctx.grad_output_quantizer,
                    (Float8Quantizer, Float8CurrentScalingQuantizer),
                ):
                    # If data is in FP8 and communication is handled
                    # with Userbuffers, we compute FP8 transposes
                    # manually
                    columnwise_usage = False
                ctx.grad_output_quantizer.set_usage(
                    rowwise=rowwise_usage,
                    columnwise=columnwise_usage,
                )

            # Prepare grad output tensor
            # Note: Cast to expected dtype and perform tensor-parallel communication
            nvtx_range_push(f"{nvtx_label}.grad_output_preprocess")
            (
                grad_output,
                grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx,
                grad_output,
                ctx.parallel_mode == "row",
                ctx.grad_output_quantizer,
            )
            nvtx_range_pop(f"{nvtx_label}.grad_output_preprocess")

            # Launch tensor-parallel communication for input tensor
            inputmat_total = None
            inputmat_total_work = None
            if ctx.backward_input_needs_gather and not ctx.ub_bulk_dgrad:
                quantizer = None
                if ctx.fp8 or ctx.debug:
                    quantizer = ctx.input_quantizer
                    if isinstance(quantizer, (Float8Quantizer, Float8CurrentScalingQuantizer)):
                        # If data is in FP8, we compute FP8 transposes manually
                        quantizer.set_usage(rowwise=True, columnwise=False)
                    else:
                        # wgrad GEMM requires input with column-wise usage
                        quantizer.set_usage(rowwise=False, columnwise=True)
                nvtx_range_push(f"{nvtx_label}.column_parallel_comm_input")
                gather_quantizer = None if ctx.force_hp_input_gather else quantizer
                inputmat_total, inputmat_total_work = gather_along_first_dim(
                    inputmat,
                    ctx.tp_group,
                    async_op=True,
                    quantizer=gather_quantizer,
                )
                nvtx_range_pop(f"{nvtx_label}.column_parallel_comm_input")
            else:
                inputmat_total = inputmat

            # Check whether to output wgrad GEMM directly into main grad
            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

            # Compute grad input tensor
            dgrad = None
            dgrad_work = None
            if ctx.requires_dgrad:

                # Update quantizer
                if ctx.grad_input_quantizer is not None:
                    ctx.grad_input_quantizer.set_usage(rowwise=True, columnwise=False)
                # dgrad GEMM
                nvtx_range_push(f"{nvtx_label}.dgrad_gemm")
                dgrad_gemm_use_split_accumulator = _2X_ACC_DGRAD
                if ctx.fp8:
                    recipe = ctx.fp8_recipe
                    if hasattr(recipe, "fp8_gemm_dgrad"):
                        dgrad_gemm_use_split_accumulator = (
                            recipe.fp8_gemm_dgrad.use_split_accumulator
                        )

                if ctx.weight_quantizer is not None and isinstance(weight_fp8, QuantizedTensor):
                    weight_fp8.update_usage(
                        rowwise_usage=ctx.weight_quantizer.rowwise_usage,
                        columnwise_usage=ctx.weight_quantizer.columnwise_usage,
                    )

                dgrad, *_, rs_out = general_gemm(
                    weight_fp8,
                    grad_output,
                    get_workspace(),
                    layout="NN",
                    grad=True,
                    quantization_params=ctx.grad_input_quantizer,
                    out=dgrad_bulk,
                    out_dtype=ctx.activation_dtype,
                    use_split_accumulator=dgrad_gemm_use_split_accumulator,
                    ub=ub_obj_dgrad,
                    ub_type=ub_type_dgrad,
                    extra_output=rs_out,
                    bulk_overlap=ctx.ub_bulk_dgrad,
                )
                nvtx_range_pop(f"{nvtx_label}.dgrad_gemm")

                # Launch tensor-parallel communication
                if ctx.ub_overlap_rs_dgrad:
                    dgrad = rs_out
                elif ctx.parallel_mode == "column" and not ctx.ub_bulk_wgrad:
                    nvtx_range_push(f"{nvtx_label}.column_parallel_comm_dgrad")
                    if ctx.sequence_parallel:
                        dgrad, dgrad_work = reduce_scatter_along_first_dim(
                            dgrad,
                            ctx.tp_group,
                            async_op=True,
                        )
                    else:
                        dgrad, dgrad_work = allreduce(dgrad, ctx.tp_group, async_op=True)
                    nvtx_range_pop(f"{nvtx_label}.column_parallel_comm_dgrad")

            # Compute grad weight tensor
            wgrad = None
            if ctx.requires_wgrad:

                # Synchronize tensor-parallel communication for input tensor
                if ctx.ub_bulk_dgrad:
                    inputmat_total = ub_obj_dgrad.get_buffer(ctx.input_quantizer)
                    if ctx.fp8:
                        if inputmat._data is None:
                            # All-gather executed on columnwise data and result is in rowwise data,
                            # so we need to fix the interleaving before WGRAD.
                            inputmat_total = _fix_gathered_fp8_transpose(
                                inputmat_total, ctx.tp_size
                            )
                        elif not is_non_tn_fp8_gemm_supported():
                            # FP8 GEMM on Hopper only supports TN layout so the gathered input must
                            # have a valid transpose.
                            inputmat_total._create_transpose()
                if inputmat_total_work is not None:
                    inputmat_total_work.wait()
                    inputmat_total_work = None
                if ctx.input_quantizer is not None and not isinstance(
                    inputmat_total, QuantizedTensor
                ):
                    # Async gather in BF16 does not asynchronously
                    # call quantizer after gather.
                    ctx.input_quantizer.set_usage(rowwise=False, columnwise=True)
                    inputmat_total = ctx.input_quantizer(inputmat_total)

                # Make sure GEMM inputs have required data
                if isinstance(inputmat_total, QuantizedTensor):
                    inputmat_total.update_usage(columnwise_usage=True)
                if isinstance(grad_output, QuantizedTensor):
                    grad_output.update_usage(columnwise_usage=True)

                # Figure out whether to use split accumulator
                use_split_accumulator = _2X_ACC_WGRAD
                if ctx.fp8:
                    recipe = ctx.fp8_recipe
                    if hasattr(recipe, "fp8_gemm_wgrad"):
                        use_split_accumulator = recipe.fp8_gemm_wgrad.use_split_accumulator

                # Output buffer for overlapping grad input
                # reduce-scatter with wgrad GEMM
                if ctx.ub_bulk_wgrad and ub_obj_wgrad.is_fp8_ubuf():
                    rs_out = torch.empty(
                        dgrad_shape, dtype=ctx.activation_dtype, device=grad_output.device
                    )

                # wgrad GEMM
                # Note: Fuse with bgrad computation if needed
                nvtx_range_push(f"{nvtx_label}.wgrad_gemm")
                general_gemm_wgrad = functools.partial(
                    general_gemm,
                    out_dtype=(
                        main_grad.dtype if ctx.fuse_wgrad_accumulation else ctx.activation_dtype
                    ),
                    workspace=get_workspace(),
                    layout="NT",
                    grad=True,
                    bias=(bias if (grad_bias is None and not ctx.fp8) else None),
                    out=main_grad if ctx.fuse_wgrad_accumulation else None,
                    use_split_accumulator=use_split_accumulator,
                    accumulate=accumulate_wgrad_into_param_main_grad,
                    quantization_params=ctx.grad_weight_quantizer,
                    ub=ub_obj_wgrad,
                    ub_type=ub_type_wgrad,
                    extra_output=rs_out,
                    bulk_overlap=ctx.ub_bulk_wgrad,
                )

                if ctx.wgrad_store is not None and ctx.wgrad_store.delay_wgrad_compute():
                    ctx.wgrad_store.put([inputmat_total, grad_output], general_gemm_wgrad)
                else:
                    wgrad, grad_bias_, _, rs_out = general_gemm_wgrad(inputmat_total, grad_output)

                    if grad_bias is None:
                        grad_bias = grad_bias_
                    del grad_bias_

                    # Deallocate input tensor
                    if ctx.owns_input:
                        clear_tensor_data(inputmat_total)
                nvtx_range_pop(f"{nvtx_label}.wgrad_gemm")

                if ctx.ub_bulk_wgrad:
                    if ub_obj_wgrad.is_fp8_ubuf():
                        dgrad = rs_out
                    else:
                        dgrad = ub_obj_wgrad.get_buffer(ctx.grad_input_quantizer, local_chunk=True)

            # Don't return grad bias if not needed
            if not ctx.use_bias:
                grad_bias = None

            # Make sure all tensor-parallel communication is finished
            if inputmat_total_work is not None:
                inputmat_total_work.wait()
                inputmat_total_work = None
            if dgrad_work is not None:
                dgrad_work.wait()
                dgrad_work = None

        if ctx.requires_wgrad:
            # Handle custom DDP from mcore.
            if (
                ctx.fuse_wgrad_accumulation
                and weight is not None
                and hasattr(weight, "grad_added_to_main_grad")
            ):
                weight.grad_added_to_main_grad = True
                if getattr(weight, "zero_out_wgrad", False):
                    wgrad = get_dummy_wgrad(
                        list(weight.main_grad.shape),
                        weight.dtype,
                        zero=True,
                    )
                else:
                    wgrad = get_dummy_wgrad(
                        list(weight.main_grad.shape),
                        weight.dtype,
                    )
            elif ctx.fuse_wgrad_accumulation:
                wgrad = None
        else:
            wgrad = None

        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            nvtx_range_push(f"{nvtx_label}.reduce_and_update_fp8_tensors")
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
            nvtx_range_pop(f"{nvtx_label}.reduce_and_update_fp8_tensors")

        # Scatter fp8 weight buffers
        if ctx.fp8 and not isinstance(weight, QuantizedTensor):
            _fsdp_scatter_tensors(ctx.fsdp_group, weight_fp8)
        return (
            wgrad,
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            grad_bias,
            None,  # is_first_microbatch
            None,  # fp8
            None,  # fp8_calibration
            None,  # wgrad_store
            None,  # input_quantizer
            None,  # weight_quantizer
            None,  # output_quantizer
            None,  # grad_input_quantizer
            None,  # grad_weight_quantizer
            None,  # grad_output_quantizer
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
            None,  # module
            None,  # skip_fp8_weight_update
            None,  # symmetric_ar_type
            None,  # debug
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
    name: str, default = `None`
        name of the module, currently used for debugging purposes.

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
        name: Optional[str] = None,
    ) -> None:
        super().__init__()

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
        self.name = name

        if TEDebugState.debug_enabled:
            self._turn_off_unsupported_features_in_debug()  # turn off userbuffers

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

    def set_meta_tensor(self, fwd: bool, recipe: Recipe) -> None:
        """Init scales and amaxes for fwd | bwd."""
        super().set_meta_tensor(fwd, recipe)

        # customize quantizers based on each recipe & layer configs
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        if recipe.float8_current_scaling():
            self._customize_quantizers_float8_current_scaling(fwd, recipe)
        # elif for other recipes (mxfp8, etc.)

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
        debug = TEDebugState.debug_enabled
        if debug:
            self._validate_name()

        if FP8GlobalStateManager.fp8_graph_capturing():
            skip_fp8_weight_update = FP8GlobalStateManager.get_skip_fp8_weight_update_tensor()
        else:
            skip_fp8_weight_update = None
        if skip_fp8_weight_update is not None:
            is_first_microbatch = False

        if self.ub_overlap_rs_fprop:
            if get_ub(self.ub_name + "_fprop").is_fp8_ubuf():
                fp8_output = True
        if self.ub_overlap_rs_dgrad:
            if get_ub(self.ub_name + "_dgrad").is_fp8_ubuf():
                fp8_grad = True

        with self.prepare_forward(
            inp,
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
                    warnings.warn(
                        "You are using quantized weights without quantized compute. "
                        "Please make sure this is intentional."
                    )
                    unfused_weights = [w.dequantize() for w in unfused_weights]

            weight_tensor = noop_cat(unfused_weights)
            if self.use_bias:
                bias_tensor = noop_cat([getattr(self, name) for name in self.bias_names])
            else:
                bias_tensor = None

            quantizers = (
                self._get_quantizers(fp8_output, fp8_grad)
                if not debug
                else self._get_debug_quantizers(fp8_output, fp8_grad)
            )
            if debug:
                if not any_feature_enabled(quantizers):
                    # If no feature is used, then run faster implementation with debug = False.
                    quantizers = self._get_quantizers(fp8_output, fp8_grad)
                    debug = False

                if isinstance(weight_tensor, QuantizedTensor):
                    raise RuntimeError("FP8 weights are not supported in debug mode.")

            (
                input_quantizer,
                weight_quantizer,
                output_quantizer,
                grad_input_quantizer,
                grad_weight_quantizer,
                grad_output_quantizer,
            ) = quantizers

            # Make sure weight tensor has correct quantizer
            # Note: Quantizer might have changed if quantization
            # recipe changed
            if weight_quantizer is not None and isinstance(weight_tensor, QuantizedTensor):
                weight_tensor._quantizer = weight_quantizer

            if torch.is_grad_enabled():
                linear_fn = _Linear.apply
                args = []
            else:
                linear_fn = _Linear.forward
                args = [None]
            args += (
                weight_tensor,
                inp,
                bias_tensor if (self.apply_bias and not self.gemm_bias_unfused_add) else None,
                is_first_microbatch,
                self.fp8,
                self.fp8_calibration,
                self.wgrad_store,
                input_quantizer,
                weight_quantizer,
                output_quantizer,
                grad_input_quantizer,
                grad_weight_quantizer,
                grad_output_quantizer,
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
                self,
                skip_fp8_weight_update,
                self.symmetric_ar_type,
                debug,
            )
            out = linear_fn(*args)
        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        return out

    def _get_quantizers(self, fp8_output, fp8_grad):
        if not self.fp8:
            return [None] * 6
        grad_input_quantizer = None
        grad_weight_quantizer = None
        grad_output_quantizer = None
        output_quantizer = None
        input_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_INPUT]
        input_quantizer.internal = False
        weight_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_WEIGHT]
        weight_quantizer.internal = True
        if fp8_output:
            output_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_OUTPUT]
        if torch.is_grad_enabled():
            grad_output_quantizer = self.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_OUTPUT1]
            grad_output_quantizer.internal = True
            if fp8_grad:
                grad_input_quantizer = self.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_INPUT1]
        return (
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            grad_input_quantizer,
            grad_weight_quantizer,
            grad_output_quantizer,
        )

    def _get_debug_quantizers(self, fp8_output, fp8_grad):
        original_quantizers = self._get_quantizers(fp8_output, fp8_grad)
        assert TEDebugState.debug_enabled
        from ...debug.pytorch.debug_quantization import DebugQuantizer

        names = ["activation", "weight", "output", "dgrad", "wgrad", "gradient"]
        return tuple(
            DebugQuantizer(self.name, name, q, self.tp_group)
            for name, q in zip(names, original_quantizers)
        )

    def _customize_quantizers_float8_current_scaling(self, fwd: bool, recipe: Recipe) -> None:
        """Customize quantizers based on current scaling recipe + linear."""
        assert (
            recipe.float8_current_scaling()
        ), "current scaling recipe quantizer customization here"
        if fwd:
            # set configs about amax epsilon and power_2_scale
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM1_INPUT
            ].force_pow_2_scales = recipe.fp8_quant_fwd_inp.power_2_scale
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM1_INPUT
            ].amax_epsilon = recipe.fp8_quant_fwd_inp.amax_epsilon
            # also set weight quantizer with same amax_epsilon & power_2_scale
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM1_WEIGHT
            ].force_pow_2_scales = recipe.fp8_quant_fwd_weight.power_2_scale
            self.quantizers["scaling_fwd"][
                tex.FP8FwdTensors.GEMM1_WEIGHT
            ].amax_epsilon = recipe.fp8_quant_fwd_weight.amax_epsilon
            # paralle related
            if self.sequence_parallel and self.parallel_mode == "column":
                # customize input_quantizer with amax reduction TP group
                self.quantizers["scaling_fwd"][
                    tex.FP8FwdTensors.GEMM1_INPUT
                ].with_amax_reduction = True
                self.quantizers["scaling_fwd"][
                    tex.FP8FwdTensors.GEMM1_INPUT
                ].amax_reduction_group = self.tp_group
        else:
            # set grad_output_quantizer with amax epsilon and power_2_scale
            self.quantizers["scaling_bwd"][
                tex.FP8BwdTensors.GRAD_OUTPUT1
            ].force_pow_2_scales = recipe.fp8_quant_bwd_grad.power_2_scale
            self.quantizers["scaling_bwd"][
                tex.FP8BwdTensors.GRAD_OUTPUT1
            ].amax_epsilon = recipe.fp8_quant_bwd_grad.amax_epsilon
            # parallel related
            if self.sequence_parallel and self.parallel_mode == "row":
                # customize grad_output_quantizer with amax reduction TP group
                self.quantizers["scaling_bwd"][
                    tex.FP8BwdTensors.GRAD_OUTPUT1
                ].with_amax_reduction = True
                self.quantizers["scaling_bwd"][
                    tex.FP8BwdTensors.GRAD_OUTPUT1
                ].amax_reduction_group = self.tp_group

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
from typing import Callable, Dict, Optional, Tuple, Union, List
from functools import reduce
from operator import mul as multiply_op
import warnings

import torch

import transformer_engine_torch as tex

from transformer_engine.common.recipe import Recipe
from transformer_engine.pytorch import torch_version

from .base import (
    fill_userbuffers_buffer_for_all_gather,
    get_dummy_wgrad,
    get_ub,
    get_workspace,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ._common import noop_cat, WeightGradStore
from ..fp8 import FP8GlobalStateManager
from ..utils import (
    cast_if_needed,
    clear_tensor_data,
    divide,
    init_method_constant,
    requires_grad,
    needs_quantized_gemm,
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
    QuantizedTensorBase,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)
from ..tensor.float8_tensor import Float8CurrentScalingQuantizer, Float8Quantizer
from ..tensor.mxfp8_tensor import MXFP8Quantizer
from ..tensor._internal.mxfp8_tensor_base import MXFP8TensorBase
from ..tensor._internal.float8_blockwise_tensor_base import Float8BlockwiseQTensorBase
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
            ub_obj = get_ub(ub_name + "_fprop")
            ub_type = tex.CommOverlapType.RS
        elif ub_overlap_ag_fprop:
            ub_obj = get_ub(ub_name + "_fprop")
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
        if with_input_all_gather_nccl or ub_overlap_ag_fprop:  # All-gather input tensor

            # Cast local input tensor if needed
            if fp8 or debug:
                if input_quantizer is None:
                    raise ValueError("Missing quantizer for input tensor")
                if not isinstance(inputmat, QuantizedTensorBase):
                    input_quantizer.set_usage(rowwise=True, columnwise=backward_needs_input)
                    if isinstance(
                        input_quantizer, (Float8Quantizer, Float8CurrentScalingQuantizer)
                    ):
                        # All-gather is not supported with FP8 column-wise data
                        input_quantizer.set_usage(columnwise=False)
                    inputmat = input_quantizer(inputmat)
                    own_quantized_input = True
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
                if isinstance(inputmat, QuantizedTensorBase):
                    inputmat.update_usage(rowwise_usage=True)
                else:
                    if input_quantizer is None:
                        raise ValueError("Missing quantizer for input tensor")
                    input_quantizer.set_usage(rowwise=True, columnwise=backward_needs_input)
                    inputmat = input_quantizer(inputmat)
                    own_quantized_input = True
            else:
                inputmat = cast_if_needed(inp, activation_dtype)  # Cast for AMP
            inputmat_total = inputmat
        nvtx_range_pop(f"{nvtx_label}.input_cast_comm")
        # ------------------------------------------------------
        # Input tensor is ready for GEMM...
        # ------------------------------------------------------

        # ------------------------------------------------------
        # Prepare weight tensor
        # ------------------------------------------------------
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

            # Get quantized weight
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
        if not fp8 and fp8_calibration:
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
            get_workspace(),
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

        # ------------------------------------------------------
        # Prepare output tensor
        # Note: Perform tensor-parallel communication
        # ------------------------------------------------------
        out = None
        if ub_overlap_rs_fprop:
            out = reduce_scatter_out
        elif parallel_mode == "row" and tp_size > 1:
            nvtx_range_push(f"{nvtx_label}.row_parallel_comm")
            out = gemm_out
            if sequence_parallel:
                out, _ = reduce_scatter_along_first_dim(out, tp_group)
            elif tensor_parallel:
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

        # ------------------------------------------------------
        # Cache state for backward pass
        # ------------------------------------------------------

        if is_grad_enabled:
            ctx.weight_quantizer = weight_quantizer
            saved_inputmat = None

            ctx.backward_input_needs_gather = (
                weight.requires_grad and parallel_mode == "column" and sequence_parallel
            )

            if backward_needs_input:
                if own_quantized_input and isinstance(inputmat, QuantizedTensorBase):
                    # For sequence parallel in vanilla FP8, rowwise data is
                    # to gather the input. For MXFP8, columnwise only data
                    # can be allgathered.
                    if (
                        isinstance(inputmat, (MXFP8TensorBase, Float8BlockwiseQTensorBase))
                        or not ctx.backward_input_needs_gather
                    ):
                        inputmat.update_usage(rowwise_usage=False, columnwise_usage=True)
                saved_inputmat = inputmat

            # Weight with column-wise usage is needed for dgrad GEMM.
            if inp.requires_grad:
                if isinstance(weightmat, QuantizedTensorBase):
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
                weightmat if fp8 and not isinstance(weight, QuantizedTensorBase) else None,
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

        # ------------------------------------------------------
        # Cached state for backward pass is ready...
        # ------------------------------------------------------

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring

        # NVTX label for profiling
        nvtx_label = "transformer_engine._Linear.backward"
        if ctx.ub_name is not None:
            nvtx_label = f"{nvtx_label}.{ctx.ub_name}"

        with torch.cuda.nvtx.range("_Linear_backward"):
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

            # Configure Userbuffers communication (comm+GEMM overlap)
            ctx.ub_obj_gradout = None
            ub_obj_dgrad = None
            ub_obj_wgrad = None
            ub_type_dgrad = None
            ub_type_wgrad = None
            dgrad_shape = [reduce(multiply_op, ctx.inp_shape[:-1]), ctx.inp_shape[-1]]
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
            else:
                if ctx.ub_bulk_dgrad:
                    # Overlap inputmat all-gather with dgrad compute
                    ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad")
                    ub_obj_dgrad = ctx.ub_obj_gradout
                    ub_type_dgrad = tex.CommOverlapType.AG
                if ctx.ub_bulk_wgrad:
                    # Overlap dgrad reduce-scatter with wgrad compute
                    ub_obj_wgrad = get_ub(ctx.ub_name + "_wgrad")
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
            if ctx.grad_output_quantizer is not None:
                quantizer = ctx.grad_output_quantizer
                quantizer.set_usage(rowwise=True, columnwise=True)
                if ctx.ub_overlap_ag:
                    # Userbuffers only supports communication for one
                    # tensor usage at a time. Configure quantizer with
                    # usage for only dgrad GEMM.
                    quantizer.set_usage(columnwise=False)

            # Prepare grad output tensor
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
            if ctx.backward_input_needs_gather:
                quantizer = None
                if ctx.fp8 or ctx.debug:
                    quantizer = ctx.input_quantizer
                    if isinstance(quantizer, (Float8Quantizer, Float8CurrentScalingQuantizer)):
                        # If data is in FP8, we compute FP8 transposes manually
                        quantizer.set_usage(rowwise=True, columnwise=False)
                    else:
                        # wgrad GEMM requires input with column-wise usage
                        quantizer.set_usage(rowwise=False, columnwise=True)
                if ctx.ub_bulk_dgrad:
                    inputmat_total, _ = fill_userbuffers_buffer_for_all_gather(
                        ub_obj_dgrad,
                        inputmat,
                        quantizer,
                        ctx.tp_group,
                    )
                else:
                    nvtx_range_push(f"{nvtx_label}.column_parallel_comm_input")
                    inputmat_total, inputmat_total_work = gather_along_first_dim(
                        inputmat,
                        ctx.tp_group,
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
            if ctx.requires_dgrad:

                # Make sure required data is available
                if isinstance(grad_output, QuantizedTensorBase):
                    grad_output.update_usage(rowwise_usage=True)
                if ctx.weight_quantizer is not None and isinstance(weight_fp8, QuantizedTensorBase):
                    weight_fp8.update_usage(columnwise_usage=True)

                # Choose whether to use GEMM kernel with split accumulator
                use_split_accumulator = _2X_ACC_DGRAD
                if ctx.fp8:
                    recipe = ctx.fp8_recipe
                    if hasattr(recipe, "fp8_gemm_dgrad"):
                        use_split_accumulator = recipe.fp8_gemm_dgrad.use_split_accumulator

                # Update grad input quantizer
                if ctx.grad_input_quantizer is not None:
                    ctx.grad_input_quantizer.set_usage(rowwise=True, columnwise=False)

                # Output buffers for Userbuffers reduce-scatter
                gemm_out = None
                reduce_scatter_out = None
                if ctx.ub_overlap_rs_dgrad:
                    reduce_scatter_out = torch.empty(
                        dgrad_shape, dtype=ctx.activation_dtype, device=grad_output_arg.device
                    )
                elif ctx.ub_bulk_wgrad:
                    gemm_out = ub_obj_wgrad.get_buffer(local_chunk=False)

                # dgrad GEMM
                # Note: dx = dy * w
                nvtx_range_push(f"{nvtx_label}.dgrad_gemm")
                gemm_out, *_, reduce_scatter_out = general_gemm(
                    weight_fp8,
                    grad_output,
                    get_workspace(),
                    layout="NN",
                    grad=True,
                    quantization_params=ctx.grad_input_quantizer,
                    out=gemm_out,
                    out_dtype=ctx.activation_dtype,
                    use_split_accumulator=use_split_accumulator,
                    ub=ub_obj_dgrad,
                    ub_type=ub_type_dgrad,
                    extra_output=reduce_scatter_out,
                    bulk_overlap=ctx.ub_bulk_dgrad,
                )
                nvtx_range_pop(f"{nvtx_label}.dgrad_gemm")

                # Prepare grad input tensor
                # Note: Perform tensor-parallel communication
                if ctx.ub_overlap_rs_dgrad:
                    dgrad = reduce_scatter_out
                elif ctx.ub_bulk_wgrad:
                    dgrad = ub_obj_wgrad.get_buffer(local_chunk=True)
                elif ctx.parallel_mode == "column" and ctx.tp_size > 1:
                    nvtx_range_push(f"{nvtx_label}.column_parallel_comm_dgrad")
                    dgrad = gemm_out
                    if ctx.sequence_parallel:
                        dgrad, dgrad_work = reduce_scatter_along_first_dim(
                            dgrad,
                            ctx.tp_group,
                            async_op=True,
                        )
                    else:
                        dgrad, dgrad_work = allreduce(dgrad, ctx.tp_group, async_op=True)
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
            if ctx.requires_wgrad:

                # Prepare input tensor
                # Note: Synchronize tensor-parallel communication and
                # make sure required data is available
                if inputmat_total_work is not None:
                    inputmat_total_work.wait()
                    inputmat_total_work = None
                if ctx.fp8 or ctx.debug:
                    if isinstance(inputmat_total, QuantizedTensorBase):
                        inputmat_total.update_usage(columnwise_usage=True)
                    else:
                        ctx.input_quantizer.set_usage(rowwise=False, columnwise=True)
                        inputmat_total = ctx.input_quantizer(inputmat_total)

                # Prepare grad output tensor
                # Note: Synchronize tensor-parallel communication and
                # make sure required data is available
                if ctx.ub_overlap_ag and isinstance(ctx.grad_output_quantizer, MXFP8Quantizer):
                    # UB does not support overlapping grad output
                    # all-gather with wgrad GEMM. Also, we can't
                    # convert row-scaled MXFP8 to column-scaled, so we
                    # can't reuse the grad output that was gathered
                    # for the dgrad GEMM. We work around with blocking
                    # all-gather for column-scaled MXFP8 data.
                    ctx.grad_output_quantizer.set_usage(rowwise=False, columnwise=True)
                    grad_output, _ = gather_along_first_dim(
                        grad_output_arg,
                        ctx.tp_group,
                        quantizer=ctx.grad_output_quantizer,
                    )
                if ctx.fp8 or ctx.debug:
                    if isinstance(grad_output, QuantizedTensorBase):
                        grad_output.update_usage(columnwise_usage=True)
                    else:
                        ctx.grad_output_quantizer.set_usage(rowwise=False, columnwise=True)
                        grad_output = ctx.grad_output_quantizer(grad_output)

                # Figure out whether to use split accumulator
                use_split_accumulator = _2X_ACC_WGRAD
                if ctx.fp8:
                    recipe = ctx.fp8_recipe
                    if hasattr(recipe, "fp8_gemm_wgrad"):
                        use_split_accumulator = recipe.fp8_gemm_wgrad.use_split_accumulator

                # Figure out whether to output wgrad GEMM directly into main grad
                if ctx.is_first_microbatch is not None:
                    accumulate_wgrad_into_param_main_grad = (
                        ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                    )
                else:
                    accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

                # Output buffer for overlapping FP8 grad input
                # reduce-scatter with wgrad GEMM
                reduce_scatter_out = None
                if ctx.ub_bulk_wgrad and ub_obj_wgrad.is_fp8_ubuf():
                    reduce_scatter_out = torch.empty(
                        dgrad_shape, dtype=ctx.activation_dtype, device=grad_output_arg.device
                    )

                # Arguments to include in wgrad GEMM closure
                wgrad_gemm_kwargs = {
                    "workspace": get_workspace(),
                    "out_dtype": (
                        main_grad.dtype if ctx.fuse_wgrad_accumulation else ctx.activation_dtype
                    ),
                    "quantization_params": ctx.grad_weight_quantizer,
                    "accumulate": accumulate_wgrad_into_param_main_grad,
                    "layout": "NT",
                    "out": main_grad if ctx.fuse_wgrad_accumulation else None,
                    "bias": (bias if (grad_bias is None and not ctx.fp8) else None),
                    "use_split_accumulator": use_split_accumulator,
                    "grad": True,
                    "ub": ub_obj_wgrad,
                    "ub_type": ub_type_wgrad,
                    "extra_output": reduce_scatter_out,
                    "bulk_overlap": ctx.ub_bulk_wgrad,
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
                if ctx.wgrad_store is not None and ctx.wgrad_store.delay_wgrad_compute():
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
                    ctx.wgrad_store.put([inputmat_total, grad_output], wgrad_gemm)
                else:

                    # Call wgrad GEMM now
                    wgrad, grad_bias_ = wgrad_gemm(inputmat_total, grad_output)

                    # Update grad bias if needed
                    if grad_bias is None:
                        grad_bias = grad_bias_
                    del grad_bias_

                    # Deallocate input tensor if permitted
                    if ctx.owns_input:
                        clear_tensor_data(inputmat_total)

                # Update grad input if overlapping reduce-scatter with wgrad GEMM
                if ctx.ub_bulk_wgrad:
                    if ub_obj_wgrad.is_fp8_ubuf():
                        dgrad = reduce_scatter_out
                    else:
                        dgrad = ub_obj_wgrad.get_buffer(local_chunk=True).clone()

            # --------------------------------------------------
            # Grad weight has been computed...
            # --------------------------------------------------

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

        # Update FP8 scaling factors if needed
        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            nvtx_range_push(f"{nvtx_label}.reduce_and_update_fp8_tensors")
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
            nvtx_range_pop(f"{nvtx_label}.reduce_and_update_fp8_tensors")

        # Scatter fp8 weight buffers
        if ctx.fp8 and not isinstance(weight, QuantizedTensorBase):
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
        elif recipe.float8_block_scaling():
            self._customize_quantizers_float8_blockwise_scaling(fwd, recipe)
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
            unfused_weights = self._get_weight_tensors()
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
        input_quantizer.internal = True
        (weight_quantizer,) = self._get_weight_quantizers()
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

    def _get_weight_tensors(self) -> List[Union[torch.Tensor, QuantizedTensorBase]]:
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
        if not self.fp8:
            return [None]
        weight_quantizer = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_WEIGHT]
        weight_quantizer.internal = True
        return [weight_quantizer]

    def _customize_quantizers_float8_blockwise_scaling(self, fwd: bool, recipe: Recipe) -> None:
        """Customize quantizers based on blockwise scaling recipe + linear."""
        assert (
            recipe.float8_block_scaling()
        ), "blockwise scaling recipe quantizer customization here"

        if fwd:
            if self.sequence_parallel and self.parallel_mode == "column":
                # set compact for inp tensor X
                self.quantizers["scaling_fwd"][
                    tex.FP8FwdTensors.GEMM1_INPUT
                ].all_gather_usage = True
        else:
            if self.sequence_parallel and self.parallel_mode == "row":
                # set compact for grad_output tensor dY
                self.quantizers["scaling_bwd"][
                    tex.FP8BwdTensors.GRAD_OUTPUT1
                ].all_gather_usage = True

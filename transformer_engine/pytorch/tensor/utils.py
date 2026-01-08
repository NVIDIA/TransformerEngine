# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Helper functions for using fp8/nvfp4 tensors as weights"""

from typing import Optional, Union, List
import math
import torch

import transformer_engine_torch as tex
from transformer_engine_torch import (
    multi_tensor_scale,
    multi_tensor_compute_scale_and_scale_inv,
    multi_tensor_compute_scale_inv_e8m0,
)

from ..quantized_tensor import QuantizedTensor, Quantizer, QuantizedTensorStorage
from .float8_tensor import Float8Tensor, Float8Quantizer, Float8CurrentScalingQuantizer
from .nvfp4_tensor import NVFP4Tensor, NVFP4Quantizer
from .mxfp8_tensor import MXFP8Tensor, MXFP8Quantizer
from .float8_blockwise_tensor import Float8BlockwiseQTensor, Float8BlockQuantizer
from ..optimizers.multi_tensor_apply import multi_tensor_applier
from ..utils import is_non_tn_fp8_gemm_supported
from ..constants import NVFP4_BLOCK_SCALING_SIZE


def replace_raw_data(tensor: QuantizedTensor, new_raw_data: torch.Tensor):
    r"""Change a quantized tensor's data buffer while preserving values

    This function modifies only the address space of the underlying
    raw data and does not alter any other tensor attributes or values.

    This may be used for custom buffer allocations, e.g. packing
    multiple parameter tensors together into a single contiguous
    buffer for ZeRO-2.

    """
    if isinstance(tensor, Float8Tensor):
        old_raw_data = tensor._data
        assert old_raw_data.dtype == new_raw_data.dtype, "The data types of raw data don't match"
        new_raw_data.detach().copy_(old_raw_data)
        tensor._data = new_raw_data
        del old_raw_data
    elif isinstance(tensor, Float8BlockwiseQTensor):
        old_raw_data = tensor._rowwise_data
        assert old_raw_data.dtype == new_raw_data.dtype, "The data types of raw data don't match"
        new_raw_data.detach().copy_(old_raw_data)
        tensor._rowwise_data = new_raw_data
        del old_raw_data
    elif isinstance(tensor, NVFP4Tensor):
        old_rowwise = tensor._rowwise_data
        assert old_rowwise.dtype == new_raw_data.dtype, "The data types of raw data don't match"
        new_rowwise_data.detach().copy_(old_rowwise)
        tensor._rowwise_data = new_rowwise_data
        del old_rowwise
    elif isinstance(tensor, MXFP8Tensor):
        raise NotImplementedError("replace_raw_data for MXFP8Tensor is not supported yet")
    else:
        raise ValueError(f"replace_raw_data for {type(tensor)} is not supported yet")


def cast_master_weights_to_fp8(
    model_weights,
    master_weights,
    start_offsets,
    group,
    fsdp_shard_model_weights=None,
    manual_post_all_gather_processing=False,
):
    r"""Helper function to cast master weights to FP8 primary weights.

    This is intended for use with ZeRO/FSDP. Each rank has a shard of
    the master weights (possibly empty) and a full copy of the model
    weights.

    Parameters
    ----------
    model_weights  : list of FP8 weights.
    master_weights : list of master weights. Typically they are FP32 weights.
    start_offsets  : list of integers, the starting index of the master weight in the model weight.
                     master_weight may be smaller than model_weight because it could be distributed
                     across multiple ranks. These offsets indicate which part of the model_weight
                     should be updated.
    group          : The distributed group to do amax reduction. Typically it's the data parallel
                     group.
    fsdp_shard_model_weights : list of FSDP shard model weights. If None, it means that the model weights are
                             not sharded. Otherwise, it means that the model weights are sharded and we get
                             target model weights data storage using the FSDP shard model weights.
    manual_post_all_gather_processing : bool, default = `False`.
                     If False, post processing will be automatically triggered during next forward.
                     If True, the timing of calling post_all_gather_processing is left to the user.
                     Note that users must call `post_all_gather_processing` if it's set to True,
                     otherwise the weights won't be updated correctly.

    """

    delayed_scaling_params = []
    current_scaling_params = []
    blockwise_scaling_params = []
    mxfp8_scaling_params = []

    if fsdp_shard_model_weights is None:
        use_fsdp_shard_model_weights = False
        fsdp_shard_model_weights = [None] * len(model_weights)
    else:
        use_fsdp_shard_model_weights = True

    for model_weight, master_weight, start_offset, fsdp_shard_model_weight in zip(
        model_weights, master_weights, start_offsets, fsdp_shard_model_weights
    ):
        # Clear `_high_precision_init_val` of model_weight automatically.
        # - Master weights are initialized from model weights, if we use fp8 primary weights to
        #   initialize master weights, the numerical values of master weights are not consistent
        #   with the numerical values when we initialize them from bf16/fp16 weights.
        # - So we add a `_high_precision_init_val` attribute to each model weight to store the
        #   original bf16/fp16 weight on cpu before casting it to fp8. And users can use
        #   `get_high_precision_init_val` to get this cpu tensor.
        # - This cpu tensor is not needed once the master weight is initialized, so users should
        #   call `clear_high_precision_init_val` to remove it after master weight is initialized.
        # - In case users don't call `clear_high_precision_init_val`, we will clear it automatically
        #   here. It's safe to clear the `_high_precision_init_val` at this time because this
        #   function is supposed to be called after the master weights are initialized and updated.
        if hasattr(model_weight, "clear_high_precision_init_val"):
            model_weight.clear_high_precision_init_val()

        if master_weight is not None:
            # When not using fp8_primary_weights, the master_weight (fp32) is first cast to
            # bf16/fp16, and then cast to fp8 during forward. Although it's not necessary when
            # fp8_primary_weights is enabled, we still keep this logic to keep numerical
            # consistency. So here we cast the master_weight to model_weight.dtype.
            master_weight = master_weight.to(model_weight.dtype)

        quantizer = model_weight._get_quantizer()
        if isinstance(quantizer, Float8Quantizer):
            delayed_scaling_params.append(
                (model_weight, master_weight, start_offset, fsdp_shard_model_weight)
            )
        elif isinstance(quantizer, Float8CurrentScalingQuantizer):
            current_scaling_params.append(
                (model_weight, master_weight, start_offset, fsdp_shard_model_weight)
            )
        elif isinstance(quantizer, Float8BlockQuantizer):
            blockwise_scaling_params.append(
                (model_weight, master_weight, start_offset, fsdp_shard_model_weight)
            )
        elif isinstance(quantizer, MXFP8Quantizer):
            mxfp8_scaling_params.append(
                (model_weight, master_weight, start_offset, fsdp_shard_model_weight)
            )
        else:
            raise ValueError(
                f"cast_master_weights_to_fp8 for {type(quantizer)} is not supported yet"
            )

    extra_args = [group, use_fsdp_shard_model_weights, manual_post_all_gather_processing]
    if len(delayed_scaling_params) > 0:
        _cast_master_weights_to_fp8_delayed_scaling(delayed_scaling_params, *extra_args)
    if len(current_scaling_params) > 0:
        _cast_master_weights_to_fp8_current_scaling(current_scaling_params, *extra_args)
    if len(blockwise_scaling_params) > 0:
        _cast_master_weights_to_fp8_blockwise_scaling(blockwise_scaling_params, *extra_args)
    if len(mxfp8_scaling_params) > 0:
        _cast_master_weights_to_fp8_mxfp8_scaling(mxfp8_scaling_params, *extra_args)


def cast_master_weights_to_nvfp4(
    model_weights, master_weights, start_offsets, group, fsdp_shard_model_weights=None,
    manual_post_all_gather_processing=False
):
    """Helper to cast master weights to NVFP4 primary weights."""

    nvfp4_params = []
    if fsdp_shard_model_weights is None:
        use_fsdp_shard_model_weights = False
        fsdp_shard_model_weights = [None] * len(model_weights)
    else:
        use_fsdp_shard_model_weights = True

    # Batch convert master_weights to model dtype (single kernel instead of N kernels)
    # All NVFP4 model_weights should have the same dtype (BF16)
    if len(model_weights) > 0:
        target_dtype = model_weights[0].dtype
        
        # Collect non-None master_weights and their indices
        non_none_indices = []
        non_none_weights = []
        sizes = []
        for i, mw in enumerate(master_weights):
            if mw is not None:
                non_none_indices.append(i)
                non_none_weights.append(mw.view(-1))
                sizes.append(mw.numel())
        
        if len(non_none_weights) > 0 and non_none_weights[0].dtype != target_dtype:
            # Concatenate, convert once, then split
            concatenated = torch.cat(non_none_weights)
            converted = concatenated.to(target_dtype)
            split_weights = torch.split(converted, sizes)
            
            # Rebuild master_weights list with converted tensors
            converted_master_weights = list(master_weights)
            for idx, split_w, orig_mw in zip(non_none_indices, split_weights, 
                                              [master_weights[i] for i in non_none_indices]):
                converted_master_weights[idx] = split_w.view(orig_mw.shape)
            master_weights = converted_master_weights

    for model_weight, master_weight, start_offset, fsdp_shard_model_weight in zip(
        model_weights, master_weights, start_offsets, fsdp_shard_model_weights
    ):
        if hasattr(model_weight, "clear_high_precision_init_val"):
            model_weight.clear_high_precision_init_val()

        quantizer = model_weight._get_quantizer()
        if isinstance(quantizer, NVFP4Quantizer):
            nvfp4_params.append(
                (model_weight, master_weight, start_offset, fsdp_shard_model_weight)
            )
        else:
            raise ValueError(
                f"cast_master_weights_to_nvfp4 only supports NVFP4 tensors, got {type(model_weight)}"
            )
    if len(nvfp4_params) > 0:
        _cast_master_weights_to_nvfp4_2d(
            nvfp4_params, group, use_fsdp_shard_model_weights=use_fsdp_shard_model_weights,
            manual_post_all_gather_processing=manual_post_all_gather_processing
        )

def _cast_master_weights_to_fp8_delayed_scaling(
    params, group, use_fsdp_shard_model_weights=False, manual_post_all_gather_processing=False
):
    r"""Helper function to cast master weights to FP8 primary weights for delayed scaling.

    Parameters
    ----------
    params : List of tuple, each tuple contains a model weight, a master weight, and an offset
             indicating the starting index of the master weight in the model weight.
    group  : The distributed group to do amax reduction. Typically it's the data parallel
             group.
    use_fsdp_shard_model_weights : bool, if True, it means that the model weights are sharded.
    """

    # Collect amaxes to do reduce-max among dp group.
    # Collect scales and scale_invs to update scale_invs of the fp8 weights.
    amaxes, scales, scale_invs = [], [], []

    for model_weight, master_weight, start_offset, shard_model_weight_raw in params:
        if not manual_post_all_gather_processing:
        # Reset transpose cache for all model weights.
            # We cannot create transpose cache here because users (like megatron) may want to
            # overlap the all-gather of model weights and forward process, so the model weight is
            # not updated currently.
            model_weight._reset_caches()

        quantizer = model_weight._get_quantizer()

        amaxes.append(quantizer.amax.view(1))
        scales.append(quantizer.scale.view(1))
        scale_invs.append(model_weight._scale_inv.view(1))

        # If master weight is None, it means that the master weight of the current model weight
        # is in other DP ranks.
        if master_weight is None:
            continue

        # If master weight is not None, start_offset must be a valid value.
        assert start_offset is not None
        assert start_offset >= 0
        end_offset = start_offset + master_weight.numel()
        assert end_offset <= model_weight.numel()

        # master_weight may be smaller than model_weight because it could be distributed across
        # multiple ranks. So we need to create a dummy weight using the raw data from model_weight.
        if not use_fsdp_shard_model_weights:
            shard_model_weight_raw = model_weight._data.view(-1)[start_offset:end_offset]
        shard_model_weight_fp8 = quantizer.create_tensor_from_data(
            shard_model_weight_raw.view(1, -1),
            model_weight.dtype,
        )

        # Cast master weight to fp8.
        quantizer.update_quantized(master_weight.view(1, -1), shard_model_weight_fp8)

    if len(amaxes) > 0:
        dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device=amaxes[0].device)

        # Reduce amaxes.
        packed_amaxes = torch.empty(len(amaxes), dtype=torch.float32, device=amaxes[0].device)
        packed_amax_views = [packed_amaxes[i].view(1) for i in range(len(amaxes))]
        multi_tensor_applier(
            multi_tensor_scale, dummy_overflow_buf, [amaxes, packed_amax_views], 1.0
        )
        torch.distributed.all_reduce(
            packed_amaxes,
            op=torch.distributed.ReduceOp.MAX,
            group=group,
        )
        multi_tensor_applier(
            multi_tensor_scale, dummy_overflow_buf, [packed_amax_views, amaxes], 1.0
        )

        # Update scale_invs.
        packed_scales = torch.empty(len(scales), dtype=torch.float32, device=scales[0].device)
        packed_scale_views = [packed_scales[i].view(1) for i in range(len(scales))]
        multi_tensor_applier(
            multi_tensor_scale, dummy_overflow_buf, [scales, packed_scale_views], 1.0
        )
        torch.reciprocal(packed_scales, out=packed_scales)
        multi_tensor_applier(
            multi_tensor_scale, dummy_overflow_buf, [packed_scale_views, scale_invs], 1.0
        )


def _cast_master_weights_to_fp8_current_scaling(
    params, group, use_fsdp_shard_model_weights=False, manual_post_all_gather_processing=False
):
    r"""Helper function to cast master weights to FP8 primary weights for current scaling.

    Parameters
    ----------
    params : List of tuple, each tuple contains a model weight, a master weight, and an offset
             indicating the starting index of the master weight in the model weight.
    group  : The distributed group to do amax reduction. Typically it's the data parallel
             group.
    use_fsdp_shard_model_weights : bool, if True, it means that the model weights are sharded.
    """

    # Parameter attributes
    device = params[0][0].device
    fp8_dtype = params[0][0]._get_quantizer().dtype
    force_pow_2_scales = params[0][0]._get_quantizer().force_pow_2_scales
    amax_epsilon = params[0][0]._get_quantizer().amax_epsilon

    # Create a dummy overflow buffer, it's needed by multi_tensor_applier.
    dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device=device)

    # Create a contiguous buffer to store amaxes temporarily, so we can perform all all-reduce
    # NCCL kernels at once.
    packed_amaxes = torch.zeros(len(params), dtype=torch.float32, device=device)
    amaxes = [packed_amaxes[i : i + 1] for i in range(len(params))]

    # Collect scales and scale_invs to update them after amax reduction.
    scales, scale_invs = [], []

    # ---------------------------------------------------------------------------------------------
    # Step 1: Iterate through all the none empty master weights and compute amax of them. Store the
    #         amaxes in a contiguous buffer. If the master weight is None, the corresponding amax
    #         will be set to 0.
    # ---------------------------------------------------------------------------------------------
    for (model_weight, master_weight, _, _), amax in zip(params, amaxes):

        # Make sure all the model weights have the same numerical options.
        quantizer = model_weight._get_quantizer()
        assert quantizer.dtype == fp8_dtype
        assert quantizer.force_pow_2_scales == force_pow_2_scales
        assert quantizer.amax_epsilon == amax_epsilon

        scales.append(quantizer.scale.view(1))
        scale_invs.append(model_weight._scale_inv.view(1))

        # Compute amax of the master weight and store it in packed_amaxes.
        if master_weight is not None:
            tex.compute_amax(master_weight, amax)

    # ---------------------------------------------------------------------------------------------
    # Step 2: Perform all-reduce on packed_amaxes to get the global amax.
    # ---------------------------------------------------------------------------------------------
    torch.distributed.all_reduce(packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=group)

    # ---------------------------------------------------------------------------------------------
    # Step 3: Update scales and scale_invs.
    # ---------------------------------------------------------------------------------------------
    if fp8_dtype == tex.DType.kFloat8E4M3:
        max_fp8 = 448.0
    elif fp8_dtype == tex.DType.kFloat8E5M2:
        max_fp8 = 57344.0
    else:
        raise ValueError(f"Unsupported FP8 dtype: {fp8_dtype}")
    multi_tensor_applier(
        multi_tensor_compute_scale_and_scale_inv,
        dummy_overflow_buf,
        [amaxes, scales, scale_invs],
        max_fp8,
        force_pow_2_scales,
        amax_epsilon,
    )

    # ---------------------------------------------------------------------------------------------
    # Step 4: Cast master weights to FP8.
    # ---------------------------------------------------------------------------------------------
    for (model_weight, master_weight, start_offset, model_weight_fragment), scale in zip(
        params, scales
    ):
        if not manual_post_all_gather_processing:
        # Reset transpose cache for all model weights.
            # We cannot create transpose cache here because users (like megatron) may want to
            # overlap the all-gather of model weights and forward process, so the model weight is
            # not updated currently.
            model_weight._reset_caches()

        # If master weight is None, it means that the master weight of the current model weight
        # is in other DP ranks.
        if master_weight is None:
            continue

        # Cast master weight to FP8
        end_offset = start_offset + master_weight.numel()
        if not use_fsdp_shard_model_weights:
            model_weight_fragment = model_weight.reshape(-1)[start_offset:end_offset]
        quantizer = Float8Quantizer(
            scale=scale,
            amax=torch.Tensor(),
            fp8_dtype=model_weight._fp8_dtype,
        )
        if use_fsdp_shard_model_weights and not isinstance(model_weight_fragment, Float8Tensor):
            # NOTE: The fsdp shard model weight may be a unit8 tensor instead of
            # a float8 tensor. We should handle this situation properly.
            model_weight_fragment = quantizer.create_tensor_from_data(
                model_weight_fragment.view(-1),
                model_weight.dtype,
            )
        quantizer.update_quantized(master_weight, model_weight_fragment)


def _cast_master_weights_to_fp8_blockwise_scaling(
    params, group, use_fsdp_shard_model_weights=False, manual_post_all_gather_processing=False
):
    r"""Helper function to cast master weights to FP8 primary weights for blockwise scaling.

    Parameters
    ----------
    params : List of tuple, each tuple contains a model weight, a master weight, and an offset
             indicating the starting index of the master weight in the model weight.
    group  : The distributed group to do amax reduction. Typically it's the data parallel
             group.
    use_fsdp_shard_model_weights : bool, if True, it means that the model weights are sharded.
    """

    # Parameter attributes
    device = params[0][0].device
    block_len = params[0][0]._get_quantizer().block_len
    fp8_dtype = params[0][0]._get_quantizer().dtype
    force_pow_2_scales = params[0][0]._get_quantizer().force_pow_2_scales
    amax_epsilon = params[0][0]._get_quantizer().amax_epsilon

    # Create a dummy overflow buffer, it's needed by multi_tensor_applier.
    dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device=device)

    # Get the total number of amax elements in all the model weights.
    cu_amax_sizes = [0]
    for model_weight, _, _, _ in params:
        scale_shape = model_weight._get_quantizer().get_scale_shape(model_weight.shape, False)
        num_amaxes = scale_shape[0] * scale_shape[1]
        cu_amax_sizes.append(cu_amax_sizes[-1] + num_amaxes)

    # Create a contiguous buffer to store amaxes temporarily, so we can perform all all-reduce
    # NCCL kernels at once.
    packed_amaxes = torch.zeros(cu_amax_sizes[-1], dtype=torch.float32, device=device)

    # ---------------------------------------------------------------------------------------------
    # Step 1: Iterate through all the none empty master weights and compute amax of them. Store the
    #         amaxes in a contiguous buffer. If a block of a master weight is empty, the
    #         corresponding amax will be set to 0.
    # ---------------------------------------------------------------------------------------------
    amaxes, scales, scale_invs = [], [], []
    for i, (model_weight, master_weight, start_offset, _) in enumerate(params):

        # Make sure all the model weights have the same numerical options.
        quantizer = model_weight._get_quantizer()
        assert block_len == quantizer.block_len
        assert fp8_dtype == quantizer.dtype
        assert force_pow_2_scales == quantizer.force_pow_2_scales
        assert amax_epsilon == quantizer.amax_epsilon

        scale_shape = quantizer.get_scale_shape(model_weight.shape, False)
        amax = packed_amaxes[cu_amax_sizes[i] : cu_amax_sizes[i + 1]].reshape(scale_shape)
        scale = torch.empty(scale_shape, dtype=torch.float32, device=device)
        scale_inv = model_weight._rowwise_scale_inv
        assert len(scale_shape) == 2
        assert len(scale_inv.shape) == 2
        assert scale_inv.shape[0] == scale_shape[0]
        assert scale_inv.shape[1] == scale_shape[1]

        amaxes.append(amax)
        scales.append(scale)
        scale_invs.append(scale_inv)

        # Compute amax of the master weight and store it in packed_amaxes.
        if master_weight is not None:
            assert len(model_weight.shape) == 2
            h, w = model_weight.shape
            tex.fp8_block_scaling_compute_partial_amax(
                master_weight, amax, h, w, start_offset, block_len
            )

    # ---------------------------------------------------------------------------------------------
    # Step 2: Perform all-reduce on packed_amaxes to get the global amax.
    # ---------------------------------------------------------------------------------------------
    torch.distributed.all_reduce(packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=group)

    # ---------------------------------------------------------------------------------------------
    # Step 3: Update scales and scale_invs.
    # ---------------------------------------------------------------------------------------------
    if fp8_dtype == tex.DType.kFloat8E4M3:
        max_fp8 = 448.0
    elif fp8_dtype == tex.DType.kFloat8E5M2:
        max_fp8 = 57344.0
    else:
        raise ValueError(f"Unsupported FP8 dtype: {fp8_dtype}")
    multi_tensor_applier(
        multi_tensor_compute_scale_and_scale_inv,
        dummy_overflow_buf,
        [amaxes, scales, scale_invs],
        max_fp8,
        force_pow_2_scales,
        amax_epsilon,
    )

    # ---------------------------------------------------------------------------------------------
    # Step 4: Cast master weights to FP8.
    # ---------------------------------------------------------------------------------------------
    for (model_weight, master_weight, start_offset, model_weight_fragment), scale in zip(
        params, scales
    ):
        if not manual_post_all_gather_processing:
            # Reset transpose cache for all model weights.
            # We cannot create columnwise data here because users (like megatron) may want to
            # overlap the all-gather of model weights and forward process, so the model weight is
            # not updated at this moment.
            model_weight.update_usage(rowwise_usage=True, columnwise_usage=False)

        # If master weight is None, it means that the master weight of the current model weight
        # is in other DP ranks.
        if master_weight is None:
            continue

        # Cast master weight to FP8
        end_offset = start_offset + master_weight.numel()
        if not use_fsdp_shard_model_weights:
            model_weight_fragment = model_weight._rowwise_data.reshape(-1)[start_offset:end_offset]
        assert len(model_weight.shape) == 2
        h, w = model_weight.shape
        tex.fp8_block_scaling_partial_cast(
            master_weight, model_weight_fragment, scale, h, w, start_offset, block_len, fp8_dtype
        )

# revisit this later
def _cast_master_weights_to_nvfp4_2d(
    params, group, use_fsdp_shard_model_weights=False, manual_post_all_gather_processing=False
):
    r"""Helper function to cast master weights to FP8 primary weights for blockwise scaling.

    Parameters
    ----------
    params : List of tuple, each tuple contains a model weight, a master weight, and an offset
             indicating the starting index of the master weight in the model weight.
    group  : The distributed group to do amax reduction. Typically it's the data parallel
             group.
    use_fsdp_shard_model_weights : bool, if True, it means that the model weights are sharded.
    """
    
    device = params[0][0].device
    block_len = NVFP4_BLOCK_SCALING_SIZE

    cu_amax_sizes = [0]
    tile_shapes: List[tuple[int, int]] = []
    row_sizes: List[int] = []
    tile_widths: List[int] = []
    scale_targets: List[torch.Tensor] = []
    amax_targets: List[Optional[torch.Tensor]] = []
    for model_weight, _, _, _ in params:
        quantizer = model_weight._get_quantizer()
        assert isinstance(quantizer, NVFP4Quantizer)
        assert quantizer.with_2d_quantization, "NVFP4 2D quantization must be enabled."
        assert len(model_weight.shape) == 2
        h, w = model_weight.shape
        tile_h = (h + block_len - 1) // block_len
        tile_w = (w + block_len - 1) // block_len
        tile_shapes.append((tile_h, tile_w))
        row_sizes.append(h)
        tile_widths.append(tile_w)
        scale_targets.append(model_weight._rowwise_scale_inv)
        amax_targets.append(model_weight._amax_rowwise)
        num_amaxes = tile_h * tile_w
        cu_amax_sizes.append(cu_amax_sizes[-1] + num_amaxes)

    packed_amaxes = torch.zeros(cu_amax_sizes[-1], dtype=torch.float32, device=device)
    packed_scales = torch.zeros(cu_amax_sizes[-1], dtype=torch.float32, device=device)

    amaxes: List[torch.Tensor] = []
    scales: List[torch.Tensor] = []
    global_amaxes = torch.zeros(len(params), dtype=torch.float32, device=device)
    global_amax_views: List[torch.Tensor] = [
        global_amaxes[i : i + 1] for i in range(len(params))
    ]

    for i, (model_weight, master_weight, start_offset, _) in enumerate(params):
        scale_shape = tile_shapes[i]
        amax = packed_amaxes[cu_amax_sizes[i] : cu_amax_sizes[i + 1]].reshape(scale_shape)
        scale = packed_scales[cu_amax_sizes[i] : cu_amax_sizes[i + 1]].reshape(scale_shape)
        global_amax_view = global_amax_views[i]

        assert model_weight._rowwise_scale_inv is not None

        amaxes.append(amax)
        scales.append(scale)

        if master_weight is not None and master_weight.numel() > 0:
            assert len(model_weight.shape) == 2
            h, w = model_weight.shape
            # master_weight is already converted to model_weight.dtype (BF16) in the caller
            tex.nvfp4_2d_compute_partial_amax(
                master_weight, amax, h, w, start_offset, block_len
            )
            tex.compute_amax(master_weight, global_amax_view)

    if packed_amaxes.numel() > 0:
        torch.distributed.all_reduce(packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=group)

    if global_amaxes.numel() > 0:
        torch.distributed.all_reduce(global_amaxes, op=torch.distributed.ReduceOp.MAX, group=group)

    # Use GPU kernel to compute global encode scales from global amaxes
    # This replaces multiple Python tensor operations with a single kernel
    global_scale_tensor = torch.empty_like(global_amaxes)
    
    tex.nvfp4_compute_global_scale(global_amaxes, global_scale_tensor)
    global_scale_views = [global_scale_tensor[i : i + 1] for i in range(len(params))]

    # Main loop: use fused kernel for scale computation + expansion + amax copy
    # This saves 2 kernel launches per parameter
    zipped_meta = zip(
        tile_shapes,
        row_sizes,
        tile_widths,
        scale_targets,
        amax_targets,
        params,
        amaxes,
        scales,
        global_scale_views,
    )
    for idx, (
        tile_shape,
        rows,
        tile_col_cnt,
        target_scale,
        target_amax,
        (model_weight, master_weight, start_offset, model_weight_fragment),
        block_amax,
        per_block_decode_scale,
        global_scale,
    ) in enumerate(zipped_meta):

        if not manual_post_all_gather_processing:
            # Reset transpose cache for all model weights.
            # We cannot create transpose cache here because users (like megatron) may want to
            # overlap the all-gather of model weights and forward process, so the model weight is
            # not updated currently.
            model_weight.update_usage(rowwise_usage=True, columnwise_usage=False)

        # Use fused kernel: computes per-block decode scale, copies global amax to target,
        # and expands to row-level FP8 scale - all in one kernel launch
        tile_rows = tile_shape[0]
        rows_padded = target_scale.shape[0]
        global_amax_view = global_amaxes[idx : idx + 1]
        
        # target_amax could be None if model_weight._amax_rowwise is None
        if target_amax is not None:
            tex.nvfp4_fused_scale(
                block_amax,
                global_amax_view,
                per_block_decode_scale,
                target_scale,
                target_amax,
                tile_rows,
                tile_col_cnt,
                rows_padded,
                block_len,
            )
        else:
            # Fallback: compute scale and expand without amax copy
            tex.nvfp4_compute_per_block_scale(block_amax, per_block_decode_scale, global_amax_view)
            tex.nvfp4_expand_scale_to_fp8(
                per_block_decode_scale,
                target_scale,
                tile_rows,
                tile_col_cnt,
                rows_padded,
                block_len,
            )

        # Only cast data for layers owned by this rank
        if master_weight is None or master_weight.numel() == 0:
            continue

        end_offset = start_offset + master_weight.numel()
        if not use_fsdp_shard_model_weights:
            rowwise_bytes = model_weight._rowwise_data.view(-1)
            byte_start = start_offset // 2
            byte_end = (end_offset + 1) // 2
            model_weight_fragment = rowwise_bytes[byte_start:byte_end]
        assert len(model_weight.shape) == 2
        h, w = model_weight.shape
        # master_weight is already converted to model_weight.dtype (BF16) in the caller
        tex.nvfp4_2d_partial_cast(
            master_weight,
            model_weight_fragment,
            per_block_decode_scale,
            global_scale,
            h,
            w,
            start_offset,
            block_len,
        )

def _cast_master_weights_to_fp8_mxfp8_scaling(
    params, group, use_fsdp_shard_model_weights=False, manual_post_all_gather_processing=False
):  # pylint: disable=unused-argument
    r"""Helper function to cast master weights to FP8 primary weights for mxfp8 scaling.

    Parameters
    ----------
    params : List of tuple, each tuple contains a model weight, a master weight, and an offset
             indicating the starting index of the master weight in the model weight.
    group  : The distributed group to do amax reduction. Typically it's the data parallel
             group.
    use_fsdp_shard_model_weights : bool, if True, it means that the model weights are sharded.
    """

    # Parameter attributes
    device = params[0][0].device
    for _, master_weight, _, _ in params:
        if master_weight is not None:
            master_weight_dtype = master_weight.dtype
            break

    # Get the total number of amax elements in all the model weights.
    cu_rowwise_amax_sizes = [0]
    cu_colwise_amax_sizes = [0]
    for model_weight, _, _, _ in params:
        rowwise_shape = model_weight._rowwise_scale_inv.shape
        assert len(rowwise_shape) == 2
        colwise_shape = model_weight._columnwise_scale_inv.shape
        assert len(colwise_shape) == 2
        cu_rowwise_amax_sizes.append(
            cu_rowwise_amax_sizes[-1] + rowwise_shape[0] * rowwise_shape[1]
        )
        cu_colwise_amax_sizes.append(
            cu_colwise_amax_sizes[-1] + colwise_shape[0] * colwise_shape[1]
        )

    # Create a contiguous buffer to store amaxes temporarily, so we can perform all all-reduce
    # NCCL kernels at once.
    packed_amaxes = torch.zeros(
        cu_rowwise_amax_sizes[-1] + cu_colwise_amax_sizes[-1],
        dtype=master_weight_dtype,
        device=device,
    )

    # ---------------------------------------------------------------------------------------------
    # Step 1: Iterate through all the none empty master weights and compute amax of them. Store the
    #         amaxes in a contiguous buffer. If a block of a master weight is empty, the
    #         corresponding amax will be set to 0.
    # ---------------------------------------------------------------------------------------------
    amaxes_rowwise, scale_invs_rowwise = [], []
    amaxes_colwise, scale_invs_colwise = [], []
    for i, (model_weight, master_weight, start_offset, _) in enumerate(params):
        rowwise_shape = model_weight._rowwise_scale_inv.shape
        colwise_shape = model_weight._columnwise_scale_inv.shape
        rowwise_start = cu_rowwise_amax_sizes[i]
        rowwise_end = cu_rowwise_amax_sizes[i + 1]
        colwise_start = cu_rowwise_amax_sizes[-1] + cu_colwise_amax_sizes[i]
        colwise_end = cu_rowwise_amax_sizes[-1] + cu_colwise_amax_sizes[i + 1]
        amax_rowwise = packed_amaxes[rowwise_start:rowwise_end].reshape(rowwise_shape)
        amax_colwise = packed_amaxes[colwise_start:colwise_end].reshape(colwise_shape)
        amaxes_rowwise.append(amax_rowwise)
        amaxes_colwise.append(amax_colwise)
        scale_invs_rowwise.append(model_weight._rowwise_scale_inv)
        scale_invs_colwise.append(model_weight._columnwise_scale_inv)

        # Compute amax of the master weight and store it in packed_amaxes.
        if master_weight is not None:
            assert len(model_weight.shape) == 2
            h, w = model_weight.shape
            tex.mxfp8_scaling_compute_partial_amax(
                master_weight, amax_rowwise, amax_colwise, h, w, start_offset
            )

    # ---------------------------------------------------------------------------------------------
    # Step 2: Perform all-reduce on packed_amaxes to get the global amax.
    # ---------------------------------------------------------------------------------------------
    torch.distributed.all_reduce(packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=group)

    # ---------------------------------------------------------------------------------------------
    # Step 3: Update scales and scale_invs.
    # ---------------------------------------------------------------------------------------------
    multi_tensor_applier(
        multi_tensor_compute_scale_inv_e8m0,
        None,  # dummy_overflow_buf
        [
            amaxes_rowwise + amaxes_colwise,
            scale_invs_rowwise + scale_invs_colwise,
        ],
    )

    # ---------------------------------------------------------------------------------------------
    # Step 4: Cast master weights to FP8.
    # ---------------------------------------------------------------------------------------------
    for (
        (model_weight, master_weight, start_offset, model_weight_fragment),
        scale_inv_rowwise,
        scale_inv_colwise,
    ) in zip(params, scale_invs_rowwise, scale_invs_colwise):
        # If master weight is None, it means that the master weight of the current model weight
        # is in other DP ranks.
        if master_weight is None:
            continue

        # Cast master weight to FP8
        end_offset = start_offset + master_weight.numel()
        if use_fsdp_shard_model_weights:
            rowwise_fragment = model_weight_fragment[0]
            colwise_fragment = model_weight_fragment[1]
        else:
            rowwise_fragment = model_weight._rowwise_data.reshape(-1)[start_offset:end_offset]
            colwise_fragment = model_weight._columnwise_data.reshape(-1)[start_offset:end_offset]
        assert len(model_weight.shape) == 2
        h, w = model_weight.shape
        tex.mxfp8_scaling_partial_cast(
            master_weight,
            rowwise_fragment,
            colwise_fragment,
            scale_inv_rowwise,
            scale_inv_colwise,
            h,
            w,
            start_offset,
        )


def post_all_gather_processing(model_weights: Union[torch.Tensor, List[torch.Tensor]]):
    """
    Post-processing after all-gather for weights in distributed optimizer.
    - Float8Tensor: may need to create a transposed view to match backend GEMM.
    - Float8BlockwiseQTensor: create column-wise storage.
    - Plain pytorch tensor: noop.
    
    For NVFP4 tensors, uses batched multi-tensor processing to reduce CPU overhead.
    """
    if not isinstance(model_weights, list):
        model_weights = [model_weights]
    
    # Collect NVFP4 tensors for batched processing
    nvfp4_tensors = []
    
    for model_weight in model_weights:
        if isinstance(model_weight, Float8Tensor):
            # Delayed scaling and per-tensor current scaling: if backend does not support
            # non-transposed FP8 GEMM, pre-create the transpose.
            if not is_non_tn_fp8_gemm_supported():
                model_weight._create_transpose()
        elif isinstance(model_weight, Float8BlockwiseQTensor):
            # Blockwise scaling: create column-wise storage.
            model_weight._create_columnwise()
        elif isinstance(model_weight, NVFP4Tensor):
            # Collect for batched processing
            nvfp4_tensors.append(model_weight)
        elif isinstance(model_weight, MXFP8Tensor):
            # MXFP8 scaling: no need to do anything.
            pass
        elif isinstance(model_weight, QuantizedTensor):
            raise ValueError(f"post_processing for {type(model_weight)} is not supported")
    
    # Batch process all NVFP4 tensors with multi-tensor approach
    if nvfp4_tensors:
        _nvfp4_multi_tensor_create_columnwise(nvfp4_tensors)


def _nvfp4_multi_tensor_create_columnwise(nvfp4_tensors: List[NVFP4Tensor]):
    """
    Batched columnwise creation for multiple NVFP4 tensors.
    Reduces CPU overhead by collecting all tensor metadata and dispatching to C++.
    """
    TILE_SIZE = 16
    
    # Prepare tensor lists for batched C++ call
    rowwise_data_list = []
    columnwise_data_list = []
    rowwise_scale_inv_list = []
    columnwise_scale_inv_list = []
    M_list = []
    K_list = []
    
    for tensor in nvfp4_tensors:
        rowwise_data = tensor._rowwise_data
        if not rowwise_data.is_contiguous():
            rowwise_data = rowwise_data.contiguous()
            tensor._rowwise_data = rowwise_data
        
        logical_shape = tensor.size()
        M, K = logical_shape[0], logical_shape[-1]
        M_tiles = (M + TILE_SIZE - 1) // TILE_SIZE
        K_tiles = (K + TILE_SIZE - 1) // TILE_SIZE
        
        # Allocate columnwise_data if needed
        if tensor._columnwise_data is None:
            # Output shape: [K, M/2] packed bytes
            columnwise_data = torch.empty(
                (K, M // 2),
                dtype=torch.uint8,
                device=rowwise_data.device,
            )
            tensor._columnwise_data = columnwise_data
        else:
            columnwise_data = tensor._columnwise_data
        
        # Allocate columnwise_scale_inv if needed
        if tensor._columnwise_scale_inv is None:
            assert tensor._quantizer is not None
            columnwise_scale_inv_shape = tensor._quantizer.get_scale_shape(logical_shape, True)
            columnwise_scale_inv = torch.empty(
                columnwise_scale_inv_shape,
                dtype=tensor._rowwise_scale_inv.dtype,
                device=tensor._rowwise_scale_inv.device,
            )
            tensor._columnwise_scale_inv = columnwise_scale_inv
        else:
            columnwise_scale_inv = tensor._columnwise_scale_inv
        
        rowwise_data_list.append(rowwise_data)
        columnwise_data_list.append(columnwise_data)
        rowwise_scale_inv_list.append(tensor._rowwise_scale_inv)
        columnwise_scale_inv_list.append(columnwise_scale_inv)
        M_list.append(M)
        K_list.append(K)
        
        # Copy amax if needed
        if tensor._amax_columnwise is None and tensor._amax_rowwise is not None:
            tensor._amax_columnwise = tensor._amax_rowwise.clone()
        elif tensor._amax_rowwise is not None:
            tensor._amax_columnwise.copy_(tensor._amax_rowwise)
    
    # Dispatch to C++ multi-tensor kernel
    tex.nvfp4_multi_tensor_create_columnwise(
        rowwise_data_list,
        columnwise_data_list,
        rowwise_scale_inv_list,
        columnwise_scale_inv_list,
        M_list,
        K_list,
    )


def is_custom(x: Optional[Union[Quantizer, QuantizedTensorStorage]] = None) -> bool:
    """Check if an object is custom.

    Returns False if x is a torch.Tensor.
    """
    if x is None or isinstance(x, torch.Tensor):
        return False
    if not isinstance(x, (Quantizer, QuantizedTensorStorage)):
        raise AssertionError("Object must be a Quantizer or QuantizedTensorStorage instance")
    return hasattr(x, "custom") and x.custom

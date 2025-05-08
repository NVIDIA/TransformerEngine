# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Helper functions for using fp8 tensors as weights"""

import torch
import transformer_engine_torch as tex
from transformer_engine_torch import multi_tensor_scale, multi_tensor_compute_scale_and_scale_inv

from .quantized_tensor import QuantizedTensor
from .float8_tensor import Float8Tensor, Float8Quantizer, Float8CurrentScalingQuantizer
from .mxfp8_tensor import MXFP8Tensor, MXFP8Quantizer
from .float8_blockwise_tensor import Float8BlockwiseQTensor, Float8BlockQuantizer
from ..optimizers.multi_tensor_apply import multi_tensor_applier


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
    elif isinstance(tensor, MXFP8Tensor):
        raise NotImplementedError("replace_raw_data for MXFP8Tensor is not supported yet")
    else:
        raise ValueError(f"replace_raw_data for {type(tensor)} is not supported yet")


def cast_master_weights_to_fp8(
    model_weights, master_weights, start_offsets, group, fsdp_shard_model_weights=None
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

    """

    delayed_scaling_params = []
    current_scaling_params = []
    blockwise_scaling_params = []

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
            raise NotImplementedError(
                "cast_master_weights_to_fp8 for MXFP8BlockScaling is not supported yet"
            )
        else:
            raise ValueError(
                f"cast_master_weights_to_fp8 for {type(quantizer)} is not supported yet"
            )

    if len(delayed_scaling_params) > 0:
        _cast_master_weights_to_fp8_delayed_scaling(
            delayed_scaling_params, group, use_fsdp_shard_model_weights
        )
    if len(current_scaling_params) > 0:
        _cast_master_weights_to_fp8_current_scaling(
            current_scaling_params, group, use_fsdp_shard_model_weights
        )
    if len(blockwise_scaling_params) > 0:
        _cast_master_weights_to_fp8_blockwise_scaling(
            blockwise_scaling_params, group, use_fsdp_shard_model_weights
        )


def _cast_master_weights_to_fp8_delayed_scaling(params, group, use_fsdp_shard_model_weights=False):
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
        # Reset transpose cache for all model weights.
        # We cannot create transpose cache here because users (like megatron) may want to overlap
        # the all-gather of model weights and forward process, so the model weight is not updated
        # currently.
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
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device=amaxes[0].device)

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


def _cast_master_weights_to_fp8_current_scaling(params, group, use_fsdp_shard_model_weights=False):
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
        # Reset transpose cache for all model weights.
        # We cannot create transpose cache here because users (like megatron) may want to overlap
        # the all-gather of model weights and forward process, so the model weight is not updated
        # currently.
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
    params, group, use_fsdp_shard_model_weights=False
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
        # Clear columnwise data for all model weights.
        # We cannot create columnwise data here because users (like megatron) may want to overlap
        # the all-gather of model weights and forward process, so the model weight is not updated
        # at this moment.
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

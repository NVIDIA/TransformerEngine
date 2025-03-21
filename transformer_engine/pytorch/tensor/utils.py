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
    elif isinstance(tensor, MXFP8Tensor):
        raise NotImplementedError("replace_raw_data for MXFP8Tensor is not supported yet")
    else:
        raise ValueError(f"replace_raw_data for {type(tensor)} is not supported yet")


def cast_master_weights_to_fp8(model_weights, master_weights, start_offsets, group):
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

    """

    delayed_scaling_params = []
    current_scaling_params = []

    for model_weight, master_weight, start_offset in zip(
        model_weights, master_weights, start_offsets
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
            delayed_scaling_params.append((model_weight, master_weight, start_offset))
        elif isinstance(quantizer, Float8CurrentScalingQuantizer):
            current_scaling_params.append((model_weight, master_weight, start_offset))
        elif isinstance(quantizer, MXFP8Quantizer):
            raise NotImplementedError(
                "cast_master_weights_to_fp8 for MXFP8BlockScaling is not supported yet"
            )
        else:
            raise ValueError(
                f"cast_master_weights_to_fp8 for {type(quantizer)} is not supported yet"
            )

    if len(delayed_scaling_params) > 0:
        _cast_master_weights_to_fp8_delayed_scaling(delayed_scaling_params, group)
    if len(current_scaling_params) > 0:
        _cast_master_weights_to_fp8_current_scaling(current_scaling_params, group)


def _cast_master_weights_to_fp8_delayed_scaling(params, group):
    r"""Helper function to cast master weights to FP8 primary weights for delayed scaling.

    Parameters
    ----------
    params : List of tuple, each tuple contains a model weight, a master weight, and an offset
             indicating the starting index of the master weight in the model weight.
    group  : The distributed group to do amax reduction. Typically it's the data parallel
             group.
    """

    # Collect amaxes to do reduce-max among dp group.
    # Collect scales and scale_invs to update scale_invs of the fp8 weights.
    amaxes, scales, scale_invs = [], [], []

    for model_weight, master_weight, start_offset in params:
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


def _cast_master_weights_to_fp8_current_scaling(params, group):
    r"""Helper function to cast master weights to FP8 primary weights for current scaling.

    Parameters
    ----------
    params : List of tuple, each tuple contains a model weight, a master weight, and an offset
             indicating the starting index of the master weight in the model weight.
    group  : The distributed group to do amax reduction. Typically it's the data parallel
             group.
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
    for (model_weight, master_weight, _), amax in zip(params, amaxes):

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
    for (model_weight, master_weight, start_offset), scale in zip(params, scales):
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
        model_weight_fragment = model_weight.reshape(-1)[start_offset:end_offset]
        quantizer = Float8Quantizer(
            scale=scale,
            amax=torch.Tensor(),
            fp8_dtype=model_weight._fp8_dtype,
        )
        quantizer.update_quantized(master_weight, model_weight_fragment)

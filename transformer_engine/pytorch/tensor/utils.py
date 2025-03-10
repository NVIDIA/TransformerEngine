# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import torch.distributed as dist

from .float8_tensor import Float8Quantizer, Float8CurrentScalingQuantizer
from .mxfp8_tensor import MXFP8Quantizer

from transformer_engine_torch import multi_tensor_scale
from ..optimizers.multi_tensor_apply import multi_tensor_applier


def cast_master_weights_to_fp8(model_weights, master_weights, start_offsets, group):
    """Helper function to cast master weights to FP8 primary weights."""
    delayed_scaling_params = []
    current_scaling_params = []

    for model_weight, master_weight, start_offset in zip(
        model_weights, master_weights, start_offsets
    ):
        quantizer = model_weight._get_quantizer()
        if isinstance(quantizer, Float8Quantizer):
            delayed_scaling_params.append((model_weight, master_weight, start_offset))
        elif isinstance(quantizer, Float8CurrentScalingQuantizer):
            current_scaling_params.append((model_weight, master_weight, start_offset))
        elif isinstance(quantizer, MXFP8Quantizer):
            raise NotImplementedError(
                f"cast_master_weights_to_fp8 for MXFP8BlockScaling is not supported yet"
            )
        else:
            raise ValueError(
                f"cast_master_weights_to_fp8 for {type(quantizer)} is not supported yet"
            )

    if len(delayed_scaling_params) > 0:
        cast_master_weights_to_fp8_delayed_scaling(delayed_scaling_params, group)
    if len(current_scaling_params) > 0:
        cast_master_weights_to_fp8_current_scaling(current_scaling_params, group)


def cast_master_weights_to_fp8_delayed_scaling(params, group):
    # Collect amaxes to do reduce-max among dp group.
    # Collect scales and scale_invs to update scale_invs of the fp8 weights.
    amaxes, scales, scale_invs = [], [], []

    for model_weight, master_weight, start_offset in params:
        # Reset transpose cache for all model weights.
        # We cannot create transpose cache here because users (like megatron) may want to overlap
        # the all-gather of model weights and forward process, so the model weight is not updated
        # currently.
        model_weight._reset_caches()

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
        quantizer = model_weight._get_quantizer()
        shard_model_weight_fp8 = quantizer.create_tensor_from_data(
            shard_model_weight_raw.view(1, -1),
            model_weight.dtype,
        )

        # When not using fp8_primary_weights, the master_weight (fp32) is first cast to bf16/fp16,
        # and then cast to fp8 during forward. Although it's not necessary when fp8_primary_weights
        # is enabled, we still keep this logic to keep numerical consistency. So here we cast the
        # master_weight to model_weight.dtype.
        master_weight = master_weight.to(model_weight.dtype)
        quantizer.update_quantized(master_weight.view(1, -1), shard_model_weight_fp8)

        amaxes.append(quantizer.amax.view(1))
        scales.append(quantizer.scale.view(1))
        scale_invs.append(model_weight._scale_inv.view(1))

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


def cast_master_weights_to_fp8_current_scaling(params, group):
    # Collect scales and scale_invs to update scale_invs of the fp8 weights.
    scales, scale_invs = [], []

    for model_weight, master_weight, start_offset in params:
        # Reset transpose cache for all model weights.
        # We cannot create transpose cache here because users (like megatron) may want to overlap
        # the all-gather of model weights and forward process, so the model weight is not updated
        # currently.
        model_weight._reset_caches()

        if master_weight is None:
            master_weight = torch.zeros([1], dtype=torch.float32, device=model_weight.device)
            shard_model_weight_raw = torch.empty([1], dtype=torch.uint8, device=model_weight.device)
        else:
            # If master weight is not None, start_offset must be a valid value.
            assert start_offset is not None
            assert start_offset >= 0
            end_offset = start_offset + master_weight.numel()
            assert end_offset <= model_weight.numel()
            # master_weight may be smaller than model_weight because it could be distributed across
            # multiple ranks. So we need to create a slice of raw data from model_weight.
            shard_model_weight_raw = model_weight._data.view(-1)[start_offset:end_offset]

        quantizer = model_weight._get_quantizer()
        quantizer.with_amax_reduction = True
        quantizer.amax_reduction_group = group
        quantizer.amax_reduction_size = dist.get_world_size(group)
        shard_model_weight_fp8 = quantizer.create_tensor_from_data(
            shard_model_weight_raw.view(1, -1),
            model_weight.dtype,
        )

        # When not using fp8_primary_weights, the master_weight (fp32) is first cast to bf16/fp16,
        # and then cast to fp8 during forward. Although it's not necessary when fp8_primary_weights
        # is enabled, we still keep this logic to keep numerical consistency. So here we cast the
        # master_weight to model_weight.dtype.
        master_weight = master_weight.to(model_weight.dtype)
        quantizer.update_quantized(master_weight.view(1, -1), shard_model_weight_fp8)

        scales.append(quantizer.scale.view(1))
        scale_invs.append(model_weight._scale_inv.view(1))

    if len(scales) > 0:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device=scales[0].device)

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

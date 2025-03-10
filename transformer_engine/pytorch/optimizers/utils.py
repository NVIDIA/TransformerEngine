# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import torch.distributed as dist

from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling, MXFP8BlockScaling
from ..fp8 import FP8GlobalStateManager

from transformer_engine_torch import multi_tensor_scale
from .multi_tensor_apply import multi_tensor_applier


def fp8_primary_weights_cast_helper(model_params, master_params, start_offsets, group):
    """Helper function to cast master weights to FP8 primary weights."""
    recipe = FP8GlobalStateManager.get_fp8_recipe()
    if isinstance(recipe, DelayedScaling):
        # TODO: Here is a bug that recipe is always DelayedScaling
        fp8_primary_weights_cast_helper_delayed_scaling(model_params, master_params, start_offsets, group)
    elif isinstance(recipe, Float8CurrentScaling):
        fp8_primary_weights_cast_helper_float8_current_scaling(model_params, master_params, start_offsets, group)
    elif isinstance(recipe, MXFP8BlockScaling):
        raise NotImplementedError("fp8_primary_weights_cast_helper for MXFP8BlockScaling is not supported yet")
    else:
        raise ValueError(f"Unsupported FP8 recipe: {type(recipe)}")


def fp8_primary_weights_cast_helper_delayed_scaling(model_params, master_params, start_offsets, group):
    for model_param, master_param, start_offset in zip(model_params, master_params, start_offsets):
        assert (master_param is not None and start_offset is not None) or (master_param is None and start_offset is None)
        if master_param is None:
            continue

        shard_model_param = model_param._data.view(-1)[start_offset : start_offset + master_param.numel()]
        quantizer = model_param._get_quantizer()
        fp8_shard_model_param = quantizer.create_tensor_from_data(
            shard_model_param.view(1, -1),
            model_param.dtype,
        )
        master_param = master_param.to(model_param.dtype)
        quantizer.update_quantized(
            master_param.view(1, -1),
            fp8_shard_model_param,
        )

    amaxes = []
    scales = []
    scale_invs = []
    for param in model_params:
        quantizer = param._get_quantizer()
        amaxes.append(quantizer.amax.view(1))
        scales.append(quantizer.scale.view(1))
        scale_invs.append(param._scale_inv.view(1))
        # Reset transpose cache
        param._reset_caches()

    if len(scales) > 0:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')

        # Update scaling factors.
        packed_scales = torch.empty(
            len(scales), dtype=torch.float32, device=scales[0].device
        )
        packed_scale_views = [packed_scales[i].view(1) for i in range(len(scales))]
        multi_tensor_applier(multi_tensor_scale, dummy_overflow_buf, [scales, packed_scale_views], 1.0)
        torch.reciprocal(packed_scales, out=packed_scales)
        multi_tensor_applier(multi_tensor_scale, dummy_overflow_buf, [packed_scale_views, scale_invs], 1.0)

        # Reduce amaxes.
        # Note: Assume each param has a separate amax.
        packed_amaxes = torch.empty(
            len(amaxes), dtype=torch.float32, device=amaxes[0].device
        )
        packed_amax_views = [packed_amaxes[i].view(1) for i in range(len(amaxes))]
        multi_tensor_applier(multi_tensor_scale, dummy_overflow_buf, [amaxes, packed_amax_views], 1.0)
        torch.distributed.all_reduce(
            packed_amaxes,
            op=torch.distributed.ReduceOp.MAX,
            group=group,
        )
        multi_tensor_applier(multi_tensor_scale, dummy_overflow_buf, [packed_amax_views, amaxes], 1.0)


def fp8_primary_weights_cast_helper_float8_current_scaling(model_params, master_params, start_offsets, group):
    for model_param, master_param, start_offset in zip(model_params, master_params, start_offsets):
        assert (master_param is not None and start_offset is not None) or (master_param is None and start_offset is None)
        if master_param is None:
            master_param = torch.zeros([1], dtype=model_param.dtype, device=model_param.device)
            shard_model_param = torch.empty([1], dtype=torch.uint8, device=model_param.device)
        else:
            shard_model_param = model_param._data.view(-1)[start_offset : start_offset + master_param.numel()]

        quantizer = model_param._get_quantizer()
        quantizer.with_amax_reduction = True
        quantizer.amax_reduction_group = group
        quantizer.amax_reduction_size = dist.get_world_size(group)
        fp8_shard_model_param = quantizer.create_tensor_from_data(
            shard_model_param.view(1, -1),
            model_param.dtype,
        )
        master_param = master_param.to(model_param.dtype)
        quantizer.update_quantized(
            master_param.view(1, -1),
            fp8_shard_model_param,
        )

    scales = []
    scale_invs = []
    for param in model_params:
        quantizer = param._get_quantizer()
        scales.append(quantizer.scale.view(1))
        scale_invs.append(param._scale_inv.view(1))
        # Reset transpose cache
        param._reset_caches()

    if len(scales) > 0:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')

        # Update scaling factors.
        packed_scales = torch.empty(
            len(scales), dtype=torch.float32, device=scales[0].device
        )
        packed_scale_views = [packed_scales[i].view(1) for i in range(len(scales))]
        multi_tensor_applier(multi_tensor_scale, dummy_overflow_buf, [scales, packed_scale_views], 1.0)
        torch.reciprocal(packed_scales, out=packed_scales)
        multi_tensor_applier(multi_tensor_scale, dummy_overflow_buf, [packed_scale_views, scale_invs], 1.0)

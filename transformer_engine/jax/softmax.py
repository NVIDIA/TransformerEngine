# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX softmax modules"""
from enum import Enum
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp

from .cpp_extensions import scaled_softmax_fwd
from .cpp_extensions import scaled_softmax_bwd
from .cpp_extensions import scaled_masked_softmax_fwd
from .cpp_extensions import scaled_masked_softmax_bwd
from .cpp_extensions import scaled_upper_triang_masked_softmax_fwd
from .cpp_extensions import scaled_upper_triang_masked_softmax_bwd
from .cpp_extensions import ScaledSoftmaxFwdPrimitive
from .cpp_extensions import ScaledMaskedSoftmaxFwdPrimitive
from .cpp_extensions import ScaledUpperTriangMaskedSoftmaxFwdPrimitive
from .sharding import get_softmax_sharding_meta, ShardingType, ShardingMeta
from .sharding import xmap_runner, extend_fsdp_sharding_meta

jax.config.update('experimental_xmap_spmd_lowering', True)
jax.config.update('experimental_xmap_spmd_lowering_manual', True)


class SoftmaxType(Enum):
    """SoftmaxType."""
    SCALED = "scaled"
    SCALED_MASKED = "scaled_masked"
    SCALED_UPPER_TRIANG_MASKED = "scaled_upper_triang_masked"


def is_softmax_kernel_available(softmax_type: SoftmaxType, batch: int, heads: int, q_seqlen: int,
                                k_seqlen: int, dtype: jnp.dtype):
    """check softmax available"""
    if softmax_type is SoftmaxType.SCALED:
        return ScaledSoftmaxFwdPrimitive.is_kernel_available(batch, heads, q_seqlen, k_seqlen,
                                                             dtype)
    if softmax_type is SoftmaxType.SCALED_MASKED:
        return ScaledMaskedSoftmaxFwdPrimitive.is_kernel_available(batch, heads, q_seqlen, k_seqlen,
                                                                   dtype)
    if softmax_type is SoftmaxType.SCALED_UPPER_TRIANG_MASKED:
        return ScaledUpperTriangMaskedSoftmaxFwdPrimitive.is_kernel_available(
            batch, heads, q_seqlen, k_seqlen, dtype)

    raise NotImplementedError


def softmax(inputs: jnp.ndarray,
            mask: Optional[jnp.ndarray] = None,
            scale_factor: Optional[float] = 1.0,
            softmax_type: Optional[SoftmaxType] = SoftmaxType.SCALED,
            sharding_type: ShardingType = ShardingType.SINGLE,
            dp_dim_index: int = 0,
            tp_dim_index: int = 1):
    """
    Softmax wrapper
    """
    assert dp_dim_index == 0, \
        "Only softmax support batch dim in the first place currently."
    assert tp_dim_index == 1, \
        "Only softmax support head dim in the second place currently."

    assert mask is None or mask.shape[tp_dim_index] == 1

    if sharding_type is ShardingType.SINGLE:
        outputs = _softmax(inputs, mask, scale_factor, softmax_type)
    else:
        dp_axis_name = "batch"
        tp_axis_name = "model"

        sharding_meta = get_softmax_sharding_meta(sharding_type,
                                                  inputs.shape,
                                                  dp_dim=dp_dim_index,
                                                  tp_dim=tp_dim_index,
                                                  dp_axis_name=dp_axis_name,
                                                  tp_axis_name=tp_axis_name)

        sharding_meta, _ = extend_fsdp_sharding_meta(sharding_meta, {0: dp_dim_index})

        inputs_ = jnp.reshape(inputs, sharding_meta.input_shapes[0])    # 0 for input
        mask_ = mask
        mask_in_axis = {}
        if mask_ is not None:

            if sharding_type in (ShardingType.DP, ShardingType.DP_TP_COL, ShardingType.DP_TP_ROW):
                # If mask is head broadcastable (heads == 1),
                # then it equals to DP sharding.
                mask_sharding_meta = get_softmax_sharding_meta(ShardingType.DP,
                                                               mask_.shape,
                                                               dp_dim=dp_dim_index,
                                                               tp_dim=tp_dim_index,
                                                               dp_axis_name=dp_axis_name,
                                                               tp_axis_name=tp_axis_name)
            else:
                mask_sharding_meta = ShardingMeta([{}], {}, {}, [mask_.shape], mask_.shape)

            mask_sharding_meta, _ = extend_fsdp_sharding_meta(mask_sharding_meta, {0: dp_dim_index})
            mask_ = jnp.reshape(mask_, mask_sharding_meta.input_shapes[0])
            mask_in_axis = mask_sharding_meta.in_axes[0]

        partial_softmax = partial(_softmax, scale_factor=scale_factor, softmax_type=softmax_type)

        in_axes = (sharding_meta.in_axes[0], mask_in_axis)
        outputs = xmap_runner(partial_softmax, in_axes, sharding_meta.out_axes,
                              sharding_meta.axis_resources, (inputs_, mask_))

        outputs = jnp.reshape(outputs, sharding_meta.output_shapes[0])

    return outputs


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def _softmax(inputs, mask, scale_factor, softmax_type):
    output, _ = _softmax_fwd(inputs, mask, scale_factor, softmax_type)
    return output


def _softmax_fwd(inputs, mask, scale_factor, softmax_type):
    if softmax_type is SoftmaxType.SCALED_MASKED:
        assert mask is not None
        outputs = scaled_masked_softmax_fwd(inputs, mask, scale_factor)
    elif softmax_type is SoftmaxType.SCALED_UPPER_TRIANG_MASKED:
        outputs = scaled_upper_triang_masked_softmax_fwd(inputs, scale_factor)
    else:
        outputs = scaled_softmax_fwd(inputs, scale_factor)

    return outputs, (outputs, mask)


def _softmax_bwd(scale_factor, softmax_type, ctx, grad_outputs):
    softmax_outputs, mask = ctx

    if softmax_type is SoftmaxType.SCALED_MASKED:
        assert mask is not None
        dgrad = scaled_masked_softmax_bwd(grad_outputs, softmax_outputs, scale_factor)
    elif softmax_type is SoftmaxType.SCALED_UPPER_TRIANG_MASKED:
        dgrad = scaled_upper_triang_masked_softmax_bwd(grad_outputs, softmax_outputs, scale_factor)
    else:
        dgrad = scaled_softmax_bwd(grad_outputs, softmax_outputs, scale_factor)

    return (dgrad, None)


_softmax.defvjp(_softmax_fwd, _softmax_bwd)

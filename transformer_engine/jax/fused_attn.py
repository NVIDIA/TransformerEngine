# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX multi-head attention modules"""

from enum import Enum
from functools import partial
import jax
import jax.numpy as jnp

from transformer_engine_jax import NVTE_Bias_Type
from transformer_engine_jax import NVTE_Mask_Type

from .cpp_extensions import cross_fused_attn_max_512_fwd, cross_fused_attn_max_512_bwd
from .cpp_extensions import self_fused_attn_max_512_fwd, self_fused_attn_max_512_bwd
from .sharding import get_fused_attn_sharding_meta
from .sharding import ShardingType
from .sharding import xmap_runner

jax.config.update('experimental_xmap_spmd_lowering', True)
jax.config.update('experimental_xmap_spmd_lowering_manual', True)


class AttnBiasType(Enum):
    """Attention Bias Type."""
    NO_BIAS = NVTE_Bias_Type.NVTE_NO_BIAS
    PRE_SCALE_BIAS = NVTE_Bias_Type.NVTE_PRE_SCALE_BIAS
    POST_SCALE_BIAS = NVTE_Bias_Type.NVTE_POST_SCALE_BIAS


class AttnMaskType(Enum):
    """Attention Mask Type."""
    NO_MASK = NVTE_Mask_Type.NVTE_NO_MASK
    PADDING_MASK = NVTE_Mask_Type.NVTE_PADDING_MASK
    CAUSAL_MASK = NVTE_Mask_Type.NVTE_CAUSAL_MASK


def self_fused_attn(qkv: jnp.ndarray,
                    bias: jnp.ndarray,
                    mask: jnp.ndarray,
                    seed: int,
                    attn_bias_type: AttnBiasType,
                    attn_mask_type: AttnMaskType,
                    scaling_factor: float,
                    dropout_probability: float,
                    is_causal_masking: bool,
                    sharding_type: ShardingType = ShardingType.SINGLE):
    """
    Self fused attention wrapper
    """
    assert sharding_type not in (ShardingType.TP_ROW, ShardingType.DP_TP_ROW), \
        "Fused_attn_max_512 does not support row-split tensor parallelism currently."

    if sharding_type is ShardingType.SINGLE:
        output = _self_fused_attn_max_512(qkv,
                                          bias,
                                          mask,
                                          seed=seed,
                                          attn_bias_type=attn_bias_type,
                                          attn_mask_type=attn_mask_type,
                                          scaling_factor=scaling_factor,
                                          dropout_probability=dropout_probability,
                                          is_causal_masking=is_causal_masking)
    else:
        dp_axis_name = "batch"
        tp_axis_name = "model"

        inputs = [qkv, bias, mask]
        batch, seqlen, _, num_head, head_dim = qkv.shape
        output_shape = [batch, seqlen, num_head, head_dim]
        sharding_meta = get_fused_attn_sharding_meta(sharding_type, [x.shape for x in inputs],
                                                     [output_shape],
                                                     dp_dims=([0, None, 0], [0]),
                                                     tp_dims=([3, 1, None], [2]),
                                                     dp_axis_name=dp_axis_name,
                                                     tp_axis_name=tp_axis_name)

        inputs_ = tuple(
            jnp.reshape(x, new_shape) for x, new_shape in zip(inputs, sharding_meta.input_shapes))

        partial_self_fused_attn_max_512 = partial(_self_fused_attn_max_512,
                                                  seed=seed,
                                                  attn_bias_type=attn_bias_type,
                                                  attn_mask_type=attn_mask_type,
                                                  scaling_factor=scaling_factor,
                                                  dropout_probability=dropout_probability,
                                                  is_causal_masking=is_causal_masking)

        output_ = xmap_runner(partial_self_fused_attn_max_512, sharding_meta.in_axes,
                              sharding_meta.out_axes[0], sharding_meta.axis_resources, inputs_)

        output = jnp.reshape(output_, sharding_meta.output_shapes[0])

    return output


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8))
def _self_fused_attn_max_512(qkv: jnp.ndarray, bias: jnp.ndarray, mask: jnp.ndarray, seed: int,
                             attn_bias_type: AttnBiasType, attn_mask_type: AttnMaskType,
                             scaling_factor: float, dropout_probability: float,
                             is_causal_masking: bool):
    output, _ = _self_fused_attn_max_512_fwd(qkv, bias, mask, seed, attn_bias_type, attn_mask_type,
                                             scaling_factor, dropout_probability, is_causal_masking)
    return output


def _self_fused_attn_max_512_fwd(qkv, bias, mask, seed, attn_bias_type, attn_mask_type,
                                 scaling_factor, dropout_probability, is_causal_masking):

    q_seqlen = jnp.sum(mask[:, :, :, 0] == 0, axis=(-1, -2), dtype=jnp.int32)
    q_cu_seqlen = jnp.cumsum(q_seqlen)
    q_cu_seqlen = jnp.hstack((0, q_cu_seqlen))

    kv_cu_seqlen = q_cu_seqlen

    output, softmax_aux = self_fused_attn_max_512_fwd(qkv, bias, q_cu_seqlen, kv_cu_seqlen, seed,
                                                      attn_bias_type.value, attn_mask_type.value,
                                                      scaling_factor, dropout_probability,
                                                      is_causal_masking)
    return output, (softmax_aux, qkv, q_cu_seqlen, kv_cu_seqlen)


def _self_fused_attn_max_512_bwd(
        seed,    # pylint: disable=unused-argument
        attn_bias_type,
        attn_mask_type,
        scaling_factor,
        dropout_probability,
        is_causal_masking,
        ctx,
        grad):
    softmax_aux, qkv, q_cu_seqlen, kv_cu_seqlen = ctx

    doutput = grad

    grad_qkv, _, grad_bias = self_fused_attn_max_512_bwd(qkv,
                                                         softmax_aux,
                                                         doutput,
                                                         q_cu_seqlen,
                                                         kv_cu_seqlen,
                                                         attn_bias_type=attn_bias_type.value,
                                                         attn_mask_type=attn_mask_type.value,
                                                         scaling_factor=scaling_factor,
                                                         dropout_probability=dropout_probability,
                                                         is_causal_masking=is_causal_masking)

    return grad_qkv, grad_bias, None


_self_fused_attn_max_512.defvjp(_self_fused_attn_max_512_fwd, _self_fused_attn_max_512_bwd)


def cross_fused_attn(q: jnp.ndarray,
                     kv: jnp.ndarray,
                     mask: jnp.ndarray,
                     seed: int,
                     attn_bias_type: AttnBiasType,
                     attn_mask_type: AttnMaskType,
                     scaling_factor: float,
                     dropout_probability: float,
                     is_causal_masking: bool,
                     sharding_type: ShardingType = ShardingType.SINGLE):
    """
    Cross multi-head attention wrapper
    """
    assert sharding_type not in (ShardingType.TP_ROW, ShardingType.DP_TP_ROW), \
        "Fused_attn_max_512 does not support row-split tensor parallelism currently."

    if sharding_type is ShardingType.SINGLE:
        output = _cross_fused_attn_max_512(q,
                                           kv,
                                           mask,
                                           seed=seed,
                                           attn_bias_type=attn_bias_type,
                                           attn_mask_type=attn_mask_type,
                                           scaling_factor=scaling_factor,
                                           dropout_probability=dropout_probability,
                                           is_causal_masking=is_causal_masking)
    else:
        dp_axis_name = "batch"
        tp_axis_name = "model"

        inputs = [q, kv, mask]
        output_shape = q.shape
        sharding_meta = get_fused_attn_sharding_meta(sharding_type, [x.shape for x in inputs],
                                                     [output_shape],
                                                     dp_dims=([0, 0, 0], [0]),
                                                     tp_dims=([2, 3, None], [2]),
                                                     dp_axis_name=dp_axis_name,
                                                     tp_axis_name=tp_axis_name)

        inputs_ = tuple(
            jnp.reshape(x, new_shape) for x, new_shape in zip(inputs, sharding_meta.input_shapes))

        partial_cross_fused_attn_max_512 = partial(_cross_fused_attn_max_512,
                                                   seed=seed,
                                                   attn_bias_type=attn_bias_type,
                                                   attn_mask_type=attn_mask_type,
                                                   scaling_factor=scaling_factor,
                                                   dropout_probability=dropout_probability,
                                                   is_causal_masking=is_causal_masking)

        output_ = xmap_runner(partial_cross_fused_attn_max_512, sharding_meta.in_axes,
                              sharding_meta.out_axes[0], sharding_meta.axis_resources, inputs_)

        output = jnp.reshape(output_, sharding_meta.output_shapes[0])

    return output


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8))
def _cross_fused_attn_max_512(q: jnp.ndarray, kv: jnp.ndarray, mask: jnp.ndarray, seed: int,
                              attn_bias_type: AttnBiasType, attn_mask_type: AttnMaskType,
                              scaling_factor: float, dropout_probability: float,
                              is_causal_masking: bool):

    output, _ = _cross_fused_attn_max_512_fwd(q, kv, mask, seed, attn_bias_type, attn_mask_type,
                                              scaling_factor, dropout_probability,
                                              is_causal_masking)
    return output


def _cross_fused_attn_max_512_fwd(q, kv, mask, seed, attn_bias_type, attn_mask_type, scaling_factor,
                                  dropout_probability, is_causal_masking):

    q_seqlen = jnp.sum(mask[:, :, :, 0] == 0, axis=(-1, -2), dtype=jnp.int32)
    q_cu_seqlen = jnp.cumsum(q_seqlen)
    q_cu_seqlen = jnp.hstack((0, q_cu_seqlen))

    kv_seqlen = jnp.sum(mask[:, :, 0, :] == 0, axis=(-1, -2), dtype=jnp.int32)
    kv_cu_seqlen = jnp.cumsum(kv_seqlen)
    kv_cu_seqlen = jnp.hstack((0, kv_cu_seqlen))

    output, softmax_aux = cross_fused_attn_max_512_fwd(q, kv, q_cu_seqlen, kv_cu_seqlen, seed,
                                                       attn_bias_type.value, attn_mask_type.value,
                                                       scaling_factor, dropout_probability,
                                                       is_causal_masking)
    return output, (softmax_aux, q, kv, q_cu_seqlen, kv_cu_seqlen)


def _cross_fused_attn_max_512_bwd(
        seed,    # pylint: disable=unused-argument
        attn_bias_type,
        attn_mask_type,
        scaling_factor,
        dropout_probability,
        is_causal_masking,
        ctx,
        grad):
    softmax_aux, q, kv, q_cu_seqlen, kv_cu_seqlen = ctx

    doutput = grad

    # TODO(rewang): remove dsoftmax for cross_fused_attn_max_512
    grad_q, grad_kv, _ = cross_fused_attn_max_512_bwd(q,
                                                      kv,
                                                      softmax_aux,
                                                      doutput,
                                                      q_cu_seqlen,
                                                      kv_cu_seqlen,
                                                      attn_bias_type=attn_bias_type.value,
                                                      attn_mask_type=attn_mask_type.value,
                                                      scaling_factor=scaling_factor,
                                                      dropout_probability=dropout_probability,
                                                      is_causal_masking=is_causal_masking)

    return grad_q, grad_kv, None


_cross_fused_attn_max_512.defvjp(_cross_fused_attn_max_512_fwd, _cross_fused_attn_max_512_bwd)

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX multi-head attention modules"""

from enum import Enum
from functools import partial
from jax.ad_checkpoint import checkpoint_name
import jax
import jax.numpy as jnp

from transformer_engine_jax import NVTE_Bias_Type
from transformer_engine_jax import NVTE_Mask_Type
from transformer_engine_jax import NVTE_QKV_Layout

from .cpp_extensions import FusedAttnHelper
from .cpp_extensions import cross_fused_attn_fwd, cross_fused_attn_bwd
from .cpp_extensions import self_fused_attn_fwd, self_fused_attn_bwd
from .cpp_extensions import fused_attn_fwd, fused_attn_bwd


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
    PADDING_CAUSAL_MASK = NVTE_Mask_Type.NVTE_PADDING_CAUSAL_MASK


class QKVLayout(Enum):
    """QKV layout"""
    BS3HD = NVTE_QKV_Layout.NVTE_BS3HD
    BSHD_BS2HD = NVTE_QKV_Layout.NVTE_BSHD_BS2HD
    BSHD_BSHD_BSHD = NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD


def canonicalize_attn_mask_type(attn_mask_type: str):
    """Convert string attn_mask_type to AttnMaskType
    TE-JAX currently fall back to the padding version kernels for the libraries integration.
    The overhead between padding and non-padding version should be small.
    However, we will lease this limitation in the near feature.
    """
    if attn_mask_type in ['causal', 'padding_causal']:
        return AttnMaskType.PADDING_CAUSAL_MASK
    if attn_mask_type in ['no_mask', 'padding']:
        return AttnMaskType.PADDING_MASK
    raise ValueError(f"Unsupported {attn_mask_type=}, "
                     "supported attn_mask_type={'no_mask', 'padding', 'causal', 'padding_causal'}")


def is_fused_attn_kernel_available(q_type, kv_type, qkv_layout, attn_bias_type, attn_mask_type,
                                   dropout_probability, num_heads_q, num_heads_kv, max_seqlen_q,
                                   max_seqlen_kv, head_dim):
    """
    To check whether the fused attention kernel is available
    """
    return FusedAttnHelper(q_type, kv_type, qkv_layout.value, attn_bias_type.value,
                           attn_mask_type.value, dropout_probability, num_heads_q, num_heads_kv,
                           max_seqlen_q, max_seqlen_kv, head_dim).is_fused_attn_kernel_available()


def self_fused_attn(qkv: jnp.ndarray, bias: jnp.ndarray, mask: jnp.ndarray, seed: jnp.ndarray,
                    attn_bias_type: AttnBiasType, attn_mask_type: AttnMaskType,
                    scaling_factor: float, dropout_probability: float, is_training: bool):
    """
    Self fused attention wrapper
    """
    output = _self_fused_attn(qkv,
                              bias,
                              mask,
                              seed,
                              attn_bias_type=attn_bias_type,
                              attn_mask_type=attn_mask_type,
                              scaling_factor=scaling_factor,
                              dropout_probability=dropout_probability,
                              is_training=is_training)

    return output


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8))
def _self_fused_attn(qkv: jnp.ndarray, bias: jnp.ndarray, mask: jnp.ndarray, seed: jnp.ndarray,
                     attn_bias_type: AttnBiasType, attn_mask_type: AttnMaskType,
                     scaling_factor: float, dropout_probability: float, is_training: bool):

    output, _ = _self_fused_attn_fwd_rule(qkv, bias, mask, seed, attn_bias_type, attn_mask_type,
                                          scaling_factor, dropout_probability, is_training)
    return output


def _self_fused_attn_fwd_rule(qkv: jnp.ndarray, bias: jnp.ndarray, mask: jnp.ndarray,
                              seed: jnp.ndarray, attn_bias_type: AttnBiasType,
                              attn_mask_type: AttnMaskType, scaling_factor: float,
                              dropout_probability: float, is_training: bool):
    if mask is None:
        batch, seqlen, *_ = qkv.shape
        actual_seqlen = jnp.full((batch,), seqlen, dtype=jnp.int32)
    else:
        mask = jnp.logical_not(mask)
        actual_seqlen = jnp.sum(mask, axis=-2, dtype=jnp.int32)[..., 0, 0]    # shape = (b,)
    output, softmax_aux, rng_state = self_fused_attn_fwd(qkv,
                                                         bias,
                                                         actual_seqlen,
                                                         seed,
                                                         attn_bias_type=attn_bias_type.value,
                                                         attn_mask_type=attn_mask_type.value,
                                                         scaling_factor=scaling_factor,
                                                         dropout_probability=dropout_probability,
                                                         is_training=is_training)
    output = checkpoint_name(output, 'context')
    softmax_aux = checkpoint_name(softmax_aux, 'context')
    rng_state = checkpoint_name(rng_state, 'context')
    return output, (qkv, bias, softmax_aux, rng_state, output, actual_seqlen)


def _self_fused_attn_bwd_rule(attn_bias_type, attn_mask_type, scaling_factor, dropout_probability,
                              is_training, ctx, dz):
    qkv, bias, softmax_aux, rng_state, output, actual_seqlen = ctx

    grad_qkv, grad_bias = self_fused_attn_bwd(qkv,
                                              bias,
                                              softmax_aux,
                                              rng_state,
                                              output,
                                              dz,
                                              actual_seqlen,
                                              attn_bias_type=attn_bias_type.value,
                                              attn_mask_type=attn_mask_type.value,
                                              scaling_factor=scaling_factor,
                                              dropout_probability=dropout_probability,
                                              is_training=is_training)

    if attn_bias_type == AttnBiasType.NO_BIAS:
        grad_bias = None

    return grad_qkv, grad_bias, None, None


_self_fused_attn.defvjp(_self_fused_attn_fwd_rule, _self_fused_attn_bwd_rule)


def cross_fused_attn(q: jnp.ndarray, kv: jnp.ndarray, bias: jnp.ndarray, mask: jnp.ndarray,
                     seed: jnp.ndarray, attn_bias_type: AttnBiasType, attn_mask_type: AttnMaskType,
                     scaling_factor: float, dropout_probability: float, is_training: bool):
    """
    Cross multi-head attention wrapper
    """

    output = _cross_fused_attn(q,
                               kv,
                               bias,
                               mask,
                               seed,
                               attn_bias_type=attn_bias_type,
                               attn_mask_type=attn_mask_type,
                               scaling_factor=scaling_factor,
                               dropout_probability=dropout_probability,
                               is_training=is_training)

    return output


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9))
def _cross_fused_attn(q: jnp.ndarray, kv: jnp.ndarray, bias: jnp.ndarray, mask: jnp.ndarray,
                      seed: jnp.ndarray, attn_bias_type: AttnBiasType, attn_mask_type: AttnMaskType,
                      scaling_factor: float, dropout_probability: float, is_training: bool):

    output, _ = _cross_fused_attn_fwd_rule(q, kv, bias, mask, seed, attn_bias_type, attn_mask_type,
                                           scaling_factor, dropout_probability, is_training)
    return output


def _cross_fused_attn_fwd_rule(q, kv, bias, mask, seed, attn_bias_type, attn_mask_type,
                               scaling_factor, dropout_probability, is_training):
    if mask is None:
        batch, s_q, *_ = q.shape
        s_kv = kv.shape[1]
        q_actual_seqlen = jnp.full((batch,), s_q, dtype=jnp.int32)
        kv_actual_seqlen = jnp.full((batch,), s_kv, dtype=jnp.int32)
    else:
        mask = jnp.logical_not(mask)
        q_actual_seqlen = jnp.sum(mask, axis=-2, dtype=jnp.int32)[..., 0, 0]    # shape = (b,)
        if attn_mask_type not in [AttnMaskType.CAUSAL_MASK, AttnMaskType.PADDING_CAUSAL_MASK]:
            kv_actual_seqlen = jnp.sum(mask, axis=-1, dtype=jnp.int32)[..., 0, 0]    # shape = (b,)
        else:
            # When mask is causal, the actual seqlen is not the last row, use max to find it
            kv_actual_seqlen = jnp.max(jnp.sum(mask, axis=-1, dtype=jnp.int32), axis=(-1, -2))

    output, softmax_aux, rng_state = cross_fused_attn_fwd(q,
                                                          kv,
                                                          bias,
                                                          q_actual_seqlen,
                                                          kv_actual_seqlen,
                                                          seed,
                                                          attn_bias_type=attn_bias_type.value,
                                                          attn_mask_type=attn_mask_type.value,
                                                          scaling_factor=scaling_factor,
                                                          dropout_probability=dropout_probability,
                                                          is_training=is_training)

    return output, (q, kv, bias, softmax_aux, rng_state, output, q_actual_seqlen, kv_actual_seqlen)


def _cross_fused_attn_bwd_rule(attn_bias_type, attn_mask_type, scaling_factor, dropout_probability,
                               is_training, ctx, dz):
    q, kv, bias, softmax_aux, rng_state, output, q_actual_seqlen, kv_actual_seqlen = ctx

    grad_q, grad_kv, grad_bias = cross_fused_attn_bwd(q,
                                                      kv,
                                                      bias,
                                                      softmax_aux,
                                                      rng_state,
                                                      output,
                                                      dz,
                                                      q_actual_seqlen,
                                                      kv_actual_seqlen,
                                                      attn_bias_type=attn_bias_type.value,
                                                      attn_mask_type=attn_mask_type.value,
                                                      scaling_factor=scaling_factor,
                                                      dropout_probability=dropout_probability,
                                                      is_training=is_training)

    if attn_bias_type == AttnBiasType.NO_BIAS:
        grad_bias = None

    return grad_q, grad_kv, grad_bias, None, None


_cross_fused_attn.defvjp(_cross_fused_attn_fwd_rule, _cross_fused_attn_bwd_rule)


def fused_attn(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, bias: jnp.ndarray, mask: jnp.ndarray,
               seed: jnp.ndarray, attn_bias_type: AttnBiasType, attn_mask_type: AttnMaskType,
               scaling_factor: float, dropout_probability: float, is_training: bool):
    """
    Dot product attention with the seperated query, key, value
    """

    output = _fused_attn(q,
                         k,
                         v,
                         bias,
                         mask,
                         seed,
                         attn_bias_type=attn_bias_type,
                         attn_mask_type=attn_mask_type,
                         scaling_factor=scaling_factor,
                         dropout_probability=dropout_probability,
                         is_training=is_training)

    return output


@partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8, 9, 10))
def _fused_attn(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, bias: jnp.ndarray,
                mask: jnp.ndarray, seed: jnp.ndarray, attn_bias_type: AttnBiasType,
                attn_mask_type: AttnMaskType, scaling_factor: float, dropout_probability: float,
                is_training: bool):

    output, _ = _fused_attn_fwd_rule(q, k, v, bias, mask, seed, attn_bias_type, attn_mask_type,
                                     scaling_factor, dropout_probability, is_training)
    return output


def _fused_attn_fwd_rule(q, k, v, bias, mask, seed, attn_bias_type, attn_mask_type, scaling_factor,
                         dropout_probability, is_training):
    if mask is None:
        batch, s_q, *_ = q.shape
        s_kv = k.shape[1]
        q_actual_seqlen = jnp.full((batch,), s_q, dtype=jnp.int32)
        kv_actual_seqlen = jnp.full((batch,), s_kv, dtype=jnp.int32)
    else:
        mask = jnp.logical_not(mask)
        q_actual_seqlen = jnp.sum(mask, axis=-2, dtype=jnp.int32)[..., 0, 0]    # shape = (b,)
        if attn_mask_type not in [AttnMaskType.CAUSAL_MASK, AttnMaskType.PADDING_CAUSAL_MASK]:
            kv_actual_seqlen = jnp.sum(mask, axis=-1, dtype=jnp.int32)[..., 0, 0]    # shape = (b,)
        else:
            # When mask is causal, the actual seqlen is not the last row, use max to find it
            kv_actual_seqlen = jnp.max(jnp.sum(mask, axis=-1, dtype=jnp.int32), axis=(-1, -2))

    output, softmax_aux, rng_state = fused_attn_fwd(q,
                                                    k,
                                                    v,
                                                    bias,
                                                    q_actual_seqlen,
                                                    kv_actual_seqlen,
                                                    seed,
                                                    attn_bias_type=attn_bias_type.value,
                                                    attn_mask_type=attn_mask_type.value,
                                                    scaling_factor=scaling_factor,
                                                    dropout_probability=dropout_probability,
                                                    is_training=is_training)

    return output, (q, k, v, bias, softmax_aux, rng_state, output, q_actual_seqlen,
                    kv_actual_seqlen)


def _fused_attn_bwd_rule(attn_bias_type, attn_mask_type, scaling_factor, dropout_probability,
                         is_training, ctx, dz):
    q, k, v, bias, softmax_aux, rng_state, output, q_actual_seqlen, kv_actual_seqlen = ctx

    grad_q, grad_k, grad_v, grad_bias = fused_attn_bwd(q,
                                                       k,
                                                       v,
                                                       bias,
                                                       softmax_aux,
                                                       rng_state,
                                                       output,
                                                       dz,
                                                       q_actual_seqlen,
                                                       kv_actual_seqlen,
                                                       attn_bias_type=attn_bias_type.value,
                                                       attn_mask_type=attn_mask_type.value,
                                                       scaling_factor=scaling_factor,
                                                       dropout_probability=dropout_probability,
                                                       is_training=is_training)

    if attn_bias_type == AttnBiasType.NO_BIAS:
        grad_bias = None

    return grad_q, grad_k, grad_v, grad_bias, None, None


_fused_attn.defvjp(_fused_attn_fwd_rule, _fused_attn_bwd_rule)

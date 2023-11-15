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
from transformer_engine_jax import NVTE_QKV_Layout

from .cpp_extensions import FusedAttnHelper
from .cpp_extensions import cross_fused_attn_fwd, cross_fused_attn_bwd
from .cpp_extensions import self_fused_attn_fwd, self_fused_attn_bwd


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


def is_fused_attn_kernel_available(q_type, kv_type, qkv_layout, attn_bias_type, attn_mask_type,
                                   dropout_probability, max_seqlen_q, max_seqlen_kv, head_dim):
    """
    To check whether the fused attention kernel is available
    """
    return FusedAttnHelper(q_type, kv_type, qkv_layout.value, attn_bias_type.value,
                           attn_mask_type.value, dropout_probability, max_seqlen_q, max_seqlen_kv,
                           head_dim).is_fused_attn_kernel_available()


def self_fused_attn(qkv: jnp.ndarray, bias: jnp.ndarray, mask: jnp.ndarray, seed: jnp.ndarray,
                    attn_bias_type: AttnBiasType, attn_mask_type: AttnMaskType,
                    scaling_factor: float, dropout_probability: float, is_training: bool):
    """
    Self fused attention wrapper
    """
    assert attn_mask_type is not AttnMaskType.NO_MASK, \
        "Currently not support AttnMaskType.NO_MASK."

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
    squeezed_mask = mask[:, :, :, 0]
    output, softmax_aux, rng_state = self_fused_attn_fwd(qkv,
                                                         bias,
                                                         squeezed_mask,
                                                         seed,
                                                         attn_bias_type=attn_bias_type.value,
                                                         attn_mask_type=attn_mask_type.value,
                                                         scaling_factor=scaling_factor,
                                                         dropout_probability=dropout_probability,
                                                         is_training=is_training)
    return output, (qkv, softmax_aux, rng_state, output, squeezed_mask)


def _self_fused_attn_bwd_rule(attn_bias_type, attn_mask_type, scaling_factor, dropout_probability,
                              is_training, ctx, dz):
    qkv, softmax_aux, rng_state, output, squeezed_mask = ctx

    grad_qkv, grad_bias = self_fused_attn_bwd(qkv,
                                              softmax_aux,
                                              rng_state,
                                              output,
                                              dz,
                                              squeezed_mask,
                                              attn_bias_type=attn_bias_type.value,
                                              attn_mask_type=attn_mask_type.value,
                                              scaling_factor=scaling_factor,
                                              dropout_probability=dropout_probability,
                                              is_training=is_training)

    if attn_bias_type == AttnBiasType.NO_BIAS:
        grad_bias = None

    return grad_qkv, grad_bias, None, None


_self_fused_attn.defvjp(_self_fused_attn_fwd_rule, _self_fused_attn_bwd_rule)


def cross_fused_attn(q: jnp.ndarray, kv: jnp.ndarray, mask: jnp.ndarray, seed: jnp.ndarray,
                     attn_bias_type: AttnBiasType, attn_mask_type: AttnMaskType,
                     scaling_factor: float, dropout_probability: float, is_training: bool):
    """
    Cross multi-head attention wrapper
    """

    output = _cross_fused_attn(q,
                               kv,
                               mask,
                               seed,
                               attn_bias_type=attn_bias_type,
                               attn_mask_type=attn_mask_type,
                               scaling_factor=scaling_factor,
                               dropout_probability=dropout_probability,
                               is_training=is_training)

    return output


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8))
def _cross_fused_attn(q: jnp.ndarray, kv: jnp.ndarray, mask: jnp.ndarray, seed: jnp.ndarray,
                      attn_bias_type: AttnBiasType, attn_mask_type: AttnMaskType,
                      scaling_factor: float, dropout_probability: float, is_training: bool):

    output, _ = _cross_fused_attn_fwd_rule(q, kv, mask, seed, attn_bias_type, attn_mask_type,
                                           scaling_factor, dropout_probability, is_training)
    return output


def _cross_fused_attn_fwd_rule(q, kv, mask, seed, attn_bias_type, attn_mask_type, scaling_factor,
                               dropout_probability, is_training):

    q_squeezed_mask = mask[:, :, :, 0]
    kv_squeezed_mask = mask[:, :, 0, :]

    output, softmax_aux = cross_fused_attn_fwd(q,
                                               kv,
                                               q_squeezed_mask,
                                               kv_squeezed_mask,
                                               seed,
                                               attn_bias_type=attn_bias_type.value,
                                               attn_mask_type=attn_mask_type.value,
                                               scaling_factor=scaling_factor,
                                               dropout_probability=dropout_probability,
                                               is_training=is_training)
    return output, (softmax_aux, q, kv, q_squeezed_mask, kv_squeezed_mask)


def _cross_fused_attn_bwd_rule(attn_bias_type, attn_mask_type, scaling_factor, dropout_probability,
                               is_training, ctx, dz):
    softmax_aux, q, kv, q_squeezed_mask, kv_squeezed_mask = ctx

    grad_q, grad_kv = cross_fused_attn_bwd(q,
                                           kv,
                                           softmax_aux,
                                           dz,
                                           q_squeezed_mask,
                                           kv_squeezed_mask,
                                           attn_bias_type=attn_bias_type.value,
                                           attn_mask_type=attn_mask_type.value,
                                           scaling_factor=scaling_factor,
                                           dropout_probability=dropout_probability,
                                           is_training=is_training)

    return grad_q, grad_kv, None, None


_cross_fused_attn.defvjp(_cross_fused_attn_fwd_rule, _cross_fused_attn_bwd_rule)

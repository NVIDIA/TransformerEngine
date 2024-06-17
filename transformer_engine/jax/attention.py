# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX multi-head attention modules"""

from enum import Enum
from functools import partial
from typing import Optional, Tuple
from jax.ad_checkpoint import checkpoint_name
import jax
import jax.numpy as jnp

from transformer_engine.transformer_engine_jax import NVTE_Bias_Type
from transformer_engine.transformer_engine_jax import NVTE_Mask_Type
from transformer_engine.transformer_engine_jax import NVTE_QKV_Layout
from transformer_engine.transformer_engine_jax import NVTE_QKV_Format
from transformer_engine.transformer_engine_jax import nvte_get_qkv_format

from . import cpp_extensions as tex


class AttnBiasType(Enum):
    """
    NO_BIAS: Softmax is performed as softmax(scale * qk)
    PRE_SCALE_BIAS: Softmax is performed as softmax(scale * (qk + bias))
    POST_SCALE_BIAS: Softmax is performed as softmax(scale * qk + bias)
    """
    NO_BIAS = NVTE_Bias_Type.NVTE_NO_BIAS
    PRE_SCALE_BIAS = NVTE_Bias_Type.NVTE_PRE_SCALE_BIAS
    POST_SCALE_BIAS = NVTE_Bias_Type.NVTE_POST_SCALE_BIAS


class AttnMaskType(Enum):
    """
    NO_MASK: No attention mask is applied.
    PADDING_MASK: Indicates the presence of paddings at the end of each sequence.
    CAUSAL_MASK: An upper triangular mask is applied to the softmax inputs.
    PADDING_CAUSAL_MASK: A combination of both causal and padding masks.
    """
    NO_MASK = NVTE_Mask_Type.NVTE_NO_MASK
    PADDING_MASK = NVTE_Mask_Type.NVTE_PADDING_MASK
    CAUSAL_MASK = NVTE_Mask_Type.NVTE_CAUSAL_MASK
    PADDING_CAUSAL_MASK = NVTE_Mask_Type.NVTE_PADDING_CAUSAL_MASK


class QKVLayout(Enum):
    """QKV layout"""
    BS3HD = NVTE_QKV_Layout.NVTE_BS3HD
    BSHD_BS2HD = NVTE_QKV_Layout.NVTE_BSHD_BS2HD
    BSHD_BSHD_BSHD = NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD
    T3HD = NVTE_QKV_Layout.NVTE_T3HD
    THD_T2HD = NVTE_QKV_Layout.NVTE_THD_T2HD
    THD_THD_THD = NVTE_QKV_Layout.NVTE_THD_THD_THD


class QKVFormat(Enum):
    """QKV format"""
    SBHD = NVTE_QKV_Format.NVTE_SBHD
    BSHD = NVTE_QKV_Format.NVTE_BSHD
    THD = NVTE_QKV_Format.NVTE_THD


def get_qkv_format(qkv_layout):
    return QKVFormat(nvte_get_qkv_format(qkv_layout.value))


def canonicalize_attn_mask_type(attn_mask_type: str):
    """Convert string attn_mask_type to AttnMaskType
    TE-JAX currently fall back to the padding version kernels for the libraries integration.
    The overhead between padding and non-padding version should be small.
    However, we will lease this limitation in the near feature.
    """
    match attn_mask_type:
        case 'no_mask':
            return AttnMaskType.NO_MASK
        case 'padding':
            return AttnMaskType.PADDING_MASK
        case 'causal':
            return AttnMaskType.CAUSAL_MASK
        case 'padding_causal' | 'causal_padding':
            return AttnMaskType.PADDING_CAUSAL_MASK
    raise ValueError(f"Unsupported {attn_mask_type=}, supported attn_mask_type="
                     "{'no_mask', 'padding', 'causal', 'padding_causal', 'causal_padding'}")


def is_fused_attn_kernel_available(q_dtype, kv_dtype, qkv_layout, attn_bias_type, attn_mask_type,
                                   dropout_probability, q_num_heads, kv_num_heads, q_max_seqlen,
                                   kv_max_seqlen, head_dim):
    """
    To check whether the fused attention kernel is supported
    """
    return tex.FusedAttnHelper(q_dtype, kv_dtype, qkv_layout.value, attn_bias_type.value,
                           attn_mask_type.value, dropout_probability, q_num_heads, kv_num_heads,
                           q_max_seqlen, kv_max_seqlen, head_dim).is_fused_attn_kernel_available()


def _obtain_batch_and_max_seqlen(qkv, qkv_layout):
    match qkv_layout:
        case QKVLayout.BS3HD | QKVLayout.T3HD:
            assert len(qkv) == 1, f"qkv must be (qkvpacked,) with {qkv_layout=}"
            batch, q_max_seqlen, *_ = qkv[0].shape
            kv_max_seqlen = q_max_seqlen
        case QKVLayout.BSHD_BS2HD | QKVLayout.THD_T2HD:
            assert len(qkv) == 2, f"qkv must be (query, kvpacked) with {qkv_layout=}"
            batch, q_max_seqlen, *_ = qkv[0].shape
            kv_max_seqlen = qkv[1].shape[1]
        case QKVLayout.BSHD_BSHD_BSHD | QKVLayout.THD_THD_THD:
            assert len(qkv) == 3, f"qkv must be (query, key, value) with {qkv_layout=}"
            batch, q_max_seqlen, *_ = qkv[0].shape
            kv_max_seqlen = qkv[1].shape[1]
        case _:
            raise ValueError(f"Unsupported {qkv_layout=}")
    return batch, q_max_seqlen, kv_max_seqlen


def fused_attn(qkv: Tuple[jnp.ndarray, ...], bias: Optional[jnp.ndarray], mask: Optional[jnp.ndarray],
               q_seq_lens: Optional[jnp.ndarray], kv_seq_lens: Optional[jnp.ndarray],
               q_seq_offsets: Optional[jnp.ndarray], kv_seq_offsets: Optional[jnp.ndarray],
               seed: jnp.ndarray, attn_bias_type: AttnBiasType, attn_mask_type: AttnMaskType,
               qkv_layout: QKVLayout, scaling_factor: float, dropout_probability: float,
               is_training: bool, max_segments_per_seq: int = 1):
    """
    Dot product attention ... (TODO): rewang
    """

    if get_qkv_format(qkv_layout) == QKVFormat.THD:
        assert mask is None, "THD format doesn't support mask, please provide the explicit " \
            "[q_seqlens, kv_seqlens, q_seq_offsets, kv_seq_offsets]."
    else:
        assert max_segments_per_seq == 1, "max_segments_per_seq should be 1 for non-THD format."

    if mask is not None:
        # convert the mask to seqlens, mask doesn't support ragged offsets
        assert all(x is None for x in [q_seq_lens, q_seq_offsets, kv_seq_lens, kv_seq_offsets])
        if attn_mask_type in [AttnMaskType.NO_MASK, AttnMaskType.CAUSAL_MASK]:
            batch, q_max_seqlen, kv_max_seqlen = _obtain_batch_and_max_seqlen(qkv, qkv_layout)
            q_seq_lens = jnp.full((batch,), q_max_seqlen, dtype=jnp.int32)
            kv_seq_lens = jnp.full((batch,), kv_max_seqlen, dtype=jnp.int32)
        else:
            assert mask is not None
            mask = jnp.logical_not(mask)
            q_seq_lens = jnp.sum(mask, axis=-2, dtype=jnp.int32)[..., 0, 0]
            if attn_mask_type == AttnMaskType.PADDING_MASK:
                kv_seq_lens = jnp.sum(mask, axis=-1, dtype=jnp.int32)[..., 0, 0]
            else:
                # When mask is causal, the actual seqlen is not the last row, use max to find it
                kv_seq_lens = jnp.max(jnp.sum(mask, axis=-1, dtype=jnp.int32), axis=(-1, -2))
    else:
        assert \
            all(x is not None for x in [q_seq_lens, q_seq_offsets, kv_seq_lens, kv_seq_offsets]), \
            "mask is None, seq_lens and seq_offsets must not be None."

    output = _fused_attn(qkv,
                         bias,
                         q_seq_lens,
                         kv_seq_lens,
                         q_seq_offsets,
                         kv_seq_offsets,
                         seed,
                         attn_bias_type=attn_bias_type,
                         attn_mask_type=attn_mask_type,
                         qkv_layout=qkv_layout,
                         scaling_factor=scaling_factor,
                         dropout_probability=dropout_probability,
                         is_training=is_training,
                         max_segments_per_seq=max_segments_per_seq)

    return output


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13))
def _fused_attn(qkv: Tuple[jnp.ndarray, ...], bias: Optional[jnp.ndarray],
                q_seq_lens: jnp.ndarray, kv_seq_lens: jnp.ndarray,
                q_seq_offsets: Optional[jnp.ndarray], kv_seq_offsets: Optional[jnp.ndarray],
                seed: jnp.ndarray, attn_bias_type: AttnBiasType, attn_mask_type: AttnMaskType,
                qkv_layout: QKVLayout, scaling_factor: float, dropout_probability: float,
                is_training: bool, max_segments_per_seq: int):
    output, _ = _fused_attn_fwd_rule(qkv, bias, q_seq_lens, kv_seq_lens, q_seq_offsets,
                                     kv_seq_offsets, seed, attn_bias_type, attn_mask_type,
                                     qkv_layout, scaling_factor, dropout_probability, is_training,
                                     max_segments_per_seq)
    return output


def _fused_attn_fwd_rule(qkv, bias, q_seq_lens, kv_seq_lens, q_seq_offsets, kv_seq_offsets,
                         seed, attn_bias_type, attn_mask_type, qkv_layout, scaling_factor,
                         dropout_probability, is_training, max_segments_per_seq):
    output, softmax_aux, rng_state = tex.fused_attn_fwd(qkv,
                                                    bias,
                                                    q_seq_lens,
                                                    kv_seq_lens,
                                                    q_seq_offsets,
                                                    kv_seq_offsets,
                                                    seed,
                                                    attn_bias_type=attn_bias_type.value,
                                                    attn_mask_type=attn_mask_type.value,
                                                    qkv_layout=qkv_layout.value,
                                                    scaling_factor=scaling_factor,
                                                    dropout_probability=dropout_probability,
                                                    is_training=is_training,
                                                    max_segments_per_seq=max_segments_per_seq)
    output = checkpoint_name(output, 'context')
    softmax_aux = checkpoint_name(softmax_aux, 'context')
    rng_state = checkpoint_name(rng_state, 'context')
    return output, (qkv, bias, q_seq_lens, kv_seq_lens, q_seq_offsets, kv_seq_offsets,
                    softmax_aux, rng_state, output)


def _fused_attn_bwd_rule(attn_bias_type, attn_mask_type, qkv_layout, scaling_factor,
                         dropout_probability, is_training, max_segments_per_seq, ctx, dz):
    qkv, bias, q_seq_lens, kv_seq_lens, q_seq_offsets, kv_seq_offsets, \
        softmax_aux, rng_state, output = ctx
    grad_qkv, grad_bias = tex.fused_attn_bwd(qkv,
                                        bias,
                                        softmax_aux,
                                        rng_state,
                                        output,
                                        dz,
                                        q_seq_lens,
                                        kv_seq_lens,
                                        q_seq_offsets,
                                        kv_seq_offsets,
                                        attn_bias_type=attn_bias_type.value,
                                        attn_mask_type=attn_mask_type.value,
                                        qkv_layout=qkv_layout.value,
                                        scaling_factor=scaling_factor,
                                        dropout_probability=dropout_probability,
                                        is_training=is_training,
                                        max_segments_per_seq=max_segments_per_seq)
    if attn_bias_type == AttnBiasType.NO_BIAS:
        grad_bias = None
    return grad_qkv, grad_bias, None, None, None, None, None


_fused_attn.defvjp(_fused_attn_fwd_rule, _fused_attn_bwd_rule)
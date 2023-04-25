# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring

from typing import Optional
import math
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax import nn as jax_nn
from jax import lax
from jax import value_and_grad, jit

from transformer_engine.jax.fmha import self_fmha, cross_fmha

# Type annotations
Array = jnp.ndarray

CASES = [(32, 512, 16, 64), (32, 128, 16, 64)]
CROSS_FMHA_CASES = [(32, 128, 512, 16, 64)]
DTYPES = [jnp.bfloat16, jnp.float16]


def make_causal_mask(x: Array, extra_batch_dims: int = 0) -> Array:
    idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
    return make_attention_mask(idxs, idxs, jnp.greater_equal, extra_batch_dims=extra_batch_dims)


def make_attention_mask(query_input, key_input, pairwise_fn=jnp.multiply, extra_batch_dims=0):
    # [batch, len_q, len_kv]
    mask = pairwise_fn(
    # [batch, len_q] -> [batch, len_q, 1]
        jnp.expand_dims(query_input, axis=-1),
    # [batch, len_q] -> [batch, 1, len_kv]
        jnp.expand_dims(key_input, axis=-2))

    # [batch, 1, len_q, len_kv]. This creates the head dim.
    mask = jnp.expand_dims(mask, axis=-3)
    mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
    return mask


def combine_biases(*masks):
    masks = [m for m in masks if m is not None]
    if not masks:
        return None
    assert all(map(lambda x: x.ndim == masks[0].ndim,
                   masks)), (f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
    mask, *other_masks = masks
    for other_mask in other_masks:
        mask = mask + other_mask
    return mask


def combine_masks(*masks: Optional[Array]):
    masks = [m for m in masks if m is not None]
    if not masks:
        return None
    assert all(map(lambda x: x.ndim == masks[0].ndim,
                   masks)), (f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
    mask, *other_masks = masks
    for other_mask in other_masks:
        mask = jnp.logical_and(mask, other_mask)
    return mask


def make_decoder_mask(decoder_target_tokens: Array,
                      decoder_causal_attention: Optional[Array] = None,
                      decoder_segment_ids: Optional[Array] = None) -> Array:
    masks = []
    # The same mask is applied to all attention heads. So the head dimension is 1,
    # i.e., the mask will be broadcast along the heads dim.
    # [batch, 1, length, length]
    causal_mask = make_causal_mask(decoder_target_tokens)

    # Positions with value 1 in `decoder_causal_attneition` can attend
    # bidirectionally.
    if decoder_causal_attention is not None:
        # [batch, 1, length, length]
        inputs_mask = make_attention_mask(decoder_causal_attention, decoder_causal_attention,
                                          jnp.logical_and)
        masks.append(jnp.logical_or(causal_mask, inputs_mask))
    else:
        masks.append(causal_mask)

    # Padding mask.
    masks.append(make_attention_mask(decoder_target_tokens > 0, decoder_target_tokens > 0))

    # Packing mask
    if decoder_segment_ids is not None:
        masks.append(make_attention_mask(decoder_segment_ids, decoder_segment_ids, jnp.equal))

    return combine_masks(*masks)


def core_attention(query,
                   key,
                   value,
                   bias=None,
                   dropout_rng=None,
                   dropout_rate=0.,
                   deterministic=False,
                   dtype=jnp.float32,
                   float32_logits=True):

    assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
    batch_dim = 0
    assert query.shape[batch_dim] == key.shape[batch_dim] == value.shape[batch_dim], (
        'q, k, v batch dims must match.')
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], ('q, k, v num_heads must match.')
    sequence_dim = 1
    assert key.shape[sequence_dim] == value.shape[sequence_dim], 'k, v lengths must match.'
    assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

    # Casting logits and softmax computation for float32 for model stability.
    if float32_logits:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)

    # `attn_weights`: [batch, num_heads, q_length, kv_length]
    attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)

    # Apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
        attn_weights = attn_weights + bias.astype(attn_weights.dtype)

    # Normalize the attention weights across `kv_length` dimension.
    attn_weights = jax_nn.softmax(attn_weights).astype(dtype)

    # Apply attention dropout.
    if not deterministic and dropout_rate > 0.:
        keep_prob = 1.0 - dropout_rate
        # T5 broadcasts along the "length" dim, but unclear which one that
        # corresponds to in positional dimensions here, assuming query dim.
        dropout_shape = list(attn_weights.shape)
        dropout_shape[-2] = 1
        keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        keep = jnp.broadcast_to(keep, attn_weights.shape)
        multiplier = (keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
        attn_weights = attn_weights * multiplier

    return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)


def jax_self_fmha(qkv, bias, q_token, kv_token, **kwargs):
    is_causal_masking = kwargs['is_causal_masking']
    if is_causal_masking:
        mask = make_decoder_mask(q_token)
    else:
        mask = make_attention_mask(q_token > 0, kv_token > 0)
    attention_bias = lax.select(mask > 0,
                                jnp.full(mask.shape, 0.).astype(qkv.dtype),
                                jnp.full(mask.shape, -1e10).astype(qkv.dtype))
    bias = combine_biases(bias, attention_bias)
    query, key, value = jnp.split(qkv, [1, 2], axis=-3)
    query = query.reshape(*query.shape[:2], *query.shape[-2:])
    key = key.reshape(*value.shape[:2], *key.shape[-2:])
    value = value.reshape(*value.shape[:2], *value.shape[-2:])
    query = query * kwargs['scaling_factor']
    output = core_attention(query,
                            key,
                            value,
                            bias=bias,
                            dropout_rate=kwargs['dropout_probability'],
                            dtype=qkv.dtype)
    return output


def jax_cross_fmha(q, kv, q_token, kv_token, **kwargs):
    is_causal_masking = kwargs['is_causal_masking']
    if is_causal_masking:
        raise NotImplementedError
    mask = make_attention_mask(q_token > 0, kv_token > 0)
    assert q.dtype == kv.dtype
    attention_bias = lax.select(mask > 0,
                                jnp.full(mask.shape, 0.).astype(q.dtype),
                                jnp.full(mask.shape, -1e10).astype(q.dtype))
    bias = attention_bias
    query = q
    key, value = jnp.split(kv, [1], axis=-3)
    key = key.reshape(*value.shape[:2], *key.shape[-2:])
    value = value.reshape(*value.shape[:2], *value.shape[-2:])
    query = query * kwargs['scaling_factor']
    output = core_attention(query,
                            key,
                            value,
                            bias=bias,
                            dropout_rate=kwargs['dropout_probability'],
                            dtype=q.dtype)
    return output


def customcall_self_fmha(qkv, bias, q_token, kv_token, **kwargs):
    is_causal_masking = kwargs['is_causal_masking']
    if is_causal_masking:
        mask = make_decoder_mask(q_token)
    else:
        mask = make_attention_mask(q_token > 0, kv_token > 0)

    # mask invert
    mask = (mask == 0)

    return self_fmha(qkv, bias, mask, **kwargs)


def customcall_cross_fmha(q, kv, q_token, kv_token, **kwargs):
    is_causal_masking = kwargs['is_causal_masking']
    if is_causal_masking:
        raise NotImplementedError
    mask = make_attention_mask(q_token > 0, kv_token > 0)

    # mask invert
    mask = (mask == 0)

    return cross_fmha(q, kv, mask, **kwargs)


class TestSelfFMHA():

    def set_input(self, b, s, h, d, dtype, is_causal_masking, pad_len):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)

        qkv_shape = (b, s, 3, h, d)
        bias_shape = (1, h, s, s)
        self.valid_len = s - pad_len

        min_val, max_val = -1, 1
        self.qkv = jax.random.uniform(subkeys[0], qkv_shape, dtype, min_val, max_val)
        self.bias = jax.random.uniform(subkeys[1], bias_shape, dtype, min_val, max_val)

        self.q_token = jnp.concatenate((jnp.ones((b, self.valid_len)), jnp.zeros((b, pad_len))),
                                       axis=-1)
        self.kv_token = self.q_token

        self.seed = 0
        self.scaling_factor = 1. / math.sqrt(d)
        self.dropout_probability = 0.
        self.is_causal_masking = is_causal_masking

    @pytest.mark.parametrize('b, s, h, d', CASES)
    @pytest.mark.parametrize('is_causal_masking', [False, True])
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_forward(self, b, s, h, d, is_causal_masking, dtype):

        self.set_input(b, s, h, d, is_causal_masking=is_causal_masking, dtype=dtype, pad_len=117)

        reference_out = jax_self_fmha(
            self.qkv,
            self.bias,
            self.q_token,
            self.kv_token,
            seed=self.seed,    # no used currently
            scaling_factor=self.scaling_factor,
            dropout_probability=self.dropout_probability,    # no used currently
            is_causal_masking=self.is_causal_masking)

        primitive_out = customcall_self_fmha(self.qkv,
                                             self.bias,
                                             self.q_token,
                                             self.kv_token,
                                             seed=self.seed,
                                             scaling_factor=self.scaling_factor,
                                             dropout_probability=self.dropout_probability,
                                             is_causal_masking=self.is_causal_masking)

        ref_valid, _ = jnp.split(reference_out, (self.valid_len,), axis=1)
        pri_valid, pri_invalid = jnp.split(primitive_out, (self.valid_len,), axis=1)

        np.testing.assert_allclose(jnp.asarray(pri_valid, np.float32),
                                   jnp.asarray(ref_valid, np.float32),
                                   rtol=1e-2,
                                   atol=5e-4)

        np.testing.assert_allclose(jnp.asarray(pri_invalid, jnp.float32),
                                   jnp.zeros_like(pri_invalid, jnp.float32))

    @pytest.mark.parametrize('b, s, h, d', CASES)
    @pytest.mark.parametrize('is_causal_masking', [False, True])
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_forward_backward(self, b, s, h, d, is_causal_masking, dtype):
        self.set_input(b,
                       s,
                       h,
                       d,
                       is_causal_masking=is_causal_masking,
                       dtype=dtype,
                       pad_len=(161 if s > 128 else 19))

        def grad_func(fmha_func, *args, **kwargs):
            # Gradient is small, use a gradient multiplier to amplify the graident
            gradient_multiplier = 1000 if dtype == jnp.bfloat16 else 10000
            if self.is_causal_masking:
                gradient_multiplier = gradient_multiplier / 10
            # Keep only valid result for the gradient
            # fmha output has shape (b, s, h, d)
            valid_fmha_ret, _ = jnp.split(fmha_func(*args, **kwargs), (self.valid_len,), axis=1)
            return (jnp.mean(valid_fmha_ret, dtype=jnp.float32) * gradient_multiplier).astype(dtype)

        kwargs = {
            'seed': self.seed,
            'scaling_factor': self.scaling_factor,
            'dropout_probability': self.dropout_probability,
            'is_causal_masking': self.is_causal_masking
        }

        # Use FP16/BF16 to sum the FMHA results may cause overflow, use FP32 for the summation
        jitted_primitive = jit(
            value_and_grad(
                lambda qkv, bias, q_token, kv_token: grad_func(
                    customcall_self_fmha, qkv, bias, q_token, kv_token, **kwargs), (0, 1)))

        jitted_reference = jit(
            value_and_grad(
                lambda qkv, bias, q_token, kv_token: grad_func(jax_self_fmha, qkv.astype(
                    jnp.float32), bias.astype(jnp.float32), q_token, kv_token, **kwargs), (0, 1)))

        primitive_out, (primitive_dqkv,
                        primitive_dbeta) = jitted_primitive(self.qkv, self.bias, self.q_token,
                                                            self.kv_token)

        reference_out, (reference_dqkv,
                        reference_dbeta) = jitted_reference(self.qkv, self.bias, self.q_token,
                                                            self.kv_token)

        np.testing.assert_allclose(jnp.asarray(primitive_out, np.float32),
                                   jnp.asarray(reference_out, np.float32),
                                   rtol=1e-4,
                                   atol=1e-5)

        valid_primitive_dqkv, invalid_primitive_dqkv = jnp.split(primitive_dqkv, (self.valid_len,),
                                                                 axis=1)
        valid_reference_dqkv, invalid_reference_dqkv = jnp.split(reference_dqkv, (self.valid_len,),
                                                                 axis=1)

        # dQ
        np.testing.assert_allclose(jnp.asarray(valid_primitive_dqkv[:, :, 0], np.float32),
                                   jnp.asarray(valid_reference_dqkv[:, :, 0], np.float32),
                                   rtol=1e-4,
                                   atol=1e-5)

        # dK
        np.testing.assert_allclose(jnp.asarray(valid_primitive_dqkv[:, :, 1], np.float32),
                                   jnp.asarray(valid_reference_dqkv[:, :, 1], np.float32),
                                   rtol=1e-4,
                                   atol=1e-5)

        # dV
        np.testing.assert_allclose(jnp.asarray(valid_primitive_dqkv[:, :, 2], np.float32),
                                   jnp.asarray(valid_reference_dqkv[:, :, 2], np.float32),
                                   rtol=1e-4,
                                   atol=1e-5)

        assert jnp.allclose(invalid_primitive_dqkv, invalid_reference_dqkv)

        # Padded part should be 0s
        assert jnp.allclose(invalid_primitive_dqkv, jnp.zeros_like(invalid_primitive_dqkv))

        # dbeta valid part
        np.testing.assert_allclose(
            jnp.asarray(primitive_dbeta[:, :, :self.valid_len, :self.valid_len], np.float32),
            jnp.asarray(reference_dbeta[:, :, :self.valid_len, :self.valid_len], np.float32),
            rtol=1e-4,
        # dbeta has a little higher diff on FP16
            atol=1.15e-5)

        # dbeta padded part
        np.testing.assert_allclose(
            jnp.asarray(primitive_dbeta[:, :, self.valid_len:, self.valid_len:], np.float32),
            jnp.asarray(reference_dbeta[:, :, self.valid_len:, self.valid_len:], np.float32))

        assert jnp.allclose(primitive_dbeta[:, :, self.valid_len:, self.valid_len:],
                            jnp.zeros_like(primitive_dbeta[:, :, self.valid_len:, self.valid_len:]))


class TestCrossFMHA():

    def set_input(self, b, s_q, s_kv, h, d, dtype, is_causal_masking, pad_len):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)

        q_shape = (b, s_q, h, d)
        kv_shape = (b, s_kv, 2, h, d)
        assert pad_len < min(s_q, s_kv)
        self.q_valid_len = s_q - pad_len
        self.kv_valid_len = s_kv - pad_len

        min_val, max_val = -1, 1
        self.q = jax.random.uniform(subkeys[0], q_shape, dtype, min_val, max_val)
        self.kv = jax.random.uniform(subkeys[1], kv_shape, dtype, min_val, max_val)

        self.q_token = jnp.concatenate((jnp.ones((b, self.q_valid_len)), jnp.zeros((b, pad_len))),
                                       axis=-1)
        self.kv_token = jnp.concatenate((jnp.ones((b, self.kv_valid_len)), jnp.zeros((b, pad_len))),
                                        axis=-1)
        self.seed = 0
        self.scaling_factor = 1. / math.sqrt(d)
        self.dropout_probability = 0.
        self.is_causal_masking = is_causal_masking

    @pytest.mark.parametrize('b, s_q, s_kv, h, d', CROSS_FMHA_CASES)
    @pytest.mark.parametrize('is_causal_masking', [False])
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_forward(self, b, s_q, s_kv, h, d, is_causal_masking, dtype):

        self.set_input(b,
                       s_q,
                       s_kv,
                       h,
                       d,
                       is_causal_masking=is_causal_masking,
                       dtype=dtype,
                       pad_len=63)

        reference_out = jax_cross_fmha(
            self.q,
            self.kv,
            self.q_token,
            self.kv_token,
            seed=self.seed,    # no used currently
            scaling_factor=self.scaling_factor,
            dropout_probability=self.dropout_probability,    # no used currently
            is_causal_masking=self.is_causal_masking)

        primitive_out = customcall_cross_fmha(self.q,
                                              self.kv,
                                              self.q_token,
                                              self.kv_token,
                                              seed=self.seed,
                                              scaling_factor=self.scaling_factor,
                                              dropout_probability=self.dropout_probability,
                                              is_causal_masking=self.is_causal_masking)

        ref_valid, _ = jnp.split(reference_out, (self.q_valid_len,), axis=1)
        pri_valid, pri_invalid = jnp.split(primitive_out, (self.q_valid_len,), axis=1)

        np.testing.assert_allclose(jnp.asarray(pri_valid, np.float32),
                                   jnp.asarray(ref_valid, np.float32),
                                   rtol=1e-2,
                                   atol=5e-4)

        np.testing.assert_allclose(jnp.asarray(pri_invalid, jnp.float32),
                                   jnp.zeros_like(pri_invalid, jnp.float32))

    @pytest.mark.parametrize('b, s_q, s_kv, h, d', CROSS_FMHA_CASES)
    @pytest.mark.parametrize('is_causal_masking', [False])
    @pytest.mark.parametrize('dtype', DTYPES)
    def test_forward_backward(self, b, s_q, s_kv, h, d, is_causal_masking, dtype):
        self.set_input(b,
                       s_q,
                       s_kv,
                       h,
                       d,
                       is_causal_masking=is_causal_masking,
                       dtype=dtype,
                       pad_len=19)

        def grad_func(fmha_func, *args, **kwargs):
            # Gradient is small, use a gradient multiplier to amplify the graident
            gradient_multiplier = 10000
            if self.is_causal_masking:
                gradient_multiplier = gradient_multiplier / 10
            # Keep only valid result for the gradient
            # fmha output has shape (b, s_q, h, d)
            valid_fmha_ret, _ = jnp.split(fmha_func(*args, **kwargs), (self.q_valid_len,), axis=1)
            return (jnp.mean(valid_fmha_ret, dtype=jnp.float32) * gradient_multiplier).astype(dtype)

        kwargs = {
            'seed': self.seed,
            'scaling_factor': self.scaling_factor,
            'dropout_probability': self.dropout_probability,
            'is_causal_masking': self.is_causal_masking
        }

        # Use FP16/BF16 to sum the FMHA results may cause overflow, use FP32 for the summation
        jitted_primitive = jit(
            value_and_grad(
                lambda q, kv, q_token, kv_token: grad_func(customcall_cross_fmha, q, kv, q_token,
                                                           kv_token, **kwargs), (0, 1)))

        jitted_reference = jit(
            value_and_grad(
                lambda q, kv, q_token, kv_token: grad_func(jax_cross_fmha, q.astype(
                    jnp.float32), kv.astype(jnp.float32), q_token, kv_token, **kwargs), (0, 1)))

        primitive_out, (primitive_dq,
                        primitive_dkv) = jitted_primitive(self.q, self.kv, self.q_token,
                                                          self.kv_token)

        reference_out, (reference_dq,
                        reference_dkv) = jitted_reference(self.q, self.kv, self.q_token,
                                                          self.kv_token)

        np.testing.assert_allclose(jnp.asarray(primitive_out, np.float32),
                                   jnp.asarray(reference_out, np.float32),
                                   rtol=1e-4,
                                   atol=1e-5)

        valid_primitive_dq, invalid_primitive_dq = jnp.split(primitive_dq, (self.q_valid_len,),
                                                             axis=1)
        valid_reference_dq, invalid_reference_dq = jnp.split(reference_dq, (self.q_valid_len,),
                                                             axis=1)

        valid_primitive_dkv, invalid_primitive_dkv = jnp.split(primitive_dkv, (self.kv_valid_len,),
                                                               axis=1)
        valid_reference_dkv, invalid_reference_dkv = jnp.split(reference_dkv, (self.kv_valid_len,),
                                                               axis=1)

        # dQ
        np.testing.assert_allclose(jnp.asarray(valid_primitive_dq, np.float32),
                                   jnp.asarray(valid_reference_dq, np.float32),
                                   rtol=1e-4,
                                   atol=1e-5)

        # dK
        np.testing.assert_allclose(jnp.asarray(valid_primitive_dkv[:, :, 0], np.float32),
                                   jnp.asarray(valid_reference_dkv[:, :, 0], np.float32),
                                   rtol=1e-4,
                                   atol=1e-5)

        # dV
        np.testing.assert_allclose(jnp.asarray(valid_primitive_dkv[:, :, 1], np.float32),
                                   jnp.asarray(valid_reference_dkv[:, :, 1], np.float32),
                                   rtol=1e-4,
                                   atol=1e-5)

        assert jnp.allclose(invalid_primitive_dq, invalid_reference_dq)
        assert jnp.allclose(invalid_primitive_dkv, invalid_reference_dkv)

        # Padded part should be 0s
        assert jnp.allclose(invalid_primitive_dq, jnp.zeros_like(invalid_primitive_dq))
        assert jnp.allclose(invalid_primitive_dkv, jnp.zeros_like(invalid_primitive_dkv))

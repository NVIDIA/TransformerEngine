# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import Optional
import math
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from jax import nn as jax_nn
from jax import lax
from jax import value_and_grad, jit

from flax.linen import combine_masks
from flax.linen import make_attention_mask
from flax.linen import make_causal_mask
from flax.linen import dot_product_attention

from transformer_engine.jax.fused_attn import self_fused_attn, cross_fused_attn
from transformer_engine.jax.fused_attn import AttnBiasType, AttnMaskType
from transformer_engine.jax.fused_attn import is_fused_attn_kernel_available

# Type annotations
Array = jnp.ndarray

SELF_CASES = [(32, 512, 16, 64), (32, 128, 16, 64)]
CROSS_CASES = [(32, 128, 512, 16, 64)]
DTYPES = [jnp.bfloat16, jnp.float16]
PAD_RATIO = [0.3]


def make_decoder_mask(tokens: Array) -> Array:
    causal_mask = make_causal_mask(tokens)
    padding_mask = make_attention_mask(tokens > 0, tokens > 0)
    return combine_masks(causal_mask, padding_mask)


def jax_self_fused_attn(qkv, bias, q_token, kv_token, dropout_rng, **kwargs):
    attn_mask_type = kwargs['attn_mask_type']
    if attn_mask_type == AttnMaskType.CAUSAL_MASK:
        mask = make_decoder_mask(q_token)
    else:
        mask = make_attention_mask(q_token > 0, kv_token > 0)

    query, key, value = jnp.split(qkv, [1, 2], axis=-3)
    query = jnp.squeeze(query)
    key = jnp.squeeze(key)
    value = jnp.squeeze(value)

    output = dot_product_attention(query,
                                   key,
                                   value,
                                   bias=bias,
                                   mask=mask,
                                   dropout_rate=kwargs['dropout_probability'],
                                   dropout_rng=dropout_rng,
                                   dtype=qkv.dtype)
    return output


def jax_cross_fused_attn(q, kv, q_token, kv_token, dropout_rng, **kwargs):
    assert q.dtype == kv.dtype

    attn_mask_type = kwargs['attn_mask_type']
    if attn_mask_type == AttnMaskType.CAUSAL_MASK:
        raise NotImplementedError
    mask = make_attention_mask(q_token > 0, kv_token > 0)

    query = q
    key, value = jnp.split(kv, [1], axis=-3)
    key = jnp.squeeze(key)
    value = jnp.squeeze(value)

    output = dot_product_attention(query,
                                   key,
                                   value,
                                   bias=None,
                                   mask=mask,
                                   dropout_rate=kwargs['dropout_probability'],
                                   dropout_rng=dropout_rng,
                                   dtype=q.dtype)
    return output


def customcall_self_fused_attn(qkv, bias, q_token, kv_token, dropout_rng, **kwargs):
    if kwargs['attn_mask_type'] == AttnMaskType.CAUSAL_MASK:
        mask = make_decoder_mask(q_token)
    else:
        mask = make_attention_mask(q_token > 0, kv_token > 0)

    # mask invert
    mask = (mask == 0)

    return self_fused_attn(qkv, bias, mask, dropout_rng, **kwargs)


def customcall_cross_fused_attn(q, kv, q_token, kv_token, dropout_rng, **kwargs):
    assert q.dtype == kv.dtype

    if kwargs['attn_mask_type'] == AttnMaskType.CAUSAL_MASK:
        raise NotImplementedError
    mask = make_attention_mask(q_token > 0, kv_token > 0)

    # mask invert
    mask = (mask == 0)

    return cross_fused_attn(q, kv, mask, dropout_rng, **kwargs)


@pytest.mark.skipif(not is_fused_attn_kernel_available(),
                    reason="Fused attention kernel is not supported.")
class TestSelfFusedAttnMax512():

    def set_input(self, b, s, h, d, dtype, attn_mask_type, pad_ratio):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)

        qkv_shape = (b, s, 3, h, d)
        bias_shape = (1, h, s, s)

        pad_len = int(s * pad_ratio)
        self.valid_len = s - pad_len

        min_val, max_val = -1, 1
        self.qkv = jax.random.uniform(subkeys[0], qkv_shape, dtype, min_val, max_val)
        self.bias = jax.random.uniform(subkeys[1], bias_shape, dtype, min_val, max_val)

        self.q_token = jnp.concatenate((jnp.ones((b, self.valid_len)), jnp.zeros((b, pad_len))),
                                       axis=-1)
        self.kv_token = self.q_token

        self.scaling_factor = 1. / math.sqrt(d)
        self.dropout_probability = 0.
        self.dropout_rng = jax.random.PRNGKey(0)
        self.attn_bias_type = AttnBiasType.POST_SCALE_BIAS
        # deterministic = not is_training
        self.deterministic = False

    @pytest.mark.parametrize('b, s, h, d', SELF_CASES)
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('attn_mask_type',
                             [AttnMaskType.PADDING_MASK, AttnMaskType.CAUSAL_MASK])
    @pytest.mark.parametrize('pad_ratio', PAD_RATIO)
    def test_forward(self, b, s, h, d, dtype, attn_mask_type, pad_ratio):

        self.set_input(b, s, h, d, dtype=dtype, attn_mask_type=attn_mask_type, pad_ratio=pad_ratio)

        primitive_out = customcall_self_fused_attn(self.qkv,
                                                   self.bias,
                                                   self.q_token,
                                                   self.kv_token,
                                                   self.dropout_rng,
                                                   attn_bias_type=self.attn_bias_type,
                                                   attn_mask_type=attn_mask_type,
                                                   scaling_factor=self.scaling_factor,
                                                   dropout_probability=self.dropout_probability,
                                                   is_training=not self.deterministic)

        reference_out = jax_self_fused_attn(self.qkv,
                                            self.bias,
                                            self.q_token,
                                            self.kv_token,
                                            self.dropout_rng,
                                            attn_mask_type=attn_mask_type,
                                            scaling_factor=self.scaling_factor,
                                            dropout_probability=self.dropout_probability)

        ref_valid, _ = jnp.split(reference_out, (self.valid_len,), axis=1)
        pri_valid, pri_invalid = jnp.split(primitive_out, (self.valid_len,), axis=1)

        np.testing.assert_allclose(jnp.asarray(pri_valid, np.float32),
                                   jnp.asarray(ref_valid, np.float32),
                                   rtol=1e-4,
                                   atol=1e-2)

        np.testing.assert_allclose(jnp.asarray(pri_invalid, jnp.float32),
                                   jnp.zeros_like(pri_invalid, jnp.float32))

    @pytest.mark.parametrize('b, s, h, d', SELF_CASES)
    @pytest.mark.parametrize('attn_mask_type',
                             [AttnMaskType.PADDING_MASK, AttnMaskType.CAUSAL_MASK])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('pad_ratio', PAD_RATIO)
    def test_forward_backward(self, b, s, h, d, dtype, attn_mask_type, pad_ratio):
        self.set_input(b, s, h, d, dtype=dtype, attn_mask_type=attn_mask_type, pad_ratio=pad_ratio)

        def grad_func(fused_attn_max_512_func, *args, **kwargs):
            # Gradient is small, use a gradient multiplier to amplify the graident
            gradient_multiplier = 1000 if dtype == jnp.bfloat16 else 10000
            if attn_mask_type == AttnMaskType.CAUSAL_MASK:
                gradient_multiplier = gradient_multiplier / 10
            # Keep only valid result for the gradient
            # fused_attn_max_512 output has shape (b, s, h, d)
            valid_fused_attn_max_512_ret, _ = jnp.split(fused_attn_max_512_func(*args, **kwargs),
                                                        (self.valid_len,),
                                                        axis=1)
            return (jnp.mean(valid_fused_attn_max_512_ret, dtype=jnp.float32) *
                    gradient_multiplier).astype(dtype)

        kwargs = {
            'attn_bias_type': self.attn_bias_type,
            'attn_mask_type': attn_mask_type,
            'scaling_factor': self.scaling_factor,
            'dropout_probability': self.dropout_probability,
            'is_training': not self.deterministic
        }

        # Use FP16/BF16 to sum the results may cause overflow, use FP32 for the summation
        jitted_primitive = jit(
            value_and_grad(
                lambda qkv, bias, q_token, kv_token, dropout_rng: grad_func(
                    customcall_self_fused_attn, qkv, bias, q_token, kv_token, dropout_rng, **kwargs
                ), (0, 1)))

        jitted_reference = jit(
            value_and_grad(
                lambda qkv, bias, q_token, kv_token, dropout_rng: grad_func(
                    jax_self_fused_attn, qkv, bias, q_token, kv_token, dropout_rng, **kwargs),
                (0, 1)))

        primitive_out, (primitive_dqkv,
                        primitive_dbeta) = jitted_primitive(self.qkv, self.bias, self.q_token,
                                                            self.kv_token, self.dropout_rng)

        reference_out, (reference_dqkv,
                        reference_dbeta) = jitted_reference(self.qkv, self.bias, self.q_token,
                                                            self.kv_token, self.dropout_rng)

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
            atol=3e-5)

        # dbeta padded part
        np.testing.assert_allclose(
            jnp.asarray(primitive_dbeta[:, :, self.valid_len:, self.valid_len:], np.float32),
            jnp.asarray(reference_dbeta[:, :, self.valid_len:, self.valid_len:], np.float32))

        assert jnp.allclose(primitive_dbeta[:, :, self.valid_len:, self.valid_len:],
                            jnp.zeros_like(primitive_dbeta[:, :, self.valid_len:, self.valid_len:]))


@pytest.mark.skipif(not is_fused_attn_kernel_available(),
                    reason="Fused attention kernel is not supported.")
class TestCrossFusedAttnMax512():

    def set_input(self, b, s_q, s_kv, h, d, dtype, attn_mask_type, pad_ratio):
        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 2)

        q_shape = (b, s_q, h, d)
        kv_shape = (b, s_kv, 2, h, d)
        q_pad_len = int(s_q * pad_ratio)
        kv_pad_len = int(s_kv * pad_ratio)
        self.q_valid_len = s_q - q_pad_len
        self.kv_valid_len = s_kv - kv_pad_len

        min_val, max_val = -1, 1
        self.q = jax.random.uniform(subkeys[0], q_shape, dtype, min_val, max_val)
        self.kv = jax.random.uniform(subkeys[1], kv_shape, dtype, min_val, max_val)

        self.q_token = jnp.concatenate((jnp.ones((b, self.q_valid_len)), jnp.zeros((b, q_pad_len))),
                                       axis=-1)
        self.kv_token = jnp.concatenate((jnp.ones((b, self.kv_valid_len)), jnp.zeros(
            (b, kv_pad_len))),
                                        axis=-1)
        self.scaling_factor = 1. / math.sqrt(d)
        self.dropout_probability = 0.
        self.dropout_rng = jax.random.PRNGKey(0)
        self.attn_bias_type = AttnBiasType.NO_BIAS
        # deterministic = not is_training
        self.deterministic = False

    @pytest.mark.parametrize('b, s_q, s_kv, h, d', CROSS_CASES)
    @pytest.mark.parametrize('attn_mask_type', [AttnMaskType.PADDING_MASK])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('pad_ratio', PAD_RATIO)
    def test_forward(self, b, s_q, s_kv, h, d, dtype, attn_mask_type, pad_ratio):

        self.set_input(b,
                       s_q,
                       s_kv,
                       h,
                       d,
                       dtype=dtype,
                       attn_mask_type=attn_mask_type,
                       pad_ratio=pad_ratio)

        primitive_out = customcall_cross_fused_attn(self.q,
                                                    self.kv,
                                                    self.q_token,
                                                    self.kv_token,
                                                    self.dropout_rng,
                                                    attn_bias_type=self.attn_bias_type,
                                                    attn_mask_type=attn_mask_type,
                                                    scaling_factor=self.scaling_factor,
                                                    dropout_probability=self.dropout_probability,
                                                    is_training=not self.deterministic)

        reference_out = jax_cross_fused_attn(self.q,
                                             self.kv,
                                             self.q_token,
                                             self.kv_token,
                                             self.dropout_rng,
                                             attn_mask_type=attn_mask_type,
                                             scaling_factor=self.scaling_factor,
                                             dropout_probability=self.dropout_probability)

        ref_valid, _ = jnp.split(reference_out, (self.q_valid_len,), axis=1)
        pri_valid, pri_invalid = jnp.split(primitive_out, (self.q_valid_len,), axis=1)

        np.testing.assert_allclose(jnp.asarray(pri_valid, np.float32),
                                   jnp.asarray(ref_valid, np.float32),
                                   rtol=1e-4,
                                   atol=2e-3)

        np.testing.assert_allclose(jnp.asarray(pri_invalid, jnp.float32),
                                   jnp.zeros_like(pri_invalid, jnp.float32))

    @pytest.mark.parametrize('b, s_q, s_kv, h, d', CROSS_CASES)
    @pytest.mark.parametrize('attn_mask_type', [AttnMaskType.PADDING_MASK])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('pad_ratio', PAD_RATIO)
    def test_forward_backward(self, b, s_q, s_kv, h, d, dtype, attn_mask_type, pad_ratio):
        self.set_input(b,
                       s_q,
                       s_kv,
                       h,
                       d,
                       dtype=dtype,
                       attn_mask_type=attn_mask_type,
                       pad_ratio=pad_ratio)

        def grad_func(fused_attn_max_512_func, *args, **kwargs):
            # Gradient is small, use a gradient multiplier to amplify the graident
            gradient_multiplier = 10000
            if attn_mask_type == AttnMaskType.CAUSAL_MASK:
                gradient_multiplier = gradient_multiplier / 10
            # Keep only valid result for the gradient
            # fused_attn_max_512 output has shape (b, s_q, h, d)
            valid_fused_attn_max_512_ret, _ = jnp.split(fused_attn_max_512_func(*args, **kwargs),
                                                        (self.q_valid_len,),
                                                        axis=1)
            return (jnp.mean(valid_fused_attn_max_512_ret, dtype=jnp.float32) *
                    gradient_multiplier).astype(dtype)

        kwargs = {
            'attn_bias_type': self.attn_bias_type,
            'attn_mask_type': attn_mask_type,
            'scaling_factor': self.scaling_factor,
            'dropout_probability': self.dropout_probability,
            'is_training': not self.deterministic
        }

        # Use FP16/BF16 to sum the results may cause overflow, use FP32 for the summation
        jitted_primitive = jit(
            value_and_grad(
                lambda q, kv, q_token, kv_token, dropout_rng: grad_func(
                    customcall_cross_fused_attn, q, kv, q_token, kv_token, dropout_rng, **kwargs),
                (0, 1)))

        jitted_reference = jit(
            value_and_grad(
                lambda q, kv, q_token, kv_token, dropout_rng: grad_func(
                    jax_cross_fused_attn, q, kv, q_token, kv_token, dropout_rng, **kwargs), (0, 1)))

        primitive_out, (primitive_dq,
                        primitive_dkv) = jitted_primitive(self.q, self.kv, self.q_token,
                                                          self.kv_token, self.dropout_rng)

        reference_out, (reference_dq,
                        reference_dkv) = jitted_reference(self.q, self.kv, self.q_token,
                                                          self.kv_token, self.dropout_rng)

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

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for fused attention"""

from dataclasses import dataclass
from functools import partial
from math import sqrt

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flax.linen import combine_masks
from flax.linen import make_attention_mask
from flax.linen.dtypes import promote_dtype
from jax import Array
from jax import value_and_grad, jit
from jax.typing import ArrayLike, DTypeLike

from transformer_engine.jax.fused_attn import AttnBiasType, AttnMaskType, QKVLayout
from transformer_engine.jax.fused_attn import self_fused_attn, cross_fused_attn, fused_attn
from transformer_engine.jax.fused_attn import is_fused_attn_kernel_available


@pytest.fixture(autouse=True, scope='function')
def clear_live_arrays():
    """
    Clear all live arrays to keep the resource clean
    """
    # Calling customcalls before jax may cause CUDA uninitialize error
    _ = jnp.zeros(0)
    yield
    for arr in jax.live_arrays():
        arr.delete()


def general_dot_product_attention(query: ArrayLike, key: ArrayLike, value: ArrayLike,
                                  bias: ArrayLike, mask: ArrayLike, deterministic: bool,
                                  dropout_rate: float, dropout_rng: ArrayLike,
                                  dtype: DTypeLike) -> Array:
    """
    Similar to flax.linen.dot_product_attention but with GQA support
    """
    query, key, value, bias = promote_dtype(query, key, value, bias, dtype=dtype)
    dtype = query.dtype

    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)

    b, s_q, h_q, d = query.shape
    _, _, h_kv, _ = key.shape
    assert (h_q % h_kv == 0) and (h_q >= h_kv)
    num_groups = h_q // h_kv
    grouped_query = jnp.reshape(query, (b, s_q, h_kv, num_groups, d))
    # logits with shape (b, h_kv, num_groups, s_q, s_kv)
    logits = jnp.einsum('...qhgd,...khd->...hgqk', grouped_query, key)

    if bias is not None:
        if bias.ndim != logits.ndim:
            bias = bias.reshape((1, *logits.shape[1:]))
        logits = logits + bias

    if mask is not None:
        if mask.ndim != logits.ndim:
            mask = jnp.expand_dims(mask, axis=-3)
        logits = jnp.where(mask, logits, jnp.finfo(dtype).min)

    softmax_out = jax.nn.softmax(logits).astype(dtype)

    if not deterministic and dropout_rate > 0.:
        keep_prob = 1.0 - dropout_rate
        keep = jax.random.bernoulli(dropout_rng, keep_prob, softmax_out.shape)
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        softmax_out = softmax_out * multiplier

    context = jnp.einsum('...hgqk,...khd->...qhgd', softmax_out, value)
    context = jnp.reshape(context, query.shape)
    return context


def is_causal_mask(mask: AttnMaskType):
    """
    Check if the mask is a causal mask
    """
    return mask in [AttnMaskType.CAUSAL_MASK, AttnMaskType.PADDING_CAUSAL_MASK]


def make_decoder_mask(q_tokens: ArrayLike, kv_tokens: ArrayLike) -> Array:
    """
    Create padded causal mask
    """
    q_idxs = jnp.broadcast_to(jnp.arange(q_tokens.shape[-1], dtype=jnp.int32), q_tokens.shape)
    kv_idxs = jnp.broadcast_to(jnp.arange(kv_tokens.shape[-1], dtype=jnp.int32), kv_tokens.shape)
    causal_mask = make_attention_mask(q_idxs, kv_idxs, jnp.greater_equal)
    padding_mask = make_attention_mask(q_tokens > 0, kv_tokens > 0)
    return combine_masks(causal_mask, padding_mask)


def jax_dpa(query, key, value, bias, q_token, kv_token, dropout_rng, **kwargs):
    """
    JAX native dot product attention implementation
    """
    attn_mask_type = kwargs['attn_mask_type']
    if is_causal_mask(attn_mask_type):
        mask = make_decoder_mask(q_token, kv_token)
    else:
        mask = make_attention_mask(q_token > 0, kv_token > 0)

    output = general_dot_product_attention(query,
                                           key,
                                           value,
                                           bias=bias,
                                           mask=mask,
                                           deterministic=not kwargs['is_training'],
                                           dropout_rate=kwargs['dropout_probability'],
                                           dropout_rng=dropout_rng,
                                           dtype=jnp.float32)
    return output.astype(query.dtype)


def customcall_fused_dpa(query, key, value, bias, q_token, kv_token, dropout_rng, **kwargs):
    """
    TE customcall dot product attention implementation
    """
    attn_mask_type = kwargs['attn_mask_type']
    if is_causal_mask(attn_mask_type):
        mask = make_decoder_mask(q_token, kv_token)
    else:
        mask = make_attention_mask(q_token > 0, kv_token > 0)

    # mask invert
    mask = jnp.logical_not(mask)

    qkv_layout = kwargs.pop('qkv_layout')
    match qkv_layout:
        case QKVLayout.BS3HD:
            query, key, value = map(partial(jnp.expand_dims, axis=-3), [query, key, value])
            qkv = jnp.concatenate((query, key, value), axis=-3)
            return self_fused_attn(qkv, bias, mask, dropout_rng, **kwargs).astype(query.dtype)
        case QKVLayout.BSHD_BS2HD:
            key, value = map(partial(jnp.expand_dims, axis=-3), [key, value])
            kv = jnp.concatenate((key, value), axis=-3)
            return cross_fused_attn(query, kv, bias, mask, dropout_rng,
                                    **kwargs).astype(query.dtype)
        case QKVLayout.BSHD_BSHD_BSHD:
            return fused_attn(query, key, value, bias, mask, dropout_rng,
                              **kwargs).astype(query.dtype)


@dataclass
class FusedAttnRunner:
    """
    Fused attention runner
    """
    batch_size: int
    max_seqlen_q: int
    max_seqlen_kv: int
    num_heads_q: int
    num_heads_kv: int
    head_dim: int
    attn_bias_type: AttnBiasType
    attn_mask_type: AttnMaskType
    dropout_prob: float
    dtype: DTypeLike
    is_training: bool
    qkv_layout: QKVLayout

    def _check_configs(self):
        if self.qkv_layout == QKVLayout.BS3HD and self.num_heads_q != self.num_heads_kv:
            pytest.skip("BS3HD layout requires num_heads_q and num_heads_kv to be equal.")

        if self.qkv_layout == QKVLayout.BS3HD and self.max_seqlen_q != self.max_seqlen_kv:
            pytest.skip("BS3HD layout requires max_seqlen_q and max_seqlen_kv to be equal.")

        if not is_fused_attn_kernel_available(
                self.dtype, self.dtype, self.qkv_layout, self.attn_bias_type, self.attn_mask_type,
                self.dropout_prob, self.num_heads_q, self.num_heads_kv, self.max_seqlen_q,
                self.max_seqlen_kv, self.head_dim):
            pytest.skip("Unsupported inputs combination or device compute capability.")

    def _setup_inputs(self):
        self._check_configs()
        key = jax.random.PRNGKey(0)
        q_key, k_key, v_key, bias_key, dropout_key = jax.random.split(key, 5)

        q_shape = (self.batch_size, self.max_seqlen_q, self.num_heads_q, self.head_dim)
        k_shape = v_shape = (self.batch_size, self.max_seqlen_kv, self.num_heads_kv, self.head_dim)
        bias_shape = (1, self.num_heads_q, self.max_seqlen_q, self.max_seqlen_kv)

        self.q = jax.random.uniform(q_key, q_shape, self.dtype, -1)
        self.k = jax.random.uniform(k_key, k_shape, self.dtype, -1)
        self.v = jax.random.uniform(v_key, v_shape, self.dtype, -1)

        with_bias = self.attn_bias_type != AttnBiasType.NO_BIAS
        self.bias = jax.random.uniform(bias_key, bias_shape, self.dtype, -1) if with_bias else None

        if self.attn_mask_type in [AttnMaskType.NO_MASK, AttnMaskType.CAUSAL_MASK]:
            pad_ratio = 0.0
        else:
            pad_ratio = 0.3

        def gen_valid(bs, max_seqlen, pad_ratio):
            pad_len = int(max_seqlen * pad_ratio)
            valid_len = max_seqlen - pad_len
            tokens = jnp.concatenate([jnp.ones((bs, valid_len)), jnp.zeros((bs, pad_len))], axis=-1)
            return valid_len, tokens

        self.valid_len_q, self.token_q = gen_valid(self.batch_size, self.max_seqlen_q, pad_ratio)
        self.valid_len_kv, self.token_kv = gen_valid(self.batch_size, self.max_seqlen_kv, pad_ratio)

        self.dropout_rng = dropout_key if self.dropout_prob > 0 else None
        self.scaling_factor = 1. / sqrt(self.head_dim)

    def test_forward(self):
        """
        Test forward without JIT
        """
        self._setup_inputs()

        args = [self.q, self.k, self.v, self.bias, self.token_q, self.token_kv, self.dropout_rng]
        kwargs = {
            'attn_bias_type': self.attn_bias_type,
            'attn_mask_type': self.attn_mask_type,
            'scaling_factor': self.scaling_factor,
            'dropout_probability': self.dropout_prob,
            'is_training': self.is_training,
            'qkv_layout': self.qkv_layout,
        }

        # Convert the outputs to float32 for the elementwise comparison
        primitive_out = customcall_fused_dpa(*args, **kwargs).astype(jnp.float32)
        reference_out = jax_dpa(*args, **kwargs).astype(jnp.float32)

        primitive_valid, primitive_invalid = jnp.split(primitive_out, (self.valid_len_q,), axis=1)
        reference_valid, _ = jnp.split(reference_out, (self.valid_len_q,), axis=1)

        # Skip elementwise comparison when dropout enabled
        if self.is_training and self.dropout_prob > 0.:
            return

        np.testing.assert_allclose(primitive_valid, reference_valid, atol=1e-2, rtol=1e-4)
        np.testing.assert_allclose(primitive_invalid, jnp.zeros_like(primitive_invalid))

    def test_backward(self):
        """
        Test value_and_grad with JIT, which includes both forward and backward
        """
        if not self.is_training:
            pytest.skip("Backward doesn't support inference")

        self._setup_inputs()

        def grad_func(func, *args, **kwargs):
            # Gradient is small, use a gradient multiplier to amplify the gradient
            gradient_multiplier = self.valid_len_q * self.num_heads_q
            if is_causal_mask(self.attn_mask_type):
                gradient_multiplier /= 10
            # Keep only valid result for the gradient
            ret_valid, _ = jnp.split(func(*args, **kwargs), (self.valid_len_q,), axis=1)
            return (jnp.mean(ret_valid, dtype=jnp.float32) * gradient_multiplier).astype(self.dtype)

        args = [self.q, self.k, self.v, self.bias, self.token_q, self.token_kv, self.dropout_rng]
        kwargs = {
            'attn_bias_type': self.attn_bias_type,
            'attn_mask_type': self.attn_mask_type,
            'scaling_factor': self.scaling_factor,
            'dropout_probability': self.dropout_prob,
            'is_training': self.is_training,
            'qkv_layout': self.qkv_layout,
        }

        # Use FP16/BF16 to sum the results may cause overflow, use FP32 for the summation
        jitted_primitive = jit(
            value_and_grad(
                lambda q, k, v, bias, *args: grad_func(customcall_fused_dpa, q, k, v, bias, *args,
                                                       **kwargs), (0, 1, 2, 3)))
        jitted_reference = jit(
            value_and_grad(
                lambda q, k, v, bias, *args: grad_func(jax_dpa, q, k, v, bias, *args, **kwargs),
                (0, 1, 2, 3)))

        primitive_out, primitive_dgrad = jitted_primitive(*args)
        reference_out, reference_dgrad = jitted_reference(*args)

        # Skip elementwise comparison when dropout enabled
        if self.dropout_prob > 0.:
            return

        np.testing.assert_allclose(primitive_out.astype(jnp.float32),
                                   reference_out.astype(jnp.float32),
                                   atol=1e-5,
                                   rtol=1e-3)

        # Convert the outputs to float32 for the elementwise comparison
        primitive_dq, primitive_dk, primitive_dv, primitive_dbias = map(
            jnp.float32, primitive_dgrad)
        reference_dq, reference_dk, reference_dv, reference_dbias = map(
            jnp.float32, reference_dgrad)

        def check_dqkv(primitive, reference, valid_len):
            primitive_valid, primitive_invalid = jnp.split(primitive, (valid_len,), axis=1)
            reference_valid, reference_invalid = jnp.split(reference, (valid_len,), axis=1)

            np.testing.assert_allclose(primitive_valid, reference_valid, atol=1e-4, rtol=1e-3)
            assert jnp.allclose(primitive_invalid, reference_invalid)
            assert jnp.allclose(primitive_invalid, jnp.zeros_like(primitive_invalid))

        check_dqkv(primitive_dq, reference_dq, self.valid_len_q)
        check_dqkv(primitive_dk, reference_dk, self.valid_len_kv)
        check_dqkv(primitive_dv, reference_dv, self.valid_len_kv)

        if self.attn_bias_type != AttnBiasType.NO_BIAS:
            # dbias valid part
            np.testing.assert_allclose(primitive_dbias[..., :self.valid_len_q, :self.valid_len_kv],
                                       reference_dbias[..., :self.valid_len_q, :self.valid_len_kv],
                                       atol=3e-5,
                                       rtol=1e-4)

            # dbias padded part
            np.testing.assert_allclose(primitive_dbias[..., self.valid_len_q:, self.valid_len_kv:],
                                       reference_dbias[..., self.valid_len_q:, self.valid_len_kv:])

            assert jnp.allclose(
                primitive_dbias[..., self.valid_len_q:, self.valid_len_kv:],
                jnp.zeros_like(primitive_dbias[..., self.valid_len_q:, self.valid_len_kv:]))


@pytest.mark.parametrize('attn_bias_type', [
    pytest.param(AttnBiasType.NO_BIAS, id='NO_BIAS'),
    pytest.param(AttnBiasType.POST_SCALE_BIAS, id='POST_SCALE_BIAS'),
])
@pytest.mark.parametrize('attn_mask_type', [
    pytest.param(AttnMaskType.NO_MASK, id='NO_MASK'),
    pytest.param(AttnMaskType.PADDING_MASK, id='PADDING'),
    pytest.param(AttnMaskType.CAUSAL_MASK, id='CAUSAL'),
    pytest.param(AttnMaskType.PADDING_CAUSAL_MASK, id='PADDING_CAUSAL'),
])
@pytest.mark.parametrize('qkv_layout', [
    pytest.param(QKVLayout.BS3HD, id='qkvpacked'),
    pytest.param(QKVLayout.BSHD_BS2HD, id='kvpacked'),
    pytest.param(QKVLayout.BSHD_BSHD_BSHD, id='separate'),
])
@pytest.mark.parametrize('dropout_prob', [0., 0.1])
@pytest.mark.parametrize('is_training',
                         [pytest.param(True, id='training'),
                          pytest.param(False, id='inference')])
@pytest.mark.parametrize(
    'dtype', [pytest.param(jnp.bfloat16, id="BF16"),
              pytest.param(jnp.float16, id="FP16")])
@pytest.mark.parametrize('b, s_q, s_kv, h_q, h_kv, d',
                         [(32, 128, 128, 16, 16, 64), (4, 2048, 2048, 12, 12, 64),
                          pytest.param(32, 512, 128, 16, 16, 64, id='32-512-128-16-16-64-cross'),
                          pytest.param(4, 2048, 2048, 12, 6, 64, id='4-2048-2048-12-6-64-GQA')])
class TestFusedAttn:
    """
    Fused attention tester
    """

    @staticmethod
    def test_forward(b, s_q, s_kv, h_q, h_kv, d, attn_bias_type, attn_mask_type, dropout_prob,
                     dtype, is_training, qkv_layout):
        """
        Test forward with parameterized configs
        """
        runner = FusedAttnRunner(b, s_q, s_kv, h_q, h_kv, d, attn_bias_type, attn_mask_type,
                                 dropout_prob, dtype, is_training, qkv_layout)
        runner.test_forward()

    @staticmethod
    def test_backward(b, s_q, s_kv, h_q, h_kv, d, attn_bias_type, attn_mask_type, dropout_prob,
                      dtype, is_training, qkv_layout):
        """
        Test backward with parameterized configs
        """
        runner = FusedAttnRunner(b, s_q, s_kv, h_q, h_kv, d, attn_bias_type, attn_mask_type,
                                 dropout_prob, dtype, is_training, qkv_layout)
        runner.test_backward()

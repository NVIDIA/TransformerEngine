# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for fused attention"""
from enum import Enum
from dataclasses import dataclass
from functools import partial
from math import sqrt

import jax
import jax.numpy as jnp
import pytest

from flax.linen import combine_masks
from flax.linen import make_attention_mask
from flax.linen.dtypes import promote_dtype
from jax import Array
from jax import value_and_grad, jit
from jax.typing import ArrayLike, DTypeLike

from transformer_engine.jax.attention import (
    AttnBiasType,
    AttnMaskType,
    QKVLayout,
    fused_attn_qkvpacked,
    fused_attn_kvpacked,
    fused_attn,
)
from transformer_engine.jax.cpp_extensions import FusedAttnHelper

from transformer_engine.transformer_engine_jax import NVTE_Fused_Attn_Backend

from utils import assert_allclose


@pytest.fixture(autouse=True, scope="module")
def init():
    """
    WAR for CUDA uninitialize error
    """
    # Calling customcalls before jax may cause CUDA uninitialize error
    _ = jnp.zeros(0)
    yield


def general_dot_product_attention(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    bias: ArrayLike,
    mask: ArrayLike,
    deterministic: bool,
    scale_factor: float,
    dropout_rate: float,
    dropout_rng: ArrayLike,
    dtype: DTypeLike,
) -> Array:
    """
    Similar to flax.linen.dot_product_attention but with GQA support
    """
    query, key, value, bias = promote_dtype(query, key, value, bias, dtype=dtype)
    dtype = query.dtype

    b, s_q, h_q, d = query.shape
    _, s_kv, h_kv, _ = key.shape
    assert (h_q % h_kv == 0) and (h_q >= h_kv)
    num_groups = h_q // h_kv
    grouped_query = jnp.reshape(query, (b, s_q, h_kv, num_groups, d))
    # logits with shape (b, h_kv, num_groups, s_q, s_kv)
    logits = scale_factor * jnp.einsum("...qhgd,...khd->...hgqk", grouped_query, key)

    if bias is not None:
        # reshape logits without groups
        logits = logits.reshape((b, h_kv * num_groups, s_q, s_kv))
        # apply post-scale bias
        logits = logits + bias
        # reshape logits back to original
        logits = logits.reshape((b, h_kv, num_groups, s_q, s_kv))

    if mask is not None:
        if mask.ndim != logits.ndim:
            mask = jnp.expand_dims(mask, axis=-3)
        logits = jnp.where(mask, jnp.finfo(dtype).min, logits)

    softmax_out = jax.nn.softmax(logits).astype(dtype)

    if not deterministic and dropout_rate > 0.0:
        keep_prob = 1.0 - dropout_rate
        keep = jax.random.bernoulli(dropout_rng, keep_prob, softmax_out.shape)
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        softmax_out = softmax_out * multiplier

    context = jnp.einsum("...hgqk,...khd->...qhgd", softmax_out, value)
    context = jnp.reshape(context, query.shape)
    return context


def is_causal_mask(mask: AttnMaskType):
    """
    Check if the mask is a causal mask
    """
    return mask in [AttnMaskType.CAUSAL_MASK, AttnMaskType.PADDING_CAUSAL_MASK]


def make_decoder_mask(q_tokens: ArrayLike, kv_tokens: ArrayLike) -> Array:
    """
    Create inverse padded causal mask where `True` means allowing the corresponding
    position to participate in attention and `False` means masking out that position.
    """
    q_idxs = jnp.broadcast_to(jnp.arange(q_tokens.shape[-1], dtype=jnp.int32), q_tokens.shape)
    kv_idxs = jnp.broadcast_to(jnp.arange(kv_tokens.shape[-1], dtype=jnp.int32), kv_tokens.shape)
    inv_causal_mask = make_attention_mask(q_idxs, kv_idxs, jnp.greater_equal)
    inv_padding_mask = make_attention_mask(q_tokens > 0, kv_tokens > 0)
    return combine_masks(inv_causal_mask, inv_padding_mask)


def make_mask(q_token: ArrayLike, kv_token: ArrayLike, attn_mask_type: AttnMaskType) -> Array:
    """
    Create attention mask based on mask type. A `True` value in the mask means
    masking out the corresponding position and a `False` value means allowing
    that position to participate in attention.
    """
    if is_causal_mask(attn_mask_type):
        inv_mask = make_decoder_mask(q_token, kv_token)
    else:
        inv_mask = make_attention_mask(q_token > 0, kv_token > 0)
    mask = jnp.logical_not(inv_mask)
    return mask


def jax_dpa(query, key, value, bias, q_token, kv_token, dropout_rng, **kwargs):
    """
    JAX native dot product attention implementation
    """
    attn_mask_type = kwargs["attn_mask_type"]
    mask = make_mask(q_token, kv_token, attn_mask_type)

    output = general_dot_product_attention(
        query,
        key,
        value,
        bias=bias,
        mask=mask,
        deterministic=not kwargs["is_training"],
        scale_factor=kwargs["scaling_factor"],
        dropout_rate=kwargs["dropout_probability"],
        dropout_rng=dropout_rng,
        dtype=jnp.float32,
    )
    return output.astype(query.dtype)


def customcall_fused_dpa(query, key, value, bias, q_token, kv_token, dropout_rng, **kwargs):
    """
    TE customcall dot product attention implementation
    """
    attn_mask_type = kwargs["attn_mask_type"]
    mask = make_mask(q_token, kv_token, attn_mask_type)

    qkv_layout = kwargs.pop("qkv_layout")
    match qkv_layout:
        case QKVLayout.BS3HD:
            query, key, value = map(partial(jnp.expand_dims, axis=-3), [query, key, value])
            qkv = jnp.concatenate((query, key, value), axis=-3)
            return fused_attn_qkvpacked(qkv, bias, mask, dropout_rng, **kwargs).astype(query.dtype)
        case QKVLayout.BSHD_BS2HD:
            key, value = map(partial(jnp.expand_dims, axis=-3), [key, value])
            kv = jnp.concatenate((key, value), axis=-3)
            return fused_attn_kvpacked(query, kv, bias, mask, dropout_rng, **kwargs).astype(
                query.dtype
            )
        case QKVLayout.BSHD_BSHD_BSHD:
            return fused_attn(query, key, value, bias, mask, dropout_rng, **kwargs).astype(
                query.dtype
            )


class BiasShape(Enum):
    """
    Enum class to represent the different bias shapes used in the fused attention.
    """

    BIAS_1HSS = "1HSS"
    BIAS_B1SS = "B1SS"
    BIAS_BHSS = "BHSS"
    BIAS_11SS = "11SS"


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
    bias_shape: BiasShape

    def _check_configs(self):
        if self.qkv_layout == QKVLayout.BS3HD and self.num_heads_q != self.num_heads_kv:
            pytest.skip("BS3HD layout requires num_heads_q and num_heads_kv to be equal.")

        if self.qkv_layout == QKVLayout.BS3HD and self.max_seqlen_q != self.max_seqlen_kv:
            pytest.skip("BS3HD layout requires max_seqlen_q and max_seqlen_kv to be equal.")

        self.backend = FusedAttnHelper(
            self.dtype,
            self.dtype,
            self.qkv_layout.value,
            self.attn_bias_type.value,
            self.attn_mask_type.value,
            self.dropout_prob,
            self.num_heads_q,
            self.num_heads_kv,
            self.max_seqlen_q,
            self.max_seqlen_kv,
            self.head_dim,
        ).get_fused_attn_backend()
        if self.backend == NVTE_Fused_Attn_Backend.NVTE_No_Backend:
            pytest.skip("Unsupported inputs combination or device compute capability.")

        if self.attn_bias_type == AttnBiasType.POST_SCALE_BIAS:
            if self.attn_mask_type not in [AttnMaskType.NO_MASK, AttnMaskType.CAUSAL_MASK]:
                pytest.skip(
                    "B1SS, BHSS and 11SS bias shapes are only supported for "
                    "AttnMaskType.NO_MASK and AttnMaskType.CAUSAL_MASK."
                )
            elif self.backend != NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen:
                pytest.skip(
                    "B1SS, BHSS and 11SS bias shapes are only supported for "
                    "the F16_arbitrary_seqlen backend."
                )

    def _setup_inputs(self):
        self._check_configs()
        key = jax.random.PRNGKey(0)
        q_key, k_key, v_key, bias_key, dropout_key = jax.random.split(key, 5)

        q_shape = (self.batch_size, self.max_seqlen_q, self.num_heads_q, self.head_dim)
        k_shape = v_shape = (self.batch_size, self.max_seqlen_kv, self.num_heads_kv, self.head_dim)

        if self.attn_bias_type == AttnBiasType.NO_BIAS:
            bias_shape = None
        elif self.bias_shape == BiasShape.BIAS_1HSS:
            bias_shape = (1, self.num_heads_q, self.max_seqlen_q, self.max_seqlen_kv)
        elif self.bias_shape == BiasShape.BIAS_B1SS:
            bias_shape = (self.batch_size, 1, self.max_seqlen_q, self.max_seqlen_kv)
        elif self.bias_shape == BiasShape.BIAS_BHSS:
            bias_shape = (self.batch_size, self.num_heads_q, self.max_seqlen_q, self.max_seqlen_kv)
        elif self.bias_shape == BiasShape.BIAS_11SS:
            bias_shape = (1, 1, self.max_seqlen_q, self.max_seqlen_kv)
        else:
            pytest.fail(f"PyTest attempted to use an unrecognized bias_layout = {self.bias_shape}!")

        self.q = jax.random.uniform(q_key, q_shape, self.dtype, -1.0)
        self.k = jax.random.uniform(k_key, k_shape, self.dtype, -1.0)
        self.v = jax.random.uniform(v_key, v_shape, self.dtype, -1.0)

        if self.attn_bias_type != AttnBiasType.NO_BIAS:
            if self.bias_shape == BiasShape.BIAS_1HSS:
                self.bias = jax.random.uniform(bias_key, bias_shape, self.dtype, -1.0)
            else:
                # [b, 1, s, s], [b, h, s, s] and [1, 1, s, s] bias shapes are workarounds for
                # an arbitrary mask where (True/False -> 0/-Inf)
                cudnn_neg_inf = -(2.0**27.0) if self.dtype == jnp.bfloat16 else -(2.0**15.0)
                self.bias = jnp.full(bias_shape, cudnn_neg_inf, dtype=self.dtype)
                max_id = min(self.max_seqlen_q, self.max_seqlen_kv)
                seq_id_size = max_id * 5 // 128  # 5 ids per interval of 128 sequences
                seq_id = jax.random.randint(bias_key, (int(seq_id_size),), 0, max_id).tolist()
                for i in range(1, len(seq_id)):
                    self.bias = self.bias.at[
                        :, :, seq_id[i - 1] : seq_id[i], seq_id[i - 1] : seq_id[i]
                    ].set(0.0)
        else:
            self.bias = None

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
        self.scaling_factor = 1.0 / sqrt(self.head_dim)

    def test_forward(self):
        """
        Test forward without JIT
        """
        self._setup_inputs()

        args = [self.q, self.k, self.v, self.bias, self.token_q, self.token_kv, self.dropout_rng]
        kwargs = {
            "attn_bias_type": self.attn_bias_type,
            "attn_mask_type": self.attn_mask_type,
            "scaling_factor": self.scaling_factor,
            "dropout_probability": self.dropout_prob,
            "is_training": self.is_training,
            "qkv_layout": self.qkv_layout,
        }

        # Convert the outputs to float32 for the elementwise comparison
        primitive_out = customcall_fused_dpa(*args, **kwargs).astype(jnp.float32)
        reference_out = jax_dpa(*args, **kwargs).astype(jnp.float32)

        if self.is_training and self.dropout_prob > 0.0:
            return

        primitive_valid, primitive_invalid = jnp.split(primitive_out, (self.valid_len_q,), axis=1)
        reference_valid, _ = jnp.split(reference_out, (self.valid_len_q,), axis=1)

        assert_allclose(primitive_invalid, jnp.zeros_like(primitive_invalid), dtype=self.dtype)
        assert_allclose(primitive_valid, reference_valid, dtype=self.dtype)

    def test_backward(self):
        """
        Test value_and_grad with JIT, which includes both forward and backward
        """

        self._setup_inputs()
        if self.attn_bias_type != AttnBiasType.NO_BIAS and self.bias_shape != BiasShape.BIAS_1HSS:
            pytest.skip("Bias gradient calculation is only supported for 1HSS bias shape.")

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
            "attn_bias_type": self.attn_bias_type,
            "attn_mask_type": self.attn_mask_type,
            "scaling_factor": self.scaling_factor,
            "dropout_probability": self.dropout_prob,
            "is_training": self.is_training,
            "qkv_layout": self.qkv_layout,
        }

        # We can compute dBias only for the [1, h, s, s] layout
        arg_nums = (0, 1, 2, 3) if self.bias_shape == BiasShape.BIAS_1HSS else (0, 1, 2)

        # Use FP16/BF16 to sum the results may cause overflow, use FP32 for the summation
        jitted_primitive = jit(
            value_and_grad(
                lambda q, k, v, bias, *args: grad_func(
                    customcall_fused_dpa, q, k, v, bias, *args, **kwargs
                ),
                arg_nums,
            )
        )
        jitted_reference = jit(
            value_and_grad(
                lambda q, k, v, bias, *args: grad_func(jax_dpa, q, k, v, bias, *args, **kwargs),
                arg_nums,
            )
        )

        primitive_out, primitive_dgrad = jitted_primitive(*args)
        reference_out, reference_dgrad = jitted_reference(*args)

        # Skip elementwise comparison when dropout enabled
        if self.dropout_prob > 0.0:
            return

        assert_allclose(
            primitive_out.astype(jnp.float32), reference_out.astype(jnp.float32), dtype=self.dtype
        )

        def check_dqkv(primitive, reference, valid_len):
            primitive_valid, primitive_invalid = jnp.split(primitive, (valid_len,), axis=1)
            reference_valid, reference_invalid = jnp.split(reference, (valid_len,), axis=1)

            assert_allclose(primitive_invalid, jnp.zeros_like(primitive_invalid), dtype=self.dtype)
            assert_allclose(primitive_invalid, reference_invalid, dtype=self.dtype)
            assert_allclose(primitive_valid, reference_valid, dtype=self.dtype)

        # Convert the outputs to float32 for the elementwise comparison
        primitive_dq, primitive_dk, primitive_dv = map(jnp.float32, primitive_dgrad[:3])
        reference_dq, reference_dk, reference_dv = map(jnp.float32, reference_dgrad[:3])

        check_dqkv(primitive_dq, reference_dq, self.valid_len_q)
        check_dqkv(primitive_dk, reference_dk, self.valid_len_kv)
        check_dqkv(primitive_dv, reference_dv, self.valid_len_kv)

        if self.attn_bias_type != AttnBiasType.NO_BIAS and self.bias_shape == BiasShape.BIAS_1HSS:
            primitive_dbias = jnp.float32(primitive_dgrad[3])
            reference_dbias = jnp.float32(reference_dgrad[3])

            assert_allclose(
                primitive_dbias[..., self.valid_len_q :, self.valid_len_kv :],
                jnp.zeros_like(primitive_dbias[..., self.valid_len_q :, self.valid_len_kv :]),
                dtype=self.dtype,
            )

            # dbias padded part
            assert_allclose(
                primitive_dbias[..., self.valid_len_q :, self.valid_len_kv :],
                reference_dbias[..., self.valid_len_q :, self.valid_len_kv :],
                dtype=self.dtype,
            )

            # dbias valid part
            assert_allclose(
                primitive_dbias[..., : self.valid_len_q, : self.valid_len_kv],
                reference_dbias[..., : self.valid_len_q, : self.valid_len_kv],
                dtype=self.dtype,
            )


@pytest.mark.parametrize(
    "attn_bias_type, bias_shape",
    [
        pytest.param(AttnBiasType.NO_BIAS, None, id="NO_BIAS"),
        pytest.param(AttnBiasType.POST_SCALE_BIAS, BiasShape.BIAS_1HSS, id="POST_SCALE_BIAS-1HSS"),
        pytest.param(AttnBiasType.POST_SCALE_BIAS, BiasShape.BIAS_B1SS, id="POST_SCALE_BIAS-B1SS"),
        pytest.param(AttnBiasType.POST_SCALE_BIAS, BiasShape.BIAS_BHSS, id="POST_SCALE_BIAS-BHSS"),
        pytest.param(AttnBiasType.POST_SCALE_BIAS, BiasShape.BIAS_11SS, id="POST_SCALE_BIAS-11SS"),
    ],
)
@pytest.mark.parametrize(
    "attn_mask_type",
    [
        pytest.param(AttnMaskType.NO_MASK, id="NO_MASK"),
        pytest.param(AttnMaskType.PADDING_MASK, id="PADDING"),
        pytest.param(AttnMaskType.CAUSAL_MASK, id="CAUSAL"),
        pytest.param(AttnMaskType.PADDING_CAUSAL_MASK, id="PADDING_CAUSAL"),
    ],
)
@pytest.mark.parametrize(
    "qkv_layout",
    [
        pytest.param(QKVLayout.BS3HD, id="QKV_PACKED"),
        pytest.param(QKVLayout.BSHD_BS2HD, id="KV_PACKED"),
        pytest.param(QKVLayout.BSHD_BSHD_BSHD, id="SEPARATE"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(jnp.bfloat16, id="BF16"),
        pytest.param(jnp.float16, id="FP16"),
    ],
)
@pytest.mark.parametrize(
    "b, s_q, s_kv, h_q, h_kv, d",
    [
        pytest.param(32, 128, 128, 16, 16, 64, id="32-128-128-16-16-64-SELF"),
        pytest.param(4, 2048, 2048, 12, 12, 64, id="4-2048-2048-12-12-64-SELF"),
        pytest.param(32, 512, 128, 16, 16, 64, id="32-512-128-16-16-64-CROSS"),
        pytest.param(4, 2048, 1024, 12, 12, 64, id="4-2048-1048-12-12-64-CROSS"),
        pytest.param(32, 128, 128, 16, 8, 64, id="32-128-128-16-8-64-GQA"),
        pytest.param(4, 2048, 2048, 12, 6, 64, id="4-2048-2048-12-6-64-GQA"),
    ],
)
@pytest.mark.parametrize(
    "dropout_prob",
    [
        pytest.param(0.0, id="DROP_0.0"),
        pytest.param(0.1, id="DROP_0.1"),
    ],
)
class TestFusedAttn:
    """
    Fused attention tester
    """

    @staticmethod
    @pytest.mark.parametrize(
        "is_training",
        [
            pytest.param(True, id="TRAINING"),
            pytest.param(False, id="INFERENCE"),
        ],
    )
    def test_forward(
        b,
        s_q,
        s_kv,
        h_q,
        h_kv,
        d,
        attn_bias_type,
        attn_mask_type,
        dropout_prob,
        dtype,
        is_training,
        qkv_layout,
        bias_shape,
    ):
        """
        Test forward with parameterized configs
        """
        runner = FusedAttnRunner(
            b,
            s_q,
            s_kv,
            h_q,
            h_kv,
            d,
            attn_bias_type,
            attn_mask_type,
            dropout_prob,
            dtype,
            is_training,
            qkv_layout,
            bias_shape,
        )
        runner.test_forward()

    @staticmethod
    def test_backward(
        b,
        s_q,
        s_kv,
        h_q,
        h_kv,
        d,
        attn_bias_type,
        attn_mask_type,
        dropout_prob,
        dtype,
        qkv_layout,
        bias_shape,
    ):
        """
        Test backward with parameterized configs
        """
        runner = FusedAttnRunner(
            b,
            s_q,
            s_kv,
            h_q,
            h_kv,
            d,
            attn_bias_type,
            attn_mask_type,
            dropout_prob,
            dtype,
            True,
            qkv_layout,
            bias_shape,
        )
        runner.test_backward()

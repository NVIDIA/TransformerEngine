# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for fused attention"""
from enum import Enum
from dataclasses import dataclass
from functools import partial
from math import sqrt
from typing import Tuple, Optional

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

from transformer_engine.jax.attention import (
    AttnBiasType,
    AttnMaskType,
    QKVLayout,
    QKVFormat,
    fused_attn,
    fused_attn_thd,
    get_qkv_format,
    make_swa_mask,
)
from transformer_engine.jax.cpp_extensions import FusedAttnHelper
from transformer_engine.transformer_engine_jax import (
    NVTE_Fused_Attn_Backend,
    get_cudnn_version,
)

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


def make_causal_mask(q_tokens: ArrayLike, kv_tokens: ArrayLike) -> Array:
    """
    Create inverse padded causal mask where `True` means allowing the corresponding
    position to participate in attention and `False` means masking out that position.
    """
    q_idxs = jnp.broadcast_to(jnp.arange(q_tokens.shape[-1], dtype=jnp.int32), q_tokens.shape)
    kv_idxs = jnp.broadcast_to(jnp.arange(kv_tokens.shape[-1], dtype=jnp.int32), kv_tokens.shape)
    inv_causal_mask = make_attention_mask(q_idxs, kv_idxs, jnp.greater_equal)
    return inv_causal_mask


def make_mask(
    q_token: ArrayLike,
    kv_token: ArrayLike,
    segment_pad_q: ArrayLike,
    segment_pad_kv: ArrayLike,
    attn_mask_type: AttnMaskType,
    window_size: Optional[Tuple[int, int]] = None,
) -> Array:
    """
    Create attention mask based on mask type. A `True` value in the mask means
    masking out the corresponding position and a `False` value means allowing
    that position to participate in attention.
    """
    inv_mask = make_attention_mask(
        q_token, kv_token, lambda x, y: (jnp.logical_and(jnp.equal(x, y), x != 0))
    )
    if is_causal_mask(attn_mask_type):
        inv_causal_mask = make_causal_mask(q_token, kv_token)
        inv_mask = combine_masks(inv_causal_mask, inv_mask)
    if segment_pad_q is not None and segment_pad_kv is not None:
        inv_pad_mask = make_attention_mask(
            segment_pad_q, segment_pad_kv, lambda x, y: jnp.logical_and(x != 1, y != 1)
        )
        inv_mask = combine_masks(inv_pad_mask, inv_mask)

    if window_size is not None:
        max_seqlen_q = inv_mask.shape[-2]
        max_seqlen_kv = inv_mask.shape[-1]
        inv_swa_mask = make_swa_mask(max_seqlen_q, max_seqlen_kv, window_size, attn_mask_type)
        inv_swa_mask = jnp.broadcast_to(inv_swa_mask, inv_mask.shape)
        # In inv_swa_mask and inv_mask 0 is masked out
        inv_mask = jnp.where(inv_mask != 0, inv_swa_mask, inv_mask)

    mask = jnp.logical_not(inv_mask)
    return mask


def get_seqlens_and_offsets(segment_ids, segment_pad):
    batch, max_seqlen = segment_ids.shape
    bincount_vmap = jax.vmap(partial(jnp.bincount, length=max_seqlen))
    seqlens_with_zero = bincount_vmap(segment_ids.astype(jnp.int32))
    seqlens = seqlens_with_zero[..., 1:]

    def _find_offsets(x):
        same_as_previous = jnp.logical_and(x[..., 1:] != x[..., :-1], x[..., 1:] != 0)
        first_column = jnp.ones((x.shape[0], 1), dtype=bool)
        same_as_previous = jnp.hstack((first_column, same_as_previous))
        return jax.vmap(partial(jnp.argwhere, size=x.shape[1], fill_value=-1))(
            same_as_previous
        ).squeeze(-1)

    offsets = _find_offsets(segment_ids)
    offsets = jnp.insert(offsets, -1, values=-1, axis=-1)
    if segment_pad is not None:
        segment_id_with_paddings = jnp.where(segment_pad, 0, segment_ids)
        padding_aware_seqlen = bincount_vmap(segment_id_with_paddings)
        output = jnp.insert(padding_aware_seqlen[..., 1:], -1, values=0, axis=-1)
    else:
        output = jnp.insert(seqlens, -1, values=0, axis=-1)
    return output, offsets


@jax.jit
def _split_valid_and_invalid(primitive, reference, pad):
    """Use JIT to speed up the verifications"""
    primitive_valid = jnp.where(pad[..., jnp.newaxis, jnp.newaxis], 0, primitive)
    primitive_invalid = jnp.where(pad[..., jnp.newaxis, jnp.newaxis], primitive, 0)
    reference_valid = jnp.where(pad[..., jnp.newaxis, jnp.newaxis], 0, reference)
    reference_invalid = jnp.where(pad[..., jnp.newaxis, jnp.newaxis], reference, 0)
    return primitive_valid, primitive_invalid, reference_valid, reference_invalid


def jax_dpa(query, key, value, bias, mask, dropout_rng, **kwargs):
    """
    JAX native dot product attention implementation
    """
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


def customcall_fused_dpa(
    query,
    key,
    value,
    bias,
    mask,
    seqlens_q,
    seqlens_kv,
    offsets_q,
    offsets_kv,
    dropout_rng,
    **kwargs,
):
    """
    TE customcall dot product attention implementation
    """
    qkv_layout = kwargs["qkv_layout"]
    is_thd = get_qkv_format(qkv_layout) == QKVFormat.THD
    match qkv_layout:
        case QKVLayout.BS3HD | QKVLayout.T3HD:
            query, key, value = map(partial(jnp.expand_dims, axis=-3), [query, key, value])
            qkv = jnp.concatenate((query, key, value), axis=-3)
            qkv_args = (qkv,)
        case QKVLayout.BSHD_BS2HD | QKVLayout.THD_T2HD:
            key, value = map(partial(jnp.expand_dims, axis=-3), [key, value])
            kv = jnp.concatenate((key, value), axis=-3)
            qkv_args = (query, kv)
        case QKVLayout.BSHD_BSHD_BSHD | QKVLayout.THD_THD_THD:
            qkv_args = (query, key, value)
        case _:
            raise ValueError(f"Unsupported {qkv_layout=}")
    if not is_thd:
        kwargs.pop("max_segments_per_seq")
        return fused_attn(qkv_args, bias, mask, dropout_rng, **kwargs).astype(query.dtype)
    return fused_attn_thd(
        qkv_args,
        bias,
        seqlens_q,
        seqlens_kv,
        offsets_q,
        offsets_kv,
        dropout_rng,
        **kwargs,
    ).astype(query.dtype)


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
    window_size: Optional[Tuple[int, int]] = None

    # See https://docs.nvidia.com/deeplearning/cudnn/latest/release-notes.html#cudnn-9-4-0 for known issue
    # generating zero-length ragged tensors. This setting adjusts the test to avoid the zero-length cases.
    def _get_max_segments_per_sequence(self):
        if 90400 <= get_cudnn_version() < 90500:
            return self.num_segments_per_seq
        else:
            # +1 for testing runtime_segments < max_segments
            return self.num_segments_per_seq + 1

    def _check_configs(self):
        # TODO(rewang): probably adds this in is_fused_attn_available
        if get_qkv_format(self.qkv_layout) == QKVFormat.THD and not self.attn_mask_type in [
            AttnMaskType.PADDING_MASK,
            AttnMaskType.PADDING_CAUSAL_MASK,
        ]:
            pytest.skip("THD format requires padding masks.")

        qkv_format = get_qkv_format(self.qkv_layout)
        if self.qkv_layout == QKVLayout.BS3HD or qkv_format == QKVFormat.THD:
            if self.max_seqlen_q != self.max_seqlen_kv:
                pytest.skip(f"{self.qkv_layout} requires max_seqlen_q == max_seqlen_kv")

        if self.qkv_layout == QKVLayout.BS3HD or self.qkv_layout == QKVLayout.T3HD:
            if self.num_heads_q != self.num_heads_kv:
                pytest.skip(f"{self.qkv_layout} requires num_heads_q == num_heads_kv")

        if self.max_seqlen_q > self.max_seqlen_kv and self.window_size is not None:
            pytest.skip(
                "seqlen_q > seqlen_kv is not supported with sliding window attention in cuDNN"
            )

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
            (-1, -1) if self.window_size is None else self.window_size,
        ).get_fused_attn_backend()
        if self.backend == NVTE_Fused_Attn_Backend.NVTE_No_Backend:
            pytest.skip("Unsupported inputs combination or device compute capability.")

        if (
            self.attn_bias_type == AttnBiasType.POST_SCALE_BIAS
            and self.bias_shape != BiasShape.BIAS_1HSS
        ):
            if self.attn_mask_type not in [
                AttnMaskType.NO_MASK,
                AttnMaskType.CAUSAL_MASK,
            ]:
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
        k_shape = v_shape = (
            self.batch_size,
            self.max_seqlen_kv,
            self.num_heads_kv,
            self.head_dim,
        )

        if self.attn_bias_type == AttnBiasType.NO_BIAS:
            bias_shape = None
        elif self.bias_shape == BiasShape.BIAS_1HSS:
            bias_shape = (1, self.num_heads_q, self.max_seqlen_q, self.max_seqlen_kv)
        elif self.bias_shape == BiasShape.BIAS_B1SS:
            bias_shape = (self.batch_size, 1, self.max_seqlen_q, self.max_seqlen_kv)
        elif self.bias_shape == BiasShape.BIAS_BHSS:
            bias_shape = (
                self.batch_size,
                self.num_heads_q,
                self.max_seqlen_q,
                self.max_seqlen_kv,
            )
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
            return tokens, jnp.logical_not(tokens)

        def generate_random_segment_ids(
            batch_size, sequence_length, num_segments, seed, with_segment_pad=True
        ):
            rng = np.random.default_rng(seed=seed)
            # [1, 1, 1, 2, 2, 3, 3, 3, 3, 0, 0], 0 means pad
            segment_ids = np.zeros((batch_size, sequence_length), dtype=int)
            # [0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1], 1 means pad
            segment_pad = np.zeros((batch_size, sequence_length), dtype=int)

            # Not include paddings
            max_segment_size = sequence_length // num_segments
            for i in range(batch_size):
                current_pos = 0
                segment_id = 1

                for _ in range(num_segments):
                    segment_size = rng.integers(1, max_segment_size + 1)
                    if current_pos + segment_size > sequence_length:
                        break
                    segment_end = current_pos + segment_size
                    segment_ids[i, current_pos:segment_end] = segment_id
                    if with_segment_pad:
                        num_valid = rng.integers(1, segment_size + 1)
                        segment_pad[i, current_pos + num_valid : segment_end] = 1
                    current_pos = segment_end
                    segment_id += 1
                segment_pad[i, current_pos:sequence_length] = 1
            return segment_ids, segment_pad

        if get_qkv_format(self.qkv_layout) == QKVFormat.THD:
            self.num_segments_per_seq = 2
            self.token_q, self.segment_pad_q = generate_random_segment_ids(
                self.batch_size, self.max_seqlen_q, self.num_segments_per_seq, seed=42
            )
            # TODO(rewang): Check if qkvpacked supported different q/kv
            # TODO(rewang): Causal with different q/kv segment_id fails
            if self.qkv_layout == QKVLayout.T3HD or is_causal_mask(self.attn_mask_type):
                self.token_kv = self.token_q
                self.segment_pad_kv = self.segment_pad_q
            else:
                self.token_kv, self.segment_pad_kv = generate_random_segment_ids(
                    self.batch_size,
                    self.max_seqlen_kv,
                    self.num_segments_per_seq,
                    seed=2024,
                )
            self.pad_q = self.segment_pad_q
            self.pad_kv = self.segment_pad_kv
        else:
            self.num_segments_per_seq = 1
            self.token_q, self.pad_q = gen_valid(self.batch_size, self.max_seqlen_q, pad_ratio)
            self.token_kv, self.pad_kv = gen_valid(self.batch_size, self.max_seqlen_kv, pad_ratio)
            self.segment_pad_q = self.segment_pad_kv = None

        self.mask = make_mask(
            self.token_q,
            self.token_kv,
            self.segment_pad_q,
            self.segment_pad_kv,
            self.attn_mask_type,
            self.window_size,
        )

        if get_qkv_format(self.qkv_layout) == QKVFormat.THD:
            self.seqlens_q, self.offsets_q = get_seqlens_and_offsets(
                self.token_q, self.segment_pad_q
            )
            self.seqlens_kv, self.offsets_kv = get_seqlens_and_offsets(
                self.token_kv, self.segment_pad_kv
            )
            self.mask_for_customcall = None  # THD format doesn't support mask
        else:
            self.seqlens_q = self.seqlens_kv = self.offsets_q = self.offsets_kv = None
            self.mask_for_customcall = self.mask

        self.dropout_rng = dropout_key if self.dropout_prob > 0 else None
        self.scaling_factor = 1.0 / sqrt(self.head_dim)

    def test_forward(self):
        """
        Test forward without JIT
        """
        self._setup_inputs()

        args = [self.q, self.k, self.v, self.bias, self.mask, self.dropout_rng]
        customcall_args = [
            self.q,
            self.k,
            self.v,
            self.bias,
            self.mask_for_customcall,
            self.seqlens_q,
            self.seqlens_kv,
            self.offsets_q,
            self.offsets_kv,
            self.dropout_rng,
        ]
        kwargs = {
            "attn_bias_type": self.attn_bias_type,
            "attn_mask_type": self.attn_mask_type,
            "scaling_factor": self.scaling_factor,
            "dropout_probability": self.dropout_prob,
            "is_training": self.is_training,
            "qkv_layout": self.qkv_layout,
            "max_segments_per_seq": self._get_max_segments_per_sequence(),
            "window_size": self.window_size,
        }

        # Convert the outputs to float32 for the elementwise comparison
        primitive_out = customcall_fused_dpa(*customcall_args, **kwargs)
        reference_out = jax_dpa(*args, **kwargs)

        if self.is_training and self.dropout_prob > 0.0:
            return

        primitive_valid, primitive_invalid, reference_valid, reference_invalid = (
            _split_valid_and_invalid(primitive_out, reference_out, self.pad_q)
        )

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
            gradient_multiplier = self.max_seqlen_q * self.num_heads_q
            if is_causal_mask(self.attn_mask_type):
                gradient_multiplier /= 10
            # Keep only valid result for the gradient
            ret_valid = jnp.where(
                self.pad_q[..., jnp.newaxis, jnp.newaxis], 0, func(*args, **kwargs)
            )
            return (jnp.mean(ret_valid, dtype=jnp.float32) * gradient_multiplier).astype(self.dtype)

        args = [self.q, self.k, self.v, self.bias, self.mask, self.dropout_rng]
        customcall_args = [
            self.q,
            self.k,
            self.v,
            self.bias,
            self.mask_for_customcall,
            self.seqlens_q,
            self.seqlens_kv,
            self.offsets_q,
            self.offsets_kv,
            self.dropout_rng,
        ]
        kwargs = {
            "attn_bias_type": self.attn_bias_type,
            "attn_mask_type": self.attn_mask_type,
            "scaling_factor": self.scaling_factor,
            "dropout_probability": self.dropout_prob,
            "is_training": self.is_training,
            "qkv_layout": self.qkv_layout,
            "max_segments_per_seq": self._get_max_segments_per_sequence(),
            "window_size": self.window_size,
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

        primitive_out, primitive_dgrad = jitted_primitive(*customcall_args)
        reference_out, reference_dgrad = jitted_reference(*args)

        # Skip elementwise comparison when dropout enabled
        if self.dropout_prob > 0.0:
            return

        assert_allclose(primitive_out, reference_out, dtype=self.dtype)

        def check_dqkv(primitive, reference, pad):
            primitive_valid, primitive_invalid, reference_valid, reference_invalid = (
                _split_valid_and_invalid(primitive, reference, pad)
            )

            assert_allclose(primitive_invalid, jnp.zeros_like(primitive_invalid), dtype=self.dtype)
            assert_allclose(primitive_invalid, reference_invalid, dtype=self.dtype)
            assert_allclose(primitive_valid, reference_valid, dtype=self.dtype)

        primitive_dq, primitive_dk, primitive_dv = primitive_dgrad[:3]
        reference_dq, reference_dk, reference_dv = reference_dgrad[:3]

        check_dqkv(primitive_dq, reference_dq, self.pad_q)
        check_dqkv(primitive_dk, reference_dk, self.pad_kv)
        check_dqkv(primitive_dv, reference_dv, self.pad_kv)

        if self.attn_bias_type != AttnBiasType.NO_BIAS and self.bias_shape == BiasShape.BIAS_1HSS:
            primitive_dbias = primitive_dgrad[3]
            reference_dbias = reference_dgrad[3]

            # Assume all batch has the same actual_seqlen, probably needs to extend the tests
            bias_mask = self.mask[0, 0]

            # Assert all masked dbias are 0s
            assert_allclose(
                jnp.where(bias_mask, primitive_dbias, 0),
                jnp.zeros_like(primitive_dbias),
                dtype=self.dtype,
            )

            # dbias padded part
            assert_allclose(
                jnp.where(bias_mask, primitive_dbias, 0),
                jnp.where(bias_mask, reference_dbias, 0),
                dtype=self.dtype,
            )

            # dbias valid part
            assert_allclose(
                jnp.where(bias_mask, 0, primitive_dbias),
                jnp.where(bias_mask, 0, reference_dbias),
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
        pytest.param(QKVLayout.T3HD, id="RAGGED_QKV_PACKED"),
        pytest.param(QKVLayout.THD_T2HD, id="RAGGED_KV_PACKED"),
        pytest.param(QKVLayout.THD_THD_THD, id="RAGGED_SEPARATE"),
    ],
)
@pytest.mark.parametrize(
    "b, s_q, s_kv, h_q, h_kv, d, dtype",
    [
        pytest.param(4, 128, 128, 16, 16, 64, jnp.bfloat16, id="4-128-128-16-16-64-BF16-SELF"),
        pytest.param(4, 128, 128, 16, 16, 64, jnp.float16, id="4-128-128-16-16-64-FP16-SELF"),
        pytest.param(2, 2048, 2048, 12, 12, 64, jnp.bfloat16, id="2-2048-2048-12-12-64-BF16-SELF"),
        pytest.param(4, 128, 256, 16, 16, 64, jnp.bfloat16, id="4-128-256-16-16-64-BF16-CROSS"),
        pytest.param(
            2,
            2048,
            1024,
            12,
            12,
            64,
            jnp.bfloat16,
            id="2-2048-1024-12-12-64-BF16-CROSS",
        ),
        pytest.param(4, 128, 128, 16, 8, 64, jnp.bfloat16, id="4-128-128-16-8-64-BF16-GQA"),
        pytest.param(2, 2048, 2048, 12, 6, 64, jnp.bfloat16, id="2-2048-2048-12-6-64-BF16-GQA"),
    ],
)
@pytest.mark.parametrize(
    "dropout_prob",
    [
        pytest.param(0.0, id="DROP_0.0"),
        pytest.param(0.1, id="DROP_0.1"),
    ],
)
@pytest.mark.parametrize(
    "swa",
    [
        pytest.param(False, id="NO_SWA"),
        pytest.param(True, id="SWA"),
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
    def _test_forward(
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
        swa,
    ):
        """
        Test forward with parameterized configs
        This test is not intended to run automatically during CI as it is time-consuming
        It is kept for development and debugging
        """
        window_size = None
        if swa:
            window_size = (s_kv // 10, 0)
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
            window_size,
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
        swa,
    ):
        """
        Test backward with parameterized configs
        """
        window_size = None
        if swa:
            window_size = (s_kv // 10, 0)
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
            window_size,
        )
        runner.test_backward()

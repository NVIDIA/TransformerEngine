# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Wrapper module for Transformer related layers with FP8 support.
"""
import functools
from enum import Enum
from typing import Any, Callable, Tuple, Sequence, Union, Optional
import numpy as np

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning

import jax.numpy as jnp
from jax import nn as jax_nn
from jax import random as jax_random
from jax import lax, vmap

from .module import DenseGeneral, LayerNormDenseGeneral, LayerNormMlpBlock

PRNGKey = Any
Shape = Tuple[int, ...]
DType = jnp.dtype
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]
Initializer = Callable[[PRNGKey, Shape, DType], Array]


def _generate_drop_path_shape(shape: Sequence[int],
                              batch_dim: int) -> Sequence[int]:
    # Generate broadcast dims for drop_path.
    drop_path_shape = list(range(0, len(shape)))
    drop_path_shape.pop(batch_dim)
    return drop_path_shape


def combine_masks(*masks: Optional[Array], dtype: DType = jnp.float32):
    """Combine attention masks.

  Args:
    *masks: set of attention mask arguments to combine, some can be None.
    dtype: final mask dtype

  Returns:
    Combined mask, reduced by logical and, returns None if no masks given.
  """
    masks = [m for m in masks if m is not None]
    if not masks:
        return None
    assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
        f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
    mask, *other_masks = masks
    for other_mask in other_masks:
        mask = jnp.logical_and(mask, other_mask)
    return mask.astype(dtype)


def combine_biases(*masks: Optional[Array]):
    """Combine attention biases.

  Args:
    *masks: set of attention bias arguments to combine, some can be None.

  Returns:
    Combined mask, reduced by summation, returns None if no masks given.
  """
    masks = [m for m in masks if m is not None]
    if not masks:
        return None
    assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
        f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
    mask, *other_masks = masks
    for other_mask in other_masks:
        mask = mask + other_mask
    return mask


def core_attention(query: Array,
                   key: Array,
                   value: Array,
                   transpose_batch_sequence: bool,
                   bias: Optional[Array] = None,
                   dropout_rng: Optional[PRNGKey] = None,
                   dropout_rate: float = 0.,
                   deterministic: bool = False,
                   dtype: DType = jnp.float32,
                   float32_logits: bool = False):
    """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Args:
    query: queries for calculating attention with shape of `[batch, q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch, kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch, kv_length,
      num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch, num_heads, q_length, kv_length]` This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.

  Returns:
    Output of shape `[batch, length, num_heads, v_depth_per_head]`.
  """
    assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
    batch_dim = 1 if transpose_batch_sequence else 0
    assert query.shape[batch_dim] == key.shape[batch_dim] == value.shape[
        batch_dim], ('q, k, v batch dims must match.')
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
        'q, k, v num_heads must match.')
    sequence_dim = 0 if transpose_batch_sequence else 1
    assert key.shape[sequence_dim] == value.shape[
        sequence_dim], 'k, v lengths must match.'
    assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

    # Casting logits and softmax computation for float32 for model stability.
    if float32_logits:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)

    # `attn_weights`: [batch, num_heads, q_length, kv_length]
    if transpose_batch_sequence:
        attn_weights = jnp.einsum('qbhd,kbhd->bhqk', query, key)
    else:
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
        keep = jax_random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        keep = jnp.broadcast_to(keep, attn_weights.shape)
        multiplier = (keep.astype(attn_weights.dtype) /
                      jnp.asarray(keep_prob, dtype=dtype))
        attn_weights = attn_weights * multiplier

    # Take the linear combination of `value`.
    if transpose_batch_sequence:
        return jnp.einsum('bhqk,kbhd->qbhd', attn_weights, value)

    return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)


dynamic_vector_slice_in_dim = vmap(lax.dynamic_slice_in_dim,
                                   in_axes=(None, 0, None, None))


class MultiHeadAttention(nn.Module):
    """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      head_dim: dimension of each head.
      dtype: the dtype of the computation.
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
  """
    head_dim: int
    num_heads: int
    dropout_rate: float = 0.
    layernorm_epsilon: float = 1e-6
    kernel_init: Initializer = nn.initializers.variance_scaling(
        1.0, 'fan_in', 'normal')
    use_bias: bool = False
    bias_init: Initializer = nn.initializers.zeros
    scale_attn_logits: bool = False
    float32_logits: bool = False  # computes logits in float32 for stability.
    scaled_query_init: bool = True
    apply_residual_connection_post_layernorm: bool = False
    output_layernorm: bool = False
    dtype: DType = jnp.float32
    fuse_qkv: bool = True
    transpose_batch_sequence: bool = True

    @nn.compact
    def __call__(self,
                 inputs_q: Array,
                 inputs_kv: Array,
                 mask: Optional[Array] = None,
                 bias: Optional[Array] = None,
                 *,
                 decode: bool = False,
                 deterministic: bool = False) -> Array:
        """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    There are two modes: decoding and non-decoding (e.g., training). The mode is
    determined by `decode` argument. For decoding, this method is called twice,
    first to initialize the cache and then for an actual decoding process. The
    two calls are differentiated by the presence of 'cached_key' in the variable
    dict. In the cache initialization stage, the cache variables are initialized
    as zeros and will be filled in the subsequent decoding process.

    In the cache initialization call, `inputs_q` has a shape [batch, length,
    q_features] and `inputs_kv`: [batch, length, kv_features]. During the
    incremental decoding stage, query, key and value all have the shape [batch,
    1, qkv_features] corresponding to a single step.

    Args:
      inputs_q: input queries of shape `[batch, q_length, q_features]`.
      inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
      mask: attention mask of shape `[batch, num_heads, q_length, kv_length]`.
      bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
      decode: Whether to prepare and use an autoregressive cache.
      deterministic: Disables dropout if set to True.

    Returns:
      output of shape `[batch, length, q_features]`.
    """
        # NOTE: T5 does not explicitly rescale the attention logits by
        #       1/sqrt(depth_kq)!  This is folded into the initializers of the
        #       linear transformations, which is equivalent under Adafactor
        depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)

        def query_init(*args):
            return self.kernel_init(
                *args) / (depth_scaling if self.scaled_query_init else 1.0)

        # Project inputs_q to multi-headed q/k/v
        # dimensions are then [batch, length, num_heads, head_dim]

        def qkv_init(key, shape, dtype):
            assert shape[-1] % 3 == 0

            q_shape = (shape[0], shape[1] // 3)
            k_shape = (shape[0], shape[1] // 3)
            v_shape = (shape[0], shape[1] // 3)

            q_kernel = query_init(key, q_shape, dtype)
            k_kernel = self.kernel_init(key, k_shape, dtype)  # pylint: disable=too-many-function-args
            v_kernel = self.kernel_init(key, v_shape, dtype)  # pylint: disable=too-many-function-args

            return jnp.concatenate([q_kernel, k_kernel, v_kernel],
                                   axis=-1,
                                   dtype=dtype)

        residual = inputs_q
        if self.fuse_qkv:
            if inputs_q is inputs_kv:
                qkv_proj, ln_out = LayerNormDenseGeneral(
                    enable_layernorm=not self.output_layernorm,
                    epsilon=self.layernorm_epsilon,
                    axis=-1,
                    features=self.num_heads * self.head_dim * 3,
                    transpose_batch_sequence=self.transpose_batch_sequence,
                    return_layernorm_output=self.
                    apply_residual_connection_post_layernorm,
                    scale_axes=('embed', ),
                    kernel_axes=('embed', 'joined_kv'),
                    kernel_init=qkv_init,
                    use_bias=self.use_bias,
                    bias_init=self.bias_init,
                    name='qkv',
                    dtype=self.dtype)(inputs_q)
                query, key, value = jnp.split(qkv_proj, [
                    self.num_heads * self.head_dim,
                    self.num_heads * self.head_dim * 2
                ],
                                              axis=-1)
                if self.scale_attn_logits:
                    query = query / depth_scaling
            else:
                query, ln_out = LayerNormDenseGeneral(
                    enable_layernorm=not self.output_layernorm,
                    epsilon=self.layernorm_epsilon,
                    axis=-1,
                    features=(self.num_heads, self.head_dim),
                    transpose_batch_sequence=self.transpose_batch_sequence,
                    return_layernorm_output=self.
                    apply_residual_connection_post_layernorm,
                    depth_scaling=depth_scaling
                    if self.scale_attn_logits else None,
                    scale_axes=('embed', ),
                    kernel_axes=('embed', 'joined_kv'),
                    use_bias=self.use_bias,
                    bias_init=self.bias_init,
                    dtype=self.dtype,
                    kernel_init=query_init,
                    name='query')(inputs_q)
                kv_proj = DenseGeneral(
                    axis=-1,
                    features=self.num_heads * self.head_dim * 2,
                    transpose_batch_sequence=self.transpose_batch_sequence,
                    kernel_axes=('embed', 'joined_kv'),
                    kernel_init=self.kernel_init,
                    use_bias=self.use_bias,
                    bias_init=self.bias_init,
                    name='kv',
                    dtype=self.dtype)(inputs_kv)
                key, value = jnp.split(kv_proj,
                                       [self.num_heads * self.head_dim],
                                       axis=-1)
        else:
            kv_projection = functools.partial(
                DenseGeneral,
                axis=-1,
                features=self.num_heads * self.head_dim,
                transpose_batch_sequence=self.transpose_batch_sequence,
                kernel_axes=('embed', 'joined_kv'),
                use_bias=self.use_bias,
                bias_init=self.bias_init,
                dtype=self.dtype)
            query, ln_out = LayerNormDenseGeneral(
                enable_layernorm=not self.output_layernorm,
                epsilon=self.layernorm_epsilon,
                axis=-1,
                features=(self.num_heads, self.head_dim),
                transpose_batch_sequence=self.transpose_batch_sequence,
                return_layernorm_output=True,
                depth_scaling=depth_scaling
                if self.scale_attn_logits else None,
                scale_axes=('embed', ),
                kernel_axes=('embed', 'joined_kv'),
                use_bias=self.use_bias,
                bias_init=self.bias_init,
                dtype=self.dtype,
                kernel_init=query_init,
                name='query')(inputs_q)

            if inputs_q is inputs_kv:
                assert ln_out is not None
                inputs_kv = ln_out

            key = kv_projection(kernel_init=self.kernel_init,
                                name='key')(inputs_kv)
            value = kv_projection(kernel_init=self.kernel_init,
                                  name='value')(inputs_kv)

        query = query.reshape(
            (query.shape[0], query.shape[1], self.num_heads, self.head_dim))
        key = key.reshape(
            (key.shape[0], key.shape[1], self.num_heads, self.head_dim))
        value = value.reshape(
            (value.shape[0], value.shape[1], self.num_heads, self.head_dim))

        if self.apply_residual_connection_post_layernorm:
            assert ln_out is not None
            residual = ln_out

        if self.transpose_batch_sequence:
            query = nn_partitioning.with_sharding_constraint(
                query, ('length', 'batch', 'heads', 'kv'))
            key = nn_partitioning.with_sharding_constraint(
                key, ('length', 'batch', 'heads', 'kv'))
            value = nn_partitioning.with_sharding_constraint(
                value, ('length', 'batch', 'heads', 'kv'))
        else:
            query = nn_partitioning.with_sharding_constraint(
                query, ('batch', 'length', 'heads', 'kv'))
            key = nn_partitioning.with_sharding_constraint(
                key, ('batch', 'length', 'heads', 'kv'))
            value = nn_partitioning.with_sharding_constraint(
                value, ('batch', 'length', 'heads', 'kv'))

        if decode:
            # Detect if we're initializing by absence of existing cache data.
            is_initialized = self.has_variable('cache', 'cached_key')

            # The key and value have dimension [batch, length, num_heads, head_dim],
            # but we cache them as [batch, num_heads, head_dim, length] as a TPU
            # fusion optimization. This also enables the "scatter via one-hot
            # broadcast" trick, which means we do a one-hot broadcast instead of a
            # scatter/gather operations, resulting in a 3-4x speedup in practice.
            def swap_dims(x):
                return x[:-3] + tuple(x[i] for i in [-2, -1, -3])

            cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                                       swap_dims(key.shape), key.dtype)
            cached_value = self.variable('cache', 'cached_value', jnp.zeros,
                                         swap_dims(value.shape), value.dtype)
            cache_index = self.variable('cache', 'cache_index',
                                        lambda: jnp.array(0, dtype=jnp.int32))
            if is_initialized:
                batch, num_heads, head_dim, length = (cached_key.value.shape)
                # During fast autoregressive decoding, we feed one position at a time,
                # and cache the keys and values step by step.
                # Sanity shape check of cached key against input query.
                expected_shape = (batch, 1, num_heads, head_dim)
                if expected_shape != query.shape:
                    raise ValueError(
                        'Autoregressive cache shape error, '
                        f"expected query shape {expected_shape} instead got {query.shape}."
                    )

                # Create a OHE of the current index. NOTE: the index is increased below.
                cur_index = cache_index.value
                one_hot_indices = jax_nn.one_hot(cur_index,
                                                 length,
                                                 dtype=key.dtype)
                # In order to update the key, value caches with the current key and
                # value, we move the length axis to the back, similar to what we did for
                # the cached ones above.
                # Note these are currently the key and value of a single position, since
                # we feed one position at a time.
                one_token_key = jnp.moveaxis(key, -3, -1)
                one_token_value = jnp.moveaxis(value, -3, -1)
                # Update key, value caches with our new 1d spatial slices.
                # We implement an efficient scatter into the cache via one-hot
                # broadcast and addition.
                key = cached_key.value + one_token_key * one_hot_indices
                value = cached_value.value + one_token_value * one_hot_indices
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1
                # Move the keys and values back to their original shapes.
                key = jnp.moveaxis(key, -1, -3)
                value = jnp.moveaxis(value, -1, -3)

                # Causal mask for cached decoder self-attention: our single query
                # position should only attend to those key positions that have already
                # been generated and cached, not the remaining zero elements.
                mask = combine_masks(
                    mask,
                    jnp.broadcast_to(
                        jnp.arange(length) <= cur_index,
                        # (1, 1, length) represent (head dim, query length, key length)
                        # query length is 1 because during decoding we deal with one
                        # index.
                        # The same mask is applied to all batch elements and heads.
                        (batch, 1, 1, length)))

                # Grab the correct relative attention bias during decoding. This is
                # only required during single step decoding.
                if bias is not None:
                    # The bias is a full attention matrix, but during decoding we only
                    # have to take a slice of it.
                    # This is equivalent to bias[..., cur_index:cur_index+1, :].
                    bias = dynamic_vector_slice_in_dim(
                        jnp.squeeze(bias, axis=0),
                        jnp.reshape(cur_index, (-1)), 1, -2)

        # Convert the boolean attention mask to an attention bias.
        if mask is not None:
            # attention mask in the form of attention bias
            attention_bias = lax.select(
                mask > 0,
                jnp.full(mask.shape, 0.).astype(self.dtype),
                jnp.full(mask.shape, -1e10).astype(self.dtype))
        else:
            attention_bias = None

        # Add provided bias term (e.g. relative position embedding).
        if bias is not None:
            attention_bias = combine_biases(attention_bias, bias)

        dropout_rng = None
        if not deterministic and self.dropout_rate > 0.:
            dropout_rng = self.make_rng('dropout')

        # Apply attention.
        x = core_attention(
            query,
            key,
            value,
            transpose_batch_sequence=self.transpose_batch_sequence,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            dtype=self.dtype,
            float32_logits=self.float32_logits)

        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))

        if self.transpose_batch_sequence:
            x = nn_partitioning.with_sharding_constraint(
                x, ('length', 'batch', 'joined_kv'))
        else:
            x = nn_partitioning.with_sharding_constraint(
                x, ('batch', 'length', 'joined_kv'))

        # Back to the original inputs dimensions.
        out = DenseGeneral(
            features=inputs_q.shape[-1],  # output dim is set to the input dim.
            transpose_batch_sequence=self.transpose_batch_sequence,
            axis=-1,
            kernel_init=self.kernel_init,
            kernel_axes=('joined_kv', 'embed'),
            use_bias=self.use_bias,
            bias_init=self.bias_init,
            dtype=self.dtype,
            name='out')(x)
        return out, residual


class LayerNorm(nn.Module):
    """T5 Layer normalization operating on the last axis of the input data."""
    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    scale_init: Initializer = nn.initializers.ones

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies layer normalization on the input."""
        x = jnp.asarray(x, jnp.float32)
        features = x.shape[-1]
        mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
        y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
        scale = nn_partitioning.param_with_axes('scale',
                                                self.scale_init, (features, ),
                                                jnp.float32,
                                                axes=('embed', ))

        scale = jnp.asarray(scale, self.dtype)
        return y * scale


class RelativePositionBiases(nn.Module):
    """Adds T5-style relative positional embeddings to the attention logits.

  Attributes:
    num_buckets: Number of buckets to bucket distances between key and query
      positions into.
    max_distance: Maximum distance before everything is lumped into the last
      distance bucket.
    num_heads: Number of heads in the attention layer. Each head will get a
      different relative position weighting.
    dtype: Type of arrays through this module.
    embedding_init: initializer for relative embedding table.
  """
    num_buckets: int
    max_distance: int
    num_heads: int
    dtype: Any
    embedding_init: Callable[..., Array] = nn.linear.default_embed_init

    @staticmethod
    def _relative_position_bucket(relative_position,
                                  bidirectional=True,
                                  num_buckets=32,
                                  max_distance=128):
        """Translate relative position to a bucket number for relative attention.

    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position.  If bidirectional=False, then positive relative positions are
    invalid.
    We use smaller buckets for small absolute relative_position and larger
    buckets for larger absolute relative_positions.  All relative
    positions >=max_distance  map to the same bucket.  All relative
    positions <=-max_distance map to the same bucket.  This should allow for
    more graceful generalization to longer sequences than the model has been
    trained on.

    Args:
      relative_position: an int32 array
      bidirectional: a boolean - whether the attention is bidirectional
      num_buckets: an integer
      max_distance: an integer

    Returns:
      a Tensor with the same shape as relative_position, containing int32
        values in the range [0, num_buckets)
    """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).astype(np.int32) * num_buckets
            n = np.abs(n)
        else:
            n = np.maximum(n, 0)
        # now n is in the range [0, inf)
        max_exact = num_buckets // 2
        is_small = (n < max_exact)
        val_if_large = max_exact + (np.log(
            n.astype(np.float32) / max_exact + np.finfo(np.float32).eps) /
                                    np.log(max_distance / max_exact) *
                                    (num_buckets - max_exact)).astype(np.int32)
        val_if_large = np.minimum(val_if_large, num_buckets - 1)
        ret += np.where(is_small, n, val_if_large)
        return ret

    @nn.compact
    def __call__(self, qlen, klen, bidirectional=True):
        """Produce relative position embedding attention biases.

    Args:
      qlen: attention query length.
      klen: attention key length.
      bidirectional: whether to allow positive memory-query relative position
        embeddings.

    Returns:
      output: `(1, len, q_len, k_len)` attention bias
    """
        context_position = np.arange(qlen, dtype=jnp.int32)[:, None]
        memory_position = np.arange(klen, dtype=jnp.int32)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=bidirectional,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance)
        relative_attention_bias = nn_partitioning.param_with_axes(
            'rel_embedding',
            self.embedding_init, (self.num_heads, self.num_buckets),
            jnp.float32,
            axes=('heads', 'relpos_buckets'))

        relative_attention_bias = jnp.asarray(relative_attention_bias,
                                              self.dtype)
        # Instead of using a slow gather, we create a leading-dimension one-hot
        # array from rp_bucket and use it to perform the gather-equivalent via a
        # contraction, i.e.:
        # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
        # This is equivalent to relative_attention_bias[:, rp_bucket]
        bcast_iota = lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1),
                                          0)
        rp_bucket_one_hot = jnp.array(rp_bucket[jnp.newaxis,
                                                ...] == bcast_iota,
                                      dtype=self.dtype)
        # --> shape (qlen, klen, num_heads)
        values = lax.dot_general(
            relative_attention_bias,
            rp_bucket_one_hot,
            (
                ((1, ), (0, )),  # rhs, lhs contracting dims
                ((), ())))  # no batched dims
        # Add a singleton batch dimension.
        # --> shape (1, num_heads, qlen, klen)
        return values[jnp.newaxis, ...]


class TransformerLayerType(Enum):
    """TransformerLayerType."""
    ENCODER = "encoder"
    DECODER = "decoder"


class TransformerLayer(nn.Module):
    """
    TransformerLayer is made up of a relative embedding
    an attention block and a feedforward network (MLP).

    Parameters
    ----------
    hidden_size: int = 512
        size of each input sample.
    mlp_hidden_size: int = 2048
        intermediate size to which input samples are projected.
    num_attention_heads: int = 8
        number of attention heads in the transformer layer.
    layernorm_epsilon: float = 1e-6
        a value added to the denominator of layer normalization
        for numerical stability.
    hidden_dropout: float = 0.1
        dropout probability for the dropout op after FC2 layer.
    attention_dropout: float = 0.1
        dropout probability for the dropout op during multi-head attention.
    mha_kernel_init: Initializer =
        flax.linen.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        used for initializing weights of QKV and Output projection weights.
    mlp_kernel_init: Initializer =
        flax.linen.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
        used for initializing weights of FC1 and FC2 layers.
    mlp_activations: Sequence[str] = ('relu', )
        the sequence of activation functions to apply after FC1. Each activation has
        its own FC1 layer.
    use_bias: bool = `False`
        Whether to enable bias shifting for QKVO projections, FC1 and FC2.
    bias_init: Initializer = nn.initializers.zeros
        used for initializing bias of QKVO projections, FC1 and FC2. Only works
        when `use_bias`=`True`.
    apply_residual_connection_post_layernorm: bool = `False`
        if set to `True`, residual connections are taken from the output
        of layer norm (default is taken from input of layer norm)
    output_layernorm: bool = `False`
        if set to `True`, layer normalization is applied on the output side,
        after the final dropout-add. default behavior is to apply layer
        normalization on the input side, before the QKV transformation.
    float32_attention_logits: bool = `False`
        if set to `True`, attention logits are executed in jax.numpy.float32.
    layer_type: TransformerLayerType = TransformerLayerType.ENCODER
        if set to TransformerLayerType.DECODER, an additional cross-attention block
        is added after self-attention.This can be used for structures like `T5`
        Transformer in conjunction with the TransformerLayerType.ENCODER option.
    enable_relative_embedding: bool = `False`
        Whether to enable relative embedding as shifting of attention logits.
    relative_embedding: flax.linen.Module = None
        The module for relative embedding execution, only works when
        `enable_relative_embedding` is `True`. Default is None, which will create
        an instance of RelativePositionBiases if `enable_relative_embedding` is `True`.
        Default instance of RelativePositionBiases:
            RelativePositionBiases(
                num_buckets=32,
                max_distance=128,
                num_heads=self.num_attention_heads,
                dtype=self.dtype,
                embedding_init=
                    flax.linen.initializers.variance_scaling(1.0, 'fan_avg', 'uniform'),
                name='relpos_bias')
    Optimization parameters
    -----------------------
    dtype: Any = jnp.float32
        controls the type used to allocate the initial parameters.
    drop_path: float = 0.0
        when > 0.0, applies stochastic depth per sample in the main
        path of the residual block.
    fuse_qkv_params: bool = `True`
        if set to `True`, `TransformerLayer` module exposes a single fused
        parameter for query-key-value for self-attention and key-value for
        cross-attention.
    transpose_batch_sequence: bool = `True`
        Whether to switch axis of batch and sequence length dimension.
        if set to `True`, then transpose inputs from (batch, seqlen, hidden)
        to (seqlen, batch, hidden)
    scale_attn_logits: bool = `False`
        Whether to scale attention logits -> (Q/sqrt(dk))K instead of QK.
    scaled_query_init: bool = `True`
        Whether to scale query on init by sqrt(dk)
    """
    hidden_size: int = 512
    mlp_hidden_size: int = 2048
    num_attention_heads: int = 8
    layernorm_epsilon: float = 1e-6
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    mha_kernel_init: Initializer = nn.initializers.variance_scaling(
        1.0, 'fan_in', 'normal')
    mlp_kernel_init: Initializer = nn.initializers.variance_scaling(
        1.0, 'fan_in', 'truncated_normal')
    mlp_activations: Sequence[str] = ('relu', )
    use_bias: bool = False
    bias_init: Initializer = nn.initializers.zeros
    apply_residual_connection_post_layernorm: bool = False
    output_layernorm: bool = False
    float32_attention_logits: bool = False
    layer_type: TransformerLayerType = TransformerLayerType.ENCODER
    enable_relative_embedding: bool = True
    relative_embedding: nn.Module = None
    dtype: Any = jnp.float32
    drop_path: float = 0.0
    fuse_qkv_params: bool = True
    transpose_batch_sequence: bool = True
    scale_attn_logits: bool = False
    scaled_query_init: bool = True

    @nn.compact
    def __call__(self,
                 inputs: Array,
                 encoded: Array = None,
                 attention_mask: Array = None,
                 encoder_decoder_mask: Array = None,
                 deterministic: bool = False,
                 decode: bool = False,
                 max_decode_length: bool = None):
        """
        The caller of TransformerLayer for forward of given relative embedding
        , attention block and feedforward network (MLP).

        Parameters
        ----------
        inputs: Array
            Input tensor.
        encoded: Array = None
            Output of the encoder block to be fed into the decoder block if using
            `layer_type=TransformerLayerType.DECODER`.
        attention_mask: Array = None
            Boolean tensor used to mask out self-attention softmax input.
        encoder_decoder_mask: Array = None
            Boolean tensor used to mask out cross-attention softmax input when
            `layer_type=TransformerLayerType.DECODER`.
        deterministic: bool = False
            Disables dropout layers if set to True.
        decode: bool = False
            Whether to prepare and use an autoregressive cache in MHA.
        max_decode_length: bool = None
            The maximum length to generate relative embedding biases when
            layer_type=TransformerLayerType.DECODER` and
            `enable_relative_embedding` is True.

        Returns
        -------
        z: Array
            Output tensor of this transformer block.
        """
        assert self.layer_type in TransformerLayerType, \
                "layer_type should be one of TransformerLayerType" \
                f", but got {self.layer_type}."

        assert self.hidden_size % self.num_attention_heads == 0, \
                "hidden_size should be multiples of num_attention_heads" \
                f", but got hidden_size={self.hidden_size} and " \
                f"num_attention_heads={self.num_attention_heads}."

        assert self.layer_type == TransformerLayerType.DECODER or \
              (self.layer_type == TransformerLayerType.ENCODER and decode is False), \
               "decode should be False when layer_type == TransformerLayerType.ENCODER."

        head_dim = self.hidden_size // self.num_attention_heads

        sequence_dim = 0 if self.transpose_batch_sequence else 1
        batch_dim = 1 - sequence_dim

        attn_bias = None
        if self.enable_relative_embedding:
            if self.relative_embedding is None:
                rel_emb = RelativePositionBiases(
                    num_buckets=32,
                    max_distance=128,
                    num_heads=self.num_attention_heads,
                    dtype=self.dtype,
                    embedding_init=nn.initializers.variance_scaling(
                        1.0, 'fan_avg', 'uniform'),
                    name='relpos_bias')
            else:
                rel_emb = self.relative_embedding

            if self.layer_type == TransformerLayerType.ENCODER:
                attn_bias = rel_emb(inputs.shape[sequence_dim],
                                    inputs.shape[sequence_dim], True)
            else:
                l = max_decode_length if decode and max_decode_length else inputs.shape[
                    sequence_dim]
                attn_bias = rel_emb(l, l, False)

        # Make name be the exactly same as T5X, since names would affect
        # RNGKey during init and apply. Myabe no need in the feature.
        if self.layer_type == TransformerLayerType.ENCODER:
            assert inputs.ndim == 3
            # pre_attn_ln_name = 'pre_attention_layer_norm'
            mha_name = 'attention'
        else:
            # pre_attn_ln_name = 'pre_self_attention_layer_norm'
            mha_name = 'self_attention'

        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        x, residual = MultiHeadAttention(
            num_heads=self.num_attention_heads,
            dtype=self.dtype,
            head_dim=head_dim,
            transpose_batch_sequence=self.transpose_batch_sequence,
            dropout_rate=self.attention_dropout,
            float32_logits=self.float32_attention_logits,
            scale_attn_logits=self.scale_attn_logits,
            scaled_query_init=self.scaled_query_init,
            layernorm_epsilon=self.layernorm_epsilon,
            apply_residual_connection_post_layernorm=self.
            apply_residual_connection_post_layernorm,
            output_layernorm=self.output_layernorm,
            fuse_qkv=self.fuse_qkv_params,
            kernel_init=self.mha_kernel_init,
            use_bias=self.use_bias,
            bias_init=self.bias_init,
            name=mha_name)(inputs,
                           inputs,
                           attention_mask,
                           attn_bias,
                           deterministic=deterministic,
                           decode=decode)

        x = nn.Dropout(rate=self.hidden_dropout,
                       broadcast_dims=(-2, ))(x, deterministic=deterministic)
        if self.drop_path > 0.0:
            drop_path_shape = _generate_drop_path_shape(x.shape, batch_dim)
            x = nn.Dropout(rate=self.drop_path,
                           broadcast_dims=drop_path_shape)(
                               x, deterministic=deterministic)
        x = x + residual

        mlp_input = x
        if self.layer_type == TransformerLayerType.DECODER:
            assert encoded is not None, \
                "encoded is required when layer_type == TransformerLayerType.DECODER."

            y, residual = MultiHeadAttention(
                num_heads=self.num_attention_heads,
                dtype=self.dtype,
                head_dim=head_dim,
                transpose_batch_sequence=self.transpose_batch_sequence,
                dropout_rate=self.attention_dropout,
                layernorm_epsilon=self.layernorm_epsilon,
                apply_residual_connection_post_layernorm=self.
                apply_residual_connection_post_layernorm,
                output_layernorm=False,  # Must do LayerNorm before MHA.
                float32_logits=self.float32_attention_logits,
                scale_attn_logits=self.scale_attn_logits,
                scaled_query_init=self.scaled_query_init,
                fuse_qkv=self.fuse_qkv_params,
                kernel_init=self.mha_kernel_init,
                use_bias=self.use_bias,
                bias_init=self.bias_init,
                name='encoder_decoder_attention')(x,
                                                  encoded,
                                                  encoder_decoder_mask,
                                                  deterministic=deterministic)
            y = nn.Dropout(rate=self.hidden_dropout,
                           broadcast_dims=(-2, ))(y,
                                                  deterministic=deterministic)
            mlp_input = y + residual

        # MlpBlock
        residual = mlp_input
        z, ln_out = LayerNormMlpBlock(
            epsilon=self.layernorm_epsilon,
            transpose_batch_sequence=self.transpose_batch_sequence,
            return_layernorm_output=self.
            apply_residual_connection_post_layernorm,
            intermediate_dim=self.mlp_hidden_size,
            activations=self.mlp_activations,
            intermediate_dropout_rate=self.hidden_dropout,
            dtype=self.dtype,
            scale_axes=('embed', ),
            kernel_init=self.mlp_kernel_init,
            kernel_axes_1=('embed', 'mlp'),
            kernel_axes_2=('mlp', 'embed'),
            use_bias=self.use_bias,
            bias_init=self.bias_init,
            name='mlp',
        )(mlp_input, deterministic=deterministic)

        if self.apply_residual_connection_post_layernorm:
            assert ln_out is not None
            residual = ln_out

        z = nn.Dropout(rate=self.hidden_dropout,
                       broadcast_dims=(-2, ))(z, deterministic=deterministic)
        if self.drop_path > 0.0:
            drop_path_shape = _generate_drop_path_shape(z.shape, batch_dim)
            z = nn.Dropout(rate=self.drop_path,
                           broadcast_dims=drop_path_shape)(
                               z, deterministic=deterministic)
        z = z + residual

        if self.output_layernorm:
            z = LayerNorm(dtype=self.dtype,
                          epsilon=self.layernorm_epsilon,
                          name="output_layer_norm")(z)

        return z

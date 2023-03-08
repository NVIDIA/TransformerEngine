# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import functools
import operator
from typing import Any, Callable, Tuple, Sequence, Union, Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from cuda import cudart
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from jax import lax, vmap
from jax import nn as jax_nn
from jax import random as jax_random

PRNGKey = Any
Shape = Tuple[int, ...]
DType = jnp.dtype
Array = Any
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision,
                                                                       lax.Precision]]
Initializer = Callable[[PRNGKey, Shape, DType], Array]


def is_fp8_supported():
    """
    Thus JAX doesn't have API to query capability
    Use cuda-python for get the compute capability
    """
    cudaSuccess = cudart.cudaError_t.cudaSuccess
    ret, gpu_id = cudart.cudaGetDevice()
    assert ret == cudaSuccess
    flag = cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor
    ret, sm_major = cudart.cudaDeviceGetAttribute(flag, gpu_id)
    assert ret == cudaSuccess
    return sm_major >= 9


def is_devices_enough(required):
    return len(jax.devices()) >= required


def _generate_drop_path_shape(shape: Sequence[int], batch_dim: int) -> Sequence[int]:
    # Generate broadcast dims for drop_path.
    drop_path_shape = list(range(0, len(shape)))
    drop_path_shape.pop(batch_dim)
    return drop_path_shape


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
    # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
    return tuple(ax if ax >= 0 else ndim + ax for ax in axes)


def _canonicalize_tuple(x):
    if isinstance(x, Iterable):
        return tuple(x)
    return (x,)


def _convert_to_activation_function(fn_or_string: Union[str, Callable]) -> Callable:
    """Convert a string to an activation function."""
    if fn_or_string == 'linear':
        return lambda x: x
    if isinstance(fn_or_string, str):
        return getattr(nn, fn_or_string)
    if callable(fn_or_string):
        return fn_or_string
    raise ValueError(f"don't know how to convert {fn_or_string} to an activation function")


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
    assert all(map(lambda x: x.ndim == masks[0].ndim,
                   masks)), (f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
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
    assert all(map(lambda x: x.ndim == masks[0].ndim,
                   masks)), (f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
    mask, *other_masks = masks
    for other_mask in other_masks:
        mask = mask + other_mask
    return mask


def dot_product_attention(query: Array,
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
    assert query.shape[batch_dim] == key.shape[batch_dim] == value.shape[batch_dim], (
        'q, k, v batch dims must match.')
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], ('q, k, v num_heads must match.')
    sequence_dim = 0 if transpose_batch_sequence else 1
    assert key.shape[sequence_dim] == value.shape[sequence_dim], 'k, v lengths must match.'
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
        multiplier = (keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
        attn_weights = attn_weights * multiplier

    # Take the linear combination of `value`.
    if transpose_batch_sequence:
        return jnp.einsum('bhqk,kbhd->qbhd', attn_weights, value)

    return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)


class DenseGeneral(nn.Module):
    """A linear transformation with flexible axes and FP8 support.

        Attributes:
        features: tuple with numbers of output features.
        axis: tuple with axes to apply the transformation on.
        dtype: the dtype of the computation (default: float32).
        kernel_init: initializer function for the weight matrix.
        use_bias: whether to add a bias to the output (default: False).
        bias_init: initializer function for the bias vector.
    """
    features: Union[Iterable[int], int]
    axis: Union[Iterable[int], int] = -1
    dtype: DType = jnp.float32
    kernel_init: Initializer = None
    kernel_axes: Tuple[str, ...] = ()
    use_bias: bool = False
    bias_init: Initializer = nn.initializers.zeros
    bias_axes: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.kernel_init is None:
            self.kernel_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
        super().__post_init__()

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along multiple dimensions.

        Args:
        inputs: The nd-array to be transformed.

        Returns:
        The transformed input.
        """
        features = _canonicalize_tuple(self.features)
        axis = _canonicalize_tuple(self.axis)

        inputs = jnp.asarray(inputs, self.dtype)
        axis = _normalize_axes(axis, inputs.ndim)

        kernel_shape = tuple(inputs.shape[ax] for ax in axis) + features
        kernel_param_shape = (np.prod([inputs.shape[ax] for ax in axis]), np.prod(features))
        kernel = nn_partitioning.param_with_axes('kernel',
                                                 self.kernel_init,
                                                 kernel_param_shape,
                                                 jnp.float32,
                                                 axes=self.kernel_axes)

        kernel = jnp.asarray(kernel, self.dtype)
        kernel = jnp.reshape(kernel, kernel_shape)

        if self.use_bias:
            bias = nn_partitioning.param_with_axes('bias',
                                                   self.bias_init, (self.features,),
                                                   self.dtype,
                                                   axes=self.bias_axes)
        else:
            bias = None

        contract_ind = tuple(range(0, len(axis)))

        y = lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))

        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: Type for the dense layer.
  """
    transpose_batch_sequence: bool
    intermediate_dim: int = 2048
    activations: Sequence[Union[str, Callable]] = ('relu',)
    kernel_init: Initializer = None
    intermediate_dropout_rate: float = 0.1
    dtype: Any = jnp.float32
    fuse_wi: bool = False

    def __post_init__(self):
        if self.kernel_init is None:
            self.kernel_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
        super().__post_init__()

    @nn.compact
    def __call__(self, inputs, deterministic: bool = False):
        """Applies Transformer MlpBlock module."""
        # Iterate over specified MLP input activation functions.
        # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.

        activations = []
        if self.fuse_wi:
            dense_name = 'wi'
            num_activations = len(self.activations)
            x = DenseGeneral(self.intermediate_dim * num_activations,
                             dtype=self.dtype,
                             kernel_init=self.kernel_init,
                             kernel_axes=('embed', 'mlp'),
                             name=dense_name)(inputs)
            x = jnp.split(x, num_activations, axis=-1)
            for idx, act_fn in enumerate(self.activations):
                x_i = _convert_to_activation_function(act_fn)(x[idx])
                activations.append(x_i)
        else:
            for idx, act_fn in enumerate(self.activations):
                dense_name = 'wi' if len(self.activations) == 1 else f'wi_{idx}'
                x = DenseGeneral(self.intermediate_dim,
                                 dtype=self.dtype,
                                 kernel_init=self.kernel_init,
                                 kernel_axes=('embed', 'mlp'),
                                 name=dense_name)(inputs)
                x = _convert_to_activation_function(act_fn)(x)
                activations.append(x)

        # Take elementwise product of above intermediate activations.
        x = functools.reduce(operator.mul, activations)
        # Apply dropout and final dense output projection.
        x = nn.Dropout(rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)    # Broadcast along length.
        if self.transpose_batch_sequence:
            x = nn_partitioning.with_sharding_constraint(x, ('length', 'batch', 'mlp'))
        else:
            x = nn_partitioning.with_sharding_constraint(x, ('batch', 'length', 'mlp'))
        output = DenseGeneral(inputs.shape[-1],
                              dtype=self.dtype,
                              kernel_init=self.kernel_init,
                              kernel_axes=('mlp', 'embed'),
                              name='wo')(x)
        return output


dynamic_vector_slice_in_dim = vmap(lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))


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

    num_heads: int
    head_dim: int
    transpose_batch_sequence: bool
    dtype: DType = jnp.float32
    dropout_rate: float = 0.
    kernel_init: Initializer = None
    float32_logits: bool = False    # computes logits in float32 for stability.
    scale_attn_logits: bool = False
    scaled_query_init: bool = True
    fuse_qkv: bool = True

    def __post_init__(self):
        if self.kernel_init is None:
            self.kernel_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        super().__post_init__()

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
        projection = functools.partial(DenseGeneral,
                                       axis=-1,
                                       features=self.num_heads * self.head_dim,
                                       kernel_axes=('embed', 'joined_kv'),
                                       dtype=self.dtype)

        # NOTE: T5 does not explicitly rescale the attention logits by
        #       1/sqrt(depth_kq)!  This is folded into the initializers of the
        #       linear transformations, which is equivalent under Adafactor
        depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
        query_init = lambda *args: self.kernel_init(*args) / (    # pylint: disable=unnecessary-lambda-assignment
            depth_scaling if self.scaled_query_init else 1.0)

        # Project inputs_q to multi-headed q/k/v
        # dimensions are then [batch, length, num_heads, head_dim]

        def qkv_init(key, shape, dtype):
            assert shape[-1] % 3 == 0

            q_shape = (shape[0], shape[1] // 3)
            k_shape = (shape[0], shape[1] // 3)
            v_shape = (shape[0], shape[1] // 3)

            q_kernel = query_init(key, q_shape, dtype)
            k_kernel = self.kernel_init(key, k_shape, dtype)    # pylint: disable=too-many-function-args
            v_kernel = self.kernel_init(key, v_shape, dtype)    # pylint: disable=too-many-function-args

            return jnp.concatenate([q_kernel, k_kernel, v_kernel], axis=-1, dtype=dtype)

        if self.fuse_qkv:
            if inputs_q is inputs_kv:
                qkv_proj = DenseGeneral(axis=-1,
                                        features=self.num_heads * self.head_dim * 3,
                                        kernel_axes=('embed', 'joined_kv'),
                                        kernel_init=qkv_init,
                                        name='qkv',
                                        dtype=self.dtype)(inputs_kv)
                query, key, value = jnp.split(
                    qkv_proj, [self.num_heads * self.head_dim, self.num_heads * self.head_dim * 2],
                    axis=-1)
                if self.scale_attn_logits:
                    query = query / depth_scaling
            else:
                query = projection(kernel_init=query_init, name='query')( \
                        (inputs_q / depth_scaling) if self.scale_attn_logits else inputs_q)
                kv_proj = DenseGeneral(axis=-1,
                                       features=self.num_heads * self.head_dim * 2,
                                       kernel_axes=('embed', 'joined_kv'),
                                       kernel_init=self.kernel_init,
                                       name='kv',
                                       dtype=self.dtype)(inputs_kv)
                key, value = jnp.split(kv_proj, [self.num_heads * self.head_dim], axis=-1)
        else:
            query = projection(kernel_init=query_init, name='query')( \
                    (inputs_q / depth_scaling) if self.scale_attn_logits else inputs_q)
            key = projection(kernel_init=self.kernel_init, name='key')(inputs_kv)
            value = projection(kernel_init=self.kernel_init, name='value')(inputs_kv)

        query = query.reshape((query.shape[0], query.shape[1], self.num_heads, self.head_dim))
        key = key.reshape((key.shape[0], key.shape[1], self.num_heads, self.head_dim))
        value = value.reshape((value.shape[0], value.shape[1], self.num_heads, self.head_dim))

        if self.transpose_batch_sequence:
            query = nn_partitioning.with_sharding_constraint(query,
                                                             ('length', 'batch', 'heads', 'kv'))
            key = nn_partitioning.with_sharding_constraint(key, ('length', 'batch', 'heads', 'kv'))
            value = nn_partitioning.with_sharding_constraint(value,
                                                             ('length', 'batch', 'heads', 'kv'))
        else:
            query = nn_partitioning.with_sharding_constraint(query,
                                                             ('batch', 'length', 'heads', 'kv'))
            key = nn_partitioning.with_sharding_constraint(key, ('batch', 'length', 'heads', 'kv'))
            value = nn_partitioning.with_sharding_constraint(value,
                                                             ('batch', 'length', 'heads', 'kv'))

        if decode:
            # Detect if we're initializing by absence of existing cache data.
            is_initialized = self.has_variable('cache', 'cached_key')
            # The key and value have dimension [batch, length, num_heads, head_dim],
            # but we cache them as [batch, num_heads, head_dim, length] as a TPU
            # fusion optimization. This also enables the "scatter via one-hot
            # broadcast" trick, which means we do a one-hot broadcast instead of a
            # scatter/gather operations, resulting in a 3-4x speedup in practice.
            swap_dims = lambda x: x[:-3] + tuple(x[i] for i in [-2, -1, -3])    # pylint: disable=unnecessary-lambda-assignment
            cached_key = self.variable('cache', 'cached_key', jnp.zeros, swap_dims(key.shape),
                                       key.dtype)
            cached_value = self.variable('cache', 'cached_value', jnp.zeros, swap_dims(value.shape),
                                         value.dtype)
            cache_index = self.variable('cache', 'cache_index',
                                        lambda: jnp.array(0, dtype=jnp.int32))
            if is_initialized:
                batch, num_heads, head_dim, length = cached_key.value.shape
                # During fast autoregressive decoding, we feed one position at a time,
                # and cache the keys and values step by step.
                # Sanity shape check of cached key against input query.
                expected_shape = (batch, 1, num_heads, head_dim)
                if expected_shape != query.shape:
                    raise ValueError(
                        'Autoregressive cache shape error, '
                        f"expected query shape {expected_shape} instead got {query.shape}.")

                # Create a OHE of the current index. NOTE: the index is increased below.
                cur_index = cache_index.value
                one_hot_indices = jax_nn.one_hot(cur_index, length, dtype=key.dtype)
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
                    bias = dynamic_vector_slice_in_dim(jnp.squeeze(bias, axis=0),
                                                       jnp.reshape(cur_index, (-1)), 1, -2)

        # Convert the boolean attention mask to an attention bias.
        if mask is not None:
            # attention mask in the form of attention bias
            attention_bias = lax.select(mask > 0,
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
        x = dot_product_attention(query,
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
            x = nn_partitioning.with_sharding_constraint(x, ('length', 'batch', 'joined_kv'))
        else:
            x = nn_partitioning.with_sharding_constraint(x, ('batch', 'length', 'joined_kv'))

        # Back to the original inputs dimensions.
        out = DenseGeneral(
            features=inputs_q.shape[-1],    # output dim is set to the input dim.
            axis=-1,
            kernel_init=self.kernel_init,
            kernel_axes=('joined_kv', 'embed'),
            dtype=self.dtype,
            name='out')(x)
        return out


class LayerNorm(nn.Module):
    """T5 Layer normalization operating on the last axis of the input data."""
    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    layernorm_type: str = 'layernorm'
    scale_init: Initializer = nn.initializers.ones
    bias_init: Initializer = nn.initializers.zeros

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies layer normalization on the input."""

        x = jnp.asarray(x, jnp.float32)
        features = x.shape[-1]

        scale = nn_partitioning.param_with_axes('scale',
                                                self.scale_init, (features,),
                                                jnp.float32,
                                                axes=('embed',))
        scale = jnp.asarray(scale, self.dtype)

        if self.layernorm_type == 'layernorm':
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
            y = (x - mean) * lax.rsqrt(var + self.epsilon)

            bias = nn_partitioning.param_with_axes('ln_bias',
                                                   self.bias_init, (features,),
                                                   jnp.float32,
                                                   axes=('embed',))
            bias = jnp.asarray(bias, self.dtype)

            y = jnp.asarray(y, self.dtype)
            z = y * scale + bias
        else:
            assert self.layernorm_type == 'rmsnorm'
            mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
            y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
            z = y * scale

        return z


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
        is_small = n < max_exact
        val_if_large = max_exact + (
            np.log(n.astype(np.float32) / max_exact + np.finfo(np.float32).eps) /
            np.log(max_distance / max_exact) * (num_buckets - max_exact)).astype(np.int32)
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
        relative_position = memory_position - context_position    # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(relative_position,
                                                   bidirectional=bidirectional,
                                                   num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        relative_attention_bias = nn_partitioning.param_with_axes(
            'rel_embedding',
            self.embedding_init, (self.num_heads, self.num_buckets),
            jnp.float32,
            axes=('heads', 'relpos_buckets'))

        relative_attention_bias = jnp.asarray(relative_attention_bias, self.dtype)
        # Instead of using a slow gather, we create a leading-dimension one-hot
        # array from rp_bucket and use it to perform the gather-equivalent via a
        # contraction, i.e.:
        # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
        # This is equivalent to relative_attention_bias[:, rp_bucket]
        bcast_iota = lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1), 0)
        rp_bucket_one_hot = jnp.array(rp_bucket[jnp.newaxis, ...] == bcast_iota, dtype=self.dtype)
        # --> shape (qlen, klen, num_heads)
        values = lax.dot_general(
            relative_attention_bias,
            rp_bucket_one_hot,
            (
                ((1,), (0,)),    # rhs, lhs contracting dims
                ((), ())))    # no batched dims
        # Add a singleton batch dimension.
        # --> shape (1, num_heads, qlen, klen)
        return values[jnp.newaxis, ...]


class EncoderLayer(nn.Module):
    """Transformer encoder layer."""
    relative_embedding: nn.Module = None
    num_heads: int = 8
    head_dim: int = 64
    dropout_rate: float = 0.1
    transpose_batch_sequence: bool = True
    float32_attention_logits: bool = False
    scale_attn_logits: bool = False
    scaled_query_init: bool = True
    mlp_dim: int = 2048
    mlp_activations: Sequence[str] = ('relu',)
    dtype: Any = jnp.float32
    apply_residual_connection_post_layernorm: bool = False
    layernorm_type: str = 'layernorm'
    output_layernorm: bool = False
    drop_path: float = 0.0
    fuse_qkv_params: bool = True
    fuse_mlp_wi: bool = False

    @nn.compact
    def __call__(self, inputs, encoder_mask=None, deterministic=False):
        # Relative position embedding as attention biases.
        sequence_dim = 0 if self.transpose_batch_sequence else 1
        batch_dim = 1 - sequence_dim

        if self.relative_embedding is None:
            rel_emb = RelativePositionBiases(num_buckets=32,
                                             max_distance=128,
                                             num_heads=self.num_heads,
                                             dtype=self.dtype,
                                             embedding_init=nn.initializers.variance_scaling(
                                                 1.0, 'fan_avg', 'uniform'),
                                             name='relpos_bias')
        else:
            rel_emb = self.relative_embedding
        encoder_bias = rel_emb(inputs.shape[sequence_dim], inputs.shape[sequence_dim], True)

        # Attention block.
        residual = inputs

        if not self.output_layernorm:
            # Attention block.
            x = LayerNorm(layernorm_type=self.layernorm_type,
                          dtype=self.dtype,
                          name="pre_attention_layer_norm")(inputs)

            if self.apply_residual_connection_post_layernorm:
                residual = x
        else:
            x = inputs

        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        x = MultiHeadAttention(num_heads=self.num_heads,
                               dtype=self.dtype,
                               head_dim=self.head_dim,
                               transpose_batch_sequence=self.transpose_batch_sequence,
                               dropout_rate=self.dropout_rate,
                               float32_logits=self.float32_attention_logits,
                               scale_attn_logits=self.scale_attn_logits,
                               scaled_query_init=self.scaled_query_init,
                               fuse_qkv=self.fuse_qkv_params,
                               name='attention')(x,
                                                 x,
                                                 encoder_mask,
                                                 encoder_bias,
                                                 deterministic=deterministic)
        x = nn.Dropout(rate=self.dropout_rate,
                       broadcast_dims=(sequence_dim,))(x, deterministic=deterministic)
        if self.drop_path > 0.0:
            drop_path_shape = _generate_drop_path_shape(x.shape, batch_dim)
            x = nn.Dropout(rate=self.drop_path,
                           broadcast_dims=drop_path_shape)(x, deterministic=deterministic)
        x = x + residual

        # MLP block.
        residual = x
        y = LayerNorm(layernorm_type=self.layernorm_type,
                      dtype=self.dtype,
                      name='pre_mlp_layer_norm')(x)

        if self.apply_residual_connection_post_layernorm:
            residual = y

        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        y = MlpBlock(
            transpose_batch_sequence=self.transpose_batch_sequence,
            intermediate_dim=self.mlp_dim,
            activations=self.mlp_activations,
            intermediate_dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            fuse_wi=self.fuse_mlp_wi,
            name='mlp',
        )(y, deterministic=deterministic)
        y = nn.Dropout(rate=self.dropout_rate,
                       broadcast_dims=(sequence_dim,))(y, deterministic=deterministic)
        if self.drop_path > 0.0:
            drop_path_shape = _generate_drop_path_shape(y.shape, batch_dim)
            y = nn.Dropout(rate=self.drop_path,
                           broadcast_dims=drop_path_shape)(y, deterministic=deterministic)
        y = y + residual

        if self.output_layernorm:
            y = LayerNorm(layernorm_type=self.layernorm_type,
                          dtype=self.dtype,
                          name="output_layer_norm")(y)
        return y


class DecoderLayer(nn.Module):
    """Transformer decoder layer that attends to the encoder."""
    relative_embedding: nn.Module = None
    num_heads: int = 8
    head_dim: int = 64
    dropout_rate: float = 0.1
    transpose_batch_sequence: bool = True
    float32_attention_logits: bool = False
    scale_attn_logits: bool = False
    scaled_query_init: bool = True
    mlp_dim: int = 2048
    mlp_activations: Sequence[str] = ('relu',)
    dtype: Any = jnp.float32
    apply_residual_connection_post_layernorm: bool = False
    output_layernorm: bool = False
    layernorm_type: str = 'layernorm'
    drop_path: float = 0.0
    fuse_qkv_params: bool = True
    fuse_mlp_wi: bool = False

    @nn.compact
    def __call__(self,
                 inputs,
                 encoded,
                 decoder_mask=None,
                 encoder_decoder_mask=None,
                 deterministic=False,
                 decode=False,
                 max_decode_length=None):

        # Relative position embedding as attention biases.
        sequence_dim = 0 if self.transpose_batch_sequence else 1
        batch_dim = 1 - sequence_dim
        l = max_decode_length if decode and max_decode_length else inputs.shape[sequence_dim]
        if self.relative_embedding is None:
            rel_emb = RelativePositionBiases(num_buckets=32,
                                             max_distance=128,
                                             num_heads=self.num_heads,
                                             dtype=self.dtype,
                                             embedding_init=nn.initializers.variance_scaling(
                                                 1.0, 'fan_avg', 'uniform'),
                                             name='relpos_bias')
        else:
            rel_emb = self.relative_embedding
        decoder_bias = rel_emb(l, l, False)

        # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
        residual = inputs

        if not self.output_layernorm:
            # Attention block.
            x = LayerNorm(layernorm_type=self.layernorm_type,
                          dtype=self.dtype,
                          name="pre_self_attention_layer_norm")(inputs)

            if self.apply_residual_connection_post_layernorm:
                residual = x
        else:
            x = inputs

        # Self-attention block
        x = MultiHeadAttention(num_heads=self.num_heads,
                               dtype=self.dtype,
                               head_dim=self.head_dim,
                               transpose_batch_sequence=self.transpose_batch_sequence,
                               dropout_rate=self.dropout_rate,
                               float32_logits=self.float32_attention_logits,
                               scale_attn_logits=self.scale_attn_logits,
                               scaled_query_init=self.scaled_query_init,
                               fuse_qkv=self.fuse_qkv_params,
                               name='self_attention')(x,
                                                      x,
                                                      decoder_mask,
                                                      decoder_bias,
                                                      deterministic=deterministic,
                                                      decode=decode)
        x = nn.Dropout(rate=self.dropout_rate,
                       broadcast_dims=(sequence_dim,))(x, deterministic=deterministic)
        if self.drop_path > 0.0:
            drop_path_shape = _generate_drop_path_shape(x.shape, batch_dim)
            x = nn.Dropout(rate=self.drop_path,
                           broadcast_dims=drop_path_shape)(x, deterministic=deterministic)
        x = x + residual

        # Encoder-Decoder block.
        residual = x
        y = LayerNorm(layernorm_type=self.layernorm_type,
                      dtype=self.dtype,
                      name='pre_cross_attention_layer_norm')(x)

        if self.apply_residual_connection_post_layernorm:
            residual = y
        y = MultiHeadAttention(num_heads=self.num_heads,
                               dtype=self.dtype,
                               head_dim=self.head_dim,
                               transpose_batch_sequence=self.transpose_batch_sequence,
                               dropout_rate=self.dropout_rate,
                               float32_logits=self.float32_attention_logits,
                               scale_attn_logits=self.scale_attn_logits,
                               scaled_query_init=self.scaled_query_init,
                               fuse_qkv=self.fuse_qkv_params,
                               name='encoder_decoder_attention')(y,
                                                                 encoded,
                                                                 encoder_decoder_mask,
                                                                 deterministic=deterministic)
        y = nn.Dropout(rate=self.dropout_rate,
                       broadcast_dims=(sequence_dim,))(y, deterministic=deterministic)
        y = y + residual

        # MLP block.
        residual = y
        z = LayerNorm(layernorm_type=self.layernorm_type,
                      dtype=self.dtype,
                      name='pre_mlp_layer_norm')(y)
        if self.apply_residual_connection_post_layernorm:
            residual = z
        z = MlpBlock(
            transpose_batch_sequence=self.transpose_batch_sequence,
            intermediate_dim=self.mlp_dim,
            activations=self.mlp_activations,
            intermediate_dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            fuse_wi=self.fuse_mlp_wi,
            name='mlp',
        )(z, deterministic=deterministic)
        z = nn.Dropout(rate=self.dropout_rate,
                       broadcast_dims=(sequence_dim,))(z, deterministic=deterministic)
        if self.drop_path > 0.0:
            drop_path_shape = _generate_drop_path_shape(z.shape, batch_dim)
            z = nn.Dropout(rate=self.drop_path,
                           broadcast_dims=drop_path_shape)(z, deterministic=deterministic)
        z = z + residual

        if self.output_layernorm:
            z = LayerNorm(layernorm_type=self.layernorm_type,
                          dtype=self.dtype,
                          name="output_layer_norm")(z)

        return z


def assert_allclose(actual,
                    desired,
                    rtol=1e-05,
                    atol=1e-08,
                    equal_nan=True,
                    err_msg='',
                    verbose=True):
    if not isinstance(actual, float):
        actual = actual.astype(jnp.float32)
    if not isinstance(desired, float):
        desired = desired.astype(jnp.float32)
    np.testing.assert_allclose(actual, desired, rtol, atol, equal_nan, err_msg, verbose)

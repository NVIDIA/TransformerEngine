# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Utility for the TE layer tests"""

import functools
import math
import operator
from typing import Any, Callable, Dict, Tuple, Sequence, Union, Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import combine_masks
from jax import lax, vmap
from jax import nn as jax_nn
from jax import random as jax_random

from transformer_engine.jax.fp8 import DType as TEDType

PRNGKey = Any
Shape = Tuple[int, ...]
DType = jnp.dtype
Array = Any
PrecisionLike = Union[
    None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]
]
Initializer = Callable[[PRNGKey, Shape, DType], Array]


def is_devices_enough(required):
    """
    Check if the available GPUs is enough
    """
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
    if fn_or_string == "linear":
        return lambda x: x
    if isinstance(fn_or_string, str):
        return getattr(nn, fn_or_string)
    if callable(fn_or_string):
        return fn_or_string
    raise ValueError(f"don't know how to convert {fn_or_string} to an activation function")


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
    assert all(
        map(lambda x: x.ndim == masks[0].ndim, masks)
    ), f"masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}"
    mask, *other_masks = masks
    for other_mask in other_masks:
        mask = mask + other_mask
    return mask


class DotProductAttention(nn.Module):
    transpose_batch_sequence: bool = True
    scale_attn_logits: bool = True
    dropout_rate: float = 0.0
    dtype: DType = jnp.float32
    float32_logits: bool = False
    """Computes dot-product attention given query, key, and value.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights.

    Args:
        dropout_rate: dropout rate
        dtype: the dtype of the computation (default: float32)
        float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
    """

    @nn.compact
    def __call__(
        self,
        query: Array,
        key: Array,
        value: Array,
        bias: Optional[Array] = None,
        deterministic: bool = False,
    ):
        """
        Args:
            query: queries for calculating attention with shape of `[batch, q_length,
            num_heads, qk_depth_per_head]`.
            key: keys for calculating attention with shape of `[batch, kv_length,
            num_gqa_groups, qk_depth_per_head]`.
            value: values to be used in attention with shape of `[batch, kv_length,
            num_gqa_groups, v_depth_per_head]`.
            bias: bias for the attention weights. This should be broadcastable to the
            shape `[batch, num_heads, q_length, kv_length]` This can be used for
            incorporating causal masks, padding masks, proximity bias, etc.
            dropout_rng: JAX PRNGKey: to be used for dropout
            deterministic: bool, deterministic or not (to apply dropout)
        Returns:
            Output of shape `[batch, length, num_heads, v_depth_per_head]`.
        """
        assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
        batch_dim = 1 if self.transpose_batch_sequence else 0
        assert (
            query.shape[batch_dim] == key.shape[batch_dim] == value.shape[batch_dim]
        ), "q, k, v batch dims must match."
        sequence_dim = 0 if self.transpose_batch_sequence else 1
        assert key.shape[sequence_dim] == value.shape[sequence_dim], "k, v lengths must match."
        assert key.shape[-2] == value.shape[-2], "k, v num_heads must match."
        assert query.shape[-1] == key.shape[-1], "q, k head_dim must match."

        if self.scale_attn_logits:
            head_dim = query.shape[-1]
            depth_scaling = jnp.sqrt(head_dim).astype(self.dtype)
            query = query / depth_scaling

        # Casting logits and softmax computation for float32 for model stability.
        if self.float32_logits:
            query = query.astype(jnp.float32)
            key = key.astype(jnp.float32)

        # `attn_weights`: [batch, num_heads, groups, q_length, kv_length]
        h_q, h_kv = query.shape[-2], key.shape[-2]
        assert (h_q % h_kv == 0) and (h_q >= h_kv)
        group_size = h_q // h_kv
        grouped_query = query.reshape((*query.shape[:2], h_kv, group_size, query.shape[-1]))

        if self.transpose_batch_sequence:
            attn_weights = jnp.einsum("qbhgd,kbhd->bhgqk", grouped_query, key)
        else:
            attn_weights = jnp.einsum("bqhgd,bkhd->bhgqk", grouped_query, key)

        # reshape back to normal DPA shape for bias/softmax/dropout
        b, h, g, q, k = attn_weights_with_groups_shape = attn_weights.shape
        attn_weights_without_groups_shape = (b, h * g, q, k)
        attn_weights = attn_weights.reshape(attn_weights_without_groups_shape)

        # Apply attention bias: masking, dropout, proximity bias, etc.
        if bias is not None:
            attn_weights = attn_weights + bias.astype(attn_weights.dtype)

        # Normalize the attention weights across `kv_length` dimension.
        attn_weights = jax_nn.softmax(attn_weights).astype(self.dtype)

        # Apply attention dropout.
        if not deterministic and self.dropout_rate > 0.0:
            keep_prob = 1.0 - self.dropout_rate
            # T5 broadcasts along the "length" dim, but unclear which one that
            # corresponds to in positional dimensions here, assuming query dim.
            dropout_shape = list(attn_weights.shape)
            dropout_rng = self.make_rng("dropout")
            keep = jax_random.bernoulli(dropout_rng, keep_prob, dropout_shape)
            multiplier = keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=self.dtype)
            attn_weights = attn_weights * multiplier

        attn_weights = attn_weights.reshape(attn_weights_with_groups_shape)

        # Take the linear combination of `value`.
        if self.transpose_batch_sequence:
            return jnp.einsum("bhgqk,kbhd->qbhgd", attn_weights, value).reshape(query.shape)

        return jnp.einsum("bhgqk,bkhd->bqhgd", attn_weights, value).reshape(query.shape)


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
            self.kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")
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
        kernel = nn_partitioning.param_with_axes(
            "kernel", self.kernel_init, kernel_param_shape, jnp.float32, axes=self.kernel_axes
        )

        kernel = jnp.asarray(kernel, self.dtype)
        kernel = jnp.reshape(kernel, kernel_shape)

        if self.use_bias:
            bias = nn_partitioning.param_with_axes(
                "bias", self.bias_init, self.features, jnp.float32, axes=self.bias_axes
            )
            bias = bias.astype(self.dtype)
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
    activations: Sequence[Union[str, Callable]] = ("relu",)
    kernel_init: Initializer = None
    intermediate_dropout_rate: float = 0.1
    intermediate_dropout_dims: Sequence[int] = ()
    use_bias: bool = False
    dtype: Any = jnp.float32
    fuse_wi: bool = True

    def __post_init__(self):
        if self.kernel_init is None:
            self.kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")
        super().__post_init__()

    @nn.compact
    def __call__(self, inputs, deterministic: bool = False):
        """Applies Transformer MlpBlock module."""
        # Iterate over specified MLP input activation functions.
        # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.

        activations = []
        if self.fuse_wi:
            dense_name = "wi"
            num_activations = len(self.activations)
            x = DenseGeneral(
                self.intermediate_dim * num_activations,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                kernel_axes=("embed", "mlp"),
                use_bias=self.use_bias,
                bias_axes="mlp",
                name=dense_name,
            )(inputs)
            x = jnp.split(x, num_activations, axis=-1)
            for idx, act_fn in enumerate(self.activations):
                x_i = _convert_to_activation_function(act_fn)(x[idx])
                activations.append(x_i)
        else:
            for idx, act_fn in enumerate(self.activations):
                dense_name = "wi" if len(self.activations) == 1 else f"wi_{idx}"
                x = DenseGeneral(
                    self.intermediate_dim,
                    dtype=self.dtype,
                    kernel_init=self.kernel_init,
                    kernel_axes=("embed", "mlp"),
                    use_bias=self.use_bias,
                    bias_axes="mlp",
                    name=dense_name,
                )(inputs)
                x = _convert_to_activation_function(act_fn)(x)
                activations.append(x)

        # Take elementwise product of above intermediate activations.
        x = functools.reduce(operator.mul, activations)
        # Apply dropout and final dense output projection.
        x = nn.Dropout(
            rate=self.intermediate_dropout_rate, broadcast_dims=self.intermediate_dropout_dims
        )(
            x, deterministic=deterministic
        )  # Broadcast along length.
        if self.transpose_batch_sequence:
            x = nn_partitioning.with_sharding_constraint(x, ("length", "batch", "mlp"))
        else:
            x = nn_partitioning.with_sharding_constraint(x, ("batch", "length", "mlp"))
        output = DenseGeneral(
            inputs.shape[-1],
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            kernel_axes=("mlp", "embed"),
            use_bias=self.use_bias,
            bias_axes="embed",
            name="wo",
        )(x)
        return output


def apply_rotary_pos_emb_alternate(
    inputs: jnp.ndarray,
    position: jnp.ndarray,
    min_timescale: int = 1,
    max_timescale: int = 10000,
):
    embedding_dim = inputs.shape[-1]
    half_embedding_dim = embedding_dim // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / embedding_dim
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction
    timescale = jnp.expand_dims(timescale, axis=tuple(range(inputs.ndim - 1)))
    position = jnp.expand_dims(position, axis=tuple(range(2, inputs.ndim)))
    sinusoid_inp = position / timescale
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    first_part = first_part.astype(inputs.dtype)
    second_part = second_part.astype(inputs.dtype)
    return jnp.concatenate([first_part, second_part], axis=-1)


def apply_rotary_pos_emb_consecutive(
    inputs: jnp.ndarray,
    position: jnp.ndarray,
    min_timescale: int = 1,
    max_timescale: int = 10000,
):
    embedding_dim = inputs.shape[-1]
    half_embedding_dim = embedding_dim // 2
    fraction = 2 * jnp.arange(0, half_embedding_dim) / embedding_dim

    inputs_shifted_left = jnp.concatenate([inputs[..., 1:], inputs[..., :1]], axis=-1)
    inputs_shifted_right = jnp.concatenate([inputs[..., -1:], inputs[..., :-1]], axis=-1)
    inputs_shifted = jax.lax.select(
        jnp.tile(
            jnp.mod(jnp.arange(embedding_dim, dtype=jnp.int32), 2),
            inputs.shape[:-1] + (1,),
        ),
        inputs_shifted_right,
        inputs_shifted_left,
    )
    fraction = jnp.repeat(fraction, 2)
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction

    position = jnp.expand_dims(position, axis=tuple(range(2, inputs.ndim)))

    sinusoid_inp = position / timescale
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    sign = jnp.sign(jnp.mod(jnp.arange(embedding_dim, dtype=jnp.int32), 2) - 0.5)
    outputs = inputs * cos + inputs_shifted * sin * sign

    return outputs


dynamic_vector_slice_in_dim = vmap(lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))


class MultiHeadAttention(nn.Module):
    """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      num_gqa_groups: number of kv attention heads
      head_dim: dimension of each head.
      dtype: the dtype of the computation.
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
    """

    num_heads: int = 8
    num_gqa_groups: int | None = None
    head_dim: int = 64
    transpose_batch_sequence: bool = True
    dtype: DType = jnp.float32
    dropout_rate: float = 0.0
    kernel_init: Initializer = None
    float32_logits: bool = False  # computes logits in float32 for stability.
    scale_attn_logits: bool = False
    scaled_query_init: bool = True
    enable_rotary_pos_emb: bool = False
    rotary_pos_emb_group_method: str = "consecutive"
    fuse_qkv: bool = True
    use_bias: bool = False

    def __post_init__(self):
        if self.kernel_init is None:
            self.kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal")
        if self.num_gqa_groups is None:
            self.num_gqa_groups = self.num_attention_heads
        super().__post_init__()

    @nn.compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array,
        mask: Optional[Array] = None,
        bias: Optional[Array] = None,
        *,
        decode: bool = False,
        deterministic: bool = False,
    ) -> Array:
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
        q_projection = functools.partial(
            DenseGeneral,
            axis=-1,
            features=self.num_heads * self.head_dim,
            kernel_axes=("embed", "joined_kv"),
            use_bias=self.use_bias,
            bias_axes="joined_kv",
            dtype=self.dtype,
        )

        kv_projection = functools.partial(
            DenseGeneral,
            axis=-1,
            features=self.num_gqa_groups * self.head_dim,
            kernel_axes=("embed", "joined_kv"),
            use_bias=self.use_bias,
            bias_axes="joined_kv",
            dtype=self.dtype,
        )

        # NOTE: T5 does not explicitly rescale the attention logits by
        #       1/sqrt(depth_kq)!  This is folded into the initializers of the
        #       linear transformations, which is equivalent under Adafactor
        depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
        query_init = lambda *args: self.kernel_init(*args) / (
            depth_scaling if self.scaled_query_init else 1.0
        )

        # Project inputs_q to multi-headed q/k/v
        # dimensions are then [batch, length, num_heads, head_dim]

        def qkv_init(key, shape, dtype):
            assert shape[-1] % 3 == 0

            q_shape = (shape[0], shape[1] // 3)
            k_shape = (shape[0], shape[1] // 3)
            v_shape = (shape[0], shape[1] // 3)

            q_kernel = query_init(key, q_shape, dtype)
            k_kernel = self.kernel_init(key, k_shape, dtype)
            v_kernel = self.kernel_init(key, v_shape, dtype)

            return jnp.concatenate([q_kernel, k_kernel, v_kernel], axis=-1, dtype=dtype)

        is_self_attn = inputs_q is inputs_kv
        is_gqa = self.num_heads != self.num_gqa_groups
        is_qkvpack = is_self_attn and not is_gqa

        if self.fuse_qkv:
            if is_qkvpack:
                qkv_proj = DenseGeneral(
                    axis=-1,
                    features=self.num_heads * self.head_dim * 3,
                    kernel_axes=("embed", "joined_kv"),
                    kernel_init=qkv_init,
                    use_bias=self.use_bias,
                    bias_axes="joined_kv",
                    name="qkv",
                    dtype=self.dtype,
                )(inputs_kv)
                query, key, value = jnp.split(
                    qkv_proj,
                    [self.num_heads * self.head_dim, self.num_heads * self.head_dim * 2],
                    axis=-1,
                )
            else:
                query = q_projection(kernel_init=query_init, name="query")(inputs_q)

                kv_proj = DenseGeneral(
                    axis=-1,
                    features=self.num_gqa_groups * self.head_dim * 2,
                    kernel_axes=("embed", "joined_kv"),
                    kernel_init=self.kernel_init,
                    use_bias=self.use_bias,
                    bias_axes="joined_kv",
                    name="kv",
                    dtype=self.dtype,
                )(inputs_kv)
                key, value = jnp.split(kv_proj, [self.num_gqa_groups * self.head_dim], axis=-1)
        else:
            query = q_projection(kernel_init=query_init, name="query")(inputs_q)
            key = kv_projection(kernel_init=self.kernel_init, name="key")(inputs_kv)
            value = kv_projection(kernel_init=self.kernel_init, name="value")(inputs_kv)

        if self.enable_rotary_pos_emb:
            batch_dim = 1 if self.transpose_batch_sequence else 0
            seq_dim = 1 - batch_dim

            q_position = jnp.expand_dims(jnp.arange(query.shape[seq_dim]), axis=batch_dim)
            k_position = jnp.expand_dims(jnp.arange(query.shape[seq_dim]), axis=batch_dim)

            if self.rotary_pos_emb_group_method == "alternate":
                apply_rotary_pos_emb = apply_rotary_pos_emb_alternate
            else:
                apply_rotary_pos_emb = apply_rotary_pos_emb_consecutive

            query = query.reshape((*query.shape[:2], self.num_heads, self.head_dim))
            key = key.reshape((*key.shape[:2], self.num_gqa_groups, self.head_dim))
            query = apply_rotary_pos_emb(query, q_position)
            key = apply_rotary_pos_emb(key, k_position)

        query = query.reshape((*query.shape[:2], self.num_heads, self.head_dim))
        key = key.reshape((*key.shape[:2], self.num_gqa_groups, self.head_dim))
        value = value.reshape((*value.shape[:2], self.num_gqa_groups, self.head_dim))

        if self.transpose_batch_sequence:
            query = nn_partitioning.with_sharding_constraint(
                query, ("length", "batch", "heads", "kv")
            )
            key = nn_partitioning.with_sharding_constraint(key, ("length", "batch", "heads", "kv"))
            value = nn_partitioning.with_sharding_constraint(
                value, ("length", "batch", "heads", "kv")
            )
        else:
            query = nn_partitioning.with_sharding_constraint(
                query, ("batch", "length", "heads", "kv")
            )
            key = nn_partitioning.with_sharding_constraint(key, ("batch", "length", "heads", "kv"))
            value = nn_partitioning.with_sharding_constraint(
                value, ("batch", "length", "heads", "kv")
            )

        if decode:
            # Detect if we're initializing by absence of existing cache data.
            is_initialized = self.has_variable("cache", "cached_key")
            # The key and value have dimension [batch, length, num_heads, head_dim],
            # but we cache them as [batch, num_heads, head_dim, length] as a TPU
            # fusion optimization. This also enables the "scatter via one-hot
            # broadcast" trick, which means we do a one-hot broadcast instead of a
            # scatter/gather operations, resulting in a 3-4x speedup in practice.
            swap_dims = lambda x: x[:-3] + tuple(x[i] for i in [-2, -1, -3])
            cached_key = self.variable(
                "cache", "cached_key", jnp.zeros, swap_dims(key.shape), key.dtype
            )
            cached_value = self.variable(
                "cache", "cached_value", jnp.zeros, swap_dims(value.shape), value.dtype
            )
            cache_index = self.variable(
                "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
            )
            if is_initialized:
                batch, num_heads, head_dim, length = cached_key.value.shape
                # During fast autoregressive decoding, we feed one position at a time,
                # and cache the keys and values step by step.
                # Sanity shape check of cached key against input query.
                expected_shape = (batch, 1, num_heads, head_dim)
                if expected_shape != query.shape:
                    raise ValueError(
                        "Autoregressive cache shape error, "
                        f"expected query shape {expected_shape} instead got {query.shape}."
                    )

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
                    jnp.logical_not(mask),
                    jnp.broadcast_to(
                        jnp.arange(length) <= cur_index,
                        # (1, 1, length) represent (head dim, query length, key length)
                        # query length is 1 because during decoding we deal with one
                        # index.
                        # The same mask is applied to all batch elements and heads.
                        (batch, 1, 1, length),
                    ),
                )

                # Grab the correct relative attention bias during decoding. This is
                # only required during single step decoding.
                if bias is not None:
                    # The bias is a full attention matrix, but during decoding we only
                    # have to take a slice of it.
                    # This is equivalent to bias[..., cur_index:cur_index+1, :].
                    bias = dynamic_vector_slice_in_dim(
                        jnp.squeeze(bias, axis=0), jnp.reshape(cur_index, (-1)), 1, -2
                    )

        # Convert the boolean attention mask to an attention bias.
        if mask is not None:
            # attention mask in the form of attention bias
            attention_bias = lax.select(
                mask > 0,
                jnp.full(mask.shape, 0.0).astype(self.dtype),
                jnp.full(mask.shape, -1e10).astype(self.dtype),
            )
        else:
            attention_bias = None

        # Add provided bias term (e.g. relative position embedding).
        if bias is not None:
            attention_bias = combine_biases(attention_bias, bias)

        # Apply attention.
        x = DotProductAttention(
            transpose_batch_sequence=self.transpose_batch_sequence,
            scale_attn_logits=self.scale_attn_logits,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            float32_logits=self.float32_logits,
        )(query, key, value, bias=attention_bias, deterministic=deterministic)

        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))

        if self.transpose_batch_sequence:
            x = nn_partitioning.with_sharding_constraint(x, ("length", "batch", "joined_kv"))
        else:
            x = nn_partitioning.with_sharding_constraint(x, ("batch", "length", "joined_kv"))

        # Back to the original inputs dimensions.
        out = DenseGeneral(
            features=inputs_q.shape[-1],  # output dim is set to the input dim.
            axis=-1,
            kernel_init=self.kernel_init,
            kernel_axes=("joined_kv", "embed"),
            use_bias=self.use_bias,
            bias_axes="embed",
            dtype=self.dtype,
            name="out",
        )(x)
        return out


class LayerNorm(nn.Module):
    """T5 Layer normalization operating on the last axis of the input data."""

    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    layernorm_type: str = "layernorm"
    zero_centered_gamma: bool = False
    scale_init: Initializer = None
    bias_init: Initializer = nn.initializers.zeros

    def __post_init__(self):
        if self.scale_init is None:
            if not self.zero_centered_gamma:
                self.scale_init = nn.initializers.ones
            else:
                self.scale_init = nn.initializers.zeros
        super().__post_init__()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies layer normalization on the input."""

        x = jnp.asarray(x, jnp.float32)
        features = x.shape[-1]

        scale = nn_partitioning.param_with_axes(
            "scale", self.scale_init, (features,), jnp.float32, axes=("embed",)
        )
        scale = jnp.asarray(scale, self.dtype)

        if self.layernorm_type == "layernorm":
            mean = jnp.mean(x, axis=-1, keepdims=True)
            var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
            y = (x - mean) * lax.rsqrt(var + self.epsilon)

            bias = nn_partitioning.param_with_axes(
                "ln_bias", self.bias_init, (features,), jnp.float32, axes=("embed",)
            )
            bias = jnp.asarray(bias, self.dtype)

            if not self.zero_centered_gamma:
                z = y * scale + bias
            else:
                z = y * (scale + 1.0) + bias
        else:
            assert self.layernorm_type == "rmsnorm"
            assert not self.zero_centered_gamma
            mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
            y = x * lax.rsqrt(mean2 + self.epsilon)
            z = y * scale

        return jnp.asarray(z, self.dtype)


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
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
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
            np.log(n.astype(np.float32) / max_exact + np.finfo(np.float32).eps)
            / np.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(np.int32)
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
            max_distance=self.max_distance,
        )
        relative_attention_bias = nn_partitioning.param_with_axes(
            "rel_embedding",
            self.embedding_init,
            (self.num_heads, self.num_buckets),
            jnp.float32,
            axes=("heads", "relpos_buckets"),
        )

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
            (((1,), (0,)), ((), ())),  # rhs, lhs contracting dims
        )  # no batched dims
        # Add a singleton batch dimension.
        # --> shape (1, num_heads, qlen, klen)
        return values[jnp.newaxis, ...]


class EncoderLayer(nn.Module):
    """Transformer encoder layer."""

    enable_relative_embedding: bool = True
    relative_embedding: nn.Module = None
    num_attention_heads: int = 8
    num_gqa_groups: int | None = None
    head_dim: int = 64
    hidden_dropout: float = 0.1
    hidden_dropout_dims: Sequence[int] = ()
    attention_dropout: float = 0.1
    intermediate_dropout: float = 0.1
    intermediate_dropout_dims: Sequence[int] = ()
    transpose_batch_sequence: bool = True
    float32_attention_logits: bool = False
    scale_attn_logits: bool = False
    scaled_query_init: bool = True
    mlp_dim: int = 2048
    mlp_activations: Sequence[str] = ("relu",)
    use_bias: bool = False
    dtype: Any = jnp.float32
    apply_residual_connection_post_layernorm: bool = False
    layernorm_type: str = "layernorm"
    layernorm_epsilon: float = 1e-6
    zero_centered_gamma: bool = False
    output_layernorm: bool = False
    drop_path: float = 0.0
    enable_rotary_pos_emb: bool = False
    rotary_pos_emb_group_method: str = "consecutive"
    fuse_qkv_params: bool = True
    fuse_mlp_wi: bool = True
    self_attn_bias_type: Any = None
    self_attn_mask_type: Any = None

    def __post_init__(self):
        if self.num_gqa_groups is None:
            self.num_gqa_groups = self.num_attention_heads
        super().__post_init__()

    @nn.compact
    def __call__(self, inputs, encoder_mask=None, deterministic=False):
        del self.self_attn_mask_type  # dummy, just align to TE's impl
        # Relative position embedding as attention biases.
        sequence_dim = 0 if self.transpose_batch_sequence else 1
        batch_dim = 1 - sequence_dim

        if self.enable_relative_embedding:
            if self.relative_embedding is None:
                rel_emb = RelativePositionBiases(
                    num_buckets=32,
                    max_distance=128,
                    num_heads=self.num_attention_heads,
                    dtype=self.dtype,
                    embedding_init=nn.initializers.variance_scaling(1.0, "fan_avg", "uniform"),
                    name="relpos_bias",
                )
            else:
                rel_emb = self.relative_embedding
            encoder_bias = rel_emb(inputs.shape[sequence_dim], inputs.shape[sequence_dim], True)
        else:
            encoder_bias = None

        # Attention block.
        residual = inputs

        if not self.output_layernorm:
            # Attention block.
            x = LayerNorm(
                layernorm_type=self.layernorm_type,
                epsilon=self.layernorm_epsilon,
                zero_centered_gamma=self.zero_centered_gamma,
                dtype=self.dtype,
                name="pre_attention_layer_norm",
            )(inputs)

            if self.apply_residual_connection_post_layernorm:
                residual = x
        else:
            x = inputs

        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        x = MultiHeadAttention(
            num_heads=self.num_attention_heads,
            num_gqa_groups=self.num_gqa_groups,
            dtype=self.dtype,
            head_dim=self.head_dim,
            transpose_batch_sequence=self.transpose_batch_sequence,
            dropout_rate=self.attention_dropout,
            float32_logits=self.float32_attention_logits,
            scale_attn_logits=self.scale_attn_logits,
            scaled_query_init=self.scaled_query_init,
            fuse_qkv=self.fuse_qkv_params,
            enable_rotary_pos_emb=self.enable_rotary_pos_emb,
            rotary_pos_emb_group_method=self.rotary_pos_emb_group_method,
            use_bias=self.use_bias,
            name="attention",
        )(x, x, encoder_mask, encoder_bias, deterministic=deterministic)
        x = nn.Dropout(rate=self.hidden_dropout, broadcast_dims=self.hidden_dropout_dims)(
            x, deterministic=deterministic
        )
        if self.drop_path > 0.0:
            drop_path_shape = _generate_drop_path_shape(x.shape, batch_dim)
            x = nn.Dropout(rate=self.drop_path, broadcast_dims=drop_path_shape)(
                x, deterministic=deterministic
            )
        x = x + residual

        # MLP block.
        residual = x
        y = LayerNorm(
            layernorm_type=self.layernorm_type,
            epsilon=self.layernorm_epsilon,
            zero_centered_gamma=self.zero_centered_gamma,
            dtype=self.dtype,
            name="pre_mlp_layer_norm",
        )(x)

        if self.apply_residual_connection_post_layernorm:
            residual = y

        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        y = MlpBlock(
            transpose_batch_sequence=self.transpose_batch_sequence,
            intermediate_dim=self.mlp_dim,
            activations=self.mlp_activations,
            intermediate_dropout_rate=self.intermediate_dropout,
            intermediate_dropout_dims=self.intermediate_dropout_dims,
            use_bias=self.use_bias,
            dtype=self.dtype,
            fuse_wi=self.fuse_mlp_wi,
            name="mlp",
        )(y, deterministic=deterministic)
        y = nn.Dropout(rate=self.hidden_dropout, broadcast_dims=self.hidden_dropout_dims)(
            y, deterministic=deterministic
        )
        if self.drop_path > 0.0:
            drop_path_shape = _generate_drop_path_shape(y.shape, batch_dim)
            y = nn.Dropout(rate=self.drop_path, broadcast_dims=drop_path_shape)(
                y, deterministic=deterministic
            )
        y = y + residual

        if self.output_layernorm:
            y = LayerNorm(
                layernorm_type=self.layernorm_type,
                epsilon=self.layernorm_epsilon,
                zero_centered_gamma=self.zero_centered_gamma,
                dtype=self.dtype,
                name="output_layernorm",
            )(y)
        return y


class DecoderLayer(nn.Module):
    """Transformer decoder layer that attends to the encoder."""

    enable_relative_embedding: bool = True
    relative_embedding: nn.Module = None
    num_attention_heads: int = 8
    num_gqa_groups: int | None = None
    head_dim: int = 64
    hidden_dropout: float = 0.1
    hidden_dropout_dims: Sequence[int] = ()
    attention_dropout: float = 0.1
    intermediate_dropout: float = 0.1
    intermediate_dropout_dims: Sequence[int] = ()
    transpose_batch_sequence: bool = True
    float32_attention_logits: bool = False
    scale_attn_logits: bool = False
    scaled_query_init: bool = True
    mlp_dim: int = 2048
    mlp_activations: Sequence[str] = ("relu",)
    use_bias: bool = False
    dtype: Any = jnp.float32
    apply_residual_connection_post_layernorm: bool = False
    output_layernorm: bool = False
    layernorm_type: str = "layernorm"
    layernorm_epsilon: float = 1e-6
    zero_centered_gamma: bool = False
    drop_path: float = 0.0
    enable_rotary_pos_emb: bool = False
    rotary_pos_emb_group_method: str = "consecutive"
    fuse_qkv_params: bool = True
    fuse_mlp_wi: bool = True
    self_attn_bias_type: Any = None
    self_attn_mask_type: Any = None

    def __post_init__(self):
        if self.num_gqa_groups is None:
            self.num_gqa_groups = self.num_attention_heads
        super().__post_init__()

    @nn.compact
    def __call__(
        self,
        inputs,
        encoded,
        decoder_mask=None,
        encoder_decoder_mask=None,
        deterministic=False,
        decode=False,
        max_decode_length=None,
    ):
        del self.self_attn_mask_type  # dummy, just align to TE's impl
        # Relative position embedding as attention biases.
        sequence_dim = 0 if self.transpose_batch_sequence else 1
        batch_dim = 1 - sequence_dim

        if self.enable_relative_embedding:
            l = max_decode_length if decode and max_decode_length else inputs.shape[sequence_dim]
            if self.relative_embedding is None:
                rel_emb = RelativePositionBiases(
                    num_buckets=32,
                    max_distance=128,
                    num_heads=self.num_attention_heads,
                    dtype=self.dtype,
                    embedding_init=nn.initializers.variance_scaling(1.0, "fan_avg", "uniform"),
                    name="relpos_bias",
                )
            else:
                rel_emb = self.relative_embedding
            decoder_bias = rel_emb(l, l, False)
        else:
            decoder_bias = None

        # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
        residual = inputs

        if not self.output_layernorm:
            # Attention block.
            x = LayerNorm(
                layernorm_type=self.layernorm_type,
                epsilon=self.layernorm_epsilon,
                zero_centered_gamma=self.zero_centered_gamma,
                dtype=self.dtype,
                name="pre_self_attention_layer_norm",
            )(inputs)

            if self.apply_residual_connection_post_layernorm:
                residual = x
        else:
            x = inputs

        # Self-attention block
        x = MultiHeadAttention(
            num_heads=self.num_attention_heads,
            num_gqa_groups=self.num_gqa_groups,
            dtype=self.dtype,
            head_dim=self.head_dim,
            transpose_batch_sequence=self.transpose_batch_sequence,
            dropout_rate=self.attention_dropout,
            float32_logits=self.float32_attention_logits,
            scale_attn_logits=self.scale_attn_logits,
            scaled_query_init=self.scaled_query_init,
            enable_rotary_pos_emb=self.enable_rotary_pos_emb,
            rotary_pos_emb_group_method=self.rotary_pos_emb_group_method,
            fuse_qkv=self.fuse_qkv_params,
            use_bias=self.use_bias,
            name="self_attention",
        )(x, x, decoder_mask, decoder_bias, deterministic=deterministic, decode=decode)
        x = nn.Dropout(rate=self.hidden_dropout, broadcast_dims=self.hidden_dropout_dims)(
            x, deterministic=deterministic
        )
        if self.drop_path > 0.0:
            drop_path_shape = _generate_drop_path_shape(x.shape, batch_dim)
            x = nn.Dropout(rate=self.drop_path, broadcast_dims=drop_path_shape)(
                x, deterministic=deterministic
            )
        x = x + residual

        # Encoder-Decoder block.
        residual = x
        y = LayerNorm(
            layernorm_type=self.layernorm_type,
            epsilon=self.layernorm_epsilon,
            zero_centered_gamma=self.zero_centered_gamma,
            dtype=self.dtype,
            name="pre_cross_attention_layer_norm",
        )(x)

        if self.apply_residual_connection_post_layernorm:
            residual = y
        y = MultiHeadAttention(
            num_heads=self.num_attention_heads,
            num_gqa_groups=self.num_gqa_groups,
            dtype=self.dtype,
            head_dim=self.head_dim,
            transpose_batch_sequence=self.transpose_batch_sequence,
            dropout_rate=self.attention_dropout,
            float32_logits=self.float32_attention_logits,
            scale_attn_logits=self.scale_attn_logits,
            scaled_query_init=self.scaled_query_init,
            enable_rotary_pos_emb=self.enable_rotary_pos_emb,
            rotary_pos_emb_group_method=self.rotary_pos_emb_group_method,
            fuse_qkv=self.fuse_qkv_params,
            use_bias=self.use_bias,
            name="encoder_decoder_attention",
        )(y, encoded, encoder_decoder_mask, deterministic=deterministic)
        y = nn.Dropout(rate=self.hidden_dropout, broadcast_dims=self.hidden_dropout_dims)(
            y, deterministic=deterministic
        )
        y = y + residual

        # MLP block.
        residual = y
        z = LayerNorm(
            layernorm_type=self.layernorm_type,
            epsilon=self.layernorm_epsilon,
            zero_centered_gamma=self.zero_centered_gamma,
            dtype=self.dtype,
            name="pre_mlp_layer_norm",
        )(y)
        if self.apply_residual_connection_post_layernorm:
            residual = z
        z = MlpBlock(
            transpose_batch_sequence=self.transpose_batch_sequence,
            intermediate_dim=self.mlp_dim,
            activations=self.mlp_activations,
            intermediate_dropout_rate=self.intermediate_dropout,
            intermediate_dropout_dims=self.intermediate_dropout_dims,
            use_bias=self.use_bias,
            dtype=self.dtype,
            fuse_wi=self.fuse_mlp_wi,
            name="mlp",
        )(z, deterministic=deterministic)
        z = nn.Dropout(rate=self.hidden_dropout, broadcast_dims=self.hidden_dropout_dims)(
            z, deterministic=deterministic
        )
        if self.drop_path > 0.0:
            drop_path_shape = _generate_drop_path_shape(z.shape, batch_dim)
            z = nn.Dropout(rate=self.drop_path, broadcast_dims=drop_path_shape)(
                z, deterministic=deterministic
            )
        z = z + residual

        if self.output_layernorm:
            z = LayerNorm(
                layernorm_type=self.layernorm_type,
                epsilon=self.layernorm_epsilon,
                zero_centered_gamma=self.zero_centered_gamma,
                dtype=self.dtype,
                name="output_layernorm",
            )(z)

        return z


def make_causal_mask(batch, seqlen, dtype=jnp.uint8):
    """
    Generate causal mask
    """
    shape = (batch, seqlen)
    idxs = jnp.broadcast_to(jnp.arange(shape[-1], dtype=jnp.int32), shape)

    mask = jnp.greater_equal(jnp.expand_dims(idxs, axis=-1), jnp.expand_dims(idxs, axis=-2))
    mask = jnp.expand_dims(mask, axis=-3)
    mask = 1 - mask
    return mask.astype(dtype)


def make_self_mask(batch, seqlen, dtype=jnp.uint8):
    """
    Generate attention mask
    """
    shape = (batch, seqlen)
    mask = jnp.ones((*shape, shape[-1]))
    mask = jnp.expand_dims(mask, axis=-3)
    mask = 1 - mask
    return mask.astype(dtype)


def assert_allclose(
    actual: Array,
    desired: Array,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    dtype: Optional[Union[DType, TEDType, np.dtype, str]] = None,
    **kwargs,
) -> None:
    """Check if two tensors are close.

    Args:
      actual: test tensor.
      desired: reference tensor.
      dtype: data type or data type name (default: inferred from
        `actual`).
      rtol: relative tolerance (default: based on `dtype`).
      atol: absolute tolerance (default: based on `dtype`).
      **kwargs: keyword arguments to pass to np.testing.assert_allclose.
    """

    # Infer data type if needed
    if dtype is None:
        if isinstance(actual, float):
            dtype = "float32"
        else:
            dtype = actual.dtype

    # Determine tolerances
    tols = {}
    if rtol is None or atol is None:
        tols = dtype_tols(dtype)
    if rtol is not None:
        tols["rtol"] = rtol
    if atol is not None:
        tols["atol"] = atol

    # Cast tensors to fp32
    if not isinstance(actual, float):
        actual = actual.astype(jnp.float32)
    if not isinstance(desired, float):
        desired = desired.astype(jnp.float32)

    # Check if tensors are close
    np.testing.assert_allclose(actual, desired, **tols, **kwargs)


def assert_tree_like_allclose(expected, actual, rtol=1e-05, atol=1e-08):
    flatten_expected, _ = jax.tree_util.tree_flatten_with_path(expected)
    flatten_actual, _ = jax.tree_util.tree_flatten_with_path(actual)

    for (expected_path, expected_value), (actual_path, actual_value) in zip(
        flatten_expected, flatten_actual
    ):
        assert expected_path == actual_path
        key_str = jax.tree_util.keystr(expected_path)
        assert_allclose(
            expected_value,
            actual_value,
            rtol=rtol,
            atol=atol,
            err_msg=f"Value of expected{key_str} and actual{key_str} is not close",
        )


def dtype_tols(
    dtype: Union[DType, TEDType, np.dtype],
    reference_value: float = 1.0,
) -> Dict[str, float]:
    """Expected numerical tolerance for a data type.

    Args:
      dtype: data type.
      reference_value: reference value (default: 1).

    Returns:
      Dictionary with "rtol" and "atol" as keys

    """

    # Convert to JAX dtype if needed
    if isinstance(dtype, TEDType):
        dtype = {
            TEDType.kByte: jnp.uint8,
            TEDType.kInt32: jnp.int32,
            TEDType.kInt64: jnp.int64,
            TEDType.kFloat32: jnp.float32,
            TEDType.kFloat16: jnp.float16,
            TEDType.kBFloat16: jnp.bfloat16,
            TEDType.kFloat8E4M3: jnp.float8_e4m3fn,
            TEDType.kFloat8E5M2: jnp.float8_e5m2,
        }[dtype]
    elif isinstance(dtype, np.dtype):
        dtype = jnp.dtype(dtype)

    # Expect bit-wise accuracy for integer dtypes
    if not jnp.issubdtype(dtype, jnp.floating):
        return dict(rtol=0, atol=0)

    # Estimate floating-point error
    finfo = jnp.finfo(dtype)
    eps_relaxed = math.pow(finfo.eps, 2 / 3)
    with jax.default_device(jax.devices("cpu")[0]):
        if isinstance(reference_value, (float, int)):
            reference_value = jnp.array(reference_value, dtype=dtype)
        else:
            reference_value = reference_value.astype(dtype)
        spacing_high = jnp.nextafter(reference_value, finfo.max) - reference_value
        spacing_low = reference_value - jnp.nextafter(reference_value, finfo.min)
        ulp = max(spacing_high.item(), spacing_low.item())
    return dict(
        rtol=eps_relaxed,
        atol=max(ulp, eps_relaxed),
    )


def sync_params_values(dst, src, transformations, sep="/"):
    """
    This function will reconstuct a tree with dst's tree_def/shape and src's value.
    transformations is a map that records the key mappings between dst and src.
    If no dst key found in the transformerations, it will fall back to src key = dst key.
    transformations = {
        dst key map 0: src key map 0,
        dst key map 1: src key map 1,
        ...
        # if dst key = src key, we don't need to add it
    }
    """
    src_values = {}
    for key, value in jax.tree_util.tree_leaves_with_path(src):
        normalized_key = sep.join(x.key for x in key)
        src_values[normalized_key] = value

    flatten_dst, dst_tree_def = jax.tree_util.tree_flatten_with_path(dst)
    synced_dst_values = []

    for key, value in flatten_dst:
        normalized_key = sep.join(x.key for x in key)
        if normalized_key in transformations:
            corresponding_src_key = transformations[normalized_key]
        else:
            corresponding_src_key = normalized_key
        synced_dst_values.append(src_values[corresponding_src_key])

    synced_dst = jax.tree_util.tree_unflatten(dst_tree_def, synced_dst_values)

    return jax.tree_util.tree_map(lambda x, y: x.reshape(y.shape), synced_dst, dst)

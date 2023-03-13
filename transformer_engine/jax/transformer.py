# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Wrapper module for Transformer related layers with FP8 support.
"""
import functools
from enum import Enum
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from jax import nn as jax_nn
from jax import random as jax_random
from jax import lax, vmap

from .module import DenseGeneral, LayerNormDenseGeneral, LayerNormMLP
from .module import LayerNorm, Softmax
from .softmax import SoftmaxType
from .sharding import infer_major_sharding_type, infer_sharding_type
from .sharding import global_shard_resource, ShardingType

PRNGKey = Any
Shape = Tuple[int, ...]
DType = jnp.dtype
Array = jnp.ndarray
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision,
                                                                       lax.Precision]]
Initializer = Callable[[PRNGKey, Shape, DType], Array]
LogicalRules = Sequence[Tuple[str, Union[str, None]]]


def _generate_drop_path_shape(shape: Sequence[int], batch_dim: int) -> Sequence[int]:
    # Generate broadcast dims for drop_path.
    drop_path_shape = list(range(0, len(shape)))
    drop_path_shape.pop(batch_dim)
    return drop_path_shape


def extend_logical_axis_rules(rules: LogicalRules) -> LogicalRules:
    """
    Extend the given Flax logical axis rules with the predefined TransformerLayer's
    logical axis rules.

    .. note::
        We currently only support single, data parallelism and standard tensor parallelism
        logical axis rules for performance reasons.

    .. warning::
        Please make sure ShardingResource is set via fp8_autocast before calling this function.

    Parameters
    ----------
    rules : Sequence[Tuple[str, Union[str, None]]]
        the base Flax logical axis rules to extend.

    Returns
    -------
    extended_rules : Sequence[Tuple[str, Union[str, None]]]
        the extended Flax logical axis rules.
    """
    rules_map = {}
    for item in rules:
        assert len(item) == 2, \
            "The logical axis rule should be like (axis_name, mesh_axis_name)."
        key = item[0]
        val = item[1]
        assert isinstance(key, str), \
            f"Thie axis_name should be str, but got {type(key)}."
        assert isinstance(val, str) or (val is None), \
            f"Thie mesh_axis_name should be str or None, but got {type(val)}."
        rules_map[key] = val

    gsr = global_shard_resource()

    te_logical_axis_rules = (('batch', gsr.dp_resource), ('embed', None), ('mlp', gsr.tp_resource),
                             ('heads', gsr.tp_resource), ('kv', None), ('qkv_dim', None),
                             ('kv_dim', None), ('joined_kv', gsr.tp_resource), ('act', None),
                             ('relpos_buckets', None), ('length', None))

    extended_rules = [*rules]
    for item in te_logical_axis_rules:
        key = item[0]
        val = item[1]
        if key in rules_map:
            assert rules_map[key] == val, \
                f"The rule diverged between TE and given rule." \
                f"Axis:{key} map to {rules_map[key]} in the given" \
                f" rules, but {val} in TE's rules."
        else:
            extended_rules.append(item)
    return tuple(extended_rules)


def _merge_mask(func, *masks: Optional[Array]):
    masks = [m for m in masks if m is not None]
    if not masks:
        return None
    assert all(map(lambda x: x.ndim == masks[0].ndim,
                   masks)), (f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
    mask, *other_masks = masks
    for other_mask in other_masks:
        mask = func(mask, other_mask)
    return mask


def combine_masks(*masks: Optional[Array], dtype: DType = jnp.float32):
    """Combine attention masks."""
    func = jnp.logical_and
    return _merge_mask(func, *masks).astype(dtype)


def combine_biases(*masks: Optional[Array]):
    """Combine attention biases."""
    func = lambda a, b: a + b
    return _merge_mask(func, *masks)


def core_attention(query: Array,
                   key: Array,
                   value: Array,
                   transpose_batch_sequence: bool,
                   softmax_type: SoftmaxType = SoftmaxType.SCALED,
                   softmax_sharding_type: ShardingType = ShardingType.SINGLE,
                   mask: Optional[Array] = None,
                   bias: Optional[Array] = None,
                   dropout_rng: Optional[PRNGKey] = None,
                   dropout_rate: float = 0.,
                   deterministic: bool = False,
                   dtype: DType = jnp.float32,
                   float32_logits: bool = False):
    """Core attention"""
    assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
    batch_dim = 1 if transpose_batch_sequence else 0
    assert query.shape[batch_dim] == key.shape[batch_dim] == value.shape[batch_dim], (
        'q, k, v batch dims must match.')
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], ('q, k, v num_heads must match.')
    sequence_dim = 0 if transpose_batch_sequence else 1
    assert key.shape[sequence_dim] == value.shape[sequence_dim], 'k, v lengths must match.'
    assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

    if float32_logits:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)

    if transpose_batch_sequence:
        attn_weights = jnp.einsum('qbhd,kbhd->bhqk', query, key)
    else:
        attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)

    attn_weights = Softmax(softmax_type=softmax_type,
                           sharding_type=softmax_sharding_type)(attn_weights, mask, bias)

    if not deterministic and dropout_rate > 0.:
        keep_prob = 1.0 - dropout_rate
        dropout_shape = list(attn_weights.shape)
        dropout_shape[-2] = 1
        keep = jax_random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        keep = jnp.broadcast_to(keep, attn_weights.shape)
        multiplier = (keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
        attn_weights = attn_weights * multiplier

    if transpose_batch_sequence:
        return jnp.einsum('bhqk,kbhd->qbhd', attn_weights, value)

    return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)


dynamic_vector_slice_in_dim = vmap(lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))


class AttentionType(Enum):
    """TransformerLayerType."""
    PADDING = "padding_attention"
    CAUSAL = "causal_attention"


class MultiHeadAttention(nn.Module):
    r"""
    Multi-head Attention (MHA), including Query,
    Key, Value and Output projection.

    Parameters
    ----------
    head_dim : int
        The hidden dimension of each attention head.
    num_heads : int
        The number of attention heads
    dropout_rate : float, default = 0.0
        Dropout probability for the dropout op during multi-head attention.
    dropout_rng_name: str, default = 'dropout'
        The key in given RNGs via flax.linen.Module.apply that
        to generate Dropout masks in the core attention.
    layernorm_type : {'layernorm', 'rmsnorm'}, default = 'layernorm'
        Indicate the type of layer normalization.
    layernorm_epsilon: float, default = 1e-6
        A value added to the denominator of layer normalization for numerical stability.
    kernel_init: Initializer, default =
        flax.linen.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        Used for initializing weights of QKV and Output projection weights.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    use_bias: bool, default = False
        Indicate whether to enable bias shifting for QKVO projections.
        If set to False, the layer will not learn additive biases.
    bias_init: Initializer, default = flax.linen.initializers.zeros
        Used for initializing bias of QKVO projections, it only works when :attr:`use_bias=True`.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    apply_residual_connection_post_layernorm : bool, default = False
        Indicate if apply residual connection with the output of layer normalization.
    output_layernorm : bool, default = False
        Indicate if apply a layer normalization at the end of MHA.
    attn_type: AttentionType, defult = AttentionType.PADDING
        Indicate the format of the attention mask in the core attention.

    Optimization parameters
    -----------------------
    dtype :jax.numpy.dtype, default  = jax.numpy.float32
        The data type used to allocate the initial parameters.
    fuse_qkv: bool, default = True
        If set to True, this module exposes a single fused
        parameter for query-key-value for self-attention and key-value for
        cross-attention.
    transpose_batch_sequence : bool, default = True
        Indicate whether the input tensors were switched axis of batch
        and sequence length dimension. if set to True, the input tensors
        should be in (seqlen, batch, hidden), otherwise (batch, seqlen, hidden).
    scale_attn_logits: bool, default = False
        Indicate whether to scale attention logits.
        If set to True, :math:`\frac{Q}{\sqrt{head_dim}*K}`,
        else :math:`Q*K`
    scaled_query_init: bool, default = `True`
        Whether to scale WQ on initialization by :math:`\sqrt{head_dim}`
    float32_logits : bool, default = False
        Whether to compute attention logits in float32.
    """

    head_dim: int
    num_heads: int
    dropout_rate: float = 0.
    dropout_rng_name: str = 'dropout'
    layernorm_type: str = "layernorm"
    layernorm_epsilon: float = 1e-6
    kernel_init: Initializer = None
    use_bias: bool = False
    bias_init: Initializer = nn.initializers.zeros
    apply_residual_connection_post_layernorm: bool = False
    output_layernorm: bool = False
    attn_type: AttentionType = AttentionType.PADDING
    dtype: DType = jnp.float32
    fuse_qkv: bool = True
    transpose_batch_sequence: bool = True
    scale_attn_logits: bool = False
    scaled_query_init: bool = True
    float32_logits: bool = False    # computes logits in float32 for stability.

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
        """
        MultiHeadAttention Layer:
        [Query, Key, Value projection] -> Dot Product Attention -> Output projection.

        Parameters
        ----------
        inputs_q : jax.numpy.ndarray
            Input tensor for query projection.
        inputs_kv : jax.numpy.ndarray
            Input tensor for key/value projection.
        mask : jax.numpy.ndarray, default = None
            Boolean tensor used to mask out self-attention softmax input.
        bias : jax.numpy.ndarray, default = None
            A tensor used to shift self-attention softmax input.
        *
        decode : bool,default = False
            Indicate whether to prepare and use an autoregressive cache.
        deterministic : bool,default = False
            Disable dropout layers if set to True.

        Returns
        -------
        outputs : jax.numpy.ndarray
            Output tensors.
        """

        depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)

        def query_init(*args):
            return self.kernel_init(*args) / (depth_scaling if self.scaled_query_init else 1.0)

        def qkv_init(key, shape, dtype):
            assert len(shape) == 3
            assert shape[-2] == 3

            q_key, k_key, v_key = jax_random.split(key, num=3)

            q_shape = (shape[0], shape[-1])
            k_shape = (shape[0], shape[-1])
            v_shape = (shape[0], shape[-1])

            q_kernel = query_init(q_key, q_shape, dtype)
            k_kernel = self.kernel_init(k_key, k_shape, dtype)
            v_kernel = self.kernel_init(v_key, v_shape, dtype)

            return jnp.stack([q_kernel, k_kernel, v_kernel], axis=-2, dtype=dtype)

        def kv_init(key, shape, dtype):
            assert len(shape) == 3
            assert shape[-2] == 2

            k_key, v_key = jax_random.split(key)

            k_shape = (shape[0], shape[-1])
            v_shape = (shape[0], shape[-1])

            k_kernel = self.kernel_init(k_key, k_shape, dtype)
            v_kernel = self.kernel_init(v_key, v_shape, dtype)

            return jnp.stack([k_kernel, v_kernel], axis=-2, dtype=dtype)

        first_sharding_type, second_sharding_type = infer_sharding_type()

        residual = inputs_q
        if self.fuse_qkv:
            if inputs_q is inputs_kv:
                qkv_proj, ln_out = LayerNormDenseGeneral(
                    enable_layernorm=not self.output_layernorm,
                    layernorm_type=self.layernorm_type,
                    epsilon=self.layernorm_epsilon,
                    axis=-1,
                    features=(3, self.num_heads * self.head_dim),
                    sharding_type=first_sharding_type,
                    transpose_batch_sequence=self.transpose_batch_sequence,
                    return_layernorm_output=self.apply_residual_connection_post_layernorm,
                    scale_axes=('embed',),
                    kernel_axes=('embed', 'qkv_dim', 'joined_kv'),
                    kernel_init=qkv_init,
                    use_bias=self.use_bias,
                    bias_init=self.bias_init,
                    name='qkv',
                    dtype=self.dtype)(inputs_q)
                query, key, value = jnp.split(qkv_proj, [1, 2], axis=-2)
                query = jnp.reshape(query, (*query.shape[:-2], -1))
                key = jnp.reshape(key, (*key.shape[:-2], -1))
                value = jnp.reshape(value, (*value.shape[:-2], -1))
                if self.scale_attn_logits:
                    query = query / depth_scaling
            else:
                query, ln_out = LayerNormDenseGeneral(
                    enable_layernorm=not self.output_layernorm,
                    layernorm_type=self.layernorm_type,
                    epsilon=self.layernorm_epsilon,
                    axis=-1,
                    features=self.num_heads * self.head_dim,
                    sharding_type=first_sharding_type,
                    transpose_batch_sequence=self.transpose_batch_sequence,
                    return_layernorm_output=self.apply_residual_connection_post_layernorm,
                    depth_scaling=depth_scaling if self.scale_attn_logits else None,
                    scale_axes=('embed',),
                    kernel_axes=('embed', 'joined_kv'),
                    use_bias=self.use_bias,
                    bias_init=self.bias_init,
                    dtype=self.dtype,
                    kernel_init=query_init,
                    name='query')(inputs_q)
                kv_proj = DenseGeneral(axis=-1,
                                       features=(2, self.num_heads * self.head_dim),
                                       sharding_type=first_sharding_type,
                                       transpose_batch_sequence=self.transpose_batch_sequence,
                                       kernel_axes=('embed', 'kv_dim', 'joined_kv'),
                                       kernel_init=kv_init,
                                       use_bias=self.use_bias,
                                       bias_init=self.bias_init,
                                       name='kv',
                                       dtype=self.dtype)(inputs_kv)
                key, value = jnp.split(kv_proj, [
                    1,
                ], axis=-2)
                key = jnp.reshape(key, (*key.shape[:-2], -1))
                value = jnp.reshape(value, (*value.shape[:-2], -1))
        else:
            kv_projection = functools.partial(
                DenseGeneral,
                axis=-1,
                features=self.num_heads * self.head_dim,
                sharding_type=first_sharding_type,
                transpose_batch_sequence=self.transpose_batch_sequence,
                kernel_axes=('embed', 'joined_kv'),
                use_bias=self.use_bias,
                bias_init=self.bias_init,
                dtype=self.dtype)
            query, ln_out = LayerNormDenseGeneral(
                enable_layernorm=not self.output_layernorm,
                layernorm_type=self.layernorm_type,
                epsilon=self.layernorm_epsilon,
                axis=-1,
                features=self.num_heads * self.head_dim,
                sharding_type=first_sharding_type,
                transpose_batch_sequence=self.transpose_batch_sequence,
                return_layernorm_output=True,
                depth_scaling=depth_scaling if self.scale_attn_logits else None,
                scale_axes=('embed',),
                kernel_axes=('embed', 'joined_kv'),
                use_bias=self.use_bias,
                bias_init=self.bias_init,
                dtype=self.dtype,
                kernel_init=query_init,
                name='query')(inputs_q)

            if inputs_q is inputs_kv:
                assert ln_out is not None
                inputs_kv = ln_out

            key = kv_projection(kernel_init=self.kernel_init, name='key')(inputs_kv)
            value = kv_projection(kernel_init=self.kernel_init, name='value')(inputs_kv)

        query = query.reshape((query.shape[0], query.shape[1], self.num_heads, self.head_dim))
        key = key.reshape((key.shape[0], key.shape[1], self.num_heads, self.head_dim))
        value = value.reshape((value.shape[0], value.shape[1], self.num_heads, self.head_dim))

        if self.apply_residual_connection_post_layernorm:
            assert ln_out is not None
            residual = ln_out

        qkv_sharding_constraint = \
            ('length', 'batch', 'heads','kv') \
            if self.transpose_batch_sequence \
            else ('batch', 'length', 'heads', 'kv')
        query = nn_partitioning.with_sharding_constraint(query, qkv_sharding_constraint)
        key = nn_partitioning.with_sharding_constraint(key, qkv_sharding_constraint)
        value = nn_partitioning.with_sharding_constraint(value, qkv_sharding_constraint)

        if decode:
            is_initialized = self.has_variable('cache', 'cached_key')

            # TODO (Ming Huang): Check performance on GPU withou swap dimensions # pylint: disable=fixme
            def swap_dims(x):
                return x[:-3] + tuple(x[i] for i in [-2, -1, -3])

            cached_key = self.variable('cache', 'cached_key', jnp.zeros, swap_dims(key.shape),
                                       key.dtype)
            cached_value = self.variable('cache', 'cached_value', jnp.zeros, swap_dims(value.shape),
                                         value.dtype)
            cache_index = self.variable('cache', 'cache_index',
                                        lambda: jnp.array(0, dtype=jnp.int32))
            if is_initialized:
                batch, num_heads, head_dim, length = cached_key.value.shape

                # Sanity shape check of cached key against input query.
                expected_shape = (batch, 1, num_heads, head_dim)
                if expected_shape != query.shape:
                    raise ValueError(
                        'Autoregressive cache shape error, '
                        f"expected query shape {expected_shape} instead got {query.shape}.")

                cur_index = cache_index.value
                one_hot_indices = jax_nn.one_hot(cur_index, length, dtype=key.dtype)
                one_token_key = jnp.moveaxis(key, -3, -1)
                one_token_value = jnp.moveaxis(value, -3, -1)
                key = cached_key.value + one_token_key * one_hot_indices
                value = cached_value.value + one_token_value * one_hot_indices
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1

                key = jnp.moveaxis(key, -1, -3)
                value = jnp.moveaxis(value, -1, -3)

                mask = combine_masks(
                    mask, jnp.broadcast_to(jnp.arange(length) <= cur_index, (batch, 1, 1, length)))

                if bias is not None:
                    bias = dynamic_vector_slice_in_dim(jnp.squeeze(bias, axis=0),
                                                       jnp.reshape(cur_index, (-1)), 1, -2)

        dropout_rng = None
        if not deterministic and self.dropout_rate > 0.:
            dropout_rng = self.make_rng(self.dropout_rng_name)

        softmax_type = SoftmaxType.SCALED
        if self.attn_type is AttentionType.PADDING:
            if mask is not None:
                softmax_type = SoftmaxType.SCALED_MASKED
        else:
            softmax_type = SoftmaxType.SCALED_UPPER_TRIANG_MASKED

        x = core_attention(query,
                           key,
                           value,
                           transpose_batch_sequence=self.transpose_batch_sequence,
                           softmax_type=softmax_type,
                           softmax_sharding_type=first_sharding_type,
                           mask=mask,
                           bias=bias,
                           dropout_rng=dropout_rng,
                           dropout_rate=self.dropout_rate,
                           deterministic=deterministic,
                           dtype=self.dtype,
                           float32_logits=self.float32_logits)

        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))

        attn_context_sharding_constraint = \
            ('length', 'batch', 'joined_kv') \
            if self.transpose_batch_sequence \
            else ('batch', 'length', 'joined_kv')
        x = nn_partitioning.with_sharding_constraint(x, attn_context_sharding_constraint)

        out = DenseGeneral(features=inputs_q.shape[-1],
                           sharding_type=second_sharding_type,
                           transpose_batch_sequence=self.transpose_batch_sequence,
                           axis=-1,
                           kernel_init=self.kernel_init,
                           kernel_axes=('joined_kv', 'embed'),
                           use_bias=self.use_bias,
                           bias_init=self.bias_init,
                           dtype=self.dtype,
                           name='out')(x)
        return out, residual


class RelativePositionBiases(nn.Module):
    """
    T5-style relative positional embeddings to the attention logits.

    Parameters
    ----------
    num_buckets : int
        The number of buckets to bucket distances between key and query positions into.
    max_distance : int
        The maximum distance before everything is lumped into the last
        distance bucket.
    num_attention_heads : int
        Number of attention heads in the transformer layer.
    embedding_init : Initializer, default = flax.linen.linear.default_embed_init
        Used for initializing relative embedding tables.
    embedding_axes : Tuple[str, ...], default = ('heads', 'relpos_buckets')
        The name of axes used to shard embedding attention bias with a corresponding mesh.

    Optimization parameters
    -----------------------
    dtype : jax.numpy.dtype, default  = jax.numpy.float32
        The data type used to allocate the initial parameters.
    """
    num_buckets: int
    max_distance: int
    num_attention_heads: int
    embedding_init: Callable[..., Array] = nn.linear.default_embed_init
    embedding_axes: Tuple[str, ...] = ('heads', 'relpos_buckets')
    dtype: DType = jnp.float32

    @nn.compact
    def __call__(self, q_seqlen, k_seqlen, bidirectional=True):
        """
        Generate relative position embedding attention biases.

        Parameters
        ----------
        q_seqlen : int
            The sequence length of query.
        k_seqlen : int
            The sequence length of key.
        bidirectional : bool, default = True
            Indicate whether to allow positive memory-query relative position
            embeddings.

        Returns
        -------
        output: jax.numpy.ndarray
            An attention bias with shape `(1, num_attention_heads, q_seqlen, k_seqlen)`.
        """
        context_position = np.arange(q_seqlen, dtype=jnp.int32)[:, None]
        memory_position = np.arange(k_seqlen, dtype=jnp.int32)[None, :]
        relative_position = memory_position - context_position

        # Compute relative position bucket
        rp_bucket = 0
        negative_rp = -relative_position
        rpb_num_buckets = self.num_buckets

        if bidirectional:
            rpb_num_buckets //= 2
            rp_bucket += (negative_rp < 0).astype(np.int32) * rpb_num_buckets
            negative_rp = np.abs(negative_rp)
        else:
            negative_rp = np.maximum(negative_rp, 0)

        rpb_max_exact = rpb_num_buckets // 2
        rpb_is_small = negative_rp < rpb_max_exact
        rpb_val_if_large = rpb_max_exact + (
            np.log(negative_rp.astype(np.float32) / rpb_max_exact + np.finfo(np.float32).eps) /
            np.log(self.max_distance / rpb_max_exact) *
            (rpb_num_buckets - rpb_max_exact)).astype(np.int32)
        rpb_val_if_large = np.minimum(rpb_val_if_large, rpb_num_buckets - 1)
        rp_bucket += np.where(rpb_is_small, negative_rp, rpb_val_if_large)

        # Compute relative attention bias
        relative_attention_bias = nn_partitioning.param_with_axes(
            'rel_embedding',
            self.embedding_init, (self.num_attention_heads, self.num_buckets),
            jnp.float32,
            axes=self.embedding_axes)

        relative_attention_bias = jnp.asarray(relative_attention_bias, self.dtype)

        bcast_iota = lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1), 0)
        rp_bucket_one_hot = jnp.array(rp_bucket[jnp.newaxis, ...] == bcast_iota, dtype=self.dtype)

        values = lax.dot_general(relative_attention_bias, rp_bucket_one_hot,
                                 (((1,), (0,)), ((), ())))
        return values[jnp.newaxis, ...]


class TransformerLayerType(Enum):
    r"""
    TransformerLayerType is an Enum class to specify a type of TransformerLayer

    Values
    ----------
    ENCODER:
        Encoder type of TransformerLayer.
    DECODER:
        Decoder type of TransformerLayer.
    """
    ENCODER = "encoder"
    DECODER = "decoder"


class TransformerLayer(nn.Module):
    r"""
    TransformerLayer is made up of a relative embedding,
    an attention block and a feedforward network (MLP).
    This standard layer is based on the paper “Attention Is All You Need”.

    Parameters
    ----------
    hidden_size: int, default = 512
        The hidden size of each input sample.
    mlp_hidden_size: int, default = 2048
        Intermediate size to which input samples are projected.
    num_attention_heads: int, default = 8
        Number of attention heads in the transformer layer.
    layernorm_type : {'layernorm', 'rmsnorm'}, default = 'layernorm'
        Indicate the type of layer normalization.
    layernorm_epsilon: float, default = 1e-6
        A value added to the denominator of layer normalization for numerical stability.
    hidden_dropout: float, default = 0.1
        Dropout probability for the dropout op after FC2 layer.
    hidden_dropout_dims: Sequence[int], default = ()
        Dimensions that will share the same dropout mask for hidden
    attention_dropout: float, default = 0.1
        Dropout probability for the dropout op during multi-head attention.
    dropout_rng_name: str, default = 'dropout'
        The key in given RNGs via flax.linen.Module.apply that for
        generating Dropout masks in the Multi-Head Attention.
    mha_kernel_init: Initializer, default =
        flax.linen.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        Used for initializing weights of QKV and Output projection weights.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    mlp_kernel_init: Initializer, default =
        flax.linen.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal')
        Used for initializing weights of FC1 and FC2 layers.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    mlp_activations: Sequence[str], default = ('relu', )
        The sequence of activation functions to apply after the first linear transformation.
        Each activation has its own transformation layer.
    use_bias: bool, default = False
        Indicate whether to enable bias shifting for QKVO projections, FC1 and FC2.
        If set to False, the layer will not learn additive biases.
    bias_init: Initializer, default = flax.linen.initializers.zeros
        Used for initializing bias of QKVO projections,
        FC1 and FC2. It only works when :attr:`use_bias=True`.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    apply_residual_connection_post_layernorm: bool, default = False
        If set to True, residual connections are taken from the output
        of layer norm (default is taken from input of layer norm)
    output_layernorm: bool, default = False
        If set to True, layer normalization is applied on the output side,
        after the final dropout-add. default behavior is to apply layer
        normalization on the input side, before the QKV transformation.
    float32_attention_logits: bool, default = False
        If set to True, attention logits are executed in jax.numpy.float32.
    layer_type: TransformerLayerType, default = TransformerLayerType.ENCODER
        If set to TransformerLayerType.DECODER, an additional cross-attention block
        is added after self-attention.this can be used for structures like `T5`
        Transformer in conjunction with the TransformerLayerType.ENCODER option.
    enable_relative_embedding: bool, default = True
        Whether to enable relative embedding as shifting of attention logits.
    relative_embedding: flax.linen.Module, default = None
        The module for relative embedding execution, only works when
        :attr:`enable_relative_embedding=True`. Default is None, which will create
        an instance of RelativePositionBiases if :attr:`enable_relative_embedding=True`.
        Default: RelativePositionBiases( num_buckets=32, max_distance=128,
        num_attention_heads=self.num_attention_heads, dtype=self.dtype,
        embedding_init=flax.linen.initializers.variance_scaling(1.0, 'fan_avg', 'uniform'),
        name='relpos_bias')

    Optimization parameters
    -----------------------
    dtype :jax.numpy.dtype, default  = jax.numpy.float32
        The data type used to allocate the initial parameters.
    drop_path: float, default = 0.0
        When > 0.0, applies stochastic depth per sample in the main
        path of the residual block.
    fuse_qkv_params: bool, default = True
        If set to True, `TransformerLayer` module exposes a single fused
        parameter for query-key-value for self-attention and key-value for
        cross-attention.
    transpose_batch_sequence : bool, default = True
        Indicate whether the input tensors were switched axis of batch
        and sequence length dimension. if set to True, the input tensors
        should be in (seqlen, batch, hidden), otherwise (batch, seqlen, hidden).
    scale_attn_logits: bool, default = False
        Indicate whether to scale attention logits.
        if set to True, :math:`\frac{Q}{\sqrt{head_dim}*K}`,
        else :math:`Q*K`
    scaled_query_init: bool, default = `True`
        Whether to scale WQ on initialization by :math:`\sqrt{head_dim}`
    """

    hidden_size: int = 512
    mlp_hidden_size: int = 2048
    num_attention_heads: int = 8
    layernorm_type: str = 'layernorm'
    layernorm_epsilon: float = 1e-6
    hidden_dropout: float = 0.1
    hidden_dropout_dims: Sequence[int] = ()
    attention_dropout: float = 0.1
    dropout_rng_name: str = 'dropout'
    mha_kernel_init: Initializer = None
    mlp_kernel_init: Initializer = None
    mlp_activations: Sequence[str] = ('relu',)
    use_bias: bool = False
    bias_init: Initializer = nn.initializers.zeros
    apply_residual_connection_post_layernorm: bool = False
    output_layernorm: bool = False
    float32_attention_logits: bool = False
    layer_type: TransformerLayerType = TransformerLayerType.ENCODER
    enable_relative_embedding: bool = True
    relative_embedding: nn.Module = None
    dtype: DType = jnp.float32
    drop_path: float = 0.0
    fuse_qkv_params: bool = True
    transpose_batch_sequence: bool = True
    scale_attn_logits: bool = False
    scaled_query_init: bool = True

    def __post_init__(self):
        if self.mha_kernel_init is None:
            self.mha_kernel_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        if self.mlp_kernel_init is None:
            self.mlp_kernel_init = nn.initializers.variance_scaling(1.0, 'fan_in',
                                                                    'truncated_normal')
        super().__post_init__()

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
        Transformer Layer: attention block and a feedforward network (MLP)

        Parameters
        ----------
        inputs : jax.numpy.ndarray
            Input tensor.
        encoded : jax.numpy.ndarray, default = None
            Output tensors of the encoder block to be fed into the decoder block if using
            :attr:`layer_type=TransformerLayerType.DECODER`.
        attention_mask : jax.numpy.ndarray, default = None
            Boolean tensor used to mask out self-attention softmax input.
        encoder_decoder_mask : jax.numpy.ndarray, default = None
            Boolean tensor used to mask out cross-attention softmax input when
            :attr:`layer_type=TransformerLayerType.DECODER`.
        deterministic: bool, default = False
            Disable dropout layers if set to True.
        decode: bool,default = False
            Indicate whether to prepare and use an autoregressive cache
            in Multi-head attention (MHA).
        max_decode_length : bool, default = None
            The maximum length to generate relative embedding biases when
            :attr:`layer_type=TransformerLayerType.DECODER` and
            :attr:`enable_relative_embedding=True`.

        Returns
        -------
        outputs : jax.numpy.ndarray
            Output tensors.
        """
        assert self.layer_type in TransformerLayerType, \
                "layer_type should be one of TransformerLayerType" \
                f", but got {self.layer_type}."

        assert self.hidden_size % self.num_attention_heads == 0, \
                "hidden_size should be multiples of num_attention_heads" \
                f", but got {self.hidden_size=} and {self.num_attention_heads=}."

        assert self.layer_type == TransformerLayerType.DECODER or \
              (self.layer_type == TransformerLayerType.ENCODER and decode is False), \
               "decode should be False when layer_type == TransformerLayerType.ENCODER."

        head_dim = self.hidden_size // self.num_attention_heads

        sequence_dim = 0 if self.transpose_batch_sequence else 1
        batch_dim = 1 - sequence_dim

        attn_bias = None
        if self.enable_relative_embedding:
            if self.relative_embedding is None:
                rel_emb = RelativePositionBiases(num_buckets=32,
                                                 max_distance=128,
                                                 num_attention_heads=self.num_attention_heads,
                                                 dtype=self.dtype,
                                                 embedding_init=nn.initializers.variance_scaling(
                                                     1.0, 'fan_avg', 'uniform'),
                                                 name='relpos_bias')
            else:
                rel_emb = self.relative_embedding

            if self.layer_type == TransformerLayerType.ENCODER:
                attn_bias = rel_emb(inputs.shape[sequence_dim], inputs.shape[sequence_dim], True)
            else:
                if decode and max_decode_length:
                    l = max_decode_length
                else:
                    l = inputs.shape[sequence_dim]
                attn_bias = rel_emb(l, l, False)

        assert inputs.ndim == 3

        self_attn_type = None
        # Make name be the exactly same as T5X, since names would affect
        # RNGKey during init and apply. Myabe no need in the feature.
        if self.layer_type == TransformerLayerType.ENCODER:
            mha_name = 'attention'
            self_attn_type = AttentionType.PADDING
        else:
            mha_name = 'self_attention'
            self_attn_type = AttentionType.CAUSAL
        assert self_attn_type is not None

        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        x, residual = MultiHeadAttention(
            num_heads=self.num_attention_heads,
            dtype=self.dtype,
            head_dim=head_dim,
            transpose_batch_sequence=self.transpose_batch_sequence,
            dropout_rate=self.attention_dropout,
            dropout_rng_name=self.dropout_rng_name,
            float32_logits=self.float32_attention_logits,
            scale_attn_logits=self.scale_attn_logits,
            scaled_query_init=self.scaled_query_init,
            layernorm_type=self.layernorm_type,
            layernorm_epsilon=self.layernorm_epsilon,
            apply_residual_connection_post_layernorm=self.apply_residual_connection_post_layernorm,
            output_layernorm=self.output_layernorm,
            attn_type=self_attn_type,
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

        def hidden_dropout(x, deterministic):
            assert isinstance(self.hidden_dropout_dims, Sequence)
            x_shape_len = len(x.shape)
            for dims in self.hidden_dropout_dims:
                assert -x_shape_len < dims < x_shape_len

            return nn.Dropout(rate=self.hidden_dropout,
                              broadcast_dims=self.hidden_dropout_dims)(x, deterministic)

        x = hidden_dropout(x, deterministic)
        if self.drop_path > 0.0:
            drop_path_shape = _generate_drop_path_shape(x.shape, batch_dim)
            x = nn.Dropout(rate=self.drop_path,
                           broadcast_dims=drop_path_shape)(x, deterministic=deterministic)
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
                dropout_rng_name=self.dropout_rng_name,
                layernorm_type=self.layernorm_type,
                layernorm_epsilon=self.layernorm_epsilon,
                apply_residual_connection_post_layernorm=self.
                apply_residual_connection_post_layernorm,
                output_layernorm=False,    # Must do LayerNorm before MHA.
                attn_type=AttentionType.PADDING,
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
            y = hidden_dropout(y, deterministic)
            mlp_input = y + residual

        # MlpBlock
        residual = mlp_input
        z, ln_out = LayerNormMLP(
            layernorm_type=self.layernorm_type,
            epsilon=self.layernorm_epsilon,
            major_sharding_type=infer_major_sharding_type(),
            transpose_batch_sequence=self.transpose_batch_sequence,
            return_layernorm_output=self.apply_residual_connection_post_layernorm,
            intermediate_dim=self.mlp_hidden_size,
            activations=self.mlp_activations,
            intermediate_dropout_rate=self.hidden_dropout,
            dtype=self.dtype,
            scale_axes=('embed',),
            kernel_init=self.mlp_kernel_init,
            kernel_axes_1=('embed', 'act', 'mlp'),
            kernel_axes_2=('mlp', 'embed'),
            use_bias=self.use_bias,
            bias_init=self.bias_init,
            name='mlp',
        )(mlp_input, deterministic=deterministic)

        if self.apply_residual_connection_post_layernorm:
            assert ln_out is not None
            residual = ln_out

        z = hidden_dropout(z, deterministic)
        if self.drop_path > 0.0:
            drop_path_shape = _generate_drop_path_shape(z.shape, batch_dim)
            z = nn.Dropout(rate=self.drop_path,
                           broadcast_dims=drop_path_shape)(z, deterministic=deterministic)
        z = z + residual

        if self.output_layernorm:
            ln_sharding_type, _ = infer_sharding_type()
            z = LayerNorm(layernorm_type=self.layernorm_type,
                          scale_axes=('embed',),
                          bias_axes=('embed',),
                          transpose_batch_sequence=self.transpose_batch_sequence,
                          dtype=self.dtype,
                          epsilon=self.layernorm_epsilon,
                          sharding_type=ln_sharding_type,
                          name="output_layer_norm")(z)

        return z

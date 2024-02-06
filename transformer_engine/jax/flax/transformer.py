# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Wrapper module for Transformer related layers with FP8 support.
"""
import functools
from enum import Enum
from math import sqrt
import os
from typing import Any, Callable, Optional, Sequence, Tuple, Union
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from jax import nn as jax_nn
from jax import random as jax_random
from jax import lax, vmap
from jax.ad_checkpoint import checkpoint_name

from .module import DenseGeneral, LayerNormDenseGeneral, LayerNormMLP
from .module import LayerNorm, Softmax
from ..fused_attn import AttnBiasType, AttnMaskType, QKVLayout
from ..fused_attn import is_fused_attn_kernel_available
from ..fused_attn import self_fused_attn, cross_fused_attn
from ..softmax import SoftmaxType
from ..sharding import num_of_devices
from ..sharding import get_sharding_map_logic_axis_to_mesh_axis
from ..sharding import with_sharding_constraint_by_logical_axes
from ..sharding import BATCH_AXES, SEQLEN_AXES, SEQLEN_TP_AXES, HEAD_AXES
from ..sharding import HIDDEN_AXES, HIDDEN_TP_AXES, JOINED_AXES
from ..sharding import W_NO_SHARD_AXES, W_FSDP_AXES, W_TP_AXES, W_JOINED_AXES

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
        We currently only support logical axis rules for single GPU training, data parallel
        training and 1D-sharding tensor parallel training.
        Refer to `Figure 3 in` `Megatron-LM tensor parallel <https://arxiv.org/pdf/1909.08053.pdf>`_
        for 1D-sharding tensor parallelism.

    .. warning::
        Please make sure ShardingResource is set via fp8_autocast before calling this function.

    .. note::
        This function is only needed when using TransformerLayer. For  other modules, such as
        DenseGeneral, please properly set axes of kernels and bias.

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
        if key in rules_map:
            rules_map[key].append(val)
        else:
            rules_map[key] = [val]

    extended_rules = [*rules]
    for item in get_sharding_map_logic_axis_to_mesh_axis().items():
        key = item[0]
        val = item[1]
        if key in rules_map:
            assert len(rules_map[key]) == 1 and rules_map[key][0] == val, \
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

    def func(a, b):
        return a + b

    return _merge_mask(func, *masks)


def core_attention(query: Array,
                   key: Array,
                   value: Array,
                   scale_factor: float,
                   transpose_batch_sequence: bool,
                   softmax_type: SoftmaxType = SoftmaxType.SCALED,
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
    sequence_dim = 0 if transpose_batch_sequence else 1
    assert key.shape[sequence_dim] == value.shape[sequence_dim], 'k, v lengths must match.'
    assert key.shape[-2] == value.shape[-2], 'k, v num_heads must match.'
    assert query.shape[-1] == key.shape[-1], 'q, k head_dim must match.'

    if float32_logits:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)

    h_q, h_kv = query.shape[-2], key.shape[-2]
    # The generated GQA kernels are slower than normal MHA kernels even when h_q == h_kv.
    # Therefore, we have to maintain two code paths.
    is_gqa = (h_q != h_kv)

    if is_gqa:
        assert (h_q % h_kv == 0) and (h_q >= h_kv)
        group_size = h_q // h_kv
        grouped_query = query.reshape((*query.shape[:2], h_kv, group_size, query.shape[-1]))

    if transpose_batch_sequence:
        if is_gqa:
            attn_weights = jnp.einsum('qbhgd,kbhd->bhgqk', grouped_query, key)
        else:
            attn_weights = jnp.einsum('qbhd,kbhd->bhqk', query, key)
    else:
        if is_gqa:
            attn_weights = jnp.einsum('bqhgd,bkhd->bhgqk', grouped_query, key)
        else:
            attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)

    attn_weights = checkpoint_name(attn_weights, 'logits')

    if is_gqa:
        b, h, g, q, k = attn_weights_with_groups_shape = attn_weights.shape
        attn_weights_without_groups_shape = (b, h * g, q, k)
        attn_weights = attn_weights.reshape(attn_weights_without_groups_shape)

    attn_weights = with_sharding_constraint_by_logical_axes(
        attn_weights, (BATCH_AXES, HEAD_AXES, SEQLEN_AXES, SEQLEN_AXES))

    # When a bias is present, the computation is performed as Softmax(attn_weights * scale + bias).
    # In this case, the scale can not fused into the Softmax module.
    if bias is not None:
        attn_weights = attn_weights * scale_factor
        fused_scale_factor = 1.
    else:
        # If no bias, the scale can be fused into Softmax module
        fused_scale_factor = scale_factor

    attn_weights = Softmax(softmax_type=softmax_type,
                           scale_factor=fused_scale_factor)(attn_weights, mask, bias).astype(dtype)

    if is_gqa:
        attn_weights = attn_weights.reshape(attn_weights_with_groups_shape)

    if not deterministic and dropout_rate > 0.:
        keep_prob = 1.0 - dropout_rate
        dropout_shape = list(attn_weights.shape)
        # TODO(rewang): add attention dropout broadcast dimension arguments for users
        keep = jax_random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        multiplier = (keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
        attn_weights = attn_weights * multiplier

    if transpose_batch_sequence:
        if is_gqa:
            return jnp.einsum('bhgqk,kbhd->qbhgd', attn_weights, value).reshape(query.shape)
        return jnp.einsum('bhqk,kbhd->qbhd', attn_weights, value)

    if is_gqa:
        return jnp.einsum('bhgqk,bkhd->bqhgd', attn_weights, value).reshape(query.shape)
    return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)


def rotary_pos_emb(x: Array, windows: Tuple[int, int], transpose_batch_sequence: bool):
    """
    Rotary Positional Embedding
    x should be in shape of
    [Batch, Seqlen, ..., Hidden] if transpose_batch_sequence is False, or
    [Seqlen, Batch, ..., Hidden] if transpose_batch_sequence is True.
    """
    embed_dim = x.shape[-1]
    half_embed_dim = embed_dim // 2
    min_window = windows[0]
    max_window = windows[1]

    fraction = 2 * jnp.arange(0, half_embed_dim) / embed_dim
    time_scales = min_window * (max_window / min_window)**fraction
    time_scales = jnp.expand_dims(time_scales, axis=tuple(range(x.ndim - 1)))

    batch_dim = 1 if transpose_batch_sequence else 0
    seq_dim = 1 - batch_dim

    positions = jnp.expand_dims(jnp.arange(x.shape[seq_dim]), axis=batch_dim)
    positions = jnp.expand_dims(positions, axis=tuple(range(2, x.ndim)))

    sinusoidal_positions = positions / time_scales
    sin = jnp.sin(sinusoidal_positions)
    cos = jnp.cos(sinusoidal_positions)

    x1, x2 = jnp.split(x, 2, axis=-1)
    part_1 = (x1 * cos - x2 * sin).astype(x.dtype)
    part_2 = (x2 * cos + x1 * sin).astype(x.dtype)

    return jnp.concatenate([part_1, part_2], axis=-1)


dynamic_vector_slice_in_dim = vmap(lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))


class MultiHeadAttention(nn.Module):    # pylint: disable=too-few-public-methods
    r"""
    Multi-head Attention (MHA), including Query,
    Key, Value and Output projection.

    .. note::

        Argument :attr:`mask` will be ignored when
        :attr:`attn_mask_type` is set to `"causal"`.

    Parameters
    ----------
    head_dim : int
        The hidden dimension of each attention head.
    num_heads : int
        The number of attention heads
    num_gqa_groups : int, default = `None`
        Number of GQA groups. When `None` is present, it is equal to num_heads.
        Grouped Query Attention is described in
        `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
        This only affects the keys and values, not the querys.
        GQA-1 is equivalent to Multi-Query Attention
        (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
        is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    dropout_rate : float, default = 0.0
        Dropout probability for the dropout op during multi-head attention.
    dropout_rng_name: str, default = 'dropout'
        The key in given RNGs via flax.linen.Module.apply that is used
        to generate Dropout masks in the core attention.
    layernorm_type : {'layernorm', 'rmsnorm'}, default = 'layernorm'
        Indicate the type of layer normalization.
    layernorm_epsilon: float, default = 1e-6
        A value added to the denominator of layer normalization for numerical stability.
    zero_centered_gamma : bool, default = False
        If set to `True`, the LayerNorm formula changes to

        .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} *
            (1 + \gamma) + \beta

        This parameter is only applicable for 'layernorm'.
    kernel_init: Initializer, default =
        flax.linen.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        Used for initializing the QKV and Output projection weights.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    use_bias: bool, default = False
        Indicate whether or not to enable bias shifting for QKVO projections.
        If set to False, the layer will not learn additive biases.
    bias_init: Initializer, default = flax.linen.initializers.zeros
        Used for initializing bias of QKVO projections, only used when :attr:`use_bias=True`.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    apply_residual_connection_post_layernorm : bool, default = False
        Indicate if apply residual connection with the output of layer normalization.
    output_layernorm : bool, default = False
        Indicate if apply a layer normalization at the end of MHA.
    attn_mask_type: {'causal', 'padding'}, default = 'causal'
        Type of attention mask passed into softmax operation.
        Introduced in v0.10.0.
    enable_rotary_pos_emb: bool, default = False
        Whether to enable rotary position embedding to projected query and key.
    rotary_pos_emb_windows: Tuple[int, int], default = (1, 10000)
        Indicate the min and max time-scales of rotary position embedding,
        only used when :attr:`enable_rotary_pos_emb=True`
    enable_sequence_parallel: bool, default = False
        Whether to enable sequence parallelism to operations except dot.

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
    num_gqa_groups: int | None = None
    dropout_rate: float = 0.
    dropout_rng_name: str = 'dropout'
    layernorm_type: str = "layernorm"
    layernorm_epsilon: float = 1e-6
    zero_centered_gamma: bool = False
    kernel_init: Initializer = None
    use_bias: bool = False
    bias_init: Initializer = nn.initializers.zeros
    apply_residual_connection_post_layernorm: bool = False
    output_layernorm: bool = False
    attn_mask_type: str = 'causal'
    enable_rotary_pos_emb: bool = False
    rotary_pos_emb_windows: Tuple[int, int] = (1, 10000)
    dtype: DType = jnp.float32
    fuse_qkv: bool = True
    transpose_batch_sequence: bool = True
    enable_sequence_parallel: bool = False
    scale_attn_logits: bool = False
    scaled_query_init: bool = True
    float32_logits: bool = False    # computes logits in float32 for stability.

    def __post_init__(self):
        if self.kernel_init is None:
            self.kernel_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        if self.num_gqa_groups is None:
            self.num_gqa_groups = self.num_heads
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

        def query_init(*args):
            depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
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

        # TODO(rewang): make it configurable for pre_scale_bias
        attn_bias_type = AttnBiasType.NO_BIAS if bias is None else AttnBiasType.POST_SCALE_BIAS

        def canonicalize_attn_mask_type(attn_mask_type):
            """
            Convert the string to AttnMaskType
            """
            if attn_mask_type == 'causal':
                return AttnMaskType.PADDING_CAUSAL_MASK
            if attn_mask_type == 'padding':
                return AttnMaskType.PADDING_MASK
            raise ValueError(f"Unsupported {attn_mask_type=}, "
                             "supported attn_mask_type = {'causal', 'padding'}")

        is_self_attn = (inputs_q is inputs_kv)
        is_gqa = (self.num_heads != self.num_gqa_groups)
        is_qkvpack = (is_self_attn and not is_gqa)
        qkv_layout = QKVLayout.BS3HD if is_self_attn else QKVLayout.BSHD_BS2HD
        attn_mask_type = canonicalize_attn_mask_type(self.attn_mask_type)

        q_seqlen = inputs_q.shape[0] if self.transpose_batch_sequence else inputs_q.shape[1]
        kv_seqlen = inputs_kv.shape[0] if self.transpose_batch_sequence else inputs_kv.shape[1]
        enable_fused_attn = int(os.getenv("NVTE_FUSED_ATTN", "0"))

        has_fused_attn_kernel = is_fused_attn_kernel_available(self.dtype, self.dtype, qkv_layout,
                                                               attn_bias_type, attn_mask_type,
                                                               self.dropout_rate, self.num_heads,
                                                               self.num_gqa_groups, q_seqlen,
                                                               kv_seqlen, self.head_dim)

        use_fused_attn = not decode and not self.transpose_batch_sequence and self.fuse_qkv and \
            has_fused_attn_kernel and \
            enable_fused_attn

        if enable_fused_attn and not use_fused_attn:
            reason = ""
            if decode:
                reason += f"decode=False is required but got {decode}, "
            if self.transpose_batch_sequence:
                reason += f"transpose_batch_sequence=False is required " \
                          f"but got {self.transpose_batch_sequence}, "
            if not self.fuse_qkv:
                reason += f"fuse_qkv=True is required but got {self.fuse_qkv}, "
            if not has_fused_attn_kernel:
                reason += "no fused attention kernel is available, "

            warnings.warn(
                f"Fused attention is not enabled. Because " \
                f"{reason}fall back to unfused attention.")

        def generate_batch_seqlen_logical_axes(is_sharded_seq):
            sequence_dim = 0 if self.transpose_batch_sequence else 1
            batch_dim = 1 - sequence_dim

            axes = [None, None]

            axes[batch_dim] = BATCH_AXES
            axes[sequence_dim] = SEQLEN_TP_AXES if is_sharded_seq else SEQLEN_AXES
            return tuple(axes)

        inputs_logical_axes_maybe_sp = (*generate_batch_seqlen_logical_axes(
            self.enable_sequence_parallel), HIDDEN_AXES)
        inputs_logical_axes_no_sp = (*generate_batch_seqlen_logical_axes(False), HIDDEN_AXES)

        inputs_q = with_sharding_constraint_by_logical_axes(inputs_q, inputs_logical_axes_maybe_sp)

        residual = inputs_q
        if self.fuse_qkv:
            if is_qkvpack:
                qkv_proj, ln_out = LayerNormDenseGeneral(
                    enable_layernorm=not self.output_layernorm,
                    layernorm_type=self.layernorm_type,
                    zero_centered_gamma=self.zero_centered_gamma,
                    epsilon=self.layernorm_epsilon,
                    axis=-1,
                    features=(3, self.num_heads * self.head_dim),
                    transpose_batch_sequence=self.transpose_batch_sequence,
                    return_layernorm_output=self.apply_residual_connection_post_layernorm,
                    scale_axes=(W_NO_SHARD_AXES,),
                    ln_bias_axes=(W_NO_SHARD_AXES,),
                    kernel_axes=(W_FSDP_AXES, W_JOINED_AXES, W_TP_AXES),
                    kernel_init=qkv_init,
                    use_bias=self.use_bias,
                    bias_init=self.bias_init,
                    bias_axes=(W_JOINED_AXES, W_TP_AXES),
                    layernorm_input_axes=inputs_logical_axes_maybe_sp,
                    dot_input_axes=inputs_logical_axes_no_sp,
                    name='qkv',
                    dtype=self.dtype)(inputs_q)
                qkv_proj = checkpoint_name(qkv_proj, 'combined_qkv_proj')
                if not use_fused_attn:
                    query, key, value = jnp.split(qkv_proj, [1, 2], axis=-2)
            else:
                query, ln_out = LayerNormDenseGeneral(
                    enable_layernorm=not self.output_layernorm,
                    layernorm_type=self.layernorm_type,
                    zero_centered_gamma=self.zero_centered_gamma,
                    epsilon=self.layernorm_epsilon,
                    axis=-1,
                    features=self.num_heads * self.head_dim,
                    transpose_batch_sequence=self.transpose_batch_sequence,
                    return_layernorm_output=(self.apply_residual_connection_post_layernorm
                                             or is_self_attn),
                    scale_axes=(W_NO_SHARD_AXES,),
                    ln_bias_axes=(W_NO_SHARD_AXES,),
                    kernel_axes=(W_FSDP_AXES, W_TP_AXES),
                    use_bias=self.use_bias,
                    bias_init=self.bias_init,
                    bias_axes=(W_TP_AXES,),
                    dtype=self.dtype,
                    kernel_init=query_init,
                    layernorm_input_axes=inputs_logical_axes_maybe_sp,
                    dot_input_axes=inputs_logical_axes_no_sp,
                    name='query')(inputs_q)

                if is_self_attn:
                    assert ln_out is not None
                    inputs_kv = ln_out

                kv_proj = DenseGeneral(axis=-1,
                                       features=(2, self.num_gqa_groups * self.head_dim),
                                       transpose_batch_sequence=self.transpose_batch_sequence,
                                       kernel_axes=(W_FSDP_AXES, W_JOINED_AXES, W_TP_AXES),
                                       kernel_init=kv_init,
                                       use_bias=self.use_bias,
                                       bias_init=self.bias_init,
                                       bias_axes=(W_JOINED_AXES, W_TP_AXES),
                                       name='kv',
                                       dtype=self.dtype)(inputs_kv)
                kv_proj = checkpoint_name(kv_proj, 'combined_kv_proj')
                if not use_fused_attn:
                    key, value = jnp.split(kv_proj, [1], axis=-2)
        else:
            kv_projection = functools.partial(
                DenseGeneral,
                axis=-1,
                features=self.num_gqa_groups * self.head_dim,
                transpose_batch_sequence=self.transpose_batch_sequence,
                kernel_axes=(W_FSDP_AXES, W_TP_AXES),
                use_bias=self.use_bias,
                bias_init=self.bias_init,
                bias_axes=(W_TP_AXES,),
                dtype=self.dtype)
            query, ln_out = LayerNormDenseGeneral(
                enable_layernorm=not self.output_layernorm,
                layernorm_type=self.layernorm_type,
                zero_centered_gamma=self.zero_centered_gamma,
                epsilon=self.layernorm_epsilon,
                axis=-1,
                features=self.num_heads * self.head_dim,
                transpose_batch_sequence=self.transpose_batch_sequence,
                return_layernorm_output=True,
                scale_axes=(W_NO_SHARD_AXES,),
                ln_bias_axes=(W_NO_SHARD_AXES,),
                kernel_axes=(W_FSDP_AXES, W_TP_AXES),
                use_bias=self.use_bias,
                bias_init=self.bias_init,
                bias_axes=(W_TP_AXES,),
                dtype=self.dtype,
                kernel_init=query_init,
                layernorm_input_axes=inputs_logical_axes_maybe_sp,
                dot_input_axes=inputs_logical_axes_no_sp,
                name='query')(inputs_q)

            if is_self_attn:
                assert ln_out is not None
                inputs_kv = ln_out

            key = kv_projection(kernel_init=self.kernel_init, name='key')(inputs_kv)
            value = kv_projection(kernel_init=self.kernel_init, name='value')(inputs_kv)

        if self.apply_residual_connection_post_layernorm:
            assert ln_out is not None
            residual = ln_out

        if self.enable_rotary_pos_emb:
            if self.fuse_qkv and use_fused_attn:
                if is_qkvpack:
                    query, key, value = jnp.split(qkv_proj, [1, 2], axis=-2)
                else:
                    key, value = jnp.split(kv_proj, [1], axis=-2)

            query = rotary_pos_emb(query, self.rotary_pos_emb_windows,
                                   self.transpose_batch_sequence)
            key = rotary_pos_emb(key, self.rotary_pos_emb_windows, self.transpose_batch_sequence)

            if use_fused_attn:
                if is_qkvpack:
                    qkv_proj = jnp.concatenate([query, key, value], axis=-2)
                else:
                    kv_proj = jnp.concatenate([key, value], axis=-2)

        if not use_fused_attn:
            query = checkpoint_name(query, 'query_proj')
            key = checkpoint_name(key, 'key_proj')
            value = checkpoint_name(value, 'value_proj')
            query = query.reshape((*query.shape[:2], self.num_heads, self.head_dim))
            key = key.reshape((*key.shape[:2], self.num_gqa_groups, self.head_dim))
            value = value.reshape((*value.shape[:2], self.num_gqa_groups, self.head_dim))
            qkv_sharding_constraint = \
                (SEQLEN_AXES, BATCH_AXES, HEAD_AXES, HIDDEN_AXES) \
                if self.transpose_batch_sequence \
                else (BATCH_AXES, SEQLEN_AXES, HEAD_AXES, HIDDEN_AXES)
            query = with_sharding_constraint_by_logical_axes(query, qkv_sharding_constraint)
            key = with_sharding_constraint_by_logical_axes(key, qkv_sharding_constraint)
            value = with_sharding_constraint_by_logical_axes(value, qkv_sharding_constraint)

        if decode:
            is_initialized = self.has_variable('cache', 'cached_key')

            cached_key = self.variable('cache', 'cached_key', jnp.zeros, key.shape, key.dtype)
            cached_value = self.variable('cache', 'cached_value', jnp.zeros, value.shape,
                                         value.dtype)
            cache_index = self.variable('cache', 'cache_index',
                                        lambda: jnp.array(0, dtype=jnp.int32))
            if is_initialized:
                if self.transpose_batch_sequence:
                    length, batch, num_heads, head_dim = cached_key.value.shape
                    expected_shape = (1, batch, num_heads, head_dim)
                    one_hot_indices_shape = (length, 1, 1, 1)
                else:
                    batch, length, num_heads, head_dim = cached_key.value.shape
                    expected_shape = (batch, 1, num_heads, head_dim)
                    one_hot_indices_shape = (1, length, 1, 1)

                # Sanity shape check of cached key against input query.
                if expected_shape != query.shape:
                    raise ValueError(
                        'Autoregressive cache shape error, '
                        f"expected query shape {expected_shape} instead got {query.shape}.")

                cur_index = cache_index.value
                one_hot_indices = jax_nn.one_hot(cur_index, length, dtype=key.dtype)
                one_hot_indices = jnp.reshape(one_hot_indices, one_hot_indices_shape)
                key = cached_key.value + key * one_hot_indices
                value = cached_value.value + value * one_hot_indices
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1

                mask = combine_masks(
                    mask, jnp.broadcast_to(jnp.arange(length) > cur_index, (batch, 1, 1, length)))

                if bias is not None:
                    bias = dynamic_vector_slice_in_dim(jnp.squeeze(bias, axis=0),
                                                       jnp.reshape(cur_index, (-1)), 1, -2)

        scale_factor = 1.0 / sqrt(self.head_dim) if self.scale_attn_logits else 1.0

        dropout_rng = None
        if not deterministic and self.dropout_rate > 0.:
            dropout_rng = self.make_rng(self.dropout_rng_name)

        if use_fused_attn:
            assert mask is not None and mask.ndim == 4    # (b, 1, s_q, s_kv)
            assert not self.transpose_batch_sequence

            seed = None
            if dropout_rng is not None:
                seed = jax.random.split(dropout_rng, num_of_devices())
                # ensure the old key never used
                del dropout_rng

            if is_qkvpack:
                qkv_proj = qkv_proj.reshape((*qkv_proj.shape[:-1], self.num_heads, self.head_dim))
                qkv_sharding_constraint = (BATCH_AXES, SEQLEN_AXES, JOINED_AXES, HEAD_AXES,
                                           HIDDEN_AXES)
                qkv_proj = with_sharding_constraint_by_logical_axes(qkv_proj,
                                                                    qkv_sharding_constraint)

                x = self_fused_attn(qkv_proj,
                                    bias,
                                    mask,
                                    seed,
                                    attn_bias_type=attn_bias_type,
                                    attn_mask_type=attn_mask_type,
                                    scaling_factor=scale_factor,
                                    dropout_probability=self.dropout_rate,
                                    is_training=not deterministic)
            else:
                assert bias is None
                query = query.reshape((*query.shape[:-1], self.num_heads, self.head_dim))
                kv_proj = kv_proj.reshape((*kv_proj.shape[:-1], self.num_gqa_groups, self.head_dim))
                q_sharding_constraint = (BATCH_AXES, SEQLEN_AXES, HEAD_AXES, HIDDEN_AXES)
                kv_sharding_constraint = (BATCH_AXES, SEQLEN_AXES, JOINED_AXES, HEAD_AXES,
                                          HIDDEN_AXES)
                query = with_sharding_constraint_by_logical_axes(query, q_sharding_constraint)
                kv_proj = with_sharding_constraint_by_logical_axes(kv_proj, kv_sharding_constraint)

                x = cross_fused_attn(query,
                                     kv_proj,
                                     bias,
                                     mask,
                                     seed,
                                     attn_bias_type=attn_bias_type,
                                     attn_mask_type=attn_mask_type,
                                     scaling_factor=scale_factor,
                                     dropout_probability=self.dropout_rate,
                                     is_training=not deterministic)
        else:

            def convert_to_softmax_type(attn_mask_type, mask):
                """
                Convert the string to SoftmaxType
                """
                if attn_mask_type == 'causal':
                    return SoftmaxType.SCALED_UPPER_TRIANG_MASKED
                if attn_mask_type == 'padding':
                    if mask is not None:
                        return SoftmaxType.SCALED_MASKED
                    return SoftmaxType.SCALED
                raise ValueError(f"Unsupported {attn_mask_type=}, "
                                 "supported attn_mask_type = {'causal', 'padding'}")

            softmax_type = convert_to_softmax_type(self.attn_mask_type, mask)

            x = core_attention(query,
                               key,
                               value,
                               scale_factor=scale_factor,
                               transpose_batch_sequence=self.transpose_batch_sequence,
                               softmax_type=softmax_type,
                               mask=mask,
                               bias=bias,
                               dropout_rng=dropout_rng,
                               dropout_rate=self.dropout_rate,
                               deterministic=deterministic,
                               dtype=self.dtype,
                               float32_logits=self.float32_logits)

            x = checkpoint_name(x, 'context')

        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))

        attn_context_sharding_constraint = \
            (SEQLEN_AXES, BATCH_AXES, HIDDEN_TP_AXES) \
            if self.transpose_batch_sequence \
            else (BATCH_AXES, SEQLEN_AXES, HIDDEN_TP_AXES)
        x = with_sharding_constraint_by_logical_axes(x, attn_context_sharding_constraint)

        out = DenseGeneral(features=inputs_q.shape[-1],
                           transpose_batch_sequence=self.transpose_batch_sequence,
                           axis=-1,
                           kernel_init=self.kernel_init,
                           kernel_axes=(W_TP_AXES, W_FSDP_AXES),
                           use_bias=self.use_bias,
                           bias_init=self.bias_init,
                           bias_axes=(W_NO_SHARD_AXES,),
                           dtype=self.dtype,
                           name='out')(x)
        out = checkpoint_name(out, 'out_proj')
        return out, residual


class RelativePositionBiases(nn.Module):    # pylint: disable=too-few-public-methods
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


class TransformerLayer(nn.Module):    # pylint: disable=too-few-public-methods
    r"""
    TransformerLayer is made up of a relative embedding,
    an attention block and a feedforward network (MLP).
    This standard layer is based on the paper “Attention Is All You Need”.

    .. note::

        Argument :attr:`attention_mask` will be ignored when
        :attr:`self_attn_mask_type` is set to `"causal"`.

    Parameters
    ----------
    hidden_size: int, default = 512
        The hidden size of each input sample.
    mlp_hidden_size: int, default = 2048
        Intermediate size to which input samples are projected.
    num_attention_heads: int, default = 8
        Number of attention heads in the transformer layer.
    num_gqa_groups : int, default = `None`
        Number of GQA groups. When `None` is present, it is equal to num_attention_heads.
        Grouped Query Attention is described in
        `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
        This only affects the keys and values, not the querys.
        GQA-1 is equivalent to Multi-Query Attention
        (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
        is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    layernorm_type : {'layernorm', 'rmsnorm'}, default = 'layernorm'
        Indicate the type of layer normalization.
    layernorm_epsilon: float, default = 1e-6
        A value added to the denominator of layer normalization for numerical stability.
    zero_centered_gamma : bool, default = False
        If set to `True`, the LayerNorm formula changes to

        .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} *
            (1 + \gamma) + \beta

        This parameter is only applicable for 'layernorm'.
    hidden_dropout: float, default = 0.1
        Dropout probability for the dropout op after FC2 layer.
    hidden_dropout_dims: Sequence[int], default = ()
        Dimensions that will share the same dropout mask for hidden
    attention_dropout: float, default = 0.1
        Dropout probability for the dropout op during multi-head attention.
    intermediate_dropout: float, default = 0.1
        Dropout probability for the dropout op after FC1 layer.
    intermediate_dropout_dims: Sequence[int], default = ()
        Dimensions that will share the same dropout mask for hidden after FC1 layer.
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
        FC1 and FC2. It is only used when :attr:`use_bias=True`.
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
    self_attn_mask_type: {'causal', 'padding'}, default = 'causal'
        Type of attention mask passed into softmax operation.
        Introduced in v0.10.0.
    enable_relative_embedding: bool, default = True
        Whether to enable relative embedding as shifting of attention logits.
    relative_embedding: flax.linen.Module, default = None
        The module for relative embedding execution, only used when
        :attr:`enable_relative_embedding=True`. Default is None, which will create
        an instance of RelativePositionBiases if :attr:`enable_relative_embedding=True`.
        Default: RelativePositionBiases( num_buckets=32, max_distance=128,
        num_attention_heads=self.num_attention_heads, dtype=self.dtype,
        embedding_init=flax.linen.initializers.variance_scaling(1.0, 'fan_avg', 'uniform'),
        name='relpos_bias')
    enable_rotary_pos_emb: bool, default = False
        Whether to enable rotary position embedding to projected query and key in MHA.
    rotary_pos_emb_windows: Tuple[int, int], default = (1, 10000)
        Indicate the min and max time-scales of rotary position embedding,
        only used when :attr:`enable_rotary_pos_emb=True`
    enable_sequence_parallel: bool, default = False
        Whether to enable sequence parallelism to operations except dot.

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
    transpose_batch_sequence : bool, default = False
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
    num_gqa_groups: int | None = None
    layernorm_type: str = 'layernorm'
    layernorm_epsilon: float = 1e-6
    zero_centered_gamma: bool = False
    hidden_dropout: float = 0.1
    hidden_dropout_dims: Sequence[int] = ()
    attention_dropout: float = 0.1
    intermediate_dropout: float = 0.1
    intermediate_dropout_dims: Sequence[int] = ()
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
    self_attn_mask_type: str = 'causal'
    enable_relative_embedding: bool = True
    relative_embedding: nn.Module = None
    enable_rotary_pos_emb: bool = False
    rotary_pos_emb_windows: Tuple[int, int] = (1, 10000)
    dtype: DType = jnp.float32
    drop_path: float = 0.0
    fuse_qkv_params: bool = True
    transpose_batch_sequence: bool = False
    enable_sequence_parallel: bool = False
    scale_attn_logits: bool = False
    scaled_query_init: bool = True

    def __post_init__(self):
        if self.mha_kernel_init is None:
            self.mha_kernel_init = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        if self.mlp_kernel_init is None:
            self.mlp_kernel_init = nn.initializers.variance_scaling(1.0, 'fan_in',
                                                                    'truncated_normal')
        if self.num_gqa_groups is None:
            self.num_gqa_groups = self.num_attention_heads
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

        def generate_batch_seqlen_logical_axes(is_shared_seq=None):
            axes = [None, None]

            is_shared_seq = self.enable_sequence_parallel if is_shared_seq is None \
                            else is_shared_seq

            axes[batch_dim] = BATCH_AXES
            axes[sequence_dim] = SEQLEN_TP_AXES if is_shared_seq else SEQLEN_AXES
            return tuple(axes)

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

        # Make name be the exactly same as T5X, since names would affect
        # RNGKey during init and apply. Myabe no need in the feature.
        if self.layer_type == TransformerLayerType.ENCODER:
            mha_name = 'attention'
        else:
            mha_name = 'self_attention'

        inputs = with_sharding_constraint_by_logical_axes(
            inputs, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES))

        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        x, residual = MultiHeadAttention(
            num_heads=self.num_attention_heads,
            dtype=self.dtype,
            head_dim=head_dim,
            num_gqa_groups=self.num_gqa_groups,
            transpose_batch_sequence=self.transpose_batch_sequence,
            enable_sequence_parallel=self.enable_sequence_parallel,
            dropout_rate=self.attention_dropout,
            dropout_rng_name=self.dropout_rng_name,
            float32_logits=self.float32_attention_logits,
            scale_attn_logits=self.scale_attn_logits,
            scaled_query_init=self.scaled_query_init,
            layernorm_type=self.layernorm_type,
            layernorm_epsilon=self.layernorm_epsilon,
            zero_centered_gamma=self.zero_centered_gamma,
            apply_residual_connection_post_layernorm=self.apply_residual_connection_post_layernorm,
            output_layernorm=self.output_layernorm,
            attn_mask_type=self.self_attn_mask_type,
            enable_rotary_pos_emb=self.enable_rotary_pos_emb,
            rotary_pos_emb_windows=self.rotary_pos_emb_windows,
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
                assert -x_shape_len <= dims < x_shape_len

            return nn.Dropout(rate=self.hidden_dropout,
                              broadcast_dims=self.hidden_dropout_dims,
                              rng_collection=self.dropout_rng_name)(x, deterministic=deterministic)

        x = with_sharding_constraint_by_logical_axes(
            x, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES))
        residual = with_sharding_constraint_by_logical_axes(
            residual, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES))

        x = hidden_dropout(x, deterministic)
        if self.drop_path > 0.0:
            drop_path_shape = _generate_drop_path_shape(x.shape, batch_dim)
            x = nn.Dropout(rate=self.drop_path,
                           broadcast_dims=drop_path_shape,
                           rng_collection=self.dropout_rng_name)(x, deterministic=deterministic)
        x = x + residual

        mlp_input = x
        if self.layer_type == TransformerLayerType.DECODER:
            assert encoded is not None, \
                "encoded is required when layer_type == TransformerLayerType.DECODER."

            x = with_sharding_constraint_by_logical_axes(
                x, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES))

            y, residual = MultiHeadAttention(
                num_heads=self.num_attention_heads,
                dtype=self.dtype,
                head_dim=head_dim,
                num_gqa_groups=self.num_gqa_groups,
                transpose_batch_sequence=self.transpose_batch_sequence,
                enable_sequence_parallel=self.enable_sequence_parallel,
                dropout_rate=self.attention_dropout,
                dropout_rng_name=self.dropout_rng_name,
                layernorm_type=self.layernorm_type,
                layernorm_epsilon=self.layernorm_epsilon,
                zero_centered_gamma=self.zero_centered_gamma,
                apply_residual_connection_post_layernorm=self.
                apply_residual_connection_post_layernorm,
                output_layernorm=False,    # Must do LayerNorm before MHA.
                attn_mask_type='padding',
                enable_rotary_pos_emb=self.enable_rotary_pos_emb,
                rotary_pos_emb_windows=self.rotary_pos_emb_windows,
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

            y = with_sharding_constraint_by_logical_axes(
                y, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES))
            residual = with_sharding_constraint_by_logical_axes(
                residual, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES))

            y = hidden_dropout(y, deterministic)
            mlp_input = y + residual

        mlp_input = with_sharding_constraint_by_logical_axes(
            mlp_input, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES))

        # MlpBlock
        residual = mlp_input
        z, ln_out = LayerNormMLP(
            layernorm_type=self.layernorm_type,
            zero_centered_gamma=self.zero_centered_gamma,
            epsilon=self.layernorm_epsilon,
            transpose_batch_sequence=self.transpose_batch_sequence,
            return_layernorm_output=self.apply_residual_connection_post_layernorm,
            intermediate_dim=self.mlp_hidden_size,
            activations=self.mlp_activations,
            intermediate_dropout_rng_name=self.dropout_rng_name,
            intermediate_dropout_rate=self.intermediate_dropout,
            intermediate_hidden_dropout_dims=self.intermediate_dropout_dims,
            dtype=self.dtype,
            scale_axes=(W_NO_SHARD_AXES,),
            ln_bias_axes=(W_NO_SHARD_AXES,),
            kernel_init=self.mlp_kernel_init,
            kernel_axes_1=(W_FSDP_AXES, W_JOINED_AXES, W_TP_AXES),
            kernel_axes_2=(W_TP_AXES, W_FSDP_AXES),
            use_bias=self.use_bias,
            bias_init=self.bias_init,
            bias_axes_1=(W_JOINED_AXES, W_TP_AXES),
            bias_axes_2=(W_NO_SHARD_AXES,),
            layernorm_input_axes=(*generate_batch_seqlen_logical_axes(), HIDDEN_AXES),
            dot_1_input_axes=(*generate_batch_seqlen_logical_axes(False), HIDDEN_AXES),
            dot_2_input_axes=(*generate_batch_seqlen_logical_axes(False), HIDDEN_TP_AXES),
            name='mlp',
        )(mlp_input, deterministic=deterministic)

        if self.apply_residual_connection_post_layernorm:
            assert ln_out is not None
            residual = ln_out

        z = with_sharding_constraint_by_logical_axes(
            z, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES))
        residual = with_sharding_constraint_by_logical_axes(
            residual, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES))

        z = hidden_dropout(z, deterministic)
        if self.drop_path > 0.0:
            drop_path_shape = _generate_drop_path_shape(z.shape, batch_dim)
            z = nn.Dropout(rate=self.drop_path,
                           broadcast_dims=drop_path_shape)(z, deterministic=deterministic)
        z = z + residual

        if self.output_layernorm:
            z = with_sharding_constraint_by_logical_axes(
                z, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES))
            z = LayerNorm(layernorm_type=self.layernorm_type,
                          zero_centered_gamma=self.zero_centered_gamma,
                          epsilon=self.layernorm_epsilon,
                          scale_axes=(W_NO_SHARD_AXES,),
                          bias_axes=(W_NO_SHARD_AXES,),
                          transpose_batch_sequence=self.transpose_batch_sequence,
                          dtype=self.dtype,
                          name="output_layer_norm")(z)

        return z

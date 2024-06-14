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
from flax.linen.attention import combine_masks
from jax import nn as jax_nn
from jax import random as jax_random
from jax import lax, vmap
from jax.ad_checkpoint import checkpoint_name

from .module import DenseGeneral, LayerNormDenseGeneral, LayerNormMLP
from .module import LayerNorm, Softmax
from ..attention import AttnBiasType, AttnMaskType, QKVLayout
from ..attention import is_fused_attn_kernel_available, canonicalize_attn_mask_type
from ..attention import fused_attn_qkvpacked, fused_attn_kvpacked, fused_attn
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
PrecisionLike = Union[
    None, str, lax.Precision, Tuple[str, str], Tuple[lax.Precision, lax.Precision]
]
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
    rules: Sequence[Tuple[str, Union[str, None]]]
        the base Flax logical axis rules to extend.

    Returns
    -------
    extended_rules: Sequence[Tuple[str, Union[str, None]]]
        the extended Flax logical axis rules.
    """
    rules_map = {}
    for item in rules:
        assert len(item) == 2, "The logical axis rule should be like (axis_name, mesh_axis_name)."
        key = item[0]
        val = item[1]
        assert isinstance(key, str), f"Thie axis_name should be str, but got {type(key)}."
        assert isinstance(val, str) or (
            val is None
        ), f"Thie mesh_axis_name should be str or None, but got {type(val)}."
        if key in rules_map:
            rules_map[key].append(val)
        else:
            rules_map[key] = [val]

    extended_rules = [*rules]
    for item in get_sharding_map_logic_axis_to_mesh_axis().items():
        key = item[0]
        val = item[1]
        if key in rules_map:
            assert len(rules_map[key]) == 1 and rules_map[key][0] == val, (
                "The rule diverged between TE and given rule."
                f"Axis:{key} map to {rules_map[key]} in the given"
                f" rules, but {val} in TE's rules."
            )
        else:
            extended_rules.append(item)
    return tuple(extended_rules)


class _UnfusedDotProductAttention(nn.Module):  # pylint: disable=too-few-public-methods
    attention_dropout: float = 0.0
    attn_mask_type: AttnMaskType = AttnMaskType.CAUSAL_MASK
    attn_bias_type: Optional[AttnBiasType] = None
    dtype: DType = jnp.float32
    float32_logits: bool = False
    scale_factor: Optional[float] = None
    transpose_batch_sequence: bool = True

    @nn.compact
    def __call__(
        self,
        query: Array,
        key: Array,
        value: Array,
        mask: Optional[Array] = None,
        bias: Optional[Array] = None,
        *,
        dropout_rng: Optional[PRNGKey] = None,
        deterministic: bool = False,
    ) -> Array:
        assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
        batch_dim = 1 if self.transpose_batch_sequence else 0
        assert (
            query.shape[batch_dim] == key.shape[batch_dim] == value.shape[batch_dim]
        ), "q, k, v batch dims must match."
        sequence_dim = 0 if self.transpose_batch_sequence else 1
        assert key.shape[sequence_dim] == value.shape[sequence_dim], "k, v lengths must match."
        assert key.shape[-2] == value.shape[-2], "k, v num_attention_heads must match."
        assert query.shape[-1] == key.shape[-1], "q, k head_dim must match."

        if self.scale_factor is None:
            scale_factor = 1.0 / sqrt(query.shape[-1])
        else:
            scale_factor = self.scale_factor
        del self.scale_factor

        if self.float32_logits:
            query = query.astype(jnp.float32)
            key = key.astype(jnp.float32)
        h_q, h_kv = query.shape[-2], key.shape[-2]
        # The generated GQA kernels are slower than normal MHA kernels even when h_q == h_kv.
        # Therefore, we have to maintain two code paths.
        is_gqa = h_q != h_kv

        if is_gqa:
            assert (h_q % h_kv == 0) and (h_q >= h_kv)
            group_size = h_q // h_kv
            grouped_query = query.reshape((*query.shape[:2], h_kv, group_size, query.shape[-1]))

        if self.transpose_batch_sequence:
            if is_gqa:
                attn_weights = jnp.einsum("qbhgd,kbhd->bhgqk", grouped_query, key)
            else:
                attn_weights = jnp.einsum("qbhd,kbhd->bhqk", query, key)
        else:
            if is_gqa:
                attn_weights = jnp.einsum("bqhgd,bkhd->bhgqk", grouped_query, key)
            else:
                attn_weights = jnp.einsum("bqhd,bkhd->bhqk", query, key)

        attn_weights = checkpoint_name(attn_weights, "logits")

        if is_gqa:
            b, h, g, q, k = attn_weights_with_groups_shape = attn_weights.shape
            attn_weights_without_groups_shape = (b, h * g, q, k)
            attn_weights = attn_weights.reshape(attn_weights_without_groups_shape)

        attn_weights = with_sharding_constraint_by_logical_axes(
            attn_weights, (BATCH_AXES, HEAD_AXES, SEQLEN_AXES, SEQLEN_AXES)
        )

        # When post_scale_bias is present, the computation is Softmax(attn_weights * scale + bias)
        # In this case, the scale can not fused into the Softmax module.
        if self.attn_bias_type == AttnBiasType.POST_SCALE_BIAS:
            attn_weights = attn_weights * scale_factor
            fused_scale_factor = 1.0
        else:
            # If not post_scale_bias, the scale can be fused into Softmax module
            fused_scale_factor = scale_factor
            if self.attn_bias_type == AttnBiasType.PRE_SCALE_BIAS:
                attn_weights += bias

        def convert_to_softmax_type(attn_mask_type, mask):
            """Convert the attn_mask_type to SoftmaxType"""
            # mask is ignored for no_mask and causal_mask
            if attn_mask_type in [AttnMaskType.NO_MASK, AttnMaskType.CAUSAL_MASK]:
                mask = None
            if attn_mask_type in [AttnMaskType.CAUSAL_MASK, AttnMaskType.PADDING_CAUSAL_MASK]:
                return SoftmaxType.SCALED_UPPER_TRIANG_MASKED, mask
            if attn_mask_type in [AttnMaskType.NO_MASK, AttnMaskType.PADDING_MASK]:
                if mask is not None:
                    return SoftmaxType.SCALED_MASKED, mask
                return SoftmaxType.SCALED, mask
            raise ValueError(
                f"Unsupported {attn_mask_type=}, supported attn_mask_type="
                "{'no_mask', 'padding', 'causal', 'padding_causal', 'causal_padding'}"
            )

        softmax_type, mask = convert_to_softmax_type(self.attn_mask_type, mask)

        attn_weights = Softmax(softmax_type=softmax_type, scale_factor=fused_scale_factor)(
            attn_weights, mask, bias
        ).astype(self.dtype)

        if is_gqa:
            attn_weights = attn_weights.reshape(attn_weights_with_groups_shape)

        if not deterministic and self.attention_dropout > 0.0:
            keep_prob = 1.0 - self.attention_dropout
            dropout_shape = list(attn_weights.shape)
            # TODO(rewang): add attention dropout broadcast dimension arguments for users
            keep = jax_random.bernoulli(dropout_rng, keep_prob, dropout_shape)
            multiplier = keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=self.dtype)
            attn_weights = attn_weights * multiplier

        if self.transpose_batch_sequence:
            if is_gqa:
                return jnp.einsum("bhgqk,kbhd->qbhgd", attn_weights, value).reshape(query.shape)
            return jnp.einsum("bhqk,kbhd->qbhd", attn_weights, value)

        if is_gqa:
            return jnp.einsum("bhgqk,bkhd->bqhgd", attn_weights, value).reshape(query.shape)
        return jnp.einsum("bhqk,bkhd->bqhd", attn_weights, value)


class _FusedDotProductAttention(nn.Module):  # pylint: disable=too-few-public-methods
    attention_dropout: float = 0.0
    attn_mask_type: AttnMaskType = AttnMaskType.CAUSAL_MASK
    attn_bias_type: Optional[AttnBiasType] = None
    dtype: DType = jnp.float32
    qkv_layout: QKVLayout = QKVLayout.BSHD_BSHD_BSHD
    scale_factor: Optional[float] = None
    transpose_batch_sequence: bool = False

    @nn.compact
    def __call__(
        self,
        query: Array,
        key: Array,
        value: Array,
        mask: Optional[Array] = None,
        bias: Optional[Array] = None,
        *,
        dropout_rng: Optional[PRNGKey] = None,
        deterministic: bool = False,
    ) -> Array:

        seed = None
        if dropout_rng is not None:
            seed = jax.random.split(dropout_rng, num_of_devices())

        if self.scale_factor is None:
            scale_factor = 1.0 / sqrt(query.shape[-1])
        else:
            scale_factor = self.scale_factor
        del self.scale_factor

        if self.qkv_layout == QKVLayout.BS3HD:
            """qkvpacked format, treat
            query: qkvpacked tensor, shape = [..., 3, h, d]
            key: ignore
            value: ignore
            """
            qkv_packed = query
            if self.transpose_batch_sequence:
                qkv_packed = qkv_packed.transpose([1, 0, 2, 3, 4])
            x = fused_attn_qkvpacked(
                qkv_packed,
                bias,
                mask,
                seed,
                attn_mask_type=self.attn_mask_type,
                attn_bias_type=self.attn_bias_type,
                scaling_factor=scale_factor,
                dropout_probability=self.attention_dropout,
                is_training=not deterministic,
            )
        elif self.qkv_layout == QKVLayout.BSHD_BS2HD:
            """kvpacked format, treat
            query: query tensor, shape = [..., h, d]
            key: kvpacked tensor, shape = [..., 2, h, d]
            value: ignore
            """
            kv_packed = key
            if self.transpose_batch_sequence:
                query = query.transpose([1, 0, 2, 3])
                kv_packed = kv_packed.transpose([1, 0, 2, 3, 4])
            x = fused_attn_kvpacked(
                query,
                kv_packed,
                bias,
                mask,
                seed,
                attn_mask_type=self.attn_mask_type,
                attn_bias_type=self.attn_bias_type,
                scaling_factor=scale_factor,
                dropout_probability=self.attention_dropout,
                is_training=not deterministic,
            )
        elif self.qkv_layout == QKVLayout.BSHD_BSHD_BSHD:
            if self.transpose_batch_sequence:
                query = query.transpose([1, 0, 2, 3])
                key = key.transpose([1, 0, 2, 3])
                value = value.transpose([1, 0, 2, 3])
            x = fused_attn(
                query,
                key,
                value,
                bias,
                mask,
                seed,
                attn_mask_type=self.attn_mask_type,
                attn_bias_type=self.attn_bias_type,
                scaling_factor=scale_factor,
                dropout_probability=self.attention_dropout,
                is_training=not deterministic,
            )
        else:
            raise ValueError(f"Unsupported {self.qkv_layout=}.")

        if self.transpose_batch_sequence:
            x = x.transpose([1, 0, 2, 3])

        return x


class DotProductAttention(nn.Module):  # pylint: disable=too-few-public-methods
    r"""
    Dot Product Attention (DPA). Allows the model to jointly attend to information from different
    representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. note::
        The DotProductAttention module supports two backends: the unfused and the fused attention
        mechanisms. The unfused attention is implemented using JAX native operations, providing
        broad compatibility and flexibility. In contrast, the fused attention uses `cuDNN fused
        attention
        <https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md>`_ for
        higher performance and lower memory usage on the supported hardwares.
        Users can select between these two backends via the :attr:`NVTE_FUSED_ATTN` environment
        variable:

        * Set :attr:`NVTE_FUSED_ATTN=0` for unfused attention (default).
        * Set :attr:`NVTE_FUSED_ATTN=1` for fused attention. If the required cuDNN fused attention
          kernel is not available on the system, a warning will be issued, and the module will
          automatically fall back to the unfused backend.

    Parameters
    ----------
    head_dim: int
        The hidden dimension of each attention head.
    num_attention_heads: int
        The number of attention heads.
    num_gqa_groups: int, default = `None`
        Number of GQA groups. When `None` is present, it is equal to num_attention_heads.
        Grouped Query Attention is described in
        `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
        This only affects the keys and values, not the querys.
        GQA-1 is equivalent to Multi-Query Attention
        (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
        is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    attention_dropout: float, default = 0.0
        Dropout probability for the dropout op after the softmax.
    attn_mask_type: str, default = 'causal'
        This parameter specifies the type of attention mask to be applied during the softmax
        operation.
        Available options are {'no_mask', 'padding', 'causal', 'causal_padding', 'padding_causal'}

        Each described below:

        * no_mask: No attention mask is applied. This means the attention will consider the
          full sequence without any restrictions.
        * padding: Indicates the presence of padding at the end of each sequence.
          Users must provide a mask with the shape [batch, 1, max_seqlen_q, max_seqlen_kv] in the
          :attr:`__call__` method to specify the padding positions.
        * causal: An upper triangular mask is applied to the softmax inputs,
          ensuring that the prediction for a certain position is only dependent on known outputs
          from positions before it.
        * causal_padding / padding_causal: A combination of both causal and padding masks.
          Both 'causal_padding' and 'padding_causal' are acceptable and have the same effect.

        .. note:: :attr:`mask` in :attr:`__call__` is ignored for 'no_mask' and 'causal'.

    attn_bias_type: Optional[str], default = None
        Type of the attention bias passed in the attention.
        Available options: {'no_bias', 'pre_scale_bias', 'post_scale_bias'}.
        When default is present, the type is automatically decided by the MHA's bias parameter.
        Where it is :attr:`post_scale_bias` if there is bias. Otherwise :attr:`no_bias` is used.
    dropout_rng_name: str, default = 'dropout'
        The key in given RNGs via flax.linen.Module.apply that is used
        to generate Dropout masks in the core attention.
    float32_logits: bool, default = False
        Whether to compute attention logits in float32 for the unfused attention backend.
        For fused attention backend, the accumulation is always float32 without the perf overhead.
    qkv_layout: str, default = 'bshd_bshd_bshd'
        Specifies the dimensional layout format for the query, key, and value tensors in __call__().
        It indicates how the inputs are processed.
        Available options: {'bs3hd', 'bshd_bs2hd', 'bshd_bshd_bshd'}. Where

        * bs3hd: query tensor is treated as a qkvpacked tensor with shape = [b, s, 3, h, d].
          key and value arguments in :attr:`__call__()` are ignored in this layout.
        * bshd_bs2hd: query tensor with shape = [b, s, h, d]. key tensor is treaded as a kvpacked
          tensor with shape = [b, s, 2, h, d]. `value` argument in :attr:`__call__()` is ignored.
        * bshd_bshd_bshd: query, key, and value are seperated with shape = [b, s, h, d].

        Explanation of denotations:

        * b: batch size
        * s: seqeuence length
        * h: num_attention_heads or num_gqa_groups
        * d: head dimension

    scale_factor: Optional[float], default = None
        Scale factor to apply on query. When :attr:`None` is present, the scale factor is equal
        to :math:`\frac{1}{\sqrt{head\_dim}}`. This is useful for model like T5X, which doesn't
        need to apply scale on query, which is to set :attr:`scale_factor=1.`.
    transpose_batch_sequence: bool, default = True
        Indicate whether the input tensors were switched axis of batch
        and sequence length dimension. if set to True, the input tensors
        should be in (seqlen, batch, ...), otherwise (batch, seqlen, ...).

    Optimization parameters
    -----------------------
    dtype: jax.numpy.dtype, default = jax.numpy.float32
        The data type used to allocate the initial parameters.
    """

    head_dim: int
    num_attention_heads: int
    num_gqa_groups: Optional[int] = None
    attention_dropout: float = 0.0
    attn_mask_type: AttnMaskType = "causal"
    attn_bias_type: AttnBiasType = None
    dtype: DType = jnp.float32
    dropout_rng_name: str = "dropout"
    float32_logits: bool = False
    qkv_layout: str = "bshd_bshd_bshd"
    scale_factor: Optional[float] = None
    transpose_batch_sequence: bool = True

    @nn.compact
    def __call__(
        self,
        query: Array,
        key: Array,
        value: Array,
        mask: Optional[Array] = None,
        bias: Optional[Array] = None,
        *,
        deterministic: bool = False,
    ) -> Array:
        """
        Parameters
        ----------
        query: jax.numpy.ndarray
            The details of query tensor representation is described in :attr:`qkv_layout`.
        key: jax.numpy.ndarrary
            The details of kery tensor representation is described in :attr:`qkv_layout`.
        value: jax.numpy.ndarrary
            The details of value tensor representation is described in :attr:`qkv_layout`.
        mask: jax.numpy.ndarray, default = None
            Boolean tensor used to mask out the attention softmax input.
            :attr:`True` means to mask out the corresponding values.
            Ignored when :attr:`self.attn_mask_type` is either 'no_mask' or 'causal'.
        bias: jax.numpy.ndarray, default = None
            A tensor used to shift attention softmax input.
        *:
            Below parameters are keyword only
        deterministic: bool, default = False
            Disable dropout layers if set to True.

        Returns
        -------
        outputs: jax.numpy.ndarray
            Output tensors.
        """

        # For internal API, we use enum to maintain
        if self.attn_bias_type is None:
            attn_bias_type = AttnBiasType.NO_BIAS if bias is None else AttnBiasType.POST_SCALE_BIAS
        else:
            attn_bias_type = AttnBiasType[self.attn_bias_type.upper()]
        attn_mask_type = canonicalize_attn_mask_type(self.attn_mask_type)
        qkv_layout = QKVLayout[self.qkv_layout.upper()]
        del self.attn_bias_type, self.attn_mask_type, self.qkv_layout

        if attn_bias_type == AttnBiasType.NO_BIAS:
            assert bias is None
        else:
            assert bias is not None

        enable_fused_attn = int(os.getenv("NVTE_FUSED_ATTN", "0"))

        sequence_dim = 0 if self.transpose_batch_sequence else 1
        seqlen_q = query.shape[sequence_dim]
        if qkv_layout == QKVLayout.BS3HD:
            seqlen_kv = seqlen_q
        else:
            seqlen_kv = key.shape[sequence_dim]

        has_fused_attn_kernel = is_fused_attn_kernel_available(
            self.dtype,
            self.dtype,
            qkv_layout,
            attn_bias_type,
            attn_mask_type,
            self.attention_dropout,
            self.num_attention_heads,
            self.num_gqa_groups,
            seqlen_q,
            seqlen_kv,
            self.head_dim,
        )

        use_fused_attn = enable_fused_attn and has_fused_attn_kernel

        if enable_fused_attn and not has_fused_attn_kernel:
            warnings.warn(
                "Fused attention is not enabled because there is no available kernel.\n"
                "Fall back to the unfused attention.\n"
                "Please try to update the cuDNN and TE to the latest version.\n"
                f"{self.dtype=}\n{qkv_layout=}\n{attn_bias_type=}\n{attn_mask_type=}\n"
                f"{self.attention_dropout=}\n{self.num_attention_heads=}\n"
                f"{self.num_gqa_groups=}\n{seqlen_q=}\n{seqlen_kv=}\n{self.head_dim=}\n"
            )

        dropout_rng = None
        if not deterministic and self.attention_dropout > 0.0:
            dropout_rng = self.make_rng(self.dropout_rng_name)

        if self.scale_factor is None:
            scale_factor = 1.0 / sqrt(self.head_dim)
        else:
            scale_factor = self.scale_factor
        del self.scale_factor

        if not use_fused_attn:
            # unfused attention only supports splitted query, key, value
            if qkv_layout == QKVLayout.BS3HD:
                query, key, value = jnp.split(query, [1, 2], axis=-3)
                query, key, value = map(
                    functools.partial(jnp.squeeze, axis=-3), [query, key, value]
                )
            elif qkv_layout == QKVLayout.BSHD_BS2HD:
                key, value = jnp.split(key, [1], axis=-3)
                key, value = map(functools.partial(jnp.squeeze, axis=-3), [key, value])
            else:
                assert qkv_layout == QKVLayout.BSHD_BSHD_BSHD

            x = _UnfusedDotProductAttention(
                attention_dropout=self.attention_dropout,
                attn_mask_type=attn_mask_type,
                attn_bias_type=attn_bias_type,
                dtype=self.dtype,
                float32_logits=self.float32_logits,
                scale_factor=scale_factor,
                transpose_batch_sequence=self.transpose_batch_sequence,
            )(query, key, value, mask, bias, dropout_rng=dropout_rng, deterministic=deterministic)
        else:
            x = _FusedDotProductAttention(
                attention_dropout=self.attention_dropout,
                attn_mask_type=attn_mask_type,
                attn_bias_type=attn_bias_type,
                dtype=self.dtype,
                scale_factor=scale_factor,
                transpose_batch_sequence=self.transpose_batch_sequence,
                qkv_layout=qkv_layout,
            )(query, key, value, mask, bias, dropout_rng=dropout_rng, deterministic=deterministic)

        return x


def rotary_pos_emb(
    x: Array,
    windows: Tuple[int, int],
    transpose_batch_sequence: bool,
    group_method: str = "consecutive",
):
    """
    Rotary Positional Embedding
    x should be in shape of
    [Batch, Seqlen, ..., Heads, Hidden] if transpose_batch_sequence is False, or
    [Seqlen, Batch, ..., Heads, Hidden] if transpose_batch_sequence is True.
    """
    hidden_dim = x.shape[-1]
    half_hidden_dim = hidden_dim // 2
    min_window = windows[0]
    max_window = windows[1]

    fraction = 2 * jnp.arange(0, half_hidden_dim) / hidden_dim
    time_scales = min_window * (max_window / min_window) ** fraction
    time_scales = jnp.expand_dims(time_scales, axis=tuple(range(x.ndim - 1)))

    batch_dim = 1 if transpose_batch_sequence else 0
    seq_dim = 1 - batch_dim

    positions = jnp.expand_dims(jnp.arange(x.shape[seq_dim]), axis=batch_dim)
    positions = jnp.expand_dims(positions, axis=tuple(range(2, x.ndim)))

    def generate_sin_cos(timescales):
        sinusoidal_positions = positions / timescales
        sin = jnp.sin(sinusoidal_positions)
        cos = jnp.cos(sinusoidal_positions)
        return sin, cos

    def alternate_impl():
        sin, cos = generate_sin_cos(time_scales)

        x1, x2 = jnp.split(x, 2, axis=-1)
        part_1 = (x1 * cos - x2 * sin).astype(x.dtype)
        part_2 = (x2 * cos + x1 * sin).astype(x.dtype)

        output = jnp.concatenate([part_1, part_2], axis=-1)
        return output

    def consecutive_impl():
        sin, cos = generate_sin_cos(jnp.repeat(time_scales, 2, axis=-1))

        x_shifted_left = jnp.roll(x, -1, axis=-1)
        x_shifted_right = jnp.roll(x, 1, axis=-1)
        x_shifted = jax.lax.select(
            jnp.tile(
                jnp.mod(jnp.arange(hidden_dim, dtype=jnp.int32), 2),
                x.shape[:-1] + (1,),
            ),
            x_shifted_right,
            x_shifted_left,
        )

        sign = jnp.sign(jnp.mod(jnp.arange(hidden_dim, dtype=jnp.int32), 2) - 0.5)

        output = x * cos + x_shifted * sin * sign
        output = output.astype(x.dtype)
        return output

    def canonicalize_group_method(gm):
        canonicalized_gm = gm.lower().strip().replace("-", "").replace("_", "")
        assert canonicalized_gm in ["consecutive", "alternate"], (
            "Invalid relative positional embedding group method. "
            f"Expect to be in []'alternate' or 'consecutive'], but got {gm}."
        )

        return canonicalized_gm

    group_method = canonicalize_group_method(group_method)

    if group_method == "alternate":
        return alternate_impl()
    return consecutive_impl()


class LoRAScope:  # pylint: disable=too-few-public-methods
    """LoRA Scope"""

    def __init__(self, qkv_proj=False, output_proj=False, mlp=False):
        self.qkv_proj = qkv_proj
        self.output_proj = output_proj
        self.mlp = mlp

    def __eq__(self, other):
        return (self.qkv_proj, self.output_proj, self.mlp) == (
            other.qkv_proj,
            other.output_proj,
            other.mlp,
        )


def _canonicalize_lora_scope(scope):

    SCOPE_NONE = "none"
    SCOPE_ALL = "all"
    SCOPE_QKV_PROJ = "qkv_proj"
    SCOPE_OUTPUT_PROJ = "output_proj"
    SCOPE_MLP = "mlp"
    SCOPE_EX_QKV_PROJ = "exclude_qkv_proj"
    SCOPE_EX_OUTPUT_PROJ = "exclude_output_proj"
    SCOPE_EX_MLP = "exclude_mlp"

    scope = SCOPE_NONE if scope is None else scope

    scope = scope.lower()

    assert scope in [
        SCOPE_NONE,
        SCOPE_ALL,
        SCOPE_QKV_PROJ,
        SCOPE_OUTPUT_PROJ,
        SCOPE_MLP,
        SCOPE_EX_QKV_PROJ,
        SCOPE_EX_OUTPUT_PROJ,
        SCOPE_EX_MLP,
    ]

    lora_scope = LoRAScope()

    if scope in [SCOPE_ALL, SCOPE_QKV_PROJ, SCOPE_EX_OUTPUT_PROJ, SCOPE_EX_MLP]:
        lora_scope.qkv_proj = True

    if scope in [SCOPE_ALL, SCOPE_OUTPUT_PROJ, SCOPE_EX_QKV_PROJ, SCOPE_EX_MLP]:
        lora_scope.output_proj = True

    if scope in [SCOPE_ALL, SCOPE_MLP, SCOPE_EX_QKV_PROJ, SCOPE_EX_OUTPUT_PROJ]:
        lora_scope.mlp = True

    return lora_scope


class MultiHeadAttention(nn.Module):  # pylint: disable=too-few-public-methods
    r"""
    Multi-head Attention (MHA), including Query,
    Key, Value and Output projection.

    Parameters
    ----------
    head_dim: int
        The hidden dimension of each attention head.
    num_attention_heads: int
        The number of attention heads.
    num_gqa_groups: int, default = `None`
        Number of GQA groups. When `None` is present, it is equal to num_attention_heads.
        Grouped Query Attention is described in
        `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
        This only affects the keys and values, not the querys.
        GQA-1 is equivalent to Multi-Query Attention
        (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
        is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    attention_dropout: float, default = 0.0
        Dropout probability for the dropout op after the softmax.
    attn_mask_type: str, default = 'causal'
        This parameter specifies the type of attention mask to be applied during the softmax
        operation.
        Available options are {'no_mask', 'padding', 'causal', 'causal_padding', 'padding_causal'}

        Each described below:

        * no_mask: No attention mask is applied. This means the attention will consider the
          full sequence without any restrictions.
        * padding: Indicates the presence of padding at the end of each sequence.
          Users must provide a mask with the shape [batch, 1, max_seqlen_q, max_seqlen_kv] in the
          :attr:`__call__` method to specify the padding positions.
        * causal: An upper triangular mask is applied to the softmax inputs,
          ensuring that the prediction for a certain position is only dependent on known outputs
          from positions before it.
        * causal_padding / padding_causal: A combination of both causal and padding masks.
          Both 'causal_padding' and 'padding_causal' are acceptable and have the same effect.

        .. note:: :attr:`mask` in :attr:`__call__` is ignored for 'no_mask' and 'causal'.

    attn_bias_type: Optional[str], default = None
        Type of the attention bias passed in the attention.
        Available options: {'no_bias', 'pre_scale_bias', 'post_scale_bias'}.
        When default is present, the type is automatically decided by the MHA's bias parameter.
        Where it is `post_scale_bias` if there is bias. Otherwise `no_bias` is used.
    dropout_rng_name: str, default = 'dropout'
        The key in given RNGs via flax.linen.Module.apply that is used
        to generate Dropout masks in the core attention.
    layernorm_type: {'layernorm', 'rmsnorm'}, default = 'layernorm'
        Indicate the type of layer normalization.
    layernorm_epsilon: float, default = 1e-6
        A value added to the denominator of layer normalization for numerical stability.
    zero_centered_gamma: bool, default = False
        If set to `True`, the LayerNorm formula changes to

        .. math::
            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} *
            (1 + \gamma) + \beta

        This parameter is only applicable for 'layernorm'.
    kernel_init: Initializer, default =
        flax.linen.initializers.variance_scaling(1.0, 'fan_in', 'normal')
        Used for initializing the QKV and output projection weights.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    use_bias: bool, default = False
        Indicate whether or not to enable bias shifting for QKV and output projections.
        If set to False, the layer will not learn additive biases.
    bias_init: Initializer, default = flax.linen.initializers.zeros
        Used for initializing bias of QKVO projections, only used when :attr:`use_bias=True`.
        It should be a callable object with three arguments (jax.random.PRNGKey, shape, dtype).
    input_layernorm: bool, default = True
        If set to False, layer normalization to the input is not applied.
    return_layernorm_output: bool, default = False
        If set to True, output of layernorm is returned from the forward together with the output
        of the linear transformation.
        Example use case: residual connection for transformer module is taken post layernorm.
    enable_rotary_pos_emb: bool, default = False
        Whether to enable rotary position embedding to projected query and key.
    rotary_pos_emb_windows: Tuple[int, int], default = (1, 10000)
        Indicate the min and max time-scales of rotary position embedding,
        only used when :attr:`enable_rotary_pos_emb=True`
    rotary_pos_emb_group_method: str, default = 'consecutive'
        Indicate the method to coupled the coordinates. It should be one of
        ['consecutive', 'alternate']. 'alternate' is to pair index :math:`i` with :math:`i + d/2`
        , d is the hidden dimension. 'consecutive' pairs index :math:`i` with :math:`i + 1`.
    low_rank_adaptation_scope: str, default = 'none'
        Indicate the scope to apply low rank adaptation. It should be one of
        ['none', 'all', 'qkv_proj', 'output_proj', 'exclude_qkv_proj', 'exclude_output_proj']
    low_rank_adaptation_dim: int, default = 32
        The dimension for low rank adaptation, only used when
        :attr:`enable_low_rank_adaptation=True`
    low_rank_adaptation_alpha: float, default = None
        The alpha for computing the scaling factor of LoRA output.
        :math:`\frac{alpha}{rank} * lora_output`. None means no scaling.
    enable_sequence_parallel: bool, default = False
        Whether to enable sequence parallelism to operations except dot.
    num_heads: int, default = None
        Deprecated. Please refer `num_attention_heads`.
    dropout_rate: float, default = None
        Deprecated. Please refer `attention_dropout`.
    output_layernorm: bool, default = None
        Deprecated. Please refer `input_layernorm`
    apply_residual_connection_post_layernorm: bool, default = None
        Deprecated. Please refer `return_layernorm_output`.

    Optimization parameters
    -----------------------
    dtype: jax.numpy.dtype, default = jax.numpy.float32
        The data type used to allocate the initial parameters.
    fuse_qkv_params: bool, default = True
        If set to True, this module exposes a single fused
        parameter for query-key-value for self-attention and key-value for
        cross-attention.
    transpose_batch_sequence: bool, default = True
        Indicate whether the input tensors were switched axis of batch
        and sequence length dimension. if set to True, the input tensors
        should be in (seqlen, batch, hidden), otherwise (batch, seqlen, hidden).
    scale_attn_logits: bool, default = False
        Indicate whether to scale attention logits.
        If set to True, :math:`\frac{Q}{\sqrt{head\_dim}*K}`,
        else :math:`Q*K`
    scaled_query_init: bool, default = True
        Whether to scale WQ on initialization by :math:`\frac{1}{\sqrt{head\_dim}}`
    float32_logits: bool, default = False
        Whether to compute attention logits in float32 for the unfused attention backend.
        For fused attention backend, the accumulation is always float32 without the perf overhead.
    fuse_qkv: bool, default = None
        Deprecated. Please refer `fuse_qkv_params`
    """

    head_dim: int
    num_attention_heads: int
    num_gqa_groups: Optional[int] = None
    attention_dropout: float = 0.0
    dropout_rng_name: str = "dropout"
    input_layernorm: bool = True
    layernorm_type: str = "layernorm"
    layernorm_epsilon: float = 1e-6
    return_layernorm_output: bool = False
    zero_centered_gamma: bool = False
    kernel_init: Initializer = None
    use_bias: bool = False
    bias_init: Initializer = nn.initializers.zeros
    attn_mask_type: str = "causal"
    attn_bias_type: Optional[str] = None
    enable_rotary_pos_emb: bool = False
    rotary_pos_emb_windows: Tuple[int, int] = (1, 10000)
    rotary_pos_emb_group_method: str = "consecutive"
    low_rank_adaptation_scope: str = "none"
    low_rank_adaptation_dim: int = 32
    low_rank_adaptation_alpha: float = None
    dtype: DType = jnp.float32
    fuse_qkv_params: bool = True
    transpose_batch_sequence: bool = True
    enable_sequence_parallel: bool = False
    scale_attn_logits: bool = False
    scaled_query_init: bool = True
    float32_logits: bool = False

    # Deprecated parameters
    num_heads: Optional[int] = None
    dropout_rate: Optional[float] = None
    output_layernorm: Optional[bool] = None
    apply_residual_connection_post_layernorm: Optional[bool] = None
    fuse_qkv: Optional[bool] = None

    def __post_init__(self):
        # Deal with the deprecated parameters
        if self.num_heads is not None:
            self.num_attention_heads = self.num_heads
            warnings.warn(
                f"{__class__}.num_heads is deprecated. It will be removed recently. "
                f"Please uses {__class__}.num_attention_heads as the new API.",
                DeprecationWarning,
            )
        if self.dropout_rate is not None:
            self.attention_dropout = self.dropout_rate
            warnings.warn(
                f"{__class__}.dropout_rate is deprecated. It will be removed recently. "
                f"Please use {__class__}.attention_dropout as the new API.",
                DeprecationWarning,
            )
        if self.apply_residual_connection_post_layernorm is not None:
            warnings.warn(
                f"{__class__}.apply_residual_connection_post_layernorm is deprecated. "
                f"It will be removed recently, please use {__class__}.return_layernorm_output.",
                DeprecationWarning,
            )
        if self.fuse_qkv is not None:
            warnings.warn(
                f"{__class__}.fuse_qkv is deprecated. It will be removed recently. "
                f"Please use {__class__}.fuse_qkv_params as the new API.",
                DeprecationWarning,
            )
        assert self.output_layernorm is None, (
            f"{__class__}.output_layernorm is deprecated. It will be removed recently. "
            f"Please use {__class__}.input_layernorm for controlling whether to apply layernorm."
        )

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
        """
        MultiHeadAttention Layer:
        [Query, Key, Value projection] -> Dot Product Attention -> Output projection.

        Parameters
        ----------
        inputs_q: jax.numpy.ndarray
            Input tensor for query projection.
        inputs_kv: jax.numpy.ndarray
            Input tensor for key/value projection.
        mask: jax.numpy.ndarray, default = None
            Boolean tensor used to mask out the attention softmax input.
            :attr:`True` means mask out the corresponding values.
            Ignored when :attr:`self.attn_mask_type` is either 'no_mask' or 'causal'.
        bias: jax.numpy.ndarray, default = None
            A tensor used to shift the attention softmax input.
        *
        decode: bool, default = False
            Indicate whether to prepare and use an autoregressive cache.
        deterministic: bool, default = False
            Disable dropout layers if set to True.

        Returns
        -------
        outputs: jax.numpy.ndarray
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

        def generate_batch_seqlen_logical_axes(is_sharded_seq):
            sequence_dim = 0 if self.transpose_batch_sequence else 1
            batch_dim = 1 - sequence_dim

            axes = [None, None]

            axes[batch_dim] = BATCH_AXES
            axes[sequence_dim] = SEQLEN_TP_AXES if is_sharded_seq else SEQLEN_AXES
            return tuple(axes)

        is_self_attn = inputs_q is inputs_kv
        is_gqa = self.num_attention_heads != self.num_gqa_groups
        is_qkvpack = is_self_attn and not is_gqa

        inputs_logical_axes_maybe_sp = (
            *generate_batch_seqlen_logical_axes(self.enable_sequence_parallel),
            HIDDEN_AXES,
        )
        inputs_logical_axes_no_sp = (*generate_batch_seqlen_logical_axes(False), HIDDEN_AXES)

        inputs_q = with_sharding_constraint_by_logical_axes(inputs_q, inputs_logical_axes_maybe_sp)

        lora_scope = _canonicalize_lora_scope(self.low_rank_adaptation_scope)

        if self.fuse_qkv_params:
            if is_qkvpack:
                qkv_proj, ln_out = LayerNormDenseGeneral(
                    enable_layernorm=self.input_layernorm,
                    layernorm_type=self.layernorm_type,
                    zero_centered_gamma=self.zero_centered_gamma,
                    epsilon=self.layernorm_epsilon,
                    axis=-1,
                    features=(3, self.num_attention_heads * self.head_dim),
                    transpose_batch_sequence=self.transpose_batch_sequence,
                    return_layernorm_output=self.return_layernorm_output,
                    scale_axes=(W_NO_SHARD_AXES,),
                    ln_bias_axes=(W_NO_SHARD_AXES,),
                    kernel_axes=(W_FSDP_AXES, W_JOINED_AXES, W_TP_AXES),
                    kernel_init=qkv_init,
                    use_bias=self.use_bias,
                    bias_init=self.bias_init,
                    bias_axes=(W_JOINED_AXES, W_TP_AXES),
                    enable_low_rank_adaptation=lora_scope.qkv_proj,
                    low_rank_adaptation_dim=self.low_rank_adaptation_dim,
                    low_rank_adaptation_alpha=self.low_rank_adaptation_alpha,
                    layernorm_input_axes=inputs_logical_axes_maybe_sp,
                    dot_input_axes=inputs_logical_axes_no_sp,
                    name="qkv",
                    dtype=self.dtype,
                )(inputs_q)
                qkv_proj = checkpoint_name(qkv_proj, "combined_qkv_proj")
                qkv_layout = QKVLayout.BS3HD
            else:
                query, ln_out = LayerNormDenseGeneral(
                    enable_layernorm=self.input_layernorm,
                    layernorm_type=self.layernorm_type,
                    zero_centered_gamma=self.zero_centered_gamma,
                    epsilon=self.layernorm_epsilon,
                    axis=-1,
                    features=self.num_attention_heads * self.head_dim,
                    transpose_batch_sequence=self.transpose_batch_sequence,
                    return_layernorm_output=(self.return_layernorm_output or is_self_attn),
                    scale_axes=(W_NO_SHARD_AXES,),
                    ln_bias_axes=(W_NO_SHARD_AXES,),
                    kernel_axes=(W_FSDP_AXES, W_TP_AXES),
                    use_bias=self.use_bias,
                    bias_init=self.bias_init,
                    bias_axes=(W_TP_AXES,),
                    enable_low_rank_adaptation=lora_scope.qkv_proj,
                    low_rank_adaptation_dim=self.low_rank_adaptation_dim,
                    low_rank_adaptation_alpha=self.low_rank_adaptation_alpha,
                    dtype=self.dtype,
                    kernel_init=query_init,
                    layernorm_input_axes=inputs_logical_axes_maybe_sp,
                    dot_input_axes=inputs_logical_axes_no_sp,
                    name="query",
                )(inputs_q)

                if is_self_attn:
                    assert ln_out is not None
                    inputs_kv = ln_out

                kv_proj = DenseGeneral(
                    axis=-1,
                    features=(2, self.num_gqa_groups * self.head_dim),
                    transpose_batch_sequence=self.transpose_batch_sequence,
                    kernel_axes=(W_FSDP_AXES, W_JOINED_AXES, W_TP_AXES),
                    kernel_init=kv_init,
                    use_bias=self.use_bias,
                    bias_init=self.bias_init,
                    bias_axes=(W_JOINED_AXES, W_TP_AXES),
                    enable_low_rank_adaptation=lora_scope.qkv_proj,
                    low_rank_adaptation_dim=self.low_rank_adaptation_dim,
                    low_rank_adaptation_alpha=self.low_rank_adaptation_alpha,
                    name="kv",
                    dtype=self.dtype,
                )(inputs_kv)
                kv_proj = checkpoint_name(kv_proj, "combined_kv_proj")
                qkv_layout = QKVLayout.BSHD_BS2HD
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
                enable_low_rank_adaptation=lora_scope.qkv_proj,
                low_rank_adaptation_dim=self.low_rank_adaptation_dim,
                low_rank_adaptation_alpha=self.low_rank_adaptation_alpha,
                dtype=self.dtype,
            )
            query, ln_out = LayerNormDenseGeneral(
                enable_layernorm=self.input_layernorm,
                layernorm_type=self.layernorm_type,
                zero_centered_gamma=self.zero_centered_gamma,
                epsilon=self.layernorm_epsilon,
                axis=-1,
                features=self.num_attention_heads * self.head_dim,
                transpose_batch_sequence=self.transpose_batch_sequence,
                return_layernorm_output=True,
                scale_axes=(W_NO_SHARD_AXES,),
                ln_bias_axes=(W_NO_SHARD_AXES,),
                kernel_axes=(W_FSDP_AXES, W_TP_AXES),
                use_bias=self.use_bias,
                bias_init=self.bias_init,
                bias_axes=(W_TP_AXES,),
                enable_low_rank_adaptation=lora_scope.qkv_proj,
                low_rank_adaptation_dim=self.low_rank_adaptation_dim,
                low_rank_adaptation_alpha=self.low_rank_adaptation_alpha,
                dtype=self.dtype,
                kernel_init=query_init,
                layernorm_input_axes=inputs_logical_axes_maybe_sp,
                dot_input_axes=inputs_logical_axes_no_sp,
                name="query",
            )(inputs_q)

            if is_self_attn:
                assert ln_out is not None
                inputs_kv = ln_out

            key = kv_projection(kernel_init=self.kernel_init, name="key")(inputs_kv)
            value = kv_projection(kernel_init=self.kernel_init, name="value")(inputs_kv)
            query = checkpoint_name(query, "query_proj")
            key = checkpoint_name(key, "key_proj")
            value = checkpoint_name(value, "value_proj")
            qkv_layout = QKVLayout.BSHD_BSHD_BSHD

        if self.enable_rotary_pos_emb:
            if qkv_layout == QKVLayout.BS3HD:
                query, key, value = jnp.split(qkv_proj, [1, 2], axis=-2)
            elif qkv_layout == QKVLayout.BSHD_BS2HD:
                key, value = jnp.split(kv_proj, [1], axis=-2)
            else:
                assert qkv_layout == QKVLayout.BSHD_BSHD_BSHD

            # No changes to memory layout, should trigger bitcast only (Ideally no Perf impact)
            query = query.reshape((*query.shape[:2], self.num_attention_heads, self.head_dim))
            key = key.reshape((*key.shape[:2], self.num_gqa_groups, self.head_dim))

            query = rotary_pos_emb(
                query,
                self.rotary_pos_emb_windows,
                self.transpose_batch_sequence,
                self.rotary_pos_emb_group_method,
            )
            key = rotary_pos_emb(
                key,
                self.rotary_pos_emb_windows,
                self.transpose_batch_sequence,
                self.rotary_pos_emb_group_method,
            )
            qkv_layout = QKVLayout.BSHD_BSHD_BSHD

        if qkv_layout == QKVLayout.BSHD_BSHD_BSHD:
            query = query.reshape((*query.shape[:2], self.num_attention_heads, self.head_dim))
            key = key.reshape((*key.shape[:2], self.num_gqa_groups, self.head_dim))
            value = value.reshape((*value.shape[:2], self.num_gqa_groups, self.head_dim))

        if decode:
            assert qkv_layout == QKVLayout.BSHD_BSHD_BSHD
            is_initialized = self.has_variable("cache", "cached_key")

            cached_key = self.variable("cache", "cached_key", jnp.zeros, key.shape, key.dtype)
            cached_value = self.variable(
                "cache", "cached_value", jnp.zeros, value.shape, value.dtype
            )
            cache_index = self.variable(
                "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
            )
            if is_initialized:
                if self.transpose_batch_sequence:
                    length, batch, num_attention_heads, head_dim = cached_key.value.shape
                    expected_shape = (1, batch, num_attention_heads, head_dim)
                    one_hot_indices_shape = (length, 1, 1, 1)
                else:
                    batch, length, num_attention_heads, head_dim = cached_key.value.shape
                    expected_shape = (batch, 1, num_attention_heads, head_dim)
                    one_hot_indices_shape = (1, length, 1, 1)

                # Sanity shape check of cached key against input query.
                if expected_shape != query.shape:
                    raise ValueError(
                        "Autoregressive cache shape error, "
                        f"expected query shape {expected_shape} instead got {query.shape}."
                    )

                cur_index = cache_index.value
                one_hot_indices = jax_nn.one_hot(cur_index, length, dtype=key.dtype)
                one_hot_indices = jnp.reshape(one_hot_indices, one_hot_indices_shape)
                key = cached_key.value + key * one_hot_indices
                value = cached_value.value + value * one_hot_indices
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1

                mask = combine_masks(
                    mask, jnp.broadcast_to(jnp.arange(length) > cur_index, (batch, 1, 1, length))
                )

                if bias is not None:
                    dynamic_vector_slice_in_dim = vmap(
                        lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None)
                    )
                    bias = dynamic_vector_slice_in_dim(
                        jnp.squeeze(bias, axis=0), jnp.reshape(cur_index, (-1)), 1, -2
                    )

        LEADING_AXES = (BATCH_AXES, SEQLEN_AXES)
        if self.transpose_batch_sequence:
            LEADING_AXES = (SEQLEN_AXES, BATCH_AXES)

        if qkv_layout == QKVLayout.BS3HD:
            qkv_proj = qkv_proj.reshape(
                *qkv_proj.shape[:2], 3, self.num_attention_heads, self.head_dim
            )
            qkv_sharding_constraint = (*LEADING_AXES, JOINED_AXES, HEAD_AXES, HIDDEN_AXES)
            qkv_proj = with_sharding_constraint_by_logical_axes(qkv_proj, qkv_sharding_constraint)
            dpa_args = [qkv_proj, None, None]
        elif qkv_layout == QKVLayout.BSHD_BS2HD:
            query = query.reshape(*query.shape[:2], self.num_attention_heads, self.head_dim)
            kv_proj = kv_proj.reshape(*kv_proj.shape[:2], 2, self.num_gqa_groups, self.head_dim)
            q_sharding_constraint = (*LEADING_AXES, HEAD_AXES, HIDDEN_AXES)
            kv_sharding_constraint = (*LEADING_AXES, JOINED_AXES, HEAD_AXES, HIDDEN_AXES)
            query = with_sharding_constraint_by_logical_axes(query, q_sharding_constraint)
            kv_proj = with_sharding_constraint_by_logical_axes(kv_proj, kv_sharding_constraint)
            dpa_args = [query, kv_proj, None]
        else:
            assert qkv_layout == QKVLayout.BSHD_BSHD_BSHD
            query = query.reshape((*query.shape[:2], self.num_attention_heads, self.head_dim))
            key = key.reshape((*key.shape[:2], self.num_gqa_groups, self.head_dim))
            value = value.reshape((*value.shape[:2], self.num_gqa_groups, self.head_dim))
            qkv_sharding_constraint = (*LEADING_AXES, HEAD_AXES, HIDDEN_AXES)
            query = with_sharding_constraint_by_logical_axes(query, qkv_sharding_constraint)
            key = with_sharding_constraint_by_logical_axes(key, qkv_sharding_constraint)
            value = with_sharding_constraint_by_logical_axes(value, qkv_sharding_constraint)
            dpa_args = [query, key, value]

        scale_factor = 1.0 / sqrt(self.head_dim) if self.scale_attn_logits else 1.0
        x = DotProductAttention(
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_gqa_groups=self.num_gqa_groups,
            attn_mask_type=self.attn_mask_type,
            attn_bias_type=self.attn_bias_type,
            attention_dropout=self.attention_dropout,
            dtype=self.dtype,
            dropout_rng_name=self.dropout_rng_name,
            float32_logits=self.float32_logits,
            qkv_layout=qkv_layout.name,
            scale_factor=scale_factor,
            transpose_batch_sequence=self.transpose_batch_sequence,
        )(*dpa_args, mask, bias, deterministic=deterministic)
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))

        attn_context_sharding_constraint = (*LEADING_AXES, HIDDEN_TP_AXES)
        x = with_sharding_constraint_by_logical_axes(x, attn_context_sharding_constraint)

        out = DenseGeneral(
            features=inputs_q.shape[-1],
            transpose_batch_sequence=self.transpose_batch_sequence,
            axis=-1,
            kernel_init=self.kernel_init,
            kernel_axes=(W_TP_AXES, W_FSDP_AXES),
            use_bias=self.use_bias,
            bias_init=self.bias_init,
            bias_axes=(W_NO_SHARD_AXES,),
            enable_low_rank_adaptation=lora_scope.output_proj,
            low_rank_adaptation_dim=self.low_rank_adaptation_dim,
            low_rank_adaptation_alpha=self.low_rank_adaptation_alpha,
            dtype=self.dtype,
            name="out",
        )(x)
        out = checkpoint_name(out, "out_proj")

        return out, ln_out


class RelativePositionBiases(nn.Module):  # pylint: disable=too-few-public-methods
    """
    T5-style relative positional embeddings to the attention logits.

    Parameters
    ----------
    num_buckets: int
        The number of buckets to bucket distances between key and query positions into.
    max_distance: int
        The maximum distance before everything is lumped into the last
        distance bucket.
    num_attention_heads: int
        Number of attention heads in the transformer layer.
    embedding_init: Initializer, default = flax.linen.linear.default_embed_init
        Used for initializing relative embedding tables.
    embedding_axes: Tuple[str, ...], default = ('heads', 'relpos_buckets')
        The name of axes used to shard embedding attention bias with a corresponding mesh.

    Optimization parameters
    -----------------------
    dtype: jax.numpy.dtype, default  = jax.numpy.float32
        The data type used to allocate the initial parameters.
    """

    num_buckets: int
    max_distance: int
    num_attention_heads: int
    embedding_init: Callable[..., Array] = nn.linear.default_embed_init
    embedding_axes: Tuple[str, ...] = ("heads", "relpos_buckets")
    dtype: DType = jnp.float32

    @nn.compact
    def __call__(self, q_seqlen, k_seqlen, bidirectional=True):
        """
        Generate relative position embedding attention biases.

        Parameters
        ----------
        q_seqlen: int
            The sequence length of query.
        k_seqlen: int
            The sequence length of key.
        bidirectional: bool, default = True
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
            np.log(negative_rp.astype(np.float32) / rpb_max_exact + np.finfo(np.float32).eps)
            / np.log(self.max_distance / rpb_max_exact)
            * (rpb_num_buckets - rpb_max_exact)
        ).astype(np.int32)
        rpb_val_if_large = np.minimum(rpb_val_if_large, rpb_num_buckets - 1)
        rp_bucket += np.where(rpb_is_small, negative_rp, rpb_val_if_large)

        # Compute relative attention bias
        relative_attention_bias = nn_partitioning.param_with_axes(
            "rel_embedding",
            self.embedding_init,
            (self.num_attention_heads, self.num_buckets),
            jnp.float32,
            axes=self.embedding_axes,
        )

        relative_attention_bias = jnp.asarray(relative_attention_bias, self.dtype)

        bcast_iota = lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1), 0)
        rp_bucket_one_hot = jnp.array(rp_bucket[jnp.newaxis, ...] == bcast_iota, dtype=self.dtype)

        values = lax.dot_general(
            relative_attention_bias, rp_bucket_one_hot, (((1,), (0,)), ((), ()))
        )
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


class TransformerLayer(nn.Module):  # pylint: disable=too-few-public-methods
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
    num_gqa_groups: int, default = `None`
        Number of GQA groups. When `None` is present, it is equal to num_attention_heads.
        Grouped Query Attention is described in
        `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
        This only affects the keys and values, not the querys.
        GQA-1 is equivalent to Multi-Query Attention
        (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
        is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    layernorm_type: {'layernorm', 'rmsnorm'}, default = 'layernorm'
        Indicate the type of layer normalization.
    layernorm_epsilon: float, default = 1e-6
        A value added to the denominator of layer normalization for numerical stability.
    zero_centered_gamma: bool, default = False
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
        Whether to compute attention logits in float32 for the unfused attention backend.
        For fused attention backend, the accumulation is always float32 without the perf overhead.
    layer_type: TransformerLayerType, default = TransformerLayerType.ENCODER
        If set to TransformerLayerType.DECODER, an additional cross-attention block
        is added after self-attention.this can be used for structures like `T5`
        Transformer in conjunction with the TransformerLayerType.ENCODER option.
    self_attn_mask_type: str, default = 'causal'
        This parameter specifies the type of attention mask to be applied during the softmax
        operation in the self attention.
        Available options are {'no_mask', 'padding', 'causal', 'causal_padding', 'padding_causal'}

        Each described below:

        * no_mask: No attention mask is applied. This means the self attention will consider the
          full sequence without any restrictions.
        * padding: Indicates the presence of padding at the end of each sequence.
          Users must provide a mask with the shape [batch, 1, max_seqlen_q, max_seqlen_kv] in the
          :attr:`__call__` method to specify the padding positions.
        * causal: An upper triangular mask is applied to the softmax inputs,
          ensuring that the prediction for a certain position is only dependent on known outputs
          from positions before it.
        * causal_padding / padding_causal: A combination of both causal and padding masks.
          Both 'causal_padding' and 'padding_causal' are acceptable and have the same effect.

        .. note:: :attr:`attention_mask` in :attr:`__call__` is ignored for 'no_mask' and 'causal'.

    self_attn_bias_type: Optional[str], default = None
        Type of the attention bias passed into the self attention.
        Available options: {'no_bias', 'pre_scale_bias', 'post_scale_bias'}.
        When default is present, the type is automatically decided by the MHA's bias parameter.
        Where it is `post_scale_bias` if there is bias. Otherwise `no_bias` is used.
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
    rotary_pos_emb_group_method: str, default = 'consecutive'
        Indicate the method to coupled the coordinates. It should be one of
        ['consecutive', 'alternate']. 'alternate' is to pair index :math:`i` with :math:`i + d/2`
        , d is the hidden dimension. 'consecutive' pairs index :math:`i` with :math:`i + 1`.
    low_rank_adaptation_scope: str, default = 'none'
        Indicate the scope to apply low rank adaptation. It should be one of
        ['none', 'all', 'qkv_proj', 'output_proj', 'mlp', 'exclude_qkv_proj',
         'exclude_output_proj', 'exclude_mlp']
    low_rank_adaptation_dim: int, default = 32
        The dimension for low rank adaptation, only used when
        :attr:`enable_low_rank_adaptation=True`
    low_rank_adaptation_alpha: float, default = None
        The alpha for computing the scaling factor of LoRA output.
        :math:`\frac{alpha}{rank} * lora_output`. None means no scaling.
    enable_sequence_parallel: bool, default = False
        Whether to enable sequence parallelism to operations except dot.

    Optimization parameters
    -----------------------
    dtype: jax.numpy.dtype, default  = jax.numpy.float32
        The data type used to allocate the initial parameters.
    drop_path: float, default = 0.0
        When > 0.0, applies stochastic depth per sample in the main
        path of the residual block.
    fuse_qkv_params: bool, default = True
        If set to True, `TransformerLayer` module exposes a single fused
        parameter for query-key-value for self-attention and key-value for
        cross-attention.
    transpose_batch_sequence: bool, default = False
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
    num_gqa_groups: Optional[int] = None
    layernorm_type: str = "layernorm"
    layernorm_epsilon: float = 1e-6
    zero_centered_gamma: bool = False
    hidden_dropout: float = 0.1
    hidden_dropout_dims: Sequence[int] = ()
    attention_dropout: float = 0.1
    intermediate_dropout: float = 0.1
    intermediate_dropout_dims: Sequence[int] = ()
    dropout_rng_name: str = "dropout"
    mha_kernel_init: Initializer = None
    mlp_kernel_init: Initializer = None
    mlp_activations: Sequence[str] = ("relu",)
    use_bias: bool = False
    bias_init: Initializer = nn.initializers.zeros
    apply_residual_connection_post_layernorm: bool = False
    output_layernorm: bool = False
    float32_attention_logits: bool = False
    layer_type: TransformerLayerType = TransformerLayerType.ENCODER
    self_attn_mask_type: str = "causal"
    self_attn_bias_type: Optional[str] = None
    enable_relative_embedding: bool = True
    relative_embedding: nn.Module = None
    enable_rotary_pos_emb: bool = False
    rotary_pos_emb_windows: Tuple[int, int] = (1, 10000)
    rotary_pos_emb_group_method: str = "consecutive"
    low_rank_adaptation_scope: str = "none"
    low_rank_adaptation_dim: int = 32
    low_rank_adaptation_alpha: float = None
    dtype: DType = jnp.float32
    drop_path: float = 0.0
    fuse_qkv_params: bool = True
    transpose_batch_sequence: bool = False
    enable_sequence_parallel: bool = False
    scale_attn_logits: bool = False
    scaled_query_init: bool = True

    def __post_init__(self):
        if self.mha_kernel_init is None:
            self.mha_kernel_init = nn.initializers.variance_scaling(1.0, "fan_in", "normal")
        if self.mlp_kernel_init is None:
            self.mlp_kernel_init = nn.initializers.variance_scaling(
                1.0, "fan_in", "truncated_normal"
            )
        if self.num_gqa_groups is None:
            self.num_gqa_groups = self.num_attention_heads
        super().__post_init__()

    @nn.compact
    def __call__(
        self,
        inputs: Array,
        encoded: Array = None,
        attention_mask: Array = None,
        encoder_decoder_mask: Array = None,
        deterministic: bool = False,
        decode: bool = False,
        max_decode_length: bool = None,
    ):
        """
        Transformer Layer: attention block and a feedforward network (MLP)

        Parameters
        ----------
        inputs: jax.numpy.ndarray
            Input tensor.
        encoded: jax.numpy.ndarray, default = None
            Output tensors of the encoder block to be fed into the decoder block if using
            :attr:`layer_type=TransformerLayerType.DECODER`.
        attention_mask : jax.numpy.ndarray, default = None
            Boolean tensor used to mask out self-attention softmax input.
            :attr:`True` means mask out the corresponding values.
            Ignored when :attr:`self.self_attn_mask_type` is either 'no_mask' or 'causal'.
        encoder_decoder_mask: jax.numpy.ndarray, default = None
            Boolean tensor used to mask out cross-attention softmax input when
            :attr:`layer_type=TransformerLayerType.DECODER`.
            :attr:`True` means mask out the corresponding values.
        deterministic: bool, default = False
            Disable dropout layers if set to True.
        decode: bool, default = False
            Indicate whether to prepare and use an autoregressive cache
            in Multi-head attention (MHA).
        max_decode_length: bool, default = None
            The maximum length to generate relative embedding biases when
            :attr:`layer_type=TransformerLayerType.DECODER` and
            :attr:`enable_relative_embedding=True`.

        Returns
        -------
        outputs: jax.numpy.ndarray
            Output tensors.
        """
        assert (
            self.layer_type in TransformerLayerType
        ), f"layer_type should be one of TransformerLayerType, but got {self.layer_type}."

        assert self.hidden_size % self.num_attention_heads == 0, (
            "hidden_size should be multiples of num_attention_heads"
            f", but got {self.hidden_size=} and {self.num_attention_heads=}."
        )

        assert self.layer_type == TransformerLayerType.DECODER or (
            self.layer_type == TransformerLayerType.ENCODER and decode is False
        ), "decode should be False when layer_type == TransformerLayerType.ENCODER."

        head_dim = self.hidden_size // self.num_attention_heads

        sequence_dim = 0 if self.transpose_batch_sequence else 1
        batch_dim = 1 - sequence_dim

        def generate_batch_seqlen_logical_axes(is_shared_seq=None):
            axes = [None, None]

            is_shared_seq = (
                self.enable_sequence_parallel if is_shared_seq is None else is_shared_seq
            )

            axes[batch_dim] = BATCH_AXES
            axes[sequence_dim] = SEQLEN_TP_AXES if is_shared_seq else SEQLEN_AXES
            return tuple(axes)

        attn_bias = None
        if self.enable_relative_embedding:
            if self.relative_embedding is None:
                rel_emb = RelativePositionBiases(
                    num_buckets=32,
                    max_distance=128,
                    num_attention_heads=self.num_attention_heads,
                    dtype=self.dtype,
                    embedding_init=nn.initializers.variance_scaling(1.0, "fan_avg", "uniform"),
                    name="relpos_bias",
                )
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
            mha_name = "attention"
        else:
            mha_name = "self_attention"

        inputs = with_sharding_constraint_by_logical_axes(
            inputs, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES)
        )

        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        residual = inputs
        x, ln_out = MultiHeadAttention(
            num_attention_heads=self.num_attention_heads,
            dtype=self.dtype,
            head_dim=head_dim,
            num_gqa_groups=self.num_gqa_groups,
            transpose_batch_sequence=self.transpose_batch_sequence,
            enable_sequence_parallel=self.enable_sequence_parallel,
            attention_dropout=self.attention_dropout,
            dropout_rng_name=self.dropout_rng_name,
            float32_logits=self.float32_attention_logits,
            scale_attn_logits=self.scale_attn_logits,
            scaled_query_init=self.scaled_query_init,
            layernorm_type=self.layernorm_type,
            layernorm_epsilon=self.layernorm_epsilon,
            zero_centered_gamma=self.zero_centered_gamma,
            return_layernorm_output=self.apply_residual_connection_post_layernorm,
            input_layernorm=not self.output_layernorm,
            attn_mask_type=self.self_attn_mask_type,
            attn_bias_type=self.self_attn_bias_type,
            enable_rotary_pos_emb=self.enable_rotary_pos_emb,
            rotary_pos_emb_windows=self.rotary_pos_emb_windows,
            rotary_pos_emb_group_method=self.rotary_pos_emb_group_method,
            low_rank_adaptation_scope=self.low_rank_adaptation_scope,
            low_rank_adaptation_dim=self.low_rank_adaptation_dim,
            low_rank_adaptation_alpha=self.low_rank_adaptation_alpha,
            fuse_qkv_params=self.fuse_qkv_params,
            kernel_init=self.mha_kernel_init,
            use_bias=self.use_bias,
            bias_init=self.bias_init,
            name=mha_name,
        )(inputs, inputs, attention_mask, attn_bias, deterministic=deterministic, decode=decode)

        def hidden_dropout(x, deterministic):
            assert isinstance(self.hidden_dropout_dims, Sequence)
            x_shape_len = len(x.shape)
            for dims in self.hidden_dropout_dims:
                assert -x_shape_len <= dims < x_shape_len

            return nn.Dropout(
                rate=self.hidden_dropout,
                broadcast_dims=self.hidden_dropout_dims,
                rng_collection=self.dropout_rng_name,
            )(x, deterministic=deterministic)

        x = with_sharding_constraint_by_logical_axes(
            x, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES)
        )
        residual = with_sharding_constraint_by_logical_axes(
            residual, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES)
        )

        x = hidden_dropout(x, deterministic)
        if self.drop_path > 0.0:
            drop_path_shape = _generate_drop_path_shape(x.shape, batch_dim)
            x = nn.Dropout(
                rate=self.drop_path,
                broadcast_dims=drop_path_shape,
                rng_collection=self.dropout_rng_name,
            )(x, deterministic=deterministic)

        if self.apply_residual_connection_post_layernorm:
            assert ln_out is not None
            residual = ln_out

        x = x + residual

        mlp_input = x
        if self.layer_type == TransformerLayerType.DECODER:
            assert (
                encoded is not None
            ), "encoded is required when layer_type == TransformerLayerType.DECODER."

            x = with_sharding_constraint_by_logical_axes(
                x, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES)
            )

            residual = x
            y, ln_out = MultiHeadAttention(
                num_attention_heads=self.num_attention_heads,
                dtype=self.dtype,
                head_dim=head_dim,
                num_gqa_groups=self.num_gqa_groups,
                transpose_batch_sequence=self.transpose_batch_sequence,
                enable_sequence_parallel=self.enable_sequence_parallel,
                attention_dropout=self.attention_dropout,
                dropout_rng_name=self.dropout_rng_name,
                layernorm_type=self.layernorm_type,
                layernorm_epsilon=self.layernorm_epsilon,
                zero_centered_gamma=self.zero_centered_gamma,
                return_layernorm_output=self.apply_residual_connection_post_layernorm,
                input_layernorm=True,  # Must do LayerNorm before MHA.
                attn_mask_type="padding",
                attn_bias_type="no_bias",
                enable_rotary_pos_emb=self.enable_rotary_pos_emb,
                rotary_pos_emb_windows=self.rotary_pos_emb_windows,
                rotary_pos_emb_group_method=self.rotary_pos_emb_group_method,
                low_rank_adaptation_scope=self.low_rank_adaptation_scope,
                low_rank_adaptation_dim=self.low_rank_adaptation_dim,
                low_rank_adaptation_alpha=self.low_rank_adaptation_alpha,
                float32_logits=self.float32_attention_logits,
                scale_attn_logits=self.scale_attn_logits,
                scaled_query_init=self.scaled_query_init,
                fuse_qkv_params=self.fuse_qkv_params,
                kernel_init=self.mha_kernel_init,
                use_bias=self.use_bias,
                bias_init=self.bias_init,
                name="encoder_decoder_attention",
            )(x, encoded, encoder_decoder_mask, deterministic=deterministic)

            y = with_sharding_constraint_by_logical_axes(
                y, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES)
            )
            residual = with_sharding_constraint_by_logical_axes(
                residual, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES)
            )

            y = hidden_dropout(y, deterministic)

            if self.apply_residual_connection_post_layernorm:
                assert ln_out is not None
                residual = ln_out

            mlp_input = y + residual

        mlp_input = with_sharding_constraint_by_logical_axes(
            mlp_input, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES)
        )

        lora_scope = _canonicalize_lora_scope(self.low_rank_adaptation_scope)

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
            enable_low_rank_adaptation=lora_scope.mlp,
            low_rank_adaptation_dim=self.low_rank_adaptation_dim,
            low_rank_adaptation_alpha=self.low_rank_adaptation_alpha,
            layernorm_input_axes=(*generate_batch_seqlen_logical_axes(), HIDDEN_AXES),
            dot_1_input_axes=(*generate_batch_seqlen_logical_axes(False), HIDDEN_AXES),
            dot_2_input_axes=(*generate_batch_seqlen_logical_axes(False), HIDDEN_TP_AXES),
            name="mlp",
        )(mlp_input, deterministic=deterministic)

        if self.apply_residual_connection_post_layernorm:
            assert ln_out is not None
            residual = ln_out

        z = with_sharding_constraint_by_logical_axes(
            z, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES)
        )
        residual = with_sharding_constraint_by_logical_axes(
            residual, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES)
        )

        z = hidden_dropout(z, deterministic)
        if self.drop_path > 0.0:
            drop_path_shape = _generate_drop_path_shape(z.shape, batch_dim)
            z = nn.Dropout(rate=self.drop_path, broadcast_dims=drop_path_shape)(
                z, deterministic=deterministic
            )
        z = z + residual

        if self.output_layernorm:
            z = with_sharding_constraint_by_logical_axes(
                z, (*generate_batch_seqlen_logical_axes(), HIDDEN_AXES)
            )
            z = LayerNorm(
                layernorm_type=self.layernorm_type,
                zero_centered_gamma=self.zero_centered_gamma,
                epsilon=self.layernorm_epsilon,
                scale_axes=(W_NO_SHARD_AXES,),
                bias_axes=(W_NO_SHARD_AXES,),
                transpose_batch_sequence=self.transpose_batch_sequence,
                dtype=self.dtype,
                name="output_layernorm",
            )(z)

        return z

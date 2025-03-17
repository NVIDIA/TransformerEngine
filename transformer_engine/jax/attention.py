# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX multi-head attention modules"""
from __future__ import annotations
from enum import Enum
from functools import partial
from typing import Optional, Tuple, Union
import warnings

from jax.ad_checkpoint import checkpoint_name
import jax
import jax.numpy as jnp
from flax.linen import make_attention_mask

from transformer_engine_jax import NVTE_Bias_Type
from transformer_engine_jax import NVTE_Mask_Type
from transformer_engine_jax import NVTE_QKV_Layout
from transformer_engine_jax import NVTE_QKV_Format
from transformer_engine_jax import nvte_get_qkv_format

from . import cpp_extensions as tex


class AttnBiasType(Enum):
    """
    NO_BIAS: Softmax is performed as softmax(scale * qk)
    PRE_SCALE_BIAS: Softmax is performed as softmax(scale * (qk + bias))
    POST_SCALE_BIAS: Softmax is performed as softmax(scale * qk + bias)
    """

    NO_BIAS = NVTE_Bias_Type.NVTE_NO_BIAS
    PRE_SCALE_BIAS = NVTE_Bias_Type.NVTE_PRE_SCALE_BIAS
    POST_SCALE_BIAS = NVTE_Bias_Type.NVTE_POST_SCALE_BIAS


class AttnMaskType(Enum):
    """
    NO_MASK: No attention mask is applied.
    PADDING_MASK: Indicates the presence of paddings at the end of each sequence.
    CAUSAL_MASK: An upper triangular mask is applied to the softmax inputs.
    PADDING_CAUSAL_MASK: A combination of both causal and padding masks.
    """

    NO_MASK = NVTE_Mask_Type.NVTE_NO_MASK
    PADDING_MASK = NVTE_Mask_Type.NVTE_PADDING_MASK
    CAUSAL_MASK = NVTE_Mask_Type.NVTE_CAUSAL_MASK
    PADDING_CAUSAL_MASK = NVTE_Mask_Type.NVTE_PADDING_CAUSAL_MASK
    CAUSAL_BOTTOM_RIGHT_MASK = NVTE_Mask_Type.NVTE_CAUSAL_BOTTOM_RIGHT_MASK
    PADDING_CAUSAL_BOTTOM_RIGHT_MASK = NVTE_Mask_Type.NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK

    def is_causal(self):
        """Returns True if the mask is a causal mask"""
        return self in [
            AttnMaskType.CAUSAL_MASK,
            AttnMaskType.PADDING_CAUSAL_MASK,
            AttnMaskType.CAUSAL_BOTTOM_RIGHT_MASK,
            AttnMaskType.PADDING_CAUSAL_BOTTOM_RIGHT_MASK,
        ]

    def is_padding(self):
        """Returns True if the mask includes padding"""
        return self in [
            AttnMaskType.PADDING_MASK,
            AttnMaskType.PADDING_CAUSAL_MASK,
            AttnMaskType.PADDING_CAUSAL_BOTTOM_RIGHT_MASK,
        ]

    def is_bottom_right(self):
        """Returns True if the causal mask is calculated from the bottom-right section"""
        return self in [
            AttnMaskType.CAUSAL_BOTTOM_RIGHT_MASK,
            AttnMaskType.PADDING_CAUSAL_BOTTOM_RIGHT_MASK,
        ]


class QKVFormat(Enum):
    """
    SBHD: q,k,v memory layout with [s, b, ..., h, d]
    BSHD: q,k,v memory layout with [b, s, ..., h, d]
    THD: q,k,v memory layout is same as BSHD, but allow multiple segments packed in a sequence.
    """

    SBHD = NVTE_QKV_Format.NVTE_SBHD
    BSHD = NVTE_QKV_Format.NVTE_BSHD
    THD = NVTE_QKV_Format.NVTE_THD


class QKVLayout(Enum):
    """
    BSHD Format:
        - BS3HD: q,k,v are interleave packed as a tensor with shape [b, s, 3, h, d].
        - BSHD_BS2HD: q with shape [b, s, h, d] and kv are interleaved with shape [b, s, 2, h, d].
        - BSHD_BSHD_BSHD: q,k,v are seperate tensors with shape [b, s, h, d]
    THD Format: Shape is same as BSHD layout but allow multiple segments packed in a sequence.
        - T3HD: q,k,v are interleave packed as a tensor with shape [b, s, 3, h, d].
        - THD_T2HD: q with shape [b, s, h, d] and kv are interleaved with shape [b, s, 2, h, d].
        - THD_THD_THD: q,k,v are seperate tensors with shape [b, s, h, d]
    """

    BS3HD = NVTE_QKV_Layout.NVTE_BS3HD
    BSHD_BS2HD = NVTE_QKV_Layout.NVTE_BSHD_BS2HD
    BSHD_BSHD_BSHD = NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD
    T3HD = NVTE_QKV_Layout.NVTE_T3HD
    THD_T2HD = NVTE_QKV_Layout.NVTE_THD_T2HD
    THD_THD_THD = NVTE_QKV_Layout.NVTE_THD_THD_THD

    def get_qkv_format(self):
        """
        Return the corresponding qkv_format (BSHD, SBHD, THD)
        """
        return QKVFormat(nvte_get_qkv_format(self.value))

    def is_qkvpacked(self):
        """
        Return True if the query, key, value is packed
        """
        return self in [QKVLayout.BS3HD, QKVLayout.T3HD]

    def is_kvpacked(self):
        """
        Return True if the key, value is packed
        """
        return self in [QKVLayout.BSHD_BS2HD, QKVLayout.THD_T2HD]

    def is_separate(self):
        """
        Return True if the query, key, value are three separate tensors
        """
        return self in [QKVLayout.BSHD_BSHD_BSHD, QKVLayout.THD_THD_THD]

    def is_thd(self):
        """
        Return True if the layout belongs to THD
        """
        return self in [QKVLayout.T3HD, QKVLayout.THD_T2HD, QKVLayout.THD_THD_THD]


class CPStrategy(Enum):
    """Defines the context parallel strategies of Jax fused attention.

    DEFAULT: Default strategy will choose automatically if context parallel axis is sharded.
    ALL_GATHER: All-gather/reduce scatter implementation.
    RING: Ring attention implementation (https://arxiv.org/abs/2310.01889).
    """

    DEFAULT = 0
    ALL_GATHER = 1
    RING = 2


def make_swa_mask(
    segment_pos_q: jnp.ndarray,
    segment_pos_kv: jnp.ndarray,
    window_size: Optional[Tuple[int, int]] = None,
    dtype: jax.typing.DTypeLike = jnp.float32,
):
    """
    Generate a sliding window mask (1 = attend, 0 = masked).

    Args:
        segment_pos_q (jnp.ndarray):
            Query positions within each segment. For example, a batch with segment_ids =
            [[1, 1, 1, 2, 2, 2, 2, 2]] yields segment_pos =
            [[0, 1, 2, 0, 1, 2, 3, 4]].
        segment_pos_kv (jnp.ndarray):
            Key/value positions within each segment.
        window_size (Optional[Tuple[int, int]], optional):
            Sliding window size for local attention, where query at position i attends to keys
            in [i - window_size[0], i + window_size[1]] inclusive. A negative number means an
            infinite window; None means no sliding window.
            Defaults to None.
        dtype (jax.typing.DTypeLike, optional):
            Mask data type. Defaults to jnp.float32.

    Returns:
        jnp.ndarray:
            The mask with shape [b, 1, max_seqlen_q, max_seqlen_kv].
    """
    if window_size is not None:
        left_window, right_window = window_size
    else:
        left_window = right_window = jnp.inf
    left_window = jnp.inf if left_window < 0 else left_window
    right_window = jnp.inf if right_window < 0 else right_window
    pos_q = jnp.expand_dims(segment_pos_q, axis=-1)
    pos_kv = jnp.expand_dims(segment_pos_kv, axis=-2)
    inv_swa_mask = (pos_kv >= pos_q - left_window) & (pos_kv <= pos_q + right_window)
    inv_swa_mask = jnp.expand_dims(inv_swa_mask, axis=-3)
    return inv_swa_mask.astype(dtype)


def canonicalize_attn_mask_type(attn_mask_type: str):
    """Convert string attn_mask_type to AttnMaskType
    TE-JAX currently fall back to the padding version kernels for the libraries integration.
    The overhead between padding and non-padding version should be small.
    However, we will lease this limitation in the near feature.
    """
    match attn_mask_type:
        case "no_mask":
            return AttnMaskType.NO_MASK
        case "padding":
            return AttnMaskType.PADDING_MASK
        case "causal":
            return AttnMaskType.CAUSAL_MASK
        case "causal_bottom_right" | "bottom_right_causal":
            return AttnMaskType.CAUSAL_BOTTOM_RIGHT_MASK
        case "padding_causal" | "causal_padding":
            return AttnMaskType.PADDING_CAUSAL_MASK
        case (
            "padding_causal_bottom_right"
            | "causal_padding_bottom_right"
            | "bottom_right_causal_padding"
            | "bottom_right_padding_causal"
        ):
            return AttnMaskType.PADDING_CAUSAL_BOTTOM_RIGHT_MASK
    raise ValueError(
        f"Unsupported {attn_mask_type=}, supported attn_mask_type={{'no_mask', 'padding', 'causal',"
        " 'padding_causal', 'causal_padding', 'causal_bottom_right',"
        " 'padding_causal_bottom_right'}"
    )


def is_fused_attn_kernel_available(
    q_dtype,
    kv_dtype,
    qkv_layout,
    attn_bias_type,
    attn_mask_type,
    dropout_probability,
    q_num_heads,
    kv_num_heads,
    q_max_seqlen,
    kv_max_seqlen,
    head_dim,
    window_size: Optional[Tuple[int, int]] = None,
):
    """
    To check whether the fused attention kernel is supported
    """

    def make_helper(attn_mask_type):
        return tex.FusedAttnHelper(
            q_dtype,
            kv_dtype,
            qkv_layout.value,
            attn_bias_type.value,
            attn_mask_type.value,
            dropout_probability,
            q_num_heads,
            kv_num_heads,
            q_max_seqlen,
            kv_max_seqlen,
            head_dim,
            (-1, -1) if window_size is None else window_size,
        )

    return make_helper(attn_mask_type).is_fused_attn_kernel_available()


def _obtain_batch_and_max_seqlen(qkv, qkv_layout):
    if qkv_layout.is_qkvpacked():
        assert len(qkv) == 1, f"qkv must be (qkvpacked,) with {qkv_layout=}"
        batch, q_max_seqlen, *_ = qkv[0].shape
        kv_max_seqlen = q_max_seqlen
    elif qkv_layout.is_kvpacked():
        assert len(qkv) == 2, f"qkv must be (query, kvpacked) with {qkv_layout=}"
        batch, q_max_seqlen, *_ = qkv[0].shape
        kv_max_seqlen = qkv[1].shape[1]
    elif qkv_layout.is_separate():
        assert len(qkv) == 3, f"qkv must be (query, key, value) with {qkv_layout=}"
        batch, q_max_seqlen, *_ = qkv[0].shape
        kv_max_seqlen = qkv[1].shape[1]
    else:
        raise ValueError(f"Unsupported {qkv_layout=}")
    return batch, q_max_seqlen, kv_max_seqlen


def reorder_causal_load_balancing(tensor, cp_size: int, tensor_format: QKVFormat):
    """Reorders a tensor for load balancing the compute of causal attention."""
    seq_dim = 1 if tensor_format == QKVFormat.BSHD else 0
    return tex.attention.reorder_causal_load_balancing(tensor, cp_size, seq_dim, False)


def inverse_reorder_causal_load_balancing(tensor, cp_size: int, tensor_format: QKVFormat):
    """Inverse operation of `reorder_causal_load_balancing`."""
    seq_dim = 1 if tensor_format == QKVFormat.BSHD else 0
    return tex.attention.reorder_causal_load_balancing(tensor, cp_size, seq_dim, True)


def _get_seqlens_and_offsets(segment_ids, max_segments_per_seq):
    # bincount map with 0s
    bincount_vmap = jax.vmap(partial(jnp.bincount, length=max_segments_per_seq + 1))
    seqlens_with_zero = bincount_vmap(segment_ids.astype(jnp.int32))
    seqlens = seqlens_with_zero[..., 1:]

    def _find_offsets(x):
        same_as_previous = jnp.logical_and(x[..., 1:] != x[..., :-1], x[..., 1:] != 0)
        first_column = x[..., :1] != 0
        same_as_previous = jnp.hstack((first_column, same_as_previous))
        return jax.vmap(partial(jnp.argwhere, size=(max_segments_per_seq + 1), fill_value=-1))(
            same_as_previous
        ).squeeze(-1)

    offsets = _find_offsets(segment_ids)
    return seqlens, offsets


def _mask_to_seqlens_offset(mask, max_segments_per_seq):
    assert mask.shape[1] == 1
    row_ids = mask.squeeze(axis=1).max(axis=-1)
    q_seqlen, q_offset = _get_seqlens_and_offsets(row_ids, max_segments_per_seq)
    col_ids = mask.squeeze(axis=1).max(axis=-2)
    kv_seqlen, kv_offset = _get_seqlens_and_offsets(col_ids, max_segments_per_seq)
    return q_seqlen, q_offset, kv_seqlen, kv_offset


def _segment_ids_pos_to_seqlens_offsets(
    segment_ids_q,
    segment_ids_kv,
    segment_pos_q,
    segment_pos_kv,
    attn_mask_type,
    window_size,
    max_segments_per_seq,
):
    # (1 = attend, 0 = masked)
    segment_mask = make_attention_mask(
        segment_ids_q,
        segment_ids_kv,
        jnp.equal,
    )
    segment_mask_with_id = make_attention_mask(
        segment_ids_q,
        segment_ids_kv,
        lambda x, y: jnp.equal(x, y) * x,
    )
    attn_mask = segment_mask
    if attn_mask_type.is_causal():
        causal_mask = make_attention_mask(
            segment_pos_q,
            segment_pos_kv,
            jnp.greater_equal,
        )
        attn_mask = jnp.logical_and(segment_mask, causal_mask)

    swa_mask = make_swa_mask(segment_pos_q, segment_pos_kv, window_size, dtype=jnp.bool)
    attn_mask = jnp.logical_and(attn_mask, swa_mask)

    attn_mask_with_id = jnp.where(attn_mask, segment_mask_with_id, 0)
    q_seqlen, q_offset, kv_seqlen, kv_offset = _mask_to_seqlens_offset(
        attn_mask_with_id, max_segments_per_seq
    )
    return q_seqlen, kv_seqlen, q_offset, kv_offset


def _segment_ids_to_seqlens(segment_ids_q, segment_ids_kv, attn_mask_type):
    # convert the mask to seqlens, mask doesn't support ragged offsets
    if not attn_mask_type.is_padding():
        q_max_seqlen = segment_ids_q.shape[-1]
        kv_max_seqlen = segment_ids_kv.shape[-1]
        q_seq_lens = jnp.full_like(q_max_seqlen, q_max_seqlen, dtype=jnp.int32)
        kv_seq_lens = jnp.full_like(kv_max_seqlen, kv_max_seqlen, dtype=jnp.int32)
    else:
        q_seq_lens = jnp.sum(segment_ids_q, axis=-1).astype(jnp.int32)
        kv_seq_lens = jnp.sum(segment_ids_kv, axis=-1).astype(jnp.int32)
    return q_seq_lens, kv_seq_lens


@jax.tree_util.register_pytree_node_class
class SequenceDescriptor:
    """A class to descibe the sequences with flexible initialization.
    - SequenceDescriptor.from_seqlens
      For non-THD (non-packed) cases, where each batch has only 1 sequence.
    - SequenceDescriptor.from_seqlens_and_offsets
      For THD (packed) cases, where each batch may have not only 1 sequence.
    - SequenceDescriptor.from_segment_ids_and_pos
      Experimental feature for THD (packed) cases with context parallelism.
    """

    seqlens: Optional[Tuple[jnp.ndarray, jnp.ndarray]]
    seq_offsets: Optional[Tuple[jnp.ndarray, jnp.ndarray]]
    segment_ids: Optional[Tuple[jnp.ndarray, jnp.ndarray]]
    segment_pos: Optional[Tuple[jnp.ndarray, jnp.ndarray]]

    def __init__(self, seqlens=None, seq_offsets=None, segment_ids=None, segment_pos=None):
        """
        Initialize to Tuple(jnp.zeros, jnp.zeros) because the primitive only accepts pure jax array
        """
        self.seqlens = (jnp.zeros(0), jnp.zeros(0)) if seqlens is None else seqlens
        self.seq_offsets = (jnp.zeros(0), jnp.zeros(0)) if seq_offsets is None else seq_offsets
        self.segment_ids = (jnp.zeros(0), jnp.zeros(0)) if segment_ids is None else segment_ids
        self.segment_pos = (jnp.zeros(0), jnp.zeros(0)) if segment_pos is None else segment_pos

    def tree_flatten(self):
        """
        Flatten method to register as a pytree node
        """
        return ((self.seqlens, self.seq_offsets, self.segment_ids, self.segment_pos), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Unflatten method to register as a pytree node
        """
        del aux_data
        return cls(*children)

    def get_seqlens_and_offsets(
        self, attn_mask_type, qkv_layout, window_size, max_segments_per_seq
    ):
        """
        Acquire the seqlens/offsets for cuDNN backend
        """
        attn_mask_type = AttnMaskType(attn_mask_type)
        qkv_layout = QKVLayout(qkv_layout)
        q_segment_ids, kv_segment_ids = self.segment_ids
        q_segment_pos, kv_segment_pos = self.segment_pos
        assert q_segment_ids.shape == q_segment_pos.shape
        assert kv_segment_ids.shape == kv_segment_pos.shape
        # No segment_ids/segment_pos
        if q_segment_ids.size + kv_segment_ids.size == 0:
            return self.seqlens, self.seq_offsets

        if qkv_layout.is_thd():
            q_seqlens, kv_seqlens, q_offsets, kv_offsets = _segment_ids_pos_to_seqlens_offsets(
                q_segment_ids,
                kv_segment_ids,
                q_segment_pos,
                kv_segment_pos,
                attn_mask_type,
                window_size,
                max_segments_per_seq,
            )
        else:
            q_seqlens, kv_seqlens = _segment_ids_to_seqlens(
                q_segment_ids,
                kv_segment_ids,
                attn_mask_type,
            )
            q_offsets = kv_offsets = jnp.zeros(0)
        return (q_seqlens, kv_seqlens), (q_offsets, kv_offsets)

    @classmethod
    def _expand_to_pair(
        cls, value: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Internal helper to ensure a single value expands into a pair (q, kv).
        """
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("Input tuple must have exactly 2 elements.")
            return value

        if isinstance(value, jnp.ndarray):
            return value, value  # Duplicate for q=kv case

        raise TypeError(
            "Expected a jax.numpy.ndarray or a tuple of two jax.numpy.ndarray, "
            f"but got {type(value).__name__}."
        )

    @classmethod
    def from_seqlens(
        cls,
        seqlens: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    ) -> SequenceDescriptor:
        """
        Factory method for inputs with sequence lengths only (non-THD).
        Args:
            seqlens(Tuple(jnp.ndarray, jnp.ndarray)) = (q_seqlens, kv_seqlens):
                - q_seqlens (jnp.ndarray):
                  Sequence lengths for the query, with shape [batch].
                - kv_seqlen (jnp.ndarray):
                  Sequence lengths for the key and value, with shape [batch].
        Return:
            A SequenceDescriptor with only seqlens initialized.
        """
        q_seqlens, kv_seqlens = cls._expand_to_pair(seqlens)
        return cls(seqlens=(q_seqlens, kv_seqlens))

    @classmethod
    def from_seqlens_and_offsets(
        cls,
        seqlens: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
        seq_offsets: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
    ) -> SequenceDescriptor:
        """
        Factory method for inputs with sequence lengths and offsets (THD).
        Args:
            seqlens(Tuple(jnp.ndarray, jnp.ndarray)) = (q_seqlens, kv_seqlens):
                - q_seqlens (jnp.ndarray):
                  Sequence lengths for the query, with shape [batch, max_seqlen].
                  Unused positions are padded with -1.
                - kv_seqlen (jnp.ndarray):
                  Sequence lengths for the key and value, with shape [batch, max_seqlen].
                  Unused positions are padded with -1.
            seq_offsets(Tuple(jnp.ndarray, jnp.ndarray)) = (q_offsets, kv_offsets)
                - q_seq_offsets (jnp.ndarray):
                  The offsets in the sequence dim for the query, with shape [batch, max_seqlen + 1].
                  Unused positions are padded with -1.
                - kv_seq_offsets (jnp.ndarray):
                  The offsets in the sequence dim for the query, with shape [batch, max_seqlen + 1].
                  Unused positions are padded with -1.
        Return:
            A SequenceDescriptor with seqlens/seq_offsets initialized.
        """
        q_seqlens, kv_seqlens = cls._expand_to_pair(seqlens)
        q_offsets, kv_offsets = cls._expand_to_pair(seq_offsets)
        return cls(seqlens=(q_seqlens, kv_seqlens), seq_offsets=(q_offsets, kv_offsets))

    @classmethod
    def from_segment_ids_and_pos(
        cls,
        segment_ids: Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]],
        segment_pos: Optional[Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]] = None,
    ) -> SequenceDescriptor:
        """
        Experimental factory method for inputs with segment IDs and optional positions. (THD)
        Args:
            segment_ids(Tuple(jnp.ndarray, jnp.ndarray)) = (q_segment_ids, kv_segment_ids):
                - q_segment_ids (jnp.ndarray):
                  Query segment ids start with 1, with shape [batch, max_seqlen].
                  0s are treated as paddings.
                - kv_segment_ids (jnp.ndarray):
                  Key, value segment ids start with 1, with shape [batch, max_seqlen].
                  0s are treated as paddings.
            segment_pos(Tuple(jnp.ndarray, jnp.ndarray)) = (q_segment_pos, kv_segment_pos)
                - q_segment_pos (jnp.ndarray):
                  The position inside each segment for query, with shape [batch, max_seqlen].
                - kv_segment_pos (jnp.ndarray):
                  The position inside each segment for key, value, with shape [batch, max_seqlen].
        Return:
            A SequenceDescriptor with segment_ids/segment_pos initialized.
        """
        q_seg_ids, kv_seg_ids = cls._expand_to_pair(segment_ids)

        if segment_pos is not None:
            segment_pos = cls._expand_to_pair(segment_pos)
        else:

            def generate_default_pos(segment_ids):
                seqlen = segment_ids.shape[-1]
                return jnp.broadcast_to(jnp.arange(seqlen), segment_ids.shape)

            q_seg_pos = generate_default_pos(q_seg_ids)
            kv_seg_pos = generate_default_pos(kv_seg_ids)
            segment_pos = (q_seg_pos, kv_seg_pos)

        return cls(
            segment_ids=(q_seg_ids, kv_seg_ids),
            segment_pos=segment_pos,
        )


def _legacy_fused_attn(
    qkv: Tuple[jnp.ndarray, ...],
    bias: Optional[jnp.ndarray],
    mask: Optional[jnp.ndarray],
    seed: Optional[jnp.ndarray],
    attn_bias_type: AttnBiasType,
    attn_mask_type: AttnMaskType,
    qkv_layout: QKVLayout,
    scaling_factor: float,
    dropout_probability: float,
    is_training: bool,
    window_size: Optional[Tuple[int, int]] = None,
    context_parallel_strategy: CPStrategy = CPStrategy.DEFAULT,
    context_parallel_causal_load_balanced: bool = False,
    context_parallel_axis: str = "",
):
    """
    Perform non-THD (non-packed) cuDNN fused attention.

    This function implements the following formula:
        BMM1 -> (PreBias) -> ScaleMaskSoftmax -> (PostBias) -> (Dropout) -> BMM2
    Args:
        qkv (Tuple[jnp.ndarray, ...]): A tuple containing query, key, and value tensors.
        It supports three formats:
            - `(qkv_packed,)`: For interleaved QKV packed format, typically used when query, key,
              and value have the same shape (e.g., self-attention).
            - `(query, kv_packed)`: For separate query and KV packed format, typically used when
              query has a different shape (e.g., cross-attention).
            - `(query, key, value)`: For separate query, key, and value tensors.
        bias (Optional[jnp.ndarray]): An optional bias tensor to be added to the attention scores.
        mask (Optional[jnp.ndarray]):
            An optional mask tensor to mask out the attention scores, `True` means mask out.
            Intra-sequence padding is not valid. The padded tokens can only on the right-most.
            Otherwise the results will be wrong.
        seed (Optional[jnp.ndarray]): Optional random seed for dropout.
        attn_bias_type (NVTE_Bias_Type): Type of attention bias.
        attn_mask_type (NVTE_Mask_Type): Type of attention mask.
        qkv_layout (NVTE_QKV_Layout): Layout of the QKV tensors.
        scaling_factor (float): Scaling factor for the attention scores.
        dropout_probability (float): Dropout probability to apply during attention.
        is_training (bool): Flag indicating whether the model is in training mode.
        window_size (Optional[Tuple[int, int]]): Sliding window size.
        context_parallel_causal_load_balanced (bool):
            Indicates the sequences are ordered for causal mask load balancing when running context parallelism.
        context_parallel_axis (str): The name of the context parallel axis.
    Returns:
        (jnp.ndarray): The output tensor from the fused attention.
    """
    assert (
        not qkv_layout.is_thd()
    ), "Please use transformer_engine.jax.attention.fused_attn_thd for THD format."

    # Check inputs qkv
    match qkv_layout:
        case NVTE_QKV_Layout.NVTE_BS3HD:
            assert len(qkv) == 1, f"qkv=(packed_qkv,) is expected with {qkv_layout=} but got {qkv=}"
        case NVTE_QKV_Layout.NVTE_BSHD_BS2HD:
            assert (
                len(qkv) == 2
            ), f"qkv=(query, packed_kv) is expected with {qkv_layout=} but got {qkv=}"
        case NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD:
            assert (
                len(qkv) == 3
            ), f"qkv=(query, key, value) is expected with {qkv_layout=} but got {qkv=}"

    # convert the mask to seqlens, mask doesn't support ragged offsets
    if not attn_mask_type.is_padding():
        batch, q_max_seqlen, kv_max_seqlen = _obtain_batch_and_max_seqlen(qkv, qkv_layout)
        q_seq_lens = jnp.full((batch,), q_max_seqlen, dtype=jnp.int32)
        kv_seq_lens = jnp.full((batch,), kv_max_seqlen, dtype=jnp.int32)
    else:
        assert mask is not None
        mask = jnp.logical_not(mask)
        q_seq_lens = jnp.sum(mask, axis=-2, dtype=jnp.int32)[..., 0, 0]
        if attn_mask_type == AttnMaskType.PADDING_MASK:
            kv_seq_lens = jnp.sum(mask, axis=-1, dtype=jnp.int32)[..., 0, 0]
        else:
            # When mask is causal, the actual seqlen is not the last row, use max to find it
            kv_seq_lens = jnp.max(jnp.sum(mask, axis=-1, dtype=jnp.int32), axis=(-1, -2))

    output = _fused_attn(
        qkv,
        bias,
        SequenceDescriptor.from_seqlens((q_seq_lens, kv_seq_lens)),
        seed,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=qkv_layout,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
        max_segments_per_seq=1,
        window_size=window_size,
        context_parallel_strategy=context_parallel_strategy,
        context_parallel_causal_load_balanced=context_parallel_causal_load_balanced,
        context_parallel_axis=context_parallel_axis,
    )

    return output


def fused_attn_thd(
    qkv: Tuple[jnp.ndarray, ...],
    bias: Optional[jnp.ndarray],
    q_seq_lens: jnp.ndarray,
    kv_seq_lens: jnp.ndarray,
    q_seq_offsets: jnp.ndarray,
    kv_seq_offsets: jnp.ndarray,
    seed: Optional[jnp.ndarray],
    attn_bias_type: AttnBiasType,
    attn_mask_type: AttnMaskType,
    qkv_layout: QKVLayout,
    scaling_factor: float,
    dropout_probability: float,
    is_training: bool,
    max_segments_per_seq: int = 1,
    window_size: Optional[Tuple[int, int]] = None,
    context_parallel_strategy: CPStrategy = CPStrategy.DEFAULT,
    context_parallel_causal_load_balanced: bool = False,
    context_parallel_axis: str = "",
):
    """
    Deprecated THD fused attn, please use fusd_attn with SequenceDescriptor
    """
    warnings.warn(
        "fused_attn_thd is deprecated, please use fused_attn with SequenceDescriptor",
        DeprecationWarning,
    )

    assert (
        qkv_layout.is_thd()
    ), "Please use transformer_engine.jax.attention.fused_attn for non-THD format."

    # Check inputs qkv
    match qkv_layout:
        case NVTE_QKV_Layout.NVTE_T3HD:
            assert len(qkv) == 1, f"qkv=(packed_qkv,) is expected with {qkv_layout=} but got {qkv=}"
        case NVTE_QKV_Layout.NVTE_THD_T2HD:
            assert (
                len(qkv) == 2
            ), f"qkv=(query, packed_kv) is expected with {qkv_layout=} but got {qkv=}"
        case NVTE_QKV_Layout.NVTE_THD_THD_THD:
            assert (
                len(qkv) == 3
            ), f"qkv=(query, key, value) is expected with {qkv_layout=} but got {qkv=}"

    batch, q_max_seqlen, kv_max_seqlen = _obtain_batch_and_max_seqlen(qkv, qkv_layout)
    assert q_seq_lens.shape == (batch, q_max_seqlen)
    assert kv_seq_lens.shape == (batch, kv_max_seqlen)
    assert q_seq_offsets.shape == (batch, q_max_seqlen + 1)
    assert kv_seq_offsets.shape == (batch, kv_max_seqlen + 1)

    output = _fused_attn(
        qkv,
        bias,
        SequenceDescriptor.from_seqlens_and_offsets(
            (q_seq_lens, kv_seq_lens), (q_seq_offsets, kv_seq_offsets)
        ),
        seed,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=qkv_layout,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
        max_segments_per_seq=max_segments_per_seq,
        window_size=window_size,
        context_parallel_strategy=context_parallel_strategy,
        context_parallel_causal_load_balanced=context_parallel_causal_load_balanced,
        context_parallel_axis=context_parallel_axis,
    )

    return output


@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))
def _fused_attn(
    qkv: Tuple[jnp.ndarray, ...],
    bias: Optional[jnp.ndarray],
    sequence_descriptor: SequenceDescriptor,
    seed: Optional[jnp.ndarray],
    attn_bias_type: AttnBiasType,
    attn_mask_type: AttnMaskType,
    qkv_layout: QKVLayout,
    scaling_factor: float,
    dropout_probability: float,
    is_training: bool,
    max_segments_per_seq: int,
    window_size: Optional[Tuple[int, int]],
    context_parallel_strategy: CPStrategy,
    context_parallel_causal_load_balanced: bool,
    context_parallel_axis: str,
):
    output, _ = _fused_attn_fwd_rule(
        qkv,
        bias,
        sequence_descriptor,
        seed,
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
        max_segments_per_seq,
        window_size,
        context_parallel_strategy,
        context_parallel_causal_load_balanced,
        context_parallel_axis,
    )
    return output


def _fused_attn_fwd_rule(
    qkv,
    bias,
    sequence_descriptor,
    seed,
    attn_bias_type,
    attn_mask_type,
    qkv_layout,
    scaling_factor,
    dropout_probability,
    is_training,
    max_segments_per_seq,
    window_size,
    context_parallel_strategy,
    context_parallel_causal_load_balanced,
    context_parallel_axis,
):
    output, softmax_aux, rng_state = tex.fused_attn_fwd(
        qkv,
        bias,
        sequence_descriptor,
        seed,
        attn_bias_type=attn_bias_type.value,
        attn_mask_type=attn_mask_type.value,
        qkv_layout=qkv_layout.value,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
        max_segments_per_seq=max_segments_per_seq,
        window_size=window_size,
        context_parallel_strategy=context_parallel_strategy,
        context_parallel_causal_load_balanced=context_parallel_causal_load_balanced,
        context_parallel_axis=context_parallel_axis,
    )
    output = checkpoint_name(output, "context")
    softmax_aux = checkpoint_name(softmax_aux, "context")
    rng_state = checkpoint_name(rng_state, "context")
    return output, (
        qkv,
        bias,
        sequence_descriptor,
        softmax_aux,
        rng_state,
        output,
    )


def _fused_attn_bwd_rule(
    attn_bias_type,
    attn_mask_type,
    qkv_layout,
    scaling_factor,
    dropout_probability,
    is_training,
    max_segments_per_seq,
    window_size,
    context_parallel_strategy,
    context_parallel_causal_load_balanced,
    context_parallel_axis,
    ctx,
    dz,
):
    (
        qkv,
        bias,
        sequence_descriptor,
        softmax_aux,
        rng_state,
        output,
    ) = ctx
    grad_qkv, grad_bias = tex.fused_attn_bwd(
        qkv,
        bias,
        softmax_aux,
        rng_state,
        output,
        dz,
        sequence_descriptor,
        attn_bias_type=attn_bias_type.value,
        attn_mask_type=attn_mask_type.value,
        qkv_layout=qkv_layout.value,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
        max_segments_per_seq=max_segments_per_seq,
        window_size=window_size,
        context_parallel_strategy=context_parallel_strategy,
        context_parallel_causal_load_balanced=context_parallel_causal_load_balanced,
        context_parallel_axis=context_parallel_axis,
    )
    if attn_bias_type == AttnBiasType.NO_BIAS:
        grad_bias = None
    return (
        grad_qkv,
        grad_bias,
        None,
        None,
    )


_fused_attn.defvjp(_fused_attn_fwd_rule, _fused_attn_bwd_rule)


def fused_attn(
    qkv: Tuple[jnp.ndarray, ...],
    bias: Optional[jnp.ndarray],
    sequence_descriptor: SequenceDescriptor,
    seed: Optional[jnp.ndarray],
    attn_bias_type: AttnBiasType,
    attn_mask_type: AttnMaskType,
    qkv_layout: QKVLayout,
    scaling_factor: float,
    dropout_probability: float,
    is_training: bool,
    max_segments_per_seq: int = 1,
    window_size: Optional[Tuple[int, int]] = None,
    context_parallel_strategy: CPStrategy = CPStrategy.DEFAULT,
    context_parallel_causal_load_balanced: bool = False,
    context_parallel_axis: str = "",
):
    """
    Perform cuDNN fused attention.

    This function implements the following formula:
        BMM1 -> (PreBias) -> ScaleMaskSoftmax -> (PostBias) -> (Dropout) -> BMM2
    Args:
        qkv (Tuple[jnp.ndarray, ...]): A tuple containing query, key, and value tensors.
        It supports three formats:
            - `(qkv_packed,)`: For interleaved QKV packed format, typically used when query, key,
              and value have the same shape (e.g., self-attention).
            - `(query, kv_packed)`: For separate query and KV packed format, typically used when
              query has a different shape (e.g., cross-attention).
            - `(query, key, value)`: For separate query, key, and value tensors.
        bias (Optional[jnp.ndarray]): An optional bias tensor to be added to the attention scores.
        sequence_descriptor (SequenceDescriptor): Descriptor for how to describe the sequence.
        seed (Optional[jnp.ndarray]): Optional random seed for dropout.
        attn_bias_type (NVTE_Bias_Type): Type of attention bias.
        attn_mask_type (NVTE_Mask_Type): Type of attention mask.
        qkv_layout (NVTE_QKV_Layout): Layout of the QKV tensors.
        scaling_factor (float): Scaling factor for the attention scores.
        dropout_probability (float): Dropout probability to apply during attention.
        is_training (bool): Flag indicating whether the model is in training mode.
        max_segments_per_seq (int):
            Indicating the maximum number of segments inside a sequence. This parameter is to
            constrain the limit usage and need to be static during the e2e training. The XLA compile
            time and memory consumption is proportional to `max_segments_per_seq`.
        window_size (Optional[Tuple[int, int]]):
            Sliding window size.
        context_parallel_causal_load_balanced (bool):
            Indicates the sequences are ordered for causal mask load balancing when running context parallelism.
        context_parallel_axis (str): The name of the context parallel axis.
    Returns:
        (jnp.ndarray): The output tensor from the fused attention.

    Examples (non-THD, also known as non-packed):
        >>> #  q_segment_ids = [[1, 1, 1, 0], [1, 1, 0, 0]], 0 means padded tokens
        >>> # kv_segment_ids = [[1, 0, 0, 0], [1, 1, 0, 0]], 0 means padded tokens
        >>> b, s, h, d = 2, 4, 12, 64
        >>> qkv = jnp.zeros((b, s, 3, h, d), dtype=jnp.bfloat16)
        >>> q_seq_lens = jnp.asarray([3, 2])
        >>> kv_seq_lens = jnp.asarray([1, 2])
        >>> sequence_desc = SequenceDescriptor.from_seqlens(
                seqlens=(q_seq_lens, kv_seq_lens))
        >>> out = fused_attn((qkv,), None, sequence_desc, None,
                             AttnBiasType.NO_BIAS, AttnMaskType.PADDING_CAUSAL_MASK,
                             QKVLayout.BS3HD, 0.125, 0, True, 3)

    Examples (THD, also known as packed):
        >>> # segment_ids = [[1, 1, 2, 3], [1, 1, 2, 0]], 0 means padded tokens
        >>> # segment_pos = [[0, 1, 0, 0], [0, 1, 0, 1]]
        >>> b, s, h, d = 2, 4, 12, 64
        >>> qkv = jnp.zeros((b, s, 3, h, d), dtype=jnp.bfloat16)
        >>> # 3 segments in first seq, 2 segments in second seq
        >>> q_seq_lens = kv_seq_lens = jnp.asarray([[2, 1, 1, -1], [2, 1, -1, -1]])
        >>> # seq_offsets need to include the end offset of the last segments
        >>> q_seq_offsets = kv_seq_offsets = jnp.asarray([[0, 2, 3, 4, -1], [0, 2, 3, -1, -1]])
        >>> sequence_desc = SequenceDescriptor.from_seqlens_and_offsets(
                seqlens=(q_seq_lens, kv_seq_lens),
                seq_offsets=(q_seq_offsets, kv_seq_offsets))
        >>> out = fused_attn((qkv,), None, sequence_desc, None,
                             AttnBiasType.NO_BIAS, AttnMaskType.PADDING_CAUSAL_MASK,
                             QKVLayout.T3HD, 0.125, 0, True, 3)
    """
    if sequence_descriptor is None or isinstance(sequence_descriptor, jnp.ndarray):
        warnings.warn(
            "Pass mask to fused_attn is deprecated, please use SequenceDescriptor instead. "
            + "See help(transformer_engine.jax.attention.SequenceDescriptor) for details.",
            DeprecationWarning,
        )
        if max_segments_per_seq != 1:
            raise ValueError("Passing mask is only supported for non-THD case.")
        return _legacy_fused_attn(
            qkv,
            bias,
            sequence_descriptor,
            seed,
            attn_bias_type=attn_bias_type,
            attn_mask_type=attn_mask_type,
            qkv_layout=qkv_layout,
            scaling_factor=scaling_factor,
            dropout_probability=dropout_probability,
            is_training=is_training,
            window_size=window_size,
            context_parallel_strategy=context_parallel_strategy,
            context_parallel_causal_load_balanced=context_parallel_causal_load_balanced,
            context_parallel_axis=context_parallel_axis,
        )
    output = _fused_attn(
        qkv,
        bias,
        sequence_descriptor,
        seed,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=qkv_layout,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
        max_segments_per_seq=max_segments_per_seq,
        window_size=window_size,
        context_parallel_strategy=context_parallel_strategy,
        context_parallel_causal_load_balanced=context_parallel_causal_load_balanced,
        context_parallel_axis=context_parallel_axis,
    )

    return output

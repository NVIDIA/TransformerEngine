# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX multi-head attention modules"""

from enum import Enum
from functools import partial
from typing import Optional, Tuple
import warnings
from jax.ad_checkpoint import checkpoint_name
import jax
import jax.numpy as jnp

from transformer_engine.transformer_engine_jax import NVTE_Bias_Type
from transformer_engine.transformer_engine_jax import NVTE_Mask_Type
from transformer_engine.transformer_engine_jax import NVTE_QKV_Layout
from transformer_engine.transformer_engine_jax import NVTE_QKV_Format
from transformer_engine.transformer_engine_jax import nvte_get_qkv_format

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


class QKVFormat(Enum):
    """
    SBHD: q,k,v memory layout with [s, b, ..., h, d]
    BSHD: q,k,v memory layout with [b, s, ..., h, d]
    THD: q,k,v memory layout is same as BSHD, but allow multiple segments packed in a sequence.
    """

    SBHD = NVTE_QKV_Format.NVTE_SBHD
    BSHD = NVTE_QKV_Format.NVTE_BSHD
    THD = NVTE_QKV_Format.NVTE_THD


def get_qkv_format(qkv_layout):
    """
    Get qkv_format from qkv_layout
    """
    return QKVFormat(nvte_get_qkv_format(qkv_layout.value))


# Following check_set_window_size() in transformer_engine/pytorch/attention.py
def check_set_window_size(
    attn_mask_type: AttnMaskType,
    window_size: Tuple[int, int] = None,
):
    """Check if sliding window size is compliant with attention mask type.
    If not, set it to the appropriate size.

         attn_mask_type                              |   window_size
    -------------------------------------------------------------------------
    NO_MASK, PADDING_MASK                            | (-1, -1) or (>=0, >=0)
    CAUSAL_MASK                                      | (-1,  0) or (>=0, 0)
    PADDING_CAUSAL_MASK                              | (-1,  0) or (>=0, 0)
    CAUSAL_BOTTOM_RIGHT_MASK                         | (-1,  0) or (>=0, 0)
    PADDING_CAUSAL_BOTTOM_RIGHT_MASK                 | (-1,  0) or (>=0, 0)
    """
    orig_window_size = window_size
    non_causal_mask_types = {AttnMaskType.NO_MASK, AttnMaskType.PADDING_MASK}
    causal_mask_types = {
        AttnMaskType.CAUSAL_MASK,
        AttnMaskType.PADDING_CAUSAL_MASK,
        AttnMaskType.CAUSAL_BOTTOM_RIGHT_MASK,
        AttnMaskType.PADDING_CAUSAL_BOTTOM_RIGHT_MASK,
    }
    if attn_mask_type in causal_mask_types:
        if orig_window_size is None:
            window_size = (-1, 0)
        elif orig_window_size == (-1, -1) or (
            orig_window_size[0] >= 0 and orig_window_size[1] != 0
        ):
            window_size = (orig_window_size[0], 0)
            warnings.warn(
                "window_size should be (-1, 0) or (>=0, 0) for attn_mask_type="
                + attn_mask_type.name
            )
        elif orig_window_size != (-1, 0) and (orig_window_size[0] < 0 or orig_window_size[1] != 0):
            assert False, (
                "window_size should be (-1, 0) or (>=0, 0) for attn_mask_type="
                + attn_mask_type.name
            )
    elif attn_mask_type in non_causal_mask_types:
        if orig_window_size is None:
            window_size = (-1, -1)
        elif orig_window_size == (-1, 0):
            window_size = (-1, -1)
            warnings.warn(
                "window_size should be (-1, -1) or (>=0, >=0) for attn_mask_type="
                + attn_mask_type.name
            )
        elif orig_window_size != (-1, -1) and (orig_window_size[0] < 0 or orig_window_size[1] < 0):
            assert False, (
                "window_size should be (-1, -1) or (>=0, >=0) for attn_mask_type="
                + attn_mask_type.name
            )
    else:
        assert False, "Invalid attn_mask_type: " + attn_mask_type.name
    return window_size


# Following get_swa_mask() in transformer_engine/pytorch/attention.py
def get_swa_mask(
    window_size: Tuple[int, int],
    max_seqlen_q: int,
    max_seqlen_kv: int,
    attn_mask_type: AttnMaskType = AttnMaskType.NO_MASK,
):
    """
    Convert sliding window `window_size` to an equivalent "`arbitrary`" mask.
    For "`causal`" mask type, the sliding window diagonal is aligned to the top left corner,
    and for other mask types, the bottom right corner.

    Parameters
    ----------
    window_size: Tuple[int, int]
        Sliding window size for local attention, where query at position i attends to keys
        in [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q
        + window_size[1]] inclusive. Special cases (-1, -1) and (-1, 0) mean no sliding
        window and causal mask specifically. Both `causal` and `causal_bottom_right` masks
        map to `window_size = (-1, 0)` and Transformer Engine distinguishes them based on
        `attn_mask_type`.
    max_seqlen_q: int
        Maximum sequence length for queries.
    max_seqlen_kv: int
        Maximum sequence length for keys and values.
    attn_mask_type: AttnMaskType, default = AttnMaskType.NO_MASK
        Attention mask type, {AttnMaskType.NO_MASK, AttnMaskType.PADDING_MASK,
        AttnMaskType.CAUSAL_MASK, AttnMaskType.PADDING_CAUSAL_MASK,
        AttnMaskType.CAUSAL_BOTTOM_RIGHT_MASK, AttnMaskType.PADDING_CAUSAL_BOTTOM_RIGHT_MASK}

    Returns
    ----------
    swa_mask: jax.numpy.tensor
        Matrix with shape [max_seqlen_q, max_seqlen_kv]. Elements with value 1 are the positions
        that will get attention, value 0 are the masked out positions.
    """
    swa_mask = jnp.ones((max_seqlen_q, max_seqlen_kv))
    causal_mask_types = {
        AttnMaskType.CAUSAL_MASK,
        AttnMaskType.PADDING_CAUSAL_MASK,
    }
    if window_size[0] >= 0:
        if attn_mask_type in causal_mask_types:
            left = window_size[0] if window_size[0] != -1 else max_seqlen_q
            right = window_size[1] if window_size[1] != -1 else max_seqlen_q
            swa_mask = jnp.triu(swa_mask, k=-left)
            swa_mask = jnp.tril(swa_mask, k=right)
        else:
            left = window_size[0] if window_size[0] != -1 else max_seqlen_kv
            right = window_size[1] if window_size[1] != -1 else max_seqlen_kv
            swa_mask = jnp.triu(swa_mask, k=max_seqlen_kv - max_seqlen_q - left)
            swa_mask = jnp.tril(swa_mask, k=max_seqlen_kv - max_seqlen_q + right)
    return swa_mask


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
    window_size_left=-1,
    window_size_right=-1,
):
    """
    To check whether the fused attention kernel is supported
    """
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
        window_size_left,
        window_size_right,
    ).is_fused_attn_kernel_available()


def _obtain_batch_and_max_seqlen(qkv, qkv_layout):
    match qkv_layout:
        case QKVLayout.BS3HD | QKVLayout.T3HD:
            assert len(qkv) == 1, f"qkv must be (qkvpacked,) with {qkv_layout=}"
            batch, q_max_seqlen, *_ = qkv[0].shape
            kv_max_seqlen = q_max_seqlen
        case QKVLayout.BSHD_BS2HD | QKVLayout.THD_T2HD:
            assert len(qkv) == 2, f"qkv must be (query, kvpacked) with {qkv_layout=}"
            batch, q_max_seqlen, *_ = qkv[0].shape
            kv_max_seqlen = qkv[1].shape[1]
        case QKVLayout.BSHD_BSHD_BSHD | QKVLayout.THD_THD_THD:
            assert len(qkv) == 3, f"qkv must be (query, key, value) with {qkv_layout=}"
            batch, q_max_seqlen, *_ = qkv[0].shape
            kv_max_seqlen = qkv[1].shape[1]
        case _:
            raise ValueError(f"Unsupported {qkv_layout=}")
    return batch, q_max_seqlen, kv_max_seqlen


def _reorder_causal_load_balancing(tensor, cp_size: int, tensor_format: QKVFormat, inverse: bool):
    match tensor_format:
        case QKVFormat.SBHD:
            seq_dim = 0
        case QKVFormat.BSHD:
            seq_dim = 1
        case _:
            raise ValueError(f"{tensor_format=} is not supported for causal load balancing.")

    if cp_size == 1:
        return tensor

    if cp_size % 2 != 0:
        raise ValueError(f"{cp_size=} must be a multiple of 2.")

    # Need to ensure we have 2 pairs to swap for balancing between cp ranks
    if tensor.shape[seq_dim] % (cp_size * 2) != 0:
        raise ValueError(f"{tensor.shape=} is not a multiple of {cp_size*2=}")

    # [B, S, H, D] -> [B, 2*cp_size, S/2*cp_size, D]
    # [S, B, H, D] -> [2*cp_size, S/2*cp_size, B, H, D]
    ori_tensor_shape = tensor.shape
    tensor = tensor.reshape(
        (
            *ori_tensor_shape[:seq_dim],
            2 * cp_size,
            ori_tensor_shape[seq_dim] // (2 * cp_size),
            *ori_tensor_shape[seq_dim + 1 :],
        )
    )

    parts = []
    if not inverse:
        for cp_rank in range(cp_size):
            # [B, S, H, D]: [B, 2*cp_size, S/2*cp_size, H, D] -> [B, 2, S/2*cp_size, H, D]
            # [S, B, H, D]: [2*cp_size, S/2*cp_size, B, H, D] -> [2, S/2*cp_size, B, H, D]
            index = jnp.array([cp_rank, (2 * cp_size - cp_rank - 1)])
            parts.append(jnp.take(tensor, index, axis=seq_dim))
    else:
        for cp_rank in range(cp_size // 2):
            # [B, S, H, D]: [B, 2*cp_size, S/2*cp_size, H, D] -> [B, 2, S/2*cp_size, H, D]
            # [S, B, H, D]: [2*cp_size, S/2*cp_size, B, H, D] -> [2, S/2*cp_size, B, H, D]
            base = 4 * cp_rank
            index = jnp.array([base, base + 2])
            parts.append(jnp.take(tensor, index, axis=seq_dim))
        for cp_rank in range(cp_size // 2):
            # [B, S, H, D]: [B, 2*cp_size, S/2*cp_size, H, D] -> [B, 2, S/2*cp_size, H, D]
            # [S, B, H, D]: [2*cp_size, S/2*cp_size, B, H, D] -> [2, S/2*cp_size, B, H, D]
            base = 2 * cp_size - 1 - 4 * cp_rank
            index = jnp.array([base, base - 2])
            parts.append(jnp.take(tensor, index, axis=seq_dim))

    # [B, S, H, D]: [B, 2*cp_size, S/2*cp_size, H, D]
    # [S, B, H, D]: [2*cp_size, S/2*cp_size, B, H, D]
    combined = jnp.stack(parts, axis=seq_dim)

    return combined.reshape(ori_tensor_shape)


def reorder_causal_load_balancing(tensor, cp_size: int, tensor_format: QKVFormat):
    """Reorders a tensor for load balancing the compute of causal attention."""
    return _reorder_causal_load_balancing(tensor, cp_size, tensor_format, False)


def inverse_reorder_causal_load_balancing(tensor, cp_size: int, tensor_format: QKVFormat):
    """Inverse operation of `reorder_causal_load_balancing`."""
    return _reorder_causal_load_balancing(tensor, cp_size, tensor_format, True)


def fused_attn(
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
    window_size_left: int = -1,
    window_size_right: int = -1,
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
        window_size_left (int): Sliding window size (the left half).
        window_size_right (int): Sliding window size (the right half).
        context_parallel_causal_load_balanced (bool):
            Indicates the sequences are ordered for causal mask load balancing when running context parallelism.
        context_parallel_axis (str): The name of the context parallel axis.
    Returns:
        (jnp.ndarray): The output tensor from the fused attention.
    """
    assert (
        get_qkv_format(qkv_layout) != QKVFormat.THD
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
    if attn_mask_type in [
        AttnMaskType.NO_MASK,
        AttnMaskType.CAUSAL_MASK,
        AttnMaskType.CAUSAL_BOTTOM_RIGHT_MASK,
    ]:
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
        q_seq_lens,
        kv_seq_lens,
        None,
        None,
        seed,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=qkv_layout,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
        max_segments_per_seq=1,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
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
    window_size_left: int = -1,
    window_size_right: int = -1,
    context_parallel_causal_load_balanced: bool = False,
    context_parallel_axis: str = "",
):
    """
    (Experimental) Perform THD (packed) cuDNN fused attention.

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
        q_seqlen (jnp.ndarray):
            Sequence lengths for the query, with shape [batch, max_seqlen]. Unused positions are
            padded with -1.
        kv_seqlen (jnp.ndarray):
            Sequence lengths for the key and value, with shape [batch, max_seqlen]. Unused positions
            are padded with -1.
        q_seq_offsets (jnp.ndarray):
            The offsets in the sequence dim for the query, with shape [batch, max_seqlen + 1].
            Unused positions are padded with -1.
        kv_seq_offsets (jnp.ndarray):
            The offsets in the sequence dim for the query, with shape [batch, max_seqlen + 1].
            Unused positions are padded with -1.
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
        context_parallel_causal_load_balanced (bool):
            Indicates the sequences are ordered for causal mask load balancing when running context parallelism.
        context_parallel_axis (str): The name of the context parallel axis.
    Returns:
        (jnp.ndarray): The output tensor from the fused attention.

    Examples:
        >>> # segment_ids = [[1, 1, 2, 3], [1, 1, 2, 0]], 0 means padded tokens
        >>> b, s, h, d = 2, 4, 12, 64
        >>> qkv = jnp.zeros((b, s, 3, h, d), dtype=jnp.bfloat16)
        >>> # 3 segments in first seq, 2 segments in second seq
        >>> q_seq_lens = kv_seq_lens = jnp.asarray([[2, 1, 1, -1], [2, 1, -1, -1]])
        >>> # seq_offsets need to include the end offset of the last segments
        >>> q_seq_offsets = kv_seq_offsets = jnp.asarray([[0, 2, 3, 4, -1], [0, 2, 3, -1, -1]])
        >>> out = fused_attn_thd((qkv,), None, q_seq_lens, kv_seq_lens,
                                 q_seq_offsets, kv_seq_offsets, None,
                                 AttnBiasType.NO_BIAS, AttnMaskType.PADDING_CAUSAL_MASK,
                                 QKVLayout.T3HD, 0.125, 0, True, 3)
    """
    assert (
        get_qkv_format(qkv_layout) == QKVFormat.THD
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
        q_seq_lens,
        kv_seq_lens,
        q_seq_offsets,
        kv_seq_offsets,
        seed,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=qkv_layout,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
        max_segments_per_seq=max_segments_per_seq,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        context_parallel_causal_load_balanced=context_parallel_causal_load_balanced,
        context_parallel_axis=context_parallel_axis,
    )

    return output


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17))
def _fused_attn(
    qkv: Tuple[jnp.ndarray, ...],
    bias: Optional[jnp.ndarray],
    q_seq_lens: jnp.ndarray,
    kv_seq_lens: jnp.ndarray,
    q_seq_offsets: Optional[jnp.ndarray],
    kv_seq_offsets: Optional[jnp.ndarray],
    seed: jnp.ndarray,
    attn_bias_type: AttnBiasType,
    attn_mask_type: AttnMaskType,
    qkv_layout: QKVLayout,
    scaling_factor: float,
    dropout_probability: float,
    is_training: bool,
    max_segments_per_seq: int,
    window_size_left: int,
    window_size_right: int,
    context_parallel_causal_load_balanced: bool,
    context_parallel_axis: str,
):
    output, _ = _fused_attn_fwd_rule(
        qkv,
        bias,
        q_seq_lens,
        kv_seq_lens,
        q_seq_offsets,
        kv_seq_offsets,
        seed,
        attn_bias_type,
        attn_mask_type,
        qkv_layout,
        scaling_factor,
        dropout_probability,
        is_training,
        max_segments_per_seq,
        window_size_left,
        window_size_right,
        context_parallel_causal_load_balanced,
        context_parallel_axis,
    )
    return output


def _fused_attn_fwd_rule(
    qkv,
    bias,
    q_seq_lens,
    kv_seq_lens,
    q_seq_offsets,
    kv_seq_offsets,
    seed,
    attn_bias_type,
    attn_mask_type,
    qkv_layout,
    scaling_factor,
    dropout_probability,
    is_training,
    max_segments_per_seq,
    window_size_left,
    window_size_right,
    context_parallel_causal_load_balanced,
    context_parallel_axis,
):
    output, softmax_aux, rng_state = tex.fused_attn_fwd(
        qkv,
        bias,
        q_seq_lens,
        kv_seq_lens,
        q_seq_offsets,
        kv_seq_offsets,
        seed,
        attn_bias_type=attn_bias_type.value,
        attn_mask_type=attn_mask_type.value,
        qkv_layout=qkv_layout.value,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
        max_segments_per_seq=max_segments_per_seq,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        context_parallel_causal_load_balanced=context_parallel_causal_load_balanced,
        context_parallel_axis=context_parallel_axis,
    )
    output = checkpoint_name(output, "context")
    softmax_aux = checkpoint_name(softmax_aux, "context")
    rng_state = checkpoint_name(rng_state, "context")
    return output, (
        qkv,
        bias,
        q_seq_lens,
        kv_seq_lens,
        q_seq_offsets,
        kv_seq_offsets,
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
    window_size_left,
    window_size_right,
    context_parallel_causal_load_balanced,
    context_parallel_axis,
    ctx,
    dz,
):
    (
        qkv,
        bias,
        q_seq_lens,
        kv_seq_lens,
        q_seq_offsets,
        kv_seq_offsets,
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
        q_seq_lens,
        kv_seq_lens,
        q_seq_offsets,
        kv_seq_offsets,
        attn_bias_type=attn_bias_type.value,
        attn_mask_type=attn_mask_type.value,
        qkv_layout=qkv_layout.value,
        scaling_factor=scaling_factor,
        dropout_probability=dropout_probability,
        is_training=is_training,
        max_segments_per_seq=max_segments_per_seq,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        context_parallel_causal_load_balanced=context_parallel_causal_load_balanced,
        context_parallel_axis=context_parallel_axis,
    )
    if attn_bias_type == AttnBiasType.NO_BIAS:
        grad_bias = None
    return grad_qkv, grad_bias, None, None, None, None, None


_fused_attn.defvjp(_fused_attn_fwd_rule, _fused_attn_bwd_rule)

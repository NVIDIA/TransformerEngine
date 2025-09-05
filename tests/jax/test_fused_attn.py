# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for fused attention"""
from enum import Enum, auto
from dataclasses import dataclass, field
from functools import partial
from math import sqrt
from typing import Tuple, Optional, Dict
import random

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flax.linen import combine_masks
from flax.linen import make_attention_mask
from flax.linen.dtypes import promote_dtype
from jax import Array
from jax import value_and_grad, jit
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.typing import ArrayLike, DTypeLike

from transformer_engine.jax import fp8_autocast
from transformer_engine.jax.sharding import MeshResource
from transformer_engine.jax.attention import (
    AttnBiasType,
    AttnMaskType,
    QKVLayout,
    QKVFormat,
    reorder_causal_load_balancing,
    inverse_reorder_causal_load_balancing,
    fused_attn,
    make_swa_mask,
    SequenceDescriptor,
    CPStrategy,
    ReorderStrategy,
)
from transformer_engine.jax.cpp_extensions import FusedAttnHelper
from transformer_engine_jax import (
    NVTE_Fused_Attn_Backend,
    get_cudnn_version,
    get_device_compute_capability,
)

from distributed_test_base import assert_equal_collectives
from utils import assert_allclose, print_debug_tensor_stats


@pytest.fixture(autouse=True, scope="module")
def init():
    """
    WAR for CUDA uninitialize error
    """
    # Calling customcalls before jax may cause CUDA uninitialize error
    _ = jnp.zeros(0)
    yield


@partial(jax.jit, static_argnums=(5, 6, 7, 9))
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
    context_shape = query.shape[:-1] + (value.shape[-1],)
    context = jnp.reshape(context, context_shape)
    return context


@jax.jit
def make_causal_mask(
    segment_ids_q: ArrayLike,
    segment_ids_kv: ArrayLike,
    segment_pos_q: ArrayLike = None,
    segment_pos_kv: ArrayLike = None,
) -> Array:
    """
    Create inverse padded causal mask where `True` means allowing the corresponding
    position to participate in attention and `False` means masking out that position.
    If segment_pos is not provided, aragne of the segment_ids will be applied.
    """
    if segment_pos_q is None:
        segment_pos_q = jnp.broadcast_to(
            jnp.arange(segment_ids_q.shape[-1], dtype=jnp.int32), segment_ids_q.shape
        )
    if segment_pos_kv is None:
        segment_pos_kv = jnp.broadcast_to(
            jnp.arange(segment_ids_kv.shape[-1], dtype=jnp.int32), segment_ids_kv.shape
        )
    inv_causal_mask = make_attention_mask(segment_pos_q, segment_pos_kv, jnp.greater_equal)
    return inv_causal_mask


@partial(jax.jit, static_argnums=(4, 5))
def make_mask(
    segment_ids_q: ArrayLike,
    segment_ids_kv: ArrayLike,
    segment_pos_q: ArrayLike,
    segment_pos_kv: ArrayLike,
    attn_mask_type: AttnMaskType,
    window_size: Optional[Tuple[int, int]] = None,
) -> Array:
    """
    Create attention mask based on mask type. A `True` value in the mask means
    masking out the corresponding position and a `False` value means allowing
    that position to participate in attention.

    - segment_ids should start with 1, and using 0s for the paddings.
      Expected that each segment starts without paddings.
    - segment_pos marks the token position in the segments.

    A example pair of segments_ids and segment_pos:
    segment_ids: [1, 1, 1, 0, 2, 2, 2, 3, 3, 3, 4, 0, 0, 5, 5, 5]
    segment_pos: [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    """
    # segment masks
    inv_mask = make_attention_mask(
        segment_ids_q, segment_ids_kv, lambda x, y: (jnp.logical_and(jnp.equal(x, y), x != 0))
    )

    if segment_pos_q is None:
        segment_pos_q = jnp.broadcast_to(
            jnp.arange(segment_ids_q.shape[-1], dtype=jnp.int32), segment_ids_q.shape
        )
    if segment_pos_kv is None:
        segment_pos_kv = jnp.broadcast_to(
            jnp.arange(segment_ids_kv.shape[-1], dtype=jnp.int32), segment_ids_kv.shape
        )

    # causal mask
    if attn_mask_type.is_causal():
        inv_causal_mask = make_attention_mask(
            segment_pos_q, segment_pos_kv, lambda x, y: jnp.greater_equal(x, y)
        )
        inv_mask = combine_masks(inv_causal_mask, inv_mask)

    # sliding window mask
    inv_swa_mask = make_swa_mask(segment_pos_q, segment_pos_kv, window_size, jnp.bool_)
    inv_mask = combine_masks(inv_mask, inv_swa_mask)
    mask = jnp.logical_not(inv_mask)
    return mask


@jax.jit
def get_seqlens_and_offsets(segment_ids):
    batch, max_seqlen = segment_ids.shape
    bincount_vmap = jax.vmap(partial(jnp.bincount, length=max_seqlen))
    seqlens_with_zero = bincount_vmap(segment_ids.astype(jnp.int32))
    seqlens = seqlens_with_zero[..., 1:]

    def _find_offsets(x):
        same_as_previous = jnp.logical_and(x[..., 1:] != x[..., :-1], x[..., 1:] != 0)
        first_column = x[..., :1] != 0
        same_as_previous = jnp.hstack((first_column, same_as_previous))
        return jax.vmap(partial(jnp.argwhere, size=x.shape[1], fill_value=-1))(
            same_as_previous
        ).squeeze(-1)

    offsets = _find_offsets(segment_ids)
    offsets = jnp.insert(offsets, offsets.shape[-1], values=-1, axis=-1)
    seqlens = jnp.insert(seqlens, seqlens.shape[-1], values=0, axis=-1)
    seqlens = jnp.where(seqlens, seqlens, -1)
    return seqlens, offsets


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
        bias,
        mask,
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
    sequence_descriptor,
    dropout_rng,
    **kwargs,
):
    """
    TE customcall dot product attention implementation
    """
    qkv_layout = kwargs["qkv_layout"]
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
    return fused_attn(qkv_args, bias, sequence_descriptor, dropout_rng, **kwargs).astype(
        query.dtype
    )


class BiasShape(Enum):
    """
    Enum class to represent the different bias shapes used in the fused attention.
    """

    _1HSS = "1HSS"
    _B1SS = "B1SS"
    _BHSS = "BHSS"
    _11SS = "11SS"


class SeqDescFormat(Enum):
    Mask = auto()
    Seqlens = auto()
    SegmentIDs = auto()


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
    head_dim_qk: int
    head_dim_v: int
    attn_bias_type: AttnBiasType
    attn_mask_type: AttnMaskType
    dropout_prob: float
    dtype: DTypeLike
    is_training: bool
    qkv_layout: QKVLayout
    bias_shape: BiasShape
    window_size: Tuple[int, int]
    seq_desc_format: SeqDescFormat

    # Specifies sharding resources for distributed tests
    number_of_devices: int = 1
    mesh_shape: tuple[int, ...] = (1, 1, 1)
    mesh_axes: tuple[str, ...] = ("dp", "cp", "tp")
    mesh_resource: MeshResource = field(default_factory=partial(MeshResource, "dp", "cp", "tp"))

    # Context parallel aux arguments
    cp_strategy: CPStrategy = CPStrategy.DEFAULT
    cp_load_balanced: bool = True

    # dictionary of expected collective comm bytes
    coll_count_ref: Optional[Dict[str, int]] = None

    # See https://docs.nvidia.com/deeplearning/cudnn/latest/release-notes.html#cudnn-9-4-0 for known issue
    # generating zero-length ragged tensors. This setting adjusts the test to avoid the zero-length cases.
    def _get_max_segments_per_sequence(self):
        if self.qkv_layout.is_thd():
            if 90400 <= get_cudnn_version() < 90500:
                return self.num_segments_per_seq
            else:
                # +1 for testing runtime_segments < max_segments
                return self.num_segments_per_seq + 1
        else:
            return 1

    def _check_configs(self):
        # TODO(rewang): probably adds this in is_fused_attn_available
        if self.qkv_layout.is_thd() and not self.attn_mask_type.is_padding():
            pytest.skip("THD format requires padding masks.")

        if self.qkv_layout.is_qkvpacked():
            if self.max_seqlen_q != self.max_seqlen_kv:
                pytest.skip(f"{self.qkv_layout} requires max_seqlen_q == max_seqlen_kv")
            if self.num_heads_q != self.num_heads_kv:
                pytest.skip(f"{self.qkv_layout} requires num_heads_q == num_heads_kv")

        if self.max_seqlen_q > self.max_seqlen_kv and self.window_size is not None:
            pytest.skip(
                "seqlen_q > seqlen_kv is not supported with sliding window attention in cuDNN"
            )

        if (
            get_device_compute_capability(0) == 100
            and self.dropout_prob == 0.1
            and self.attn_bias_type is not AttnBiasType.NO_BIAS
        ):
            pytest.skip(
                "For sm100, bprop kernel support for dropout + determinism (bias) is not supported"
            )
        # Test the MLA case where head dims for qk differ from head dims for v, only if the tensors
        # are provided in BSHD_BSHD_BSHD or THD_THD_THD formats
        if self.head_dim_qk != self.head_dim_v and not self.qkv_layout.is_separate():
            pytest.skip(
                "For head_dim_qk != head_dim_v, it is necessary that the QKV layout "
                "is either BSHD_BSHD_BSHD or THD_THD_THD"
            )

        self.backend = FusedAttnHelper(
            self.is_training,
            self.dtype,
            self.dtype,
            self.qkv_layout,
            self.attn_bias_type,
            self.attn_mask_type,
            self.dropout_prob,
            self.num_heads_q,
            self.num_heads_kv,
            self.max_seqlen_q,
            self.max_seqlen_kv,
            self.head_dim_qk,
            self.head_dim_v,
            (-1, -1) if self.window_size is None else self.window_size,
        ).get_fused_attn_backend()
        if self.backend != NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen:
            pytest.skip("Unsupported inputs combination or device compute capability.")

        if (
            self.attn_bias_type == AttnBiasType.POST_SCALE_BIAS
            and self.bias_shape != BiasShape._1HSS
        ):
            if self.attn_mask_type.is_padding():
                pytest.skip(
                    "B1SS, BHSS and 11SS bias shapes are only supported for non-padding mask"
                )
            elif self.backend != NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen:
                pytest.skip(
                    "B1SS, BHSS and 11SS bias shapes are only supported for "
                    "the F16_arbitrary_seqlen backend."
                )

    def _setup_inputs(self):
        self._check_configs()

        # Create a mesh for distributed tests
        self.devices = np.asarray(jax.devices()[: self.number_of_devices]).reshape(*self.mesh_shape)
        self.mesh = Mesh(self.devices, self.mesh_axes)
        self.dp_size = self.mesh.shape.get(self.mesh_resource.dp_resource, 1)
        self.cp_size = self.mesh.shape.get(self.mesh_resource.cp_resource, 1)
        self.tp_size = self.mesh.shape.get(self.mesh_resource.tpsp_resource, 1)

        key = jax.random.PRNGKey(0)
        q_key, k_key, v_key, bias_key, dropout_key = jax.random.split(key, 5)

        q_shape = (self.batch_size, self.max_seqlen_q, self.num_heads_q, self.head_dim_qk)
        k_shape = (self.batch_size, self.max_seqlen_kv, self.num_heads_kv, self.head_dim_qk)
        v_shape = (self.batch_size, self.max_seqlen_kv, self.num_heads_kv, self.head_dim_v)

        if self.attn_bias_type == AttnBiasType.NO_BIAS:
            bias_shape = None
        elif self.bias_shape == BiasShape._1HSS:
            bias_shape = (1, self.num_heads_q, self.max_seqlen_q, self.max_seqlen_kv)
        elif self.bias_shape == BiasShape._B1SS:
            bias_shape = (self.batch_size, 1, self.max_seqlen_q, self.max_seqlen_kv)
        elif self.bias_shape == BiasShape._BHSS:
            bias_shape = (
                self.batch_size,
                self.num_heads_q,
                self.max_seqlen_q,
                self.max_seqlen_kv,
            )
        elif self.bias_shape == BiasShape._11SS:
            bias_shape = (1, 1, self.max_seqlen_q, self.max_seqlen_kv)
        else:
            pytest.fail(f"PyTest attempted to use an unrecognized bias_layout = {self.bias_shape}!")

        self.q = jax.random.uniform(q_key, q_shape, self.dtype, -1.0)
        self.k = jax.random.uniform(k_key, k_shape, self.dtype, -1.0)
        self.v = jax.random.uniform(v_key, v_shape, self.dtype, -1.0)

        if self.attn_bias_type != AttnBiasType.NO_BIAS:
            if self.bias_shape == BiasShape._1HSS:
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

        if self.attn_mask_type.is_padding():
            pad_ratio = 0.3
        else:
            pad_ratio = 0.0

        def gen_valid(bs, max_seqlen, pad_ratio):
            pad_len = int(max_seqlen * pad_ratio)
            valid_len = max_seqlen - pad_len
            tokens = jnp.concatenate([jnp.ones((bs, valid_len)), jnp.zeros((bs, pad_len))], axis=-1)
            return tokens, jnp.logical_not(tokens)

        def generate_random_segment_ids(
            batch_size,
            sequence_length,
            num_segments,
            seed,
            with_segment_pad=True,
            min_segment_len=None,
        ):
            rng = np.random.default_rng(seed=seed)
            # [1, 1, 1, 2, 2, 3, 3, 3, 3, 0, 0], 0 means pad
            segment_ids = np.zeros((batch_size, sequence_length), dtype=np.int32)
            segment_pos = np.zeros((batch_size, sequence_length), dtype=np.int32)
            # [0, 1, 2, 0, 1, 0, 1, 2, 3, 0, 0]
            # [0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1], 1 means pad
            segment_pad = np.zeros((batch_size, sequence_length), dtype=np.int32)

            # Not include paddings
            max_segment_size = sequence_length // num_segments
            for i in range(batch_size):
                current_pos = 0
                segment_id = 1

                for seg_id in range(num_segments):
                    # min_segment_len is to force kv_len >= q_len because cuDNN kernels failed
                    # TODO(rewang): Remove this constrain after cuDNN supports
                    min_segment_size = 1
                    if min_segment_len is not None:
                        min_segment_size = min_segment_len[i][seg_id]
                    segment_size = rng.integers(min_segment_size, max_segment_size + 1)
                    if current_pos + segment_size > sequence_length:
                        break
                    segment_end = current_pos + segment_size
                    segment_ids[i, current_pos:segment_end] = segment_id
                    segment_pos[i, current_pos:segment_end] = np.arange(segment_size)
                    if with_segment_pad:
                        num_valid = rng.integers(min_segment_size, segment_size + 1)
                        segment_pad[i, current_pos + num_valid : segment_end] = 1
                    current_pos = segment_end
                    segment_id += 1
                segment_pad[i, current_pos:sequence_length] = 1

            segment_ids, segment_pos, segment_pad = map(
                jnp.asarray, [segment_ids, segment_pos, segment_pad]
            )
            segment_ids = jnp.where(segment_pad, 0, segment_ids)
            return segment_ids, segment_pos, segment_pad

        if self.qkv_layout.is_thd():
            self.num_segments_per_seq = 2
            self.segment_ids_q, self.segment_pos_q, self.pad_q = generate_random_segment_ids(
                self.batch_size, self.max_seqlen_q, self.num_segments_per_seq, seed=42
            )
            self.seqlens_q, self.offsets_q = get_seqlens_and_offsets(self.segment_ids_q)
            # TODO(rewang): record only self attention and find the reason of cross attention
            if self.qkv_layout == QKVLayout.T3HD or self.max_seqlen_q == self.max_seqlen_kv:
                self.segment_ids_kv = self.segment_ids_q
                self.segment_pos_kv = self.segment_pos_q
                self.pad_kv = self.pad_q
            else:
                # Force kv_len >= q_len for swa, otherwise, cuDNN kernels don't support
                min_segment_len = None if self.window_size is None else self.seqlens_q
                self.segment_ids_kv, self.segment_pos_kv, self.pad_kv = generate_random_segment_ids(
                    self.batch_size,
                    self.max_seqlen_kv,
                    self.num_segments_per_seq,
                    seed=2024,
                    min_segment_len=min_segment_len,
                )
            self.seqlens_kv, self.offsets_kv = get_seqlens_and_offsets(self.segment_ids_kv)
        else:
            self.num_segments_per_seq = 1
            self.segment_ids_q, self.pad_q = gen_valid(
                self.batch_size, self.max_seqlen_q, pad_ratio
            )
            self.segment_ids_kv, self.pad_kv = gen_valid(
                self.batch_size, self.max_seqlen_kv, pad_ratio
            )
            self.segment_pos_q = self.segment_pos_kv = None
            self.seqlens_q = self.seqlens_kv = self.offsets_q = self.offsets_kv = None

        # For reference code
        self.mask = make_mask(
            self.segment_ids_q,
            self.segment_ids_kv,
            self.segment_pos_q,
            self.segment_pos_kv,
            self.attn_mask_type,
            self.window_size,
        )

        if self.cp_size > 1 and self.cp_load_balanced:
            if self.qkv_layout.is_thd():
                reorder_strategy = ReorderStrategy.Striped
            else:
                reorder_strategy = ReorderStrategy.DualChunkSwap

            seq_dim = 0 if self.qkv_layout.get_qkv_format() == QKVFormat.SBHD else 1
            self.cp_reorder_fn = partial(
                reorder_causal_load_balancing,
                strategy=reorder_strategy,
                cp_size=self.cp_size,
                seq_dim=seq_dim,
            )
            self.cp_inverse_reorder_fn = partial(
                inverse_reorder_causal_load_balancing,
                strategy=reorder_strategy,
                cp_size=self.cp_size,
                seq_dim=seq_dim,
            )
        else:
            # no-ops for non cp or non load balanced
            self.cp_reorder_fn = lambda x: x
            self.cp_inverse_reorder_fn = lambda x: x

        # Test different input formats
        if self.qkv_layout.is_thd():
            match self.seq_desc_format:
                case SeqDescFormat.Mask:
                    pytest.skip("THD doesn't support mask input")
                case SeqDescFormat.Seqlens:
                    self.sequence_desciptor = SequenceDescriptor.from_seqlens_and_offsets(
                        (self.seqlens_q, self.seqlens_kv),
                        (self.offsets_q, self.offsets_kv),
                    )
                case SeqDescFormat.SegmentIDs:
                    self.sequence_desciptor = SequenceDescriptor.from_segment_ids_and_pos(
                        (
                            self.cp_reorder_fn(self.segment_ids_q),
                            self.cp_reorder_fn(self.segment_ids_kv),
                        ),
                        (
                            self.cp_reorder_fn(self.segment_pos_q),
                            self.cp_reorder_fn(self.segment_pos_kv),
                        ),
                    )
                case _:
                    raise ValueError(f"Unknown {self.seq_desc_format=}")
        else:
            match self.seq_desc_format:
                case SeqDescFormat.Mask:
                    if self.attn_mask_type == AttnMaskType.NO_MASK:
                        self.sequence_desciptor = None
                    else:
                        self.sequence_desciptor = make_mask(
                            self.segment_ids_q,
                            self.segment_ids_kv,
                            self.segment_pos_q,
                            self.segment_pos_kv,
                            self.attn_mask_type,
                        )
                case SeqDescFormat.Seqlens:
                    self.sequence_desciptor = SequenceDescriptor.from_seqlens(
                        (
                            self.segment_ids_q.sum(axis=-1).astype(jnp.int32),
                            self.segment_ids_kv.sum(axis=-1).astype(jnp.int32),
                        ),
                    )
                case SeqDescFormat.SegmentIDs:
                    self.sequence_desciptor = SequenceDescriptor.from_segment_ids_and_pos(
                        (self.segment_ids_q, self.segment_ids_kv),
                        None,
                    )
                case _:
                    raise ValueError(f"Unknown {self.seq_desc_format=}")

        self.dropout_rng = dropout_key if self.dropout_prob > 0 else None
        self.scaling_factor = 1.0 / sqrt(self.head_dim_qk)

        # Setup distributed sharding specs
        # Setup shardings for distributed tests
        self.qkvo_psec = PartitionSpec(
            self.mesh_resource.dp_resource,
            self.mesh_resource.cp_resource,
            self.mesh_resource.tpsp_resource,
            None,
        )
        self.qkvo_sharding = NamedSharding(self.mesh, self.qkvo_psec)

        mask_pspec = PartitionSpec(
            self.mesh_resource.dp_resource, None, self.mesh_resource.cp_resource, None
        )
        self.mask_sharding = NamedSharding(self.mesh, mask_pspec)

        match self.seq_desc_format:
            case SeqDescFormat.Mask:
                self.seq_desc_sharding = self.mask_sharding
            case _:

                def to_dp_shardings(x):
                    if x.ndim == 1:
                        pspec = PartitionSpec(self.mesh_resource.dp_resource)
                    else:
                        pspec = PartitionSpec(
                            self.mesh_resource.dp_resource, self.mesh_resource.cp_resource
                        )
                    return NamedSharding(self.mesh, pspec)

                self.seq_desc_sharding = jax.tree.map(to_dp_shardings, self.sequence_desciptor)

        if self.bias_shape == BiasShape._1HSS:
            self.bias_pspec = PartitionSpec(
                None, self.mesh_resource.tpsp_resource, self.mesh_resource.cp_resource, None
            )
        elif self.bias_shape == BiasShape._B1SS:
            self.bias_pspec = PartitionSpec(
                self.mesh_resource.dp_resource, None, self.mesh_resource.cp_resource, None
            )
        elif self.bias_shape == BiasShape._11SS:
            self.bias_pspec = PartitionSpec(None, None, self.mesh_resource.cp_resource, None)
        else:
            self.bias_pspec = PartitionSpec()
        self.bias_sharding = NamedSharding(self.mesh, self.bias_pspec)

        self.dropout_rng_pspec = PartitionSpec(
            None,
        )
        self.dropout_rng_sharding = NamedSharding(self.mesh, self.dropout_rng_pspec)

        self.logit_scale_pspec = PartitionSpec(None, None, self.mesh_resource.cp_resource, None)
        self.logit_scale_sharding = NamedSharding(self.mesh, self.logit_scale_pspec)

        # [batch][max_segments_per_batch]
        # TODO(mgoldfarb-nvidia): Will need to handle CP cases of replicated or distributed length/offset.
        self.seq_length_offset_pspec = PartitionSpec(self.mesh_resource.dp_resource, None)
        self.seq_length_offset_sharding = NamedSharding(self.mesh, self.seq_length_offset_pspec)

    def test_forward(self):
        """
        Test forward without JIT
        """
        self._setup_inputs()

        args = [self.q, self.k, self.v, self.bias, self.mask, self.dropout_rng]

        customcall_args = [
            # Put test data onto each GPU for distributed.
            # TODO(mgoldfarb-nvidia): We will need to add reordering for bias, mas and
            # THD params once we support those features on CP.
            jax.device_put(self.cp_reorder_fn(self.q), self.qkvo_sharding),
            jax.device_put(self.cp_reorder_fn(self.k), self.qkvo_sharding),
            jax.device_put(self.cp_reorder_fn(self.v), self.qkvo_sharding),
            jax.device_put(self.bias, self.bias_sharding),
            jax.device_put(self.sequence_desciptor, self.seq_desc_sharding),
            jax.device_put(self.dropout_rng, self.dropout_rng_sharding),
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
            "context_parallel_strategy": self.cp_strategy,
            "context_parallel_causal_load_balanced": self.cp_load_balanced,
        }

        customcall_fused_dpa_jit = jit(
            partial(customcall_fused_dpa, **kwargs),
            static_argnames=kwargs.keys(),
            in_shardings=[
                self.qkvo_sharding,
                self.qkvo_sharding,
                self.qkvo_sharding,
                self.bias_sharding,
                self.seq_desc_sharding,
                self.dropout_rng_sharding,
            ],
        )

        with self.mesh, fp8_autocast(mesh_resource=self.mesh_resource):
            primitive_out = customcall_fused_dpa_jit(*customcall_args)
            primitive_out = self.cp_inverse_reorder_fn(primitive_out)

        reference_out = jax_dpa(*args, **kwargs)

        if self.is_training and self.dropout_prob > 0.0:
            return

        primitive_valid, primitive_invalid, reference_valid, reference_invalid = (
            _split_valid_and_invalid(primitive_out, reference_out, self.pad_q)
        )

        assert_allclose(primitive_invalid, jnp.zeros_like(primitive_invalid), dtype=self.dtype)
        assert_allclose(primitive_valid, reference_valid, dtype=self.dtype)

        if self.coll_count_ref is not None:
            with self.mesh, fp8_autocast(mesh_resource=self.mesh_resource):
                target_hlo = (
                    customcall_fused_dpa_jit.lower(*customcall_args, **kwargs).compile().as_text()
                )
            assert_equal_collectives(target_hlo, self.coll_count_ref)

    def test_backward(self):
        """
        Test value_and_grad with JIT, which includes both forward and backward.

        If coll_count_ref is not None then the HLO of the backwrds function
        HLO will be examined for the expected comms.
        """

        self._setup_inputs()

        def grad_func(func, *args, cp_reverse_out=False, **kwargs):
            # Gradient is small, use a gradient multiplier to amplify the gradient
            gradient_multiplier = self.max_seqlen_q * self.num_heads_q
            if self.attn_mask_type.is_causal():
                gradient_multiplier /= 10
            # Keep only valid result for the gradient
            if not cp_reverse_out:
                ret_valid = jnp.where(
                    self.pad_q[..., jnp.newaxis, jnp.newaxis],
                    0,
                    func(*args, **kwargs),
                )
            else:
                ret_valid = jnp.where(
                    self.pad_q[..., jnp.newaxis, jnp.newaxis],
                    0,
                    self.cp_inverse_reorder_fn(func(*args, **kwargs)),
                )
            return (
                jnp.mean(ret_valid.astype(jnp.float32), dtype=jnp.float32) * gradient_multiplier
            ).astype(self.dtype)

        args = [self.q, self.k, self.v, self.bias, self.mask, self.dropout_rng]
        customcall_args = [
            # TODO(mgoldfarb-nvidia): We will need to add reordering for bias, mas and
            # THD params once we support those features on CP.
            jax.device_put(self.cp_reorder_fn(self.q), self.qkvo_sharding),
            jax.device_put(self.cp_reorder_fn(self.k), self.qkvo_sharding),
            jax.device_put(self.cp_reorder_fn(self.v), self.qkvo_sharding),
            jax.device_put(self.bias, self.bias_sharding),
            jax.device_put(self.sequence_desciptor, self.seq_desc_sharding),
            jax.device_put(self.dropout_rng, self.dropout_rng_sharding),
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
            "context_parallel_strategy": self.cp_strategy,
            "context_parallel_causal_load_balanced": self.cp_load_balanced,
        }

        # We can compute dBias only for the [1, h, s, s] layout
        if self.bias_shape == BiasShape._1HSS:
            arg_nums = (0, 1, 2, 3)
            grad_shardings = (
                self.qkvo_sharding,
                self.qkvo_sharding,
                self.qkvo_sharding,
                self.bias_sharding,
            )
        else:
            arg_nums = (0, 1, 2)
            grad_shardings = (self.qkvo_sharding, self.qkvo_sharding, self.qkvo_sharding)

        # Use FP16/BF16 to sum the results may cause overflow, use FP32 for the summation
        jitted_primitive = jit(
            value_and_grad(
                lambda q, k, v, bias, *args: grad_func(
                    customcall_fused_dpa, q, k, v, bias, *args, cp_reverse_out=True, **kwargs
                ),
                arg_nums,
            ),
            in_shardings=(
                self.qkvo_sharding,
                self.qkvo_sharding,
                self.qkvo_sharding,
                self.bias_sharding,
                self.seq_desc_sharding,
                self.dropout_rng_sharding,
            ),
            out_shardings=(None, grad_shardings),
        )
        jitted_reference = jit(
            value_and_grad(
                lambda q, k, v, bias, *args: grad_func(jax_dpa, q, k, v, bias, *args, **kwargs),
                arg_nums,
            )
        )

        with self.mesh, fp8_autocast(mesh_resource=self.mesh_resource):
            primitive_out, primitive_dgrad = jitted_primitive(*customcall_args)

        reference_out, reference_dgrad = jitted_reference(*args)

        # Skip elementwise comparison when dropout enabled
        if self.dropout_prob > 0.0:
            return

        print_debug_tensor_stats(f"primitive_out", primitive_out)
        print_debug_tensor_stats(f"reference_grad_valid", reference_out)
        print_debug_tensor_stats(f"diff_grad", jnp.abs(primitive_out - reference_out))
        assert_allclose(primitive_out, reference_out, dtype=self.dtype)

        def check_dqkv(primitive, reference, pad, idx):
            primitive_valid, primitive_invalid, reference_valid, reference_invalid = (
                _split_valid_and_invalid(primitive, reference, pad)
            )

            print_debug_tensor_stats(f"primitive_grad_valid[{idx}]", primitive_valid[idx])
            print_debug_tensor_stats(f"reference_grad_valid[{idx}]", reference_valid[idx])
            print_debug_tensor_stats(
                f"diff_grad[{idx}]", jnp.abs(primitive_valid[idx] - reference_valid[idx])
            )

            assert_allclose(primitive_invalid, jnp.zeros_like(primitive_invalid), dtype=self.dtype)
            assert_allclose(primitive_invalid, reference_invalid, dtype=self.dtype)
            assert_allclose(primitive_valid, reference_valid, dtype=self.dtype)

        primitive_dq, primitive_dk, primitive_dv = primitive_dgrad[:3]
        reference_dq, reference_dk, reference_dv = reference_dgrad[:3]

        primitive_dq = self.cp_inverse_reorder_fn(primitive_dq)
        primitive_dk = self.cp_inverse_reorder_fn(primitive_dk)
        primitive_dv = self.cp_inverse_reorder_fn(primitive_dv)

        check_dqkv(primitive_dq, reference_dq, self.pad_q, 0)
        check_dqkv(primitive_dk, reference_dk, self.pad_kv, 1)
        check_dqkv(primitive_dv, reference_dv, self.pad_kv, 2)

        if self.attn_bias_type != AttnBiasType.NO_BIAS and self.bias_shape == BiasShape._1HSS:
            # TODO(mgoldfarb-nvidia): Inverse reorder bias once supported by a CP implementation.

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

        if self.coll_count_ref is not None:
            with self.mesh, fp8_autocast(mesh_resource=self.mesh_resource):
                target_hlo = jitted_primitive.lower(*customcall_args).compile().as_text()
            assert_equal_collectives(target_hlo, self.coll_count_ref)


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
    "b, s_q, s_kv, h_q, h_kv, d_qk, d_v, dtype",
    [
        pytest.param(
            2, 2048, 2048, 12, 12, 64, 64, jnp.bfloat16, id="2-2048-2048-12-12-64-64-BF16-SELF"
        ),
        pytest.param(
            2,
            2048,
            1024,
            12,
            12,
            64,
            64,
            jnp.bfloat16,
            id="2-2048-1024-12-12-64-64-BF16-CROSS",
        ),
        pytest.param(
            2, 2048, 2048, 12, 6, 64, 64, jnp.bfloat16, id="2-2048-2048-12-6-64-64-BF16-GQA"
        ),
        pytest.param(
            4, 128, 128, 16, 16, 64, 64, jnp.float16, id="4-128-128-16-16-64-64-FP16-SELF"
        ),
        pytest.param(
            4, 128, 128, 16, 16, 64, 32, jnp.float16, id="4-128-128-16-16-64-32-FP16-SELF"
        ),
        pytest.param(
            2,
            2048,
            1024,
            12,
            12,
            64,
            32,
            jnp.bfloat16,
            id="2-2048-1024-12-12-64-32-BF16-CROSS",
        ),
        pytest.param(
            2, 2048, 2048, 12, 6, 128, 64, jnp.float16, id="2-2048-2048-12-6-128-64-FP16-GQA"
        ),
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
@pytest.mark.parametrize(
    "seq_desc_format",
    [
        pytest.param(SeqDescFormat.Mask, id="Mask"),
        pytest.param(SeqDescFormat.Seqlens, id="Seqlens"),
        pytest.param(SeqDescFormat.SegmentIDs, id="SegmentIDs"),
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
    @pytest.mark.parametrize(
        "attn_bias_type, bias_shape",
        [
            pytest.param(AttnBiasType.NO_BIAS, None, id="NO_BIAS"),
            pytest.param(AttnBiasType.POST_SCALE_BIAS, BiasShape._1HSS, id="POST_SCALE_BIAS-1HSS"),
            pytest.param(AttnBiasType.POST_SCALE_BIAS, BiasShape._B1SS, id="POST_SCALE_BIAS-B1SS"),
            pytest.param(AttnBiasType.POST_SCALE_BIAS, BiasShape._BHSS, id="POST_SCALE_BIAS-BHSS"),
            pytest.param(AttnBiasType.POST_SCALE_BIAS, BiasShape._11SS, id="POST_SCALE_BIAS-11SS"),
        ],
    )
    def _test_forward(
        b,
        s_q,
        s_kv,
        h_q,
        h_kv,
        d_qk,
        d_v,
        attn_bias_type,
        attn_mask_type,
        dropout_prob,
        dtype,
        is_training,
        qkv_layout,
        bias_shape,
        swa,
        seq_desc_format,
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
            d_qk,
            d_v,
            attn_bias_type,
            attn_mask_type,
            dropout_prob,
            dtype,
            is_training,
            qkv_layout,
            bias_shape,
            window_size,
            seq_desc_format,
        )
        runner.test_forward()

    @staticmethod
    @pytest.mark.parametrize(
        "attn_bias_type, bias_shape",
        [
            pytest.param(AttnBiasType.NO_BIAS, None, id="NO_BIAS"),
            pytest.param(AttnBiasType.POST_SCALE_BIAS, BiasShape._1HSS, id="POST_SCALE_BIAS-1HSS"),
        ],
    )
    def test_backward(
        b,
        s_q,
        s_kv,
        h_q,
        h_kv,
        d_qk,
        d_v,
        attn_bias_type,
        attn_mask_type,
        dropout_prob,
        dtype,
        qkv_layout,
        bias_shape,
        swa,
        seq_desc_format,
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
            d_qk,
            d_v,
            attn_bias_type,
            attn_mask_type,
            dropout_prob,
            dtype,
            True,
            qkv_layout,
            bias_shape,
            window_size,
            seq_desc_format,
        )
        runner.test_backward()

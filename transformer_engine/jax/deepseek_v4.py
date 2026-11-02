# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Differentiable per-kernel JAX APIs for DeepSeek-V4 CSA and HCA.

The cuDNN Frontend package owns only the raw CuTeDSL forward/backward calls.
Transformer Engine owns automatic differentiation and the public JAX contract.
These functions intentionally do not compose a higher-level attention module.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from .attention import AttnMaskType, QKVLayout, SequenceDescriptor
from .cpp_extensions.attention import generate_cu_seqlen
from .sharding import _get_mesh


def _batch_axes_spec(batch_axes: tuple[str, ...], rank: int) -> P:
    """Return a leading-dimension shard_map spec for active batch axes."""
    mesh = _get_mesh()
    active_axes = tuple(
        axis for axis in batch_axes if axis in mesh.axis_names and mesh.shape[axis] > 1
    )
    if not active_axes:
        return P(*((None,) * rank))
    leading_axis: str | tuple[str, ...] = active_axes[0] if len(active_axes) == 1 else active_axes
    return P(leading_axis, *((None,) * (rank - 1)))


def _batched_shard_map(fn, in_specs, out_spec, batch_axes):
    """Map a batched CSA kernel locally over MaxText DP/FSDP/EP axes.

    ``shard_map`` marks these axes manual inside the body.  Thus the opaque
    CuTe calls receive local packed tensors rather than inducing an all-gather
    to recover the global batch.
    """
    mesh = _get_mesh()
    if (
        mesh is None
        or mesh.empty
        or not any(axis in mesh.axis_names and mesh.shape[axis] > 1 for axis in batch_axes)
    ):
        return fn
    return shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_spec, check_rep=False)


def _cudnn_compressor_forward(
    kv, score, ape, cu_seqlens, cu_seqlens_comp, *, total_comp, ratio, coff
):
    from cudnn import csa_compressor_forward_jax_sm100

    return csa_compressor_forward_jax_sm100(
        kv,
        score,
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        total_comp=total_comp,
        ratio=ratio,
        coff=coff,
    )


def _cudnn_compressor_backward(
    kv, score, ape, cu_seqlens, cu_seqlens_comp, grad_out, *, ratio, coff
):
    from cudnn import csa_compressor_backward_jax_sm100

    return csa_compressor_backward_jax_sm100(
        kv,
        score,
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        grad_out,
        ratio=ratio,
        coff=coff,
    )


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def _csa_compressor(kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp, ratio, coff):
    return _cudnn_compressor_forward(
        kv,
        score,
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        total_comp=total_comp,
        ratio=ratio,
        coff=coff,
    )


def _csa_compressor_fwd(kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp, ratio, coff):
    out = _cudnn_compressor_forward(
        kv,
        score,
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        total_comp=total_comp,
        ratio=ratio,
        coff=coff,
    )
    return out, (kv, score, ape, cu_seqlens, cu_seqlens_comp)


def _csa_compressor_bwd(total_comp, ratio, coff, residual, grad_out):
    del total_comp
    kv, score, ape, cu_seqlens, cu_seqlens_comp = residual
    grad_kv, grad_score, grad_ape = _cudnn_compressor_backward(
        kv,
        score,
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        grad_out.astype(jnp.bfloat16),
        ratio=ratio,
        coff=coff,
    )
    return grad_kv, grad_score, grad_ape, None, None


_csa_compressor.defvjp(_csa_compressor_fwd, _csa_compressor_bwd)


def csa_compressor(
    kv: Any,
    score: Any,
    ape: Any,
    cu_seqlens: Any,
    cu_seqlens_comp: Any,
    *,
    total_comp: int,
    ratio: int = 4,
    coff: int = 2,
) -> Any:
    """Differentiable SM100 CSA/HCA compressor kernel.

    Inputs use the raw packed kernel layout. Sequence metadata is treated as
    non-differentiable; gradients are returned for ``kv``, ``score``, and ``ape``.
    ``ratio=4, coff=2`` is DSv4 CSA; ``ratio=128, coff=1`` is DSv4 HCA.
    """
    return _csa_compressor(kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp, ratio, coff)


def csa_compressor_batched(
    kv: Any,
    score: Any,
    ape: Any,
    *,
    batch_axes: tuple[str, ...] = (),
    ratio: int = 4,
    coff: int = 2,
) -> Any:
    """Run the differentiable compressor independently on local batch shards.

    ``batch_axes`` may contain several mesh axes.  MaxText passes
    ``("data", "fsdp", "expert")`` so FSDP and EP both act as data
    parallelism for this attention-only region.

    Each sequence yields ``sequence // ratio`` blocks; a trailing partial block
    is dropped (and receives zero gradient), matching the DSv4 reference.
    """
    if kv.ndim != 3 or score.shape != kv.shape:
        raise ValueError("kv and score must have shape [batch, sequence, width]")
    if kv.shape[2] % coff:
        raise ValueError(f"compressor width {kv.shape[2]} is not divisible by coff={coff}")
    if kv.shape[1] < ratio:
        raise ValueError(f"compressor requires sequence >= ratio ({ratio})")
    batch_spec = _batch_axes_spec(batch_axes, 3)

    def body(local_kv, local_score, local_ape):
        batch, sequence, width = local_kv.shape
        blocks = sequence // ratio
        cu_seqlens = jnp.arange(batch + 1, dtype=jnp.int32) * sequence
        cu_seqlens_comp = jnp.arange(batch + 1, dtype=jnp.int32) * blocks
        out = csa_compressor(
            local_kv.reshape(batch * sequence, width),
            local_score.reshape(batch * sequence, width),
            local_ape,
            cu_seqlens,
            cu_seqlens_comp,
            total_comp=batch * blocks,
            ratio=ratio,
            coff=coff,
        )
        return out.reshape(batch, blocks, width // coff)

    return _batched_shard_map(
        body,
        (batch_spec, batch_spec, P(None, None)),
        _batch_axes_spec(batch_axes, 3),
        batch_axes,
    )(kv, score, ape)


def hca_compressor_batched(
    kv: Any,
    score: Any,
    ape: Any,
    *,
    batch_axes: tuple[str, ...] = (),
    ratio: int = 128,
) -> Any:
    """DSv4 HCA compressor: non-overlapping softmax pooling over ``ratio`` tokens.

    ``kv``/``score`` are ``[batch, sequence, head_dim]`` and ``ape`` is
    ``[ratio, head_dim]``; returns ``[batch, sequence // ratio, head_dim]``.
    """
    return csa_compressor_batched(kv, score, ape, batch_axes=batch_axes, ratio=ratio, coff=1)


def _cudnn_indexer_forward(
    q,
    k,
    weights,
    *,
    ratio,
    sm_scale,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
):
    from cudnn import indexer_forward_jax_sm100

    return indexer_forward_jax_sm100(
        q,
        k,
        weights,
        ratio=ratio,
        sm_scale=sm_scale,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )


def _indexer_reference(q, k, weights, ratio, sm_scale):
    dots = jnp.einsum("bshd,bwkd->bhsw", q.astype(jnp.float32), k.astype(jnp.float32))
    scores = jnp.einsum(
        "bhsw,bsh->bsw",
        jax.nn.relu(dots) * sm_scale,
        weights.astype(jnp.bfloat16).astype(jnp.float32),
    )
    q_positions = jnp.arange(q.shape[1])[:, None]
    block_ends = ratio * (jnp.arange(k.shape[1])[None, :] + 1)
    return jnp.where(block_ends <= q_positions + 1, scores, -jnp.inf)


def _indexer_packed_reference(q, k, w, cu_q, cu_k, ratio, scale, max_q, max_k):
    width = (max_k + 3) // 4 * 4
    qp = jnp.pad(q, ((0, max_q), (0, 0), (0, 0)))
    kp = jnp.pad(k, ((0, width), (0, 0), (0, 0)))
    wp = jnp.pad(w, ((0, max_q), (0, 0)))

    def segment(i, scores):
        qs = jax.lax.dynamic_slice_in_dim(qp, cu_q[i], max_q)
        ks = jax.lax.dynamic_slice_in_dim(kp, cu_k[i], width)
        ws = jax.lax.dynamic_slice_in_dim(wp, cu_q[i], max_q)
        # Clear invalid slices before arithmetic: output masking alone can leave
        # 0 * NaN in reverse AD when adjacent segments/capacity contain NaNs.
        q_valid = jnp.arange(max_q) < cu_q[i + 1] - cu_q[i]
        k_valid = jnp.arange(width) < cu_k[i + 1] - cu_k[i]
        qs = jnp.where(q_valid[:, None, None], qs, 0)
        ks = jnp.where(k_valid[:, None, None], ks, 0)
        ws = jnp.where(q_valid[:, None], ws, 0)
        # Reuse existing dense math + ratio-causal mask on one segment only.
        values = _indexer_reference(qs[None], ks[None], ws[None], ratio, scale)[0]
        values = jnp.where(jnp.arange(width) < cu_k[i + 1] - cu_k[i], values, -jnp.inf)
        rows = cu_q[i] + jnp.arange(max_q)
        rows = jnp.where(jnp.arange(max_q) < cu_q[i + 1] - cu_q[i], rows, q.shape[0])
        return scores.at[rows].set(values, mode="drop")

    initial = jnp.full((q.shape[0], width), -jnp.inf, jnp.float32)
    return jax.lax.fori_loop(0, cu_q.shape[0] - 1, segment, initial)


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _dsa_indexer(q, k, weights, cu_q, cu_k, ratio, sm_scale, max_q, max_k):
    return _dsa_indexer_fwd(q, k, weights, cu_q, cu_k, ratio, sm_scale, max_q, max_k)[0]


def _dsa_indexer_fwd(q, k, weights, cu_q, cu_k, ratio, sm_scale, max_q, max_k):
    out = _cudnn_indexer_forward(
        q,
        k,
        weights,
        ratio=ratio,
        sm_scale=sm_scale,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
    )
    return out, (q, k, weights, cu_q, cu_k)


def _dsa_indexer_bwd(ratio, sm_scale, max_q, max_k, residual, grad_out):
    q, k, weights, cu_q, cu_k = residual

    def reference(q_, k_, w_):
        if cu_q is None:
            return _indexer_reference(q_, k_, w_, ratio, sm_scale)
        return _indexer_packed_reference(q_, k_, w_, cu_q, cu_k, ratio, sm_scale, max_q, max_k)

    _, pullback = jax.vjp(reference, q, k, weights)
    grad_q, grad_k, grad_weights = pullback(grad_out)
    return grad_q, grad_k, grad_weights, None, None


_dsa_indexer.defvjp(_dsa_indexer_fwd, _dsa_indexer_bwd)


def dsa_indexer(
    q: Any,
    k: Any,
    weights: Any,
    *,
    ratio: int = 4,
    sm_scale: float = 1.0,
    sequence_descriptor: SequenceDescriptor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
) -> Any:
    """Differentiable SM100 indexer, using cuDNN forward and a JAX reference VJP.

    Dense inputs use BSHD/BSHD/BSH. Packed inputs use Q [Tq,64,128],
    K [Tk,1,128], weights [Tq,64], and a caller-created SequenceDescriptor
    with explicit INT32 lengths [1,M] and offsets [1,M+1]. Only one contiguous
    pack is supported: valid offsets must equal cumulative lengths, segments
    start at position zero, and unused slots are trailing -1 entries. Callers
    own these values and valid endpoints within storage capacity; holes and
    segment-ID descriptors are not supported. This is not context parallelism.

    Packed calls require positive static maximum Q/K lengths, with maxK at
    least ceil(maxQ/ratio). Returns FP32 [Tq,ceil4(maxK)], with segment-local
    key columns and -inf for invalid entries. Metadata is not differentiated.
    """
    cu_q = cu_k = None
    if q.ndim == 3:
        if not isinstance(sequence_descriptor, SequenceDescriptor):
            raise ValueError("THD requires a SequenceDescriptor with explicit lengths and offsets")
        sd = sequence_descriptor
        if any(x.size for x in (*sd.segment_ids, *sd.segment_pos)):
            raise ValueError("THD indexer requires explicit lengths/offsets, not segment IDs")
        q_len, k_len = sd.seqlens
        q_off, k_off = sd.seq_offsets
        if (
            q_len.ndim != 2
            or q_len.shape[0] != 1
            or q_len.shape[1] < 1
            or k_len.shape != q_len.shape
            or q_off.shape != (1, q_len.shape[1] + 1)
            or k_off.shape != q_off.shape
            or any(x.dtype != jnp.int32 for x in (q_len, k_len, q_off, k_off))
        ):
            raise ValueError("THD descriptor requires INT32 lengths [1,M] and offsets [1,M+1]")
        (q_len, k_len), _ = sd.get_seqlens_and_offsets(
            AttnMaskType.PADDING_CAUSAL_MASK,
            QKVLayout.THD_THD_THD,
            (-1, -1),
            q_len.shape[1],
        )
        # For contiguous packs, lengths give native prefixes directly. The
        # existing utility maps unused -1 slots to repeated terminal offsets.
        cu_q, cu_k = generate_cu_seqlen(q_len.flatten()), generate_cu_seqlen(k_len.flatten())
    elif q.ndim != 4 or any(
        x is not None for x in (sequence_descriptor, max_seqlen_q, max_seqlen_k)
    ):
        raise ValueError("dense indexer requires BSHD inputs without packed metadata")
    return _dsa_indexer(q, k, weights, cu_q, cu_k, ratio, sm_scale, max_seqlen_q, max_seqlen_k)


def dsa_indexer_batched(
    q: Any,
    k: Any,
    weights: Any,
    *,
    batch_axes: tuple[str, ...] = (),
    ratio: int = 4,
    sm_scale: float = 1.0,
) -> Any:
    """Run the differentiable indexer locally on each batch shard."""
    batch_spec_q = _batch_axes_spec(batch_axes, q.ndim)
    batch_spec_k = _batch_axes_spec(batch_axes, k.ndim)
    batch_spec_w = _batch_axes_spec(batch_axes, weights.ndim)
    return _batched_shard_map(
        lambda local_q, local_k, local_weights: dsa_indexer(
            local_q, local_k, local_weights, ratio=ratio, sm_scale=sm_scale
        ),
        (batch_spec_q, batch_spec_k, batch_spec_w),
        _batch_axes_spec(batch_axes, 3),
        batch_axes,
    )(q, k, weights)


def _cudnn_sparse_attention_forward(
    q, kv, topk_indices, topk_length, attn_sink, *, indexer_topk, softmax_scale
):
    from cudnn import sparse_attention_forward_jax_sm100

    return sparse_attention_forward_jax_sm100(
        q,
        kv,
        topk_indices,
        topk_length,
        attn_sink,
        indexer_topk=indexer_topk,
        softmax_scale=softmax_scale,
    )


def _cudnn_sparse_attention_backward(
    q,
    kv,
    out,
    grad_out,
    lse,
    attn_sink,
    topk_indices,
    topk_length,
    *,
    softmax_scale,
):
    from cudnn import sparse_attention_backward_jax_sm100

    return sparse_attention_backward_jax_sm100(
        q,
        kv,
        out,
        grad_out,
        lse,
        attn_sink,
        topk_indices,
        topk_length,
        softmax_scale=softmax_scale,
    )


@partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _dsa_sparse_attention(q, kv, topk_indices, topk_length, attn_sink, indexer_topk, softmax_scale):
    return _cudnn_sparse_attention_forward(
        q,
        kv,
        topk_indices,
        topk_length,
        attn_sink,
        indexer_topk=indexer_topk,
        softmax_scale=softmax_scale,
    )[0]


def _dsa_sparse_attention_fwd(
    q, kv, topk_indices, topk_length, attn_sink, indexer_topk, softmax_scale
):
    out, _, lse, _ = _cudnn_sparse_attention_forward(
        q,
        kv,
        topk_indices,
        topk_length,
        attn_sink,
        indexer_topk=indexer_topk,
        softmax_scale=softmax_scale,
    )
    return out, (q, kv, out, lse, attn_sink, topk_indices, topk_length)


def _dsa_sparse_attention_bwd(indexer_topk, softmax_scale, residual, grad_out):
    del indexer_topk
    q, kv, out, lse, attn_sink, topk_indices, topk_length = residual
    grad_q, grad_kv, grad_sink = _cudnn_sparse_attention_backward(
        q,
        kv,
        out,
        grad_out.astype(jnp.bfloat16),
        lse,
        attn_sink,
        topk_indices,
        topk_length,
        softmax_scale=softmax_scale,
    )
    return grad_q, grad_kv, None, None, grad_sink


_dsa_sparse_attention.defvjp(_dsa_sparse_attention_fwd, _dsa_sparse_attention_bwd)


def dsa_sparse_attention(
    q: Any,
    kv: Any,
    topk_indices: Any,
    topk_length: Any,
    attn_sink: Any,
    *,
    indexer_topk: int = 512,
    softmax_scale: float = 1.0,
) -> Any:
    """Differentiable SM100 DSv4 sparse-attention kernel.

    Gradients are returned for Q, KV, and the per-head attention sink. Sparse
    indices and lengths are non-differentiable metadata. ``indexer_topk=0``
    runs without an indexer prefix (HCA's fixed local + compressed index list).
    """
    return _dsa_sparse_attention(
        q,
        kv,
        topk_indices,
        topk_length,
        attn_sink,
        indexer_topk,
        softmax_scale,
    )


def dsa_sparse_attention_batched(
    q: Any,
    kv: Any,
    topk_indices: Any,
    topk_length: Any,
    attn_sink: Any,
    *,
    batch_axes: tuple[str, ...] = (),
    indexer_topk: int = 512,
    softmax_scale: float = 1.0,
) -> Any:
    """Run sparse attention on local batch shards using local KV indices.

    ``topk_indices`` has shape ``[batch, query_length, selected_kv]`` and
    must be relative to that batch element's KV row.  This deliberate contract
    prevents a global-batch all-gather before the raw CuTe kernel.
    """
    if q.ndim != 4 or kv.ndim != 3 or topk_indices.ndim != 3:
        raise ValueError("expected q=[B,S,H,D], kv=[B,K,D], and indices=[B,S,N]")
    if topk_length.shape != q.shape[:2]:
        raise ValueError("topk_length must have shape [batch, query_length]")
    q_spec = _batch_axes_spec(batch_axes, 4)
    kv_spec = _batch_axes_spec(batch_axes, 3)
    indices_spec = _batch_axes_spec(batch_axes, 3)
    lengths_spec = _batch_axes_spec(batch_axes, 2)

    def body(local_q, local_kv, local_indices, local_lengths, local_sink):
        batch, query_length, heads, head_dim = local_q.shape
        kv_length = local_kv.shape[1]
        # The kernel indexes the flattened [batch * kv_length] KV; shift each
        # row's valid (non-negative) indices into its batch element's KV rows.
        row_offset = (jnp.arange(batch, dtype=jnp.int32) * kv_length)[:, None, None]
        global_indices = jnp.where(local_indices >= 0, local_indices + row_offset, local_indices)
        out = dsa_sparse_attention(
            local_q.reshape(batch * query_length, heads, head_dim),
            local_kv.reshape(batch * kv_length, head_dim),
            global_indices.reshape(batch * query_length, local_indices.shape[-1]),
            local_lengths.reshape(batch * query_length),
            local_sink,
            indexer_topk=indexer_topk,
            softmax_scale=softmax_scale,
        )
        return out.reshape(batch, query_length, heads, head_dim)

    return _batched_shard_map(
        body,
        (q_spec, kv_spec, indices_spec, lengths_spec, P(None)),
        _batch_axes_spec(batch_axes, 4),
        batch_axes,
    )(q, kv, topk_indices, topk_length, attn_sink)


__all__ = [
    "csa_compressor",
    "csa_compressor_batched",
    "dsa_indexer",
    "dsa_indexer_batched",
    "dsa_sparse_attention",
    "dsa_sparse_attention_batched",
    "hca_compressor_batched",
]

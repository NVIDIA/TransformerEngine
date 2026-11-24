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
    from cudnn.ops import dsa

    return dsa.compressor_forward_jax(
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
    from cudnn.ops import dsa

    return dsa.compressor_backward_jax(
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
    return _csa_compressor_fwd(
        kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp, ratio, coff
    )[0]


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
    kv: jnp.ndarray,
    score: jnp.ndarray,
    ape: jnp.ndarray,
    cu_seqlens: jnp.ndarray,
    cu_seqlens_comp: jnp.ndarray,
    *,
    total_comp: int,
    ratio: int = 4,
    coff: int = 2,
) -> jnp.ndarray:
    """Differentiable SM100 CSA/HCA compressor kernel.

    ``kv`` and ``score`` are BF16 ``[T, coff * D]`` arrays; ``ape`` is FP32
    ``[ratio, coff * D]`` additive position bias applied to the pooling logits.
    ``ratio=4, coff=2`` selects overlapping CSA pooling; ``ratio=128, coff=1``
    selects non-overlapping HCA pooling. ``coff`` is the number of projected
    channel groups combined into each output row of width ``D``.

    ``cu_seqlens`` and ``cu_seqlens_comp`` are matching INT32 ``[M+1]``
    cumulative token/block counts starting at zero. Each sequence contributes
    ``length // ratio`` blocks; trailing partial blocks are dropped. Callers
    supply these counts and a static ``total_comp >= cu_seqlens_comp[-1]``
    allocation capacity. Returns BF16 ``[total_comp, D]``; only rows before
    ``cu_seqlens_comp[-1]`` represent valid blocks.

    Gradients are returned for ``kv``, ``score``, and ``ape``; metadata and
    configuration are not differentiated.
    """
    return _csa_compressor(kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp, ratio, coff)


def csa_compressor_batched(
    kv: jnp.ndarray,
    score: jnp.ndarray,
    ape: jnp.ndarray,
    *,
    batch_axes: tuple[str, ...] = (),
    ratio: int = 4,
    coff: int = 2,
) -> jnp.ndarray:
    """Run the differentiable compressor independently on local batch shards.

    Batched form of :func:`csa_compressor`: BF16 ``kv``/``score`` have shape
    ``[B,S,coff*D]``, FP32 ``ape`` is shared with shape ``[ratio,coff*D]``,
    and the result is BF16 ``[B,S//ratio,D]``. Token/block prefixes are built
    internally for the equal-length batch elements.

    ``batch_axes`` names mesh axes that shard the leading batch dimension;
    inactive axes are ignored and ``()`` adds no batch sharding. Position bias
    is replicated. This batches independent sequences, not context parallelism.

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
    kv: jnp.ndarray,
    score: jnp.ndarray,
    ape: jnp.ndarray,
    *,
    batch_axes: tuple[str, ...] = (),
    ratio: int = 128,
) -> jnp.ndarray:
    """DSv4 HCA compressor: non-overlapping softmax pooling over ``ratio`` tokens.

    ``kv``/``score`` are ``[batch, sequence, head_dim]`` and ``ape`` is
    ``[ratio, head_dim]``; returns ``[batch, sequence // ratio, head_dim]``.
    This is :func:`csa_compressor_batched` with ``coff=1``: the same dtypes,
    batch sharding, trailing-block behavior, and gradient contract apply.
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
    from cudnn.ops import dsa

    return dsa.indexer_forward_jax(
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


def _indexer_prefixes(q, sequence_descriptor, max_seqlen_q, max_seqlen_k):
    """Share the contiguous-pack metadata contract between scores and sparse KL."""
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
    return cu_q, cu_k


def dsa_indexer(
    q: jnp.ndarray,
    k: jnp.ndarray,
    weights: jnp.ndarray,
    *,
    ratio: int = 4,
    sm_scale: float = 1.0,
    sequence_descriptor: SequenceDescriptor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
) -> jnp.ndarray:
    """Differentiable SM100 indexer, using cuDNN forward and a JAX reference VJP.

    BF16 dense inputs are Q ``[B,Sq,64,128]``, K ``[B,Sk,1,128]``, and
    weights ``[B,Sq,64]``; returns FP32 scores ``[B,Sq,Sk]``. ``Sk`` must be
    a multiple of four. The supported ``ratio`` is 4. Scores sum the weighted
    per-head ReLU of Q/K dot products, scaled by ``sm_scale`` (default 1.0).
    Only completed compressed blocks are causally visible; invalid scores
    are ``-inf``.

    Packed BF16 inputs use Q [Tq,64,128], K [Tk,1,128], weights [Tq,64],
    and a caller-created SequenceDescriptor
    with explicit INT32 lengths [1,M] and offsets [1,M+1]. Only one contiguous
    pack is supported: valid offsets must equal cumulative lengths, segments
    start at position zero, and unused slots are trailing -1 entries. Callers
    own these values and valid endpoints within storage capacity; holes and
    segment-ID descriptors are not supported. This is not context parallelism.

    Packed calls require positive static maximum Q/K lengths, with maxK at
    least ceil(maxQ/ratio). Returns FP32 [Tq,ceil4(maxK)], with segment-local
    key columns and -inf for invalid entries; ceil4 rounds up to a multiple of
    four. Metadata is not differentiated; Q, K, and weights receive gradients.

    Returns scores, not selected indices. The caller owns top-k selection and
    the indexer training objective: discrete top-k indices do not propagate
    attention-output gradients back to the indexer scores.
    """
    cu_q, cu_k = _indexer_prefixes(q, sequence_descriptor, max_seqlen_q, max_seqlen_k)
    return _dsa_indexer(q, k, weights, cu_q, cu_k, ratio, sm_scale, max_seqlen_q, max_seqlen_k)


def dsa_indexer_batched(
    q: jnp.ndarray,
    k: jnp.ndarray,
    weights: jnp.ndarray,
    *,
    batch_axes: tuple[str, ...] = (),
    ratio: int = 4,
    sm_scale: float = 1.0,
) -> jnp.ndarray:
    """Run the dense :func:`dsa_indexer` locally on each batch shard.

    Input/output shapes, dtypes, scales, and gradients follow its dense
    contract. ``batch_axes`` follows :func:`csa_compressor_batched`; this
    wrapper does not accept packed sequence metadata.
    """
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


def _cudnn_indexer_sparse_backward(
    q,
    k,
    weights,
    target,
    probabilities,
    indices,
    grad_loss,
    sm_scale,
    cu_q,
    cu_k,
    max_q,
    max_k,
):
    from cudnn.ops import dsa

    return dsa.indexer_backward_jax(
        q,
        k,
        weights,
        target,
        probabilities,
        indices,
        grad_loss,
        sm_scale=sm_scale,
        clip_log=False,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
    )


@partial(jax.custom_vjp, nondiff_argnums=(8, 9, 10, 11))
def _indexer_sparse_kl(
    q, k, weights, scores, indices, target, cu_q, cu_k, sm_scale, eps, max_q, max_k
):
    return _indexer_sparse_kl_fwd(
        q, k, weights, scores, indices, target, cu_q, cu_k, sm_scale, eps, max_q, max_k
    )[0]


def _indexer_sparse_kl_fwd(
    q, k, weights, scores, indices, target, cu_q, cu_k, sm_scale, eps, max_q, max_k
):
    del sm_scale, max_q, max_k
    if cu_q is None:
        valid = (indices >= 0) & (indices < k.shape[1])
    else:
        rows = jnp.arange(q.shape[0], dtype=jnp.int32)
        segment = jnp.minimum(jnp.searchsorted(cu_q, rows, side="right") - 1, cu_q.size - 2)
        lengths = cu_k[segment + 1] - cu_k[segment]
        valid = (rows[:, None] < cu_q[-1]) & (indices >= 0) & (indices < lengths[:, None])
    selected = jnp.take_along_axis(scores, jnp.clip(indices, 0, scores.shape[-1] - 1), axis=-1)
    valid = valid & jnp.isfinite(selected)
    has_values = jnp.any(valid, axis=-1, keepdims=True)
    logits = jnp.where(valid, selected, -jnp.inf)
    # An empty query contributes zero; avoid log_softmax of an all-infinite row.
    logits = jnp.where(has_values, logits, 0.0)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_probs = jnp.where(valid, log_probs, 0.0)
    probabilities = jnp.where(valid, jnp.exp(log_probs), 0.0)
    target = jnp.where(valid, target, 0.0)
    loss = jnp.sum(jnp.where(target > 0, target * (jnp.log(target + eps) - log_probs), 0.0))
    return loss, (q, k, weights, target, probabilities, jnp.where(valid, indices, -1), cu_q, cu_k)


def _indexer_sparse_kl_bwd(sm_scale, eps, max_q, max_k, residual, grad_out):
    del eps
    q, k, weights, target, probabilities, indices, cu_q, cu_k = residual
    # FE averages over local physical query capacity; this API returns a sum.
    capacity = q.shape[0] if cu_q is not None else q.shape[0] * q.shape[1]
    grad_loss = jnp.asarray(grad_out * capacity, jnp.float32).reshape(1)
    dq, dw, dk = _cudnn_indexer_sparse_backward(
        q,
        k,
        weights,
        target,
        probabilities,
        indices,
        grad_loss,
        sm_scale,
        cu_q,
        cu_k,
        max_q,
        max_k,
    )
    return dq, dk.reshape(k.shape).astype(k.dtype), dw, None, None, None, None, None


_indexer_sparse_kl.defvjp(_indexer_sparse_kl_fwd, _indexer_sparse_kl_bwd)


def dsa_indexer_sparse_kl_loss(
    q: jnp.ndarray,
    k: jnp.ndarray,
    weights: jnp.ndarray,
    scores: jnp.ndarray,
    indices: jnp.ndarray,
    target: jnp.ndarray,
    *,
    sm_scale: float = 1.0,
    eps: float = 1e-8,
    batch_axes: tuple[str, ...] = (),
    sequence_descriptor: SequenceDescriptor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
) -> jnp.ndarray:
    """Sum the sparse indexer KL, with native gradients for Q/K/weights only.

    BSHD Q/K/weights are BF16 [B,S,64,128]/[B,K,1,128]/[B,S,64]. Scores are
    FP32 [B,S,K] from these operands and ``sm_scale``. Unique local INT32
    indices and detached FP32 target have shape [B,S,512]; use -1/zero for
    inactive slots. Targets retain their mass. Empty rows contribute zero.
    The scalar is a global sum over batch shards, without loss coefficient
    or token normalization. Scores, indices and teacher receive no gradients.

    Packed THD uses Q [Tq,64,128], K [Tk,1,128], weights [Tq,64], scores
    [Tq,ceil4(maxK)] and indices/target [Tq,512]. The explicit contiguous-pack
    SequenceDescriptor and positive static max lengths follow :func:`dsa_indexer`;
    indices are segment-local compressed-key columns. Empty segments, invalid
    slots and unused capacity contribute zero gradients, even with poisoned
    unused storage. Active operands must be finite. THD does not accept batch
    sharding axes. Requires FE ``indexer_backward_jax(..., clip_log=False)``
    with packed metadata support for THD.
    """
    cu_q, cu_k = _indexer_prefixes(q, sequence_descriptor, max_seqlen_q, max_seqlen_k)
    if q.shape[-2:] != (64, 128):
        raise ValueError("sparse KL requires indexer H64/D128")
    if cu_q is None:
        b, sq = q.shape[:2]
        if k.ndim != 4 or k.shape[0] != b or k.shape[2:] != (1, 128):
            raise ValueError("sparse KL requires BSHD K with one KV head")
        score_shape = (b, sq, k.shape[1])
    else:
        if batch_axes:
            raise ValueError("THD sparse KL does not support batch sharding axes")
        if k.ndim != 3 or k.shape[1:] != (1, 128) or not q.shape[0] or not k.shape[0]:
            raise ValueError("THD sparse KL requires nonempty Q/K capacity and one KV head")
        if any(
            not isinstance(n, int) or isinstance(n, bool) or n <= 0
            for n in (max_seqlen_q, max_seqlen_k)
        ):
            raise ValueError("THD sparse KL requires positive static max Q/K lengths")
        score_shape = (q.shape[0], -(-max_seqlen_k // 4) * 4)
    if weights.shape != q.shape[:-1] or scores.shape != score_shape:
        raise ValueError("sparse KL score/weight shape mismatch")
    if indices.shape != (*q.shape[:-2], 512) or target.shape != indices.shape:
        raise ValueError("sparse KL requires 512 selected slots, padded with -1/zero")
    if any(x.dtype != jnp.bfloat16 for x in (q, k, weights)):
        raise TypeError("sparse KL q/k/weights must be bfloat16")
    if scores.dtype != jnp.float32 or target.dtype != jnp.float32 or indices.dtype != jnp.int32:
        raise TypeError("sparse KL scores/target must be float32 and indices int32")
    operands = (q, k, weights, scores, indices, target)
    if cu_q is not None:
        return _indexer_sparse_kl(*operands, cu_q, cu_k, sm_scale, eps, max_seqlen_q, max_seqlen_k)
    partials = _batched_shard_map(
        lambda *xs: _indexer_sparse_kl(*xs, None, None, sm_scale, eps, None, None).reshape(1),
        tuple(_batch_axes_spec(batch_axes, x.ndim) for x in operands),
        _batch_axes_spec(batch_axes, 1),
        batch_axes,
    )(*operands)
    return jnp.sum(partials)


def _cudnn_sparse_attention_forward(
    q, kv, topk_indices, topk_length, attn_sink, *, indexer_topk, softmax_scale
):
    from cudnn.ops import dsa

    return dsa.sparse_attention_forward_jax(
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
    from cudnn.ops import dsa

    return dsa.sparse_attention_backward_jax(
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
    return _dsa_sparse_attention_fwd(
        q, kv, topk_indices, topk_length, attn_sink, indexer_topk, softmax_scale
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
    q: jnp.ndarray,
    kv: jnp.ndarray,
    topk_indices: jnp.ndarray,
    topk_length: jnp.ndarray,
    attn_sink: jnp.ndarray,
    *,
    indexer_topk: int = 512,
    softmax_scale: float = 1.0,
) -> jnp.ndarray:
    """Differentiable SM100 DSv4 sparse-attention kernel.

    ``q`` is BF16 ``[Tq,64,512]`` and shared ``kv`` is BF16 ``[Tkv,512]``.
    INT32 ``topk_indices`` is ``[Tq,N]``, with physical width ``N`` a multiple
    of 64. Indices address rows of the supplied flattened KV array; ``-1``
    marks an unused slot. INT32 ``topk_length`` is ``[Tq]`` and bounds the
    prefix of slots considered per query, with values in ``[0,N]``. Callers
    construct these lists, including causal and sequence-boundary masking.

    FP32 ``attn_sink`` is ``[64]``: a per-head logit contributing to the
    softmax denominator with zero value. ``softmax_scale`` multiplies Q/K
    logits and defaults to 1.0; no inverse-square-root scale is inferred.
    ``indexer_topk`` is the prefix width for cuDNN's auxiliary indexer LSE
    statistic (0, 512, 1024, or 2048, no larger than ``N``), not the total
    attended width. Use 0 for HCA. TE returns only the BF16 output matching
    Q's shape; the auxiliary statistic is not exposed.

    Gradients are returned for Q, KV, and the attention sink. Sparse indices,
    lengths, and configuration are not differentiated.
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
    q: jnp.ndarray,
    kv: jnp.ndarray,
    topk_indices: jnp.ndarray,
    topk_length: jnp.ndarray,
    attn_sink: jnp.ndarray,
    *,
    batch_axes: tuple[str, ...] = (),
    indexer_topk: int = 512,
    softmax_scale: float = 1.0,
) -> jnp.ndarray:
    """Run sparse attention on local batch shards using local KV indices.

    Batched form of :func:`dsa_sparse_attention`: Q is ``[B,S,64,512]``, KV
    is ``[B,K,512]``, indices are ``[B,S,N]``, lengths are ``[B,S]``, and the
    output matches Q. Dtypes, scales, and gradients follow the raw function;
    the FP32 ``[64]`` sink is shared across the batch.

    Nonnegative indices are relative to each example's KV rows ``[0,K)``;
    this wrapper offsets them into the local flattened KV array and preserves
    ``-1`` padding. ``batch_axes`` follows :func:`csa_compressor_batched`.
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
    "dsa_indexer_sparse_kl_loss",
    "dsa_sparse_attention",
    "dsa_sparse_attention_batched",
    "hca_compressor_batched",
]

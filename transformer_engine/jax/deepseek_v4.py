# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Differentiable per-kernel JAX APIs for DeepSeek-V4 CSA.

The cuDNN Frontend package owns only the raw CuTeDSL forward/backward calls.
Transformer Engine owns automatic differentiation and the public JAX contract.
These functions intentionally do not compose a higher-level attention module.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp


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
def _csa_compressor(
    kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp, ratio, coff
):
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


def _csa_compressor_fwd(
    kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp, ratio, coff
):
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
    """Differentiable SM100 CSA compressor kernel.

    Inputs use the raw packed kernel layout. Sequence metadata is treated as
    non-differentiable; gradients are returned for ``kv``, ``score``, and ``ape``.
    """
    return _csa_compressor(
        kv, score, ape, cu_seqlens, cu_seqlens_comp, total_comp, ratio, coff
    )


def _cudnn_indexer_forward(q, k, weights, *, ratio, sm_scale):
    from cudnn import indexer_forward_jax_sm100

    return indexer_forward_jax_sm100(
        q, k, weights, ratio=ratio, sm_scale=sm_scale
    )


def _indexer_reference(q, k, weights, ratio, sm_scale):
    dots = jnp.einsum(
        "bshd,bwkd->bhsw", q.astype(jnp.float32), k.astype(jnp.float32)
    )
    scores = jnp.einsum(
        "bhsw,bsh->bsw",
        jax.nn.relu(dots) * sm_scale,
        weights.astype(jnp.bfloat16).astype(jnp.float32),
    )
    q_positions = jnp.arange(q.shape[1])[:, None]
    block_ends = ratio * (jnp.arange(k.shape[1])[None, :] + 1)
    return jnp.where(block_ends <= q_positions + 1, scores, -jnp.inf)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _dsa_indexer(q, k, weights, ratio, sm_scale):
    return _cudnn_indexer_forward(q, k, weights, ratio=ratio, sm_scale=sm_scale)


def _dsa_indexer_fwd(q, k, weights, ratio, sm_scale):
    out = _cudnn_indexer_forward(q, k, weights, ratio=ratio, sm_scale=sm_scale)
    return out, (q, k, weights)


def _dsa_indexer_bwd(ratio, sm_scale, residual, grad_out):
    q, k, weights = residual
    _, pullback = jax.vjp(
        lambda q_, k_, weights_: _indexer_reference(
            q_, k_, weights_, ratio, sm_scale
        ),
        q,
        k,
        weights,
    )
    return pullback(grad_out)


_dsa_indexer.defvjp(_dsa_indexer_fwd, _dsa_indexer_bwd)


def dsa_indexer(
    q: Any,
    k: Any,
    weights: Any,
    *,
    ratio: int = 4,
    sm_scale: float = 1.0,
) -> Any:
    """Differentiable SM100 DSv4 indexer score kernel.

    The CuTeDSL kernel implements the forward pass. Until cuDNN exposes an
    equivalent compact JAX backward call, TE computes its exact mathematical
    VJP with JAX operations.
    """
    return _dsa_indexer(q, k, weights, ratio, sm_scale)


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
def _dsa_sparse_attention(
    q, kv, topk_indices, topk_length, attn_sink, indexer_topk, softmax_scale
):
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


def _dsa_sparse_attention_bwd(
    indexer_topk, softmax_scale, residual, grad_out
):
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


_dsa_sparse_attention.defvjp(
    _dsa_sparse_attention_fwd, _dsa_sparse_attention_bwd
)


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
    indices and lengths are non-differentiable metadata.
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


__all__ = ["csa_compressor", "dsa_indexer", "dsa_sparse_attention"]

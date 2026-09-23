# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Experimental PyTorch bindings for cuDNN Frontend 1.29.0 DSv4 primitives.

Compression, selection, sparse attention forward, and backward call cuDNN
Frontend APIs. There are no numerical fallbacks. Inputs use packed sequences;
model projections/norm/RoPE are external.
"""

from importlib import import_module
from typing import Optional

import torch
from torch.autograd.function import once_differentiable

__all__ = ["compress", "select_blocks", "attention"]


def _namespace(name):
    try:
        return getattr(import_module("cudnn"), name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "DSv4 requires nvidia-cudnn-frontend>=1.29.0 "
            f"with the cudnn.{name} namespace."
        ) from exc


def _sparse_attention_forward():
    try:
        return getattr(_namespace("DSA"), "sparse_attention_forward_wrapper")
    except AttributeError as exc:
        raise ImportError(
            "DSv4 attention requires cudnn.DSA.sparse_attention_forward_wrapper "
            "from nvidia-cudnn-frontend>=1.29.0."
        ) from exc


def _validate_tensors(reference, bf16, fp32, cu_seqlens, cu_seqlens_comp):
    """Check metadata without copying packed sequence offsets to the CPU.

    Callers own prefix contents: start at zero, monotonic, final offsets equal
    valid row counts, and compressed lengths floor(length / ratio).
    """
    if not reference.is_cuda or torch.cuda.get_device_capability(reference.device) != (
        10,
        0,
    ):
        raise ValueError("This DSv4 implementation requires an SM100 GPU.")
    for tensor in (reference, *bf16, *fp32, cu_seqlens, cu_seqlens_comp):
        if tensor.device != reference.device or not tensor.is_contiguous():
            raise ValueError("DSv4 inputs must be contiguous on the same CUDA device.")
    if any(t.dtype != torch.bfloat16 for t in (reference, *bf16)):
        raise TypeError("DSv4 activations must be BF16.")
    if any(t.dtype != torch.float32 for t in fp32):
        raise TypeError("DSv4 position biases and sinks must be FP32.")
    for offsets in (cu_seqlens, cu_seqlens_comp):
        if offsets.dtype != torch.int32 or offsets.ndim != 1 or offsets.numel() < 2:
            raise ValueError("Sequence prefixes must be INT32 [batch+1].")
    if cu_seqlens.shape != cu_seqlens_comp.shape:
        raise ValueError("Local and compressed prefixes must describe the same batch.")


class _Compress(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, kv, score, ape, cu_seqlens, cu_seqlens_comp, ratio, overlap, total_comp
    ):
        coff = 2 if overlap else 1
        if kv.ndim != 2 or kv.shape[-1] % coff:
            raise ValueError("kv must have shape [tokens, coff * head_dim].")
        head_dim = kv.shape[-1] // coff
        result = _namespace("CSA").csa_compressor_forward_wrapper(
            kv,
            score,
            ape,
            cu_seqlens,
            cu_seqlens_comp,
            ratio=ratio,
            head_dim=head_dim,
            coff=coff,
            total_comp=total_comp,
        )
        ctx.save_for_backward(kv, score, ape, cu_seqlens, cu_seqlens_comp)
        ctx.config = ratio, head_dim, coff
        return result["out"]

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        ratio, head_dim, coff = ctx.config
        # cuDNN owns scratch allocation and dAPE zeroing. Autograd may supply
        # a strided/expanded gradient, while this kernel requires contiguous data.
        result = _namespace("CSA").csa_compressor_backward_wrapper(
            *ctx.saved_tensors,
            grad_out.contiguous(),
            ratio=ratio,
            head_dim=head_dim,
            coff=coff,
        )
        return (
            result["grad_kv"],
            result["grad_score"],
            result["grad_ape"],
            None,
            None,
            None,
            None,
            None,
        )


def compress(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    *,
    ratio: int,
    overlap: bool,
    total_comp: int,
) -> torch.Tensor:
    """Gated pooling, with cuDNN forward and first-order backward.

    kv/score: contiguous BF16 [T, coff*D]; ape: FP32 [ratio, coff*D].
    coff is 2 for overlapping pooling, 1 otherwise. Sequence prefixes are
    CUDA INT32 [B+1]; compressed segment lengths are floor(length / ratio).
    Returns BF16 [total_comp, D]. Per-segment tails are dropped by cuDNN.

    Pass total_comp explicitly because inferring it from a CUDA prefix would
    synchronize. It may be a capacity; only rows below cu_seqlens_comp[-1]
    are valid. Remaining kernel-specific constraints are checked by cuDNN.
    """
    if ratio <= 0 or total_comp < 0:
        raise ValueError("ratio must be positive and total_comp nonnegative.")
    if kv.ndim != 2 or score.shape != kv.shape or ape.shape != (ratio, kv.shape[-1]):
        raise ValueError("Expected kv/score [T,coff*D] and ape [ratio,coff*D].")
    _validate_tensors(kv, (score,), (ape,), cu_seqlens, cu_seqlens_comp)
    return _Compress.apply(
        kv,
        score,
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        ratio,
        overlap,
        total_comp,
    )


@torch.no_grad()
def select_blocks(
    query: torch.Tensor,
    compressed_key: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    *,
    top_k: int,
    ratio: int,
    max_seqlen: int,
    max_compressed_seqlen: int,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """CSA selection using cuDNN's SM100 combined score + Top-K operation.

    This is the indexer, with its own projected Q/K/weights, not attention Q/KV.
    BF16 query [T,H,128], compressed_key [Tc,128], weights [T,H]; H=32 or 64.
    Full packed-sequence prefill only: causal timeline offsets start at zero.
    Returns INT32 [T,top_k] indices into compressed_key's global packed rows;
    -1 denotes an invalid slot. These are NOT IDs into local+compressed KV.

    Selection is discrete. Scale defaults to 1/sqrt(index head dim); apply any
    separate head-reduction scaling to weights before this call. This helper
    does not implement the separate indexer auxiliary training loss or promise
    autograd through the selected IDs.
    """
    if query.ndim != 3 or query.shape[1] not in (32, 64) or query.shape[2] != 128:
        raise ValueError("CSA index query must have shape [T,32 or 64,128].")
    if compressed_key.ndim != 2 or compressed_key.shape[1] != 128:
        raise ValueError("CSA index key must have shape [Tc,128].")
    if weights.shape != query.shape[:2]:
        raise ValueError("CSA index weights must have shape [T,H].")
    if ratio <= 0 or top_k <= 0 or max_seqlen <= 0 or max_compressed_seqlen <= 0:
        raise ValueError(
            "CSA ratio, top_k and maximum sequence lengths must be positive."
        )
    _validate_tensors(query, (compressed_key, weights), (), cu_seqlens, cu_seqlens_comp)
    if scale is None:
        scale = query.shape[-1] ** -0.5
    result = _namespace("DSA").indexer_forward_top_k_wrapper(
        query,
        compressed_key.unsqueeze(1),
        weights,
        top_k=top_k,
        ratio=ratio,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens_comp,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_compressed_seqlen,
        precision="bf16",
        topk_indices_global=True,
        return_softmax=False,
        sm_scale=scale,
    )
    return result["indices"]


def _attention_indices(
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    window_size: int,
    ratio: int,
    indices: Optional[torch.Tensor],
    max_compressed_seqlen: Optional[int],
) -> torch.Tensor:
    """Map local and compressed entries into one concatenated KV address space."""
    total_q = query.shape[0]
    device = query.device
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    seq_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=device),
        lengths,
        output_size=total_q,
    )
    token = torch.arange(total_q, device=device)
    seq_start = cu_seqlens[seq_ids]
    position = token - seq_start

    local_position = position[:, None] - torch.arange(
        window_size - 1,
        -1,
        -1,
        device=device,
    )
    local = seq_start[:, None] + local_position
    local = torch.where(local_position >= 0, local, -1)

    comp_start = cu_seqlens_comp[seq_ids]
    comp_end = cu_seqlens_comp[seq_ids + 1]
    if indices is None:
        if max_compressed_seqlen is None:
            raise ValueError("max_compressed_seqlen is required for HCA attention.")
        comp_position = torch.arange(max_compressed_seqlen, device=device)[None, :]
        compressed = comp_start[:, None] + comp_position
        valid = (compressed < comp_end[:, None]) & (
            (comp_position + 1) * ratio <= position[:, None] + 1
        )
    else:
        if indices.ndim != 2 or indices.shape[0] != total_q:
            raise ValueError("CSA indices must have shape [total_tokens, top_k].")
        compressed = indices
        comp_position = compressed - comp_start[:, None]
        valid = (
            (compressed >= comp_start[:, None])
            & (compressed < comp_end[:, None])
            & ((comp_position + 1) * ratio <= position[:, None] + 1)
        )
    compressed = torch.where(valid, compressed + query.shape[0], -1)
    # cuDNN accepts an arbitrary logical width and pads its internal kernel
    # layout, so preserve only the model-visible local + compressed slots.
    return torch.cat((local, compressed), dim=1).to(torch.int32).contiguous()


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, kv, indices, sink, scale):
        result = _sparse_attention_forward()(
            query,
            kv,
            indices,
            attn_sink=sink,
            softmax_scale=scale,
            # Prefix LSE is only needed by the optional indexer-loss path.
            indexer_topk=0,
        )
        out, lse = result["out"], result["lse"]
        ctx.save_for_backward(query, kv, out, lse, sink, indices)
        ctx.scale = scale
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        query, kv, out, lse, sink, indices = ctx.saved_tensors
        result = _namespace("DSA").sparse_attention_backward_wrapper(
            query,
            kv,
            out,
            grad_out.contiguous(),
            lse,
            sink,
            indices,
            softmax_scale=ctx.scale,
        )
        return result["dq"], result["dkv"], None, result["d_sink"], None


def attention(
    query: torch.Tensor,
    local_kv: torch.Tensor,
    compressed_kv: torch.Tensor,
    sink: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    *,
    window_size: int,
    ratio: int,
    indices: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    max_compressed_seqlen: Optional[int] = None,
) -> torch.Tensor:
    """Joint local + compressed DSv4 attention with first-order autograd.

    Full packed-sequence prefill: query [T,H,D], local_kv [T,D], compressed_kv
    [Tc,D], sink [H]. Q/KV have already had the model's norm/RoPE applied.
    cu_seqlens describes both query and local KV; cu_seqlens_comp describes
    valid compressed rows, excluding capacity padding. Values are the first
    512 channels of each shared KV row (Q/K width may also be 576).

    Local positions satisfy max(0,q-window_size+1) <= k <= q per sequence.
    Compressed block j is eligible when (j+1)*ratio <= q+1. With indices (CSA),
    use its selected, valid, causal compressed rows; without indices (HCA),
    use every eligible compressed row. IDs refer to compressed_kv only.

    One cuDNN sparse-attention call normalizes the combined visible entries with
    the sink and provides the KV-only LSE consumed by cuDNN backward. Output is
    [T,H,512], before output unrotation/projection. Scale defaults to D**-0.5.
    No dense attention mask is part of this interface. HCA must provide
    max_compressed_seqlen to bound its dense compressed-history metadata without
    reading a CUDA prefix on host.
    """
    if query.ndim != 3 or query.shape[1] != 64:
        raise ValueError("DSv4 attention requires query [T,64,D].")
    if query.shape[-1] not in (512, 576):
        raise ValueError("attention head dimension must be 512 or 576.")
    if local_kv.ndim != 2 or compressed_kv.ndim != 2:
        raise ValueError("local_kv and compressed_kv must be 2D packed tensors.")
    if (
        local_kv.shape[-1] != query.shape[-1]
        or compressed_kv.shape[-1] != query.shape[-1]
    ):
        raise ValueError("Q and both KV tensors must have the same QK dimension.")
    if local_kv.shape[0] != query.shape[0]:
        raise ValueError("Full-sequence attention requires one local KV row per query.")
    if sink.shape != (query.shape[1],):
        raise ValueError("sink must have shape [64].")
    if window_size <= 0 or ratio <= 0:
        raise ValueError("window_size and ratio must be positive.")
    if indices is None and (max_compressed_seqlen is None or max_compressed_seqlen < 0):
        raise ValueError("HCA requires a nonnegative max_compressed_seqlen.")
    _validate_tensors(
        query, (local_kv, compressed_kv), (sink,), cu_seqlens, cu_seqlens_comp
    )
    if indices is not None:
        if indices.dtype != torch.int32 or indices.device != query.device:
            raise ValueError("CSA indices must be CUDA INT32 on the query device.")
    if scale is None:
        scale = query.shape[-1] ** -0.5
    selected = _attention_indices(
        query,
        compressed_kv,
        cu_seqlens,
        cu_seqlens_comp,
        window_size,
        ratio,
        indices,
        max_compressed_seqlen,
    )
    return _Attention.apply(
        query.contiguous(),
        torch.cat((local_kv, compressed_kv), dim=0),
        selected,
        sink.contiguous(),
        scale,
    )

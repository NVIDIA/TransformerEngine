# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrappers for cuDNN DSA/CSA primitives."""

from importlib import import_module
from typing import Optional

import torch
from torch.autograd.function import once_differentiable

__all__ = [
    "compress",
    "select_indices",
    "sparse_attention",
    "dense_indexer_loss_scores",
    "dense_indexer_loss_backward",
]


def _namespace(name):
    try:
        return getattr(import_module("cudnn"), name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"Sparse attention requires nvidia-cudnn-frontend>=1.29.0 with the cudnn.{name}"
            " namespace."
        ) from exc


def _dsa_op(name):
    try:
        return getattr(_namespace("DSA"), name)
    except AttributeError as exc:
        raise ImportError(
            f"Sparse attention requires cudnn.DSA.{name} from nvidia-cudnn-frontend>=1.29.0."
        ) from exc


def _validate_tensors(reference, bf16, fp32, cu_seqlens, cu_seqlens_comp):
    """Check tensor metadata without copying sequence boundaries to the CPU."""
    if not reference.is_cuda or torch.cuda.get_device_capability(reference.device) not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        raise ValueError(
            "These cuDNN sparse attention kernels require an SM100, SM103, or SM107 GPU."
        )
    for tensor in (reference, *bf16, *fp32, cu_seqlens, cu_seqlens_comp):
        if tensor.device != reference.device or not tensor.is_contiguous():
            raise ValueError("Inputs must be contiguous on the same CUDA device.")
    if any(t.dtype != torch.bfloat16 for t in (reference, *bf16)):
        raise TypeError("Activations must be BF16.")
    if any(t.dtype != torch.float32 for t in fp32):
        raise TypeError("Position biases and sinks must be FP32.")
    for offsets in (cu_seqlens, cu_seqlens_comp):
        if offsets.dtype != torch.int32 or offsets.ndim != 1 or offsets.numel() < 2:
            raise ValueError("Sequence prefixes must be INT32 [batch+1].")
    if cu_seqlens.shape != cu_seqlens_comp.shape:
        raise ValueError("Local and compressed prefixes must describe the same batch.")


class _Compress(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, score, ape, cu_seqlens, cu_seqlens_comp, ratio, overlap, total_comp):
        """Run cuDNN compression and save its backward inputs."""
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
        """Propagate gradients through cuDNN compression."""
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
    """Compress token blocks with cuDNN forward and backward.

    Overlapping compression combines the current complete block with the previous
    block; otherwise each block is compressed independently. Trailing input tokens
    that do not fill a complete block are excluded from compression. This operation
    accepts projected tensors; it does not apply the module's projections, RMSNorm
    or RoPE. See the sparse_attention README for supported dtypes and devices.

    Parameters
    ----------
    kv, score : torch.Tensor
        Contiguous projected KV and gate scores of shape ``[T,coff*D]`` on the
        same device. ``coff`` is 2 when ``overlap=True``, otherwise 1.
    ape : torch.Tensor
        FP32 learned additive position embedding ``[ratio,coff*D]``,
        added to gate scores before softmax over each compression block.
    cu_seqlens, cu_seqlens_comp : torch.Tensor
        Contiguous CUDA INT32 ``[B+1]`` cumulative input/compressed token counts.
        Start at zero; each increment is the corresponding sequence's token
        count. Compressed counts are ``floor(sequence_length / ratio)``.
    ratio : int
        Positive number of input tokens per block producing one compressed token.
    overlap : bool
        Whether to include the previous block using its separate projections.
    total_comp : int
        Allocated compressed output length, at least ``cu_seqlens_comp[-1]``.

    Returns
    -------
    torch.Tensor
        Compressed KV ``[total_comp,D]`` before RMSNorm and RoPE. Only indices
        below ``cu_seqlens_comp[-1]`` contain valid compressed tokens; remaining
        output is padding. Autograd propagates gradients to kv, score and ape.
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
def select_indices(
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
    """Select compressed KV tokens using cuDNN's index scores and top-k operation.

    Inputs are the indexer's projected Q/K/head weights after compression/RoPE;
    this operation does not project or compress inputs. It supports complete
    sequences starting at position zero. See the sparse_attention README for
    supported dtypes and devices.

    Parameters
    ----------
    query : torch.Tensor
        Contiguous index queries ``[T,H,128]``, with H=32 or 64.
    compressed_key : torch.Tensor
        Contiguous packed compressed index keys ``[Tc,128]``.
    weights : torch.Tensor
        Contiguous projected per-token head weights ``[T,H]``.
    cu_seqlens, cu_seqlens_comp : torch.Tensor
        Contiguous CUDA INT32 ``[B+1]`` cumulative query/compressed token counts,
        starting at zero. Adjacent differences give each sequence's token count.
    top_k : int
        Positive maximum number of compressed KV token indices per query token.
    ratio : int
        Positive number of input tokens per compressed block.
    max_seqlen : int
        Maximum input sequence length, passed to cuDNN as ``max_seqlen_q``.
    max_compressed_seqlen : int
        Size passed to cuDNN as ``max_seqlen_k``. Use
        ``(max_seqlen + ratio - 1) // ratio``, as Indexer does, even though
        ``cu_seqlens_comp`` counts only complete compressed blocks.
    scale : float, optional
        Score multiplier. None uses ``query.shape[-1] ** -0.5``. Indexer includes
        the positive head-normalization factor using
        ``(head_dim * n_heads) ** -0.5``. Pass raw projected weights; scaling then
        rounding weights back to their activation dtype can change close top-k results.

    Returns
    -------
    torch.Tensor
        CUDA INT32 ``[T,top_k]`` global indices into packed compressed_key, one
        list per query token; -1 marks invalid entries. Indices address compressed
        KV only, before adding the local KV offset. Selection has no autograd;
        train the indexer with a separate caller-owned auxiliary loss.
    """
    if query.ndim != 3 or query.shape[1] not in (32, 64) or query.shape[2] != 128:
        raise ValueError("CSA index query must have shape [T,32 or 64,128].")
    if compressed_key.ndim != 2 or compressed_key.shape[1] != 128:
        raise ValueError("CSA index key must have shape [Tc,128].")
    if weights.shape != query.shape[:2]:
        raise ValueError("CSA index weights must have shape [T,H].")
    if ratio <= 0 or top_k <= 0 or max_seqlen <= 0 or max_compressed_seqlen <= 0:
        raise ValueError("CSA ratio, top_k and maximum sequence lengths must be positive.")
    _validate_tensors(query, (compressed_key, weights), (), cu_seqlens, cu_seqlens_comp)
    if scale is None:
        scale = query.shape[-1] ** -0.5
    result = _dsa_op("indexer_forward_top_k_wrapper")(
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


def _pad_index_heads(query, weights):
    if query.ndim != 4 or query.shape[0] != 1 or query.shape[2] not in (32, 64):
        raise ValueError("Dense indexer loss requires B1 [B,S,32 or 64,128] queries.")
    if query.shape[-1] != 128 or weights.shape != query.shape[:3]:
        raise ValueError("Dense indexer weights must match B1 index queries.")
    if query.dtype != torch.bfloat16 or weights.dtype != torch.bfloat16:
        raise TypeError("Dense indexer Q and weights must be BF16.")
    pad = 64 - query.shape[2]
    if pad:
        return torch.nn.functional.pad(query, (0, 0, 0, pad)), torch.nn.functional.pad(
            weights, (0, pad)
        )
    return query, weights


def dense_indexer_loss_scores(
    index_q,
    index_k,
    index_w,
    attn_q,
    compressed_kv,
    full_lse,
    *,
    ratio: int,
    index_scale: float,
    attention_scale: float,
):
    """Recompute indexer and teacher scores for a single-sequence auxiliary loss.

    See the sparse_attention README for support requirements and the loss example.
    These primitives do not attach an autograd loss to the indexer tensors.

    Parameters
    ----------
    index_q, index_k, index_w : torch.Tensor
        Indexer query/key/head weights of shapes ``[1,S,H,128]``, ``[1,Sc,128]``
        and ``[1,S,H]``, with H=32 or 64.
    attn_q, compressed_kv : torch.Tensor
        Teacher attention queries ``[1,S,64,D]`` and compressed KV ``[1,Sc,D]``.
        Use detached teacher tensors so the auxiliary loss trains only the indexer.
    full_lse : torch.Tensor
        Detached FP32 ``[1,S,64]`` log-sum-exp of teacher logits over the local
        window, every eligible compressed KV token, and the sink. The saved sparse
        attention LSE excludes the sink and covers only selected compressed tokens,
        so it cannot be used as this dense teacher denominator.
    ratio : int
        Number of input tokens per compressed block.
    index_scale, attention_scale : float
        Score multipliers for the indexer and teacher attention, respectively.

    Returns
    -------
    tuple of dict
        ``(index_score, attn_score)`` cuDNN dictionaries with ``out`` and ``denom``
        tensors, passed to ``dense_indexer_loss_backward``. The caller owns the
        auxiliary objective and propagation through its live indexer tensors.
    """
    if any(t.dtype != torch.bfloat16 for t in (index_k, attn_q, compressed_kv)):
        raise TypeError("Dense indexer keys and attention Q/K must be BF16.")
    if full_lse.dtype != torch.float32 or full_lse.shape != attn_q.shape[:3]:
        raise ValueError("full_lse must be FP32 [B,S,attention_heads].")
    padded_q, padded_w = _pad_index_heads(index_q, index_w)
    index_score = _dsa_op("dense_indexer_score_recompute_wrapper")(
        padded_q,
        index_k.unsqueeze(2),
        padded_w,
        sm_scale=index_scale,
        ratio=ratio,
    )
    attn_score = _dsa_op("dense_attn_score_recompute_wrapper")(
        attn_q,
        compressed_kv.unsqueeze(2),
        full_lse,
        attention_scale,
        ratio=ratio,
    )
    return index_score, attn_score


def dense_indexer_loss_backward(
    index_q,
    index_k,
    index_w,
    index_score,
    attn_score,
    grad_loss,
    *,
    ratio: int,
    index_scale: float,
    loss_coeff: float,
):
    """Compute indexer gradients for KL loss averaged over query tokens, with B=1.

    Use the same tensors, compression ratio and index scale as in
    ``dense_indexer_loss_scores``. See the sparse_attention README loss example.

    Parameters
    ----------
    index_q, index_k, index_w : torch.Tensor
        Live indexer query/key/head weights ``[1,S,H,128]``, ``[1,Sc,128]`` and
        ``[1,S,H]``, with H=32 or 64. H=32 is padded internally for cuDNN.
    index_score, attn_score : dict
        Indexer and teacher dictionaries returned by ``dense_indexer_loss_scores``.
    grad_loss : torch.Tensor
        Scalar upstream gradient for the auxiliary loss.
    ratio : int
        Number of input tokens per compressed block.
    index_scale : float
        Index score multiplier used to compute index_score.
    loss_coeff : float
        Multiplier for the auxiliary loss and its gradients.

    Returns
    -------
    tuple of torch.Tensor
        ``(d_query, d_key, d_weights)`` matching the input tensor shapes. The caller
        must propagate these gradients through the live indexer tensors, for example
        with ``torch.autograd.backward``; this helper does not do that propagation.
    """
    padded_q, padded_w = _pad_index_heads(index_q, index_w)
    result = _dsa_op("dense_indexer_backward_wrapper")(
        padded_q,
        padded_w,
        index_k,
        attn_score["out"].clone(),
        attn_score["denom"],
        index_score["out"].clone(),
        index_score["denom"],
        grad_loss=grad_loss.contiguous(),
        sm_scale=index_scale,
        loss_coeff=loss_coeff,
        ratio=ratio,
    )
    heads = index_q.shape[2]
    return (
        result["d_index_q"][:, :, :heads].contiguous(),
        result["d_index_k"],
        result["d_weights"][:, :, :heads].contiguous(),
    )


def _prepare_sparse_kv_indices(
    query: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    window_size: int,
    ratio: int,
    indices: Optional[torch.Tensor],
    max_compressed_seqlen: Optional[int],
) -> torch.Tensor:
    """Build a list of KV token indices for each query token.

    Query is already in packed thd format [T,H,D]; only its token count and device are used.
    This helper does not flatten it. Return INT32 [T,window_size+compressed_slots]
    indices into torch.cat((local_kv, compressed_kv), dim=0).

    Local KV has T tokens. Compressed indices count from the start of the entire
    packed compressed tensor, including preceding sequences. Add T to address
    compressed KV after local KV in the concatenated tensor. Invalid entries are -1.
    CSA supplies top-k indices from select_indices; HCA supplies indices=None and
    this helper includes every eligible compressed token.
    """
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
    # Example: lengths=[4,11,17], T=32; query 7 is position 3 of sequence two.
    # With indices=None and max_compressed_seqlen=8 and 5, respectively:
    # window=ratio=2: cu_comp=[0,2,7,15]; local=[6,7]; compressed=[2,3].
    #   Combined KV indices: [6,7,34,35,-1,-1,-1,-1,-1,-1].
    # window=ratio=3: cu_comp=[0,1,4,9]; local=[5,6,7]; compressed=[1].
    #   Combined KV indices: [5,6,7,33,-1,-1,-1,-1].
    # Ratio 3 illustrates only this helper's index arithmetic, not cuDNN compression support.
    # Sliding-window attention still needs the original KV tokens after compression.
    # The combined KV tensor contains all T original tokens, followed by Tc compressed entries,
    # so compressed KV token indices receive offset T. Here T=32:
    # ratio 2: [2,3] -> [32+2,32+3]=[34,35]; ratio 3: [1] -> [32+1]=[33].
    # -1 marks a slot with no valid KV token for this query.
    # CSA fills unused top-k slots with -1 when fewer than top_k entries are eligible.
    compressed = torch.where(valid, compressed + query.shape[0], -1)
    # cuDNN accepts an arbitrary logical width and pads its internal kernel
    # layout, so preserve only the model-visible local + compressed slots.
    return torch.cat((local, compressed), dim=1).to(torch.int32).contiguous()


class _SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, kv, indices, sink, scale):
        """Attend query [T,64,D] to concatenated KV [T+Tc,D] using INT32 KV token indices.

        indices [T,window_size+compressed_slots] contains one list per query token
        addressing concatenated local + compressed KV, with -1 for invalid entries.
        No compressed query is constructed.
        """
        result = _dsa_op("sparse_attention_forward_wrapper")(
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
        """Propagate output gradients through cuDNN sparse attention."""
        query, kv, out, lse, sink, indices = ctx.saved_tensors
        result = _dsa_op("sparse_attention_backward_wrapper")(
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


def sparse_attention(
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
    """Run joint local and compressed KV attention with autograd.

    This full-sequence interface requires one local KV token for each query token.
    The caller prepares projected, normalized and rotated Q/KV. DSv4Attention
    converts batched inputs to packed token order before calling this function.
    See the sparse_attention README for supported dtypes and devices.

    Parameters
    ----------
    query : torch.Tensor
        Contiguous packed ``thd`` queries ``[T,64,D]``, with D=512 or 576.
    local_kv : torch.Tensor
        Contiguous local KV ``[T,D]``, in the same sequence/token order as query.
    compressed_kv : torch.Tensor
        Contiguous packed compressed KV ``[Tc,D]``. Only indices below
        ``cu_seqlens_comp[-1]`` are valid; any remaining output is padding.
    sink : torch.Tensor
        FP32 sink logits ``[64]``, one per attention head.
    cu_seqlens, cu_seqlens_comp : torch.Tensor
        Contiguous CUDA INT32 ``[B+1]`` cumulative query/local KV and compressed KV
        token counts. Start at zero; adjacent differences give sequence lengths.
    window_size : int
        Positive local window size including the current query token: local
        positions satisfy ``max(0,q-window_size+1) <= k <= q`` within each sequence.
    ratio : int
        Positive compression block length. Compressed token j in a sequence is
        eligible when ``(j+1)*ratio <= q+1`` for query position q.
    indices : torch.Tensor, optional
        CSA: CUDA INT32 ``[T,K]`` global token indices into packed compressed_kv
        only, with -1 for invalid entries. The helper checks sequence bounds and
        causality, then adds T for concatenated local + compressed KV. None selects
        HCA behavior: include all eligible compressed tokens. See the example in
        ``_prepare_sparse_kv_indices`` for the index spaces.
    scale : float, optional
        Attention score multiplier. None uses ``query.shape[-1] ** -0.5``.
    max_compressed_seqlen : int, optional
        Required for HCA: upper bound on complete compressed tokens in any sequence,
        used to build the index lists without reading CUDA sequence counts on host.
        Unused when indices is supplied.

    Returns
    -------
    torch.Tensor
        Attention output ``[T,64,512]`` before inverse RoPE/output projection.
        Values use the first 512 KV channels. One softmax includes the visible local
        and compressed KV tokens plus the sink. Autograd returns gradients for
        query, both KV tensors and sink; selection indices have no gradients.
    """
    if query.ndim != 3 or query.shape[1] != 64:
        raise ValueError("cuDNN sparse attention requires query [T,64,D].")
    if query.shape[-1] not in (512, 576):
        raise ValueError("attention head dimension must be 512 or 576.")
    if local_kv.ndim != 2 or compressed_kv.ndim != 2:
        raise ValueError("local_kv and compressed_kv must be 2D packed tensors.")
    if local_kv.shape[-1] != query.shape[-1] or compressed_kv.shape[-1] != query.shape[-1]:
        raise ValueError("Q and both KV tensors must have the same QK dimension.")
    if local_kv.shape[0] != query.shape[0]:
        raise ValueError(
            "Full-sequence DSv4 attention requires equal query and local KV token counts."
        )
    if sink.shape != (query.shape[1],):
        raise ValueError("sink must have shape [64].")
    if window_size <= 0 or ratio <= 0:
        raise ValueError("window_size and ratio must be positive.")
    if indices is None and (max_compressed_seqlen is None or max_compressed_seqlen < 0):
        raise ValueError("HCA requires a nonnegative max_compressed_seqlen.")
    _validate_tensors(query, (local_kv, compressed_kv), (sink,), cu_seqlens, cu_seqlens_comp)
    if indices is not None:
        if indices.dtype != torch.int32 or indices.device != query.device:
            raise ValueError("CSA indices must be CUDA INT32 on the query device.")
    if scale is None:
        scale = query.shape[-1] ** -0.5
    selected = _prepare_sparse_kv_indices(
        query,
        cu_seqlens,
        cu_seqlens_comp,
        window_size,
        ratio,
        indices,
        max_compressed_seqlen,
    )
    return _SparseAttention.apply(
        query.contiguous(),
        torch.cat((local_kv, compressed_kv), dim=0),
        selected,
        sink.contiguous(),
        scale,
    )

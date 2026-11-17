# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Index projections, compression, and cuDNN compressed KV selection."""

import torch

from transformer_engine.pytorch.module import Linear

from .compressor import Compressor
from .dsa_cudnn import select_indices
from .dsa_rope import apply_rotary


class Indexer(torch.nn.Module):
    """Select compressed KV tokens using the indexer's own Q/K/weight projections.

    Accept batched [B,S,D] or packed thd [T,D] hidden states and matching query residuals.
    Standalone calls do not interpret sequence-major [S,B,D] input; DSv4Attention
    transposes that layout first. Return global compressed KV token indices;
    return_context=True also returns live Q/K/head weights for a caller-owned loss.
    See the sparse_attention README for support and training requirements.

    Parameters
    ----------
    hidden_size : int
        Input hidden-state width for compressed-key and head-weight projections.
    q_lora_rank : int
        Width of the query-down residual supplied to forward.
    head_dim : int
        Index query/key width; the current selection operation uses 128.
    n_heads : int
        Number of index heads, 32 or 64. Also sets the per-token scoring-weight width.
    top_k : int
        Maximum compressed KV tokens to select per query token.
    ratio : int
        Input tokens per compression block. Indexer uses overlapping compression.
    eps : float
        Epsilon for the compressed-key RMSNorm.
    device : torch.device or str
        Device on which to allocate parameters.
    params_dtype : torch.dtype
        Dtype used to initialize projection and compression parameters.
    """

    def __init__(
        self,
        hidden_size,
        q_lora_rank,
        head_dim,
        n_heads,
        top_k,
        ratio,
        eps,
        device,
        params_dtype,
    ):
        super().__init__()
        self.compressor = Compressor(
            hidden_size,
            head_dim,
            ratio,
            True,
            eps,
            device,
            params_dtype,
            weight_width=n_heads,
        )
        kw = {"bias": False, "device": device, "params_dtype": params_dtype}
        self.q_proj = Linear(q_lora_rank, n_heads * head_dim, **kw)
        self.head_dim, self.n_heads = head_dim, n_heads
        self.top_k, self.ratio = top_k, ratio

    def forward(
        self,
        hidden_states,
        q_residual,
        cu,
        cu_comp,
        token_rope,
        compressed_rope,
        *,
        total_comp,
        max_seqlen,
        return_context=False,
    ):
        """Select compressed KV entries and optionally expose live indexer tensors.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Batched ``[B,S,hidden_size]`` with equal sequence length S, or packed ``thd``
            ``[T,hidden_size]`` with concatenated sequences.
        q_residual : torch.Tensor
            Query-down residual with matching leading axes and width ``q_lora_rank``.
        cu, cu_comp : torch.Tensor
            CUDA INT32 ``[B+1]`` cumulative input/compressed token counts, starting
            at zero. Each compressed increment is the input sequence length
            divided by ratio, rounded down.
        token_rope, compressed_rope : tuple of torch.Tensor
            FP32 cosine/sine pairs ``[1,max_positions,1,rotary_width]`` for
            token and compressed-window positions, respectively.
        total_comp : int
            Allocated compressed output length: exactly ``B * (S // ratio)`` for
            ``[B,S,hidden_size]`` input; packed ``thd`` output may include padding after the valid
            compressed KV tokens.
        max_seqlen : int
            Maximum sequence length used to bound selection. For ``[B,S,hidden_size]``
            input this must equal S; for packed ``thd`` input use the longest sequence length.
        return_context : bool, default = False
            Also return differentiable Q/K/head weights for a caller-owned loss.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            INT32 ``[T,min(top_k,max_seqlen // ratio)]`` global compressed KV token
            indices addressing compressed KV only, with invalid entries set to -1.
            With context, return
            ``(indices, query[T,n_heads,head_dim], key[total_comp,head_dim],
            weights[T,n_heads])``. Selected token indices do not carry gradients.
        """
        n_comp = max_seqlen // self.ratio
        comp_shape = (total_comp,) if hidden_states.ndim == 2 else (hidden_states.shape[0], n_comp)
        index_compressed, weights = self.compressor(
            hidden_states, cu, cu_comp, total_comp=total_comp
        )
        # The singleton axis is the shared index-key head, not a tunable head count.
        key = (
            apply_rotary(
                index_compressed.reshape(*comp_shape, 1, self.head_dim),
                *compressed_rope,
                cu_comp,
            )
            .reshape(total_comp, self.head_dim)
            .contiguous()
        )
        query = self.q_proj(q_residual).reshape(
            *hidden_states.shape[:-1], self.n_heads, self.head_dim
        )
        query = (
            apply_rotary(query, *token_rope, cu)
            .reshape(-1, self.n_heads, self.head_dim)
            .contiguous()
        )
        # Top-k token indices are discrete; training index projections needs a separate loss.
        weights = weights.reshape(-1, self.n_heads).contiguous()
        indices = select_indices(
            query,
            key,
            weights,
            cu,
            cu_comp,
            top_k=min(self.top_k, n_comp),
            ratio=self.ratio,
            max_seqlen=max_seqlen,
            # Pass ceil(max_seqlen / ratio) as cuDNN max_seqlen_k; cu_comp still
            # counts only complete blocks, using floor(length / ratio) per sequence.
            max_compressed_seqlen=(max_seqlen + self.ratio - 1) // self.ratio,
            scale=(self.head_dim * self.n_heads) ** -0.5,
        )
        return (indices, query, key, weights) if return_context else indices

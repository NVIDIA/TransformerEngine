# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CSA index projections, compression, and cuDNN block selection."""

import torch

from transformer_engine.pytorch.module import Linear

from .compressor import _Compressor
from .dsa_cudnn import select_blocks
from .dsa_rope import apply_rotary


class _Indexer(torch.nn.Module):
    """Select compressed rows using the indexer's own Q/K/weight projections."""

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
        *,
        fused,
    ):
        super().__init__()
        self.compressor = _Compressor(
            hidden_size,
            head_dim,
            ratio,
            True,
            eps,
            device,
            params_dtype,
            fused=fused,
            weight_width=n_heads if fused else 0,
        )
        kw = {"bias": False, "device": device, "params_dtype": params_dtype}
        self.q_proj = Linear(q_lora_rank, n_heads * head_dim, **kw)
        if not fused:
            self.weights_proj = Linear(hidden_size, n_heads, **kw)
        self.head_dim, self.n_heads = head_dim, n_heads
        self.top_k, self.ratio, self.fused = top_k, ratio, fused

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
        """Select compressed rows and optionally expose live tensors for the indexer loss."""
        n_comp = max_seqlen // self.ratio
        comp_shape = (total_comp,) if hidden_states.ndim == 2 else (hidden_states.shape[0], n_comp)
        index_compressed = self.compressor(hidden_states, cu, cu_comp, total_comp=total_comp)
        if self.fused:
            index_compressed, weights = index_compressed
        else:
            weights = self.weights_proj(hidden_states)
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
        # Top-k IDs are discrete; training index projections needs a separate loss.
        weights = weights.reshape(-1, self.n_heads).contiguous()
        indices = select_blocks(
            query,
            key,
            weights,
            cu,
            cu_comp,
            top_k=min(self.top_k, n_comp),
            ratio=self.ratio,
            max_seqlen=max_seqlen,
            # cuDNN's compile bound covers the partial trailing ratio even
            # though cu_comp contains only complete compressed blocks.
            max_compressed_seqlen=(max_seqlen + self.ratio - 1) // self.ratio,
            scale=(self.head_dim * self.n_heads) ** -0.5,
        )
        return (indices, query, key, weights) if return_context else indices

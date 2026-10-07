# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DSv4 attention core and unpadded CSA/HCA attention layer."""

from typing import Optional

import torch

from transformer_engine.pytorch.module import Linear, RMSNorm

from .compressor import _Compressor
from .dsa_cudnn import attention as _attention
from .dsa_rope import _DSv4RotaryEmbedding, apply_rotary
from .indexer import _Indexer

__all__ = ["DSv4Attention", "DSv4HybridAttention"]


class DSv4Attention(torch.nn.Module):
    """Experimental causal joint local + compressed attention for DSv4.

    Parameter-free core: the caller owns projections, compression, normalization,
    RoPE, and the learned sink. Use :mod:`dsa_cudnn` for gated pooling
    and CSA selection. Pass selected indices for CSA; omit them and supply
    ``max_compressed_seqlen`` for HCA.

    Inputs are contiguous CUDA BF16 query [T,64,D], local KV [T,D], compressed
    KV [Tc,D], FP32 sink [64], and CUDA INT32 sequence prefixes [B+1]. D is 512
    or 576; values use the first 512 KV channels. Output is [T,64,512], retaining
    the head axis for DSv4's output unrotation. Prefixes start at zero and end
    at valid row counts; compressed lengths are floor(sequence length / ratio).
    CSA indices are distinct global packed compressed-row IDs, or -1 for padding.

    Only full sequences with positions starting at zero are supported. Local
    keys satisfy max(0,q-window_size+1) <= k <= q. Compressed block j becomes
    visible when (j+1)*ratio <= q+1. All visible local and compressed entries
    share one softmax with the sink; its value contribution is zero.

    Requires SM100 and cuDNN Frontend 1.29.0. Forward and first-order backward
    use cuDNN's DSA namespace. No dropout, arbitrary masks, cache/decode, FP8,
    distributed attention, or higher-order gradients. This module does not use
    DPA backend-selection flags. cuDNN operations are loaded on execution.
    """

    def __init__(self, *, window_size: int, ratio: int):
        super().__init__()
        if window_size <= 0 or ratio <= 0:
            raise ValueError("window_size and ratio must be positive.")
        self.window_size = window_size
        self.ratio = ratio

    def forward(
        self,
        query: torch.Tensor,
        local_kv: torch.Tensor,
        compressed_kv: torch.Tensor,
        sink: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_comp: torch.Tensor,
        *,
        indices: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        max_compressed_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """Evaluate the joint attention, with scale defaulting to D**-0.5."""
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager

        if FP8GlobalStateManager.is_fp8_enabled():
            raise NotImplementedError("DSv4Attention supports BF16 only; disable TE FP8 autocast.")
        return _attention(
            query,
            local_kv,
            compressed_kv,
            sink,
            cu_seqlens,
            cu_seqlens_comp,
            window_size=self.window_size,
            ratio=self.ratio,
            indices=indices,
            scale=scale,
            max_compressed_seqlen=max_compressed_seqlen,
        )


class DSv4HybridAttention(torch.nn.Module):
    """DSv4 CSA/HCA attention from unpadded ``[S,B,D]`` to ``[S,B,D]``.

    This first-pass layer handles full-sequence BF16 attention forward/backward
    on SM100. By default, projections that share an input use one contiguous
    TE Linear weight and split its output before the separate DSv4 operations.
    It uses TE Linear/RMSNorm, the cuDNN DSv4 compressor and selector, and the
    existing parameter-free cuDNN attention core. Sequences in a batch have
    the same length and start at position zero. The model has 64 heads of width
    512; CSA index heads have width 128, matching the current cuDNN kernels.
    The low-level core's 576-wide Q/K mode returns only 512 value channels,
    which has no verified output-unrotation contract here. Cache/decode,
    padding, TP/CP, FP8, sliding-only attention, and the indexer auxiliary
    loss are unsupported. In CSA, top-k is discrete; the indexer projections
    do not receive gradients from the returned hidden states alone.
    """

    def __init__(
        self,
        hidden_size: int,
        q_lora_rank: int,
        *,
        layer_type: str,
        head_dim: int,
        rope_head_dim: int,
        sliding_window: int,
        compression_ratio: int,
        o_groups: int,
        o_lora_rank: int,
        index_n_heads: int = 64,
        index_topk: int = 512,
        rope_theta: float = 160000.0,
        rope: Optional[torch.nn.Module] = None,
        rms_norm_eps: float = 1e-6,
        max_seqlen: Optional[int] = None,
        device: str = "cuda",
        params_dtype: torch.dtype = torch.float32,
        input_format: str = "sbd",
        _fuse_projections: bool = True,
    ):
        super().__init__()
        # Keep the current kernel geometry explicit without exposing untested sizes.
        self.num_heads = 64
        self.index_head_dim = 128
        if layer_type not in ("compressed_sparse_attention", "heavily_compressed_attention"):
            raise ValueError("Only DSv4 CSA and HCA are supported.")
        if input_format not in ("sbd", "bsd"):
            raise ValueError("input_format must be 'sbd' or 'bsd'.")
        if head_dim != 512 or rope_head_dim < 2 or rope_head_dim % 2 or rope_head_dim > head_dim:
            raise ValueError("head_dim must be 512; rope_head_dim must be even and fit.")
        if any(
            v <= 0
            for v in (
                hidden_size,
                q_lora_rank,
                sliding_window,
                compression_ratio,
                o_groups,
                o_lora_rank,
            )
        ):
            raise ValueError("DSv4 dimensions, window size and ratio must be positive.")
        if self.num_heads * head_dim % o_groups:
            raise ValueError("o_groups must divide the attention output channels.")
        is_csa = layer_type == "compressed_sparse_attention"
        if is_csa and (
            index_n_heads not in (32, 64) or index_topk <= 0 or rope_head_dim > self.index_head_dim
        ):
            raise ValueError(
                "CSA requires 32 or 64 index heads, positive top-k, and RoPE width <= 128."
            )

        kw = {"bias": False, "device": device, "params_dtype": params_dtype}
        if _fuse_projections:
            # Query-down and local-KV projections read the same hidden states.
            # Keep slices explicit below while one contiguous weight feeds the GEMM.
            self.q_a_kv_proj = Linear(
                hidden_size,
                q_lora_rank + head_dim,
                **kw,
            )
        else:
            self.q_a_proj = Linear(hidden_size, q_lora_rank, **kw)
        # LayerNormLinear would normalize before projection; DSv4 normalizes after it.
        self.q_a_norm = RMSNorm(q_lora_rank, eps=rms_norm_eps, device=device, dtype=params_dtype)
        self.q_b_proj = Linear(q_lora_rank, self.num_heads * head_dim, **kw)
        if not _fuse_projections:
            self.kv_proj = Linear(hidden_size, head_dim, **kw)
        self.kv_norm = RMSNorm(head_dim, eps=rms_norm_eps, device=device, dtype=params_dtype)
        self.compressor = _Compressor(
            hidden_size,
            head_dim,
            compression_ratio,
            is_csa,
            rms_norm_eps,
            device,
            params_dtype,
            fused=_fuse_projections,
        )
        if is_csa:
            self.indexer = _Indexer(
                hidden_size,
                q_lora_rank,
                self.index_head_dim,
                index_n_heads,
                index_topk,
                compression_ratio,
                rms_norm_eps,
                device,
                params_dtype,
                fused=_fuse_projections,
            )
        # HF/Megatron use one independent block per output group. A normal TE
        # Linear forward mixes groups, so use its weight in the block multiply.
        self.o_a_proj = Linear(self.num_heads * head_dim // o_groups, o_groups * o_lora_rank, **kw)
        self.o_b_proj = Linear(o_groups * o_lora_rank, hidden_size, **kw)
        self.sinks = torch.nn.Parameter(
            torch.zeros(self.num_heads, device=device, dtype=params_dtype)
        )
        self.core_attention = DSv4Attention(window_size=sliding_window, ratio=compression_ratio)
        # A model can supply its RoPE frequencies (for example YaRN) without
        # duplicating that policy here. It must return the same token/compressed
        # (cos, sin) pairs as _DSv4RotaryEmbedding.forward.
        self.rope = (
            rope
            if rope is not None
            else _DSv4RotaryEmbedding(
                compression_ratio, rope_head_dim, rope_theta, device, max_seqlen
            )
        )
        self.hidden_size, self.q_lora_rank = hidden_size, q_lora_rank
        self.fused_projections = _fuse_projections
        self.head_dim, self.rope_head_dim = head_dim, rope_head_dim
        self.rope_theta = rope_theta if rope is None else None
        self.rms_norm_eps = rms_norm_eps
        self.compression_ratio = compression_ratio
        self.o_groups, self.o_lora_rank = o_groups, o_lora_rank
        self.is_csa = is_csa
        self.input_format = input_format

    def forward(self, hidden_states: torch.Tensor, *, return_indexer_context: bool = False):
        """Run attention in the selected format and optionally expose CSA context."""
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states must have the selected format and hidden_size.")
        if return_indexer_context and not self.is_csa:
            raise ValueError("Only CSA has an indexer context.")
        if self.input_format == "sbd":
            # cuDNN consumes batch-major packed rows. B=1 needs only a view;
            # larger batches require a physical repack with the current core.
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        batch, seq, _ = hidden_states.shape
        if return_indexer_context and batch != 1:
            raise ValueError("The CSA indexer context is currently supported for batch=1.")
        n_comp = seq // self.compression_ratio
        if not n_comp:
            raise ValueError("Sequence must contain at least one complete compression window.")
        cu = torch.arange(batch + 1, device=hidden_states.device, dtype=torch.int32) * seq
        cu_comp = torch.arange(batch + 1, device=hidden_states.device, dtype=torch.int32) * n_comp

        token_rope, compressed_rope = self.rope(seq, hidden_states.device)

        if self.fused_projections:
            q_a, local_kv_projected = self.q_a_kv_proj(hidden_states).split(
                (self.q_lora_rank, self.head_dim), dim=-1
            )
        else:
            q_a = self.q_a_proj(hidden_states)
            local_kv_projected = self.kv_proj(hidden_states)
        q_residual = self.q_a_norm(q_a)
        q = self.q_b_proj(q_residual).reshape(batch, seq, self.num_heads, self.head_dim)
        # Query-up norm is unweighted; TE RMSNorm would add a learned scale.
        q = torch.nn.functional.rms_norm(q, (self.head_dim,), eps=self.rms_norm_eps)
        q = (
            apply_rotary(q, *token_rope)
            .reshape(batch * seq, self.num_heads, self.head_dim)
            .contiguous()
        )
        local_kv = apply_rotary(self.kv_norm(local_kv_projected).unsqueeze(2), *token_rope)
        local_kv = local_kv.reshape(batch * seq, self.head_dim).contiguous()
        compressed_kv = apply_rotary(
            self.compressor(hidden_states, cu, cu_comp).reshape(batch, n_comp, 1, self.head_dim),
            *compressed_rope,
        )
        compressed_kv = compressed_kv.reshape(batch * n_comp, self.head_dim).contiguous()

        indices = None
        index_q = index_k = index_w = None
        if self.is_csa:
            # Megatron trains this tower with a separate loss, without sending
            # its gradients into the shared hidden/query projections.
            index_result = self.indexer(
                hidden_states.detach(),
                q_residual.detach(),
                cu,
                cu_comp,
                token_rope,
                compressed_rope,
                return_context=return_indexer_context,
            )
            if return_indexer_context:
                indices, index_q, index_k, index_w = index_result
            else:
                indices = index_result

        output = self.core_attention(
            q,
            local_kv,
            compressed_kv,
            self.sinks.float(),
            cu,
            cu_comp,
            indices=indices,
            max_compressed_seqlen=None if self.is_csa else n_comp,
        ).reshape(batch, seq, self.num_heads, self.head_dim)
        cos, sin = token_rope
        output = apply_rotary(output, cos, -sin).reshape(batch, seq, self.o_groups, -1)
        weight = self.o_a_proj.weight.reshape(self.o_groups, self.o_lora_rank, -1)
        grouped = torch.einsum("bsgd,grd->bsgr", output, weight).flatten(2)
        result = self.o_b_proj(grouped)
        if self.input_format == "sbd":
            result = result.transpose(0, 1).contiguous()
        if return_indexer_context:
            return result, {
                "index_q": index_q,
                "index_k": index_k,
                "index_w": index_w,
                "indices": indices,
                "attn_q": q.detach(),
                "local_kv": local_kv.detach(),
                "compressed_kv": compressed_kv.detach(),
                "sink": self.sinks.float().detach(),
            }
        return result

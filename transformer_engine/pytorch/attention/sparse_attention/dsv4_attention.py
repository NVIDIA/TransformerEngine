# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DSv4 CSA/HCA hidden-state attention layer."""

from typing import Optional

import torch

from transformer_engine.pytorch.module import Linear, RMSNorm

from .compressor import Compressor
from .dsa_cudnn import sparse_attention as _sparse_attention
from .dsa_rope import _apply_rotary_query, _DSv4RotaryEmbedding, apply_rotary
from .indexer import Indexer

__all__ = ["DSv4Attention"]


@torch.compile
def _compiled_q_rms_norm(q: torch.Tensor, eps: float) -> torch.Tensor:
    """Compile DSv4's weightless per-head query normalization."""
    return torch.nn.functional.rms_norm(q, (q.shape[-1],), eps=eps)


def _packed_prefixes(hidden_states, cu_seqlens, max_seqlen, ratio):
    """Build compressed sequence prefixes and output capacity."""
    if not isinstance(cu_seqlens, torch.Tensor):
        raise ValueError("THD requires contiguous INT32 cu_seqlens_q [B+1] on the input device.")
    if (
        cu_seqlens.dtype != torch.int32
        or cu_seqlens.ndim != 1
        or cu_seqlens.numel() < 2
        or not cu_seqlens.is_contiguous()
        or cu_seqlens.device != hidden_states.device
    ):
        raise ValueError("THD requires contiguous INT32 cu_seqlens_q [B+1] on the input device.")
    if not isinstance(max_seqlen, int) or max_seqlen < ratio:
        raise ValueError("THD requires integer max_seqlen_q >= compression_ratio.")
    # The sum of per-sequence floors can be smaller than floor(T / ratio).
    # Allocate that safe capacity without reading CUDA prefix contents on host;
    # cuDNN uses the true device prefix to exclude unused trailing rows.
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    cu_comp = torch.cat((cu_seqlens.new_zeros(1), (lengths // ratio).cumsum(0, dtype=torch.int32)))
    total_comp = hidden_states.shape[0] // ratio
    if not total_comp:
        raise ValueError("Sequences must contain at least one complete compression window.")
    return cu_comp, total_comp, max_seqlen


class DSv4Attention(torch.nn.Module):
    """DSv4 CSA/HCA attention in ``sbhd``, ``bshd``, or packed ``thd`` format.

    Composes TE projections/norms, Compressor, Indexer, and RoPE with cuDNN
    sparse attention. Supports full-sequence BF16 forward/backward on SM100
    with 64 heads of width 512 and CSA index heads of width 128. Shared-input
    projections are fused by default. Each sequence starts at position zero.
    Cache/decode, physical inter-sequence padding, TP/SP/CP, FP8, and sliding-only
    attention are unsupported. The model owns the indexer auxiliary loss;
    discrete top-k selection does not train the indexer from attention output alone.
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
        qkv_format: str = "sbhd",
        _fuse_projections: bool = True,
    ):
        super().__init__()
        # Keep the current kernel geometry explicit without exposing untested sizes.
        self.num_heads = 64
        self.index_head_dim = 128
        if layer_type not in ("compressed_sparse_attention", "heavily_compressed_attention"):
            raise ValueError("Only DSv4 CSA and HCA are supported.")
        if qkv_format not in ("sbhd", "bshd", "thd"):
            raise ValueError("qkv_format must be 'sbhd', 'bshd', or 'thd'.")
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
            self.kv_proj = Linear(hidden_size, head_dim, **kw)
        # LayerNormLinear would normalize before projection; DSv4 normalizes after it.
        self.q_a_norm = RMSNorm(q_lora_rank, eps=rms_norm_eps, device=device, dtype=params_dtype)
        self.q_b_proj = Linear(q_lora_rank, self.num_heads * head_dim, **kw)
        self.kv_norm = RMSNorm(head_dim, eps=rms_norm_eps, device=device, dtype=params_dtype)
        self.compressor = Compressor(
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
            self.indexer = Indexer(
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
        self.sliding_window = sliding_window
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
        self.qkv_format = qkv_format

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        return_indexer_context: bool = False,
    ):
        """Run full self-attention and optionally expose single-sequence CSA context.

        Packed calls require contiguous INT32 ``cu_seqlens_q`` on the input
        device and ``max_seqlen_q``. Prefixes have no padding/global offsets;
        their contents are caller-owned and are not checked on the host.
        These metadata arguments are accepted only for ``thd``.
        """
        packed = self.qkv_format == "thd"
        if (
            hidden_states.ndim != (2 if packed else 3)
            or hidden_states.shape[-1] != self.hidden_size
        ):
            raise ValueError("hidden_states must have the selected format and hidden_size.")
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager

        if FP8GlobalStateManager.is_fp8_enabled():
            raise NotImplementedError("DSv4Attention supports BF16 only; disable TE FP8 autocast.")
        if return_indexer_context and not self.is_csa:
            raise ValueError("Only CSA has an indexer context.")
        if self.qkv_format == "sbhd":
            # cuDNN consumes batch-major packed rows. B=1 needs only a view;
            # larger batches require a physical repack with the current core.
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        if packed:
            cu = cu_seqlens_q
            cu_comp, total_comp, seq = _packed_prefixes(
                hidden_states, cu, max_seqlen_q, self.compression_ratio
            )
            batch = cu.numel() - 1
            comp_shape = (total_comp,)
        else:
            if cu_seqlens_q is not None or max_seqlen_q is not None:
                raise ValueError("cu_seqlens_q and max_seqlen_q are only supported with THD input.")
            batch, seq, _ = hidden_states.shape
            n_comp = seq // self.compression_ratio
            if not n_comp:
                raise ValueError("Sequence must contain at least one complete compression window.")
            cu = torch.arange(batch + 1, device=hidden_states.device, dtype=torch.int32) * seq
            cu_comp = (
                torch.arange(batch + 1, device=hidden_states.device, dtype=torch.int32) * n_comp
            )
            total_comp = batch * n_comp
            comp_shape = (batch, n_comp)
        if return_indexer_context and batch != 1:
            raise ValueError("The CSA indexer context is currently supported for batch=1.")
        token_shape = hidden_states.shape[:-1]

        token_rope, compressed_rope = self.rope(seq, hidden_states.device)

        if self.fused_projections:
            q_a, local_kv_projected = self.q_a_kv_proj(hidden_states).split(
                (self.q_lora_rank, self.head_dim), dim=-1
            )
        else:
            q_a = self.q_a_proj(hidden_states)
            local_kv_projected = self.kv_proj(hidden_states)
        q_residual = self.q_a_norm(q_a)
        q = self.q_b_proj(q_residual).reshape(*token_shape, self.num_heads, self.head_dim)
        # Query-up norm is unweighted; TE RMSNorm would add a learned scale.
        q = _compiled_q_rms_norm(q, self.rms_norm_eps)
        q = (
            _apply_rotary_query(q, *token_rope, cu)
            .reshape(-1, self.num_heads, self.head_dim)
            .contiguous()
        )
        local_kv = apply_rotary(self.kv_norm(local_kv_projected).unsqueeze(-2), *token_rope, cu)
        local_kv = local_kv.reshape(-1, self.head_dim).contiguous()
        compressed_kv = apply_rotary(
            self.compressor(hidden_states, cu, cu_comp, total_comp=total_comp).reshape(
                *comp_shape, 1, self.head_dim
            ),
            *compressed_rope,
            cu_comp,
        )
        compressed_kv = compressed_kv.reshape(total_comp, self.head_dim).contiguous()

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
                total_comp=total_comp,
                max_seqlen=seq,
                return_context=return_indexer_context,
            )
            if return_indexer_context:
                indices, index_q, index_k, index_w = index_result
            else:
                indices = index_result

        output = _sparse_attention(
            q,
            local_kv,
            compressed_kv,
            self.sinks.float(),
            cu,
            cu_comp,
            window_size=self.sliding_window,
            ratio=self.compression_ratio,
            indices=indices,
            max_compressed_seqlen=None if self.is_csa else seq // self.compression_ratio,
        ).reshape(*token_shape, self.num_heads, self.head_dim)
        cos, sin = token_rope
        output = apply_rotary(output, cos, -sin, cu).reshape(*token_shape, self.o_groups, -1)
        weight = self.o_a_proj.weight.reshape(self.o_groups, self.o_lora_rank, -1)
        grouped = torch.einsum("...gd,grd->...gr", output, weight).flatten(-2)
        result = self.o_b_proj(grouped)
        if self.qkv_format == "sbhd":
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

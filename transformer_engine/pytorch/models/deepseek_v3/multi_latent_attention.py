# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-Latent Attention (MLA) block as used in DeepSeekV3."""

from typing import Optional, Union

import torch

from transformer_engine.pytorch.module import Linear, LayerNormLinear
from transformer_engine.pytorch.attention import DotProductAttention
from transformer_engine.pytorch.models.deepseek_v3.mla_rope import (
    apply_mla_rope_kv,
    apply_mla_rope_q,
    build_rope_tables,
)

__all__ = ["MultiLatentAttention"]


class MultiLatentAttention(torch.nn.Module):
    """
    Multi-Latent Attention as used in DeepSeekV3.

    Queries and key-values are projected through low-rank latents
    (``q_lora_rank``, ``kv_lora_rank``); RMSNorm on each latent is fused into
    the up-projection (:class:`LayerNormLinear` with RMSNorm). Each query/key
    head is split into a ``qk_nope_head_dim`` part and a ``qk_rope_head_dim``
    part; RoPE is applied only to the rope part, and the key rope part comes
    from a single shared head broadcast to all heads. Attention runs through
    :class:`DotProductAttention` with asymmetric head dims
    ``kv_channels=(qk_nope_head_dim + qk_rope_head_dim, v_head_dim)``, which
    supports the cuDNN fused attention backend.

    RoPE uses the fused MLA kernels from :mod:`.mla_rope` (in-place on the
    query rope slice, single-pass key/value assembly); the rope slice follows
    the HF/Megatron DeepSeekV3 convention (interleaved weights, NeoX output).

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    num_attention_heads : int
                         number of attention heads.
    q_lora_rank : int, default = 1536
                 rank of the query latent.
    kv_lora_rank : int, default = 512
                  rank of the key-value latent.
    qk_nope_head_dim : int, default = 128
                      per-head dim of the non-rotary query/key part.
    qk_rope_head_dim : int, default = 64
                      per-head dim of the rotary query/key part.
    v_head_dim : int, default = 128
                per-head dim of the values.
    attention_dropout : float, default = 0.0
                       dropout probability on attention scores.
    attn_mask_type : str, default = "causal"
                    attention mask type passed to :class:`DotProductAttention`.
    layernorm_epsilon : float, default = 1e-6
                       epsilon of the latent RMSNorms (matches DeepSeekV3).
    rotary_base : float, default = 10000.0
                 RoPE base.
    softmax_scale : float, optional
                   softmax scale; defaults to ``1/sqrt(qk head dim)`` inside
                   :class:`DotProductAttention`.
    qkv_format : str, default = "sbhd"
                layout of the input/output tensors.
    params_dtype : torch.dtype, optional
                  dtype of module parameters.
    tp_group : ProcessGroup, optional
              tensor-parallel process group for the up/output projections.
    tp_size : int, default = 1
             tensor-parallel world size.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        attention_dropout: float = 0.0,
        attn_mask_type: str = "causal",
        layernorm_epsilon: float = 1e-6,
        rotary_base: float = 10000.0,
        softmax_scale: Optional[float] = None,
        qkv_format: str = "sbhd",
        params_dtype: Optional[torch.dtype] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        tp_size: int = 1,
        device: Union[torch.device, str] = "cuda",
    ) -> None:
        super().__init__()

        assert qkv_format in ("sbhd", "bshd"), "MultiLatentAttention supports sbhd/bshd formats."
        assert num_attention_heads % tp_size == 0

        self.qkv_format = qkv_format
        self.num_attention_heads = num_attention_heads
        self.num_attention_heads_per_partition = num_attention_heads // tp_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank

        common = {"bias": False, "params_dtype": params_dtype, "device": device}
        tp = {"tp_group": tp_group, "tp_size": tp_size}

        self.q_down_proj = Linear(hidden_size, q_lora_rank, **common)
        self.q_up_proj = LayerNormLinear(
            q_lora_rank,
            num_attention_heads * self.qk_head_dim,
            normalization="RMSNorm",
            eps=layernorm_epsilon,
            parallel_mode="column" if tp_size > 1 else None,
            **tp,
            **common,
        )
        self.kv_down_proj = Linear(hidden_size, kv_lora_rank + qk_rope_head_dim, **common)
        self.kv_up_proj = LayerNormLinear(
            kv_lora_rank,
            num_attention_heads * (qk_nope_head_dim + v_head_dim),
            normalization="RMSNorm",
            eps=layernorm_epsilon,
            parallel_mode="column" if tp_size > 1 else None,
            **tp,
            **common,
        )
        self.out_proj = Linear(
            num_attention_heads * v_head_dim,
            hidden_size,
            parallel_mode="row" if tp_size > 1 else None,
            **tp,
            **common,
        )

        self.rotary_base = rotary_base
        self._rope_tables: Optional[tuple] = None

        self.core_attention = DotProductAttention(
            num_attention_heads,
            kv_channels=(self.qk_head_dim, v_head_dim),
            attention_dropout=attention_dropout,
            qkv_format=qkv_format,
            attn_mask_type=attn_mask_type,
            softmax_scale=softmax_scale,
            tp_group=tp_group,
            tp_size=tp_size,
        )

    def _rope_tables_for(self, seq_len: int, device: torch.device):
        if self._rope_tables is None or self._rope_tables[0].shape[0] < seq_len:
            self._rope_tables = build_rope_tables(
                seq_len, self.qk_rope_head_dim, base=self.rotary_base, device=device
            )
        cos, sin = self._rope_tables
        return cos[:seq_len], sin[:seq_len]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attn_mask_type: Optional[str] = None,
        checkpoint_core_attention: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : torch.Tensor
                       input of shape ``[sq, b, h]`` (sbhd) or ``[b, sq, h]`` (bshd).
        attention_mask : torch.Tensor, optional
                        boolean mask passed to :class:`DotProductAttention`.
        attn_mask_type : str, optional
                        override of the constructor's mask type.
        checkpoint_core_attention : bool, default = False
                                   checkpoint the core attention computation.
        """
        seq_dim = 0 if self.qkv_format == "sbhd" else 1
        seq_len = hidden_states.shape[seq_dim]
        heads = self.num_attention_heads_per_partition

        q = self.q_up_proj(self.q_down_proj(hidden_states))
        q = q.view(*q.shape[:-1], heads, self.qk_head_dim)

        kv_down = self.kv_down_proj(hidden_states)
        kv_latent, k_pos = torch.split(kv_down, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_up_proj(kv_latent)
        kv = kv.view(*kv.shape[:-1], heads, self.qk_nope_head_dim + self.v_head_dim)

        cos, sin = self._rope_tables_for(seq_len, hidden_states.device)
        q = apply_mla_rope_q(
            q, cos, sin, self.qk_nope_head_dim, self.qk_rope_head_dim, self.qkv_format
        )
        k, v = apply_mla_rope_kv(
            kv,
            k_pos.unsqueeze(-2),
            cos,
            sin,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
            self.qkv_format,
        )

        context = self.core_attention(
            q,
            k,
            v,
            attention_mask=attention_mask,
            qkv_format=self.qkv_format,
            attn_mask_type=attn_mask_type,
            checkpoint_core_attention=checkpoint_core_attention,
        )
        return self.out_proj(context)

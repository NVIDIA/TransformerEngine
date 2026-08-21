# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DeepSeekV3 transformer layer."""

from typing import Optional, Union

import torch

from transformer_engine.pytorch.module import LayerNormMLP, RMSNorm
from transformer_engine.pytorch.models.deepseek_v3.multi_latent_attention import (
    MultiLatentAttention,
)
from transformer_engine.pytorch.models.deepseek_v3.moe import DeepSeekV3MoE

__all__ = ["DeepSeekV3Layer"]


class DeepSeekV3Layer(torch.nn.Module):
    """
    A full DeepSeekV3 transformer layer, analogous to
    :class:`TransformerLayer`: pre-RMSNorm + :class:`MultiLatentAttention`,
    then either a dense SwiGLU MLP (:class:`LayerNormMLP` with RMSNorm, used
    for the first dense layers of DeepSeekV3) or :class:`DeepSeekV3MoE`, each
    with a residual connection.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    num_attention_heads : int
                         number of attention heads.
    ffn_hidden_size : int
                     ffn size of the dense MLP (used when ``num_experts`` is
                     ``None``).
    num_experts : int, optional
                 number of routed experts; ``None`` makes this a dense layer.
    moe_ffn_hidden_size : int, optional
                         ffn size of each routed expert (required with MoE).
    hidden_dropout : float, default = 0.0
                    dropout probability on the residual branches.
    **kwargs
             kwargs common to the submodules (``q_lora_rank``, ``kv_lora_rank``,
             ``qk_nope_head_dim``, ``qk_rope_head_dim``, ``v_head_dim``,
             ``attention_dropout``, ``attn_mask_type``, ``qkv_format``, ``topk``,
             ``num_groups``, ``group_topk``, ``routed_scaling_factor``,
             ``shared_expert_ffn_hidden_size``, EP options, ...), forwarded to
             :class:`MultiLatentAttention` and :class:`DeepSeekV3MoE`.
    """

    _MLA_KWARGS = frozenset(
        {
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "attention_dropout",
            "attn_mask_type",
            "rotary_base",
            "softmax_scale",
            "qkv_format",
            "tp_group",
            "tp_size",
        }
    )
    _MOE_KWARGS = frozenset(
        {
            "topk",
            "num_groups",
            "group_topk",
            "routed_scaling_factor",
            "shared_expert_ffn_hidden_size",
            "expert_bias_update_rate",
            "ep_group",
            "ep_max_tokens_per_rank",
            "ep_recv_capacity_per_rank",
            "ep_alignment",
        }
    )

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        ffn_hidden_size: Optional[int] = None,
        num_experts: Optional[int] = None,
        moe_ffn_hidden_size: Optional[int] = None,
        hidden_dropout: float = 0.0,
        layernorm_epsilon: float = 1e-5,
        params_dtype: Optional[torch.dtype] = None,
        device: Union[torch.device, str] = "cuda",
        **kwargs,
    ) -> None:
        super().__init__()

        unknown = set(kwargs) - self._MLA_KWARGS - self._MOE_KWARGS
        if unknown:
            raise TypeError(f"Unexpected keyword arguments: {sorted(unknown)}")
        mla_kwargs = {k: v for k, v in kwargs.items() if k in self._MLA_KWARGS}
        moe_kwargs = {k: v for k, v in kwargs.items() if k in self._MOE_KWARGS}

        self.hidden_dropout = hidden_dropout

        self.input_layernorm = RMSNorm(
            hidden_size, eps=layernorm_epsilon, device=device, dtype=params_dtype
        )
        self.self_attention = MultiLatentAttention(
            hidden_size,
            num_attention_heads,
            params_dtype=params_dtype,
            device=device,
            **mla_kwargs,
        )

        if num_experts is None:
            assert ffn_hidden_size is not None, "Dense layers require ffn_hidden_size."
            self.pre_mlp_layernorm = None
            self.mlp = LayerNormMLP(
                hidden_size,
                ffn_hidden_size,
                eps=layernorm_epsilon,
                normalization="RMSNorm",
                activation="swiglu",
                bias=False,
                params_dtype=params_dtype,
                device=device,
            )
        else:
            assert moe_ffn_hidden_size is not None, "MoE layers require moe_ffn_hidden_size."
            self.pre_mlp_layernorm = RMSNorm(
                hidden_size, eps=layernorm_epsilon, device=device, dtype=params_dtype
            )
            self.mlp = DeepSeekV3MoE(
                hidden_size,
                moe_ffn_hidden_size,
                num_experts,
                params_dtype=params_dtype,
                device=device,
                **moe_kwargs,
            )

    def _residual_add(self, out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.dropout(out, p=self.hidden_dropout, training=self.training)
        return residual + out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        checkpoint_core_attention: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : torch.Tensor
                       input of shape ``[sq, b, h]`` (sbhd) or ``[b, sq, h]`` (bshd).
        attention_mask : torch.Tensor, optional
                        boolean attention mask.
        checkpoint_core_attention : bool, default = False
                                   checkpoint the core attention computation.
        """
        attention_out = self.self_attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            checkpoint_core_attention=checkpoint_core_attention,
        )
        hidden_states = self._residual_add(attention_out, hidden_states)

        if self.pre_mlp_layernorm is not None:
            mlp_out = self.mlp(self.pre_mlp_layernorm(hidden_states))
        else:
            mlp_out = self.mlp(hidden_states)
        return self._residual_add(mlp_out, hidden_states)

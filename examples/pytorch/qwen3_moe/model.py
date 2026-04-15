"""Qwen3 MoE model implementation using TransformerEngine modules.

Same architecture as HuggingFace Transformers Qwen3MoeForCausalLM, with PyTorch modules replaced by TransformerEngine
equivalents for FP8 training and fused kernels.

TE module mapping (HF -> TE):
    self_attn (full block)        ->  te.MultiheadAttention (fused LN + QKV + QK-norm + RoPE + attn + O)
    post_attn_layernorm (MoE)     ->  te.RMSNorm
    expert MLP (SwiGLU)           ->  te_ops.Sequential(GroupedLinear, SwiGLU, GroupedLinear)
    final norm                    ->  te.RMSNorm
    lm_head                       ->  te.Linear
    RoPE frequencies              ->  te.RotaryPositionEmbedding
"""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.pytorch import attention as te_attention
from transformer_engine.pytorch import ops as te_ops

import config as qwen3_moe_config


def _make_init_fn(std: float) -> Callable[[torch.Tensor], None]:
    """Create a normal-distribution weight initializer for TE modules."""

    def _init(weight: torch.Tensor) -> None:
        nn.init.normal_(weight, mean=0.0, std=std)

    return _init


class Qwen3MoeRouter(nn.Module):
    """Top-k softmax router for MoE expert selection.

    Computes softmax over expert logits, selects the top-k experts per token, and returns outputs in the mask format
    expected by ``te.moe_permute_with_probs`` / ``te.moe_unpermute``.

    Args:
        hidden_size: Dimensionality of input hidden states.
        num_experts: Total number of experts.
        top_k: Number of experts selected per token (top-k).
        norm_topk_prob: Whether to renormalize top-k probabilities to sum to 1.
        initializer_range: Std for normal initialization of the routing weight.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool,
        initializer_range: float,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        nn.init.normal_(self.weight, mean=0.0, std=initializer_range)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute expert routing for a batch of tokens.

        Args:
            hidden_states: ``(num_tokens, hidden_size)``.

        Returns:
            merging_probs: ``(num_tokens, num_experts)`` with top-k entries filled, rest zero.
            routing_map: ``(num_tokens, num_experts)`` int32 mask.
            tokens_per_expert: ``(num_experts,)`` token counts.
            router_logits: ``(num_tokens, num_experts)`` pre-softmax logits.
        """
        router_logits = F.linear(hidden_states, self.weight)  # pylint: disable=not-callable
        probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)

        if self.norm_topk_prob:
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        routing_map = torch.zeros(
            hidden_states.shape[0], self.num_experts, dtype=torch.int32, device=hidden_states.device
        )
        routing_map.scatter_(1, topk_indices, 1)

        merging_probs = torch.zeros_like(probs)
        merging_probs.scatter_(1, topk_indices, topk_probs)

        tokens_per_expert = routing_map.sum(dim=0)
        return merging_probs, routing_map, tokens_per_expert, router_logits


class Qwen3MoeBlock(nn.Module):
    """Mixture-of-Experts feed-forward block with SwiGLU activation.

    Routes tokens to top-k experts via ``Qwen3MoeRouter``, then applies per-expert SwiGLU using
    ``te_ops.Sequential(GroupedLinear, SwiGLU, GroupedLinear)`` for fused batched GEMMs + activation.  Token dispatch
    and combine are handled by ``te.moe_permute_with_probs`` / ``te.moe_unpermute``.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: qwen3_moe_config.Qwen3MoeConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts

        self.router = Qwen3MoeRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            norm_topk_prob=config.norm_topk_prob,
            initializer_range=config.initializer_range,
        )

        self.expert_mlp = te_ops.Sequential(
            te_ops.GroupedLinear(
                config.num_experts, config.hidden_size, 2 * config.moe_intermediate_size, bias=False
            ),
            te_ops.SwiGLU(),
            te_ops.GroupedLinear(
                config.num_experts, config.moe_intermediate_size, config.hidden_size, bias=False
            ),
        )
        init_fn = _make_init_fn(config.initializer_range)
        for param in self.expert_mlp.parameters():
            init_fn(param)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts and apply SwiGLU.

        Args:
            hidden_states: ``(batch, seq_len, hidden_size)``.

        Returns:
            output: ``(batch, seq_len, hidden_size)`` after expert computation.
            router_logits: ``(batch * seq_len, num_experts)`` pre-softmax logits.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        merging_probs, routing_map, tokens_per_expert, router_logits = self.router(hidden_flat)

        num_out_tokens = self.router.top_k * router_logits.shape[0]
        permuted_input, _, row_id_map = te.moe_permute_with_probs(
            hidden_flat, merging_probs, routing_map, num_out_tokens=num_out_tokens
        )

        # Expert computation (fused GroupedLinear -> SwiGLU -> GroupedLinear).
        # Pass tokens_per_expert tensor directly — no .tolist() CPU sync.
        expert_out = self.expert_mlp(permuted_input, tokens_per_expert, tokens_per_expert)

        # Combine: scatter back to original order with probability weighting
        output = te.moe_unpermute(
            expert_out, row_id_map, merging_probs=merging_probs, restore_shape=hidden_flat.shape
        )

        return output.view(batch_size, seq_len, hidden_dim), router_logits


class Qwen3MoeDecoderLayer(nn.Module):
    """Pre-norm decoder layer: fused attention + MoE feed-forward.

    Architecture: ``input_layernorm + MultiheadAttention`` (fused inside TE) followed by
    ``post_attention_layernorm + Qwen3MoeBlock``, both with residual connections.

    Args:
        config: Model configuration.
        layer_idx: Zero-based layer index (passed as ``layer_number=idx+1`` to ``te.MultiheadAttention`` for internal
            bookkeeping).
    """

    def __init__(
        self,
        config: qwen3_moe_config.Qwen3MoeConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        init_fn = _make_init_fn(config.initializer_range)

        self.self_attn = te.MultiheadAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            kv_channels=config.head_dim,
            num_gqa_groups=config.num_key_value_heads,
            attention_dropout=config.attention_dropout,
            layernorm_epsilon=config.rms_norm_eps,
            init_method=init_fn,
            output_layer_init_method=init_fn,
            layer_number=layer_idx + 1,
            attn_mask_type="causal",
            input_layernorm=True,
            normalization="RMSNorm",
            bias=config.attention_bias,
            qkv_format="bshd",
            qk_norm_type="RMSNorm",
            qk_norm_eps=config.rms_norm_eps,
            qk_norm_before_rope=True,
        )

        self.post_attention_layernorm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MoeBlock(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply attention and MoE feed-forward with residual connections.

        Args:
            hidden_states: ``(batch, seq_len, hidden_size)``.
            freqs: Rotary position embedding frequencies from ``te.RotaryPositionEmbedding``.
            attention_mask: Optional ``(batch, seq_len)`` mask (1 = valid).

        Returns:
            hidden_states: ``(batch, seq_len, hidden_size)``.
            router_logits: ``(batch * seq_len, num_experts)`` from the MoE block.
        """
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states, attention_mask=attention_mask, rotary_pos_emb=freqs
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, router_logits


class Qwen3MoeModel(nn.Module):
    """Qwen3 MoE transformer backbone: embedding + decoder stack + final norm.

    Embeds input token IDs, applies rotary position embeddings, runs through ``num_hidden_layers`` decoder layers, and
    applies a final RMSNorm.  Returns hidden states and per-layer router logits (for auxiliary loss).

    Args:
        config: Model configuration.
    """

    def __init__(self, config: qwen3_moe_config.Qwen3MoeConfig) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=config.initializer_range)

        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)]
        )
        self.norm = te.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = te_attention.RotaryPositionEmbedding(
            dim=config.head_dim, rotary_base=config.rope_theta
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_ids: ``(batch, seq_len)`` token IDs.
            attention_mask: Optional ``(batch, seq_len)`` mask (1 = valid).

        Returns:
            hidden_states: ``(batch, seq_len, hidden_size)``.
            all_router_logits: List of per-layer router logit tensors.
        """
        hidden_states = self.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]

        freqs = self.rotary_emb(max_seq_len=seq_len)
        freqs = freqs.to(device=hidden_states.device, dtype=torch.float32)

        all_router_logits: list[torch.Tensor] = []
        for layer in self.layers:
            hidden_states, router_logits = layer(hidden_states, freqs, attention_mask)
            all_router_logits.append(router_logits)

        hidden_states = self.norm(hidden_states)
        return hidden_states, all_router_logits


class Qwen3MoeForCausalLM(nn.Module):
    """Qwen3 MoE causal language model: backbone + LM head.

    Wraps ``Qwen3MoeModel`` and adds a ``te.Linear`` LM head that projects hidden states to vocabulary logits.

    Args:
        config: Model configuration.
    """

    def __init__(self, config: qwen3_moe_config.Qwen3MoeConfig) -> None:
        super().__init__()
        self.model = Qwen3MoeModel(config)

        init_fn = _make_init_fn(config.initializer_range)
        self.lm_head = te.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            init_method=init_fn,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            input_ids: ``(batch, seq_len)`` token IDs.
            attention_mask: Optional ``(batch, seq_len)`` padding mask.

        Returns:
            logits: ``(batch, seq_len, vocab_size)``.
            all_router_logits: Per-layer router logits from MoE layers.
        """
        hidden_states, all_router_logits = self.model(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        return logits, all_router_logits

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import re
import gc
from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer_engine.pytorch as te
from transformer_engine.pytorch import GroupedLinear

import transformers
from transformers.models.mixtral.modeling_mixtral import (
    MixtralModel,
    MixtralForCausalLM,
    MixtralConfig,
)
from transformers.modeling_utils import _add_variant, load_state_dict
from transformers.utils import WEIGHTS_INDEX_NAME
from transformers.utils.hub import get_checkpoint_shard_files


@contextmanager
def replace_moe_block(te_moe_cls):
    """
    Replace `MixtralSparseMoeBlock` with custom `TEMixtralSparseMoeBlock`.
    """
    original_moe_cls = transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock
    transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock = te_moe_cls
    try:
        yield
    finally:
        transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock = original_moe_cls


class TEMixtralSparseMoeBlock(nn.Module):
    """
    Transformer Engine optimized MoE block using GroupedLinear for parallel expert processing.

    Replaces HuggingFace's `MixtralSparseMoeBlock` with two key improvements:
    1. `te.GroupedLinear` processes all experts in a single batched GEMM call instead of
       looping over individual `nn.Linear` layers, reducing kernel launch overhead.
    2. `te.moe_permute` / `te.moe_unpermute` efficiently reorder tokens by expert assignment
       so the batched GEMM sees contiguous data per expert.

    The SwiGLU gate and up projections (w1/w3) are combined into one GroupedLinear layer
    (out_features = 2 * ffn_dim) to match the pattern used by TE's TransformerLayer.

    Args:
        config: MixtralConfig
    """

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # Keep HuggingFace router unchanged — not in the GEMM critical path.
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # Combined gate_proj (w1) + up_proj (w3) for all experts.
        # Weight{i} shape: [2 * ffn_dim, hidden_dim]
        self.experts_gate_up = GroupedLinear(
            num_gemms=self.num_experts,
            in_features=self.hidden_dim,
            out_features=2 * self.ffn_dim,
            bias=False,
        )

        # down_proj (w2) for all experts.
        # Weight{i} shape: [hidden_dim, ffn_dim]
        self.experts_down = GroupedLinear(
            num_gemms=self.num_experts,
            in_features=self.ffn_dim,
            out_features=self.hidden_dim,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, sequence_length, hidden_dim]

        Returns:
            final_hidden_states: [batch_size, sequence_length, hidden_dim]
            router_logits: [batch_size * sequence_length, num_experts]
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        # Flatten to [num_tokens, hidden_dim]
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_states_flat.shape[0]

        # ── Router ──────────────────────────────────────────────────────────
        router_logits = self.gate(hidden_states_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        # ── Permute tokens by expert assignment ─────────────────────────────
        # moe_permute reorders tokens so all tokens for expert-0 are contiguous,
        # followed by all tokens for expert-1, etc.
        permuted_tokens, row_id_map = te.moe_permute(
            hidden_states_flat,
            selected_experts.to(torch.int32),
            num_out_tokens=None,
            max_token_num=num_tokens,
        )

        # ── m_splits: tokens routed to each expert ──────────────────────────
        # selected_experts shape: [num_tokens, top_k].
        # Count the number of (token, top_k_slot) pairs assigned to each expert.
        # Each such pair becomes one row in the permuted tensor, so the sum over
        # all experts equals num_tokens * top_k — the total permuted-tensor rows.
        m_splits = [int((selected_experts == i).sum().item()) for i in range(self.num_experts)]

        # ── Expert computation ──────────────────────────────────────────────
        # Gate + Up projection (combined), then SwiGLU, then Down projection.
        intermediate = self.experts_gate_up(permuted_tokens, m_splits=m_splits)
        gate_proj, up_proj = intermediate.chunk(2, dim=-1)
        intermediate_act = F.silu(gate_proj) * up_proj
        expert_outputs = self.experts_down(intermediate_act, m_splits=m_splits)

        # ── Unpermute and apply routing weights ─────────────────────────────
        final_hidden_states = te.moe_unpermute(
            expert_outputs,
            row_id_map,
            probs=routing_weights,
        )

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class TEMixtralForCausalLM:
    """
    Causal LM created with `MixtralForCausalLM` where every `MixtralSparseMoeBlock`
    is monkey-patched with `TEMixtralSparseMoeBlock` before model initialisation.

    Args:
        config: MixtralConfig
    """

    def __new__(cls, config: MixtralConfig):
        with replace_moe_block(te_moe_cls=TEMixtralSparseMoeBlock):
            mixtral_for_causal_lm = MixtralForCausalLM(config)
        return mixtral_for_causal_lm

    @classmethod
    def from_pretrained_local(cls, pretrained_model_name_or_path, *args, config, **kwargs):
        """
        Custom method adapted from `from_pretrained` in HuggingFace Transformers.

        Loads a sharded Mixtral checkpoint from a local directory and maps the
        weights into the TE-optimised model (including expert weight packing
        into GroupedLinear tensors).
        """
        torch.set_default_dtype(kwargs.get("torch_dtype", torch.bfloat16))

        vanilla_model = cls(config)
        subfolder = ""
        variant = None

        if os.path.isfile(
            os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                _add_variant("model.safetensors.index.json", variant),
            )
        ):
            archive_file = os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                _add_variant("model.safetensors.index.json", variant),
            )
            is_sharded = True
        elif os.path.isfile(
            os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                _add_variant(WEIGHTS_INDEX_NAME, variant),
            )
        ):
            archive_file = os.path.join(
                pretrained_model_name_or_path,
                subfolder,
                _add_variant(WEIGHTS_INDEX_NAME, variant),
            )
            is_sharded = True
        else:
            raise AssertionError("Only sharded PyTorch checkpoint format is supported.")

        resolved_archive_file, _ = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            archive_file,
        )

        if not is_sharded:
            assert not isinstance(resolved_archive_file, list)
            resolved_archive_file = [resolved_archive_file]

        for shard_file in resolved_archive_file:
            state_dict = load_state_dict(shard_file)
            replace_params(state_dict, vanilla_model.state_dict(), config)
            vanilla_model.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()

        return vanilla_model


def replace_params(hf_state_dict, te_state_dict, config):
    """
    Copy HuggingFace Mixtral parameters into the TE model's state dict.

    Non-MoE parameters (attention, norms, embeddings) are passed through
    unchanged via `load_state_dict(..., strict=False)`.  Expert weights are
    packed into the GroupedLinear tensors here.

    HF naming:
        model.layers.{L}.block_sparse_moe.experts.{E}.w1.weight  — gate_proj
        model.layers.{L}.block_sparse_moe.experts.{E}.w3.weight  — up_proj
        model.layers.{L}.block_sparse_moe.experts.{E}.w2.weight  — down_proj

    TE naming (GroupedLinear stores one tensor per expert):
        model.layers.{L}.block_sparse_moe.experts_gate_up.weight{E}
        model.layers.{L}.block_sparse_moe.experts_down.weight{E}
    """
    all_layer_prefixes = set()
    for param_key in hf_state_dict.keys():
        m = re.match(r"model\.layers\.\d+\.", param_key)
        if m is not None:
            all_layer_prefixes.add(m.group())

    for layer_prefix in all_layer_prefixes:
        moe_prefix = layer_prefix + "block_sparse_moe."

        for expert_idx in range(config.num_local_experts):
            hf_expert_prefix = moe_prefix + f"experts.{expert_idx}."
            te_gate_up_key = moe_prefix + f"experts_gate_up.weight{expert_idx}"
            te_down_key = moe_prefix + f"experts_down.weight{expert_idx}"

            # gate_proj (w1) occupies the first ffn_dim rows of the combined weight.
            if hf_expert_prefix + "w1.weight" in hf_state_dict:
                te_state_dict[te_gate_up_key].data[: config.intermediate_size] = hf_state_dict[
                    hf_expert_prefix + "w1.weight"
                ].data

            # up_proj (w3) occupies the second ffn_dim rows.
            if hf_expert_prefix + "w3.weight" in hf_state_dict:
                te_state_dict[te_gate_up_key].data[config.intermediate_size :] = hf_state_dict[
                    hf_expert_prefix + "w3.weight"
                ].data

            # down_proj (w2) maps directly.
            if hf_expert_prefix + "w2.weight" in hf_state_dict:
                te_state_dict[te_down_key].data[:] = hf_state_dict[
                    hf_expert_prefix + "w2.weight"
                ].data

    return all_layer_prefixes

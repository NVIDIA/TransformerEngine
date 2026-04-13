# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import re
import gc
from contextlib import contextmanager
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer_engine.pytorch as te
from transformer_engine.pytorch import GroupedLinear

import transformers
from transformers.models.mixtral.modeling_mixtral import (
    MixtralForCausalLM,
    MixtralConfig,
)
from transformers.modeling_utils import _add_variant, load_state_dict
from transformers.utils import WEIGHTS_INDEX_NAME
from transformers.utils.hub import get_checkpoint_shard_files


@contextmanager
def replace_moe_block(te_moe_cls):
    """Context manager: swap MixtralSparseMoeBlock for TEMixtralSparseMoeBlock."""
    original = transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock
    transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock = te_moe_cls
    try:
        yield
    finally:
        transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock = original


class TEMixtralSparseMoeBlock(nn.Module):
    """TE-optimized MoE block replacing HuggingFace MixtralSparseMoeBlock.

    Key improvements over the HF implementation:
    - ``te.GroupedLinear`` processes all experts in a single batched GEMM instead
      of a Python loop over individual ``nn.Linear`` layers.
    - ``te.moe_permute`` / ``te.moe_unpermute`` reorder tokens by expert assignment
      with zero data copies beyond what cuBLAS needs.
    - ``torch.bincount`` computes per-expert token counts in one GPU kernel with a
      single host transfer (``tolist()``), avoiding the 8 blocking ``.item()`` calls
      that killed performance in the naive loop approach.

    The SwiGLU gate (w1) and up (w3) projections are fused into one GroupedLinear
    (out_features = 2 * ffn_dim) so the two half-results can be split after the GEMM.

    Args:
        config: MixtralConfig
    """

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # Router — not in the GEMM critical path; keep as plain nn.Linear.
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # Combined gate_proj (w1) + up_proj (w3) for all experts.
        # Per-expert weight shape: [2 * ffn_dim, hidden_dim]
        self.experts_gate_up = GroupedLinear(
            num_gemms=self.num_experts,
            in_features=self.hidden_dim,
            out_features=2 * self.ffn_dim,
            bias=False,
        )

        # down_proj (w2) for all experts.
        # Per-expert weight shape: [hidden_dim, ffn_dim]
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
            router_logits:       [batch_size * sequence_length, num_experts]
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)  # [T, H]

        # ── Router ──────────────────────────────────────────────────────────
        router_logits = self.gate(hidden_states_flat)  # [T, num_experts]
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # routing_weights stays float32 — required by moe_unpermute merging_probs.

        # ── Permute tokens by expert assignment ─────────────────────────────
        # moe_permute reorders the flat token tensor so that all tokens assigned
        # to expert-0 appear first, then expert-1, etc., ready for GroupedLinear.
        permuted_tokens, row_id_map = te.moe_permute(
            hidden_states_flat,
            selected_experts.to(torch.int32),
            map_type="index",
        )

        # ── Per-expert token counts (m_splits) ──────────────────────────────
        # bincount runs entirely on GPU; tolist() does ONE host transfer for all
        # experts, versus the previous loop that called .item() once per expert
        # (8 blocking GPU→CPU syncs that wiped out the GroupedLinear speedup).
        m_splits = (
            torch.bincount(selected_experts.reshape(-1), minlength=self.num_experts).int().tolist()
        )

        # ── Expert FFN: gate_up → SwiGLU → down ─────────────────────────────
        gate_up = self.experts_gate_up(permuted_tokens, m_splits=m_splits)
        gate_proj, up_proj = gate_up.chunk(2, dim=-1)
        expert_out = self.experts_down(F.silu(gate_proj) * up_proj, m_splits=m_splits)

        # ── Unpermute and weight-sum over top-k experts ──────────────────────
        final_hidden_states = te.moe_unpermute(
            expert_out,
            row_id_map,
            merging_probs=routing_weights,
            map_type="index",
        )

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits


class TEMixtralForCausalLM:
    """Factory that builds a MixtralForCausalLM with every MoE block replaced by
    TEMixtralSparseMoeBlock.

    Usage — convert an already-loaded HF model (works with device_map="auto")::

        from transformers import AutoModelForCausalLM
        from te_mixtral import TEMixtralForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mixtral-8x7B-v0.1",
            torch_dtype=torch.bfloat16,
            device_map="auto",   # distributes across 2x H100
        )
        te_model = TEMixtralForCausalLM.from_hf_model(hf_model)
    """

    def __new__(cls, config: MixtralConfig):
        with replace_moe_block(TEMixtralSparseMoeBlock):
            return MixtralForCausalLM(config)

    @classmethod
    def from_hf_model(cls, hf_model: MixtralForCausalLM) -> MixtralForCausalLM:
        """Convert an in-memory HuggingFace Mixtral model to use TE GroupedLinear experts.

        Compatible with models loaded via ``device_map="auto"`` (multi-GPU).  Each MoE
        block is rebuilt on the same device as the original block's gate weight.

        Args:
            hf_model: Loaded ``MixtralForCausalLM``.

        Returns:
            A ``MixtralForCausalLM`` whose MoE blocks are ``TEMixtralSparseMoeBlock``
            instances with packed GroupedLinear weights.
        """
        config = hf_model.config
        dtype = next(hf_model.parameters()).dtype
        torch.set_default_dtype(dtype)

        # Build the TE model on CPU; weights are placed correctly below.
        te_model = cls(config)

        # Populate the full TE state dict from HF weights, then load in one shot.
        te_state = te_model.state_dict()
        hf_state = hf_model.state_dict()  # pulls all params to CPU
        _pack_expert_weights(hf_state, te_state, config)
        te_model.load_state_dict(te_state)

        # Mirror the device placement of the original model.
        # For device_map="auto", hf_model.hf_device_map maps submodule names to devices.
        device_map = getattr(hf_model, "hf_device_map", None)
        if device_map:
            # Replicate the same device map on the TE model.
            from accelerate import dispatch_model

            te_model = dispatch_model(te_model, device_map=device_map)
        else:
            te_model.to(next(hf_model.parameters()).device)

        return te_model

    @classmethod
    def from_pretrained_local(
        cls,
        pretrained_model_name_or_path: str,
        *args,
        config: MixtralConfig,
        **kwargs,
    ) -> MixtralForCausalLM:
        """Load a sharded local Mixtral checkpoint directly into a TE model.

        Iterates over checkpoint shards one at a time (memory-efficient) and packs
        expert weights into GroupedLinear tensors as each shard is processed.

        Args:
            pretrained_model_name_or_path: Path to the directory containing the
                sharded checkpoint (``model.safetensors.index.json`` or
                ``pytorch_model.bin.index.json``).
            config: ``MixtralConfig`` for the model.
            **kwargs: Forwarded to HF helpers; ``torch_dtype`` is respected.

        Returns:
            A TE-optimised ``MixtralForCausalLM``.
        """
        torch.set_default_dtype(kwargs.get("torch_dtype", torch.bfloat16))

        te_model = cls(config)
        te_state = te_model.state_dict()  # template — filled shard by shard

        subfolder, variant = "", None

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
        else:
            raise AssertionError("Only sharded checkpoint format is supported.")

        resolved_archive_file, _ = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            archive_file,
        )

        for shard_file in resolved_archive_file:
            hf_shard = load_state_dict(shard_file)
            _pack_expert_weights(hf_shard, te_state, config)
            del hf_shard
            gc.collect()

        te_model.load_state_dict(te_state)
        return te_model


def _pack_expert_weights(hf_state_dict: dict, te_state_dict: dict, config: MixtralConfig) -> None:
    """Pack HuggingFace per-expert weights into TE GroupedLinear tensors.

    Modifies ``te_state_dict`` in-place using ``.copy_()`` so the result is
    correct regardless of whether ``state_dict()`` returns shared-storage views
    or independent copies (the behaviour changed across PyTorch versions).

    HF naming::

        model.layers.{L}.block_sparse_moe.experts.{E}.w1.weight  # gate_proj
        model.layers.{L}.block_sparse_moe.experts.{E}.w3.weight  # up_proj
        model.layers.{L}.block_sparse_moe.experts.{E}.w2.weight  # down_proj

    TE naming (GroupedLinear, one tensor per expert)::

        model.layers.{L}.block_sparse_moe.experts_gate_up.weight{E}  # w1 || w3
        model.layers.{L}.block_sparse_moe.experts_down.weight{E}     # w2

    All non-expert parameters present in ``hf_state_dict`` that share a name
    with a key in ``te_state_dict`` are also copied (attention, norms,
    embeddings, router gate).
    """
    expert_keys_consumed: set = set()

    layer_prefixes: set = set()
    for key in hf_state_dict:
        m = re.match(r"model\.layers\.\d+\.", key)
        if m:
            layer_prefixes.add(m.group())

    for layer_prefix in layer_prefixes:
        moe_prefix = layer_prefix + "block_sparse_moe."

        for e in range(config.num_local_experts):
            hf_pfx = moe_prefix + f"experts.{e}."
            te_gu = moe_prefix + f"experts_gate_up.weight{e}"
            te_dn = moe_prefix + f"experts_down.weight{e}"

            w1 = hf_pfx + "w1.weight"
            w3 = hf_pfx + "w3.weight"
            w2 = hf_pfx + "w2.weight"

            if w1 in hf_state_dict:
                te_state_dict[te_gu][: config.intermediate_size].copy_(hf_state_dict[w1])
                expert_keys_consumed.add(w1)
            if w3 in hf_state_dict:
                te_state_dict[te_gu][config.intermediate_size :].copy_(hf_state_dict[w3])
                expert_keys_consumed.add(w3)
            if w2 in hf_state_dict:
                te_state_dict[te_dn].copy_(hf_state_dict[w2])
                expert_keys_consumed.add(w2)

    # Pass through all remaining params that exist in both dicts (attention, norms, etc.).
    for key, tensor in hf_state_dict.items():
        if key not in expert_keys_consumed and key in te_state_dict:
            te_state_dict[key].copy_(tensor)

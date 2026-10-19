# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""HuggingFace Mixtral -> Transformer Engine state-dict mapping.

Two top-level entry points share the same top-level / attention / layernorm
plumbing and only differ in how they place the expert MoE weights:

  * :func:`replace_params_bf16` — used by ``te_mixtral.py`` (Improvements 1/2,
    BF16). Expert weights land in stacked ``mlp.experts_{gate_up,down}_weight``
    parameters (loop path) and/or per-expert ``mlp.experts_{gate_up,down}.weight{i}``
    parameters (Sequential-Op ``GroupedLinear``).

  * :func:`replace_params_mxfp8` — used by ``te_mixtral_mxfp8.py`` (Improvement 3,
    MXFP8). Expert gate (``w1``) and up (``w3``) rows are row-interleaved in
    blocks of 32 to match the GLU layout the fused MXFP8 grouped-MLP kernel
    expects.
"""

from __future__ import annotations

import re

import torch
import torch.distributed as dist
from transformers import MixtralConfig


# Block size for the gate/up interleaved layout. Must match the
# ``glu_interleave_size`` configured on ``ScaledSwiGLU`` in the MXFP8 MoE
# block (see ``te_mixtral_mxfp8.py``).
GLU_INTERLEAVE_SIZE = 32


# ---------------------------------------------------------------------------
# Low-level copy helpers
# ---------------------------------------------------------------------------


def _copy_param(target: torch.Tensor, source: torch.Tensor) -> None:
    """Copy ``source`` into ``target`` preserving the target's dtype/device."""
    target.copy_(source.to(device=target.device, dtype=target.dtype))


def _copy_qkv_proj_to_fused(
    fused_qkv: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_kind: str,
    config: MixtralConfig,
) -> None:
    """Copy one HF Q/K/V projection into the TE fused QKV layout.

    TE interleaves the heads as ``[Q_g_0, ..., Q_g_{h-1}, K_g, V_g]`` per
    KV group ``g``; HF stores Q/K/V as separate projections.
    """
    head_num = config.num_attention_heads
    num_query_groups = config.num_key_value_heads
    heads_per_group = head_num // num_query_groups
    hidden_size = config.hidden_size
    head_size = hidden_size // head_num
    qkv_total_dim = head_num + 2 * num_query_groups

    fused_view = fused_qkv.view(qkv_total_dim, head_size, hidden_size)
    proj_weight = proj_weight.to(device=fused_view.device, dtype=fused_view.dtype)

    if proj_kind == "q":
        q_view = proj_weight.view(head_num, head_size, hidden_size)
        for i in range(num_query_groups):
            start = (heads_per_group + 2) * i
            end = start + heads_per_group
            fused_view[start:end].copy_(q_view[i * heads_per_group : (i + 1) * heads_per_group])
    elif proj_kind == "k":
        k_view = proj_weight.view(num_query_groups, head_size, hidden_size)
        for i in range(num_query_groups):
            fused_view[(heads_per_group + 2) * i + heads_per_group].copy_(k_view[i])
    elif proj_kind == "v":
        v_view = proj_weight.view(num_query_groups, head_size, hidden_size)
        for i in range(num_query_groups):
            fused_view[(heads_per_group + 2) * i + heads_per_group + 1].copy_(v_view[i])
    else:
        raise ValueError(f"Unsupported proj_kind: {proj_kind}")


def _interleave_gate_up(
    gate: torch.Tensor,
    up: torch.Tensor,
    interleave: int = GLU_INTERLEAVE_SIZE,
) -> torch.Tensor:
    """Interleave HF gate (``w1``) and up (``w3``) rows in blocks of ``interleave``.

    HF stacks gate-then-up along the output dim (``[I gate rows; I up rows]``);
    the fused MXFP8 kernel reads gate_up's output in the GLU-interleaved layout
    ``[B gate; B up; B gate; B up; ...]`` with ``B = interleave``.
    """
    intermediate_size, hidden = gate.shape
    if up.shape != (intermediate_size, hidden):
        raise ValueError(f"gate and up shape mismatch: {gate.shape} vs {up.shape}")
    if intermediate_size % interleave != 0:
        raise ValueError(f"intermediate_size {intermediate_size} must be divisible by {interleave}")
    g = gate.reshape(intermediate_size // interleave, interleave, hidden)
    u = up.reshape(intermediate_size // interleave, interleave, hidden)
    stacked = torch.stack([g, u], dim=1)  # [I/B, 2, B, H]
    return stacked.reshape(2 * intermediate_size, hidden).contiguous()


# ---------------------------------------------------------------------------
# Shared per-layer plumbing (top-level, attention, router gate)
# ---------------------------------------------------------------------------


def _ep_rank_from_config(config: MixtralConfig) -> tuple[int, int]:
    """Return (ep_size, ep_rank). EP rank is global_rank % ep_size."""
    ep_size = int(getattr(config, "expert_parallel_size", 1))
    world_rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    ep_rank = world_rank % ep_size if ep_size > 1 else 0
    return ep_size, ep_rank


def _collect_layer_prefixes(hf_state_dict: dict) -> set[str]:
    prefixes = set()
    for key in hf_state_dict.keys():
        m = re.match(r"model\.layers\.\d+\.", key)
        if m is not None:
            prefixes.add(m.group())
    return prefixes


def _copy_top_level(hf_state_dict: dict, te_state_dict: dict) -> None:
    direct = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
        "model.rotary_emb.inv_freq": "model.rotary_emb.inv_freq",
    }
    for hf_key, te_key in direct.items():
        if hf_key in hf_state_dict and te_key in te_state_dict:
            _copy_param(te_state_dict[te_key], hf_state_dict[hf_key])


def _copy_attention_and_layernorms(
    hf_state_dict: dict, te_state_dict: dict, layer_prefix: str, config: MixtralConfig
) -> None:
    direct = {
        layer_prefix
        + "input_layernorm.weight": layer_prefix
        + "self_attention.layernorm_qkv.layer_norm_weight",
        layer_prefix + "self_attn.o_proj.weight": layer_prefix + "self_attention.proj.weight",
        layer_prefix
        + "post_attention_layernorm.weight": layer_prefix
        + "post_attention_layernorm.weight",
    }
    for hf_key, te_key in direct.items():
        if hf_key in hf_state_dict and te_key in te_state_dict:
            _copy_param(te_state_dict[te_key], hf_state_dict[hf_key])

    fused_qkv_key = layer_prefix + "self_attention.layernorm_qkv.weight"
    if fused_qkv_key in te_state_dict:
        qkv_sources = {
            "q": layer_prefix + "self_attn.q_proj.weight",
            "k": layer_prefix + "self_attn.k_proj.weight",
            "v": layer_prefix + "self_attn.v_proj.weight",
        }
        for proj_kind, hf_key in qkv_sources.items():
            if hf_key in hf_state_dict:
                _copy_qkv_proj_to_fused(
                    te_state_dict[fused_qkv_key], hf_state_dict[hf_key], proj_kind, config
                )


def _copy_router_gate(hf_state_dict: dict, te_state_dict: dict, layer_prefix: str) -> None:
    candidates = (
        layer_prefix + "mlp.gate.weight",
        layer_prefix + "block_sparse_moe.gate.weight",
    )
    te_gate_key = layer_prefix + "mlp.gate.weight"
    for hf_key in candidates:
        if hf_key in hf_state_dict and te_gate_key in te_state_dict:
            _copy_param(te_state_dict[te_gate_key], hf_state_dict[hf_key])
            return


def _packed_expert_candidates(layer_prefix: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    gate_up = (
        layer_prefix + "mlp.experts.gate_up_proj",
        layer_prefix + "block_sparse_moe.experts.gate_up_proj",
    )
    down = (
        layer_prefix + "mlp.experts.down_proj",
        layer_prefix + "block_sparse_moe.experts.down_proj",
    )
    return gate_up, down


def _sequential_op_keys(te_state_dict: dict, layer_prefix: str) -> tuple[list[str], list[str]]:
    """Return sorted lists of TE Sequential-Op ``weight{i}`` keys, if present."""
    gate_up_prefix = layer_prefix + "mlp.experts_gate_up."
    down_prefix = layer_prefix + "mlp.experts_down."

    def _weight_index(key: str) -> int:
        match = re.search(r"weight(\d+)$", key)
        assert match is not None
        return int(match.group(1))

    gate_up_keys = sorted(
        (k for k in te_state_dict if k.startswith(gate_up_prefix) and re.search(r"weight\d+$", k)),
        key=_weight_index,
    )
    down_keys = sorted(
        (k for k in te_state_dict if k.startswith(down_prefix) and re.search(r"weight\d+$", k)),
        key=_weight_index,
    )
    return gate_up_keys, down_keys


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def replace_params_bf16(
    hf_state_dict: dict, te_state_dict: dict, config: MixtralConfig
) -> set[str]:
    """Map HF Mixtral weights into the BF16 TE state dict.

    Expert weights are placed in stacked ``mlp.experts_{gate_up,down}_weight``
    parameters (loop path) and/or per-expert Sequential-Op ``weight{i}``
    parameters (grouped_op path). Both formats are written so a single
    checkpoint loader supports either ``expert_ffn_mode``.

    Supports both packed HF MoE tensors (``mlp.experts.gate_up_proj``) and
    older per-expert tensors (``experts.{i}.w{1,2,3}.weight``).
    """
    ep_size, ep_rank = _ep_rank_from_config(config)
    layer_prefixes = _collect_layer_prefixes(hf_state_dict)
    _copy_top_level(hf_state_dict, te_state_dict)

    for layer_prefix in layer_prefixes:
        _copy_attention_and_layernorms(hf_state_dict, te_state_dict, layer_prefix, config)
        _copy_router_gate(hf_state_dict, te_state_dict, layer_prefix)

        packed_gate_up_candidates, packed_down_candidates = _packed_expert_candidates(layer_prefix)
        te_gate_up_key = layer_prefix + "mlp.experts_gate_up_weight"
        te_down_key = layer_prefix + "mlp.experts_down_weight"

        # Path A: stacked param (loop path) <- packed HF tensor.
        for hf_key in packed_gate_up_candidates:
            if hf_key in hf_state_dict and te_gate_up_key in te_state_dict:
                te_gate_up = te_state_dict[te_gate_up_key]
                local_experts = te_gate_up.shape[0]
                expert_start = ep_rank * local_experts if ep_size > 1 else 0
                expert_end = expert_start + local_experts
                _copy_param(
                    te_state_dict[te_gate_up_key],
                    hf_state_dict[hf_key][expert_start:expert_end],
                )
                break
        for hf_key in packed_down_candidates:
            if hf_key in hf_state_dict and te_down_key in te_state_dict:
                te_down = te_state_dict[te_down_key]
                local_experts = te_down.shape[0]
                expert_start = ep_rank * local_experts if ep_size > 1 else 0
                expert_end = expert_start + local_experts
                _copy_param(
                    te_state_dict[te_down_key],
                    hf_state_dict[hf_key][expert_start:expert_end],
                )
                break

        # Path B: Sequential-Op per-expert params <- packed HF tensor.
        te_gate_up_op_keys, te_down_op_keys = _sequential_op_keys(te_state_dict, layer_prefix)
        if te_gate_up_op_keys and te_down_op_keys:
            num_local_experts = len(te_gate_up_op_keys)
            expert_start = ep_rank * num_local_experts if ep_size > 1 else 0
            expert_end = expert_start + num_local_experts
            for hf_key in packed_gate_up_candidates:
                if hf_key in hf_state_dict:
                    hf_gate_up = hf_state_dict[hf_key][expert_start:expert_end]
                    for expert_idx, te_key in enumerate(te_gate_up_op_keys):
                        _copy_param(te_state_dict[te_key], hf_gate_up[expert_idx])
                    break
            for hf_key in packed_down_candidates:
                if hf_key in hf_state_dict:
                    hf_down = hf_state_dict[hf_key][expert_start:expert_end]
                    for expert_idx, te_key in enumerate(te_down_op_keys):
                        _copy_param(te_state_dict[te_key], hf_down[expert_idx])
                    break

        # Path C: older HF format with per-expert w1/w2/w3 -> stacked params.
        if te_gate_up_key in te_state_dict and te_down_key in te_state_dict:
            te_gate_up = te_state_dict[te_gate_up_key]
            te_down = te_state_dict[te_down_key]
            num_local_experts = te_gate_up.shape[0]
            expert_start = ep_rank * num_local_experts if ep_size > 1 else 0
            for expert_idx in range(num_local_experts):
                global_expert_idx = expert_start + expert_idx
                for expert_prefix in (
                    layer_prefix + f"mlp.experts.{global_expert_idx}.",
                    layer_prefix + f"block_sparse_moe.experts.{global_expert_idx}.",
                ):
                    w1_key = expert_prefix + "w1.weight"
                    w3_key = expert_prefix + "w3.weight"
                    w2_key = expert_prefix + "w2.weight"
                    if w1_key in hf_state_dict:
                        te_gate_up[expert_idx, : config.intermediate_size].copy_(
                            hf_state_dict[w1_key].to(
                                device=te_gate_up.device, dtype=te_gate_up.dtype
                            )
                        )
                    if w3_key in hf_state_dict:
                        te_gate_up[expert_idx, config.intermediate_size :].copy_(
                            hf_state_dict[w3_key].to(
                                device=te_gate_up.device, dtype=te_gate_up.dtype
                            )
                        )
                    if w2_key in hf_state_dict:
                        te_down[expert_idx].copy_(
                            hf_state_dict[w2_key].to(device=te_down.device, dtype=te_down.dtype)
                        )

        # Path D: older HF format -> Sequential-Op per-expert params.
        if te_gate_up_op_keys and te_down_op_keys:
            num_local_experts = len(te_gate_up_op_keys)
            expert_start = ep_rank * num_local_experts if ep_size > 1 else 0
            for expert_idx in range(num_local_experts):
                global_expert_idx = expert_start + expert_idx
                gate_up = te_state_dict[te_gate_up_op_keys[expert_idx]]
                down = te_state_dict[te_down_op_keys[expert_idx]]
                for expert_prefix in (
                    layer_prefix + f"mlp.experts.{global_expert_idx}.",
                    layer_prefix + f"block_sparse_moe.experts.{global_expert_idx}.",
                ):
                    w1_key = expert_prefix + "w1.weight"
                    w3_key = expert_prefix + "w3.weight"
                    w2_key = expert_prefix + "w2.weight"
                    if w1_key in hf_state_dict:
                        gate_up[: config.intermediate_size].copy_(
                            hf_state_dict[w1_key].to(device=gate_up.device, dtype=gate_up.dtype)
                        )
                    if w3_key in hf_state_dict:
                        gate_up[config.intermediate_size :].copy_(
                            hf_state_dict[w3_key].to(device=gate_up.device, dtype=gate_up.dtype)
                        )
                    if w2_key in hf_state_dict:
                        down.copy_(hf_state_dict[w2_key].to(device=down.device, dtype=down.dtype))

    return layer_prefixes


def replace_params_mxfp8(
    hf_state_dict: dict, te_state_dict: dict, config: MixtralConfig
) -> set[str]:
    """Map HF Mixtral weights into the MXFP8 TE state dict.

    Per-expert gate/up rows are interleaved in blocks of 32 to match the GLU
    layout that the fused MXFP8 grouped-MLP kernel reads.

    Supports both packed HF tensors (``mlp.experts.gate_up_proj`` of shape
    ``[E, 2I, H]``) and older per-expert tensors
    (``mlp.experts.{i}.w{1,2,3}.weight``).
    """
    ep_size, ep_rank = _ep_rank_from_config(config)
    layer_prefixes = _collect_layer_prefixes(hf_state_dict)
    _copy_top_level(hf_state_dict, te_state_dict)

    for layer_prefix in layer_prefixes:
        _copy_attention_and_layernorms(hf_state_dict, te_state_dict, layer_prefix, config)
        _copy_router_gate(hf_state_dict, te_state_dict, layer_prefix)

        te_gate_up_op_keys, te_down_op_keys = _sequential_op_keys(te_state_dict, layer_prefix)
        if not (te_gate_up_op_keys and te_down_op_keys):
            continue

        num_local_experts = len(te_gate_up_op_keys)
        expert_start = ep_rank * num_local_experts if ep_size > 1 else 0
        expert_end = expert_start + num_local_experts
        intermediate_size = config.intermediate_size

        packed_gate_up_candidates, packed_down_candidates = _packed_expert_candidates(layer_prefix)

        # Path A: newer HF format with packed gate_up tensor [E, 2I, H].
        packed_gate_up_done = False
        for hf_key in packed_gate_up_candidates:
            if hf_key not in hf_state_dict:
                continue
            hf_gate_up = hf_state_dict[hf_key][expert_start:expert_end]
            for expert_idx, te_key in enumerate(te_gate_up_op_keys):
                gate = hf_gate_up[expert_idx, :intermediate_size]
                up = hf_gate_up[expert_idx, intermediate_size:]
                interleaved = _interleave_gate_up(gate, up, GLU_INTERLEAVE_SIZE)
                _copy_param(te_state_dict[te_key], interleaved)
            packed_gate_up_done = True
            break

        packed_down_done = False
        for hf_key in packed_down_candidates:
            if hf_key not in hf_state_dict:
                continue
            hf_down = hf_state_dict[hf_key][expert_start:expert_end]
            for expert_idx, te_key in enumerate(te_down_op_keys):
                _copy_param(te_state_dict[te_key], hf_down[expert_idx])
            packed_down_done = True
            break

        if packed_gate_up_done and packed_down_done:
            continue

        # Path B: older HF format with per-expert w1/w2/w3 weights.
        for expert_idx in range(num_local_experts):
            global_expert_idx = expert_start + expert_idx
            for expert_prefix in (
                layer_prefix + f"mlp.experts.{global_expert_idx}.",
                layer_prefix + f"block_sparse_moe.experts.{global_expert_idx}.",
            ):
                w1_key = expert_prefix + "w1.weight"
                w3_key = expert_prefix + "w3.weight"
                w2_key = expert_prefix + "w2.weight"
                if w1_key in hf_state_dict and w3_key in hf_state_dict and w2_key in hf_state_dict:
                    gate = hf_state_dict[w1_key]
                    up = hf_state_dict[w3_key]
                    interleaved = _interleave_gate_up(gate, up, GLU_INTERLEAVE_SIZE)
                    _copy_param(te_state_dict[te_gate_up_op_keys[expert_idx]], interleaved)
                    _copy_param(te_state_dict[te_down_op_keys[expert_idx]], hf_state_dict[w2_key])
                    break

    return layer_prefixes

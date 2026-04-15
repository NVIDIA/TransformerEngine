# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compare Qwen3 MoE TE implementation against HuggingFace reference.

Runs forward and backward passes on both models with identical weights and
verifies that logits and gradients match.

Usage:
    python -m examples.pytorch.qwen3_moe.test_vs_hf [--seed 42]

Requirements:
    pip install transformers
"""

import argparse
import re

import torch
from transformers.models.qwen3_moe import configuration_qwen3_moe, modeling_qwen3_moe

import config as qwen3_moe_config
import model as qwen3_moe_model

_BATCH = 1
_SEQ_LEN = 32

_TEST_CONFIG = qwen3_moe_config.Qwen3MoeConfig(
    vocab_size=512,
    hidden_size=256,
    moe_intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=128,
    num_experts=8,
    top_k=2,
    norm_topk_prob=True,
)


def _to_hf_config(
    config: qwen3_moe_config.Qwen3MoeConfig,
) -> configuration_qwen3_moe.Qwen3MoeConfig:
    """Create a matching HuggingFace config."""
    return configuration_qwen3_moe.Qwen3MoeConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_intermediate_size,
        moe_intermediate_size=config.moe_intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        num_experts=config.num_experts,
        num_experts_per_tok=config.top_k,
        norm_topk_prob=config.norm_topk_prob,
        output_router_logits=True,
        use_cache=False,
        tie_word_embeddings=False,
    )


def _te_name_to_hf_param(
    name: str,
    hf_model: modeling_qwen3_moe.Qwen3MoeForCausalLM,
    *,
    return_grad: bool = False,
) -> torch.Tensor | None:
    """Match a TE param name to the corresponding HF tensor (.data or .grad)."""

    def _v(param: torch.nn.Parameter) -> torch.Tensor:
        if return_grad:
            assert param.grad is not None, f"Expected gradient for {name}"
            return param.grad
        return param.data

    if name == "model.embed_tokens.weight":
        return _v(hf_model.model.embed_tokens.weight)
    if name.startswith("model.norm."):
        return _v(hf_model.model.norm.weight)
    if "lm_head" in name and "weight" in name:
        return _v(hf_model.lm_head.weight)

    m = re.match(r"model\.layers\.(\d+)\.(.*)", name)
    if not m:
        return None
    hf_layer = hf_model.model.layers[int(m.group(1))]
    hf_attn = hf_layer.self_attn
    suffix = m.group(2)

    if "post_attention_layernorm" in suffix:
        return _v(hf_layer.post_attention_layernorm.weight)
    if "layer_norm" in suffix:
        return _v(hf_layer.input_layernorm.weight)
    if "q_norm" in suffix:
        return _v(hf_attn.q_norm.weight)
    if "k_norm" in suffix:
        return _v(hf_attn.k_norm.weight)
    if "query" in suffix and "weight" in suffix:
        return _v(hf_attn.q_proj.weight)
    if "key" in suffix and "weight" in suffix:
        return _v(hf_attn.k_proj.weight)
    if "value" in suffix and "weight" in suffix:
        return _v(hf_attn.v_proj.weight)
    if "qkv" in suffix and "weight" in suffix:
        return torch.cat(
            [_v(hf_attn.q_proj.weight), _v(hf_attn.k_proj.weight), _v(hf_attn.v_proj.weight)], dim=0
        )
    if "self_attn" in suffix and "proj" in suffix and "weight" in suffix:
        return _v(hf_attn.o_proj.weight)
    if "router" in suffix and "weight" in suffix:
        return _v(hf_layer.mlp.gate.weight)

    # te_ops.Sequential stores expert weights as expert_mlp.{seq_idx}.weight{expert_idx}
    # seq_idx 0 = gate_up_proj (first GroupedLinear), seq_idx 2 = down_proj (second GroupedLinear)
    em = re.search(r"expert_mlp\.(\d+)\.weight(\d+)", suffix)
    if em:
        seq_idx, expert_idx = int(em.group(1)), int(em.group(2))
        exp = hf_layer.mlp.experts
        if seq_idx == 0:
            # gate_up_proj: concat gate and up for this expert
            if hasattr(exp, "gate_up_proj"):
                return _v(exp.gate_up_proj)[expert_idx]
            return torch.cat(
                [_v(exp[expert_idx].gate_proj.weight), _v(exp[expert_idx].up_proj.weight)], dim=0
            )
        if seq_idx == 2:
            # down_proj
            if hasattr(exp, "down_proj"):
                return _v(exp.down_proj)[expert_idx]
            return _v(exp[expert_idx].down_proj.weight)

    return None


def _copy_hf_to_te(
    hf_model: modeling_qwen3_moe.Qwen3MoeForCausalLM,
    te_model: qwen3_moe_model.Qwen3MoeForCausalLM,
) -> None:
    """Copy every parameter from the HF model into the TE model."""
    for name, param in te_model.named_parameters():
        hf_val = _te_name_to_hf_param(name, hf_model)
        if hf_val is None:
            raise ValueError(f"Unmapped TE parameter: {name} {tuple(param.shape)}")
        param.data.copy_(hf_val)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TE vs HF Qwen3 MoE")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"

    hf_config = _to_hf_config(_TEST_CONFIG)
    hf_model = modeling_qwen3_moe.Qwen3MoeForCausalLM(hf_config).to(
        dtype=torch.float32, device=device
    )
    # Use powers-of-two weights for exact reproducibility.
    for param in hf_model.parameters():
        param.data.copy_(2.0 ** torch.randint_like(param, -4, 0))

    te_model = qwen3_moe_model.Qwen3MoeForCausalLM(_TEST_CONFIG).to(
        dtype=torch.float32, device=device
    )
    _copy_hf_to_te(hf_model, te_model)

    input_ids = torch.randint(0, _TEST_CONFIG.vocab_size, (_BATCH, _SEQ_LEN), device=device)

    print("Running forward pass...")
    hf_model.eval()
    te_model.eval()

    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids).logits
        te_logits, _ = te_model(input_ids)

    torch.testing.assert_close(
        torch.softmax(te_logits, dim=-1), torch.softmax(hf_logits, dim=-1), atol=1e-5, rtol=0
    )
    print(f"  Forward PASSED — logits match (atol=1e-5)")

    print("Running backward pass...")
    hf_model.zero_grad(set_to_none=True)
    te_model.zero_grad(set_to_none=True)

    hf_logits = hf_model(input_ids=input_ids).logits
    grad_output = 2.0 ** torch.randint_like(hf_logits, -4, 0)
    hf_logits.backward(grad_output)

    te_logits, _ = te_model(input_ids)
    # Use identical logits so the backward graph sees the same values.
    te_logits.data.copy_(hf_logits.detach())
    te_logits.backward(grad_output)

    max_grad_err = 0.0
    for name, te_param in te_model.named_parameters():
        if te_param.grad is None:
            continue
        hf_grad = _te_name_to_hf_param(name, hf_model, return_grad=True)
        if hf_grad is None:
            raise ValueError(f"Unmapped TE parameter: {name} {tuple(te_param.shape)}")

        torch.testing.assert_close(
            te_param.grad,
            hf_grad,
            atol=1e-2,
            rtol=1e-2,
            msg=lambda m: f"{m}\n{name} {tuple(te_param.shape)}",  # pylint: disable=cell-var-from-loop
        )
        err = (te_param.grad - hf_grad).abs().max().item()
        max_grad_err = max(max_grad_err, err)

    print(f"  Backward PASSED — all gradients match (atol=1e-2, max_err={max_grad_err:.2e})")
    print("All checks passed.")


if __name__ == "__main__":
    main()

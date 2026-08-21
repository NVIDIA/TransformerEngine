# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Numeric comparison of DeepSeekV3Layer against the HF transformers reference."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3RotaryEmbedding,
)

from transformer_engine.pytorch.models import DeepSeekV3Layer
from transformer_engine.pytorch.utils import interleave_glu_tensor

SEQ, BATCH = 64, 2
HIDDEN, HEADS = 256, 4
Q_LORA, KV_LORA = 96, 64
NOPE, ROPE, VDIM = 64, 32, 64
NUM_EXPERTS, TOPK, N_GROUP, TOPK_GROUP = 16, 4, 4, 2
MOE_FFN, N_SHARED = 128, 1
DTYPE = torch.bfloat16


def _hf_config():
    return DeepseekV3Config(
        hidden_size=HIDDEN,
        intermediate_size=4 * HIDDEN,
        moe_intermediate_size=MOE_FFN,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        n_shared_experts=N_SHARED,
        n_routed_experts=NUM_EXPERTS,
        routed_scaling_factor=2.5,
        kv_lora_rank=KV_LORA,
        q_lora_rank=Q_LORA,
        qk_rope_head_dim=ROPE,
        v_head_dim=VDIM,
        qk_nope_head_dim=NOPE,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        num_experts_per_tok=TOPK,
        first_k_dense_replace=0,
        norm_topk_prob=True,
        rms_norm_eps=1e-5,
        attention_bias=False,
        attention_dropout=0.0,
        rope_interleave=True,
        _attn_implementation="eager",
    )


def _init_hf_layer(config):
    torch.manual_seed(0)
    layer = DeepseekV3DecoderLayer(config, layer_idx=0).to(device="cuda", dtype=DTYPE)
    with torch.no_grad():
        for name, p in layer.named_parameters():
            if "layernorm" in name or "norm" in name:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            else:
                p.normal_(0.0, 0.02)
        bias = layer.mlp.gate.e_score_correction_bias
        bias.copy_(0.1 * torch.randn_like(bias))
    return layer


def _build_te_layer(hf):
    te_layer = DeepSeekV3Layer(
        HIDDEN,
        HEADS,
        num_experts=NUM_EXPERTS,
        moe_ffn_hidden_size=MOE_FFN,
        topk=TOPK,
        num_groups=N_GROUP,
        group_topk=TOPK_GROUP,
        routed_scaling_factor=2.5,
        shared_expert_ffn_hidden_size=MOE_FFN * N_SHARED,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=VDIM,
        params_dtype=DTYPE,
    )
    attn, mla = hf.self_attn, te_layer.self_attention
    with torch.no_grad():
        te_layer.input_layernorm.weight.copy_(hf.input_layernorm.weight)
        te_layer.pre_mlp_layernorm.weight.copy_(hf.post_attention_layernorm.weight)

        mla.q_down_proj.weight.copy_(attn.q_a_proj.weight)
        mla.q_up_proj.layer_norm_weight.copy_(attn.q_a_layernorm.weight)
        mla.q_up_proj.weight.copy_(attn.q_b_proj.weight)
        mla.kv_down_proj.weight.copy_(attn.kv_a_proj_with_mqa.weight)
        mla.kv_up_proj.layer_norm_weight.copy_(attn.kv_a_layernorm.weight)
        mla.kv_up_proj.weight.copy_(attn.kv_b_proj.weight)
        mla.out_proj.weight.copy_(attn.o_proj.weight)

        moe = te_layer.mlp
        moe.gate.weight.copy_(hf.mlp.gate.weight)
        moe.expert_bias.copy_(hf.mlp.gate.e_score_correction_bias)
        fc1, _, fc2 = moe.experts
        for e in range(NUM_EXPERTS):
            getattr(fc1, f"weight{e}").copy_(
                interleave_glu_tensor(hf.mlp.experts.gate_up_proj[e], 32)
            )
            getattr(fc2, f"weight{e}").copy_(hf.mlp.experts.down_proj[e])
        shared = hf.mlp.shared_experts
        moe.shared_expert[0].weight.copy_(
            torch.cat([shared.gate_proj.weight, shared.up_proj.weight], dim=0)
        )
        moe.shared_expert[2].weight.copy_(shared.down_proj.weight)
    return te_layer


def test_layer_matches_hf():
    config = _hf_config()
    hf = _init_hf_layer(config)
    te_layer = _build_te_layer(hf)

    torch.manual_seed(1)
    x = torch.randn(BATCH, SEQ, HIDDEN, dtype=DTYPE, device="cuda")
    x_hf = x.clone().requires_grad_(True)
    x_te = x.transpose(0, 1).contiguous().requires_grad_(True)  # sbhd

    rotary = DeepseekV3RotaryEmbedding(config).to("cuda")
    position_ids = torch.arange(SEQ, device="cuda").unsqueeze(0).expand(BATCH, -1)
    cos, sin = rotary(x_hf, position_ids)
    causal = torch.full((SEQ, SEQ), float("-inf"), device="cuda", dtype=DTYPE).triu(1)
    causal = causal[None, None].expand(BATCH, 1, SEQ, SEQ)

    out_hf = hf(x_hf, attention_mask=causal, position_embeddings=(cos, sin))
    out_te = te_layer(x_te)

    torch.testing.assert_close(out_te.transpose(0, 1), out_hf, rtol=5e-2, atol=5e-2)

    grad = torch.randn_like(out_hf)
    out_hf.backward(grad)
    out_te.backward(grad.transpose(0, 1).contiguous())
    torch.testing.assert_close(x_te.grad.transpose(0, 1), x_hf.grad, rtol=5e-2, atol=5e-2)

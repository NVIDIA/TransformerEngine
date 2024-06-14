# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import torch
from typing import Tuple
from tests.pytorch.fused_attn.test_fused_attn import ModelConfig
from transformer_engine.pytorch.distributed import _set_cuda_rng_state
from transformer_engine.pytorch.attention import DotProductAttention

# Initialize RNG state
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
_cpu_rng_state = torch.get_rng_state()
_cuda_rng_state = torch.cuda.get_rng_state()

_NVTE_DEBUG = int(os.getenv("NVTE_DEBUG", "0"))


def reset_rng_states() -> None:
    """Revert back to initial RNG state"""
    torch.set_rng_state(_cpu_rng_state)
    _set_cuda_rng_state(_cuda_rng_state)


def _run_dot_product_attention(
    dtype: torch.dtype,
    config: ModelConfig,
    qkv_layout: str,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run DotProductAttention module with one forward pass and one backward pass"""

    reset_rng_states()
    seqlens_q = torch.full(
        [config.batch_size], config.max_seqlen_q, dtype=torch.int32, device="cuda"
    )
    seqlens_kv = torch.full(
        [config.batch_size], config.max_seqlen_kv, dtype=torch.int32, device="cuda"
    )
    inp = torch.randn(
        [config.batch_size, config.max_seqlen_q, 3, config.num_heads, config.head_dim],
        dtype=dtype,
        device="cuda",
    )
    q = inp[:, :, 0, :, :]
    k = inp[:, :, 1, :, :]
    v = inp[:, :, 2, :, :]
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    out_grad = torch.randn(
        [config.batch_size, config.max_seqlen_q, config.num_heads * config.head_dim],
        dtype=dtype,
        device="cuda",
    )

    # Create attention mask / bias
    attention_mask = None
    bias = None
    if config.attn_mask_type == "arbitrary":
        attention_mask = torch.randint(
            -10,
            10,
            [config.batch_size, config.num_heads, config.max_seqlen_q, config.max_seqlen_kv],
        ).to(dtype=torch.bool, device="cuda")
    if config.attn_bias_type == "post_scale_bias":
        # convert mask to bias
        attention_mask = torch.randint(
            -10,
            10,
            [config.batch_size, config.num_heads, config.max_seqlen_q, config.max_seqlen_kv],
        ).to(dtype=torch.bool, device="cuda")
        bias = attention_mask.clone()
        neginf = -(2**50) if dtype == torch.bfloat16 else -(2**15)
        bias = torch.where(bias == 0, 0, neginf).to(dtype=dtype, device="cuda")
        bias.requires_grad = False
        attention_mask = None

    block = DotProductAttention(
        config.num_heads,
        config.head_dim,
        num_gqa_groups=config.num_gqa_groups,
        qkv_format="bshd",
        attention_dropout=config.dropout_p,
        sequence_parallel=False,
        tp_size=1,
        get_rng_state_tracker=None,
        tp_group=None,
        layer_number=1,
    ).to(dtype=dtype, device="cuda")

    # Run a forward and backward pass
    out = None
    if config.attn_mask_type == "arbitrary":
        out = block(
            q,
            k,
            v,
            attention_mask=attention_mask,  # attention_mask
            qkv_format="bshd",
            attn_mask_type=config.attn_mask_type,  # 'arbitrary'
            core_attention_bias_type=config.attn_bias_type,  # 'no_bias'
            core_attention_bias=bias,  # None
        )
        out.backward(out_grad)

    if config.attn_bias_type == "post_scale_bias":
        out = block(
            q,
            k,
            v,
            attention_mask=attention_mask,  # None
            qkv_format="bshd",
            attn_mask_type=config.attn_mask_type,  # no_mask
            core_attention_bias_type=config.attn_bias_type,  # 'post_scale_bias'
            core_attention_bias=bias,  # bias
        )
        out.backward(out_grad)

    return out, (q.grad, k.grad, v.grad)


dtype = torch.bfloat16
model_configs = {
    #      test:             b,  h, hg,  d,   sq,  skv,   p,        mask,             bias
    "test_mask": ModelConfig(4, 16, 16, 64, 2048, 2048, 0.0, "arbitrary", "no_bias"),
    "test_bias": ModelConfig(4, 16, 16, 64, 2048, 2048, 0.0, "no_mask", "post_scale_bias"),
}

print("Run with post_scale_bias:")
config = model_configs["test_bias"]
fused_attn_fwd, fused_attn_bwd = _run_dot_product_attention(dtype, config, "bs3hd")

print("Run with arbitrary mask:")
config = model_configs["test_mask"]
unfused_attn_fwd, unfused_attn_bwd = _run_dot_product_attention(dtype, config, "bs3hd")

torch.testing.assert_close(unfused_attn_fwd, fused_attn_fwd, atol=2.5e-2, rtol=2.5e-2)
for i in range(3):
    torch.testing.assert_close(unfused_attn_bwd[i], fused_attn_bwd[i], atol=2.5e-2, rtol=2.5e-2)
print("Test passed!")

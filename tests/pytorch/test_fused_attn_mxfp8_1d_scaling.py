# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# See LICENSE for license information.

"""Regression test for MXFP8 1D scaling V shape in fused attention backward."""

import pytest
import torch
import transformer_engine.pytorch as te


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not te.fp8.FP8GlobalStateManager.is_fp8_available(),
    reason="FP8 not available on this device",
)
@pytest.mark.parametrize("head_dim", [64, 128])
def test_fused_attn_mxfp8_1d_scaling_bwd_v_shape(head_dim):
    """Backward must produce dV with the same shape as V for MXFP8 1D scaling."""
    batch, seqlen, num_heads = 2, 128, 2
    device = "cuda"

    q = torch.randn(
        batch, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn(
        batch, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    v = torch.randn(
        batch, seqlen, num_heads, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True
    )

    attn = te.pytorch.DotProductAttention(
        num_attention_heads=num_heads,
        kv_channels=head_dim,
        qkv_format="bshd",
        attention_type="self",
    ).to(device=device, dtype=torch.bfloat16)

    with te.fp8.fp8_autocast(
        enabled=True,
        fp8_recipe=te.fp8.MXFP8Recipe(scaling_mode=te.fp8.MXFP8ScalingMode.VECTOR_1D),
    ):
        out = attn(q, k, v)

    loss = out.sum()
    loss.backward()


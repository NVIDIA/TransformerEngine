# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import pytest
import torch
from transformer_engine.pytorch.attention import RotaryPositionEmbedding, apply_rotary_pos_emb, fused_apply_rotary_pos_emb

def get_atol(dtype: torch.dtype) -> float:
    if dtype == torch.bfloat16:
        return 1e-2
    elif dtype == torch.float16:
        return 1e-3
    return 1e-6

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_length", [2048, 4096])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("rotary_percent", [0.5, 1.0])
def test_fused_rope(dtype: torch.dtype, seq_length: int, hidden_size: int, rotary_percent: float) -> None:
    device = torch.device("cuda:0")
    batch_size, head_num = 2, 64
    t = torch.rand((seq_length, batch_size, head_num, hidden_size), dtype=dtype, device=device, requires_grad=True)
    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent)
    emb = rotary_pos_emb(seq_length)

    # unfused
    output_unfused = apply_rotary_pos_emb(t, emb)
    output_unfused.sum().backward()
    grad_unfused = t.grad.detach().clone()
    t.grad = None

    # fused
    output_fused = fused_apply_rotary_pos_emb(t, emb)
    output_fused.sum().backward()
    grad_fused = t.grad.detach().clone()

    assert torch.allclose(output_unfused, output_fused, atol=get_atol(dtype))
    assert torch.allclose(grad_unfused, grad_fused, atol=get_atol(dtype))

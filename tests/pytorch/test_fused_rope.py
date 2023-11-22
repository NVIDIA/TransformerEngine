# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import pytest
import torch
from typing import Callable, Tuple, Union
from transformer_engine.pytorch.attention import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
)


def get_atol(dtype: torch.dtype) -> float:
    if dtype == torch.bfloat16:
        return 1e-2
    elif dtype == torch.float16:
        return 1e-3
    return 1e-6


def _overlapping_grad(output: torch.Tensor) -> torch.Tensor:
    return output.sum() * 2


def _non_overlapping_grad(output: torch.Tensor) -> torch.Tensor:
    t = torch.ones_like(output)
    return torch.sum(output * t)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_length", [2048, 4096])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("rotary_percent", [0.5, 1.0])
@pytest.mark.parametrize("transpose", [None, (0, 1), (2, 3)])
@pytest.mark.parametrize("transpose_output_memory", [False, True])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
def test_fused_rope(
    dtype: torch.dtype,
    seq_length: int,
    hidden_size: int,
    rotary_percent: float,
    transpose: Union[Tuple, None],
    transpose_output_memory: bool,
    loss_func: Callable,
) -> None:
    device = torch.device("cuda:0")
    batch_size, head_num = 2, 64
    t = torch.rand(
        (seq_length, batch_size, head_num, hidden_size),
        dtype=dtype,
        device=device,
    )
    if transpose:
        t = t.transpose(*transpose)
        t = t.reshape((seq_length, batch_size, head_num, hidden_size))
    t.requires_grad = True

    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent)
    emb = rotary_pos_emb(seq_length)

    # unfused
    output_unfused = apply_rotary_pos_emb(t, emb, fused=False)
    loss_unfused = loss_func(output_unfused)
    loss_unfused.backward()
    grad_unfused = t.grad.detach().clone()
    t.grad = None

    # fused
    output_fused = apply_rotary_pos_emb(
        t, emb, fused=True, transpose_output_memory=transpose_output_memory
    )
    loss_fused = loss_func(output_fused)
    loss_fused.backward()
    grad_fused = t.grad.detach().clone()
    t.grad = None

    assert torch.allclose(output_unfused, output_fused, atol=get_atol(dtype))
    assert torch.allclose(grad_unfused, grad_fused, atol=get_atol(dtype))
    assert output_fused.transpose(0, 1).is_contiguous() is transpose_output_memory

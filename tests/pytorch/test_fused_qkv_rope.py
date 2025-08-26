# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
from typing import Callable, Tuple, Union
import math
import torch
import pytest
from transformer_engine.pytorch.attention.rope import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
    apply_fused_qkv_rotary_pos_emb,
)


# Gradient is a broadcasted scalar
def _overlapping_grad(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return query.sum() * 2 + key.sum() * 2 + value.sum() * 2


# Gradient is a full tensor
def _non_overlapping_grad(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    t1 = torch.ones_like(query)
    t2 = torch.ones_like(key)
    t3 = torch.ones_like(value)
    return torch.sum(query * t1) + torch.sum(key * t2) + torch.sum(value * t3)


@pytest.mark.parametrize("start_positions", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_length", [2, 8, 2048, 4096])
@pytest.mark.parametrize("hidden_size", [64, 128, 256])
@pytest.mark.parametrize("rotary_percent", [0.5, 1.0])
@pytest.mark.parametrize("margin", [0, 10])
@pytest.mark.parametrize("tensor_format", ["sbhd", "bshd"])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("interleaved", [True, False])
def test_fused_qkv_rope(
    dtype: torch.dtype,
    seq_length: int,
    hidden_size: int,
    rotary_percent: float,
    margin: int,
    tensor_format: str,
    loss_func: Callable,
    cp_size: int,
    interleaved: bool,
    start_positions: bool,
) -> None:
    if margin == 0 and start_positions == True:
        # This makes sure that the `start_positions` offsets being applied
        # are with the maximum length of the rope embeddings.
        pytest.skip("Skipping test with margin=0 and start_positions=True")

    if start_positions == True and cp_size > 1:
        # `start_positions` is only supported for `cp_size=1` and inference.
        pytest.skip("Skipping test with cp_size>1 and start_positions=True")

    if seq_length - margin < 0:
        pytest.skip("Skipping test with seq_length - margin < 0")

    device = torch.device("cuda:0")
    batch_size, head_num = 2, 64

    t = torch.rand(
        (seq_length - margin, batch_size, head_num, hidden_size * 6),
        dtype=dtype,
        device=device,
    )

    # Get arbitrary offsets to be used with RoPE for all the sequences
    start_positions = (
        torch.randint(0, margin, (batch_size,), dtype=torch.int32, device=device)
        if start_positions
        else None
    )

    if tensor_format == "bshd":
        t = t.transpose(0, 1).contiguous()
    t.requires_grad = True

    rotary_pos_emb_q = RotaryPositionEmbedding(hidden_size, rotary_percent, interleaved=interleaved)
    emb_q = rotary_pos_emb_q(seq_length * cp_size)
    rotary_pos_emb_k = RotaryPositionEmbedding(hidden_size, rotary_percent, interleaved=interleaved)
    emb_k = rotary_pos_emb_k(seq_length * cp_size)

    for cp_rank in range(cp_size):
        # unfused
        # The fused kernel computes in float32 internally, so we force the unfused func to use float32
        # for more accurate comparison

        t_clone = t.clone()
        (query, key, value) = torch.split(
            t_clone, [hidden_size * 4, hidden_size, hidden_size], dim=3
        )
        query = query.reshape(query.shape[0], query.shape[1], head_num * 4, hidden_size)

        query_unfused = apply_rotary_pos_emb(
            query,
            emb_q,
            tensor_format=tensor_format,
            start_positions=start_positions,
            interleaved=interleaved,
            fused=True,
            cp_size=cp_size,
            cp_rank=cp_rank,
        ).to(dtype)

        key_unfused = apply_rotary_pos_emb(
            key,
            emb_k,
            tensor_format=tensor_format,
            start_positions=start_positions,
            interleaved=interleaved,
            fused=True,
            cp_size=cp_size,
            cp_rank=cp_rank,
        ).to(dtype)

        value_unfused = value
        loss_unfused = loss_func(query_unfused, key_unfused, value_unfused)

        if not isinstance(start_positions, torch.Tensor):
            loss_unfused.backward()
            grad_unfused = t.grad.detach().clone()

        t.grad = None

        # fused
        query_fused, key_fused, value_fused = apply_fused_qkv_rotary_pos_emb(
            t,
            emb_q,
            emb_k,
            tensor_format=tensor_format,
            start_positions=start_positions,
            interleaved=interleaved,
            cp_size=cp_size,
            cp_rank=cp_rank,
            qkv_split_arg_list=[hidden_size * 4, hidden_size, hidden_size],
        )
        loss_fused = loss_func(query_fused, key_fused, value_fused)

        if not isinstance(start_positions, torch.Tensor):
            loss_fused.backward()
            grad_fused = t.grad.detach().clone()
        t.grad = None

        torch.testing.assert_close(query_fused, query_unfused)
        torch.testing.assert_close(key_fused, key_unfused)
        torch.testing.assert_close(value_fused, value_unfused)

        if not isinstance(start_positions, torch.Tensor):
            torch.testing.assert_close(grad_fused, grad_unfused)

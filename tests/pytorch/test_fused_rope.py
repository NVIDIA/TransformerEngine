# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import math
import pytest
import torch
from typing import Callable, Tuple, Union
from transformer_engine.pytorch.dot_product_attention.rope import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
)


# Gradient is a broadcasted scalar
def _overlapping_grad(output: torch.Tensor) -> torch.Tensor:
    return output.sum() * 2


# Gradient is a full tensor
def _non_overlapping_grad(output: torch.Tensor) -> torch.Tensor:
    t = torch.ones_like(output)
    return torch.sum(output * t)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_length", [2048, 4096])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("rotary_percent", [0.5, 1.0])
@pytest.mark.parametrize("margin", [0, 10])
@pytest.mark.parametrize("transpose", [None, (0, 1), (2, 3)])
@pytest.mark.parametrize("tensor_format", ["sbhd", "bshd"])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("interleaved", [True, False])
def test_fused_rope(
    dtype: torch.dtype,
    seq_length: int,
    hidden_size: int,
    rotary_percent: float,
    margin: int,
    transpose: Union[Tuple, None],
    tensor_format: str,
    loss_func: Callable,
    cp_size: int,
    interleaved: bool,
) -> None:
    device = torch.device("cuda:0")
    batch_size, head_num = 2, 64
    t = torch.rand(
        (seq_length - margin, batch_size, head_num, hidden_size),
        dtype=dtype,
        device=device,
    )
    if tensor_format == "bshd":
        t = t.transpose(0, 1).contiguous()
    if transpose:
        t = t.transpose(*transpose).contiguous().transpose(*transpose)
    t.requires_grad = True

    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent, interleaved=interleaved)
    emb = rotary_pos_emb(seq_length * cp_size)
    assert emb.is_contiguous()

    for cp_rank in range(cp_size):
        # unfused
        # The fused kernel computes in float32 internally, so we force the unfused func to use float32
        # for more accurate comparison
        output_unfused = apply_rotary_pos_emb(
            t.float(),
            emb,
            tensor_format=tensor_format,
            interleaved=interleaved,
            fused=False,
            cp_size=cp_size,
            cp_rank=cp_rank,
        ).to(dtype)
        loss_unfused = loss_func(output_unfused)
        loss_unfused.backward()
        grad_unfused = t.grad.detach().clone()
        t.grad = None

        # fused
        output_fused = apply_rotary_pos_emb(
            t,
            emb,
            tensor_format=tensor_format,
            interleaved=interleaved,
            fused=True,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )
        loss_fused = loss_func(output_fused)
        loss_fused.backward()
        grad_fused = t.grad.detach().clone()
        t.grad = None

        torch.testing.assert_close(output_fused, output_unfused)
        torch.testing.assert_close(grad_fused, grad_unfused)
        assert output_fused.is_contiguous()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("rotary_percent", [0.5, 1.0])
@pytest.mark.parametrize("transpose", [None, (1, 2)])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("interleaved", [True, False])
def test_fused_rope_thd(
    dtype: torch.dtype,
    hidden_size: int,
    rotary_percent: float,
    transpose: Union[Tuple, None],
    loss_func: Callable,
    cp_size: int,
    interleaved: bool,
) -> None:
    device = torch.device("cuda:0")
    batch_size, head_num = 2, 64
    cu_seqlens = [0, 400, 542, 711, 727, 752, 1270, 1426, 1450, 1954, 2044, 2048]
    if cp_size > 1:
        cu_seqlens_padded = [0]
        for i in range(1, len(cu_seqlens)):
            cu_seqlens_padded.append(
                cu_seqlens_padded[i - 1]
                + math.ceil((cu_seqlens[i] - cu_seqlens[i - 1]) / (cp_size * 2)) * (cp_size * 2)
            )
    else:
        cu_seqlens_padded = cu_seqlens
    cu_seqlens_padded = torch.tensor(
        cu_seqlens_padded,
        dtype=torch.int32,
        device=device,
    )
    t = torch.rand(
        (cu_seqlens_padded[-1] // cp_size, head_num, hidden_size),
        dtype=dtype,
        device=device,
    )
    if transpose:
        t = t.transpose(*transpose).contiguous().transpose(*transpose)
    t.requires_grad = True

    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent, interleaved=interleaved)
    emb = rotary_pos_emb(cu_seqlens_padded[-1])
    assert emb.is_contiguous()

    for cp_rank in range(cp_size):
        # unfused
        # The fused kernel computes in float32 internally, so we force the unfused func to use float32
        # for more accurate comparison
        output_unfused = apply_rotary_pos_emb(
            t.float(),
            emb,
            tensor_format="thd",
            interleaved=interleaved,
            fused=False,
            cu_seqlens=cu_seqlens_padded,
            cp_size=cp_size,
            cp_rank=cp_rank,
        ).to(dtype)
        loss_unfused = loss_func(output_unfused)
        loss_unfused.backward()
        grad_unfused = t.grad.detach().clone()
        t.grad = None

        # fused
        output_fused = apply_rotary_pos_emb(
            t,
            emb,
            interleaved=interleaved,
            fused=True,
            tensor_format="thd",
            cu_seqlens=cu_seqlens_padded,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )
        loss_fused = loss_func(output_fused)
        loss_fused.backward()
        grad_fused = t.grad.detach().clone()
        t.grad = None

        torch.testing.assert_close(output_fused, output_unfused)
        torch.testing.assert_close(grad_fused, grad_unfused)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_length", [2048, 4096])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("rotary_percent", [0.5, 1.0])
@pytest.mark.parametrize("margin", [10])
@pytest.mark.parametrize("start_positions", [True, False])
@pytest.mark.parametrize("tensor_format", ["bshd", "sbhd"])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
def test_fused_rope_non_thd_staggered_inputs(
    dtype: torch.dtype,
    tensor_format: str,
    loss_func: Callable,
    start_positions: bool,
    seq_length: int,
    hidden_size: int,
    rotary_percent: float,
    margin: int,
) -> None:
    if margin == 0 and start_positions == True:
        # If sequence to encode has the same lesngth as length of encoding
        # there is no space left for starting with positions >0.
        pytest.skip("Skipping test with margin=0 and start_positions=True")

    device = torch.device("cuda:0")
    batch_size, head_num = 8, 64

    start_positions = (
        torch.randint(0, margin, (batch_size,), dtype=torch.int32, device=device)
        if start_positions
        else None
    )

    running_seq_len = seq_length - margin

    t = torch.rand(
        (running_seq_len - margin, batch_size, head_num, hidden_size),
        dtype=dtype,
        device=device,
    )
    if tensor_format == "bshd":
        t = t.transpose(0, 1).contiguous()
    t.requires_grad = True

    rope_module = RotaryPositionEmbedding(hidden_size, rotary_percent)
    rope_emb = rope_module(seq_length)

    # Apply RoPE with start_positions (unfused)
    #
    # The fused kernel computes in float32 internally, so we force the unfused func to use float32
    # for more accurate comparison
    output_unfused = apply_rotary_pos_emb(
        t.float(),
        rope_emb,
        start_positions=start_positions,
        tensor_format=tensor_format,
        fused=False,
    ).to(dtype)
    loss_unfused = loss_func(output_unfused)
    loss_unfused.backward()
    grad_unfused = t.grad.detach().clone()
    t.grad = None

    # Apply RoPE with start_positions (fused)
    output_fused = apply_rotary_pos_emb(
        t,
        rope_emb,
        start_positions=start_positions,
        tensor_format=tensor_format,
        fused=True,
    )
    loss_fused = loss_func(output_fused)
    loss_fused.backward()
    grad_fused = t.grad.detach().clone()
    t.grad = None

    torch.testing.assert_close(output_fused, output_unfused)
    torch.testing.assert_close(grad_fused, grad_unfused)
    assert output_fused.is_contiguous()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("rotary_percent", [1.0])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
# cp_size is restricted to `1` deliberately
@pytest.mark.parametrize("cp_size", [1])
@pytest.mark.parametrize("margin", [10])
@pytest.mark.parametrize("start_positions", [True, False])
def test_fused_rope_thd_staggered_inputs(
    dtype: torch.dtype,
    hidden_size: int,
    rotary_percent: float,
    loss_func: Callable,
    cp_size: int,
    margin: int,
    start_positions: bool,
) -> None:
    if margin == 0 and start_positions == True:
        # This makes sure that the `start_positions` offsets being applied
        # are with the maximum length of the rope embeddings.
        pytest.skip("Skipping test with margin=0 and start_positions=True")

    if start_positions == True and cp_size > 1:
        # Note: `start_positions` is used only during inference and context
        # parallelism to our best knowledge isn't used during inference. This
        # path hasn't been tested and so skipping it.
        pytest.skip("Skipping test with cp_size>1 and start_positions=True")

    device = torch.device("cuda:0")
    batch_size, head_num = 2, 64
    cu_seqlens = [0, 400, 542, 711, 727, 752, 1270, 1426, 1450, 1954, 2044, 2048]

    cu_seqlens = torch.tensor(
        cu_seqlens,
        dtype=torch.int32,
        device=device,
    )
    start_positions = (
        torch.randint(0, margin, (len(cu_seqlens) - 1,), dtype=torch.int32, device=device)
        if start_positions
        else None
    )
    t = torch.rand(
        (cu_seqlens[-1] // cp_size, head_num, hidden_size),
        dtype=dtype,
        device=device,
    )
    t.requires_grad = True

    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent)
    emb = rotary_pos_emb(cu_seqlens[-1])
    assert emb.is_contiguous()

    # unfused
    # The fused kernel computes in float32 internally, so we force the unfused func to use float32
    # for more accurate comparison
    output_unfused = apply_rotary_pos_emb(
        t.float(),
        emb,
        start_positions=start_positions,
        tensor_format="thd",
        fused=False,
        cu_seqlens=cu_seqlens,
        cp_size=cp_size,  # cp_size is restricted to `1` deliberately
        cp_rank=0,
    ).to(dtype)
    loss_unfused = loss_func(output_unfused)
    loss_unfused.backward()
    grad_unfused = t.grad.detach().clone()
    t.grad = None

    # fused
    output_fused = apply_rotary_pos_emb(
        t,
        emb,
        start_positions=start_positions,
        fused=True,
        tensor_format="thd",
        cu_seqlens=cu_seqlens,
        cp_size=cp_size,  # cp_size is restricted to `1` deliberately
        cp_rank=0,
    )
    loss_fused = loss_func(output_fused)
    loss_fused.backward()
    grad_fused = t.grad.detach().clone()
    t.grad = None

    torch.testing.assert_close(output_fused, output_unfused)
    torch.testing.assert_close(grad_fused, grad_unfused)

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
from typing import Callable, Tuple, Union, List
import math
import torch
import pytest
from transformer_engine.pytorch.attention.rope import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
    apply_fused_qkv_rotary_pos_emb,
)


# Gradient is a broadcasted scalar
def _overlapping_grad(output: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(output, List):
        return sum(t.sum() * 2 for t in output)
    else:
        return output.sum() * 2


# Gradient is a full tensor
def _non_overlapping_grad(output: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(output, List):
        return sum(torch.sum(t * torch.ones_like(t)) for t in output)
    else:
        t = torch.ones_like(output)
        return torch.sum(output * t)


@pytest.mark.parametrize("start_positions", [True, False])
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
    start_positions: bool,
) -> None:
    if margin == 0 and start_positions == True:
        # This makes sure that the `start_positions` offsets being applied
        # are with the maximum length of the rope embeddings.
        pytest.skip("Skipping test with margin=0 and start_positions=True")

    device = torch.device("cuda:0")
    batch_size, head_num = 2, 64
    t = torch.rand(
        (seq_length - margin, batch_size, head_num, hidden_size),
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
            start_positions=start_positions,
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
            start_positions=start_positions,
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


@pytest.mark.parametrize("margin", [10])
@pytest.mark.parametrize("start_positions", [True, False])
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
    start_positions: bool,
    margin: int,
) -> None:
    device = torch.device("cuda:0")
    batch_size, head_num = 2, 64
    cu_seqlens = [0, 400, 542, 711, 727, 752, 1270, 1426, 1450, 1954, 2044, 2048]

    # Get arbitrary offsets to be used with RoPE for all the sequences
    start_positions = (
        torch.randint(0, margin, (len(cu_seqlens) - 1,), dtype=torch.int32, device=device)
        if start_positions
        else None
    )

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
    # A long frequency table with many packed spans exercises the default
    # linear-grid dispatch heuristic without a test-only override.
    emb = rotary_pos_emb(8192)
    assert emb.is_contiguous()

    for cp_rank in range(cp_size):
        # unfused
        # The fused kernel computes in float32 internally, so we force the unfused func to use float32
        # for more accurate comparison
        output_unfused = apply_rotary_pos_emb(
            t.float(),
            emb,
            start_positions=start_positions,
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
            start_positions=start_positions,
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
        assert output_fused.is_contiguous()


@pytest.mark.parametrize("start_positions", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("rotary_percent", [1.0])
@pytest.mark.parametrize("loss_func", [_overlapping_grad])
@pytest.mark.parametrize("cp_size", [2])
@pytest.mark.parametrize("interleaved", [False, True])
def test_unfused_rope_thd_vs_bshd(
    dtype: torch.dtype,
    hidden_size: int,
    rotary_percent: float,
    loss_func: Callable,
    cp_size: int,
    interleaved: bool,
    start_positions: bool,
) -> None:
    """
    This is just a sanity check to ensure that the unfused RoPE in THD/SBHD/BSHD
    formats are the same.
    """
    device = torch.device("cuda:0")
    seqlen, max_seqlen = 16, 2048
    batch_size, head_num = 4, 256

    # NOTE: dtype=torch.int32 is important, otherwise the cumsum will be in int64 and
    # that causes unexpected issues.
    seq_lens = torch.tensor([seqlen for _ in range(batch_size)], dtype=torch.int32)

    cu_seqlens = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int32), seq_lens]), dim=0).to(
        device=device, dtype=torch.int32
    )

    # Create a tensor in THD format
    thd = torch.rand(
        (cu_seqlens[-1] // cp_size, head_num, hidden_size),
        dtype=dtype,
        device=device,
    )
    thd.requires_grad = True

    # Clone the tensor to create a tensor in BSHD format
    bshd = thd.view(batch_size, -1, head_num, hidden_size).clone().detach()
    bshd = bshd.to(dtype=dtype, device=device)
    bshd.requires_grad = True

    # Clone the tensor to create a tensor in SBHD format
    sbhd = bshd.transpose(1, 0).clone().detach()
    sbhd = sbhd.to(dtype=dtype, device=device)
    sbhd.requires_grad = True

    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent, interleaved=interleaved)
    emb = rotary_pos_emb(max_seqlen)
    assert emb.is_contiguous()

    start_positions = cu_seqlens[:-1] if start_positions else None

    for cp_rank in range(cp_size):
        # unfused bshd
        output_unfused_bshd = apply_rotary_pos_emb(
            bshd.float(),
            emb,
            start_positions=start_positions,
            interleaved=interleaved,
            fused=False,
            tensor_format="bshd",
            cu_seqlens=cu_seqlens,
            cp_size=cp_size,
            cp_rank=cp_rank,
        ).to(dtype)
        loss_unfused_bshd = loss_func(output_unfused_bshd)
        loss_unfused_bshd.backward()
        grad_unfused_bshd = bshd.grad.detach().clone()
        bshd.grad = None

        # unfused sbhd
        output_unfused_sbhd = apply_rotary_pos_emb(
            sbhd.float(),
            emb,
            start_positions=start_positions,
            interleaved=interleaved,
            fused=False,
            tensor_format="sbhd",
            cu_seqlens=cu_seqlens,
            cp_size=cp_size,
            cp_rank=cp_rank,
        ).to(dtype)

        loss_unfused_sbhd = loss_func(output_unfused_sbhd)
        loss_unfused_sbhd.backward()
        grad_unfused_sbhd = sbhd.grad.detach().clone()
        sbhd.grad = None

        # unfused thd
        output_unfused_thd = apply_rotary_pos_emb(
            thd.float(),
            emb,
            start_positions=start_positions,
            tensor_format="thd",
            interleaved=interleaved,
            fused=False,
            cu_seqlens=cu_seqlens,
            cp_size=cp_size,
            cp_rank=cp_rank,
        ).to(dtype)

        loss_unfused_thd = loss_func(output_unfused_thd)
        loss_unfused_thd.backward()
        grad_unfused_thd = thd.grad.detach().clone()
        thd.grad = None

        torch.testing.assert_close(
            output_unfused_bshd.reshape(*output_unfused_thd.shape), output_unfused_thd
        )
        torch.testing.assert_close(
            output_unfused_sbhd.transpose(1, 0).reshape(*output_unfused_thd.shape),
            output_unfused_thd,
        )
        torch.testing.assert_close(
            grad_unfused_bshd.reshape(*grad_unfused_thd.shape), grad_unfused_thd
        )
        torch.testing.assert_close(
            grad_unfused_sbhd.transpose(1, 0).reshape(*grad_unfused_thd.shape),
            grad_unfused_thd,
        )

        assert output_unfused_thd.is_contiguous()
        assert output_unfused_bshd.is_contiguous()
        assert output_unfused_sbhd.is_contiguous()


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
        loss_unfused = loss_func([query_unfused, key_unfused, value_unfused])

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
        loss_fused = loss_func([query_fused, key_fused, value_fused])

        if not isinstance(start_positions, torch.Tensor):
            loss_fused.backward()
            grad_fused = t.grad.detach().clone()
        t.grad = None

        torch.testing.assert_close(query_fused, query_unfused)
        torch.testing.assert_close(key_fused, key_unfused)
        torch.testing.assert_close(value_fused, value_unfused)

        if not isinstance(start_positions, torch.Tensor):
            torch.testing.assert_close(grad_fused, grad_unfused)


def test_rotary_position_embedding_forward_with_autocast_gives_same_result_as_without_autocast():
    rope_layer = RotaryPositionEmbedding(128)

    rope_embeddings_no_autocast = rope_layer(max_seq_len=1024)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        rope_embeddings_autocast = rope_layer(max_seq_len=1024)

    torch.testing.assert_close(
        rope_embeddings_no_autocast.to(dtype=torch.bfloat16),
        rope_embeddings_autocast.to(dtype=torch.bfloat16),
        atol=1e-8,
        rtol=1e-8,
    )


def _fp8_current_scaling_available() -> bool:
    import transformer_engine.pytorch as te

    return te.is_fp8_available()


@pytest.mark.skipif(not _fp8_current_scaling_available(), reason="FP8 is not available")
@pytest.mark.parametrize("seq_length", [128, 1024, 2048 + 8])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("groups, q_per_group", [(8, 4), (4, 1), (2, 8)])
@pytest.mark.parametrize("pow2", [False, True])
def test_fused_rope_fp8_quantize_matches_unfused(
    seq_length: int, batch_size: int, groups: int, q_per_group: int, pow2: bool
) -> None:
    """Deferred RoPE + FP8 quantize is bitwise equal to TE's RoPE followed by quantization."""
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.attention import deferred_qkv_rope

    head_dim = 128
    split = [q_per_group * head_dim, head_dim, head_dim]
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    qkv = (
        torch.randn(
            seq_length, batch_size, groups, sum(split), dtype=torch.bfloat16, device=device
        )
        * 4
    )
    emb = RotaryPositionEmbedding(head_dim, 1.0)(seq_length + 3)

    q_ref, k_ref, v_ref = apply_fused_qkv_rotary_pos_emb(qkv, emb, emb, split)
    quantizer = te.Float8CurrentScalingQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        device=device,
        rowwise=True,
        columnwise=False,
        force_pow_2_scales=pow2,
    )
    ref_data = torch.cat([t.reshape(-1) for t in (q_ref, k_ref, v_ref)])
    ref = quantizer(ref_data)

    q, k, v = apply_fused_qkv_rotary_pos_emb(qkv, emb, emb, split, defer_for_fp8_attention=True)
    assert deferred_qkv_rope.get_state(q) is not None
    q_fp8, k_fp8, v_fp8 = deferred_qkv_rope.quantize_deferred(q, k, v, quantizer)

    torch.testing.assert_close(q_fp8._scale_inv, ref._scale_inv, rtol=0, atol=0)
    out = torch.cat([t._data.reshape(-1) for t in (q_fp8, k_fp8, v_fp8)])
    assert torch.equal(out.view(torch.uint8), ref._data.view(torch.uint8))
    assert q_fp8.shape == q_ref.shape and k_fp8.shape == k_ref.shape


@pytest.mark.parametrize("seq_length", [128, 1024])
def test_deferred_rope_materialize_and_backward(seq_length: int) -> None:
    """Placeholders filled by materialize() and their gradient match the regular fused RoPE."""
    from transformer_engine.pytorch.attention import deferred_qkv_rope

    head_dim, groups, q_per_group, batch_size = 128, 8, 4, 2
    split = [q_per_group * head_dim, head_dim, head_dim]
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    qkv = torch.randn(
        seq_length, batch_size, groups, sum(split), dtype=torch.bfloat16, device=device
    ).requires_grad_()
    emb = RotaryPositionEmbedding(head_dim, 1.0)(seq_length)

    ref = apply_fused_qkv_rotary_pos_emb(qkv, emb, emb, split)
    loss_ref = sum(t.float().sum() * 2 for t in ref)
    (grad_ref,) = torch.autograd.grad(loss_ref, qkv)

    out = apply_fused_qkv_rotary_pos_emb(qkv, emb, emb, split, defer_for_fp8_attention=True)
    assert deferred_qkv_rope.get_state(out[0]) is not None
    deferred_qkv_rope.materialize(*out)
    for a, b in zip(out, ref):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    loss = sum(t.float().sum() * 2 for t in out)
    (grad,) = torch.autograd.grad(loss, qkv)
    torch.testing.assert_close(grad, grad_ref, rtol=0, atol=0)


def test_deferred_rope_unsupported_falls_back() -> None:
    """Unsupported inputs return ordinary tensors instead of placeholders."""
    from transformer_engine.pytorch.attention import deferred_qkv_rope

    split = [4 * 64, 64, 64]  # head dim != 128
    qkv = torch.randn(16, 2, 8, sum(split), dtype=torch.bfloat16, device="cuda:0")
    emb = RotaryPositionEmbedding(64, 1.0)(16)
    q, _, _ = apply_fused_qkv_rotary_pos_emb(qkv, emb, emb, split, defer_for_fp8_attention=True)
    assert deferred_qkv_rope.get_state(q) is None

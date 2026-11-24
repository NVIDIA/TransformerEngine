# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
from typing import Callable, Tuple, Union, List
import math
import torch
import pytest
from transformer_engine.pytorch.attention import rope
from transformer_engine.pytorch.attention.rope import (
    RotaryPositionEmbedding,
    _triton_rope_module,
    apply_mla_rope_kv,
    build_mla_rope_tables,
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


@pytest.mark.parametrize("layout", ["batched_q", "packed_kv"])
def test_dsa_triton_rope_adapters(monkeypatch, layout):
    """Compare production Q/KV adapter roles with eager outputs, gradients and aliasing."""
    if not torch.cuda.is_available():
        pytest.skip("Triton DSA RoPE requires CUDA")
    if rope._triton_rope_module() is None:
        pytest.skip("Triton DSL is unavailable")
    torch.manual_seed(23)
    inplace = layout == "batched_q"
    if inplace:
        x = torch.randn(2, 17, 64, 512, device="cuda", dtype=torch.bfloat16)
        boundaries, seq = [0, 17, 34], 17
    else:
        # Ragged lengths 3 and 5, followed by one padded compressed-token position.
        x = torch.randn(9, 1, 512, device="cuda", dtype=torch.bfloat16)
        boundaries, seq = [0, 3, 8], 5
    (cos, sin), _ = rope.build_dsv4_rope_tables(seq, 4, 64, 160000.0, x.device)
    cu = torch.tensor(boundaries, device=x.device, dtype=torch.int32)
    gradient = torch.randn_like(x)

    eager = rope.apply_dsa_rope_eager
    reference_input = x.detach().clone().requires_grad_()
    reference = eager(reference_input, cos, sin, cu)
    if not inplace:
        # Check eager packed indexing against separate sequences, including padding at position 0.
        with torch.no_grad():
            expected = torch.cat(
                [
                    eager(
                        x[start:end].unsqueeze(0), cos[:, : end - start], sin[:, : end - start]
                    ).squeeze(0)
                    for start, end in ((0, 3), (3, 8), (8, 9))
                ]
            )
        torch.testing.assert_close(reference, expected)
    reference_grad = torch.autograd.grad(reference, reference_input, gradient)[0]

    if inplace:
        with pytest.raises(ValueError, match="power of two"):
            eager(x, cos[..., :48], sin[..., :48], cu)
        with pytest.raises(ValueError, match="allow_inplace=True"):
            rope.apply_dsa_rope(x, cos, sin, allow_inplace=True)
        with pytest.raises(ValueError, match="allow_inplace=True"):
            rope.apply_dsa_rope(x.transpose(0, 1), cos, sin, cu, allow_inplace=True)
        with pytest.raises(ValueError, match="allow_inplace=True"):
            rope.apply_dsa_rope(reference_input, cos, sin, cu, allow_inplace=True)
        torch.testing.assert_close(reference_input, x, rtol=0, atol=0)

    monkeypatch.setattr(
        rope,
        "apply_dsa_rope_eager",
        lambda *_: pytest.fail("Triton dispatch fell back to eager RoPE"),
    )
    candidate_source = x.detach().clone().requires_grad_()
    candidate_input = candidate_source.clone() if inplace else candidate_source
    candidate_ptr = candidate_input.data_ptr()
    candidate = rope.apply_dsa_rope(candidate_input, cos, sin, cu, allow_inplace=inplace)
    candidate_grad = torch.autograd.grad(candidate, candidate_source, gradient)[0]

    assert (candidate.data_ptr() == candidate_ptr) == inplace
    torch.testing.assert_close(candidate, reference, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(candidate_grad, reference_grad, atol=0.03125, rtol=0.002)


@pytest.mark.parametrize("interleaved", [False, True])
def test_dsa_mla_rope_q_kernel_packed(interleaved):
    """Check both Q output layouts and gradients for the current CP1/noninverse callers."""
    if not torch.cuda.is_available():
        pytest.skip("Triton RoPE requires CUDA")
    kernels = rope._triton_rope_module()
    if kernels is None:
        pytest.skip("Triton DSL is unavailable")
    import triton

    torch.manual_seed(53)
    # Two sequences of lengths 4 and 6, then one padded token using position zero.
    positions = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0]
    cu = torch.tensor([0, 4, 10], device="cuda", dtype=torch.int32)
    x = torch.randn(11, 3, 80, device="cuda", dtype=torch.bfloat16)
    # Contiguous columns with a larger row stride exercise explicit table strides.
    cos = torch.randn(6, 128, device="cuda")[:, :64]
    sin = torch.randn(6, 128, device="cuda")[:, :64]
    reference_input = x.detach().requires_grad_()
    tail = reference_input[..., 16:].float()
    c, s = cos[positions, None], sin[positions, None]
    left = tail[..., 0::2] * c[..., :32] - tail[..., 1::2] * s[..., :32]
    right = tail[..., 1::2] * c[..., 32:] + tail[..., 0::2] * s[..., 32:]
    rotated = (
        torch.stack((left, right), -1).flatten(-2) if interleaved else torch.cat((left, right), -1)
    )
    reference = torch.cat((reference_input[..., :16], rotated.to(x.dtype)), -1)
    gradient = torch.randn_like(x)
    expected_grad = torch.autograd.grad(reference, reference_input, gradient)[0]
    candidate, actual_grad = x.clone(), gradient.clone()
    grid = lambda meta: (11, triton.cdiv(3, meta["BLOCK_H"]))
    for kernel, tensor in (
        (kernels.rotary_fwd_q_kernel, candidate),
        (kernels.rotary_bwd_q_kernel, actual_grad),
    ):
        kernel[grid](
            tensor,
            cos,
            sin,
            16,
            64,
            3,
            1,
            2,
            cu,
            None,
            tensor.stride(0),
            tensor.stride(1),
            cos.stride(0),
            sin.stride(0),
            0,
            1,
            INVERSE=False,
            REMOVE_INTERLEAVING=interleaved,
        )
    torch.testing.assert_close(candidate, reference, atol=0.03125, rtol=0.002)
    torch.testing.assert_close(actual_grad, expected_grad, atol=0.03125, rtol=0.002)


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(_triton_rope_module() is None, reason="MLA RoPE requires Triton")
def test_mla_rope_kv(monkeypatch):
    """Compare MLA keys, values, and input gradients with independent PyTorch equations.

    Split the KV input into per-head key and value channels. Rotate the shared
    positional key for each token and append it to every head's key. The value
    channels pass through unchanged. Backward sums the positional-key gradients
    across heads.
    """
    if not torch.cuda.is_available():
        pytest.skip("MLA RoPE requires CUDA")
    kernels = _triton_rope_module()
    # Force a partial block for the three heads without changing the shared autotune cache.
    backward_kernel = kernels.triton.autotune(
        configs=[kernels.triton.Config({"BLOCK_H": 4})], key=[]
    )(kernels.rotary_bwd_kv_kernel.fn)
    monkeypatch.setattr(kernels, "rotary_bwd_kv_kernel", backward_kernel)
    torch.manual_seed(59)
    kv = torch.randn(7, 2, 3, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    positional = torch.randn(7, 2, 1, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cos, sin = build_mla_rope_tables(7, emb_dim=64, device=kv.device)
    key, value = apply_mla_rope_kv(
        kv,
        positional,
        cos_table=cos,
        sin_table=sin,
        head_dim_nope=16,
        head_dim_v=16,
    )
    ref_kv = kv.detach().requires_grad_()
    ref_pos = positional.detach().requires_grad_()
    c, s = cos[:, None, None], sin[:, None, None]
    left = ref_pos[..., 0::2].float() * c[..., :32] - ref_pos[..., 1::2].float() * s[..., :32]
    right = ref_pos[..., 1::2].float() * c[..., 32:] + ref_pos[..., 0::2].float() * s[..., 32:]
    ref_key = torch.cat(
        (ref_kv[..., :16], torch.cat((left, right), -1).to(kv.dtype).expand(-1, -1, 3, -1)), -1
    )
    ref_value = ref_kv[..., 16:]
    gradients = (torch.randn_like(key), torch.randn_like(value))
    actual_grads = torch.autograd.grad((key, value), (kv, positional), gradients)
    ref_grads = torch.autograd.grad((ref_key, ref_value), (ref_kv, ref_pos), gradients)
    for actual, expected in zip((key, value, *actual_grads), (ref_key, ref_value, *ref_grads)):
        torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.02)

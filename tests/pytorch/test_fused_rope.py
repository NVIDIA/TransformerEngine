# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
from typing import Callable, Optional, Tuple, Union, List
import math
import torch
import pytest
from transformer_engine.pytorch.attention.rope import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
    apply_fused_qkv_rotary_pos_emb,
    apply_mla_rope_for_q,
    apply_mla_rope_for_kv,
)

# ---------------------------------------------------------------------------
# Megatron-LM Triton MLA YARN RoPE reference (self-contained copy)
# Source: github.com/NVIDIA/Megatron-LM  megatron/core/fusions/fused_mla_yarn_rope_apply.py
# ---------------------------------------------------------------------------

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if HAVE_TRITON:

    @triton.jit
    def _megatron_get_thd_token_idx(cu_seqlens, pid_m, seq_num, cp_rank, cp_size):
        token_idx = -1
        this_seq_len = 0
        seq_idx = 0
        last_cum_seqlen = tl.load(cu_seqlens) // cp_size
        while seq_idx < seq_num:
            cur_cum_seqlen = tl.load(cu_seqlens + seq_idx + 1) // cp_size
            if token_idx == -1 and cur_cum_seqlen > pid_m:
                token_idx = pid_m - last_cum_seqlen
                this_seq_len = cur_cum_seqlen - last_cum_seqlen
            last_cum_seqlen = cur_cum_seqlen
            seq_idx += 1
        if cp_size > 1:
            if token_idx < this_seq_len // 2:
                token_idx = token_idx + cp_rank * this_seq_len // 2
            else:
                token_idx = (token_idx - this_seq_len // 2) + (
                    2 * cp_size - cp_rank - 1
                ) * this_seq_len // 2
        return token_idx

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_H": 1}),
            triton.Config({"BLOCK_H": 2}),
            triton.Config({"BLOCK_H": 4}),
            triton.Config({"BLOCK_H": 8}),
            triton.Config({"BLOCK_H": 16}),
            triton.Config({"BLOCK_H": 32}),
            triton.Config({"BLOCK_H": 64}),
            triton.Config({"BLOCK_H": 128}),
        ],
        key=["emb_dim", "head_num"],
        restore_value=["Q"],
    )
    @triton.jit
    def _megatron_rotary_fwd_q_kernel(
        Q, COS, SIN, qk_head_dim,
        emb_dim: tl.constexpr, head_num: tl.constexpr,
        batch_size, seq_num, cu_seqlens_q,
        stride_x_seq, stride_x_nheads,
        cp_rank, cp_size,
        BLOCK_H: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_head = tl.program_id(axis=1)
        if cu_seqlens_q is None:
            token_idx = pid_m // batch_size
        else:
            token_idx = _megatron_get_thd_token_idx(
                cu_seqlens_q, pid_m, seq_num, cp_rank, cp_size
            )
        cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        cos_right = tl.load(
            COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2)
        )
        sin_right = tl.load(
            SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2)
        )
        cos_left = cos_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        sin_left = sin_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        cos_right = cos_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        sin_right = sin_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        Q = Q + pid_m * stride_x_seq + pid_head * BLOCK_H * stride_x_nheads
        x_off = tl.arange(0, BLOCK_H)[:, None] * stride_x_nheads + qk_head_dim
        mask = x_off < head_num * stride_x_nheads
        x_1_off = x_off + tl.arange(0, emb_dim // 2)[None, :] * 2
        x_2_off = x_1_off + 1
        x_1 = tl.load(Q + x_1_off, mask=mask)
        x_2 = tl.load(Q + x_2_off, mask=mask)
        x_left = x_1 * cos_left - x_2 * sin_left
        x_right = x_2 * cos_right + x_1 * sin_right
        x_left_off = x_off + tl.arange(0, emb_dim // 2)[None, :]
        x_right_off = x_left_off + emb_dim // 2
        tl.store(Q + x_left_off, x_left, mask=mask)
        tl.store(Q + x_right_off, x_right, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_H": 1}),
            triton.Config({"BLOCK_H": 2}),
            triton.Config({"BLOCK_H": 4}),
            triton.Config({"BLOCK_H": 8}),
            triton.Config({"BLOCK_H": 16}),
            triton.Config({"BLOCK_H": 32}),
            triton.Config({"BLOCK_H": 64}),
            triton.Config({"BLOCK_H": 128}),
        ],
        key=["emb_dim", "k_dim", "v_dim", "head_num"],
    )
    @triton.jit
    def _megatron_rotary_fwd_kv_kernel(
        KV, K_POS_EMB, O_KEY, O_VALUE, COS, SIN,
        emb_dim: tl.constexpr, k_dim: tl.constexpr,
        v_dim: tl.constexpr, head_num: tl.constexpr,
        batch_size, seq_num, cu_seqlens_kv,
        stride_kv_seq, stride_kv_nheads, stride_emb_seq,
        stride_k_seq, stride_k_nheads,
        stride_v_seq, stride_v_nheads,
        cp_rank, cp_size,
        BLOCK_H: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_head = tl.program_id(axis=1)
        if cu_seqlens_kv is None:
            token_idx = pid_m // batch_size
        else:
            token_idx = _megatron_get_thd_token_idx(
                cu_seqlens_kv, pid_m, seq_num, cp_rank, cp_size
            )
        cos_left = tl.load(COS + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        sin_left = tl.load(SIN + token_idx * emb_dim + tl.arange(0, emb_dim // 2))
        cos_right = tl.load(
            COS + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2)
        )
        sin_right = tl.load(
            SIN + token_idx * emb_dim + emb_dim // 2 + tl.arange(0, emb_dim // 2)
        )
        KV_ptr = KV + pid_m * stride_kv_seq + pid_head * BLOCK_H * stride_kv_nheads
        kv_off = tl.arange(0, BLOCK_H)[:, None] * stride_kv_nheads
        mask = kv_off < head_num * stride_kv_nheads
        k_in_off = kv_off + tl.arange(0, k_dim)[None, :]
        v_in_off = kv_off + k_dim + tl.arange(0, v_dim)[None, :]
        k = tl.load(KV_ptr + k_in_off, mask=mask)
        v = tl.load(KV_ptr + v_in_off, mask=mask)
        K_ptr = O_KEY + pid_m * stride_k_seq + pid_head * BLOCK_H * stride_k_nheads
        V_ptr = O_VALUE + pid_m * stride_v_seq + pid_head * BLOCK_H * stride_v_nheads
        k_out_off = (
            tl.arange(0, BLOCK_H)[:, None] * stride_k_nheads
            + tl.arange(0, k_dim)[None, :]
        )
        v_out_off = (
            tl.arange(0, BLOCK_H)[:, None] * stride_v_nheads
            + tl.arange(0, v_dim)[None, :]
        )
        tl.store(K_ptr + k_out_off, k, mask=mask)
        tl.store(V_ptr + v_out_off, v, mask=mask)
        EMB = K_POS_EMB + pid_m * stride_emb_seq
        x_1 = tl.load(EMB + tl.arange(0, emb_dim // 2) * 2)
        x_2 = tl.load(EMB + tl.arange(0, emb_dim // 2) * 2 + 1)
        x_left = x_1 * cos_left - x_2 * sin_left
        x_right = x_2 * cos_right + x_1 * sin_right
        x_left = x_left.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        x_right = x_right.expand_dims(0).broadcast_to(BLOCK_H, emb_dim // 2)
        x_left_off = (
            tl.arange(0, BLOCK_H)[:, None] * stride_k_nheads
            + k_dim
            + tl.arange(0, emb_dim // 2)[None, :]
        )
        x_right_off = x_left_off + emb_dim // 2
        tl.store(K_ptr + x_left_off, x_left, mask=mask)
        tl.store(K_ptr + x_right_off, x_right, mask=mask)

    def _triton_mla_rope_q(
        t: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        qk_head_dim: int,
        emb_dim: int,
    ) -> torch.Tensor:
        """Call Megatron-LM's Triton Q kernel with TE-style 2-D cos/sin tables (SBHD only)."""
        s, b, nheads, headdim = t.shape
        q = t.clone().view(-1, nheads, headdim)
        total = q.shape[0]
        cos_2d = cos_table[:s].contiguous()
        sin_2d = sin_table[:s].contiguous()
        grid = lambda META: (total, triton.cdiv(nheads, META["BLOCK_H"]))
        _megatron_rotary_fwd_q_kernel[grid](
            q, cos_2d, sin_2d, qk_head_dim, emb_dim, nheads,
            b, None, None, q.stride(0), q.stride(1), 0, 1,
        )
        return q.view(s, b, nheads, headdim)

    def _triton_mla_rope_kv(
        kv: torch.Tensor,
        k_pos_emb: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        k_dim: int,
        v_dim: int,
        emb_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call Megatron-LM's Triton KV kernel with TE-style tables (SBHD only)."""
        s, b, nheads, headdim = kv.shape
        kv_3d = kv.contiguous().view(-1, nheads, headdim)
        emb_2d = k_pos_emb.contiguous().view(-1, emb_dim)
        total = kv_3d.shape[0]
        cos_2d = cos_table[:s].contiguous()
        sin_2d = sin_table[:s].contiguous()
        o_key = kv_3d.new_empty(total, nheads, emb_dim + k_dim)
        o_value = kv_3d.new_empty(total, nheads, v_dim)
        grid = lambda META: (total, triton.cdiv(nheads, META["BLOCK_H"]))
        _megatron_rotary_fwd_kv_kernel[grid](
            kv_3d, emb_2d, o_key, o_value, cos_2d, sin_2d,
            emb_dim, k_dim, v_dim, nheads,
            b, None, None,
            kv_3d.stride(0), kv_3d.stride(1), emb_2d.stride(0),
            o_key.stride(0), o_key.stride(1),
            o_value.stride(0), o_value.stride(1),
            0, 1,
        )
        return o_key.view(s, b, nheads, emb_dim + k_dim), o_value.view(s, b, nheads, v_dim)


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
    emb = rotary_pos_emb(cu_seqlens_padded[-1])
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
            grad_unfused_sbhd.transpose(1, 0).reshape(*grad_unfused_thd.shape), grad_unfused_thd
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


# ---------------------------------------------------------------------------
# MLA YARN RoPE tests
# ---------------------------------------------------------------------------


def _make_cos_sin_tables(max_seq_len: int, emb_dim: int, device: torch.device):
    """Generate deterministic cos/sin tables mimicking YARN-style RoPE frequencies.

    Returns 4-D tensors of shape ``[max_seq_len, 1, 1, emb_dim]`` (new API format).
    Use ``cos_table[:, 0, 0, :]`` to get the 2-D ``[max_seq_len, emb_dim]`` view
    needed by reference functions.
    """
    half = emb_dim // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)
    freqs_right = torch.outer(pos, inv_freq * 1.3)
    cos_2d = torch.cat([freqs.cos(), freqs_right.cos()], dim=-1).contiguous()
    sin_2d = torch.cat([freqs.sin(), freqs_right.sin()], dim=-1).contiguous()
    cos_table = cos_2d.unsqueeze(1).unsqueeze(1).contiguous()
    sin_table = sin_2d.unsqueeze(1).unsqueeze(1).contiguous()
    return cos_table, sin_table


def _ref_mla_yarn_rope_q(
    t: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    qk_head_dim: int,
    emb_dim: int,
    tensor_format: str = "sbhd",
) -> torch.Tensor:
    """Pure PyTorch reference for MLA YARN RoPE on Q."""
    if tensor_format == "bshd":
        t = t.transpose(0, 1)  # -> sbhd

    s = t.shape[0]
    half = emb_dim // 2
    cos_L = cos_table[:s, :half].unsqueeze(1).unsqueeze(1)
    sin_L = sin_table[:s, :half].unsqueeze(1).unsqueeze(1)
    cos_R = cos_table[:s, half:].unsqueeze(1).unsqueeze(1)
    sin_R = sin_table[:s, half:].unsqueeze(1).unsqueeze(1)

    prefix = t[..., :qk_head_dim]
    tail = t[..., qk_head_dim:]
    x1 = tail[..., 0::2]
    x2 = tail[..., 1::2]

    out_left = x1 * cos_L - x2 * sin_L
    out_right = x2 * cos_R + x1 * sin_R
    out = torch.cat([prefix, out_left, out_right], dim=-1)

    if tensor_format == "bshd":
        out = out.transpose(0, 1)
    return out


def _ref_mla_yarn_rope_kv(
    kv: torch.Tensor,
    k_pos_emb: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    k_dim: int,
    v_dim: int,
    emb_dim: int,
    tensor_format: str = "sbhd",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference for MLA YARN RoPE on KV."""
    if tensor_format == "bshd":
        kv = kv.transpose(0, 1)
        k_pos_emb = k_pos_emb.transpose(0, 1)

    s, b, h, _ = kv.shape
    half = emb_dim // 2
    cos_L = cos_table[:s, :half]
    sin_L = sin_table[:s, :half]
    cos_R = cos_table[:s, half:]
    sin_R = sin_table[:s, half:]

    k_content = kv[..., :k_dim]
    v = kv[..., k_dim:]

    # k_pos_emb: [s, b, emb_dim] -> interleaved read
    x1 = k_pos_emb[..., 0::2]
    x2 = k_pos_emb[..., 1::2]

    # cos/sin are [s, half], k_pos_emb terms are [s, b, half]
    rope_left = x1 * cos_L.unsqueeze(1) - x2 * sin_L.unsqueeze(1)
    rope_right = x2 * cos_R.unsqueeze(1) + x1 * sin_R.unsqueeze(1)

    # Broadcast across heads: [s, b, half] -> [s, b, h, half]
    rope_left = rope_left.unsqueeze(2).expand(s, b, h, half)
    rope_right = rope_right.unsqueeze(2).expand(s, b, h, half)

    o_key = torch.cat([k_content, rope_left, rope_right], dim=-1)
    o_value = v

    if tensor_format == "bshd":
        o_key = o_key.transpose(0, 1)
        o_value = o_value.transpose(0, 1)
    return o_key, o_value


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_length", [128, 512])
@pytest.mark.parametrize("emb_dim", [64, 128])
@pytest.mark.parametrize("qk_head_dim", [64, 128])
@pytest.mark.parametrize("tensor_format", ["sbhd"])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
def test_fused_mla_rope_q(
    dtype: torch.dtype,
    seq_length: int,
    emb_dim: int,
    qk_head_dim: int,
    tensor_format: str,
    loss_func: Callable,
) -> None:
    device = torch.device("cuda:0")
    batch_size, head_num = 2, 16
    head_dim = qk_head_dim + emb_dim

    t = torch.rand(
        (seq_length, batch_size, head_num, head_dim),
        dtype=dtype,
        device=device,
    )
    if tensor_format == "bshd":
        t = t.transpose(0, 1).contiguous()
    t.requires_grad = True

    cos_table, sin_table = _make_cos_sin_tables(seq_length, emb_dim, device)
    cos_2d = cos_table[:, 0, 0, :]
    sin_2d = sin_table[:, 0, 0, :]

    # --- PyTorch reference (float32 for precision, uses 2D tables) ---
    output_ref = _ref_mla_yarn_rope_q(
        t.float(), cos_2d, sin_2d, qk_head_dim, emb_dim, tensor_format
    ).to(dtype)
    loss_ref = loss_func(output_ref)
    loss_ref.backward()
    grad_ref = t.grad.detach().clone()
    t.grad = None

    # --- Fused CUDA kernel (4D tables, no emb_dim arg) ---
    output_fused = apply_mla_rope_for_q(
        t, cos_table, sin_table, qk_head_dim, tensor_format=tensor_format
    )
    loss_fused = loss_func(output_fused)
    loss_fused.backward()
    grad_fused = t.grad.detach().clone()
    t.grad = None

    torch.testing.assert_close(output_fused, output_ref)
    torch.testing.assert_close(grad_fused, grad_ref)
    assert output_fused.is_contiguous()

    # --- Megatron-LM Triton reference (forward only, uses 2D tables) ---
    if HAVE_TRITON and tensor_format == "sbhd":
        output_triton = _triton_mla_rope_q(
            t.float(), cos_2d, sin_2d, qk_head_dim, emb_dim
        ).to(dtype)
        torch.testing.assert_close(output_fused, output_triton)
        torch.testing.assert_close(output_ref, output_triton)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_length", [128, 512])
@pytest.mark.parametrize("emb_dim", [64, 128])
@pytest.mark.parametrize("k_dim", [64, 128])
@pytest.mark.parametrize("v_dim", [64, 128])
@pytest.mark.parametrize("tensor_format", ["sbhd"])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
def test_fused_mla_rope_kv(
    dtype: torch.dtype,
    seq_length: int,
    emb_dim: int,
    k_dim: int,
    v_dim: int,
    tensor_format: str,
    loss_func: Callable,
) -> None:
    device = torch.device("cuda:0")
    batch_size, head_num = 2, 16

    kv = torch.rand(
        (seq_length, batch_size, head_num, k_dim + v_dim),
        dtype=dtype,
        device=device,
    )
    k_pos_emb = torch.rand(
        (seq_length, batch_size, emb_dim),
        dtype=dtype,
        device=device,
    )
    if tensor_format == "bshd":
        kv = kv.transpose(0, 1).contiguous()
        k_pos_emb = k_pos_emb.transpose(0, 1).contiguous()
    kv.requires_grad = True
    k_pos_emb.requires_grad = True

    cos_table, sin_table = _make_cos_sin_tables(seq_length, emb_dim, device)
    cos_2d = cos_table[:, 0, 0, :]
    sin_2d = sin_table[:, 0, 0, :]

    # Reference (float32 for precision, uses 2D tables)
    okey_ref, oval_ref = _ref_mla_yarn_rope_kv(
        kv.float(), k_pos_emb.float(), cos_2d, sin_2d, k_dim, v_dim, emb_dim, tensor_format
    )
    okey_ref = okey_ref.to(dtype)
    oval_ref = oval_ref.to(dtype)
    loss_ref = loss_func([okey_ref, oval_ref])
    loss_ref.backward()
    grad_kv_ref = kv.grad.detach().clone()
    grad_emb_ref = k_pos_emb.grad.detach().clone()
    kv.grad = None
    k_pos_emb.grad = None

    # Fused CUDA kernel (4D tables, no emb_dim arg)
    okey_fused, oval_fused = apply_mla_rope_for_kv(
        kv, k_pos_emb, cos_table, sin_table, k_dim, v_dim, tensor_format=tensor_format
    )
    loss_fused = loss_func([okey_fused, oval_fused])
    loss_fused.backward()
    grad_kv_fused = kv.grad.detach().clone()
    grad_emb_fused = k_pos_emb.grad.detach().clone()
    kv.grad = None
    k_pos_emb.grad = None

    torch.testing.assert_close(okey_fused, okey_ref)
    torch.testing.assert_close(oval_fused, oval_ref)
    torch.testing.assert_close(grad_kv_fused, grad_kv_ref)
    torch.testing.assert_close(grad_emb_fused, grad_emb_ref)
    assert okey_fused.is_contiguous()
    assert oval_fused.is_contiguous()

    # --- Megatron-LM Triton reference (forward only, uses 2D tables) ---
    if HAVE_TRITON and tensor_format == "sbhd":
        okey_triton, oval_triton = _triton_mla_rope_kv(
            kv.float(), k_pos_emb.float(), cos_2d, sin_2d, k_dim, v_dim, emb_dim
        )
        okey_triton = okey_triton.to(dtype)
        oval_triton = oval_triton.to(dtype)
        torch.testing.assert_close(okey_fused, okey_triton)
        torch.testing.assert_close(oval_fused, oval_triton)
        torch.testing.assert_close(okey_ref, okey_triton)
        torch.testing.assert_close(oval_ref, oval_triton)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("emb_dim", [64, 128])
@pytest.mark.parametrize("qk_head_dim", [64])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
def test_fused_mla_rope_q_thd(
    dtype: torch.dtype,
    emb_dim: int,
    qk_head_dim: int,
    loss_func: Callable,
) -> None:
    device = torch.device("cuda:0")
    head_num = 16
    head_dim = qk_head_dim + emb_dim

    cu_seqlens = torch.tensor([0, 120, 280, 512], dtype=torch.int32, device=device)
    total_seq = int(cu_seqlens[-1].item())

    t = torch.rand((total_seq, head_num, head_dim), dtype=dtype, device=device)
    t.requires_grad = True

    cos_table, sin_table = _make_cos_sin_tables(total_seq, emb_dim, device)
    cos_2d = cos_table[:, 0, 0, :]
    sin_2d = sin_table[:, 0, 0, :]

    # Build reference by splitting into per-sequence sbhd tensors (uses 2D tables)
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    ref_outputs = []
    for seq_start, seq_len in zip(cu_seqlens[:-1].tolist(), seqlens):
        chunk = t[seq_start : seq_start + seq_len].unsqueeze(1)  # [s, 1, h, d]
        ref_out = _ref_mla_yarn_rope_q(
            chunk.float(), cos_2d[:seq_len], sin_2d[:seq_len], qk_head_dim, emb_dim
        )
        ref_outputs.append(ref_out.squeeze(1))
    output_ref = torch.cat(ref_outputs, dim=0).to(dtype)
    loss_ref = loss_func(output_ref)
    loss_ref.backward()
    grad_ref = t.grad.detach().clone()
    t.grad = None

    # Fused kernel with THD format (4D tables, no emb_dim arg)
    output_fused = apply_mla_rope_for_q(
        t, cos_table, sin_table, qk_head_dim,
        tensor_format="thd", cu_seqlens=cu_seqlens,
    )
    loss_fused = loss_func(output_fused)
    loss_fused.backward()
    grad_fused = t.grad.detach().clone()
    t.grad = None

    torch.testing.assert_close(output_fused, output_ref)
    torch.testing.assert_close(grad_fused, grad_ref)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("emb_dim", [64, 128])
@pytest.mark.parametrize("k_dim", [64])
@pytest.mark.parametrize("v_dim", [64])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
def test_fused_mla_rope_kv_thd(
    dtype: torch.dtype,
    emb_dim: int,
    k_dim: int,
    v_dim: int,
    loss_func: Callable,
) -> None:
    device = torch.device("cuda:0")
    head_num = 16

    cu_seqlens = torch.tensor([0, 120, 280, 512], dtype=torch.int32, device=device)
    total_seq = int(cu_seqlens[-1].item())

    kv = torch.rand((total_seq, head_num, k_dim + v_dim), dtype=dtype, device=device)
    k_pos_emb = torch.rand((total_seq, emb_dim), dtype=dtype, device=device)
    kv.requires_grad = True
    k_pos_emb.requires_grad = True

    cos_table, sin_table = _make_cos_sin_tables(total_seq, emb_dim, device)
    cos_2d = cos_table[:, 0, 0, :]
    sin_2d = sin_table[:, 0, 0, :]

    # Build reference by splitting into per-sequence sbhd tensors (uses 2D tables)
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    ref_keys, ref_vals = [], []
    for seq_start, seq_len in zip(cu_seqlens[:-1].tolist(), seqlens):
        kv_chunk = kv[seq_start : seq_start + seq_len].unsqueeze(1)       # [s, 1, h, d]
        emb_chunk = k_pos_emb[seq_start : seq_start + seq_len].unsqueeze(1)  # [s, 1, emb_dim]
        okey, oval = _ref_mla_yarn_rope_kv(
            kv_chunk.float(), emb_chunk.float(),
            cos_2d[:seq_len], sin_2d[:seq_len],
            k_dim, v_dim, emb_dim,
        )
        ref_keys.append(okey.squeeze(1))
        ref_vals.append(oval.squeeze(1))
    okey_ref = torch.cat(ref_keys, dim=0).to(dtype)
    oval_ref = torch.cat(ref_vals, dim=0).to(dtype)
    loss_ref = loss_func([okey_ref, oval_ref])
    loss_ref.backward()
    grad_kv_ref = kv.grad.detach().clone()
    grad_emb_ref = k_pos_emb.grad.detach().clone()
    kv.grad = None
    k_pos_emb.grad = None

    # Fused kernel with THD format (4D tables, no emb_dim arg)
    okey_fused, oval_fused = apply_mla_rope_for_kv(
        kv, k_pos_emb, cos_table, sin_table, k_dim, v_dim,
        tensor_format="thd", cu_seqlens=cu_seqlens,
    )
    loss_fused = loss_func([okey_fused, oval_fused])
    loss_fused.backward()
    grad_kv_fused = kv.grad.detach().clone()
    grad_emb_fused = k_pos_emb.grad.detach().clone()
    kv.grad = None
    k_pos_emb.grad = None

    torch.testing.assert_close(okey_fused, okey_ref)
    torch.testing.assert_close(oval_fused, oval_ref)
    torch.testing.assert_close(grad_kv_fused, grad_kv_ref)
    torch.testing.assert_close(grad_emb_fused, grad_emb_ref)

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import math
import pytest
import torch
from typing import Callable, Tuple, Union
from transformer_engine.pytorch.attention import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
)


def _get_thd_freqs_on_this_cp_rank(
    cp_rank: int, cp_size: int, x: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    if cp_size > 1:
        cp_seg = x.size(0) // 2
        full_seqlen = cp_size * x.size(0)
        return torch.cat(
            [
                freqs[cp_rank * cp_seg : (cp_rank + 1) * cp_seg],
                freqs[full_seqlen - (cp_rank + 1) * cp_seg : full_seqlen - cp_rank * cp_seg],
            ]
        )
    else:
        return freqs[: x.size(0)]


def apply_rotary_pos_emb_thd(
    t: torch.Tensor,
    cu_seqlens: torch.Tensor,
    freqs: torch.Tensor,
    start_positions: torch.Tensor,
    cp_size: int = 1, 
    cp_rank: int = 0,
) -> torch.Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]
        start_positions (Tensor): Tensor of shape [b] determining the beginning offsets
                         of frequeuncies applied to  sequences.

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """
    cu_seqlens = cu_seqlens // cp_size
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    if cp_size > 1:
        return torch.cat(
            [
                apply_rotary_pos_emb(
                    x.unsqueeze(1), _get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, x, freqs)
                )
                for x in torch.split(t, seqlens)
            ]
        ).squeeze(1)
    else:
        if start_positions is None:
            return torch.cat(
                [
                    apply_rotary_pos_emb(x.unsqueeze(1), freqs[: x.size(0)])
                    for x in torch.split(t, seqlens)
                ]
            ).squeeze(1)
        else:
            return torch.cat(
                [
                    apply_rotary_pos_emb(
                        x.unsqueeze(1), freqs[start_positions[i] : (x.size(0) + start_positions[i])]
                    )
                    for i, x in enumerate(torch.split(t, seqlens))
                ]
            ).squeeze(1)
        


def apply_rotary_pos_emb_with_start_positions(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    start_positions: Union[torch.Tensor, None] = None,
    fused: bool = False
) -> torch.Tensor:
    """
    Apply rotary positional embedding tensor to the input tensor.
    This is non-fused version which supports start_positions parameters.
    Non-fused implementation with start_positions is slow, thus it is not included in the
    Transformer Engine directly.

    Parameters
    ----------
    t: torch.Tensor
        Input tensor of shape `[s, b, h, d]`, `[b, s, h, d]` or `[t, h, d]`, on which
        rotary positional embedding will be applied.
    freqs: torch.Tensor
        Rotary positional embedding tensor of shape `[s2, 1, 1, d2]` and dtype 'float',
        with `s2 >= s` and `d2 <= d`.
    tensor_format: {'sbhd', 'bshd'}, default = 'sbhd'
    start_positions: torch.Tensor, default = None.
        We may not want begin all the sequences from the 0 embedding.
        This tensor argument allows that.
    """

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        change sign so the last dimension becomes [-odd, +even]
        """
        x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    if start_positions is None:
        return apply_rotary_pos_emb(t, freqs, tensor_format=tensor_format)

    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]

    # Only apply the rotary embeddings up to the sequence length of the running
    # input.
    assert (
        cur_seq_len <= max_seq_len
    ), f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"

    if tensor_format == "bshd":
        t = t.transpose(0, 1)
    # cos/sin first then dtype conversion for better precision
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # shifted_sin, shifted_cos will have the same shape as t. They will contain
    # scaling factors shifted for each sequence by the corresponding start_positions offset.

    shifted_sin = sin_[:cur_seq_len].expand(t.shape).clone()
    shifted_cos = cos_[:cur_seq_len].expand(t.shape).clone()

    for b in range(start_positions.shape[0]):
        assert max_seq_len >= start_positions[b]
        shifted_freq = slice(start_positions[b], (start_positions[b] + cur_seq_len))
        shifted_sin[:, b, :] = sin_[shifted_freq, 0, ...]
        shifted_cos[:, b, :] = cos_[shifted_freq, 0, ...]

    t = (t * shifted_cos) + (_rotate_half(t) * shifted_sin)
    out = torch.cat((t, t_pass), dim=-1)

    if tensor_format == "bshd":
        out = out.transpose(0, 1).contiguous()

    return out


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
@pytest.mark.parametrize("start_positions", [True, False])
@pytest.mark.parametrize("transpose", [None, (0, 1), (2, 3)])
@pytest.mark.parametrize("tensor_format", ["sbhd", "bshd"])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
def test_fused_rope(
    dtype: torch.dtype,
    seq_length: int,
    hidden_size: int,
    rotary_percent: float,
    margin: int,
    start_positions: bool,
    transpose: Union[Tuple, None],
    tensor_format: str,
    loss_func: Callable,
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

    if margin == 0 and start_positions == True:
        # If sequence to encode has the same length as length of encoding
        # there is no space left for starting with positions >0.
        pytest.skip("Skipping test with margin=0 and start_positions=True")

    start_positions = (
        torch.randint(0, margin, (batch_size,), dtype=torch.int32, device=device)
        if start_positions
        else None
    )

    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent)
    emb = rotary_pos_emb(seq_length)

    # unfused
    # The fused kernel computes in float32 internally, so we force the unfused func to use float32
    # for more accurate comparison
    output_unfused = apply_rotary_pos_emb_with_start_positions(
        t.float(), emb, tensor_format=tensor_format, start_positions=start_positions, fused=False
    ).to(dtype)
    loss_unfused = loss_func(output_unfused)
    loss_unfused.backward()
    grad_unfused = t.grad.detach().clone()
    t.grad = None

    # fused
    output_fused = apply_rotary_pos_emb(
        t, emb, tensor_format=tensor_format, fused=True, start_positions=start_positions
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
@pytest.mark.parametrize("start_positions", [True, False])
@pytest.mark.parametrize("cp_size", [1, 2, 3])
def test_fused_rope_thd(
    dtype: torch.dtype,
    hidden_size: int,
    rotary_percent: float,
    transpose: Union[Tuple, None],
    loss_func: Callable,
    start_positions: bool,
    cp_size: int,
) -> None:
    device = torch.device("cuda:0")
    batch_size, head_num = 2, 64
    cu_seqlens = [0, 400, 542, 711, 727, 752, 1270, 1426, 1450, 1954, 2044, 2048]
    if cp_size > 1:
        start_positions = False
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

    start_positions = (
        torch.randint(0, 20, (cu_seqlens_padded.shape[-1],), dtype=torch.int32, device=device)
        if start_positions
        else None
    )

    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent)
    emb = rotary_pos_emb(cu_seqlens_padded[-1])

    for cp_rank in range(cp_size):
        # unfused
        # The fused kernel computes in float32 internally, so we force the unfused func to use float32
        # for more accurate comparison
        output_unfused = apply_rotary_pos_emb_thd(
            t.float(), cu_seqlens_padded, emb, start_positions, cp_size, cp_rank
        ).to(dtype)
        loss_unfused = loss_func(output_unfused)
        loss_unfused.backward()
        grad_unfused = t.grad.detach().clone()
        t.grad = None

        # fused
        output_fused = apply_rotary_pos_emb(
            t,
            emb,
            tensor_format="thd",
            fused=True,
            start_positions=start_positions,
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
def test_fused_rope_staggered_inputs(
    dtype: torch.dtype,
    seq_length: int,
    hidden_size: int,
    rotary_percent: float,
    margin: int,
    start_positions: bool,
    tensor_format: str,
    loss_func: Callable,
) -> None:
    if margin == 0 and start_positions == True:
        # If sequence to encode has the same length as length of encoding
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

    #
    # 1. Apply RoPE with start_positions
    #
    # The fused kernel computes in float32 internally, so we force the unfused func to use float32
    # for more accurate comparison
    output_unfused = apply_rotary_pos_emb_with_start_positions(
        t.float(), rope_emb, tensor_format=tensor_format, start_positions=start_positions, fused=False
    ).to(dtype)
    loss_unfused = loss_func(output_unfused)
    loss_unfused.backward()
    grad_unfused = t.grad.detach().clone()
    t.grad = None

    # fused
    output_fused = apply_rotary_pos_emb(
        t, rope_emb, tensor_format=tensor_format, fused=True, start_positions=start_positions
    )
    loss_fused = loss_func(output_fused)
    loss_fused.backward()
    grad_fused = t.grad.detach().clone()
    t.grad = None

    
    torch.testing.assert_close(output_fused, output_unfused)
    torch.testing.assert_close(grad_fused, grad_unfused)
    assert output_fused.is_contiguous()

    #
    # 2. Create RoPE with staggered embeddings in `sb1d` format and apply to 
    #    all the sequences in a batch. `start_positions` should be None in this
    #    case.
    #
    start = torch.zeros((batch_size,), dtype=int, device=device) if start_positions is None else start_positions
    staggered_rope_emb = torch.cat([rope_emb[idx : idx + running_seq_len] for idx in start], dim = 1)
   
    output_staggered = apply_rotary_pos_emb(
        t, staggered_rope_emb, tensor_format=tensor_format, fused=True, start_positions=None
    )
    loss_staggered = loss_func(output_staggered)
    loss_staggered.backward()
    grad_staggered = t.grad.detach().clone()
    t.grad = None

    assert output_staggered.is_contiguous()
    torch.testing.assert_close(output_staggered, output_fused)
    torch.testing.assert_close(grad_staggered, grad_fused)

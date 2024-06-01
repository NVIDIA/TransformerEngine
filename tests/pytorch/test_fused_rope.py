# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import pytest
import torch
from typing import Callable, Dict, Tuple, Union
from transformer_engine.pytorch.attention import (
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
)


def apply_rotary_pos_emb_thd(
    t: torch.Tensor, cu_seqlens: torch.Tensor, freqs: torch.Tensor, start_positions: torch.Tensor
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
    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    if start_positions is None:
        return torch.cat(
            [
                apply_rotary_pos_emb(x.unsqueeze(1), freqs[:x.size(0)])
                for x in torch.split(t, seqlens)
            ]
        ).squeeze(1)
    else:
        return torch.cat(
            [
                apply_rotary_pos_emb(x.unsqueeze(1), freqs[start_positions[i]:(x.size(0) + start_positions[i])])
                for i, x in enumerate(torch.split(t, seqlens))
            ]
        ).squeeze(1)


def apply_rotary_pos_emb_with_start_positions(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    start_positions: Union[torch.Tensor, None] = None,
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

    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]

    # Only apply the rotary embeddings up to the sequence length of the running
    # input.
    assert cur_seq_len <= max_seq_len, (
        f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
    )
    if start_positions is None:
        freqs = freqs[:cur_seq_len]
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)  # [seq, 1, 1, dim] -> [1, seq, 1, dim]
    # cos/sin first then dtype conversion for better precision
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if start_positions is not None:
        if tensor_format == "bshd":
            sin_1 = sin_[:, :cur_seq_len, :, :].expand(t.shape).clone()
            cos_1 = cos_[:, :cur_seq_len, :, :].expand(t.shape).clone()
            sin_2 = sin_.expand((t.shape[0], -1, t.shape[2], t.shape[3])).clone()
            cos_2 = cos_.expand((t.shape[0], -1, t.shape[2], t.shape[3])).clone()

        else:
            sin_1 = sin_[:cur_seq_len].expand(t.shape).clone()
            cos_1 = cos_[:cur_seq_len].expand(t.shape).clone()
            sin_2 = sin_.expand((-1, t.shape[1], t.shape[2], t.shape[3])).clone()
            cos_2 = cos_.expand((-1, t.shape[1], t.shape[2], t.shape[3])).clone()
        for b in range(start_positions.shape[0]):
            assert max_seq_len >= start_positions[b]
            if tensor_format == "bshd":
                sin_1[b, :] = sin_2[b, start_positions[b]:(start_positions[b] + cur_seq_len), :]
                cos_1[b, :] = cos_2[b, start_positions[b]:(start_positions[b] + cur_seq_len), :]
            else:
                sin_1[:, b, :] = sin_2[start_positions[b]:(start_positions[b] + cur_seq_len), b, :]
                cos_1[:, b, :] = cos_2[start_positions[b]:(start_positions[b] + cur_seq_len), b, :]
        t = (t * cos_1) + (_rotate_half(t) * sin_1)
        return torch.cat((t, t_pass), dim=-1)

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * cos_) + (_rotate_half(t) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def get_tol(dtype: torch.dtype) -> Dict:
    if dtype == torch.bfloat16:
        return dict(atol=1e-2, rtol=1e-2)
    elif dtype == torch.float16:
        return dict(atol=1e-3, rtol=1e-3)
    return dict(atol=1e-5, rtol=1.3e-6)


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
@pytest.mark.parametrize("tensor_format", ["bshd", "sbhd"])
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

    if margin == 0:
        start_positions = False
    start_positions = torch.randint(0, margin, (batch_size,), dtype=torch.int32, device=device) if start_positions else None



    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent)
    emb = rotary_pos_emb(seq_length)

    # unfused
    output_unfused = apply_rotary_pos_emb_with_start_positions(
        t, emb, tensor_format=tensor_format, start_positions=start_positions
    )
    loss_unfused = loss_func(output_unfused)
    loss_unfused.backward()
    grad_unfused = t.grad.detach().clone()
    t.grad = None

    # fused
    output_fused = apply_rotary_pos_emb(
        t,
        emb,
        tensor_format=tensor_format,
        fused=True,
        start_positions=start_positions
    )
    loss_fused = loss_func(output_fused)
    loss_fused.backward()
    grad_fused = t.grad.detach().clone()
    t.grad = None

    torch.testing.assert_close(output_fused, output_unfused, **get_tol(dtype))
    torch.testing.assert_close(grad_fused, grad_unfused, **get_tol(dtype))
    assert output_fused.is_contiguous()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("rotary_percent", [0.5, 1.0])
@pytest.mark.parametrize("transpose", [None, (1, 2)])
@pytest.mark.parametrize("loss_func", [_overlapping_grad, _non_overlapping_grad])
@pytest.mark.parametrize("start_positions", [True, False])
def test_fused_rope_thd(
    dtype: torch.dtype,
    hidden_size: int,
    rotary_percent: float,
    transpose: Union[Tuple, None],
    loss_func: Callable,
    start_positions: bool,
) -> None:
    device = torch.device("cuda:0")
    batch_size, head_num = 2, 64
    cu_seqlens = torch.tensor(
        [0, 400, 542, 711, 727, 752, 1270, 1426, 1450, 1954, 2044, 2048],
        dtype=torch.int32,
        device=device,
    )
    t = torch.rand(
        (cu_seqlens[-1], head_num, hidden_size),
        dtype=dtype,
        device=device,
    )
    if transpose:
        t = t.transpose(*transpose).contiguous().transpose(*transpose)
    t.requires_grad = True

    start_positions = torch.randint(0, 20, (cu_seqlens.shape[-1],), dtype=torch.int32, device=device) if start_positions else None

    rotary_pos_emb = RotaryPositionEmbedding(hidden_size, rotary_percent)
    emb = rotary_pos_emb(cu_seqlens[-1])

    # unfused
    output_unfused = apply_rotary_pos_emb_thd(t, cu_seqlens, emb, start_positions=start_positions)
    loss_unfused = loss_func(output_unfused)
    loss_unfused.backward()
    grad_unfused = t.grad.detach().clone()
    t.grad = None

    # fused
    output_fused = apply_rotary_pos_emb(
        t, emb, fused=True, tensor_format="thd", cu_seqlens=cu_seqlens, start_positions=start_positions
    )
    loss_fused = loss_func(output_fused)
    loss_fused.backward()
    grad_fused = t.grad.detach().clone()
    t.grad = None

    torch.testing.assert_close(output_fused, output_unfused, **get_tol(dtype))
    torch.testing.assert_close(grad_fused, grad_unfused, **get_tol(dtype))

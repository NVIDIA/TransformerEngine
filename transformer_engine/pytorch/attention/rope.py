# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Rotary Position Embedding implementation of different types along with helper functions
"""
from typing import Optional, Tuple, Union
import torch

import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.fused_attn import QKVFormat


__all__ = ["RotaryPositionEmbedding", "apply_rotary_pos_emb"]


class RotaryPositionEmbedding(torch.nn.Module):
    """
    Implements Rotary Position Embedding from https://arxiv.org/abs/2104.09864.
    """

    def __init__(
        self,
        dim: int,
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[int] = None,
        pretrained_max_position_embeddings: Optional[int] = None,
        rotary_base: float = 10000.0,
        interleaved: bool = False,
    ):
        """
        Parameters
        ----------
        dim: int
            Rotary embedding dimension.
        rotary_percent: float, default = 1.0
            Percent of rotary dimension to use for rotary position embeddings.
        seq_len_interpolation_factor: int, default = None
            If not None, discrete positions will be interpolated by this factor via the trick in
            https://arxiv.org/abs/2306.15595
        pretrained_max_position_embeddings: int, default = None
            Pre-trained max_position_embeddings before position interpolation.
        rotary_base: float, default = 10000.0
            Base of the rotary position embedding.
        interleaved: bool, default = False
            Whether to use interleaved rotary position embedding.
        """
        super().__init__()
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.rotary_base = rotary_base
        inv_freq = 1.0 / (
            self.rotary_base
            ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device())
                / dim
            )
        )
        self.register_buffer("inv_freq", inv_freq)
        self.pretrained_max_position_embeddings = pretrained_max_position_embeddings
        self.interleaved = interleaved

    def forward(self, max_seq_len: int, offset: int = 0):
        """
        Create rotary position embedding frequencies.

        Parameters
        ----------
        max_seq_len: int
            Sequence length of a sample.
        offset: int, default = 0
            Fixed offset for frequencies.
        """
        seq = (
            torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            + offset
        )

        if (
            self.pretrained_max_position_embeddings is not None
            and self.seq_len_interpolation_factor is not None
        ):
            if (
                max_seq_len
                > self.pretrained_max_position_embeddings * self.seq_len_interpolation_factor
            ):
                # dynamic linear scaling (length > position we have learned)
                seq *= 1 / (max_seq_len / self.pretrained_max_position_embeddings)
            else:
                # fixed linear scaling
                seq *= 1 / self.seq_len_interpolation_factor

        freqs = torch.einsum("i , j -> i j", seq, self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        if not self.interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
                freqs.shape[0], -1
            )
        # emb [seq_length, .., dim]
        return emb.reshape(emb.size(0), 1, 1, emb.size(1))


class FusedRoPEFunc(torch.autograd.Function):
    """
    Function for FusedRoPE

    This implementation assumes the input tensor to be in `sbhd`, `bshd` or `thd` format and
    the RoPE tensor to be of shape (s, 1, 1, d). It accepts arbitrary memory layouts to avoid
    the expensive `.contiguous()` calls, thus it may not achieve the best memory access pattern.
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        start_positions: Union[torch.Tensor, None] = None,
        tensor_format: str = "sbhd",
        interleaved: bool = False,
        cu_seqlens: Union[torch.Tensor, None] = None,
        cp_size: int = 1,
        cp_rank: int = 0,
    ) -> torch.Tensor:
        """Fused RoPE forward."""

        if freqs.dtype != torch.float32:
            freqs = freqs.float()
        assert tensor_format in (
            "sbhd",
            "bshd",
            "thd",
        ), f"Unsupported tensor_format: {tensor_format}."
        output = tex.fused_rope_forward(
            t,
            freqs,
            start_positions,
            QKVFormat[tensor_format],
            interleaved,
            cu_seqlens,
            cp_size,
            cp_rank,
        )
        ctx.save_for_backward(freqs, cu_seqlens)
        ctx.tensor_format = tensor_format
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.interleaved = interleaved

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        """Fused RoPE backward."""
        freqs, cu_seqlens = ctx.saved_tensors
        grad_input = tex.fused_rope_backward(
            grad_output,
            freqs,
            QKVFormat[ctx.tensor_format],
            ctx.interleaved,
            cu_seqlens,
            ctx.cp_size,
            ctx.cp_rank,
        )

        return grad_input, None, None, None, None, None, None, None


def _rotate_half(x: torch.Tensor, interleaved: bool) -> torch.Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x: torch.Tensor. Input tensor.
        interleaved: bool. Whether to use interleaved rotary position embedding.

    Returns:
        Tensor: Tensor rotated half.
    """
    if not interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    # interleaved
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _apply_rotary_pos_emb_base(
    t: torch.Tensor,
    freqs: torch.Tensor,
    start_positions: torch.Tensor = None,
    tensor_format: str = "sbhd",
    interleaved: bool = False,
) -> torch.Tensor:
    """
    Base implementation of applying rotary positional embedding tensor to the input tensor.

    Parameters
    ----------
    t: torch.Tensor
        Input tensor of shape `[s, b, h, d]` or `[b, s, h, d]`, on which rotary positional
        embedding will be applied.
    freqs: torch.Tensor
        Rotary positional embedding tensor of shape `[s2, 1, 1, d2]` and dtype 'float',
        with `s2 >= s` and `d2 <= d`.
    start_positions: torch.Tensor, default = None.
        Tokens in a sequence `i` should be applied with position encoding offset by
        `start_positions[i]`. If `start_positions=None`, there's no offset.
    tensor_format: {'sbhd', 'bshd'}, default = 'sbhd'
        Should be `bshd` if `t` is of shape `[bs, seq, ...]`, or `sbhd` if `t` is of shape
        `[seq, bs, ...]`.
    interleaved: bool, default = False
        Whether to use interleaved rotary position embedding.
    """
    max_seq_len = freqs.shape[0]
    cur_seq_len = t.shape[1] if tensor_format == "bshd" else t.shape[0]

    # In case `start_positions` are provided, create a staggered `freqs` tensor
    # offset by the values in `start_positions`.
    # `start_positions` is only supported for `cp_size=1` and inference.
    if start_positions is not None:
        max_offset = torch.max(start_positions)
        assert (
            max_offset + cur_seq_len <= max_seq_len
        ), f"Rotary Embeddings only suppported up to {max_seq_len} sequence length!"

        # Stack staggered rope embeddings along the batch dimension
        freqs = torch.concatenate([freqs[i : i + cur_seq_len] for i in start_positions], dim=1)

        # Note that from this point, `freqs` has a shape `(s,b,1,d)`.

    # Only apply the rotary embeddings up to the sequence length of the running
    # input.
    assert (
        cur_seq_len <= max_seq_len
    ), f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
    freqs = freqs[:cur_seq_len]

    # [seq, 1, 1, dim] -> [1, seq, 1, dim] or
    # [seq, b, 1, dim] -> [b, seq, 1, dim]
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)
    # cos/sin first then dtype conversion for better precision
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * cos_) + (_rotate_half(t, interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def _get_freqs_on_this_cp_rank(
    freqs: torch.Tensor, seqlen: int, cp_size: int, cp_rank: int
) -> torch.Tensor:
    """Get the position embedding on the current context parallel rank.

    Args:
        freqs: torch.Tensor. Positional embedding tensor in shape `[s2, 1, 1, d2]`.
        seqlen: int. Length of the current sequence.
        cp_size: int. Context parallel world size.
        cp_rank: int. Context parallel rank.
    """
    if cp_size > 1:
        cp_seg = seqlen // 2
        full_seqlen = cp_size * seqlen
        return torch.cat(
            [
                freqs[cp_rank * cp_seg : (cp_rank + 1) * cp_seg],
                freqs[full_seqlen - (cp_rank + 1) * cp_seg : full_seqlen - cp_rank * cp_seg],
            ]
        )

    # cp_size == 1
    return freqs


def apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    start_positions: Union[torch.Tensor, None] = None,
    interleaved: bool = False,
    fused: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> torch.Tensor:
    """
    Apply rotary positional embedding tensor to the input tensor.

    Support matrix:
    Fused/Unfused:
        Training:
            qkv_formats:            "thd", "bshd", "sbhd"
            context parallel:       yes
            start_positions:        no
            interleaving:           yes
        Inference:
            qkv_formats:            "thd", "bshd", "sbhd"
            context parallelism:    no
            start_positions:        yes
            interleaving:            yes

    Parameters
    ----------
    t: torch.Tensor
        Input tensor of shape `[s, b, h, d]`, `[b, s, h, d]` or `[t, h, d]`, on which
        rotary positional embedding will be applied.
    freqs: torch.Tensor
        Rotary positional embedding tensor of shape `[s2, 1, 1, d2]` and dtype 'float',
        with `s2 >= s` and `d2 <= d`.
    start_positions: torch.Tensor, default = None.
        Tokens in a sequence `i` should be applied with position encoding offset by
        `start_positions[i]`. If `start_positions=None`, there's no offset.
    tensor_format: {'sbhd', 'bshd', 'thd'}, default = 'sbhd'
        is `bshd` if `t` is of shape `[bs, seq, ...]`, or `sbhd` if `t` is
        of shape `[seq, bs, ...]`. 'thd' is only supported when `fused` is True.
    interleaved: bool, default = False
        Whether to use interleaved rotary position embedding.
    fused: bool, default = False
        Whether to use a fused applying RoPE implementation.
    cu_seqlens: torch.Tensor, default = None.
        Cumulative sum of sequence lengths in a batch for `t`, with shape [b + 1] and
        dtype torch.int32. Only valid when `tensor_format` is 'thd'.
        Should be `cu_seqlens_padded` when cp_size > 1.
    cp_size: int, default = 1.
        Context parallel world size. Only valid when `tensor_format` is 'thd' and `fused` is True.
    cp_rank: int, default = 0.
        Context parallel rank. Only valid when `tensor_format` is 'thd' and `fused` is True.
    """

    # `start_positions` is only supported for `cp_size=1` and inference.
    assert not (
        cp_size > 1 and start_positions is not None
    ), """start_positions != None with CP SIZE > 1 is not supported!"""

    assert (
        tensor_format != "thd" or cu_seqlens is not None
    ), "cu_seqlens must not be None when tensor_format is 'thd'."

    if fused:
        return FusedRoPEFunc.apply(
            t, freqs, start_positions, tensor_format, interleaved, cu_seqlens, cp_size, cp_rank
        )

    # Unfused THD format
    if tensor_format == "thd":
        cu_seqlens = cu_seqlens // cp_size
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        # The following code essentially splits the `thd` tensor into corresponding
        # `s1hd` tensors (for each sequence) and applies rotary embedding to
        # those sequences individually.
        # Note that if `start_positions` is not `None`, then for each sequence,
        # it's corresponding rope offset is also supplied from `start_positions`
        # individually.
        return torch.cat(
            [
                _apply_rotary_pos_emb_base(
                    x.unsqueeze(1),
                    _get_freqs_on_this_cp_rank(freqs, x.size(0), cp_size, cp_rank),
                    start_positions=(
                        start_positions[idx : idx + 1] if start_positions is not None else None
                    ),
                    interleaved=interleaved,
                )
                for idx, x in enumerate(torch.split(t, seqlens))
            ]
        ).squeeze(1)

    # Unfused SBHD/BSHD format
    if tensor_format == "sbhd":
        seqlen = t.size(0)
    elif tensor_format == "bshd":
        seqlen = t.size(1)
    else:
        raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
    return _apply_rotary_pos_emb_base(
        t,
        _get_freqs_on_this_cp_rank(freqs, seqlen, cp_size, cp_rank),
        start_positions,
        tensor_format,
        interleaved=interleaved,
    )

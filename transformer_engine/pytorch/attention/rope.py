# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Rotary Position Embedding implementation of different types along with helper functions
"""
from functools import lru_cache
from typing import Optional, Tuple, Union, List
import torch

import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import QKVFormat
from transformer_engine.pytorch.device import te_device_type
from transformer_engine.pytorch.utils import canonicalize_device


__all__ = [
    "RotaryPositionEmbedding",
    "apply_rotary_pos_emb",
    "apply_fused_qkv_rotary_pos_emb",
    "DSv4RotaryEmbedding",
    "build_dsv4_rope_tables",
    "apply_dsa_rope",
    "apply_dsa_rope_eager",
]


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

        This function is particularly sensitive to the use of mixed precision, so we disable the
        autocast context if it is enabled.

        Parameters
        ----------
        max_seq_len: int
            Sequence length of a sample.
        offset: int, default = 0
            Fixed offset for frequencies.
        """
        with torch.autocast(enabled=False, device_type=te_device_type()):
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
        ctx.save_for_backward(freqs, cu_seqlens, start_positions)
        ctx.tensor_format = tensor_format
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.interleaved = interleaved

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        """Fused RoPE backward."""
        freqs, cu_seqlens, start_positions = ctx.saved_tensors
        grad_input = tex.fused_rope_backward(
            grad_output,
            freqs,
            start_positions,
            QKVFormat[ctx.tensor_format],
            ctx.interleaved,
            cu_seqlens,
            ctx.cp_size,
            ctx.cp_rank,
        )

        return grad_input, None, None, None, None, None, None, None, None


class FusedQKVRoPEFunc(torch.autograd.Function):
    """
    Function for FusedQKVRoPE

    This implementation accepts combined QKV tensor in `bshd` or `sbhd` format. Q and K RoPE tensors are the additional required inputs.
    The RoPE tensors should be of shape (s, 1, 1, d). It produces 3 outputs: Q, K after RoPE, V is the same as input.
    """

    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,
        q_freqs: torch.Tensor,
        k_freqs: torch.Tensor,
        qkv_split_arg_list: List[int],
        start_positions: Union[torch.Tensor, None] = None,
        tensor_format: str = "sbhd",
        interleaved: bool = False,
        cp_size: int = 1,
        cp_rank: int = 0,
    ) -> torch.Tensor:
        """Fused RoPE forward."""

        if q_freqs.dtype != torch.float32:
            q_freqs = q_freqs.float()
        if k_freqs.dtype != torch.float32:
            k_freqs = k_freqs.float()
        assert tensor_format in (
            "sbhd",
            "bshd",
        ), f"Unsupported tensor_format: {tensor_format}."
        assert qkv.is_contiguous(), "QKV Tensor should be contiguous."
        assert q_freqs.is_contiguous(), "q_freqs Tensor should be contiguous."
        assert k_freqs.is_contiguous(), "k_freqs Tensor should be contiguous."
        output = tex.fused_qkv_rope_forward(
            qkv,
            q_freqs,
            k_freqs,
            start_positions,
            qkv_split_arg_list,
            QKVFormat[tensor_format],
            interleaved,
            cp_size,
            cp_rank,
        )
        ctx.save_for_backward(q_freqs, k_freqs)
        ctx.tensor_format = tensor_format
        ctx.qkv_split_arg_list = qkv_split_arg_list
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.interleaved = interleaved
        return output

    @staticmethod
    def backward(
        ctx, grad_output_q: torch.Tensor, grad_output_k: torch.Tensor, grad_output_v: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        """Fused RoPE backward."""
        q_freqs, k_freqs = ctx.saved_tensors

        grad_output_q = grad_output_q.contiguous()
        grad_output_k = grad_output_k.contiguous()
        grad_output_v = grad_output_v.contiguous()

        grad_input = tex.fused_qkv_rope_backward(
            grad_output_q,
            grad_output_k,
            grad_output_v,
            q_freqs,
            k_freqs,
            ctx.qkv_split_arg_list,
            QKVFormat[ctx.tensor_format],
            ctx.interleaved,
            ctx.cp_size,
            ctx.cp_rank,
        )

        return grad_input, None, None, None, None, None, None, None, None


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
    tensor_format: str = "sbhd",
    interleaved: bool = False,
) -> torch.Tensor:
    """
    Base implementation of applying rotary positional embedding tensor to the input tensor.

    Parameters
    ----------
    t : torch.Tensor
        Input tensor of shape `[s, b, h, d]` or `[b, s, h, d]`, on which rotary positional
        embedding will be applied.
    freqs : torch.Tensor
        Rotary positional embedding tensor of shape `[s2, 1, 1, d2]` or `[s2, b, 1, d2]`
        and dtype 'float', with `s2 >= s` and `d2 <= d`.
    tensor_format : {'sbhd', 'bshd'}, default = 'sbhd'
        Should be `bshd` if `t` is of shape `[bs, seq, ...]`, or `sbhd` if `t` is of shape
        `[seq, bs, ...]`.
    interleaved : bool, default = False
        Whether to use interleaved rotary position embedding.
    """
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
        freqs: torch.Tensor. Positional embedding tensor of shape `[s2, 1, 1, d2]`.
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
    return freqs[:seqlen]


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
            start_positions:        yes
            interleaving:           yes
        Inference:
            qkv_formats:            "thd", "bshd", "sbhd"
            context parallelism:    no
            start_positions:        yes
            interleaving:           yes

    Parameters
    ----------
    t : torch.Tensor
        Input tensor of shape `[s, b, h, d]`, `[b, s, h, d]` or `[t, h, d]`, on which
        rotary positional embedding will be applied.
    freqs : torch.Tensor
        Rotary positional embedding tensor of shape `[s2, 1, 1, d2]` and dtype 'float',
        with `s2 >= s` and `d2 <= d`.
    start_positions : torch.Tensor, default = None.
        Tokens in a sequence `i` should be applied with position encoding offset by
        `start_positions[i]`. If `start_positions=None`, there's no offset.
    tensor_format : {'sbhd', 'bshd', 'thd'}, default = 'sbhd'
        is `bshd` if `t` is of shape `[bs, seq, ...]`, or `sbhd` if `t` is
        of shape `[seq, bs, ...]`. 'thd' is only supported when `fused` is True.
    interleaved : bool, default = False
        Whether to use interleaved rotary position embedding.
    fused : bool, default = False
        Whether to use a fused applying RoPE implementation.
    cu_seqlens : torch.Tensor, default = None.
        Cumulative sum of sequence lengths in a batch for `t`, with shape [b + 1] and
        dtype torch.int32. Only valid when `tensor_format` is 'thd'.
        Should be `cu_seqlens_padded` when cp_size > 1.
    cp_size : int, default = 1.
        Context parallel world size. Only valid when `tensor_format` is 'thd' and `fused` is True.
    cp_rank : int, default = 0.
        Context parallel rank. Only valid when `tensor_format` is 'thd' and `fused` is True.
    """
    assert (
        tensor_format != "thd" or cu_seqlens is not None
    ), "cu_seqlens must not be None when tensor_format is 'thd'."

    # Fused apply rope logic for THD/BSHD/SBHD formats
    if fused:
        return FusedRoPEFunc.apply(
            t, freqs, start_positions, tensor_format, interleaved, cu_seqlens, cp_size, cp_rank
        )

    # Unfused apply rope logic for THD format
    if tensor_format == "thd":
        cu_seqlens = cu_seqlens // cp_size
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        # The following code essentially splits the `thd` tensor into corresponding
        # `s1hd` tensors (for each sequence) and applies rotary embedding to
        # those sequences individually.
        # Note that if `start_positions` is not `None`, then for each sequence,
        # the freqs supplied are offset by the corresponding `start_positions` value.
        return torch.cat(
            [
                _apply_rotary_pos_emb_base(
                    x.unsqueeze(1),
                    _get_freqs_on_this_cp_rank(
                        (
                            freqs[start_positions[idx] :] if start_positions is not None else freqs
                        ),  # offset the freqs
                        x.size(0),
                        cp_size,
                        cp_rank,
                    ),
                    interleaved=interleaved,
                )
                for idx, x in enumerate(torch.split(t, seqlens))
            ]
        ).squeeze(1)

    # Unfused apply rope logic for SBHD/BSHD format follows ...

    if tensor_format == "sbhd":
        seqlen = t.size(0)
    elif tensor_format == "bshd":
        seqlen = t.size(1)
    else:
        raise ValueError(f"Unsupported tensor_format: {tensor_format}.")

    if start_positions is not None:
        max_offset = torch.max(start_positions)
        assert (
            max_offset + seqlen * cp_size <= freqs.shape[0]
        ), f"Rotary Embeddings only suppported up to {freqs.shape[0]} sequence length!"

        # Stack staggered rope embeddings along the batch dimension
        freqs = torch.concatenate([freqs[i : i + seqlen * cp_size] for i in start_positions], dim=1)
        # Note that from this point, `freqs` has a shape `(s,b,1,d)`.

    return _apply_rotary_pos_emb_base(
        t,
        _get_freqs_on_this_cp_rank(freqs, seqlen, cp_size, cp_rank),
        tensor_format,
        interleaved=interleaved,
    )


def apply_fused_qkv_rotary_pos_emb(
    qkv: torch.Tensor,
    q_freqs: torch.Tensor,
    k_freqs: torch.Tensor,
    qkv_split_arg_list: List[int],
    tensor_format: str = "sbhd",
    start_positions: Union[torch.Tensor, None] = None,
    interleaved: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,  # pylint: disable=unused-argument
    cp_size: int = 1,
    cp_rank: int = 0,
) -> torch.Tensor:
    """
    Apply rotary positional embedding tensor to the input qkv tensor.

    Support matrix:
    Fused:
        Training:
            qkv_formats:            "bshd", "sbhd"
            context parallel:       yes
            start_positions:        no
            interleaving:           yes
        Inference:
            qkv_formats:            "bshd", "sbhd"
            context parallelism:    no
            start_positions:        yes
            interleaving:           yes

    Parameters
    ----------
    qkv : torch.Tensor
        Input tensor of shape `[s, b, h, d]` or `[b, s, h, d]`, on which
        rotary positional embedding will be applied. This tensor has q, k, v concatenated
        along the last dimension.
    q_freqs : torch.Tensor
        Rotary positional embedding Q tensor of shape `[s2, 1, 1, d2]` and dtype 'float',
        with `s2 >= s` and `d2 <= d`.
    k_freqs : torch.Tensor
        Rotary positional embedding K tensor of shape `[s2, 1, 1, d2]` and dtype 'float',
        with `s2 >= s` and `d2 <= d`.
    qkv_split_arg_list : List[int]
        List of integers that specify the split of the qkv tensor. The list should have 3 elements,
        the first element is the number of elements in the q tensor, the second element is the number
        of elements in the k tensor, and the third element is the number of elements in the v tensor.
        The sum of the elements in the list should be equal to the last dimension of the qkv tensor.
    start_positions : torch.Tensor, default = None.
        Tokens in a sequence `i` should be applied with position encoding offset by
        `start_positions[i]`. If `start_positions=None`, there's no offset.
    tensor_format : {'sbhd', 'bshd'}, default = 'sbhd'
        is `bshd` if `qkv` is of shape `[bs, seq, ...]`, or `sbhd` if `qkv` is
        of shape `[seq, bs, ...]`.
    interleaved : bool, default = False
        Whether to use interleaved rotary position embedding.
    cp_size : int, default = 1.
        Context parallel world size.
    cp_rank : int, default = 0.
        Context parallel rank.
    """

    # `start_positions` is only supported for `cp_size=1` and inference.
    assert not (
        cp_size > 1 and start_positions is not None
    ), """start_positions != None with CP SIZE > 1 is not supported!"""

    assert tensor_format != "thd", "'thd' tensor_format not supported currently."

    return FusedQKVRoPEFunc.apply(
        qkv,
        q_freqs,
        k_freqs,
        qkv_split_arg_list,
        start_positions,
        tensor_format,
        interleaved,
        cp_size,
        cp_rank,
    )


def _build_rope_tables(seq_len, inv_freq, *, interleaved):
    """Build FP32 cosine/sine tables for the DSv4 and MLA wrappers.

    Callers supply FP32 inverse frequencies [rotary_width / 2]. Both outputs have
    shape [seq_len, rotary_width]. Interleaved layout repeats compact values
    [a, b] as [a, a, b, b] for DSv4; otherwise duplicate halves as [a, b, a, b]
    for MLA.
    """
    positions = torch.arange(seq_len, device=inv_freq.device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    cos, sin = angles.cos(), angles.sin()
    if interleaved:
        return cos.repeat_interleave(2, dim=-1), sin.repeat_interleave(2, dim=-1)
    return torch.cat((cos, cos), dim=-1), torch.cat((sin, sin), dim=-1)


# DSA: trailing interleaved pairs with packed sequence positions.


def _validate_dsa_rope_width(width, head_dim=None):
    """Require the power-of-two rotary width used by the Triton Q RoPE kernels."""
    if width < 4 or width & (width - 1):
        raise ValueError("DSv4 RoPE width must be a power of two >= 4.")
    if head_dim is not None and width > head_dim:
        raise ValueError("DSv4 RoPE width must not exceed the input head dimension.")


def build_dsv4_rope_tables(seq, ratio, width, theta, device):
    """Return FP32 token/window pairs for a power-of-two rotary width >= 4."""
    _validate_dsa_rope_width(width)
    inv_freq = theta ** (-torch.arange(0, width, 2, device=device, dtype=torch.float32) / width)
    cos, sin = _build_rope_tables(seq, inv_freq, interleaved=True)
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    window_positions = torch.arange(seq // ratio, device=device) * ratio
    return (cos, sin), (cos[:, window_positions], sin[:, window_positions])


class DSv4RotaryEmbedding(torch.nn.Module):
    """Cache FP32 token frequencies and compressed-window start frequencies.

    Buffers are not saved in state_dict. Module.to() moves the cache while keeping
    its frequencies in FP32, even when a different floating-point dtype is requested.

    Parameters
    ----------
    ratio : int
        Number of input tokens per compressed window.
    width : int
        Rotary dimension, a power of two >= 4.
    theta : float
        Rotary frequency base.
    device : torch.device or str
        Device for the optional initial cache.
    max_seqlen : int, optional
        Initial cache length. Otherwise allocate lazily on the first forward call.
    """

    def __init__(self, ratio, width, theta, device, max_seqlen=None):
        super().__init__()
        _validate_dsa_rope_width(width)
        self.ratio, self.width, self.theta = ratio, width, theta
        cos, sin = (None, None)
        if max_seqlen is not None:
            (cos, sin), _ = build_dsv4_rope_tables(max_seqlen, ratio, width, theta, device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _apply(self, fn):
        cos, sin = self.cos, self.sin
        super()._apply(fn)
        if cos is not None:
            # Module.to(bfloat16) must not round the reusable frequencies.
            self.cos = cos.to(device=self.cos.device)
            self.sin = sin.to(device=self.sin.device)
        return self

    def forward(self, seq, device):
        """Return FP32 token and compressed ``(cos, sin)`` pairs.

        Parameters
        ----------
        seq : int
            Token sequence length. Grow the cache if needed.
        device : torch.device or str
            Requested output device. Move an existing cache when it differs.

        Returns
        -------
        tuple
            Token pair with shape [1, seq, 1, width] and compressed pair with shape
            [1, seq // ratio, 1, width]. Compressed positions are window starts
            0, ratio, 2 * ratio, ... . Returned tensors are views of the cache.
        """
        device = canonicalize_device(device)
        if self.cos is None or self.cos.shape[1] < seq:
            (self.cos, self.sin), _ = build_dsv4_rope_tables(
                seq, self.ratio, self.width, self.theta, device
            )
        elif self.cos.device != device:
            self.cos = self.cos.to(device=device)
            self.sin = self.sin.to(device=device)
        n_comp = seq // self.ratio
        return (self.cos[:, :seq], self.sin[:, :seq]), (
            self.cos[:, : n_comp * self.ratio : self.ratio],
            self.sin[:, : n_comp * self.ratio : self.ratio],
        )


def apply_dsa_rope_eager(x, cos, sin, cu_seqlens=None):
    """Rotate trailing pairs in FP32 for a power-of-two width >= 4, then restore dtype."""
    width = cos.shape[-1]
    _validate_dsa_rope_width(width, x.shape[-1])
    if x.ndim == 3:
        if cu_seqlens is None:
            raise ValueError("Packed RoPE requires cu_seqlens.")
        token_indices = torch.arange(x.shape[0], device=x.device)
        sequences = torch.bucketize(token_indices, cu_seqlens[1:], right=True)
        positions = token_indices - cu_seqlens[sequences]
        # Compressed tensors may have output padding after cu_seqlens[-1].
        positions = torch.where(token_indices < cu_seqlens[-1], positions, 0)
        cos, sin = cos[0, positions], sin[0, positions]
    tail = x[..., -width:]
    pair = torch.stack((-tail[..., 1::2], tail[..., 0::2]), -1).flatten(-2)
    rotated = (tail.float() * cos + pair.float() * sin).to(x.dtype)
    return torch.cat((x[..., :-width], rotated), -1)


@lru_cache(maxsize=1)
def _triton_rope_module():
    """Load optional Triton Q RoPE kernels for DSA and MLA and separate MLA KV kernels."""
    try:
        from transformer_engine.common.triton import dsa_mla_rope
    except ImportError:
        return None
    return dsa_mla_rope


def _can_use_dsa_rope_triton(x, cos, sin, cu_seqlens):
    """Keep unsupported inputs on the behaviorally identical eager path."""
    return (
        cu_seqlens is not None
        and x.is_cuda
        and x.ndim in (3, 4)
        and x.stride(-1) == 1
        and cos.shape == sin.shape
        and not cos.requires_grad
        and not sin.requires_grad
        and _triton_rope_module() is not None
    )


def _dsa_triton_rope_table(table):
    """Convert TE's repeated pairs to the Triton kernel's duplicated-half tables."""
    pairs = table[0, :, 0, ::2]
    return torch.cat((pairs, pairs), dim=-1)


def _launch_rope_q_triton(
    x, cos, sin, head_dim_nope, head_dim_rope, cu_seqlens, interleaved, module, backward
):
    kernel = module.rotary_bwd_q_kernel if backward else module.rotary_fwd_q_kernel
    heads = x.shape[-2]
    if cu_seqlens is not None:
        x = x.reshape(-1, heads, x.shape[-1])
        total, batch, sequences = x.shape[0], 1, cu_seqlens.numel() - 1
        token_stride, head_stride = x.stride(0), x.stride(1)
    else:
        seq_len, batch, heads, _ = x.shape
        total, sequences = seq_len * batch, None
        token_stride, head_stride = x.stride(1), x.stride(2)

    def grid(meta):
        return total, module.triton.cdiv(heads, meta["BLOCK_H"])

    kernel[grid](
        x,
        cos,
        sin,
        head_dim_nope,
        head_dim_rope,
        heads,
        batch,
        sequences,
        cu_seqlens,
        None,
        token_stride,
        head_stride,
        cos.stride(0),
        sin.stride(0),
        0,
        1,
        INVERSE=False,
        REMOVE_INTERLEAVING=interleaved,
    )


class _RoPEQTriton(torch.autograd.Function):
    """Apply Q RoPE with independent positioning, layout and mutation policies."""

    @staticmethod
    def forward(ctx, x, cos, sin, head_dim_nope, head_dim_rope, cu_seqlens, interleaved, mutation):
        """Rotate a copy or the input according to the wrapper's mutation policy."""
        output = x.clone(memory_format=torch.contiguous_format) if mutation == "copy" else x
        module = _triton_rope_module()
        _launch_rope_q_triton(
            output, cos, sin, head_dim_nope, head_dim_rope, cu_seqlens, interleaved, module, False
        )
        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.module = module
        ctx.head_dim_nope, ctx.head_dim_rope = head_dim_nope, head_dim_rope
        ctx.interleaved, ctx.mutation = interleaved, mutation
        if mutation == "marked_inplace":
            ctx.mark_dirty(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Preserve DSA's gradient copy and MLA's existing gradient mutation."""
        cos, sin, cu_seqlens = ctx.saved_tensors
        # Legacy MLA mutates input without mark_dirty and also reuses incoming gradients.
        # Keep that behavior separate from tensor layout and sequence positioning.
        grad_input = (
            grad_output
            if ctx.mutation == "legacy_inplace"
            else grad_output.clone(memory_format=torch.contiguous_format)
        )
        _launch_rope_q_triton(
            grad_input,
            cos,
            sin,
            ctx.head_dim_nope,
            ctx.head_dim_rope,
            cu_seqlens,
            ctx.interleaved,
            ctx.module,
            True,
        )
        return grad_input, None, None, None, None, None, None, None


def apply_dsa_rope(x, cos, sin, cu_seqlens=None, *, allow_inplace=False):
    """Rotate trailing interleaved pairs, using Triton when its contract is satisfied.

    By default, preserve the input. With allow_inplace=True, the caller must supply
    a fresh buffer safe for autograd mutation. This requires a contiguous Triton
    input that is not a leaf requiring gradients, otherwise raise ValueError.
    Both paths require a power-of-two rotary width >= 4 that fits the input head.
    """
    use_triton = _can_use_dsa_rope_triton(x, cos, sin, cu_seqlens)
    if allow_inplace and (
        not use_triton or not x.is_contiguous() or (x.is_leaf and x.requires_grad)
    ):
        raise ValueError(
            "allow_inplace=True requires a contiguous Triton-compatible input "
            "that is not a leaf requiring gradients."
        )
    if use_triton:
        _validate_dsa_rope_width(cos.shape[-1], x.shape[-1])
        return _RoPEQTriton.apply(
            x,
            _dsa_triton_rope_table(cos),
            _dsa_triton_rope_table(sin),
            x.shape[-1] - cos.shape[-1],
            cos.shape[-1],
            cu_seqlens,
            True,
            "marked_inplace" if allow_inplace else "copy",
        )
    return apply_dsa_rope_eager(x, cos, sin, cu_seqlens)


# MLA adapters from Megatron-LM fused_mla_yarn_rope_apply.py.
# The default tables use plain RoPE (base=10000), not YaRN scaling.
# Unlike DSA, these SBHD adapters convert interleaved input to split-half output.


_MLA_HEAD_DIM_ROPE = 64
_MLA_HEAD_DIM_NOPE = 128
_MLA_HEAD_DIM_V = 128
_MLA_ROTARY_BASE = 10000


def build_mla_rope_tables(
    seq_len: int,
    emb_dim: int = _MLA_HEAD_DIM_ROPE,
    base: int = _MLA_ROTARY_BASE,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build FP32 duplicated-half MLA tables with plain RoPE frequencies."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, emb_dim, 2, dtype=torch.float32, device=device) / emb_dim)
    )
    cos, sin = _build_rope_tables(seq_len, inv_freq, interleaved=False)
    return cos.contiguous(), sin.contiguous()


def _mla_flattened_token_stride(tensor: torch.Tensor) -> int:
    if tensor.dim() == 4:
        return tensor.stride(1)
    return tensor.stride(0)


class _MLARoPEKVTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, k_pos_emb, cos, sin, head_dim_nope, head_dim_rope, head_dim_v):
        """Split SBHD KV and append shared rotated position channels to each key head."""
        s, b, nheads, _ = kv.shape
        total = s * b
        module = _triton_rope_module()

        o_key = kv.new_empty(s, b, nheads, head_dim_nope + head_dim_rope)
        o_value = kv.new_empty(s, b, nheads, head_dim_v)

        def grid_kv(meta):
            return total, module.triton.cdiv(nheads, meta["BLOCK_H"])

        module.rotary_fwd_kv_kernel[grid_kv](
            kv,
            k_pos_emb,
            o_key,
            o_value,
            cos,
            sin,
            head_dim_rope,
            head_dim_nope,
            head_dim_v,
            nheads,
            b,
            None,
            None,
            _mla_flattened_token_stride(kv),
            kv.stride(2),
            _mla_flattened_token_stride(k_pos_emb),
            _mla_flattened_token_stride(o_key),
            o_key.stride(2),
            _mla_flattened_token_stride(o_value),
            o_value.stride(2),
            0,
            1,
        )

        ctx.save_for_backward(cos, sin)
        ctx.head_dim_nope = head_dim_nope
        ctx.head_dim_rope = head_dim_rope
        ctx.head_dim_v = head_dim_v
        ctx.nheads = nheads
        ctx.s = s
        ctx.b = b
        return o_key, o_value

    @staticmethod
    def backward(ctx, dk_out, dv_out):
        """Gather KV gradients and reduce shared positional gradients across heads."""
        cos, sin = ctx.saved_tensors
        s, b, nheads = ctx.s, ctx.b, ctx.nheads
        ndp, ndr, ndv = ctx.head_dim_nope, ctx.head_dim_rope, ctx.head_dim_v
        total = s * b
        module = _triton_rope_module()

        d_kv = dk_out.new_empty(s, b, nheads, ndp + ndv)
        d_emb = dk_out.new_empty(s, b, 1, ndr)

        def grid_kv(meta):
            return total, module.triton.cdiv(nheads, meta["BLOCK_H"])

        module.rotary_bwd_kv_kernel[grid_kv](
            dk_out,
            dv_out,
            d_kv,
            d_emb,
            cos,
            sin,
            ndr,
            ndp,
            ndv,
            nheads,
            b,
            None,
            None,
            _mla_flattened_token_stride(dk_out),
            dk_out.stride(2),
            _mla_flattened_token_stride(dv_out),
            dv_out.stride(2),
            _mla_flattened_token_stride(d_kv),
            d_kv.stride(2),
            _mla_flattened_token_stride(d_emb),
            0,
            1,
        )
        return d_kv, d_emb, None, None, None, None, None


def _apply_mla_rope_q_with_tables(
    q: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    head_dim_nope: int = _MLA_HEAD_DIM_NOPE,
    head_dim_rope: int = _MLA_HEAD_DIM_ROPE,
) -> torch.Tensor:
    if _triton_rope_module() is not None:
        return _RoPEQTriton.apply(
            q,
            cos_table,
            sin_table,
            head_dim_nope,
            head_dim_rope,
            None,
            False,
            "legacy_inplace",
        )
    return _apply_mla_rope_q_eager(q, cos_table, sin_table, head_dim_nope, head_dim_rope)


def _apply_mla_rope_kv_with_tables(
    kv: torch.Tensor,
    k_pos_emb: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    head_dim_nope: int = _MLA_HEAD_DIM_NOPE,
    head_dim_rope: int = _MLA_HEAD_DIM_ROPE,
    head_dim_v: int = _MLA_HEAD_DIM_V,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _triton_rope_module() is not None:
        return _MLARoPEKVTriton.apply(
            kv,
            k_pos_emb,
            cos_table,
            sin_table,
            head_dim_nope,
            head_dim_rope,
            head_dim_v,
        )
    return _apply_mla_rope_kv_eager(
        kv,
        k_pos_emb,
        cos_table,
        sin_table,
        head_dim_nope,
        head_dim_rope,
        head_dim_v,
    )


def apply_mla_rope_q(
    q: torch.Tensor,
    head_dim_nope: int = _MLA_HEAD_DIM_NOPE,
    head_dim_rope: int = _MLA_HEAD_DIM_ROPE,
    base: int = _MLA_ROTARY_BASE,
    cos_table: torch.Tensor | None = None,
    sin_table: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate trailing SBHD query channels from interleaved to split-half order."""
    if cos_table is None or sin_table is None:
        s = q.shape[0]
        cos_table, sin_table = build_mla_rope_tables(
            s,
            emb_dim=head_dim_rope,
            base=base,
            device=q.device,
        )
    return _apply_mla_rope_q_with_tables(
        q,
        cos_table,
        sin_table,
        head_dim_nope,
        head_dim_rope,
    )


def apply_mla_rope_kv(
    kv: torch.Tensor,
    k_pos_emb: torch.Tensor,
    head_dim_nope: int = _MLA_HEAD_DIM_NOPE,
    head_dim_rope: int = _MLA_HEAD_DIM_ROPE,
    head_dim_v: int = _MLA_HEAD_DIM_V,
    base: int = _MLA_ROTARY_BASE,
    cos_table: torch.Tensor | None = None,
    sin_table: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split SBHD KV and rotate shared positional keys into split-half order."""
    if cos_table is None or sin_table is None:
        s = kv.shape[0]
        cos_table, sin_table = build_mla_rope_tables(
            s,
            emb_dim=head_dim_rope,
            base=base,
            device=kv.device,
        )
    return _apply_mla_rope_kv_with_tables(
        kv,
        k_pos_emb,
        cos_table,
        sin_table,
        head_dim_nope,
        head_dim_rope,
        head_dim_v,
    )


def apply_mla_rope(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_pos_emb: torch.Tensor,
    head_dim_nope: int = _MLA_HEAD_DIM_NOPE,
    head_dim_rope: int = _MLA_HEAD_DIM_ROPE,
    head_dim_v: int = _MLA_HEAD_DIM_V,
    base: int = _MLA_ROTARY_BASE,
    cos_table: torch.Tensor | None = None,
    sin_table: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the MLA query and shared-key rotations using the same tables."""
    if cos_table is None or sin_table is None:
        s = q.shape[0]
        cos_table, sin_table = build_mla_rope_tables(
            s,
            emb_dim=head_dim_rope,
            base=base,
            device=q.device,
        )
    q = _apply_mla_rope_q_with_tables(q, cos_table, sin_table, head_dim_nope, head_dim_rope)
    k, v = _apply_mla_rope_kv_with_tables(
        kv,
        k_pos_emb,
        cos_table,
        sin_table,
        head_dim_nope,
        head_dim_rope,
        head_dim_v,
    )
    return q, k, v


def _rotate_mla_interleaved_to_neox(
    x: torch.Tensor, cos_table: torch.Tensor, sin_table: torch.Tensor
) -> torch.Tensor:
    cos_ = cos_table[:, None, None, :].to(x.dtype)
    sin_ = sin_table[:, None, None, :].to(x.dtype)
    half_dim = x.shape[-1] // 2
    x_1 = x[..., 0::2]
    x_2 = x[..., 1::2]
    x_left = x_1 * cos_[..., :half_dim] - x_2 * sin_[..., :half_dim]
    x_right = x_2 * cos_[..., half_dim:] + x_1 * sin_[..., half_dim:]
    return torch.cat((x_left, x_right), dim=-1)


def _apply_mla_rope_q_eager(q, cos_table, sin_table, head_dim_nope, head_dim_rope):
    q_nope = q[..., :head_dim_nope]
    q_rope = q[..., head_dim_nope : head_dim_nope + head_dim_rope]
    q_rope = _rotate_mla_interleaved_to_neox(q_rope, cos_table, sin_table)
    return torch.cat((q_nope, q_rope), dim=-1)


def _apply_mla_rope_kv_eager(
    kv,
    k_pos_emb,
    cos_table,
    sin_table,
    head_dim_nope,
    head_dim_rope,  # pylint: disable=unused-argument
    head_dim_v,
):
    k_nope = kv[..., :head_dim_nope]
    v = kv[..., head_dim_nope : head_dim_nope + head_dim_v]
    k_rope = _rotate_mla_interleaved_to_neox(k_pos_emb, cos_table, sin_table).expand(
        -1, -1, kv.shape[2], -1
    )
    return torch.cat((k_nope, k_rope), dim=-1), v

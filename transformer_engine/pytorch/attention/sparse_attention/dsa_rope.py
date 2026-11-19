# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Trailing-channel, interleaved DSA RoPE for full-sequence attention."""

from functools import lru_cache

import torch

from transformer_engine.pytorch.utils import canonicalize_device


def rotary_embeddings(seq, ratio, width, theta, device):
    """Return token and compressed-window (cos, sin) pairs in FP32."""
    inv_freq = theta ** (-torch.arange(0, width, 2, device=device, dtype=torch.float32) / width)
    angles = torch.outer(torch.arange(seq, device=device, dtype=torch.float32), inv_freq)
    cos = angles.cos().repeat_interleave(2, dim=-1)[None, :, None, :]
    sin = angles.sin().repeat_interleave(2, dim=-1)[None, :, None, :]
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
        Even rotary dimension.
    theta : float
        Rotary frequency base.
    device : torch.device or str
        Device for the optional initial cache.
    max_seqlen : int, optional
        Initial cache length. Otherwise allocate lazily on the first forward call.
    """

    def __init__(self, ratio, width, theta, device, max_seqlen=None):
        super().__init__()
        self.ratio, self.width, self.theta = ratio, width, theta
        cos, sin = (None, None)
        if max_seqlen is not None:
            (cos, sin), _ = rotary_embeddings(max_seqlen, ratio, width, theta, device)
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
            (self.cos, self.sin), _ = rotary_embeddings(
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
    """Rotate trailing interleaved pairs in FP32, then restore input dtype."""
    if x.ndim == 3:
        if cu_seqlens is None:
            raise ValueError("Packed RoPE requires cu_seqlens.")
        token_indices = torch.arange(x.shape[0], device=x.device)
        sequences = torch.bucketize(token_indices, cu_seqlens[1:], right=True)
        positions = token_indices - cu_seqlens[sequences]
        # Compressed tensors may have output padding after cu_seqlens[-1].
        positions = torch.where(token_indices < cu_seqlens[-1], positions, 0)
        cos, sin = cos[0, positions], sin[0, positions]
    width = cos.shape[-1]
    tail = x[..., -width:]
    pair = torch.stack((-tail[..., 1::2], tail[..., 0::2]), -1).flatten(-2)
    rotated = (tail.float() * cos + pair.float() * sin).to(x.dtype)
    return torch.cat((x[..., :-width], rotated), -1)


@lru_cache(maxsize=1)
def _triton_rope_module():
    """Load the optional Triton DSA RoPE implementation."""
    try:
        from transformer_engine.common.triton import dsa_rope
    except ImportError:
        return None
    return dsa_rope


def _can_use_triton(x, cos, sin, cu_seqlens):
    """Keep unsupported inputs on the behaviorally identical eager path."""
    return (
        cu_seqlens is not None
        and x.is_cuda
        and x.ndim in (3, 4)
        and x.stride(-1) == 1
        and cos.shape == sin.shape
        and cos.shape[-1] >= 4
        # The Triton kernels use tl.arange over half the rotary width.
        and cos.shape[-1] & (cos.shape[-1] - 1) == 0
        and not cos.requires_grad
        and not sin.requires_grad
        and _triton_rope_module() is not None
    )


def _rope_tables(table):
    """Convert TE's repeated pairs to the Triton kernel's duplicated-half tables."""
    pairs = table[0, :, 0, ::2]
    return torch.cat((pairs, pairs), dim=-1)


def _launch_triton(packed, cos, sin, cu_seqlens, module, backward):
    kernel = module.rotary_bwd_q_kernel if backward else module.rotary_fwd_q_kernel
    heads, dim = packed.shape[-2:]

    def grid(meta):
        return packed.shape[0], module.triton.cdiv(heads, meta["BLOCK_H"])

    kernel[grid](
        packed,
        cos,
        sin,
        dim - cos.shape[-1],
        cos.shape[-1],
        heads,
        1,
        cu_seqlens.numel() - 1,
        cu_seqlens,
        None,
        packed.stride(0),
        packed.stride(1),
        cos.stride(0),
        sin.stride(0),
        0,
        1,
        INVERSE=False,
        REMOVE_INTERLEAVING=True,
    )


class _TritonRotary(torch.autograd.Function):
    """Apply the shared Triton kernels with optional input mutation."""

    @staticmethod
    def forward(ctx, x, cos, sin, cu_seqlens, module, allow_inplace):
        """Rotate a contiguous buffer and save the frequency tables for backward."""
        # Contiguity ensures the packed reshape aliases the buffer we return.
        inplace = allow_inplace and x.is_contiguous()
        output = x if inplace else x.clone(memory_format=torch.contiguous_format)
        packed = output.reshape(-1, x.shape[-2], x.shape[-1])
        _launch_triton(packed, cos, sin, cu_seqlens, module, False)
        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.module = module
        ctx.shape = x.shape
        if inplace:
            ctx.mark_dirty(x)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Apply the backward rotation to a copy of the output gradient."""
        cos, sin, cu_seqlens = ctx.saved_tensors
        grad_input = grad_output.clone(memory_format=torch.contiguous_format)
        packed = grad_input.reshape(-1, ctx.shape[-2], ctx.shape[-1])
        _launch_triton(packed, cos, sin, cu_seqlens, ctx.module, True)
        return grad_input, None, None, None, None, None


def apply_dsa_rope(x, cos, sin, cu_seqlens=None, *, allow_inplace=False):
    """Rotate trailing interleaved pairs, using Triton when its contract is satisfied.

    By default, preserve the input. With allow_inplace=True, the caller must supply
    a fresh buffer safe for autograd mutation; only contiguous Triton inputs are
    mutated. Noncontiguous inputs and the eager fallback remain out of place.
    """
    if _can_use_triton(x, cos, sin, cu_seqlens):
        return _TritonRotary.apply(
            x,
            _rope_tables(cos),
            _rope_tables(sin),
            cu_seqlens,
            _triton_rope_module(),
            allow_inplace,
        )
    return apply_dsa_rope_eager(x, cos, sin, cu_seqlens)

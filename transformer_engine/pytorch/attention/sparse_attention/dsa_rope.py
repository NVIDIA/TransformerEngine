# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DSv4's trailing-channel, interleaved RoPE for full-sequence attention."""

from functools import lru_cache

import torch


def rotary_embeddings(seq, ratio, width, theta, device):
    """Return token and compressed-window (cos, sin) pairs in FP32."""
    inv_freq = theta ** (-torch.arange(0, width, 2, device=device, dtype=torch.float32) / width)
    angles = torch.outer(torch.arange(seq, device=device, dtype=torch.float32), inv_freq)
    cos = angles.cos().repeat_interleave(2, dim=-1)[None, :, None, :]
    sin = angles.sin().repeat_interleave(2, dim=-1)[None, :, None, :]
    window_positions = torch.arange(seq // ratio, device=device) * ratio
    return (cos, sin), (cos[:, window_positions], sin[:, window_positions])


class _DSv4RotaryEmbedding(torch.nn.Module):
    """Cache shared FP32 token frequencies; compressed positions are window starts."""

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
        """Return cached token and compressed-window frequency pairs."""
        if self.cos is None or self.cos.shape[1] < seq:
            (self.cos, self.sin), _ = rotary_embeddings(
                seq, self.ratio, self.width, self.theta, device
            )
        n_comp = seq // self.ratio
        return (self.cos[:, :seq], self.sin[:, :seq]), (
            self.cos[:, : n_comp * self.ratio : self.ratio],
            self.sin[:, : n_comp * self.ratio : self.ratio],
        )


def _apply_rotary_eager(x, cos, sin, cu_seqlens=None):
    """Rotate trailing interleaved pairs in FP32, then restore input dtype."""
    if x.ndim == 3:
        if cu_seqlens is None:
            raise ValueError("Packed RoPE requires cu_seqlens.")
        rows = torch.arange(x.shape[0], device=x.device)
        sequences = torch.bucketize(rows, cu_seqlens[1:], right=True)
        positions = rows - cu_seqlens[sequences]
        # Compressed tensors may have unused capacity past the final prefix.
        positions = torch.where(rows < cu_seqlens[-1], positions, 0)
        cos, sin = cos[0, positions], sin[0, positions]
    width = cos.shape[-1]
    tail = x[..., -width:]
    pair = torch.stack((-tail[..., 1::2], tail[..., 0::2]), -1).flatten(-2)
    rotated = (tail.float() * cos + pair.float() * sin).to(x.dtype)
    return torch.cat((x[..., :-width], rotated), -1)


@lru_cache(maxsize=1)
def _triton_rope_module():
    """Load the optional Triton implementation only when DSv4 can use it."""
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
        and cos.shape[-1] % 4 == 0
        and not cos.requires_grad
        and not sin.requires_grad
        and _triton_rope_module() is not None
    )


def _rope_tables(table):
    """Convert TE's repeated pairs to the Triton kernel's duplicated-half tables."""
    pairs = table[0, :, 0, ::2]
    return torch.cat((pairs, pairs), dim=-1)


def _apply_rotary_triton(x, cos, sin, cu_seqlens):
    """Apply Triton RoPE out of place so callers retain the eager aliasing contract."""
    module = _triton_rope_module()
    if module is None:
        raise RuntimeError("Triton RoPE is unavailable.")
    return _TritonRotary.apply(x, _rope_tables(cos), _rope_tables(sin), cu_seqlens, module)


def _launch_triton(packed, cos, sin, cu_seqlens, module, backward):
    kernel = (
        module._autotuned_mla_rope_bwd_inplace_kernel
        if backward
        else module._autotuned_mla_rope_fwd_inplace_kernel
    )
    heads, dim = packed.shape[-2:]
    grid = lambda meta: (packed.shape[0], module.triton.cdiv(heads, meta["BLOCK_H"]))
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
    """Pair the retained in-place Triton kernels behind an out-of-place autograd API."""

    @staticmethod
    def forward(ctx, x, cos, sin, cu_seqlens, module):
        # cuDNN attention saves its output for backward. Cloning also preserves
        # apply_rotary's established out-of-place contract for every call site.
        output = x.clone(memory_format=torch.contiguous_format)
        packed = output.reshape(-1, x.shape[-2], x.shape[-1])
        _launch_triton(packed, cos, sin, cu_seqlens, module, False)
        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.module = module
        ctx.shape = x.shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin, cu_seqlens = ctx.saved_tensors
        grad_input = grad_output.clone(memory_format=torch.contiguous_format)
        packed = grad_input.reshape(-1, ctx.shape[-2], ctx.shape[-1])
        _launch_triton(packed, cos, sin, cu_seqlens, ctx.module, True)
        return grad_input, None, None, None, None


class _TritonQueryRotary(torch.autograd.Function):
    """Rotate the fresh normalized query in place; keep backward out of place."""

    @staticmethod
    def forward(ctx, query, cos, sin, cu_seqlens, module):
        packed = query.reshape(-1, query.shape[-2], query.shape[-1])
        _launch_triton(packed, cos, sin, cu_seqlens, module, False)
        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.module = module
        ctx.shape = query.shape
        ctx.mark_dirty(query)
        return query

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin, cu_seqlens = ctx.saved_tensors
        grad_input = grad_output.clone(memory_format=torch.contiguous_format)
        packed = grad_input.reshape(-1, ctx.shape[-2], ctx.shape[-1])
        _launch_triton(packed, cos, sin, cu_seqlens, ctx.module, True)
        return grad_input, None, None, None, None


def _apply_rotary_query(query, cos, sin, cu_seqlens):
    """Use the fresh query tensor as Triton output when its native path is available."""
    if not _can_use_triton(query, cos, sin, cu_seqlens):
        return _apply_rotary_eager(query, cos, sin, cu_seqlens)
    module = _triton_rope_module()
    return _TritonQueryRotary.apply(query, _rope_tables(cos), _rope_tables(sin), cu_seqlens, module)


def apply_rotary(x, cos, sin, cu_seqlens=None):
    """Rotate trailing interleaved pairs, using Triton when its contract is satisfied."""
    if _can_use_triton(x, cos, sin, cu_seqlens):
        return _apply_rotary_triton(x, cos, sin, cu_seqlens)
    return _apply_rotary_eager(x, cos, sin, cu_seqlens)

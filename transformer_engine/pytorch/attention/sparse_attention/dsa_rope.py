# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DSv4's trailing-channel, interleaved RoPE for full-sequence attention."""

from functools import lru_cache

import torch
import transformer_engine_torch as tex


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


def _apply_rotary_eager(x, cos, sin):
    """Rotate trailing interleaved pairs in FP32, then restore input dtype."""
    width = cos.shape[-1]
    tail = x[..., -width:]
    pair = torch.stack((-tail[..., 1::2], tail[..., 0::2]), -1).flatten(-2)
    rotated = (tail.float() * cos + pair.float() * sin).to(x.dtype)
    return torch.cat((x[..., :-width], rotated), -1)


@lru_cache(maxsize=1)
def _cute_rope_module():
    """Load the optional CuTe DSL implementation only when DSv4 can use it."""
    try:
        from . import _dsa_rope_cute
    except ImportError:
        return None
    return _dsa_rope_cute


def _can_use_cute(x, cos, sin, cu_seqlens):
    """Keep unsupported inputs on the behaviorally identical eager path."""
    return (
        cu_seqlens is not None
        and x.is_cuda
        and torch.cuda.get_device_capability(x.device) == (10, 0)
        and x.ndim == 4
        and x.stride(-1) == 1
        and cos.shape == sin.shape
        and cos.shape[-1] % 4 == 0
        and not cos.requires_grad
        and not sin.requires_grad
        and _cute_rope_module() is not None
    )


def _rope_tables(table):
    """Convert TE's repeated pairs to the CuTe kernel's duplicated-half tables."""
    pairs = table[0, :, 0, ::2]
    return torch.cat((pairs, pairs), dim=-1)


def _apply_rotary_cute(x, cos, sin, cu_seqlens):
    """Apply CuTe RoPE out of place so callers retain the eager aliasing contract."""
    module = _cute_rope_module()
    if module is None:
        raise RuntimeError("CuTe DSL RoPE is unavailable.")
    return _CuTeRotary.apply(x, _rope_tables(cos), _rope_tables(sin), cu_seqlens, module)


def _launch_cute(packed, cos, sin, cu_seqlens, module, backward):
    native = (
        packed.dtype == torch.bfloat16
        and cos.dtype == torch.float32
        and cu_seqlens.dtype == torch.int32
        and tex.dsv4_rope_cutedsl_(packed, cos, sin, cu_seqlens, backward)
    )
    if not native:
        (module.backward_inplace if backward else module.forward_inplace)(
            packed, cos, sin, cu_seqlens
        )


class _CuTeRotary(torch.autograd.Function):
    """Pair the retained in-place CuTe kernels behind an out-of-place autograd API."""

    @staticmethod
    def forward(ctx, x, cos, sin, cu_seqlens, module):
        # cuDNN attention saves its output for backward. Cloning also preserves
        # apply_rotary's established out-of-place contract for every call site.
        output = x.clone(memory_format=torch.contiguous_format)
        packed = output.reshape(-1, x.shape[-2], x.shape[-1])
        _launch_cute(packed, cos, sin, cu_seqlens, module, False)
        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.module = module
        ctx.shape = x.shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        cos, sin, cu_seqlens = ctx.saved_tensors
        grad_input = grad_output.clone(memory_format=torch.contiguous_format)
        packed = grad_input.reshape(-1, ctx.shape[-2], ctx.shape[-1])
        _launch_cute(packed, cos, sin, cu_seqlens, ctx.module, True)
        return grad_input, None, None, None, None


class _CuTeQueryRotary(torch.autograd.Function):
    """Rotate the fresh normalized query in place; keep backward out of place."""

    @staticmethod
    def forward(ctx, query, cos, sin, cu_seqlens, module):
        packed = query.reshape(-1, query.shape[-2], query.shape[-1])
        _launch_cute(packed, cos, sin, cu_seqlens, module, False)
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
        _launch_cute(packed, cos, sin, cu_seqlens, ctx.module, True)
        return grad_input, None, None, None, None


def _apply_rotary_query(query, cos, sin, cu_seqlens):
    """Use the fresh query tensor as CuTe output when its native path is available."""
    if not _can_use_cute(query, cos, sin, cu_seqlens):
        return _apply_rotary_eager(query, cos, sin)
    module = _cute_rope_module()
    return _CuTeQueryRotary.apply(query, _rope_tables(cos), _rope_tables(sin), cu_seqlens, module)


def apply_rotary(x, cos, sin, cu_seqlens=None):
    """Rotate trailing interleaved pairs, using CuTe when its contract is satisfied."""
    if _can_use_cute(x, cos, sin, cu_seqlens):
        return _apply_rotary_cute(x, cos, sin, cu_seqlens)
    return _apply_rotary_eager(x, cos, sin)

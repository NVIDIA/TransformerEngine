# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Learned DSv4 projections and norm around cuDNN compression."""

import torch

from transformer_engine.pytorch.module import Linear, RMSNorm

from .dsa_cudnn import compress


class _Compressor(torch.nn.Module):
    """Project KV/gates and normalize the cuDNN-pooled rows."""

    def __init__(
        self,
        hidden_size,
        width,
        ratio,
        overlap,
        eps,
        device,
        params_dtype,
        *,
        fused=False,
        weight_width=0,
    ):
        super().__init__()
        projected = width * (2 if overlap else 1)
        kw = {"bias": False, "device": device, "params_dtype": params_dtype}
        if fused:
            # Fuse the KV, gate, and optional index-weight projections of x.
            # Keep one contiguous weight for the shared projection call.
            self.fused_proj = Linear(hidden_size, 2 * projected + weight_width, **kw)
        else:
            self.kv_proj = Linear(hidden_size, projected, **kw)
            self.gate_proj = Linear(hidden_size, projected, **kw)
        self.position_bias = torch.nn.Parameter(
            torch.zeros(ratio, projected, device=device, dtype=params_dtype)
        )
        self.kv_norm = RMSNorm(width, eps=eps, device=device, dtype=params_dtype)
        self.ratio, self.overlap = ratio, overlap
        self.projected, self.weight_width = projected, weight_width

    def forward(self, x, cu_seqlens, cu_seqlens_comp, *, total_comp):
        """Project and compress KV rows, returning optional fused index weights."""
        if hasattr(self, "fused_proj"):
            projected = self.fused_proj(x)
            widths = (
                (self.projected, self.projected, self.weight_width)
                if self.weight_width
                else (self.projected, self.projected)
            )
            parts = projected.split(widths, dim=-1)
            kv, gate = parts[:2]
        else:
            kv, gate = self.kv_proj(x), self.gate_proj(x)
        pooled = compress(
            kv.reshape(-1, self.projected).contiguous(),
            gate.reshape(-1, self.projected).contiguous(),
            # cuDNN requires FP32 bias; the cast preserves gradients to the parameter.
            self.position_bias.float(),
            cu_seqlens,
            cu_seqlens_comp,
            ratio=self.ratio,
            overlap=self.overlap,
            total_comp=total_comp,
        )
        if x.ndim == 2:
            # cuDNN only guarantees valid rows below the final prefix. Clear
            # unused capacity before RMSNorm so it cannot poison weight gradients.
            valid = torch.arange(total_comp, device=x.device) < cu_seqlens_comp[-1]
            pooled = pooled.masked_fill(~valid[:, None], 0)
        compressed = self.kv_norm(pooled)
        return (compressed, parts[2]) if self.weight_width else compressed

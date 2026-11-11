# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Learned projections and normalization around cuDNN compression."""

import torch

from transformer_engine.pytorch.module import Linear, RMSNorm

from .dsa_cudnn import compress


class Compressor(torch.nn.Module):
    """Project KV/gates for HCA or CSA, compress with cuDNN, and apply RMSNorm.

    Non-overlapping compression combines each block of ``ratio`` tokens into
    one compressed KV entry. Overlapping compression mixes the current block
    with the entire preceding block to create one compressed KV entry, using
    separate learned KV/gate projections for the current and preceding blocks.
    Neighboring compressed KV entries therefore share an entire source block,
    but use different projections of its tokens.
    Weights are computed per KV channel as ``softmax(gates + position_bias)``.

    For ``ratio=2`` and ``[a b c d e f g h i j k]``:

    - Non-overlapping windows: ``[a b]``, ``[c d]``, ``[e f]``, ``[g h]``, ``[i j]``.
    - Overlapping windows: ``[a b]``, ``[a b c d]``, ``[c d e f]``,
      ``[e f g h]``, ``[g h i j]``.

    The first overlapping entry masks the missing previous block. Both modes emit
    ``floor(length / ratio)`` entries per sequence; incomplete trailing blocks
    (``k`` here) are excluded from compression. In DSv4Attention, these tokens
    remain available through local sliding-window attention.

    Accept batch-major ``[B,S,D]`` or packed ``[T,D]`` hidden states with
    token/compressed prefixes. Return ``[total_comp,width]`` rows before RoPE,
    plus projected head weights when ``weight_width`` is nonzero in fused mode.
    """

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
        if weight_width and not fused:
            raise ValueError("weight_width requires fused=True.")
        projected = width * (2 if overlap else 1)
        kw = {"bias": False, "device": device, "params_dtype": params_dtype}
        if fused:
            # Fuse the KV, gate, and optional index-weight projections of hidden states.
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

    def forward(self, hidden_states, cu_seqlens, cu_seqlens_comp, *, total_comp):
        """Project and compress KV entries before RoPE.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Batch-major ``[B,S,hidden_size]`` or packed ``[T,hidden_size]`` input.
        cu_seqlens, cu_seqlens_comp : torch.Tensor
            Contiguous CUDA INT32 ``[B+1]`` token/compressed prefixes. Compressed
            segment lengths are ``floor(length / ratio)``.
        total_comp : int
            Output row capacity. For dense input, use exactly ``B * (S // ratio)``;
            packed input may reserve extra rows beyond ``cu_seqlens_comp[-1]``.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Normalized ``[total_comp,width]`` KV rows. When ``weight_width > 0``,
            also return head weights with shape ``hidden_states.shape[:-1] +
            (weight_width,)``. Only rows below the final compressed prefix are valid.
        """
        if hasattr(self, "fused_proj"):
            projected = self.fused_proj(hidden_states)
            widths = (
                (self.projected, self.projected, self.weight_width)
                if self.weight_width
                else (self.projected, self.projected)
            )
            parts = projected.split(widths, dim=-1)
            kv, gate = parts[:2]
        else:
            kv, gate = self.kv_proj(hidden_states), self.gate_proj(hidden_states)
        compressed_kv = compress(
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
        if hidden_states.ndim == 2:
            # cuDNN only guarantees valid rows below the final prefix. Clear
            # unused capacity before RMSNorm so it cannot poison weight gradients.
            valid = torch.arange(total_comp, device=hidden_states.device) < cu_seqlens_comp[-1]
            compressed_kv = compressed_kv.masked_fill(~valid[:, None], 0)
        compressed = self.kv_norm(compressed_kv)
        return (compressed, parts[2]) if self.weight_width else compressed

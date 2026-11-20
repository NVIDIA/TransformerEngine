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
    Weights are computed per KV channel as ``softmax(gates + ape)``, where
    ``ape`` is the learned additive position embedding.

    For ``ratio=2`` and ``[a b c d e f g h i j k]``:

    - Non-overlapping windows: ``[a b]``, ``[c d]``, ``[e f]``, ``[g h]``, ``[i j]``.
    - Overlapping windows: ``[a b]``, ``[a b c d]``, ``[c d e f]``,
      ``[e f g h]``, ``[g h i j]``.

    The first overlapping entry masks the missing previous block. Both modes emit
    ``floor(length / ratio)`` compressed KV tokens per sequence. Trailing tokens
    that do not fill a complete block (``k`` here) are excluded from compression.
    In DSv4Attention, they remain available through local sliding-window attention.

    Accept batched ``[B,S,D]`` or packed ``thd`` hidden states ``[T,D]`` with
    cumulative sequence token counts. Standalone calls do not interpret
    sequence-major ``[S,B,D]`` input; DSv4Attention performs that conversion.
    Return ``[total_comp,width]`` compressed KV representations
    before RoPE, plus projected head weights when ``weight_width`` is nonzero.

    See the sparse_attention README for support requirements.

    Parameters
    ----------
    hidden_size : int
        Input hidden-state width.
    width : int
        Width of each compressed KV representation.
    ratio : int
        Number of input tokens per compression block.
    overlap : bool
        Whether each output also uses the preceding complete block.
    eps : float
        Epsilon for RMSNorm after compression.
    device : torch.device or str
        Device on which to allocate parameters.
    params_dtype : torch.dtype
        Dtype used to initialize projection, position-bias and normalization parameters.
    weight_width : int, default = 0
        Additional per-input-token head-weight projection width. Indexer sets this
        to n_heads for its scoring weights; DSv4Attention's main KV compressor
        leaves it at zero.
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
        weight_width=0,
    ):
        super().__init__()
        projected = width * (2 if overlap else 1)
        kw = {"bias": False, "device": device, "params_dtype": params_dtype}
        # Fuse the KV, gate, and optional index-weight projections of hidden states.
        # Keep one contiguous weight for the shared projection call.
        self.fused_proj = Linear(hidden_size, 2 * projected + weight_width, **kw)
        self.ape = torch.nn.Parameter(
            torch.zeros(ratio, projected, device=device, dtype=params_dtype)
        )
        self.kv_norm = RMSNorm(width, eps=eps, device=device, dtype=params_dtype)
        self.ratio, self.overlap = ratio, overlap
        self.projected, self.weight_width = projected, weight_width

    def forward(self, hidden_states, cu_seqlens, cu_seqlens_comp, *, total_comp):
        """Project and compress KV blocks before RoPE.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Batched ``[B,S,hidden_size]`` with equal sequence length S, or packed ``thd``
            ``[T,hidden_size]`` with sequences concatenated along the token dimension.
        cu_seqlens, cu_seqlens_comp : torch.Tensor
            Contiguous CUDA INT32 ``[B+1]`` cumulative input/compressed token counts,
            starting at zero. Each compressed count increment is the corresponding
            input sequence length divided by ratio, rounded down.
        total_comp : int
            Allocated compressed output length. For ``[B,S,hidden_size]`` input, use
            exactly ``B * (S // ratio)``; DSv4Attention allocates ``T // ratio`` for ``thd``
            input, which may include output padding after the
            valid compressed tokens counted by ``cu_seqlens_comp[-1]``.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Normalized compressed KV ``[total_comp,width]``. If ``weight_width > 0``,
            return ``(compressed_kv, weights)`` with per-input-token head weights of
            shape ``hidden_states.shape[:-1] + (weight_width,)``. Indexer uses this
            projection for scoring; the main KV compressor does not. Only compressed
            token indices below ``cu_seqlens_comp[-1]`` are valid; the rest is padding.
        """
        projected = self.fused_proj(hidden_states)
        widths = (
            (self.projected, self.projected, self.weight_width)
            if self.weight_width
            else (self.projected, self.projected)
        )
        parts = projected.split(widths, dim=-1)
        kv, gate = parts[:2]
        compressed_kv = compress(
            kv.reshape(-1, self.projected).contiguous(),
            gate.reshape(-1, self.projected).contiguous(),
            # cuDNN requires FP32 APE. The cast preserves gradients to the parameter.
            self.ape.float(),
            cu_seqlens,
            cu_seqlens_comp,
            ratio=self.ratio,
            overlap=self.overlap,
            total_comp=total_comp,
        )
        if hidden_states.ndim == 2:
            # cuDNN guarantees valid output only for the cu_seqlens_comp[-1] compressed tokens.
            # Zero output from cu_seqlens_comp[-1] onward before RMSNorm so
            # uninitialized values cannot affect weight gradients.
            valid = torch.arange(total_comp, device=hidden_states.device) < cu_seqlens_comp[-1]
            compressed_kv = compressed_kv.masked_fill(~valid[:, None], 0)
        compressed = self.kv_norm(compressed_kv)
        return (compressed, parts[2]) if self.weight_width else compressed

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-Latent Attention (MLA) block as used in DeepSeekV3."""

import torch

__all__ = ["MultiLatentAttention"]


class MultiLatentAttention(torch.nn.Module):
    """
    Multi-Latent Attention with low-rank Q/KV down-projections and a
    decoupled RoPE/NoPE head split, composed from :class:`Linear`,
    :class:`LayerNormLinear` and :class:`DotProductAttention`
    (``kv_channels=(head_dim_qk, head_dim_v)``).

    .. warning:: Work in progress, not functional yet.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("MultiLatentAttention is under development")

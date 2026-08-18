# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DeepSeekV3 transformer layer."""

import torch

__all__ = ["DeepSeekV3Layer"]


class DeepSeekV3Layer(torch.nn.Module):
    """
    A full DeepSeekV3 transformer layer, analogous to
    :class:`TransformerLayer`: :class:`MultiLatentAttention` followed by
    either a dense :class:`LayerNormMLP` (first layers) or
    :class:`DeepSeekV3MoE`, with the same residual and fused
    bias-dropout-add plumbing as :class:`TransformerLayer`.

    .. warning:: Work in progress, not functional yet.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("DeepSeekV3Layer is under development")

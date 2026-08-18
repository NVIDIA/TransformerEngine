# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DeepSeekV3 MoE block: sigmoid router with aux-loss-free bias, shared +
routed experts."""

import torch

__all__ = ["DeepSeekV3MoE"]


class DeepSeekV3MoE(torch.nn.Module):
    """
    DeepSeekV3-style Mixture of Experts block composed from TE MoE
    primitives: ``fused_topk_with_score_function`` (sigmoid score function,
    expert bias, grouped top-k), ``moe_permute_with_probs``/``moe_unpermute``,
    :class:`GroupedLinear` routed experts, a shared expert
    (:class:`LayerNormMLP`), ``Fp8Padding``/``Fp8Unpadding`` and optional
    expert parallelism via ``ep_dispatch``/``ep_combine``.

    .. warning:: Work in progress, not functional yet.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("DeepSeekV3MoE is under development")

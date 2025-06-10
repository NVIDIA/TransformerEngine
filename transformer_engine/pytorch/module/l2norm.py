# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""L2Norm API"""
from typing import Optional

import torch

from transformer_engine.pytorch.ops import L2Norm as _L2NormOp
from ..jit import (
    set_jit_fusion_options,
    warmup_jit_l2norm_all_dtypes,
)

__all__ = ["L2Norm"]


class L2Norm(_L2NormOp):
    r"""L2 Normalization

    Applies L2 normalization over the last dimension of input tensors.
    This is a parameter-free normalization that scales each vector to unit L2 norm.

    .. math::
        y = \frac{x}{\sqrt{\sum_{i} x_i^2 + \varepsilon}}

    This operation is used e.g. for query-key normalization in attention mechanisms.

    Parameters
    ----------
    eps : float, default = 1e-6
        A value added to the denominator for numerical stability
    device: torch.device, default = default CUDA device
        Tensor device
    dtype: torch.dtype, default = default dtype
        Tensor datatype
    seq_length: int, default = None
        sequence length of input samples. Needed for JIT Warmup, a technique where jit fused
        functions are warmed up before training to ensure same kernels are used for forward
        propagation and activation recompute phase.
    micro_batch_size: int, default = None
        batch size per training step. Needed for JIT Warmup, a technique where jit
        fused functions are warmed up before training to ensure same kernels are
        used for forward propagation and activation recompute phase.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        **kwargs,
    ) -> None:

        # Initialize L2Norm operation
        super().__init__(
            eps=eps,
            **kwargs,
        )

        # JIT warmup for L2Norm fused operations
        if seq_length and micro_batch_size:
            device = getattr(self, "device", torch.device("cuda"))
            if hasattr(device, "type") and device.type == "cuda":
                set_jit_fusion_options()
                # For L2Norm, we don't know the hidden size until forward pass,
                # but we can warm up with common sizes used in attention mechanisms
                common_hidden_sizes = [768, 1024, 2048, 4096, 8192]
                for hidden_size in common_hidden_sizes:
                    warmup_jit_l2norm_all_dtypes(hidden_size, seq_length, micro_batch_size)

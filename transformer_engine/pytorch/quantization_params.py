# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Parameters needed for quantization using different recipes."""

import torch
from transformer_engine_torch import DType as TE_DType

class QuantizationParams:
    def __init__(self):
        pass

class Float8Params(QuantizationParams):
    scale: torch.Tensor
    amax: torch.Tensor
    dtype: TE_DType

    def __init__(self,
                 scale: torch.Tensor,
                 amax: torch.Tensor,
                 dtype: TE_DType):
        super().__init__()
        self.scale = scale
        self.amax = amax
        self.dtype = dtype


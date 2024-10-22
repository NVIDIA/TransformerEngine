# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Quantization metadata class"""

from transformer_engine.common.recipe import (
        Recipe, DelayedScaling
)
import torch

import transformer_engine_torch as tex

from .fp8 import get_fp8_te_dtype
from .tensor import QuantizedTensor, Float8Tensor

from .quantization_params import Float8Params

class QMeta:
    def __init__(self,
                 recipe: Recipe,
                 num_tensors: int,
                 forward: bool):
        if isinstance(recipe, DelayedScaling):
            self.recipe_type = DelayedScaling
            self.scale = torch.ones(num_tensors, dtype=torch.float32, device="cuda")
            self.amax_history = torch.zeros(
                recipe.amax_history_len,
                num_tensors,
                dtype=torch.float32,
                device="cuda",
            )
            self.fp8_type = get_fp8_te_dtype(recipe, forward)
        raise ValueError("Unknown recipe type.")

    def quantize(self,
                 tensor: torch.Tensor,
                 index: int,
                 *,
                 rowwise: bool = True,
                 columnwise: bool = True) -> QuantizedTensor:
        if self.recipe_type == DelayedScaling:
            return Float8Tensor.quantize(tensor,
                                         self.get_quantization_params(index),
                                         rowwise_usage=rowwise,
                                         columnwise_usage=columnwise)
        raise NotImplementedError("Not implemented yet!")

    def quantize_param(self,
                 tensor: torch.Tensor,
                 index: int):
        if self.recipe_type == DelayedScaling:
            pass
        raise NotImplementedError("Not implemented yet! Same as quantize but also sets proxy")

    def get_quantization_params(self,
                                index: int):
        # Could be cached
        if self.recipe_type == DelayedScaling:
            return Float8Params(scale=self.scale[index],
                                amax=self.amax_history[0][index],
                                dtype=self.fp8_type)
        raise NotImplementedError("Not implemented yet!")

    def calibrate(self,
                  tensor: torch.Tensor,
                  index: int):
        if self.recipe_type == DelayedScaling:
            amin, amax = tensor.aminmax()
            self.amax_history[0][index] = torch.max(-amin, amax).float()
            return
        raise NotImplementedError("Not implemented yet!")

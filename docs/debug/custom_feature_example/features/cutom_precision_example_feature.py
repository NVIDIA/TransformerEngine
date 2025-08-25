# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Custom precision example feature"""

from typing import Dict, Optional, Tuple
from nvdlfw_inspect.logging import MetricLogger
from nvdlfw_inspect.registry import Registry, api_method

import torch
from transformer_engine.debug.features.api import TEConfigAPIMapper
from transformer_engine.pytorch.tensor import Quantizer



def custom_precision_quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    amax = torch.amax(tensor)
    scale = 1.0 / amax
    q_tensor = tensor * scale

    # tensor to -1/0/1 range (-1, -0.5) -> -1, (-0.5, , 0.5) -> 0, (0.5, 1) -> 1
    out_tensor = torch.where(q_tensor < -0.5, -1, torch.where(q_tensor > 0.5, 1, 0))
    return out_tensor, scale


def custom_precision_dequantize(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    return tensor * scale


@Registry.register_feature(namespace="transformer_engine")
class CustomPrecisionExampleFeature(TEConfigAPIMapper):

    @api_method
    def modify_tensor_enabled(
        self, config, layer_name: str, tensor_name: str, gemm: str, iteration: int
    ):  
        """API call used to determine whether to run process_tensor() in the forward."""
        return True, iteration + 1

    @api_method
    def modify_tensor(
        self,
        config,
        layer_name: str,
        gemm: str,
        tensor_name: str,
        tensor: torch.Tensor,
        iteration: int,
        default_quantizer: Quantizer,
        out: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None,
    ):  # pylint: disable=unused-argument
        """API call used to process the tensor."""

        q_tensor, scale = custom_precision_quantize(tensor)

        MetricLogger.log_scalar(
            f"custom_precision_scale {layer_name}_{gemm}_{tensor_name}",
            scale,
            iteration=iteration,
        )

        dq_tensor = custom_precision_dequantize(q_tensor, scale)
        return dq_tensor
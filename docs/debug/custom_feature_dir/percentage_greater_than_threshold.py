# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PercentageGreaterThanThreshold Feature support for nvidia-dlframework-inspect"""

from typing import Dict, Optional

import torch

from nvdlfw_inspect.registry import Registry, api_method
from nvdlfw_inspect.logging import MetricLogger

from transformer_engine.debug.features.api import TEConfigAPIMapper
from transformer_engine.pytorch.tensor import QuantizedTensor, Quantizer


# Class should inherit from TEConfigAPIMapper and be registered to the transformer_engine namespace.
@Registry.register_feature(namespace="transformer_engine")
class PercentageGreaterThanThreshold(TEConfigAPIMapper):

    @api_method
    def inspect_tensor(
        self,
        config: Dict,
        layer_name: str,
        tensor_name: str,
        iteration: int,
        tp_group: torch.distributed.ProcessGroup,
        tensor: torch.Tensor,
        rowwise_quantized_tensor: Optional[torch.Tensor | QuantizedTensor] = None,
        columnwise_quantized_tensor: Optional[torch.Tensor | QuantizedTensor] = None,
        quantizer: Optional[Quantizer] = None,
    ):
        # API call inspect_tensor is used to gather the data about the tensor.
        # All API calls are documented in the `Precision debug tools / API / Calls to Nvidia-DL-Framework-Inspect`
        # section of the documentation.

        threshold = config["threshold"]
        percentage = (tensor > threshold).sum() / tensor.numel()

        # MetricLogger is a class from nvidia-dlframework-inspect.
        # By using it we can also use functionalities provided by nvidia-dlframework-inspect,
        # like logging to TensorBoard, etc.
        MetricLogger.log_scalar(
            f"{layer_name}_{tensor_name}_percentage_greater_than_threshold", percentage, iteration
        )

    @api_method
    def inspect_tensor_enabled(
        self, config: Dict, layer_name: str, tensor_name: str, iteration: int
    ):
        # This call is used by TE to determine if the unfused debug layer - which is slower - needs to be run.
        # It returns a tuple (bool, int), where the int indicates the next iteration when the feature will be enabled
        # and bool indicates if the feature should be enabled at the current iteration.

        run_current = iteration % config["freq"] == 0
        # run in next multiple of freq
        next_iter = iteration + (config["freq"] - iteration % config["freq"])
        return run_current, next_iter

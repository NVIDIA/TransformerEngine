# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LogNvfp4TensorStats Feature support for nvidia-dlframework-inspect"""

from typing import Dict, Optional
from contextlib import contextmanager

import torch
import nvdlfw_inspect.api as debug_api

from nvdlfw_inspect.debug_features.log_tensor_stats import LogTensorStats as BaseLogTensorStats
from nvdlfw_inspect.registry import Registry, api_method

from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS
from transformer_engine.pytorch.tensor import Quantizer, QuantizedTensor
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
from transformer_engine.debug.features.utils import get_reduction_params, next_enabled_iter
from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import NVFP4TensorStorage


@Registry.register_feature(namespace="transformer_engine")
class LogNvfp4TensorStats(BaseLogTensorStats):
    """Logs statistics of NVFP4 quantized tensors.

    In distributed runs each rank first computes its local statistics; the values
    are gathered the next time `debug_api.step()` is called.  Remember to call
    `debug_api.step()` every training step so the logs are flushed.

    The feature is micro-batch aware: if several forward/backward passes occur
    between successive `debug_api.step()` calls, statistics are accumulated for all
    tensors except weights.

    Collecting NVFP4 statistics is expensive. Choosing a larger `freq` reduces the
    overhead, and if the feature is skipped for a step the additional cost is
    minimal.  When no other debug feature is active, the layer runs at normal
    Transformer Engine speed.

    Parameters
    ----------

        stats: List[str]
            List of statistics to collect. Available stats:
                - underflows% - percentage of non-zero elements clipped to 0 (from packed FP4 data)
                - mse - mean squared error = sum((quantized_tensor - original_tensor)**2) / num_elements

        tensors/tensors_struct: List[str]
            list of tensors to log
                - activation,
                - gradient,
                - weight,

        freq: Optional[int], default = 1
            frequency of logging stats, stats will be logged every `freq` steps
        start_step: Optional[int], default = None
            start step of logging stats
        end_step: Optional[int], default = None
            end step of logging stats
        start_end_list: Optional[list([int, int])], default = None
            non-overlapping list of (start, end) pairs in incremental order. If not None, will ignore start_step and end_step

    Example
    -------
    .. code-block:: yaml

        example_nvfp4_tensor_stat_collection:
            enabled: True
            layers:
                layer_types: [layernorm_linear]
            transformer_engine:
                LogNvfp4TensorStats:
                    enabled: True
                    tensors_struct:
                    - tensor: activation
                      stats: [underflows%, mse]
                      freq: 1
                    - tensor: gradient
                      stats: [underflows%, mse]
                      freq: 5
                      start_step: 0
                      end_step: 80
    """

    def check_if_stat_is_supported(self, stat: str):
        """Returns True if stat is supported, raises ValueError otherwise."""
        supported_stats = [
            "underflows%",
            "mse",
        ]
        if stat not in supported_stats:
            raise ValueError(
                f"Stat {stat} is not supported for NVFP4. Supported stats: {supported_stats}"
            )
        return True

    def get_stat_with_prefix(self, stat: str) -> str:
        """Add nvfp4_ prefix to stat name for use in stats_computation."""
        return f"nvfp4_{stat}"

    @contextmanager
    def update_aux_dict(
        self,
        aux_dict: Dict,
        quantized_tensor: QuantizedTensor,
        quantizer: Quantizer,
        original_tensor: torch.Tensor,
    ):
        """
        Updates the aux_dict with the quantized tensor and additional NVFP4-specific data.
        Yields the aux_dict.
        """
        aux_dict = {
            "nvfp4": quantized_tensor,
            "original_tensor": original_tensor,
        }

        try:
            yield aux_dict
        finally:
            pass

    @api_method
    def inspect_tensor_enabled(
        self, config: Dict, layer_name: str, tensor_name: str, iteration: int
    ):  # pylint: disable=unused-argument
        """API call used to determine whether to run inspect_tensor() in the forward."""
        run_current, next_iter = next_enabled_iter(
            config.get("start_step", None),
            config.get("end_step", None),
            config.get("start_end_list", None),
            config.get("freq", 1),
            iteration,
        )
        STATS_BUFFERS.layers_to_next_iter[layer_name] = next_iter
        return run_current, next_iter

    @api_method
    def inspect_tensor(
        self,
        config: Dict,
        layer_name: str,
        tensor_name: str,
        iteration: int,
        tp_group,
        tensor: torch.Tensor,
        rowwise_quantized_tensor: Optional[QuantizedTensor] = None,
        columnwise_quantized_tensor: Optional[QuantizedTensor] = None,
        quantizer: Optional[Quantizer] = None,
    ):
        """
        API call used to collect the data about the tensor after process_tensor()/quantization.
        """
        assert rowwise_quantized_tensor is columnwise_quantized_tensor
        assert (
            quantizer is not None
        ), "[NVTORCH INSPECT ERROR] LogNvfp4TensorStats cannot be run without NVFP4 quantizer."

        quantized_tensor = rowwise_quantized_tensor

        # Ensure we're working with NVFP4 tensors
        if not isinstance(quantizer, NVFP4Quantizer):
            raise ValueError(
                "[NVTORCH INSPECT ERROR] LogNvfp4TensorStats requires NVFP4Quantizer, "
                f"but got {type(quantizer).__name__}"
            )

        assert isinstance(
            quantized_tensor, NVFP4TensorStorage
        ), "[NVTORCH INSPECT ERROR] LogNvfp4TensorStats quantized_tensor must be a QuantizedTensor."

        for stat in config["stats"]:
            self.check_if_stat_is_supported(stat)

        start_step = config.get("start_step", None)
        end_step = config.get("end_step", None)
        start_end_list = config.get("start_end_list", None)
        if start_end_list is not None:
            start_end_list = tuple(tuple(int(x) for x in interval) for interval in start_end_list)

        options = (
            start_step,
            end_step,
            start_end_list,
            "nvfp4",
        )

        skip_reduction, reduction_group, reduce_within_microbatch = get_reduction_params(
            tensor_name, tp_group
        )

        # Add nvfp4_ prefix to all stats for internal use
        prefixed_stats = [self.get_stat_with_prefix(stat) for stat in config["stats"]]

        STATS_BUFFERS.try_add_buffer(
            layer_name=layer_name,
            tensor_name=tensor_name,
            stats=prefixed_stats,
            options=options,
            reduction_group=reduction_group,
            reduce_within_microbatch=reduce_within_microbatch,
        )

        with self.update_aux_dict(
            aux_dict={},
            quantized_tensor=quantized_tensor,
            quantizer=quantizer,
            original_tensor=tensor,
        ) as aux_dict:
            STATS_BUFFERS.feed(
                layer_name,
                tensor_name,
                options,
                tensor,
                iteration,
                skip_reduction,
                aux_dict=aux_dict,
            )

        debug_api.log_message(
            f"Feature={self.__class__.__name__}, API=inspect_tensor: {tensor_name}",
            layer_name,
            extra_cachable_args=(tensor_name,),
        )

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LogTensorStats Feature support for nvidia-dlframework-inspect"""

from typing import Dict, Optional

import torch

from nvdlfw_inspect.debug_features.log_tensor_stats import LogTensorStats as BaseLogTensorStats
from nvdlfw_inspect.registry import Registry, api_method
import nvdlfw_inspect.api as debug_api

from transformer_engine.pytorch.tensor import QuantizedTensor, Quantizer
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import Float8TensorStorage
from transformer_engine.pytorch.tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS
from transformer_engine.debug.features.utils import next_enabled_iter, get_reduction_params


@Registry.register_feature(namespace="transformer_engine")
class LogTensorStats(BaseLogTensorStats):
    """
    This feature handles the logging of basic tensor statistics.

    For a distributed setting, the auxiliary stats are computed for each node and gathered after the `debug_api.step()` call. Do not forget to invoke `debug_api.step()` at every step to log stats!

    `LogTensorStats` supports micro-batching. If multiple forward/backward passes are invoked per `debug_api.step()`, then stats for all tensors except weights will be accumulated.

    `LogTensorStats` can induce significant overhead. To mitigate this issue, logging stats with `freq > 1` is recommended. If `LogTensorStats` is not used in a given step, the overhead is smaller. Moreover, if no other feature is used for the layer, the TE layer will run as fast as it would without `debug_api` initialized.

    Parameters
    ----------
    stats: List[str]
        list of statistics to log

            - min
            - max
            - mean
            - std
            - l1_norm
            - l2_norm
            - cur_amax – maximal absolute value of a tensor,
            - dynamic_range – equal to `torch.log2(amax) - torch.log2(amin)`
    tensors/tensors_struct: List[str]
        list of tensors to log

            - activation
            - gradient
            - weight
            - output
            - wgrad
            - dgrad
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

        example_tensor_stat_collection:
            enabled: True
            layers:
                layer_name_regex_pattern: .*(fc1|self_attention).*
            transformer_engine:
                LogTensorStats:
                    enabled: True
                    tensors_struct:
                        - tensor: activation
                          stats: [mean]
                          freq: 10
                          start_step: 5
                          end_step: 100
                        - tensor: gradient
                          stats: [mean, max, min]
                          freq: 2
                          start_end_list: [[0, 20], [80, 100]]
                        - tensor: weight
                          stats: [dynamic_range]
    """

    def _get_supported_stats_list(self):
        """Returns stats this feature can log."""
        return BaseLogTensorStats._get_supported_stats_list(None) | {"cur_amax", "dynamic_range"}

    @api_method
    def inspect_tensor_enabled(
        self, config: Dict, layer_name: str, tensor_name: str, iteration: int
    ):  # pylint: disable=unused-argument
        """API call used to determine whether to run look_at_tensor_before_process() in the forward."""
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
        tp_group: torch.distributed.ProcessGroup,
        tensor: torch.Tensor,
        rowwise_quantized_tensor: Optional[torch.Tensor | QuantizedTensor] = None,
        columnwise_quantized_tensor: Optional[torch.Tensor | QuantizedTensor] = None,
        quantizer: Optional[Quantizer] = None,
    ):  # pylint: disable=unused-argument
        """API call used to collect the data about the tensor before process_tensor()/quantization."""

        assert (
            type(tensor) not in [Float8Tensor, Float8TensorStorage, MXFP8Tensor, MXFP8TensorStorage]
            and tensor.dtype != torch.uint8
        ), (
            f"[NVTORCH INSPECT ERROR] Tensor {tensor_name} must be in high precision when using"
            " log_tensor_stats. Use log_fp8_tensor_stats for FP8 tensors."
        )

        options = (
            config.get("start_step", None),
            config.get("end_step", None),
            config.get("start_end_list", None),
        )

        skip_reduction, reduction_group, reduce_within_microbatch = get_reduction_params(
            tensor_name, tp_group
        )

        for stat in config["stats"]:
            assert (
                stat in self._get_supported_stats_list()
            ), f"[NVTORCH INSPECT ERROR] Statistic {stat} is not supported."

        STATS_BUFFERS.try_add_buffer(
            layer_name=layer_name,
            tensor_name=tensor_name,
            stats=config["stats"],
            options=options,
            reduction_group=reduction_group,
            reduce_within_microbatch=reduce_within_microbatch,
        )

        STATS_BUFFERS.feed(layer_name, tensor_name, options, tensor, iteration, skip_reduction)

        debug_api.log_message(
            f"Feature={self.__class__.__name__}, API=look_at_tensor_before_process: {tensor_name}",
            layer_name,
            extra_cachable_args=(tensor_name),
        )

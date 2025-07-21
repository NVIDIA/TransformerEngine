# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LogFp8TensorStats Feature support for nvidia-dlframework-inspect"""

from typing import Dict, Union

import torch

import nvdlfw_inspect.api as debug_api
from nvdlfw_inspect.debug_features.log_tensor_stats import LogTensorStats as BaseLogTensorStats
from nvdlfw_inspect.registry import Registry, api_method

from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS
from transformer_engine.pytorch.tensor import QuantizedTensor
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
from transformer_engine.pytorch.tensor._internal.float8_tensor_base import Float8TensorBase
from transformer_engine.pytorch.tensor._internal.mxfp8_tensor_base import MXFP8TensorBase
from transformer_engine.debug.pytorch.debug_state import TEDebugState


@Registry.register_feature(namespace="transformer_engine")
class LogFp8TensorStats(BaseLogTensorStats):
    """
    This feature handles logging of FP8 tensor stats.


    In a distributed setting, the auxiliary stats are computed on each rank and gathered after
    the `debug_api.step()` call. Do not forget to invoke `debug_api.step()` at every step to log
    stats!

    `LogFp8TensorStats` supports micro-batching. If multiple forward/backward passes are invoked
    per `debug_api.step()`, then stats for all tensors except weights will be accumulated.

    `LogFp8TensorStats` can induce significant overhead. To mitigate this issue, logging stats
    with `freq > 1` is recommended. If `LogFp8TensorStats` is not used in a given step, the
    overhead is smaller. If no other feature is used for the layer, the TE layer will
    run as fast as it would without `debug_api` initialized.

    Parameters
    ----------

        stats: List[str]
            list of statistics to log

                - underflows% - percentage of elements of the tensor equal to 0,
        tensors/tensors_struct: List[str]
            list of tensors to log

                - activation
                - gradient
                - weight
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

        example_fp8_tensor_stat_collection:
            enabled: True
            layers:
                layer_types: [layernorm_linear]
            transformer_engine:
                LogFp8TensorStats:
                    enabled: True
                    tensors_struct:
                    - tensor: activation
                    stats: [underflows%]
                    freq: 1
                    - tensor: gradient
                    stats: [underflows%]
                    freq: 5
                    start_step: 0
                    end_step: 80
    """

    def _get_supported_stats_list(self):
        """Returns stats this feature can log."""
        return {"underflows%"}

    @api_method
    def inspect_tensor_postquantize_enabled(
        self, config: Dict, layer_name: str, gemm: str, tensor_name: str, iteration: int
    ):  # pylint: disable=unused-argument
        """API call used to determine whether to run inspect_tensor_postquantize() in the forward."""
        # check whether logging should happen in this iteration
        return self._check_params(config, layer_name, iteration=iteration)

    @api_method
    def inspect_tensor_postquantize(
        self,
        config: Dict,
        layer_name: str,
        tensor_name: str,
        tensor: Union[torch.Tensor, QuantizedTensor],
        rowwise: bool,
        iteration: int,
        tp_group: torch.distributed.ProcessGroup,
    ):
        """
        API call used to collect the data about the tensor after process_tensor()/quantization.
        """

        assert type(tensor) in [Float8Tensor, Float8TensorBase, MXFP8Tensor, MXFP8TensorBase], (
            f"[NVTORCH INSPECT ERROR] Tensor {tensor_name} must be a quantized tensor when using"
            " log_fp8_tensor_stats. Use log_tensor_stats for high precision tensors."
        )

        # This API can be invoked twice - with the tensor and with the transpose.
        # We want to collect the stats once.
        if not rowwise:
            return  # tensor was already seen rowwise in the other gemm

        options = (
            config.get("start_step", None),
            config.get("end_step", None),
            config.get("start_end_list", None),
            "fp8",
        )

        skip_reduction = False
        reduction_group = debug_api.get_tensor_reduction_group()
        reduce_within_microbatch = tensor_name != "weight"
        if tensor_name == "weight":
            if TEDebugState.weight_tensor_tp_group_reduce:
                reduction_group = tp_group
            else:
                skip_reduction = True

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
            f"Feature={self.__class__.__name__}, API=inspect_tensor_postquantize: {tensor_name}",
            layer_name,
            extra_cachable_args=(tensor_name,),
        )

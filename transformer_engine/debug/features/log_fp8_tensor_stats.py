# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LogFp8TensorStats Feature support for nvidia-dlframework-inspect"""

from typing import Dict, Optional, List, Tuple
from contextlib import contextmanager

import torch
import nvdlfw_inspect.api as debug_api


from nvdlfw_inspect.debug_features.log_tensor_stats import LogTensorStats as BaseLogTensorStats
from nvdlfw_inspect.registry import Registry, api_method

from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS
from transformer_engine.pytorch.tensor import Quantizer, QuantizedTensor
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.debug.features.utils import get_reduction_params, next_enabled_iter


ALL_RECIPE_NAMES = ["fp8_delayed_scaling", "fp8_current_scaling", "mxfp8", "fp8_block_scaling"]


def _get_recipe_name(quantizer: Optional[Quantizer]):
    if quantizer is None:
        return ""
    if isinstance(quantizer, Float8Quantizer):
        return "fp8_delayed_scaling"
    if isinstance(quantizer, Float8CurrentScalingQuantizer):
        return "fp8_current_scaling"
    if isinstance(quantizer, MXFP8Quantizer):
        return "mxfp8"
    if isinstance(quantizer, Float8BlockQuantizer):
        return "fp8_block_scaling"
    raise ValueError(f"Unsupported quantizer type: {type(quantizer)}")


def _get_new_quantizer(recipe_name, fp8_dtype):
    if recipe_name == "fp8_block_scaling":
        return Float8BlockQuantizer(fp8_dtype=fp8_dtype, rowwise=True, columnwise=True)
    if recipe_name == "fp8_current_scaling":
        return Float8CurrentScalingQuantizer(
            fp8_dtype=fp8_dtype, device=torch.device("cuda"), rowwise=True, columnwise=True
        )
    if recipe_name == "mxfp8":
        return MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=True, columnwise=True)
    if recipe_name == "fp8_delayed_scaling":
        raise ValueError("Cannot recreate quantizer for fp8_delayed_scaling")
    raise ValueError(f"Unsupported recipe name: {recipe_name}")


@Registry.register_feature(namespace="transformer_engine")
class LogFp8TensorStats(BaseLogTensorStats):
    """
    Logs statistics of quantized tensors.

    Supports computing statistics for current recipe, but also
    allows to see what would happend if different recipes were used for these tensors in current iteration.
    For example, during delayed-scaling training you may wish to track
    "current_scaling_underflows%" to measure the accuracy of the current scaling
    factors; note that this requires an extra cast and therefore adds overhead.
    Using a logging frequency (`freq`) greater than 1 is recommended in this case.
    Computing the stats matching the training recipe does not require an extra cast.

    Statistics are identified by the pattern `<recipe>_<stat>` with optional `_columnwise` suffix (e.g.
    `delayed_scaling_underflows%` or `mxfp8_scale_inv_min_columnwise`).
    One can provide `<stat>` only, then the current training recipe is used.

    Stats for delayed-scaling cannot be collected if delayed-scaling is not the current training recipe.

    In distributed runs each rank first computes its local statistics; the values
    are gathered the next time `debug_api.step()` is called.  Remember to call
    `debug_api.step()` every training step so the logs are flushed.

    The feature is micro-batch aware: if several forward/backward passes occur
    between successive `debug_api.step()` calls, statistics are accumulated for all
    tensors except weights.

    Collecting FP8 statistics is expensive. Choosing a larger `freq` reduces the
    overhead, and if the feature is skipped for a step the additional cost is
    minimal.  When no other debug feature is active, the layer runs at normal
    Transformer Engine speed.

    Parameters
    ----------

        stats: List[str]
            Each stat is a string of the form `<recipe>_<stat>`, with an optional `_columnwise` suffix (i.e., `<recipe>_<stat>_columnwise`).
            If only `<recipe>` is omitted, the current training recipe is used.
            For mxfp8 and fp8_block_scaling `_columnwise` suffix can be provided. Then stat is computed on columnwise(transpose)
            version of the tensor, which can be numerically different from rowwise (non-transpose) tensors.
            "_columnwise" suffix is not supported for fp8_delayed_scaling and fp8_current_scaling.

            recipes:
                - fp8_delayed_scaling,
                - fp8_current_scaling,
                - mxfp8,
                - fp8_block_scaling,

            stats:
                - underflows% - percentage of non-zero elements of tensor clipped to 0 after quantization,
                - overflows% - percentage of elements of tensor that were clipped to the max/min value of the FP8 range - supported only for fp8_delayed_scaling,
                - scale_inv_min - minimum of the inverse of the scaling factors,
                - scale_inv_max - maximum of the inverse of the scaling factors,
                - mse - mean squared error of the quantized tensor and the original tensor = sum((quantized_tensor - original_tensor)**2) / num_elements,

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

        example_fp8_tensor_stat_collection:
            enabled: True
            layers:
                layer_types: [layernorm_linear]
            transformer_engine:
                LogFp8TensorStats:
                    enabled: True
                    tensors_struct:
                    - tensor: activation
                    stats: [mxfp8_underflows%]
                    freq: 1
                    - tensor: gradient
                    stats: [underflows%]
                    freq: 5
                    start_step: 0
                    end_step: 80
    """

    def check_if_stat_is_supported(self, stat: str, current_recipe: str):
        """Returns True if stat is supported, raises ValueError otherwise."""
        columnwise = stat.endswith("_columnwise")
        if columnwise:
            stat = stat[: -len("_columnwise")]
        recipe_from_stat, _ = self.get_recipe_from_stat(stat, default_recipe=current_recipe)
        stat_without_recipe = stat.replace(recipe_from_stat + "_", "")

        if current_recipe == "" and recipe_from_stat == "":
            raise ValueError(
                f"Stat {stat} does not contain a recipe name and the current recipe is not set."
            )

        if recipe_from_stat != "" and recipe_from_stat not in ALL_RECIPE_NAMES:
            raise ValueError(f"Stat {stat} contains an unsupported recipe name: {recipe_from_stat}")

        if recipe_from_stat in ["fp8_delayed_scaling", "fp8_current_scaling"] and columnwise:
            raise ValueError(
                f"Stat {stat} is not supported. Columnwise tensor statistics are not supported for"
                " fp8_delayed_scaling and fp8_current_scaling."
            )

        if recipe_from_stat == "fp8_delayed_scaling" and stat_without_recipe == "overflows%":
            return True

        if recipe_from_stat in ["fp8_block_scaling"] and torch.cuda.get_device_capability()[0] < 9:
            raise ValueError(f"Stat {stat} needs Hopper or later GPU.")

        if recipe_from_stat == "mxfp8" and torch.cuda.get_device_capability()[0] < 10:
            raise ValueError(f"Stat {stat} needs Blackwell or later GPU.")

        supported_stats = ["underflows%", "scale_inv_min", "scale_inv_max", "mse"]
        if stat_without_recipe not in supported_stats:
            raise ValueError(
                f"Stat {stat} contains an unsupported stat name: {stat_without_recipe}"
            )

        return True

    def get_recipe_from_stat(self, stat: str, default_recipe: str = ""):
        """Returns the recipe name from the stat string."""
        columnwise_stat = stat.endswith("_columnwise")
        for recipe_name in ALL_RECIPE_NAMES:
            if recipe_name in stat:
                return recipe_name, columnwise_stat
        return default_recipe, columnwise_stat

    @contextmanager
    def update_aux_dict(
        self,
        aux_dict: Dict,
        recipe_name: str,
        quantized_tensor: QuantizedTensor,
        quantizer: Quantizer,
        original_tensor: torch.Tensor,
        recipes_in_stats: List[Tuple[str, bool]],
    ):
        """
        Updates the aux_dict with the quantized tensor for each recipe provided in recipes_in_stats.
        It allows to compute stats for different recipes in the same iteration,
        without recomputing the quantized tensor for each recipe for each stat.
        Also updates usage of the quantized tensor with rowwise and columnwise usage.
        Yields the aux_dict.
        Needs to clean after usage, because it possibly change the usage of the quantized tensor.
        """
        fp8_dtype = None
        if recipe_name in ["fp8_delayed_scaling", "fp8_current_scaling", "fp8_block_scaling"]:
            assert isinstance(
                quantizer, (Float8Quantizer, Float8CurrentScalingQuantizer, Float8BlockQuantizer)
            )
            fp8_dtype = quantizer.dtype

        aux_dict = {
            recipe_name: quantized_tensor,
        }

        old_rowwise_usage = quantizer.rowwise_usage
        old_columnwise_usage = quantizer.columnwise_usage
        for cur_recipe_name, cur_columnwise_stat in recipes_in_stats:
            if recipe_name is not cur_recipe_name:
                quantizer = _get_new_quantizer(cur_recipe_name, fp8_dtype)
                aux_dict[cur_recipe_name] = quantizer(original_tensor)
            elif isinstance(quantized_tensor, QuantizedTensor):
                if cur_columnwise_stat:
                    quantized_tensor.update_usage(columnwise_usage=True)
                else:
                    quantized_tensor.update_usage(rowwise_usage=True)
                aux_dict[""] = quantized_tensor
                aux_dict[cur_recipe_name] = quantized_tensor
        try:
            yield aux_dict
        finally:
            if isinstance(quantized_tensor, QuantizedTensor):
                quantized_tensor.update_usage(
                    rowwise_usage=old_rowwise_usage, columnwise_usage=old_columnwise_usage
                )

    @api_method
    def inspect_tensor_enabled(
        self, config: Dict, layer_name: str, tensor_name: str, iteration: int
    ):  # pylint: disable=unused-argument
        """API call used to determine whether to run inspect_tensor_postquantize() in the forward."""
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
    ):
        """
        API call used to collect the data about the tensor after process_tensor()/quantization.
        """
        assert rowwise_quantized_tensor is columnwise_quantized_tensor
        assert (
            quantizer is not None
        ), "[NVTORCH INSPECT ERROR] LogFp8TensorStats cannot be run without low-precision recipe."

        quantized_tensor = rowwise_quantized_tensor
        assert isinstance(
            quantized_tensor, QuantizedTensor
        ), "[NVTORCH INSPECT ERROR] LogFp8TensorStats quantized_tensor must be a QuantizedTensor."
        recipe_name = _get_recipe_name(quantizer)

        for stat in config["stats"]:
            self.check_if_stat_is_supported(stat, recipe_name)

        start_step = config.get("start_step", None)
        end_step = config.get("end_step", None)
        start_end_list = config.get("start_end_list", None)
        if start_end_list is not None:
            start_end_list = tuple(tuple(int(x) for x in interval) for interval in start_end_list)

        options = (
            start_step,
            end_step,
            start_end_list,
            "fp8",
        )

        skip_reduction, reduction_group, reduce_within_microbatch = get_reduction_params(
            tensor_name, tp_group
        )

        STATS_BUFFERS.try_add_buffer(
            layer_name=layer_name,
            tensor_name=tensor_name,
            stats=config["stats"],
            options=options,
            reduction_group=reduction_group,
            reduce_within_microbatch=reduce_within_microbatch,
        )

        recipes_in_stats = [
            self.get_recipe_from_stat(stat, default_recipe=recipe_name) for stat in config["stats"]
        ]

        with self.update_aux_dict(
            aux_dict={},
            recipe_name=recipe_name,
            quantized_tensor=quantized_tensor,
            quantizer=quantizer,
            original_tensor=tensor,
            recipes_in_stats=recipes_in_stats,
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

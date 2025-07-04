# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LogFp8TensorStats Feature support for nvidia-dlframework-inspect"""

from typing import Dict, Optional

import torch

import nvdlfw_inspect.api as debug_api
from nvdlfw_inspect.debug_features.log_tensor_stats import LogTensorStats as BaseLogTensorStats
from nvdlfw_inspect.registry import Registry, api_method

from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS
from transformer_engine.pytorch.tensor import Quantizer
from transformer_engine.pytorch.tensor.float8_tensor import (
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.debug.pytorch.debug_state import TEDebugState


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
        return Float8CurrentScalingQuantizer(fp8_dtype=fp8_dtype, device=torch.device("cuda"))
    if recipe_name == "mxfp8":
        return MXFP8Quantizer(fp8_dtype=fp8_dtype)
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
            If `_columnwise` is provided, then stat is computed on columnwise(transpose) version of the tensor,
            which can be numerically different from rowwise (non-transpose) tensors for mxfp8 and fp8_block_scaling.
            For fp8_delayed_scaling and fp8_current_scaling, the columnwise tensors are
            simply the transpose of the rowwise tensors with the same scaling factors.

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
        if stat.endswith("_columnwise"):
            stat = stat[: -len("_columnwise")]
        recipe_from_stat = self.get_recipe_from_stat(stat)
        stat_without_recipe = stat.replace(recipe_from_stat + "_", "")

        if current_recipe == "" and recipe_from_stat == "":
            raise ValueError(
                f"Stat {stat} does not contain a recipe name and the current recipe is not set."
            )

        if recipe_from_stat != "" and recipe_from_stat not in ALL_RECIPE_NAMES:
            raise ValueError(f"Stat {stat} contains an unsupported recipe name: {recipe_from_stat}")

        if recipe_from_stat == "fp8_delayed_scaling" and stat_without_recipe == "overflows%":
            return True

        if (
            recipe_from_stat in ["mxfp8", "fp8_block_scaling", "fp8_current_scaling"]
            and torch.cuda.get_device_capability()[0] < 9
        ):
            raise ValueError(f"Stat {stat} needs Hopper or later GPU.")

        supported_stats = ["underflows%", "scale_inv_min", "scale_inv_max", "mse"]
        if stat_without_recipe not in supported_stats:
            raise ValueError(
                f"Stat {stat} contains an unsupported stat name: {stat_without_recipe}"
            )

        return True

    def get_recipe_from_stat(self, stat: str):
        """Returns the recipe name from the stat string."""
        for recipe_name in ALL_RECIPE_NAMES:
            if recipe_name in stat:
                return recipe_name
        return ""

    @api_method
    def inspect_tensor_all_enabled(
        self, config: Dict, layer_name: str, gemm: str, tensor_name: str, iteration: int
    ):  # pylint: disable=unused-argument
        """API call used to determine whether to run inspect_tensor_postquantize() in the forward."""
        # check whether logging should happen in this iteration
        return self._check_params(config, layer_name, iteration=iteration)

    @api_method
    def inspect_tensor_all(
        self,
        config: Dict,
        layer_name: str,
        tensor_name: str,
        iteration: int,
        tp_group: torch.distributed.ProcessGroup,
        original_tensor: torch.Tensor,
        quantized_tensor_rowwise: Optional[torch.Tensor] = None,
        quantized_tensor_columnwise: Optional[torch.Tensor] = None,
        quantizer: Optional[Quantizer] = None,
    ):
        """
        API call used to collect the data about the tensor after process_tensor()/quantization.
        """
        assert quantized_tensor_rowwise is quantized_tensor_columnwise
        quantized_tensor = quantized_tensor_rowwise
        recipe_name = _get_recipe_name(quantizer)

        for stat in config["stats"]:
            self.check_if_stat_is_supported(stat, recipe_name)

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

        STATS_BUFFERS.try_add_buffer(
            layer_name=layer_name,
            tensor_name=tensor_name,
            stats=config["stats"],
            options=options,
            reduction_group=reduction_group,
            reduce_within_microbatch=reduce_within_microbatch,
        )

        recipes_in_stats = [
            self.get_recipe_from_stat(stat)
            for stat in config["stats"]
            if self.get_recipe_from_stat(stat) != ""
        ]

        fp8_dtype = None
        if recipe_name in ["fp8_delayed_scaling", "fp8_current_scaling", "fp8_block_scaling"]:
            assert isinstance(
                quantizer, (Float8Quantizer, Float8CurrentScalingQuantizer, Float8BlockQuantizer)
            )
            fp8_dtype = quantizer.dtype

        aux_dict = {
            recipe_name: quantized_tensor,
        }
        for cur_recipe_name in recipes_in_stats:
            if recipe_name is not cur_recipe_name:
                quantizer = _get_new_quantizer(recipe_name, fp8_dtype)
                aux_dict[cur_recipe_name] = quantizer(original_tensor)

        STATS_BUFFERS.feed(
            layer_name,
            tensor_name,
            options,
            original_tensor,
            iteration,
            skip_reduction,
            aux_dict=aux_dict,
        )

        debug_api.log_message(
            f"Feature={self.__class__.__name__}, API=inspect_tensor_all: {tensor_name}",
            layer_name,
            extra_cachable_args=(tensor_name,),
        )

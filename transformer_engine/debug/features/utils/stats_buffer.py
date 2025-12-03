# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Buffer used for LogTensorStats and LogFp8TensorStats features.
Buffers are fed with tensors, they compute necessary stats and save them.
When log() is called, they gather stats from all nodes, compute combined final stats and log them.
"""


from collections import defaultdict
from typing import Dict
import torch

from nvdlfw_inspect.utils import gather_along_first_dim
from nvdlfw_inspect.logging import MetricLogger

from transformer_engine.debug.features.utils.stats_computation import (
    STATS,
    DEPENDENCIES,
    stats_to_num,
)
from transformer_engine.debug.pytorch.debug_state import TEDebugState


class _Buffer:
    """
    Buffer stores temporary statistics for one tensor for one layer.
    It also can synchronize between nodes and log final stats.
    """

    def __init__(self, layer_name, tensor_name, stats, reduction_group, reduce_within_microbatch):
        self.layer_name = layer_name
        self.tensor_name = tensor_name
        self.reduction_group = reduction_group
        self.reduce_within_microbatch = reduce_within_microbatch
        self.stats_to_log = stats

        self.stats_to_compute = set()
        for stat in stats:
            self.stats_to_compute = self.stats_to_compute | DEPENDENCIES[stat]

        self._buffer = torch.zeros(len(STATS), dtype=torch.float32).cuda()
        self._new_buffer = self._buffer.clone()
        self._tmp_buffer = self._buffer.clone()

        # in case of data parallelism it is possible that layer will not be run on one node
        # modified is set to True if node is run
        # we do not take not run nodes into account
        self.modified = torch.tensor([False], dtype=torch.bool).cuda()
        self.iteration = None
        self.skip_reduction = False

    def _reset_before_next_step(self):
        """Resets the state after the logging."""
        self.modified[0] = False

    def _gather_helper_stats(self) -> torch.Tensor:
        """
        If tensor stats should be accumulated among many nodes,
        this method gathers all stats from the nodes where the stat was modified.
        """
        if self.skip_reduction:
            return self._buffer.unsqueeze(0)
        mask = gather_along_first_dim(self.modified, process_group=self.reduction_group)[0]
        gathered_buffer, _ = gather_along_first_dim(
            self._buffer.unsqueeze(0), process_group=self.reduction_group
        )
        return gathered_buffer[mask.to(torch.bool)]

    def feed(self, tensor, iteration, aux_dict=None):
        """
        feed() is used to add tensor for computing the statistics.
        Because of the microbatching, feed() can be used multiple
        times for one log().

        The aux_dict is used to share common computation between different stats.
        For example for LogFp8TensorStats in can contain quantized tensors in different precisions.

        The main reason of this design: need to combine results for already processed
        tensors with the result of the new tensor.
        """

        self.iteration = iteration

        # If the stats are not reduced within microbatch and
        # buffer was fed, we do not change the stats for the tensor.
        # It is used for weights and microbatching.
        if self.modified[0] and not self.reduce_within_microbatch:
            return

        if (
            tensor.numel() == 0
            if hasattr(tensor, "numel")
            else all((t is None or t.numel() == 0) for t in tensor.get_data_tensors())
        ):
            return

        # save stats for tensor to tmp buffer
        for stat_name in self.stats_to_compute:
            fn, _ = STATS[stat_name]
            self._tmp_buffer[stats_to_num[stat_name]] = fn(tensor, aux_dict)

        # [num_buffers, num_stats]
        buffers = torch.cat((self._buffer.unsqueeze(0), self._tmp_buffer.unsqueeze(0)), dim=0)

        for stat_name in self.stats_to_compute:
            fn, combinator = STATS[stat_name]
            if self.modified[0]:
                self._new_buffer[stats_to_num[stat_name]] = combinator(buffers)
            else:
                fn = STATS[stat_name][0]
                self._new_buffer[stats_to_num[stat_name]] = fn(tensor, aux_dict)

        self._buffer.copy_(self._new_buffer)

        self.modified[0] = True

    def log(self):
        """
        Log the tensor stats and resets buffer.
        """
        # [num_active_nodes, num_stats]
        gathered_helper_stats = self._gather_helper_stats()

        if not self.modified[0]:
            return {}
        output = {}
        for stat_name in self.stats_to_log:
            combiner = STATS[stat_name][1]
            stat_value = combiner(gathered_helper_stats)

            # Convert stat key to string for logging (uses __str__ for named tuples)
            stat_name_str = str(stat_name)

            MetricLogger.log_scalar(
                f"{self.layer_name}_{self.tensor_name}_{stat_name_str}", stat_value, self.iteration
            )
            output[(self.layer_name, self.tensor_name, stat_name, self.iteration)] = (
                stat_value  # for debugging purposes
            )
        self._reset_before_next_step()
        return output


class StatsBuffers:
    """
    StatsBuffers class represents all buffers of the statistics for all tensors.
    It is used to feed the tensors to the correct buffers.
    """

    def __init__(self):
        self.buffers = {}  # (layer_name, tensor_name) -> buffer
        self.reduction_group_to_buffer = defaultdict(list)

        # Logging stats involves synchronization between nodes
        # and non-trivial cpu overhead.
        # It should be only done if absolutely necessary.
        # This variables helps to determine if we can reduce.
        self.at_least_one_layer_fed = False
        self.layers_to_next_iter: Dict[str, int] = {}

    def _if_run_reduction(self) -> bool:
        """
        Returns True if reduction should be run.

        This is the case if at least one layer logged stats.
        If not, it may be the case that some layer was not run on this node.
        If we know that such layers on all other nodes do not log this time,
        we can not reduce. If this in not the case, we should reduce.

        To ensure corretness, we assume that every layer is invoked at first forward pass.
        If this is not the case, hang might happen.
        """
        if self.at_least_one_layer_fed:
            return True
        iteration = TEDebugState.get_iteration()
        layers_to_remove = []
        for layer_name, next_iter in self.layers_to_next_iter.items():
            # When next_iter is None the feature will no longer run.
            if next_iter is None:
                layers_to_remove.append(layer_name)
                continue
            # Note that layer can be not run for many iterations,
            # in this case we will synchronize until every step until we get any information from it.
            if iteration >= next_iter:
                return True

        for layer_name in layers_to_remove:
            self.layers_to_next_iter.pop(layer_name, None)
        return False

    def reset(self):
        """Resets all buffers."""
        self.buffers = {}  # (layer_name, tensor_name) -> buffer
        self.reduction_group_to_buffer = defaultdict(list)
        self.at_least_one_layer_fed = False
        self.layers_to_next_iter: Dict[str, int] = {}

    def try_add_buffer(
        self, layer_name, tensor_name, stats, options, reduction_group, reduce_within_microbatch
    ):
        """If buffer for such combination of stats/tensor_name/... is not present, this method creates it."""
        if (layer_name, tensor_name, options) in self.buffers:
            return
        buffer = _Buffer(layer_name, tensor_name, stats, reduction_group, reduce_within_microbatch)
        self.buffers[(layer_name, tensor_name, options)] = buffer
        self.reduction_group_to_buffer[reduction_group].append(buffer)

    def feed(
        self, layer_name, tensor_name, options, tensor, iteration, skip_reduction, aux_dict=None
    ):
        """
        Feeds the tensor into the respective buffer.

        The aux_dict is used to share common computation between different stats.
        For example for LogFp8TensorStats in can contain quantized tensors in different precisions.
        """
        self.at_least_one_layer_fed = True
        buffer = self.buffers[(layer_name, tensor_name, options)]
        buffer.feed(tensor, iteration, aux_dict)
        buffer.skip_reduction = skip_reduction

    def log_stats(self):
        """Logs the stats from all the buffers."""
        if not self._if_run_reduction():
            return {}

        output = {}
        for reduction_group, buffers in self.reduction_group_to_buffer.items():
            changed_buffers = [
                (i, buffer)
                for i, buffer in enumerate(buffers)
                if gather_along_first_dim(
                    buffer.modified.unsqueeze(0), process_group=reduction_group
                )[0].any()
            ]
            for _, buffer in changed_buffers:
                stats = buffer.log()
                output.update(stats)
        self.at_least_one_layer_fed = False
        return output


STATS_BUFFERS = StatsBuffers()

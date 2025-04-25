# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Buffer used for LogTensorStats and LogFp8TensorStats features.
Buffers are fed with tensors, they compute necessary stats and save them.
When log() is called, they gather stats from all nodes, compute combined final stats and log them.
"""


from collections import defaultdict
import torch

from nvdlfw_inspect.utils import gather_along_first_dim
from nvdlfw_inspect.logging import MetricLogger

from transformer_engine.debug.features.utils.stats_computation import (
    STATS,
    DEPENDENCIES,
    stats_to_num,
)


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
        return gathered_buffer[mask.to(bool)]

    def feed(self, tensor, iteration):
        """
        feed() is used to add tensor for computing the statistics.
        Because of the microbatching, feed() can be used multiple
        times for one log().

        The main reason of this design: need to combine results for already processed
        tensors with the result of the new tensor.
        """

        self.iteration = iteration

        # If the stats are not reduced within microbatch and
        # buffer was fed, we do not change the stats for the tensor.
        # It is used for weights and microbatching.
        if self.modified[0] and not self.reduce_within_microbatch:
            return

        # save stats for tensor to tmp buffer
        for stat_name in self.stats_to_compute:
            fn, _ = STATS[stat_name]
            self._tmp_buffer[stats_to_num[stat_name]] = fn(tensor)

        # [num_buffers, num_stats]
        buffers = torch.cat((self._buffer.unsqueeze(0), self._tmp_buffer.unsqueeze(0)), dim=0)

        for stat_name in self.stats_to_compute:
            fn, combinator = STATS[stat_name]
            if self.modified[0]:
                self._new_buffer[stats_to_num[stat_name]] = combinator(buffers)
            else:
                fn = STATS[stat_name][0]
                self._new_buffer[stats_to_num[stat_name]] = fn(tensor)

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

            MetricLogger.log_scalar(
                f"{self.layer_name}_{self.tensor_name}_{stat_name}", stat_value, self.iteration
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

    def reset(self):
        """Resets all buffers."""
        self.buffers = {}  # (layer_name, tensor_name) -> buffer
        self.reduction_group_to_buffer = defaultdict(list)

    def try_add_buffer(
        self, layer_name, tensor_name, stats, options, reduction_group, reduce_within_microbatch
    ):
        """If buffer for such combination of stats/tensor_name/... is not present, this method creates it."""
        if (layer_name, tensor_name, options) in self.buffers:
            return
        buffer = _Buffer(layer_name, tensor_name, stats, reduction_group, reduce_within_microbatch)
        self.buffers[(layer_name, tensor_name, options)] = buffer
        self.reduction_group_to_buffer[reduction_group].append(buffer)

    def feed(self, layer_name, tensor_name, options, tensor, iteration, skip_reduction):
        """Feeds the tensor into the respective buffer."""
        buffer = self.buffers[(layer_name, tensor_name, options)]
        buffer.feed(tensor, iteration)
        buffer.skip_reduction = skip_reduction

    def log_stats(self):
        """Logs the stats from all the buffers."""
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

        return output


STATS_BUFFERS = StatsBuffers()

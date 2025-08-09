# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Managing the state of all the debugged layers.
"""

import sys


class TEDebugState:
    """
    A class to manage the state of debug layers.
    """

    layer_count = 1
    layers_initialized = {}
    weight_tensor_tp_group_reduce = True
    debug_enabled = None

    @classmethod
    def initialize(cls):
        """
        If debug_api module is initialized, then sets cls.debug_enabled to True.
        """

        if "nvdlfw_inspect" in sys.modules:
            import nvdlfw_inspect.api as debug_api

            if cls.debug_enabled is False and debug_api.DEBUG_MANAGER is not None:
                # This method is invoked when initializing TE modules.
                # If this error is thrown, it means that some TE module had been initialized before
                # debug_api was initialized, and now a new TE module is being initialized.
                # This is likely to be a bug.
                raise RuntimeError(
                    "[nv_dlfw_inspect] nv_dlfw_inspect module should be initialized before"
                    " initialization of the first TE module"
                )
            cls.debug_enabled = debug_api.DEBUG_MANAGER is not None

    @classmethod
    def _reset(cls):
        """Resets layer count and stats buffers."""
        from ..features.utils.stats_buffer import STATS_BUFFERS

        STATS_BUFFERS.reset()
        cls.debug_enabled = None
        cls.layers_initialized.clear()

    @classmethod
    def get_layer_count(cls):
        """
        Layer counter is used when layer names are not provided to modules by the user.
        """
        lc = cls.layer_count
        cls.layer_count += 1
        return lc

    @classmethod
    def set_weight_tensor_tp_group_reduce(cls, enabled):
        """Sets weight tensor reduction mode."""
        cls.weight_tensor_tp_group_reduce = enabled

    @classmethod
    def get_iteration(cls):
        """Returns the current iteration."""
        import nvdlfw_inspect.api as debug_api

        return debug_api.DEBUG_MANAGER._trainer_iteration_count


def set_weight_tensor_tp_group_reduce(enabled):
    """Sets weight tensor reduction mode."""
    TEDebugState.set_weight_tensor_tp_group_reduce(enabled)

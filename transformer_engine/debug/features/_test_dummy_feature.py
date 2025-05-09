# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Feature doing nothing, used for testing purposes."""

from nvdlfw_inspect.registry import Registry, api_method
from transformer_engine.debug.features.api import TEConfigAPIMapper


@Registry.register_feature(namespace="transformer_engine")
class TestDummyFeature(TEConfigAPIMapper):
    """
    This is feature used only in tests. It invokes look_at_tensor_before_process
    and does nothing.

    If no features are used, then TE layer automatically switches to the non-debug mode.
    This feature is invoked for each GEMM to prevent this behavior.
    """

    @api_method
    def inspect_tensor_enabled(self, *_args, **_kwargs):
        """API call used to determine whether to run look_at_tensor_before_process
        in the forward pass."""
        return True

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Feature doing nothing, used for testing purposes."""

from nvdlfw_inspect.registry import Registry, api_method
from transformer_engine.debug.features.api import TEConfigAPIMapper

# Module-level counters for tracking invocations
# NOTE: These must be accessed via the full module path
# (transformer_engine.debug.features._test_dummy_feature._inspect_tensor_enabled_call_count)
# to ensure the same module instance is used when the feature is loaded by the debug framework
# and when imported by tests. Using just the variable name would create separate instances
# in different import contexts.
_inspect_tensor_enabled_call_count = 0
_inspect_tensor_call_count = 0


@Registry.register_feature(namespace="transformer_engine")
class TestDummyFeature(TEConfigAPIMapper):
    """
    This is feature used only in tests. It invokes inspect_tensor and does nothing.

    If no features are used, then TE layer automatically switches to the non-debug mode.
    This feature is invoked for each GEMM to prevent this behavior.

    Config options:
    - inspect_only_once: if True, return (False, None) from inspect_tensor_enabled to test caching behavior

    Note: This feature always tracks invocations for testing purposes.
    """

    @api_method
    def inspect_tensor_enabled(self, config, *_args, **_kwargs):
        """API call used to determine whether to run inspect_tensor in the forward pass.

        Always tracks calls for testing purposes.

        Returns:
        - If inspect_only_once=True in config: returns (False, None) - check once, never call inspect_tensor
        - Otherwise: returns True - feature is always enabled
        """
        # Access counter via full module path to ensure we're modifying the same module-level
        # variable regardless of import context (debug framework vs test import)
        import transformer_engine.debug.features._test_dummy_feature as dummy_feature  # pylint: disable=import-self

        dummy_feature._inspect_tensor_enabled_call_count += 1

        inspect_only_once = config.get("inspect_only_once", False)
        if inspect_only_once:
            return False, None
        return True

    @api_method
    def inspect_tensor(self, _config, *_args, **_kwargs):
        """This method does nothing but always tracks invocations for testing."""
        # Access counter via full module path to ensure shared state across import contexts
        import transformer_engine.debug.features._test_dummy_feature as dummy_feature  # pylint: disable=import-self

        dummy_feature._inspect_tensor_call_count += 1

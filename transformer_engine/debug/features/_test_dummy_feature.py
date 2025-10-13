# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Feature doing nothing, used for testing purposes."""

from nvdlfw_inspect.registry import Registry, api_method
from transformer_engine.debug.features.api import TEConfigAPIMapper

import transformer_engine

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
        transformer_engine.debug.features._test_dummy_feature._inspect_tensor_enabled_call_count += 1
        
        inspect_only_once = config.get("inspect_only_once", False)
        if inspect_only_once:
            return False, None
        return True

    @api_method
    def inspect_tensor(self, config, *_args, **_kwargs):
        """This method does nothing but always tracks invocations for testing."""
        transformer_engine.debug.features._test_dummy_feature._inspect_tensor_call_count += 1

    @classmethod
    def reset_call_counts(cls):
        """Reset the call counters for testing."""
        transformer_engine.debug.features._test_dummy_feature._inspect_tensor_enabled_call_count = 0
        transformer_engine.debug.features._test_dummy_feature._inspect_tensor_call_count = 0

    @classmethod
    def get_inspect_tensor_enabled_call_count(cls):
        """Get the number of times inspect_tensor_enabled was called."""
        transformer_engine.debug.features._test_dummy_feature._inspect_tensor_enabled_call_count
        return _inspect_tensor_enabled_call_count

    @classmethod
    def get_inspect_tensor_call_count(cls):
        """Get the number of times inspect_tensor was called."""
        transformer_engine.debug.features._test_dummy_feature._inspect_tensor_call_count
        return _inspect_tensor_call_count

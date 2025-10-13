# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.


import pytest
import torch
import transformer_engine.pytorch as te

import nvdlfw_inspect.api as debug_api

from transformer_engine.debug.pytorch.debug_state import TEDebugState


def test_layer_switches_to_nondebug_mode(configs_dir, feature_dirs):
    """
    Test that layers switch to non-debug mode when no features are active.
    
    Uses TestDummyFeature with inspect_only_once=True, which makes inspect_tensor_enabled return (False, None).
    The TE should:
    1. Call inspect_tensor_enabled to check if feature is needed
    2. Never call inspect_tensor
    3. Allow layers to switch to non-debug mode for optimal performance,
       so that inspect_tensor_enabled is never called again.
    """

    debug_api.end_debug()
    TEDebugState._reset()

    try:
        debug_api.initialize(
            config_file=configs_dir + "/test_switch_to_nondebug_mode.yaml",
            feature_dirs=feature_dirs
        )
        from transformer_engine.debug.features._test_dummy_feature import TestDummyFeature
        TestDummyFeature.reset_call_counts()

        model = te.Linear(256, 256, name="test_linear").cuda()
        x = torch.randn(8, 256, 256).cuda()

        # Run multiple iterations with is_first_microbatch
        for i in range(20):
            is_first_microbatch = (i % 2 == 0)  # Alternate between True and False
            y = model(x, is_first_microbatch=is_first_microbatch)
            y.sum().backward()
            debug_api.step()

        # Verify inspect_tensor_enabled was called only once per tensor 
        # (input, activation, weight, output, wgrad, dgrad)
        enabled_call_count = TestDummyFeature.get_inspect_tensor_enabled_call_count()
        assert enabled_call_count == 6, (
            "inspect_tensor_enabled should be called to check if feature is needed for each tensor "
            "(input, activation, weight, output, wgrad, dgrad)"
        )

        # Verify inspect_tensor was never called - it should not be called if inspect_tensor_enabled returns (False, None)
        inspect_call_count = TestDummyFeature.get_inspect_tensor_call_count()
        assert inspect_call_count == 0, (
            f"inspect_tensor was called {inspect_call_count} times, "
            f"but should never be called when inspect_tensor_enabled returns (False, None)"
        )

    finally:
        debug_api.end_debug()
        TEDebugState._reset()

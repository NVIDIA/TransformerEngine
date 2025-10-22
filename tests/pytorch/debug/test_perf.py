# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.


import pytest
import torch
import transformer_engine.pytorch as te

import nvdlfw_inspect.api as debug_api

from transformer_engine.debug.pytorch.debug_state import TEDebugState


@pytest.mark.parametrize("use_microbatching", [False, True])
def test_layer_switches_to_nondebug_mode(configs_dir, feature_dirs, use_microbatching):
    """
    Test that layers switch to non-debug mode when no features are active.

    Uses TestDummyFeature with inspect_only_once=True, which makes inspect_tensor_enabled return (False, None).
    The TE should:
    1. Call inspect_tensor_enabled to check if feature is needed
    2. Never call inspect_tensor
    3. Allow layers to switch to non-debug mode for optimal performance,
       so that inspect_tensor_enabled is never called again.

    Tests both with and without microbatching to ensure proper behavior in both scenarios.
    """

    try:
        debug_api.initialize(
            config_file=configs_dir + "/test_switch_to_nondebug_mode.yaml",
            feature_dirs=feature_dirs,
        )
        import transformer_engine.debug.features._test_dummy_feature as dummy_feature

        # Reset counters
        dummy_feature._inspect_tensor_enabled_call_count = 0
        dummy_feature._inspect_tensor_call_count = 0

        model = te.Linear(256, 256, name="test_linear").cuda()
        x = torch.randn(8, 256, 256).cuda()

        # Run multiple iterations
        for i in range(20):
            if use_microbatching:
                # Alternate between first and non-first microbatch
                is_first_microbatch = i % 2 == 0
                y = model(x, is_first_microbatch=is_first_microbatch)
            else:
                # Run without specifying is_first_microbatch
                y = model(x)
            y.sum().backward()
            debug_api.step()

        # Verify inspect_tensor_enabled was called only once per tensor
        # (activation, weight, gradient, output, wgrad, dgrad)
        enabled_call_count = dummy_feature._inspect_tensor_enabled_call_count
        microbatch_info = "with microbatching" if use_microbatching else "without microbatching"
        assert enabled_call_count == 6, (
            f"inspect_tensor_enabled was called {enabled_call_count} times ({microbatch_info}), "
            "but should be called 6 times to check if feature is needed for each tensor "
            "(activation, weight, gradient, output, wgrad, dgrad)"
        )

        # Verify inspect_tensor was never called - it should not be called if inspect_tensor_enabled returns (False, None)
        inspect_call_count = dummy_feature._inspect_tensor_call_count
        assert inspect_call_count == 0, (
            f"inspect_tensor was called {inspect_call_count} times ({microbatch_info}), "
            "but should never be called when inspect_tensor_enabled returns (False, None)"
        )

    finally:
        debug_api.end_debug()
        TEDebugState._reset()

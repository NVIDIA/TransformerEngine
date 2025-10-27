# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.


import pytest
import torch
import transformer_engine.pytorch as te
import time

import nvdlfw_inspect.api as debug_api

from transformer_engine.debug.pytorch.debug_state import TEDebugState


def _run_cpu_overhead(debug_tools_initialized, layer, configs_dir, feature_dirs):
    debug_api.end_debug()
    TEDebugState._reset()
    if debug_tools_initialized:
        # This config log stats starting from 0, every N iterations for huge N >> NUM_ITERS.
        # So after 1 warm-up iteration, this layers should work in non-debug mode.
        debug_api.initialize(
            config_file=configs_dir + "/perf_config.yaml", feature_dirs=feature_dirs
        )

    try:
        if layer == "linear":
            model = torch.nn.Sequential(
                te.Linear(1, 1, name="linear1"), te.Linear(1, 1, name="linear2")
            ).cuda()
            NUM_ITERS = 1800
        elif layer == "transformer":
            model = torch.nn.Sequential(
                te.TransformerLayer(1, 1, 1, name="transformer1"),
                te.TransformerLayer(1, 1, 1, name="transformer2"),
            ).cuda()
            NUM_ITERS = 200

        NUM_INVOCATIONS_PER_ITER = 10

        x = torch.randn(1, 1, 1).cuda()

        y = model(x)
        y.sum().backward()
        debug_api.step()
        torch.cuda.synchronize()

        time_start = time.time()
        for i in range(NUM_ITERS):
            for _ in range(NUM_INVOCATIONS_PER_ITER):
                y = model(x)
                y.sum().backward()
            if debug_tools_initialized:
                debug_api.step()
        torch.cuda.synchronize()
        time_end = time.time()

    finally:
        if debug_tools_initialized:
            debug_api.end_debug()

    return time_end - time_start


@pytest.mark.parametrize("layer", ["linear", "transformer"])
def test_cpu_overhead(layer, configs_dir, feature_dirs):
    # runs one layer many times on very small tensor
    # - gpu time should be negligible, so time should be dominated by cpu time.
    # if layers does not invoke any feature in current iteration,
    # then it changed into non-debug mode and should not have any non-negligible cpu overhead
    # compared to layer without debug tools initialized.

    with_debug_tools = _run_cpu_overhead(True, layer, configs_dir, feature_dirs)
    without_debug_tools = _run_cpu_overhead(False, layer, configs_dir, feature_dirs)

    print(f"with_debug_tools: {with_debug_tools} s")
    print(f"without_debug_tools: {without_debug_tools} s")

    assert with_debug_tools < without_debug_tools * 1.25  # 25% overhead margin

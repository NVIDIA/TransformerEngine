# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""conftest for tests/jax"""
import os
import jax
import pytest
from collections import defaultdict
import time


import transformer_engine.jax
from transformer_engine_jax import get_device_compute_capability


@pytest.fixture(autouse=True, scope="function")
def clear_live_arrays():
    """
    Clear all live arrays to keep the resource clean
    """
    yield
    for arr in jax.live_arrays():
        arr.delete()


@pytest.fixture(autouse=True, scope="module")
def enable_fused_attn_after_hopper():
    """
    Enable fused attn for hopper+ arch.
    Fused attn kernels on pre-hopper arch are not deterministic.
    """
    if get_device_compute_capability(0) >= 90:
        os.environ["NVTE_FUSED_ATTN"] = "1"
    yield
    if "NVTE_FUSED_ATTN" in os.environ:
        del os.environ["NVTE_FUSED_ATTN"]


class TestTimingPlugin:
    """
    Plugin to measure test execution time. Enable test timing by setting NVTE_JAX_TEST_TIMING=1
    in the environment.
    """

    def __init__(self):
        self.test_timings = defaultdict(list)

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item):
        item._timing_start = time.time()

    @pytest.hookimpl(trylast=True)
    def pytest_runtest_teardown(self, item, nextitem):
        if hasattr(item, "_timing_start"):
            duration = time.time() - item._timing_start

            # Extract base function name without parameters
            test_name = item.name
            if "[" in test_name:
                base_name = test_name.split("[")[0]
            else:
                base_name = test_name

            self.test_timings[base_name].append(duration)

    def pytest_sessionfinish(self, session, exitstatus):
        print("\n" + "=" * 80)
        print("TEST RUNTIME SUMMARY (grouped by function)")
        print("=" * 80)

        total_overall = 0
        for test_name, durations in sorted(self.test_timings.items()):
            total_time = sum(durations)
            count = len(durations)
            avg_time = total_time / count if count > 0 else 0
            total_overall += total_time

            print(f"{test_name:<60} | {count:3}x | {total_time:7.2f}s | avg: {avg_time:6.2f}s")

        print("=" * 80)
        print(f"{'TOTAL RUNTIME':<60} | {'':>3}  | {total_overall:7.2f}s |")
        print("=" * 80)


def pytest_configure(config):
    if os.getenv("NVTE_JAX_TEST_TIMING", "0") == "1":
        config.pluginmanager.register(TestTimingPlugin(), "test_timing")

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""conftest for tests/jax"""
import os
import jax
import pytest

from transformer_engine.transformer_engine_jax import get_device_compute_capability


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

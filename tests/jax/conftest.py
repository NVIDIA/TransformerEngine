# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""conftest for tests/jax"""
import jax
import pytest


@pytest.fixture(autouse=True, scope="function")
def clear_live_arrays():
    """
    Clear all live arrays to keep the resource clean
    """
    yield
    for arr in jax.live_arrays():
        arr.delete()

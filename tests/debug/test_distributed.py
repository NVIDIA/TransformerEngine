# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import subprocess
from pathlib import Path

import pytest
import torch

"""
    Distributed numerics tests

    These tests test the numerical corectness of the TransformerEngine layers.
    Tests are parametrized by the layer and fp8 precision.
    One test consists of running multiple configurations from file run_numerics.py
    Such design is due to the fact the initialization of one test is long
    - 2 processes need to start and load torch and TE. Multiple configurations
    are run in one test - this reduces the initialization overhead.

"""


if torch.cuda.device_count() < 2:
    pytest.skip("Distributed training needs at least 2 GPUs.")

TEST_ROOT = Path(__file__).parent.resolve()
NUM_PROCS: int = min(4, torch.cuda.device_count())
LAUNCH_CMD = ["torchrun", f"--nproc_per_node={NUM_PROCS}"]


def test_debug_distributed(feature_dirs):
    test_path = TEST_ROOT / "run_distributed.py"
    test_cmd = LAUNCH_CMD + [str(test_path), f"--feature_dirs={feature_dirs[0]}"]

    result = subprocess.run(test_cmd, env=os.environ, capture_output=True, check=False)
    if result.returncode != 0:
        raise AssertionError(result.stderr.decode())

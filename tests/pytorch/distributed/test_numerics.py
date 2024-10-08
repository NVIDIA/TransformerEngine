# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import subprocess
from pathlib import Path

import pytest
import torch
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

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

fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()

TEST_ROOT = Path(__file__).parent.resolve()
NUM_PROCS: int = min(4, torch.cuda.device_count())
LAUNCH_CMD = ["torchrun", f"--nproc_per_node={NUM_PROCS}"]


def _run_test(fp8):
    test_path = TEST_ROOT / "run_numerics.py"
    test_cmd = LAUNCH_CMD + [str(test_path)]

    if fp8:
        test_cmd += ["--fp8"]

    result = subprocess.run(test_cmd, env=os.environ, capture_output=True, check=False)
    if result.returncode != 0 or "NUMERICAL CHECK FAILED" in result.stderr.decode():
        raise AssertionError(result.stderr.decode())


all_boolean = [True, False]


@pytest.mark.parametrize("fp8", all_boolean)
def test_distributed(fp8):
    if fp8 and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    _run_test(fp8)

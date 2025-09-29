# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = (
    FP8GlobalStateManager.is_fp8_block_scaling_available()
)
nvfp4_available, reason_for_no_nvfp4 = FP8GlobalStateManager.is_nvfp4_available()

TEST_ROOT = Path(__file__).parent.resolve()
NUM_PROCS: int = min(4, torch.cuda.device_count())
LAUNCH_CMD = ["torchrun", f"--nproc_per_node={NUM_PROCS}"]


def _run_test(quantization):
    test_path = TEST_ROOT / "run_numerics.py"
    test_cmd = LAUNCH_CMD + [str(test_path)]

    if quantization is not None:
        test_cmd += ["--quantization", quantization]

    result = subprocess.run(test_cmd, env=os.environ, check=False)
    assert result.returncode == 0


all_boolean = [True, False]


@pytest.mark.parametrize(
    "quantization", [None, "fp8", "mxfp8", "fp8_cs", "fp8_block_scaling", "nvfp4"]
)
def test_distributed(quantization):
    if quantization == "fp8" and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if quantization == "fp8_cs" and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if quantization == "mxfp8" and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    if quantization == "fp8_block_scaling" and not fp8_block_scaling_available:
        pytest.skip(reason_for_no_fp8_block_scaling)
    if quantization == "nvfp4" and not nvfp4_available:
        pytest.skip(reason_for_no_nvfp4)
    _run_test(quantization)

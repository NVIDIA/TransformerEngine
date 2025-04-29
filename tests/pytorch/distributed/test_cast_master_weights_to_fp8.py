# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import subprocess
from pathlib import Path

import pytest
import torch
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager


if torch.cuda.device_count() < 2:
    pytest.skip("cast_master_weights_to_fp8 test needs at least 2 GPUs.")

fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = (
    FP8GlobalStateManager.is_fp8_block_scaling_available()
)

TEST_ROOT = Path(__file__).parent.resolve()
NUM_PROCS: int = min(2, torch.cuda.device_count())
LAUNCH_CMD = ["torchrun", f"--nproc_per_node={NUM_PROCS}"]


def _run_test(quantization):
    test_path = TEST_ROOT / "run_cast_master_weights_to_fp8.py"
    test_cmd = LAUNCH_CMD + [str(test_path)] + ["--quantization", quantization]
    result = subprocess.run(test_cmd, env=os.environ, check=False)
    assert result.returncode == 0


@pytest.mark.parametrize("quantization", ["fp8", "fp8_cs", "fp8_block"])
def test_cast_master_weights_to_fp8(quantization):
    if quantization in ("fp8", "fp8_cs") and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if quantization == "fp8_block" and not fp8_block_scaling_available:
        pytest.skip(reason_for_no_fp8_block_scaling)
    _run_test(quantization)

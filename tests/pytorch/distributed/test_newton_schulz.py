# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for distributed Newton-Schulz inverse square root."""

import os
import subprocess
from pathlib import Path

import pytest
import torch

if torch.cuda.device_count() < 2:
    pytest.skip("Newton-Schulz tests require at least 2 GPUs.", allow_module_level=True)

TEST_ROOT = Path(__file__).parent.resolve()
NUM_PROCS = torch.cuda.device_count()
LAUNCH_CMD = ["torchrun", f"--nproc_per_node={NUM_PROCS}"]


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("matrix_size", [256])
@pytest.mark.parametrize("num_iterations", [5, 15])
def test_newton_schulz(dtype, matrix_size, num_iterations):
    """Test distributed Newton-Schulz matrix orthogonalization."""
    test_path = TEST_ROOT / "run_newton_schulz.py"
    test_cmd = LAUNCH_CMD + [
        str(test_path),
        f"--dtype={dtype}",
        f"--matrix-size={matrix_size}",
        f"--num-iterations={num_iterations}",
    ]
    if dtype == "bfloat16":
        test_cmd += ["--atol=5e-2", "--rtol=5e-2"]

    result = subprocess.run(test_cmd, env=os.environ, capture_output=True, check=False)
    if (
        result.returncode != 0
        or "NUMERICAL CHECK FAILED" in result.stderr.decode()
        or "NUMERICAL CHECK PASSED" not in result.stdout.decode()
    ):
        raise AssertionError(
            "Newton-Schulz test failed.\n"
            f"stdout: {result.stdout.decode()}\n"
            f"stderr: {result.stderr.decode()}"
        )

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for distributed Newton-Schulz matrix orthogonalization."""

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
ORTHOGONALITY_SHAPES = [
    (NUM_PROCS * 64, NUM_PROCS * 64),
    (NUM_PROCS * 64, NUM_PROCS * 96),
    (NUM_PROCS * 96, NUM_PROCS * 64),
]
REFERENCE_SHAPES = [(NUM_PROCS * 64, NUM_PROCS * 64)]


def _run_test(dtype, matrix_shape, num_iterations, coeff_type, check):
    rows, cols = matrix_shape
    test_path = TEST_ROOT / "run_newton_schulz.py"
    test_cmd = LAUNCH_CMD + [
        str(test_path),
        f"--check={check}",
        f"--dtype={dtype}",
        f"--matrix-rows={rows}",
        f"--matrix-cols={cols}",
        f"--num-iterations={num_iterations}",
        f"--coeff-type={coeff_type}",
    ]
    if dtype == "bfloat16":
        test_cmd += ["--atol=5e-2", "--rtol=5e-2"]

    result = subprocess.run(test_cmd, env=os.environ, capture_output=True, check=False, timeout=300)
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


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("matrix_shape", ORTHOGONALITY_SHAPES)
@pytest.mark.parametrize("num_iterations,coeff_type", [(5, "quintic"), (8, "polar_express")])
def test_orthogonality(dtype, matrix_shape, num_iterations, coeff_type):
    """Test distributed Newton-Schulz orthogonality."""
    _run_test(dtype, matrix_shape, num_iterations, coeff_type, "orthogonality")


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("matrix_shape", REFERENCE_SHAPES)
@pytest.mark.parametrize("num_iterations,coeff_type", [(5, "quintic"), (8, "polar_express")])
def test_against_reference(dtype, matrix_shape, num_iterations, coeff_type):
    """Test distributed Newton-Schulz against a local reference implementation."""
    _run_test(dtype, matrix_shape, num_iterations, coeff_type, "reference")

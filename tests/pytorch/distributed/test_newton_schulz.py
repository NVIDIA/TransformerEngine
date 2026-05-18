# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for distributed Newton-Schulz matrix orthogonalization."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

NUM_PROCS = torch.cuda.device_count()
if NUM_PROCS < 1:
    pytest.skip("Newton-Schulz tests require at least 1 GPU.", allow_module_level=True)

TEST_ROOT = Path(__file__).parent.resolve()


def _run_worker(num_procs: int) -> None:
    test_path = TEST_ROOT / "run_newton_schulz.py"
    test_cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={num_procs}",
        str(test_path),
    ]
    result = subprocess.run(
        test_cmd,
        env=os.environ,
        capture_output=True,
        check=False,
        timeout=1200,
    )
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


def test_newton_schulz_single_gpu():
    """Test cuSolverMp Newton-Schulz with a single-rank GPU grid."""
    _run_worker(1)


@pytest.mark.skipif(
    NUM_PROCS < 2, reason="Distributed Newton-Schulz tests require at least 2 GPUs."
)
def test_newton_schulz_distributed():
    """Launch one parallel job that runs all multi-GPU Newton-Schulz checks."""
    _run_worker(NUM_PROCS)

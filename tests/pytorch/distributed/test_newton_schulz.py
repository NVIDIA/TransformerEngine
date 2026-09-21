# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for distributed Newton-Schulz matrix orthogonalization."""

import mmap
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformer_engine.common import _get_shared_object_file

NUM_PROCS = torch.cuda.device_count()
if NUM_PROCS < 1:
    pytest.skip("Newton-Schulz tests require at least 1 GPU.", allow_module_level=True)


def _built_with_cusolvermp() -> bool:
    """Whether Transformer Engine was compiled with NVTE_WITH_CUSOLVERMP=1.

    There is no Python-level query for this build option, but cuSolverMp is a
    link-time dependency, so libtransformer_engine.so has a DT_NEEDED entry for
    libcusolverMp.so if and only if it was built with support.
    """
    so_path = _get_shared_object_file("core")
    with open(so_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        return mm.find(b"libcusolverMp") != -1


pytestmark = pytest.mark.skipif(
    not _built_with_cusolvermp(),
    reason="Transformer Engine not built with cuSolverMp (NVTE_WITH_CUSOLVERMP=0).",
)

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

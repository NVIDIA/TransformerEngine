# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Pytest driver — spawns run_ep.py under torchrun and asserts the suite passed."""

import os
import subprocess
from pathlib import Path

import pytest
import torch

TEST_ROOT = Path(__file__).parent.resolve()
WORKER = TEST_ROOT / "run_ep.py"
LAUNCHER = TEST_ROOT / "run_test_ep.sh"


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="EP requires >= 4 GPUs")
def test_multi_process_ep():
    """Launch the EP unit-test suite across all visible GPUs.

    Short timeout so a hang on any rank surfaces fast rather than burning CI time.
    """
    timeout_s = int(os.environ.get("NVTE_TEST_EP_TIMEOUT_S", "180"))
    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        env={**os.environ, "KEEP_EP_LOGS": "1", "TEST_TIMEOUT_S": str(timeout_s)},
        timeout=timeout_s + 30,
        check=False,
    )
    assert proc.returncode == 0, f"EP test suite failed (rc={proc.returncode})"

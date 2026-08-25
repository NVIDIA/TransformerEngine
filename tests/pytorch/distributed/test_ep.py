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


def _count_launcher_passes() -> int:
    # Count run_pass invocations so the outer timeout scales as passes are added.
    n = 0
    for line in LAUNCHER.read_text().splitlines():
        s = line.strip()
        if s.startswith("run_pass ") or s.startswith("run_pass\t"):
            n += 1
    return max(n, 1)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="EP requires >= 4 GPUs")
def test_multi_process_ep():
    """Launch the EP unit-test suite across all visible GPUs.

    Per-pass timeout stays short so a hang on any rank surfaces fast; the outer
    pytest budget scales with the number of passes the launcher runs.
    """
    per_pass_s = int(os.environ.get("NVTE_TEST_EP_TIMEOUT_S", "180"))
    outer_s = per_pass_s * _count_launcher_passes() + 60
    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        env={**os.environ, "KEEP_EP_LOGS": "1", "TEST_TIMEOUT_S": str(per_pass_s)},
        timeout=outer_s,
        check=False,
    )
    assert proc.returncode == 0, f"EP test suite failed (rc={proc.returncode})"

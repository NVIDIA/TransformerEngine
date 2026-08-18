# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Pytest driver — spawns run_deepseek_ep.py under torchrun and asserts it passed."""

import os
import subprocess
from pathlib import Path

import pytest
import torch

TEST_ROOT = Path(__file__).parent.resolve()
LAUNCHER = TEST_ROOT / "run_test_deepseek_ep.sh"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="DeepSeek EP requires >= 2 GPUs")
def test_multi_process_deepseek_ep():
    timeout_s = int(os.environ.get("NVTE_TEST_EP_TIMEOUT_S", "180"))
    proc = subprocess.run(
        ["bash", str(LAUNCHER)],
        env={**os.environ, "KEEP_EP_LOGS": "1", "TEST_TIMEOUT_S": str(timeout_s)},
        timeout=timeout_s + 30,
        check=False,
    )
    assert proc.returncode == 0, f"DeepSeek EP test suite failed (rc={proc.returncode})"

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import subprocess
from pathlib import Path

import pytest
import torch

TEST_ROOT = Path(__file__).parent.resolve()
NUM_PROCS = min(8, torch.cuda.device_count())
LAUNCH_CMD = ["torchrun", f"--nproc_per_node={NUM_PROCS}"]


def _has_nvlink() -> bool:
    # NCCL EP falls back to the network transport and deadlocks on PCIe-only nodes.
    out = subprocess.run(
        ["nvidia-smi", "nvlink", "--status"], capture_output=True, text=True, check=False
    ).stdout
    return "GB/s" in out


@pytest.mark.skipif(NUM_PROCS < 2, reason="EP requires >= 2 GPUs")
@pytest.mark.skipif(not _has_nvlink(), reason="NCCL EP requires NVLink")
def test_deepseek_layer_ep():
    result = subprocess.run(
        LAUNCH_CMD + [str(TEST_ROOT / "run_models.py")], env=os.environ, check=False, timeout=300
    )
    assert result.returncode == 0

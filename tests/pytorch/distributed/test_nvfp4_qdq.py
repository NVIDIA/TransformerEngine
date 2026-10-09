# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pytest launcher for distributed NVFP4 QDQ reference parity."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import transformer_engine.pytorch as te


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="QDQ amax reduction requires two GPUs")
def test_nvfp4_qdq_amax_reduction():
    available, reason = te.is_nvfp4_available(return_reason=True)
    if not available:
        pytest.skip(reason)
    script = Path(__file__).with_name("run_nvfp4_qdq.py")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(script),
        ],
        env=os.environ.copy(),
        check=True,
    )

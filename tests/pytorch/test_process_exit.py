# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""The process exits cleanly after NVRTC kernels have been loaded."""

import os
import subprocess
import sys

import pytest
import torch

# LayerNorm runs NVRTC-compiled kernels, so rtc::KernelManager holds loaded CUDA modules that its
# static destructor unloads at exit.
SCRIPT = """
import torch
import transformer_engine.pytorch as te

layer = te.LayerNorm(256).cuda()
x = torch.randn(64, 256, device="cuda", requires_grad=True)
layer(x).sum().backward()
torch.cuda.synchronize()
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@pytest.mark.skipif(bool(int(os.getenv("NVTE_DISABLE_NVRTC", "0"))), reason="Requires NVRTC")
def test_exit_after_nvrtc_kernels():
    # MALLOC_PERTURB_ makes glibc overwrite freed memory, so an exit-time destructor that reads an
    # already-destroyed object crashes every time rather than only sometimes.
    env = dict(os.environ, MALLOC_PERTURB_="165")
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, f"exit status {result.returncode}\n{result.stderr}"

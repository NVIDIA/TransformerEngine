# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import pytest
import subprocess
from pathlib import Path
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
import torch
from packaging.version import Version as PkgVersion


def get_torch_version():
    """Get pytorch version from __version__"""

    def get_torch_version_str():
        import torch

        return str(torch.__version__)

    return PkgVersion(get_torch_version_str())


if torch.cuda.device_count() < 4:
    pytest.skip("FSDP2 test requires at least 4 GPUs.")

if torch.cuda.device_count() % 2 != 0:
    pytest.skip("Number of device should be divided by 2.")

if not get_torch_version() >= PkgVersion("2.4"):
    pytest.skip("FSDP2 requires PyTorch >= 2.4.0 with FSDP 2 support.")

fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()

TEST_ROOT = Path(__file__).parent.resolve()
NUM_PROCS: int = torch.cuda.device_count()
LAUNCH_CMD = ["torchrun", f"--nproc_per_node={NUM_PROCS}"]


def _run_test(fp_init, sharding_dims):
    test_path = TEST_ROOT / "run_fsdp2_model.py"
    test_cmd = LAUNCH_CMD + [str(test_path)]

    if fp_init:
        test_cmd += ["--fp8-init"]
    if len(sharding_dims) == 1:
        test_cmd += ["--sharding-dims", str(sharding_dims[0])]
    elif len(sharding_dims) == 2:
        test_cmd += ["--sharding-dims", str(sharding_dims[0]), str(sharding_dims[1])]
    else:
        assert False
    result = subprocess.run(test_cmd, env=os.environ, capture_output=True, check=False)
    if result.returncode != 0:
        raise AssertionError(result.stderr.decode())


all_boolean = [True, False]
sharding_dims = [[NUM_PROCS], [2, NUM_PROCS // 2]]


@pytest.mark.parametrize("sharding_dims", sharding_dims)
@pytest.mark.parametrize("fp8_init", all_boolean)
def test_distributed(fp8_init, sharding_dims):
    if fp8_init and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    _run_test(fp8_init, sharding_dims)

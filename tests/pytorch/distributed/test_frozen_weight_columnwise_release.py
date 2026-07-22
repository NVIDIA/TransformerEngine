# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Distributed tests for NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE.

Each case launches a torchrun subprocess (see run_fsdp2_frozen_release.py
and run_tpsp_frozen_release.py for the executed checks).
"""

import os
import subprocess
from pathlib import Path

import pytest
import torch
import transformer_engine.pytorch as te

if torch.cuda.device_count() < 2:
    pytest.skip(
        "NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE distributed tests need at least 2 GPUs.",
        allow_module_level=True,
    )

fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)

TEST_ROOT = Path(__file__).parent.resolve()
NUM_PROCS: int = 2
LAUNCH_CMD = ["torchrun", f"--nproc_per_node={NUM_PROCS}"]

_CASES = (
    ("fsdp2_reshard_true", "run_fsdp2_frozen_release.py", ["--reshard-after-forward", "1"]),
    ("fsdp2_reshard_false", "run_fsdp2_frozen_release.py", ["--reshard-after-forward", "0"]),
    ("tp_sp_smoke", "run_tpsp_frozen_release.py", []),
)


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case[0])
def test_distributed_frozen_columnwise_release(case):
    _, script, extra_args = case
    test_cmd = LAUNCH_CMD + [str(TEST_ROOT / script)] + extra_args
    result = subprocess.run(test_cmd, env=os.environ, check=False)
    assert result.returncode == 0

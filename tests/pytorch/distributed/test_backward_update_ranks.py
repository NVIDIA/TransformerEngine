# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import subprocess
from pathlib import Path

import pytest
import torch
import transformer_engine.pytorch as te

if torch.cuda.device_count() < 2:
    pytest.skip("Distributed training needs at least 2 GPUs.", allow_module_level=True)

fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)

TEST_ROOT = Path(__file__).parent.resolve()
NUM_PROCS: int = min(4, torch.cuda.device_count())
LAUNCH_CMD = ["torchrun", f"--nproc_per_node={NUM_PROCS}"]


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("case", ["skipped_module_backward", "no_backward_in_scope"])
def test_backward_update_ranks(case):
    test_cmd = LAUNCH_CMD + [str(TEST_ROOT / "run_backward_update_ranks.py"), "--case", case]
    result = subprocess.run(test_cmd, env=os.environ, check=False, timeout=600)
    assert result.returncode == 0

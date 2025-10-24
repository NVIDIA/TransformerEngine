# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import subprocess
from pathlib import Path

import pytest
import torch
import transformer_engine.pytorch as te

"""
    Distributed numerics tests

    This numerical test aims for zero tolerance test for absolute confidence in numerics.
    In the case of NVFP4, with the custom NVFP4 quantization, we matched bitwise
    result with the native silicon. For distrbuted test cases, we can do the same by thing
    by comparing BF16 AG results with the low precision AG results at layer level.
"""


if torch.cuda.device_count() < 2:
    pytest.skip("Distributed training needs at least 2 GPUs.")

fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)
nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)

TEST_ROOT = Path(__file__).parent.resolve()
NUM_PROCS: int = min(4, torch.cuda.device_count())
LAUNCH_CMD = ["torchrun", f"--nproc_per_node={NUM_PROCS}"]


def _run_test(quantization, batch_size, hidden_size, out_size):
    test_path = TEST_ROOT / "run_numerics_exact.py"
    test_cmd = LAUNCH_CMD + [str(test_path)]

    test_cmd += ["--quantization", quantization]
    test_cmd += ["--batch-size", str(batch_size)]
    test_cmd += ["--hidden-size", str(hidden_size)]
    test_cmd += ["--out-size", str(out_size)]

    result = subprocess.run(test_cmd, env=os.environ, check=False)
    assert result.returncode == 0


all_boolean = [True, False]


@pytest.mark.parametrize("quantization", ["nvfp4"])
@pytest.mark.parametrize(
    "batch_size, hidden_size, out_size",
    [
        (64, 128, 128),
        (128, 128, 128),
        (128, 256, 256),
        (512, 1024, 768),
        (512, 256, 1024),
        (2048, 2048, 2048),
    ],
)
def test_distributed(quantization, batch_size, hidden_size, out_size):
    if quantization == "nvfp4" and not nvfp4_available:
        pytest.skip(reason_for_no_nvfp4)

    _run_test(quantization, batch_size, hidden_size, out_size)

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Launch the mixed CP backward checks on supported GPU configurations."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


@pytest.mark.parametrize("cp_size,tp_size", [(2, 1), (4, 1), (8, 1), (2, 2), (4, 2)])
def test_mixed_context_parallel(cp_size, tp_size, tmp_path):
    if torch.cuda.device_count() < cp_size * tp_size:
        pytest.skip("Not enough GPUs for this CP/TP configuration")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Mixed CP backward currently targets SM100")
    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
        _flash_attn_bwd_v4,
    )

    if _flash_attn_bwd_v4 is None:
        pytest.skip("FlashAttention-4 backward is unavailable")
    env = dict(
        os.environ,
        TEST_TP_SIZE=str(tp_size),
        XML_LOG_DIR=str(tmp_path),
        NVTE_FLASH_ATTN="0",
        NVTE_FUSED_ATTN="1",
        NVTE_FUSED_ATTN_BACKEND="1",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={cp_size * tp_size}",
            str(Path(__file__).with_name("run_mixed_cp.py")),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import pathlib
import sys

import pytest
import torch

from transformer_engine.pytorch import get_device_compute_capability

_current_file = pathlib.Path(__file__).resolve()
sys.path.append(str(_current_file.parent.parent))
from utils import run_distributed


@pytest.mark.skipif(get_device_compute_capability() < (9, 0), reason="THD format requires sm90+.")
@pytest.mark.parametrize("world_size", [2, 4])
def test_cp_thd_tail_padding(world_size):
    """THD + CP(p2p) with tail padding: deterministic and matches no-CP reference.

    Regression test for https://github.com/NVIDIA/TransformerEngine/issues/3331 —
    the pad_between_seqs auto-detect must route tail-padded THD batches to the
    exact per-rank path under context parallelism.
    """
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Test requires {world_size} GPUs!")

    run_distributed(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={world_size}",
            "--standalone",
            str(_current_file.parent / "run_cp_thd_tail_padding.py"),
        ],
        env={
            **os.environ,
            "NVTE_FLASH_ATTN": "0",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        },
        timeout=600,
    )

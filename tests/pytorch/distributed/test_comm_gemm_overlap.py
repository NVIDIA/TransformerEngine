# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import os
import subprocess
from pathlib import Path

import pytest
import torch
import transformer_engine
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

RNG_SEED:     int = 1234
NUM_PROCS:    int = torch.cuda.device_count()
SEQ_LENGTH:   int = 1024
BATCH_SIZE:   int = 2
NUM_HEADS:    int = 64
HEAD_DIM:     int = 128
HIDDEN_SIZE:  int = NUM_HEADS * HEAD_DIM

TE_PATH = Path(os.getenv("TE_PATH", str(Path(transformer_engine.__file__).resolve().parent.parent)))
BASE_CMD = [
    'torchrun', f'--nproc-per-node={torch.cuda.device_count()}',
    str(TE_PATH / 'examples' / 'pytorch' / 'comm_gemm_overlap' / 'test_gemm.py'),
    '--check-numerics'
]

os.environ['UB_SKIPMC'] = '1'  # CI environment may not support multicast so use CUDA IPC instead.
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()

@pytest.mark.parametrize(
    "fp8,p2p,comm_type,aggregate",
    [(False, True,  'AG', False), (False, True, 'AG', True),
     (True,  True,  'AG', False), (True,  True, 'AG', True),
     (False, False, 'RS', False), (False, True, 'RS', False),
     (True,  False, 'RS', False), (True,  True, 'RS', False)],
    ids=['BF16 RING-EXCHANGE ALL-GATHER', 'BF16 AGGREGATED RING-EXCHANGE ALL-GATHER',
         'FP8 RING-EXCHANGE ALL-GATHER',  'FP8 AGGREGATED RING-EXCHANGE ALL-GATHER',
         'BF16 COLLECTIVE REDUCE-SCATTER', 'BF16 RING-EXCHANGE REDUCE-SCATTER',
         'FP8 COLLECTIVE REDUCE-SCATTER',  'FP8 RING-ECHANGE REDUCE-SCATTER'])
def test_split_gemm_overlap(fp8, p2p, comm_type, aggregate):
    """Test communication overlap with split GEMM."""
    if fp8 and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    test_cmd = BASE_CMD + [ '--comm-type', comm_type ]
    if fp8:
        test_cmd.append('--fp8')
    if p2p:
        test_cmd.append('--p2p')
    if aggregate:
        test_cmd.append('--aggregate')
    subprocess.run(
        test_cmd,
        env=os.environ,
        check=True
    )

# NOTE: Atomic GEMM overlaps suffer from hangs. Test is disabled while issue is being actively
#       worked on.

# @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
# @pytest.mark.parametrize(
#     "p2p", [False, True],
#     ids=['RING-EXCHANGE ALL-GATHER + COLLECTIVE REDUCE-SCATTER',
#          'RING-EXCHANGE ALL-GATHER + RING-EXHCANGE REDUCE-SCATTER'])
# def test_paired_atomic_gemm_overlap_fp8(p2p):
#     """Test communication overlap with atomic GEMM."""
#     test_cmd = BASE_CMD + ['--fp8 --atomic']
#     if p2p:
#         test_cmd.append('--p2p')
#     subprocess.run(
#         test_cmd,
#         env=os.environ,
#         check=True
#     )

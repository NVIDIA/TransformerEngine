# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import os
import subprocess
from pathlib import Path

import pytest
import torch
import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()

RNG_SEED: int = 1234
SEQ_LENGTH: int = 2024
BATCH_SIZE: int = 2
NUM_HEADS: int = 64
HEAD_DIM: int = 128

TEST_ROOT = Path(__file__).parent.resolve()
NUM_PROCS: int = min(torch.cuda.device_count(), 4)
LAUNCH_CMD = ["torchrun", f"--nproc_per_node={NUM_PROCS}"]

# Fall back on CUDA IPC if the platform does not support CUDA multicast
if not tex.device_supports_multicast():
    os.environ["UB_SKIPMC"] = "1"

# Force GPU kernels to launch in the order they're executed by the host CPU
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"


@pytest.mark.skipif(NUM_PROCS < 2, reason="Comm+GEMM overlap requires at least 2 GPUs.")
@pytest.mark.parametrize(
    "fp8,p2p,comm_type,aggregate,atomic,bulk",
    [
        # FP8, P2P, Type, Aggregate, Atomic, Bulk
        (False, True, "AG", False, False, False),
        (False, True, "AG", True, False, False),
        (True, True, "AG", False, False, False),
        (True, True, "AG", True, False, False),
        (False, False, "RS", False, False, False),
        (False, True, "RS", False, False, False),
        (True, False, "RS", False, False, False),
        (True, True, "RS", False, False, False),
        (True, False, "RS", False, True, False),
        (True, True, "RS", False, True, False),
        (False, False, "AG", False, False, True),
        (False, False, "RS", False, False, True),
    ],
    ids=[
        "  AG -> SPLIT GEMM | BF16 | RING-EXCHANGE ",
        "  AG -> SPLIT GEMM | BF16 | RING-EXCHANGE (2X AGGREGATED) ",
        "  AG -> SPLIT GEMM | FP8  | RING-EXCHANGE ",
        "  AG -> SPLIT GEMM | FP8  | RING-EXCHANGE (2X AGGREGATED) ",
        "  SPLIT GEMM -> RS | BF16 | PIPELINE ",
        "  SPLIT GEMM -> RS | BF16 | RING-EXCHANGE ",
        "  SPLIT GEMM -> RS | FP8  | PIPELINE ",
        "  SPLIT GEMM -> RS | FP8  | RING-EXCHANGE ",
        " ATOMIC GEMM -> RS | FP8  | PIPELINE ",
        " ATOMIC GEMM -> RS | FP8  | RING-EXCHANGE ",
        "    BULK AG & GEMM | BF16 | PIPELINE ",
        "    BULK RS & GEMM | BF16 | PIPELINE ",
    ],
)
def test_gemm_with_overlap(fp8, p2p, comm_type, aggregate, atomic, bulk):
    """
    Test comm+GEMM overlap algorithms with direct calls to
    te.cpp_extensions.gemm or te.cpp_extensions.fp8_gemm
    """
    test_path = TEST_ROOT / "run_gemm_with_overlap.py"
    test_cmd = (
        LAUNCH_CMD
        + [str(test_path)]
        + [
            "--check-numerics",
            f"--seed={RNG_SEED}",
            f"--seq-length={SEQ_LENGTH}",
            f"--batch-size={BATCH_SIZE}",
            f"--num-heads={NUM_HEADS}",
            f"--head-dim={HEAD_DIM}",
            f"--comm-type={comm_type}",
        ]
    )

    if bulk:
        test_cmd.append("--bulk-overlap")
    else:
        if fp8:
            if not fp8_available:
                pytest.skip(reason_for_no_fp8)
            test_cmd.append("--fp8")
        if p2p:
            test_cmd.append("--p2p")
        if aggregate:
            test_cmd.append("--aggregate")
        if atomic:
            if torch.cuda.get_device_properties(0).major < 9:
                pytest.skip("Device compute capability 9.0 or higher is required for Atomic GEMM.")
            test_cmd.append("--atomic")

    output = subprocess.run(test_cmd, env=os.environ, text=True, capture_output=True, check=False)
    assert "NUMERICAL CHECK PASSED" in str(output)

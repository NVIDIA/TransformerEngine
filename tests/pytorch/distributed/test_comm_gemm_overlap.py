# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import os
import subprocess
from pathlib import Path

import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine.pytorch.cpp_extensions as tex
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

if torch.cuda.device_count() < 2:
    pytest.skip("Comm+GEMM overlap requires at least 2 GPUs.")

fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()

RNG_SEED: int = 1234
SEQ_LENGTH: int = 512
BATCH_SIZE: int = 2
NUM_HEADS: int = 12
HEAD_DIM: int = 64
TE_LAYERS = [
    te.Linear,
    te.LayerNormLinear,
    te.LayerNormMLP,
    te.MultiheadAttention,
    te.TransformerLayer,
]

TEST_ROOT = Path(__file__).parent.resolve()
NUM_PROCS: int = min(torch.cuda.device_count(), 4)
LAUNCH_CMD = ["torchrun", f"--nproc_per_node={NUM_PROCS}"]
if tex.ubuf_built_with_mpi():
    LAUNCH_CMD = ["mpirun", "-np", str(NUM_PROCS), "--oversubscribe", "--quiet", "python"]

# Fall back on CUDA IPC if the platform does not support CUDA multicast
if not tex.device_supports_multicast():
    os.environ["UB_SKIPMC"] = "1"

# Force GPU kernels to launch in the order they're executed by the host CPU
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"


def _run_gemm_with_overlap(comm_type, bulk, p2p, atomic, fp8_in, fp8_out, aggregate):
    test_path = TEST_ROOT / "run_gemm_with_overlap.py"
    test_cmd = LAUNCH_CMD + [
        str(test_path),
        "--check-numerics",
        f"--seed={RNG_SEED}",
        f"--seq-length={SEQ_LENGTH}",
        f"--batch-size={BATCH_SIZE}",
        f"--num-heads={NUM_HEADS}",
        f"--head-dim={HEAD_DIM}",
        f"--comm-type={comm_type}",
    ]

    if bulk:
        test_cmd.append("--bulk-overlap")
    else:
        if fp8_in:
            if not fp8_available:
                pytest.skip(reason_for_no_fp8)
            test_cmd.append("--fp8")
            if fp8_out:
                test_cmd.append("--fp8-output")
        if p2p:
            test_cmd.append("--p2p")
        if aggregate:
            test_cmd.append("--aggregate")
        if atomic:
            if torch.cuda.get_device_properties(0).major < 9:
                pytest.skip("Device compute capability 9.0 or higher required for Atomic GEMM.")
            test_cmd.append("--atomic")

    result = subprocess.run(test_cmd, env=os.environ, capture_output=True, check=False)
    if (
        result.returncode != 0
        or "NUMERICAL CHECK FAILED" in result.stderr.decode()
        or "NUMERICAL CHECK PASSED" not in result.stdout.decode()
    ):
        raise AssertionError(result.stderr.decode())


def _run_layer_with_overlap(layer_type, fp8, fp8_init):
    test_path = TEST_ROOT / "run_layer_with_overlap.py"
    test_cmd = LAUNCH_CMD + [
        str(test_path),
        f"--seed={RNG_SEED}",
        f"--seq-length={SEQ_LENGTH}",
        f"--batch-size={BATCH_SIZE}",
        f"--num-heads={NUM_HEADS}",
        f"--head-dim={HEAD_DIM}",
        f"--layer-type={layer_type}",
    ]

    if fp8:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        test_cmd.append("--fp8")
        if fp8_init:
            test_cmd.append("--fp8-init")

    os.environ["PYTORCH_JIT"] = "0"
    os.environ["NVTE_TORCH_COMPILE"] = "0"
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"

    result = subprocess.run(test_cmd, env=os.environ, capture_output=True, check=False)

    os.unsetenv("PYTORCH_JIT")
    os.unsetenv("NVTE_TORCH_COMPILE")
    os.unsetenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO")

    if (
        result.returncode != 0
        or "NUMERICAL CHECK FAILED" in result.stderr.decode()
        or "NUMERICAL CHECK PASSED" not in result.stdout.decode()
    ):
        raise AssertionError(result.stderr.decode())


@pytest.mark.parametrize(
    "fp8,aggregate",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
    ids=[
        " BF16 IN - RING-EXCHANGE ",
        " BF16 IN - RING-EXCHANGE - 2x AGGREGATED ",
        " FP8  IN - RING-EXCHANGE ",
        " FP8  IN - RING-EXCHANGE - 2x AGGREGATED ",
    ],
)
def test_split_all_gather_overlaps(fp8, aggregate):
    """
    Test (split GEMM -> all-gather) overlaps with direct calls to te.cpp_extensions.gemm or
    te.cpp_extensions.fp8_gemm.
    """
    _run_gemm_with_overlap("AG", False, True, False, fp8, False, aggregate)


@pytest.mark.parametrize(
    "fp8_in,fp8_out,p2p",
    [
        (False, False, False),
        (False, False, True),
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ],
    ids=[
        " BF16 IN - BF16 OUT - PIPELINE ",
        " BF16 IN - BF16 OUT - RING-EXCHANGE ",
        " FP8  IN - BF16 OUT - PIPELINE ",
        " FP8  IN - BF16 OUT - RING-EXCHANGE ",
        " FP8  IN - FP8  OUT - PIPELINE ",
        " FP8  IN - FP8  OUT - RING-EXCHANGE ",
    ],
)
def test_split_reduce_scatter_overlaps(fp8_in, fp8_out, p2p):
    """
    Test (reduce-scatter -> split GEMM) overlaps with direct calls to te.cpp_extensions.gemm or
    te.cpp_extensions.fp8_gemm.
    """
    _run_gemm_with_overlap("RS", False, p2p, False, fp8_in, fp8_out, False)


@pytest.mark.parametrize(
    "ag_type,rs_type,p2p,fp8_out",
    [
        (0, 0, False, False),
        (0, 1, False, False),
        (0, 1, False, True),
        (0, 2, False, False),
        (0, 2, False, True),
        (0, 0, True, False),
        (0, 0, True, True),
        (1, 0, True, False),
        (1, 0, True, True),
    ],
    ids=[
        " NON-ATOMIC AG   - NON-ATOMIC RS   - PIPELINE      - BF16 OUT ",
        " NON-ATOMIC AG   - ATOMIC RS       - PIPELINE      - BF16 OUT ",
        " NON-ATOMIC AG   - ATOMIC RS       - PIPELINE      - FP8  OUT ",
        " NON-ATOMIC AG   - MULTI-ATOMIC RS - PIPELINE      - BF16 OUT ",
        " NON-ATOMIC AG   - MULTI-ATOMIC RS - PIPELINE      - FP8  OUT ",
        " NON-ATOMIC AG   - NON-ATOMIC RS   - RING-EXCHANGE - BF16 OUT ",
        " NON-ATOMIC AG   - NON-ATOMIC RS   - RING-EXCHANGE - FP8  OUT ",
        " MULTI-ATOMIC AG - NON-ATOMIC RS   - RING-EXCHANGE - BF16 OUT ",
        " MULTI-ATOMIC AG - NON-ATOMIC RS   - RING-EXCHANGE - FP8  OUT ",
    ],
)
def test_atomic_gemm_overlaps(ag_type, rs_type, p2p, fp8_out):
    """
    Test paired (all-gather -> atomic GEMM) and (atomic GEMM -> reduce-scatter) overlaps with
    direct calls to te.cpp_extensions.gemm or te.cpp_extensions.fp8_gemm.
    """
    os.environ["NVTE_AG_P2P_MULTI_ATOMIC"] = str(ag_type)
    os.environ["NVTE_RS_STRIDED_ATOMIC"] = str(rs_type)
    _run_gemm_with_overlap("AG", False, p2p, True, True, fp8_out, False)


@pytest.mark.parametrize(
    "comm_type,fp8",
    [
        ("AG", False),
        ("RS", False),
        ("RS", True),
    ],
    ids=[" ALL-GATHER     - BF16 ", " REDUCE-SCATTER - BF16 ", " REDUCE-SCATTER - FP8 "],
)
def test_bulk_overlaps(comm_type, fp8):
    """
    Test bulk overlaps with direct calls to te.cpp_extensions.gemm or te.cpp_extensions.fp8_gemm.
    """
    _run_gemm_with_overlap(comm_type, True, False, False, fp8, False, False)


@pytest.mark.parametrize(
    "layer_type",
    [layer.__name__ for layer in TE_LAYERS],
    ids=[(" " + layer.__name__ + " ") for layer in TE_LAYERS],
)
@pytest.mark.parametrize(
    "fp8,fp8_init",
    [
        (False, False),
        (True, False),
        (True, True),
    ],
    ids=[
        " BF16 GEMM - BF16 PARAMS ",
        " FP8  GEMM - BF16 PARAMS ",
        " FP8  GEMM - FP8  PARAMS ",
    ],
)
def test_layers_with_overlap(layer_type, fp8, fp8_init):
    """
    Test Transformer Engine layers with comm+GEMM overlap.
    """
    _run_layer_with_overlap(layer_type, fp8, fp8_init)

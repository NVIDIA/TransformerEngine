# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

RNG_SEED: int = 42
SEQ_LENGTH: int = 1024
BATCH_SIZE: int = 2
NUM_HEADS: int = 16
HEAD_DIM: int = 48
TE_LAYERS = [
    te.Linear,
    te.LayerNormLinear,
    te.LayerNormMLP,
    te.MultiheadAttention,
    te.TransformerLayer,
]
MAX_LAYER_NAME_LENGTH = max([len(layer.__name__) for layer in TE_LAYERS])

TEST_ROOT = Path(__file__).parent.resolve()
NUM_PROCS: int = torch.cuda.device_count()
LAUNCH_CMD = ["torchrun", f"--nproc_per_node={NUM_PROCS}"]
if tex.ubuf_built_with_mpi():
    LAUNCH_CMD = ["mpirun", "-np", str(NUM_PROCS), "--oversubscribe", "--quiet", "python"]

# Fall back on CUDA IPC if the platform does not support CUDA multicast
if not tex.device_supports_multicast():
    os.environ["UB_SKIPMC"] = "1"

# Force GPU kernels to launch in the order they're executed by the host CPU
os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

# Clear torch.dynamo caches
torch._dynamo.reset()


def _run_gemm_with_overlap(comm_type, bulk, p2p, atomic, fp8):
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
        if fp8:
            if not fp8_available:
                pytest.skip(reason_for_no_fp8)
            test_cmd.append("--fp8")
        if p2p:
            test_cmd.append("--p2p")
        if atomic:
            if torch.cuda.get_device_properties(0).major != 9:
                pytest.skip("Atomic GEMM is requires device compute capability 9.x (Hopper).")
            test_cmd.append("--atomic")

    result = subprocess.run(test_cmd, env=os.environ, capture_output=True, check=False)
    if (
        result.returncode != 0
        or "NUMERICAL CHECK FAILED" in result.stderr.decode()
        or "NUMERICAL CHECK PASSED" not in result.stdout.decode()
    ):
        raise AssertionError(result.stderr.decode())


def _run_layer_with_overlap(layer_type, linear_parallel_mode, overlap_rs_dgrad, fp8, fp8_recipe):
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
    if layer_type in [te.Linear.__name__, te.LayerNormLinear.__name__]:
        test_cmd.append(f"--linear-parallel-mode={linear_parallel_mode}")

    if overlap_rs_dgrad:
        test_cmd.append("--overlap-rs-dgrad")

    if fp8:
        if not fp8_available:
            pytest.skip(reason_for_no_fp8)
        test_cmd.append("--fp8")
        test_cmd.append(f"--fp8-recipe={fp8_recipe}")

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
    "fp8",
    (False, True),
    ids=[" BF16 - RING-EXCHANGE ", " FP8  - RING-EXCHANGE "],
)
def test_split_all_gather_overlaps(fp8):
    """
    Test (split GEMM -> all-gather) overlaps with direct calls to te.cpp_extensions.gemm or
    te.cpp_extensions.fp8_gemm.
    """
    _run_gemm_with_overlap("AG", False, True, False, fp8)


@pytest.mark.parametrize(
    "fp8,p2p",
    [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ],
    ids=[
        " BF16 - PIPELINE ",
        " BF16 - RING-EXCHANGE ",
        " FP8  - PIPELINE ",
        " FP8  - RING-EXCHANGE ",
    ],
)
def test_split_reduce_scatter_overlaps(fp8, p2p):
    """
    Test (reduce-scatter -> split GEMM) overlaps with direct calls to te.cpp_extensions.gemm or
    te.cpp_extensions.fp8_gemm.
    """
    _run_gemm_with_overlap("RS", False, p2p, False, fp8)


@pytest.mark.parametrize(
    "comm_type, fp8, connections",
    [
        ("AG", False, 1),
        ("RS", False, 1),
        ("RS", True, 1),
        ("AG", False, 8),
        ("RS", False, 8),
        ("RS", True, 8),
    ],
    ids=[
        "ALL-GATHER     - BF16 - 1 connections",
        "REDUCE-SCATTER - BF16 - 1 connections",
        "REDUCE-SCATTER - FP8  - 1 connections",
        "ALL-GATHER     - BF16 - 8 connections",
        "REDUCE-SCATTER - BF16 - 8 connections",
        "REDUCE-SCATTER - FP8  - 8 connections",
    ],
)
def test_bulk_overlaps(comm_type, fp8, connections):
    """
    Test bulk overlaps with direct calls to te.cpp_extensions.gemm or te.cpp_extensions.fp8_gemm.
    """
    if connections == 8:
        if torch.cuda.get_device_properties(0).major != 9:
            pytest.skip(
                "CUDA_DEVICE_MAX_CONNECTIONS=8 test only applies to devices with compute capability"
                " 9.0 (HOPPER ARCH)."
            )
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "8"
        _run_gemm_with_overlap(comm_type, True, False, False, fp8)
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    else:
        _run_gemm_with_overlap(comm_type, True, False, False, fp8)


@pytest.mark.parametrize(
    "fp8",
    (False,),
    ids=[
        " BF16 ",
    ],
)
@pytest.mark.parametrize(
    "layer_type,linear_parallel_mode,overlap_rs_dgrad",
    [
        (te.Linear.__name__, "row", False),
        (te.Linear.__name__, "column", False),
        (te.Linear.__name__, "column", True),
        (te.LayerNormLinear.__name__, "row", False),
        (te.LayerNormLinear.__name__, "column", False),
        (te.LayerNormLinear.__name__, "column", True),
    ]
    + list(
        zip(
            [layer.__name__ for layer in TE_LAYERS[2:] for _ in range(2)],
            [None] * len(TE_LAYERS[2:]) * 2,
            [False, True] * len(TE_LAYERS[2:]),
        )
    ),
    ids=[
        f" {te.Linear.__name__} - ROW-PARALLEL ",
        f" {te.Linear.__name__} - COL-PARALLEL - BULK DGRAD/WGRAD ",
        f" {te.Linear.__name__} - COL-PARLALEL - DGRAD+RS ",
        f" {te.LayerNormLinear.__name__} - ROW-PARALLEL ",
        f" {te.LayerNormLinear.__name__} - COL-PARALLEL - BULK DGRAD/WGRAD ",
        f" {te.LayerNormLinear.__name__} - COL-PARALLEL - DGRAD+RS ",
    ]
    + [
        " " + " - ".join(test_name_parts) + " "
        for test_name_parts in zip(
            [layer.__name__ for layer in TE_LAYERS[2:] for _ in range(2)],
            ["BULK DGRAD/WGRAD", "DGRAD+RS"] * len(TE_LAYERS[2:]),
        )
    ],
)
def test_layers_with_overlap_bf16(layer_type, linear_parallel_mode, overlap_rs_dgrad, fp8):
    """
    Test Transformer Engine layers with comm+GEMM overlap.
    """
    _run_layer_with_overlap(layer_type, linear_parallel_mode, overlap_rs_dgrad, fp8, None)


@pytest.mark.parametrize(
    "fp8_recipe", ["delayed", "tensorwise"], ids=[" DELAYED SCALING ", " CURRENT SCALING "]
)
@pytest.mark.parametrize(
    "fp8",
    (True,),
    ids=[
        " FP8  ",
    ],
)
@pytest.mark.parametrize(
    "layer_type,linear_parallel_mode,overlap_rs_dgrad",
    [
        (te.Linear.__name__, "row", False),
        (te.Linear.__name__, "column", False),
        (te.Linear.__name__, "column", True),
        (te.LayerNormLinear.__name__, "row", False),
        (te.LayerNormLinear.__name__, "column", False),
        (te.LayerNormLinear.__name__, "column", True),
    ]
    + list(
        zip(
            [layer.__name__ for layer in TE_LAYERS[2:] for _ in range(2)],
            [None] * len(TE_LAYERS[2:]) * 2,
            [False, True] * len(TE_LAYERS[2:]),
        )
    ),
    ids=[
        f" {te.Linear.__name__} - ROW-PARALLEL ",
        f" {te.Linear.__name__} - COL-PARALLEL - BULK DGRAD/WGRAD ",
        f" {te.Linear.__name__} - COL-PARLALEL - DGRAD+RS ",
        f" {te.LayerNormLinear.__name__} - ROW-PARALLEL ",
        f" {te.LayerNormLinear.__name__} - COL-PARALLEL - BULK DGRAD/WGRAD ",
        f" {te.LayerNormLinear.__name__} - COL-PARALLEL - DGRAD+RS ",
    ]
    + [
        " " + " - ".join(test_name_parts) + " "
        for test_name_parts in zip(
            [layer.__name__ for layer in TE_LAYERS[2:] for _ in range(2)],
            ["BULK DGRAD/WGRAD", "DGRAD+RS"] * len(TE_LAYERS[2:]),
        )
    ],
)
def test_layers_with_overlap_fp8(
    layer_type, linear_parallel_mode, overlap_rs_dgrad, fp8, fp8_recipe
):
    """
    Test Transformer Engine layers with comm+GEMM overlap.
    """
    _run_layer_with_overlap(layer_type, linear_parallel_mode, overlap_rs_dgrad, fp8, fp8_recipe)

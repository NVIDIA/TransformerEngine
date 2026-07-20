# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pytest launcher for hybrid TP/SP distributed tests."""

import os
import subprocess
from pathlib import Path

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.utils import is_non_tn_fp8_gemm_supported

if torch.cuda.device_count() < 2:
    pytest.skip(
        "Distributed TP/SP tests need at least 2 GPUs.",
        allow_module_level=True,
    )

fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)

TEST_ROOT = Path(__file__).parent.resolve()
NUM_PROCS = min(2, torch.cuda.device_count())
LAUNCH_CMD = ["torchrun", f"--nproc_per_node={NUM_PROCS}"]

xfail_hopper_columnwise_per_tensor_fp8 = pytest.mark.xfail(
    condition=not is_non_tn_fp8_gemm_supported(),
    strict=True,
    reason=(
        "Hopper does not yet support columnwise-only per-tensor FP8 quantization; "
        "tracked by NVIDIA/TransformerEngine#3158"
    ),
)


def _run_test(quantization: str, test: str = "all"):
    script = TEST_ROOT / "run_hybrid_tp_sp.py"
    cmd = LAUNCH_CMD + [str(script), "--quantization", quantization, "--test", test]
    result = subprocess.run(cmd, env=os.environ, check=False)
    assert result.returncode == 0, (
        f"run_hybrid_tp_sp.py (quantization={quantization}, test={test})"
        f" exited with code {result.returncode}"
    )


# ──────────────────────────────────────────────────────────────────────
# Hybrid FP8 current scaling
# ──────────────────────────────────────────────────────────────────────
# Exercises TP amax reduction and SP gather paths.


@pytest.mark.skipif(not fp8_available, reason=f"FP8: {reason_for_no_fp8}")
@xfail_hopper_columnwise_per_tensor_fp8
def test_hybrid_fp8_linear():
    """Linear TP/SP coverage for hybrid FP8 current scaling."""
    _run_test("hybrid_fp8", "linear")


@pytest.mark.skipif(not fp8_available, reason=f"FP8: {reason_for_no_fp8}")
@xfail_hopper_columnwise_per_tensor_fp8
def test_hybrid_fp8_linear_vs_vanilla():
    """Same-topology Linear parity against Float8CurrentScaling."""
    _run_test("hybrid_fp8", "linear_vs_vanilla")


@pytest.mark.skipif(not fp8_available, reason=f"FP8: {reason_for_no_fp8}")
@xfail_hopper_columnwise_per_tensor_fp8
def test_hybrid_fp8_layernorm_linear_vs_vanilla():
    """Same-topology LayerNormLinear parity against vanilla FP8."""
    _run_test("hybrid_fp8", "layernorm_linear_vs_vanilla")


@pytest.mark.skipif(not fp8_available, reason=f"FP8: {reason_for_no_fp8}")
@xfail_hopper_columnwise_per_tensor_fp8
def test_hybrid_fp8_layernorm_mlp_vs_vanilla():
    """Same-topology LayerNormMLP parity against vanilla FP8."""
    _run_test("hybrid_fp8", "layernorm_mlp_vs_vanilla")


@pytest.mark.skipif(not fp8_available, reason=f"FP8: {reason_for_no_fp8}")
@xfail_hopper_columnwise_per_tensor_fp8
def test_hybrid_fp8_layernorm_linear():
    """LayerNormLinear TP/SP coverage for hybrid FP8."""
    _run_test("hybrid_fp8", "layernorm_linear")


@pytest.mark.skipif(not fp8_available, reason=f"FP8: {reason_for_no_fp8}")
@xfail_hopper_columnwise_per_tensor_fp8
def test_hybrid_fp8_layernorm_mlp():
    """LayerNormMLP TP/SP coverage for hybrid FP8."""
    _run_test("hybrid_fp8", "layernorm_mlp")


@pytest.mark.skipif(not fp8_available, reason=f"FP8: {reason_for_no_fp8}")
@xfail_hopper_columnwise_per_tensor_fp8
def test_hybrid_fp8_transformer_layer():
    """TransformerLayer TP/SP coverage for hybrid FP8."""
    _run_test("hybrid_fp8", "transformer_layer")


@pytest.mark.skipif(not fp8_available, reason=f"FP8: {reason_for_no_fp8}")
def test_hybrid_fp8_identity_linear():
    """Linear TP/SP coverage for FP8 forward plus Identity backward."""
    _run_test("hybrid_fp8_identity", "linear")


@pytest.mark.skipif(not fp8_available, reason=f"FP8: {reason_for_no_fp8}")
def test_identity_all_modules():
    """All-Identity TP/SP end-to-end coverage for every supported TE module."""
    _run_test("identity", "all")


# ──────────────────────────────────────────────────────────────────────
# Hybrid MXFP8
# ──────────────────────────────────────────────────────────────────────
# Covers per-block scale layout through TP shards.


@pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
def test_hybrid_mxfp8_linear():
    _run_test("hybrid_mxfp8", "linear")


@pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
def test_hybrid_mxfp8_linear_vs_vanilla():
    _run_test("hybrid_mxfp8", "linear_vs_vanilla")


@pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
def test_hybrid_mxfp8_layernorm_linear_vs_vanilla():
    _run_test("hybrid_mxfp8", "layernorm_linear_vs_vanilla")


@pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
def test_hybrid_mxfp8_layernorm_mlp_vs_vanilla():
    _run_test("hybrid_mxfp8", "layernorm_mlp_vs_vanilla")


@pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
def test_hybrid_mxfp8_layernorm_linear():
    _run_test("hybrid_mxfp8", "layernorm_linear")


@pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
def test_hybrid_mxfp8_layernorm_mlp():
    _run_test("hybrid_mxfp8", "layernorm_mlp")


@pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
def test_hybrid_mxfp8_transformer_layer():
    _run_test("hybrid_mxfp8", "transformer_layer")


@pytest.mark.skipif(not mxfp8_available, reason=f"MXFP8: {reason_for_no_mxfp8}")
def test_hybrid_mxfp8_identity_linear():
    """Linear TP/SP coverage for MXFP8 forward plus Identity backward."""
    _run_test("hybrid_mxfp8_identity", "linear")


# ──────────────────────────────────────────────────────────────────────
# Hybrid NVFP4
# ──────────────────────────────────────────────────────────────────────
# Same-format NVFP4 coverage with base role-wise settings.


@pytest.mark.skipif(not nvfp4_available, reason=f"NVFP4: {reason_for_no_nvfp4}")
def test_hybrid_nvfp4_linear():
    _run_test("hybrid_nvfp4", "linear")


@pytest.mark.skipif(not nvfp4_available, reason=f"NVFP4: {reason_for_no_nvfp4}")
def test_hybrid_nvfp4_linear_vs_vanilla():
    _run_test("hybrid_nvfp4", "linear_vs_vanilla")


@pytest.mark.skipif(not nvfp4_available, reason=f"NVFP4: {reason_for_no_nvfp4}")
def test_hybrid_nvfp4_layernorm_linear():
    _run_test("hybrid_nvfp4", "layernorm_linear")


@pytest.mark.skipif(not nvfp4_available, reason=f"NVFP4: {reason_for_no_nvfp4}")
def test_hybrid_nvfp4_layernorm_mlp():
    _run_test("hybrid_nvfp4", "layernorm_mlp")


@pytest.mark.skipif(not nvfp4_available, reason=f"NVFP4: {reason_for_no_nvfp4}")
def test_hybrid_nvfp4_transformer_layer():
    _run_test("hybrid_nvfp4", "transformer_layer")


# ──────────────────────────────────────────────────────────────────────
# Cross-format hybrid: MXFP8 rowwise + NVFP4 columnwise
# ──────────────────────────────────────────────────────────────────────
# No single vanilla recipe exists for bitwise comparison.

_cross_format_available = mxfp8_available and nvfp4_available
_reason_for_no_cross_format = reason_for_no_mxfp8 if not mxfp8_available else reason_for_no_nvfp4


@pytest.mark.skipif(
    not _cross_format_available, reason=f"MXFP8+NVFP4: {_reason_for_no_cross_format}"
)
def test_hybrid_mxfp8_nvfp4_linear():
    _run_test("hybrid_mxfp8_nvfp4", "linear")


@pytest.mark.skipif(
    not _cross_format_available, reason=f"MXFP8+NVFP4: {_reason_for_no_cross_format}"
)
def test_hybrid_mxfp8_nvfp4_layernorm_linear():
    _run_test("hybrid_mxfp8_nvfp4", "layernorm_linear")


@pytest.mark.skipif(
    not _cross_format_available, reason=f"MXFP8+NVFP4: {_reason_for_no_cross_format}"
)
def test_hybrid_mxfp8_nvfp4_layernorm_mlp():
    _run_test("hybrid_mxfp8_nvfp4", "layernorm_mlp")


@pytest.mark.skipif(
    not _cross_format_available, reason=f"MXFP8+NVFP4: {_reason_for_no_cross_format}"
)
def test_hybrid_mxfp8_nvfp4_transformer_layer():
    _run_test("hybrid_mxfp8_nvfp4", "transformer_layer")

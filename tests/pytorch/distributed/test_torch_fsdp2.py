# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import subprocess
from pathlib import Path

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import fp8

NUM_PROCS: int = torch.cuda.device_count()


def check_nvfp4_support():
    supported, reason = fp8.check_nvfp4_support()
    if supported and torch.cuda.get_device_capability()[0] == 12:
        return (
            False,
            (
                "NVFP4BlockScaling is failing on SM120 with "
                "hadamard_transform/hadamard_transform_cast_fusion.cu:672 in function "
                "rht_gemm_ntt_w_sfc: CUDA Error: invalid argument"
            ),
        )

    return supported, reason


# Each entry: (recipe_class_name, check_fn)
_FP8_RECIPE_CONFIGS = [
    ("DelayedScaling", fp8.check_fp8_support),
    ("Float8CurrentScaling", fp8.check_fp8_support),
    ("Float8BlockScaling", fp8.check_fp8_block_scaling_support),
    ("MXFP8BlockScaling", fp8.check_mxfp8_support),
    ("NVFP4BlockScaling", check_nvfp4_support),
]


def _parametrize_fp8_recipes():
    """Generate pytest.param objects with skip marks for unsupported FP8 recipes."""
    params = []
    for name, check_fn in _FP8_RECIPE_CONFIGS:
        supported, reason = check_fn()
        params.append(
            pytest.param(
                name,
                id=name,
                marks=pytest.mark.skipif(not supported, reason=reason),
            )
        )
    return params


@pytest.fixture(params=_parametrize_fp8_recipes())
def fp_recipe(request):
    """Parametrized fixture providing FP8 recipe Hydra overrides for each supported TE recipe."""
    return request.param


def _run_test(fp_init, sharding_dims, recipe, layer_type):
    test_path = Path(__file__).parent.resolve() / "run_fsdp2_model.py"
    test_cmd = ["torchrun", f"--nproc_per_node={NUM_PROCS}", str(test_path)]

    if fp_init:
        test_cmd += ["--fp8-init"]

    if len(sharding_dims) == 1:
        test_cmd += ["--sharding-dims", str(sharding_dims[0])]
    elif len(sharding_dims) == 2:
        test_cmd += ["--sharding-dims", str(sharding_dims[0]), str(sharding_dims[1])]
    else:
        assert False
    test_cmd += ["--recipe", recipe]
    test_cmd += ["--layer-type", layer_type]

    subprocess.run(test_cmd, env=os.environ, check=True)


@pytest.mark.skipif(NUM_PROCS % 2 != 0, reason="Requires even number of GPUs")
@pytest.mark.skipif(not te.torch_version() >= (2, 4, 0), reason="Requires PyTorch 2.4.0+")
@pytest.mark.parametrize("sharding_dims", ([NUM_PROCS], [2, NUM_PROCS // 2]))
@pytest.mark.parametrize("fp8_init", (False, True))
@pytest.mark.parametrize("layer_type", ("LayerNormLinear", "TransformerLayer"))
def test_distributed(fp8_init, sharding_dims, fp_recipe, layer_type):

    if fp_recipe in ("Float8BlockScaling", "NVFP4BlockScaling") and fp8_init:
        pytest.xfail(f"{fp_recipe} + fp8_init: test_fp8_fsdp2_allgather is currently failing.")

    _run_test(fp8_init, sharding_dims, fp_recipe, layer_type)


## ── FusedAdam + FSDP2 tests ─────────────────────────────────────────


def _run_fused_adam_test(test_name, recipe="delayed_scaling"):
    """Launch an FSDP2 + FusedAdam test via torchrun."""
    test_path = Path(__file__).parent.resolve() / "run_fsdp2_fused_adam.py"
    nproc = min(NUM_PROCS, 2)  # These tests only need 2 GPUs
    test_cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        str(test_path),
        "--test",
        test_name,
        "--recipe",
        recipe,
    ]

    subprocess.run(test_cmd, env=os.environ, check=True)


@pytest.mark.skipif(NUM_PROCS < 2, reason="Requires 2+ GPUs")
def test_fsdp2_fused_adam_fp8_master_weights(fp_recipe):
    """FusedAdam(master_weights=True) + FSDP2 + quantized_model_init (meta device init)."""
    if fp_recipe in ("NVFP4BlockScaling",):
        pytest.xfail(
            f"{fp_recipe}: quantized_model_init and FSDP2 is not currently supported, since the "
            "block tensor is dequantized before we flatten it for FSDP2."
        )
    _run_fused_adam_test("fused_adam_fp8_master_weights", fp_recipe)


@pytest.mark.skipif(NUM_PROCS < 2, reason="Requires 2+ GPUs")
def test_fsdp2_fused_adam_fp8_master_weights_no_meta(fp_recipe):
    """FusedAdam(master_weights=True) + FSDP2 + quantized_model_init (CUDA init, no meta device).

    Block-scaling QuantizedTensors (MXFP8, Float8Blockwise, NVFP4) are wrapper
    subclasses with data_ptr() == 0.  Without meta-device init, FSDP2's
    reset_sharded_param() crashes with 'invalid python storage'.
    Per-tensor FP8 (DelayedScaling, Float8CurrentScaling) works because
    Float8Tensor's storage is accessible.
    """
    if fp_recipe in ("MXFP8BlockScaling", "Float8BlockScaling", "NVFP4BlockScaling"):
        pytest.xfail(
            f"{fp_recipe}: FSDP2 without meta-device init crashes on block-scaling "
            "QuantizedTensor wrapper subclasses (data_ptr() == 0). "
            "Use device='meta' + reset_parameters() after sharding."
        )
    _run_fused_adam_test("fused_adam_fp8_master_weights_no_meta", fp_recipe)


@pytest.mark.skipif(NUM_PROCS < 2, reason="Requires 2+ GPUs")
def test_fsdp2_fused_adam_bf16(fp_recipe):
    """FusedAdam(master_weights=True) + FSDP2 + bf16 params (no FP8)."""
    _run_fused_adam_test("fused_adam_bf16", fp_recipe)


@pytest.mark.skipif(NUM_PROCS < 2, reason="Requires 2+ GPUs")
def test_fsdp2_fused_adam_fp8_no_master(fp_recipe):
    """FusedAdam(master_weights=False) + FSDP2 + FP8 params."""
    if fp_recipe in ("MXFP8BlockScaling", "Float8BlockScaling", "NVFP4BlockScaling"):
        pytest.xfail(
            f"{fp_recipe}: FusedAdam without master_weights does not support "
            "block-scaling quantized tensors. Use master_weights=True."
        )
    _run_fused_adam_test("fused_adam_fp8_no_master", fp_recipe)


@pytest.mark.skipif(NUM_PROCS < 2, reason="Requires 2+ GPUs")
def test_fsdp2_fused_adam_bf16_store_param_remainders(fp_recipe):
    """FusedAdam(master_weights=True, store_param_remainders=True) + FSDP2 + bf16."""
    _run_fused_adam_test("fused_adam_bf16_store_param_remainders", fp_recipe)


@pytest.mark.skipif(NUM_PROCS < 2, reason="Requires 2+ GPUs")
def test_fsdp2_dcp_output_parity(fp_recipe):
    """DCP save/load round-trip into a fresh model produces identical outputs."""
    if fp_recipe == "MXFP8BlockScaling":
        pytest.xfail(
            "MXFP8BlockScaling: FusedAdam CUDA kernel does not support "
            "MXFP8 quantized tensors, causing illegal memory access"
        )

    if fp_recipe == "NVFP4BlockScaling":
        pytest.xfail(
            "NVFP4BlockScaling: DCP load_state_dict triggers reset_sharded_param() "
            "which calls data_ptr() on NVFP4Tensor wrapper subclass with invalid storage"
        )

    if fp_recipe == "Float8BlockScaling" and torch.cuda.get_device_capability()[0] == 12:
        pytest.xfail(
            "Float8BlockScaling is failing on SM120 with RuntimeError: "
            "transformer_engine/common/transpose/quantize_transpose_vector_blockwise.cu:534 "
            "in function quantize_transpose_vector_blockwise: Assertion failed: pow2_scale. On "
            "Blackwell and newer, the FP8 block scaling recipe is emulated with MXFP8, which "
            "requires using power of two scaling factors."
        )

    _run_fused_adam_test("dcp_output_parity", fp_recipe)


@pytest.mark.skipif(NUM_PROCS < 2, reason="Requires 2+ GPUs")
def test_fsdp2_dcp_output_parity_async(fp_recipe):
    """DCP save/load round-trip into a fresh model produces identical outputs."""
    if fp_recipe == "MXFP8BlockScaling":
        pytest.xfail(
            "MXFP8BlockScaling: FusedAdam CUDA kernel does not support "
            "MXFP8 quantized tensors, causing illegal memory access: "
            "/transformer_engine/common/multi_tensor/multi_tensor_apply.cuh:92 in function "
            "multi_tensor_apply: CUDA Error: an illegal memory access was encountered"
        )

    if fp_recipe == "NVFP4BlockScaling":
        pytest.xfail(
            "NVFP4BlockScaling: DCP load_state_dict triggers reset_sharded_param() "
            "which calls data_ptr() on NVFP4Tensor wrapper subclass with invalid storage"
        )

    if fp_recipe == "Float8BlockScaling":
        pytest.xfail(
            "Float8BlockScaling: async DCP save/load round-trip produces different model "
            "outputs — quantization metadata (scales) is not correctly persisted through "
            "async distributed checkpointing. On SM120, additionally fails with pow2_scale "
            "assertion in quantize_transpose_vector_blockwise."
        )

    _run_fused_adam_test("dcp_output_parity_async", fp_recipe)


@pytest.mark.skipif(NUM_PROCS < 2, reason="Requires 2+ GPUs")
def test_fsdp2_safetensors_fp32_export(fp_recipe):
    """Export FP32 model from optimizer master weights to safetensors."""
    if fp_recipe == "MXFP8BlockScaling":
        pytest.xfail(
            "MXFP8BlockScaling: FusedAdam CUDA kernel does not support "
            "MXFP8 quantized tensors, causing illegal memory access"
        )
    _run_fused_adam_test("safetensors_fp32_export", fp_recipe)


@pytest.mark.skipif(NUM_PROCS < 2, reason="Requires 2+ GPUs")
@pytest.mark.xfail(
    reason=(
        "fuse_wgrad_accumulation is incompatible with vanilla FSDP2: "
        "autograd Function.apply unwraps DTensors to local tensors, so "
        "main_grad (set on the DTensor) is inaccessible during backward. "
        "Additionally, the fused wgrad GEMM bypasses FSDP2's reduce-scatter."
    ),
    raises=subprocess.CalledProcessError,
    strict=True,
)
def test_fsdp2_fuse_wgrad_accumulation(fp_recipe):
    """fuse_wgrad_accumulation=True + FSDP2 -- expected to fail."""
    _run_fused_adam_test("fuse_wgrad_accumulation", fp_recipe)


def test_dummy() -> None:
    """Dummy test

    pytest returns exit code 5 if all tests are skipped.

    """
    pass

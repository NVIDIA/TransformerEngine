# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Uniform fixed-token THD attention benchmarks and correctness tests.

Configs here are intended for benchmarking and stress testing on top of the
core correctness suite in test_attention_with_cp.py. Most configs use larger
batch sizes / sequence lengths than the core suite.

The bench/profile machinery (timing loop, cudaProfilerStart/Stop, seqlen pattern
arg) lives in run_attention_with_cp.py.
"""

import os
import sys
import pathlib
import logging
import copy
import json
import subprocess
import pytest
import torch
from transformer_engine.pytorch import (
    get_device_compute_capability,
    get_cudnn_version,
)

_current_file = pathlib.Path(__file__).resolve()
sys.path.append(str(_current_file.parent.parent))
from utils import ModelConfig, get_available_attention_backends, run_distributed

pytest_logging_level = logging.getLevelName(logging.root.level)


def get_bash_arguments(num_gpus_per_node, **kwargs):
    args = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=" + str(num_gpus_per_node),
    ]
    te_path = os.getenv("TE_PATH", "/opt/transformerengine")
    args.append(os.path.join(te_path, "tests/pytorch/attention/run_attention_with_cp.py"))
    for k, v in kwargs.items():
        args.append(f"{k}={v}")
    return args


# Uniform 512K-token configs (32 query heads, 8 GQA groups, d=128).
model_configs_fused_attn = {
    # Fixed 512K tokens with different sequence-count/length decompositions.
    "uniform_1x512k": ModelConfig(1, 524288, 32, 128, num_gqa_groups=8, attn_mask_type="causal"),
    "uniform_2x256k": ModelConfig(2, 262144, 32, 128, num_gqa_groups=8, attn_mask_type="causal"),
    "uniform_4x128k": ModelConfig(4, 131072, 32, 128, num_gqa_groups=8, attn_mask_type="causal"),
    "uniform_8x64k": ModelConfig(8, 65536, 32, 128, num_gqa_groups=8, attn_mask_type="causal"),
    "uniform_16x32k": ModelConfig(16, 32768, 32, 128, num_gqa_groups=8, attn_mask_type="causal"),
    "uniform_32x16k": ModelConfig(32, 16384, 32, 128, num_gqa_groups=8, attn_mask_type="causal"),
    "uniform_64x8k": ModelConfig(64, 8192, 32, 128, num_gqa_groups=8, attn_mask_type="causal"),
    "uniform_128x4k": ModelConfig(128, 4096, 32, 128, num_gqa_groups=8, attn_mask_type="causal"),
    "uniform_256x2k": ModelConfig(256, 2048, 32, 128, num_gqa_groups=8, attn_mask_type="causal"),
    # Fixed-token SWA probes use the same shapes with a 1024-token left window.
    "uniform_1x512k_swa1024": ModelConfig(
        1,
        524288,
        32,
        128,
        num_gqa_groups=8,
        attn_mask_type="causal",
        window_size=(1024, 0),
    ),
    "uniform_2x256k_swa1024": ModelConfig(
        2,
        262144,
        32,
        128,
        num_gqa_groups=8,
        attn_mask_type="causal",
        window_size=(1024, 0),
    ),
    "uniform_4x128k_swa1024": ModelConfig(
        4,
        131072,
        32,
        128,
        num_gqa_groups=8,
        attn_mask_type="causal",
        window_size=(1024, 0),
    ),
    "uniform_8x64k_swa1024": ModelConfig(
        8,
        65536,
        32,
        128,
        num_gqa_groups=8,
        attn_mask_type="causal",
        window_size=(1024, 0),
    ),
    "uniform_16x32k_swa1024": ModelConfig(
        16,
        32768,
        32,
        128,
        num_gqa_groups=8,
        attn_mask_type="causal",
        window_size=(1024, 0),
    ),
}


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 7), reason="cuDNN 8.9.7+ is required.")
@pytest.mark.skipif(get_device_compute_capability() < (9, 0), reason="THD requires sm90+.")
@pytest.mark.parametrize("model", model_configs_fused_attn.keys())
@pytest.mark.parametrize("qkv_format", ["thd"])
@pytest.mark.parametrize("cp_comm_type", ["p2p", "all_gather", "a2a"])
@pytest.mark.parametrize("kernel_backend", ["FusedAttention", "FlashAttention"])
@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_cp_benchmark_configs(model, qkv_format, cp_comm_type, kernel_backend, cp_size):
    """Check CP correctness and run a one-iteration benchmark."""
    if cp_size > torch.cuda.device_count():
        pytest.skip(f"Test requires {cp_size} GPUs")

    config = copy.deepcopy(model_configs_fused_attn[model])
    deterministic = os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1") == "0"
    config.context_parallel = True
    config.cp_comm_type = cp_comm_type

    has_swa = config.window_size != (-1, 0) and config.window_size != (-1, -1)
    if has_swa and cp_comm_type == "p2p":
        pytest.skip("p2p does not support sliding window")

    # THD uses padding mask types for backend availability check
    config.attn_mask_type = "padding_causal"

    available, flash_version, *_ = get_available_attention_backends(
        config,
        qkv_dtype=torch.bfloat16,
        qkv_layout="_".join([qkv_format] * 3),
        cp_size=cp_size,
        pad_between_seqs=kernel_backend == "FusedAttention",
        deterministic=deterministic,
    )
    backend_index = 1 if kernel_backend == "FusedAttention" else 0
    if not available[backend_index]:
        pytest.skip(f"{kernel_backend} not available for this config")
    if (
        kernel_backend == "FlashAttention"
        and cp_comm_type == "all_gather"
        and flash_version.major < 3
    ):
        pytest.skip("THD all-gather requires FlashAttention 3 or newer")

    process = run_distributed(
        get_bash_arguments(
            num_gpus_per_node=cp_size,
            dtype="bf16",
            model=model,
            qkv_format=qkv_format,
            kernel_backend=kernel_backend,
            cp_comm_type=cp_comm_type,
            deterministic=deterministic,
            thd_seqlen_pattern="max",
            benchmark=1,
            log_level=pytest_logging_level,
        ),
        stdout=subprocess.PIPE,
    )
    print(process.stdout, end="")
    results = [
        json.loads(line.removeprefix("CP_BENCH_RESULT "))
        for line in process.stdout.splitlines()
        if line.startswith("CP_BENCH_RESULT ")
    ]
    assert len(results) == 1
    result = results[0]
    assert result["model"] == model
    assert result["backend"] == kernel_backend
    assert result["communication_type"] == cp_comm_type
    assert result["cp_size"] == cp_size
    assert result["qkv_format"] == qkv_format
    assert result["dtype"] == "bf16"
    assert result["is_training"] is True
    assert result["thd_seqlen_pattern"] == "max"
    assert result["warmup_iterations"] == 10
    assert result["timed_iterations"] == 1
    assert len(result["iteration_rank_max_ms"]) == 1
    assert 0 < result["latency_ms"] < float("inf")

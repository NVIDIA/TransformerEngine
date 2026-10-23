# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark/profile configurations and cross-backend consistency test for CP attention.

Configs here are intended for benchmarking and stress testing on top of the
core correctness suite in test_attention_with_cp.py. Most configs use larger
batch sizes / sequence lengths than the core suite. Variable-length THD inputs
use the `thd_seqlen_pattern` attribute, plumbed through to run_attention_with_cp.py.

The bench/profile machinery (timing loop, cudaProfilerStart/Stop, save dir, seqlen
pattern arg) lives in run_attention_with_cp.py.
"""

import os
import sys
import pathlib
import logging
import copy
import tempfile
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


# Benchmark/stress configs (llama3_8b-like: 32 heads, 8 GQA, d=128).
model_configs_fused_attn = {
    # Llama3-8b-shaped, varying seqlen
    "bench_8k": ModelConfig(2, 8192, 32, 128, num_gqa_groups=8, attn_mask_type="causal"),
    "bench_16k": ModelConfig(1, 16384, 32, 128, num_gqa_groups=8, attn_mask_type="causal"),
    "bench_32k": ModelConfig(1, 32768, 32, 128, num_gqa_groups=8, attn_mask_type="causal"),
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
    "uniform_1x128k": ModelConfig(
        1,
        131072,
        32,
        128,
        num_gqa_groups=8,
        attn_mask_type="causal",
    ),
    "uniform_1x128k_swa1024": ModelConfig(
        1,
        131072,
        32,
        128,
        num_gqa_groups=8,
        attn_mask_type="causal",
        window_size=(1024, 0),
    ),
    # 2688-wide model with 32 query heads: 2688 / 32 = 84 channels per head.
    "uniform_1x128k_h32_d2688": ModelConfig(
        1,
        131072,
        32,
        84,
        num_gqa_groups=8,
        attn_mask_type="causal",
    ),
    "uniform_1x128k_h32_d2688_swa1024": ModelConfig(
        1,
        131072,
        32,
        84,
        num_gqa_groups=8,
        attn_mask_type="causal",
        window_size=(1024, 0),
    ),
    # THD stress: higher batch / longer seqlen than core suite
    "cp_thd_0": ModelConfig(8, 8192, 12, 128, attn_mask_type="causal"),  # MHA b=8
    "cp_thd_1": ModelConfig(8, 8192, 12, 128),  # MHA b=8 non-causal
    "cp_thd_2": ModelConfig(16, 4096, 12, 128, attn_mask_type="causal"),  # MHA b=16
    "cp_thd_3": ModelConfig(8, 8192, 12, 128, num_gqa_groups=2, attn_mask_type="causal"),  # GQA b=8
    # THD + SWA
    "cp_thd_swa_0": ModelConfig(
        8, 8192, 12, 128, attn_mask_type="causal", window_size=(512, 0)
    ),  # MHA SWA causal
    "cp_thd_swa_1": ModelConfig(
        8, 8192, 12, 128, num_gqa_groups=2, attn_mask_type="causal", window_size=(512, 0)
    ),  # GQA SWA causal
    "cp_thd_swa_2": ModelConfig(
        8, 8192, 12, 128, attn_mask_type="causal", window_size=(512, 512)
    ),  # MHA SWA causal+right
    "cp_thd_swa_3": ModelConfig(
        8, 8192, 12, 128, num_gqa_groups=2, attn_mask_type="causal", window_size=(512, 512)
    ),  # GQA SWA causal+right
}


# Variable-length training-workload configs used to exercise non-uniform THD batches.
_training_workloads = {
    "bucket32k": (
        ModelConfig(4, 32768, 32, 128, num_gqa_groups=8, attn_mask_type="causal"),
        "24576,28672,30720,32768",
    ),
    "bucket64k": (
        ModelConfig(4, 65536, 32, 128, num_gqa_groups=8, attn_mask_type="causal"),
        "57344,61440,63488,65536",
    ),
    "mixed32k": (
        ModelConfig(8, 32768, 32, 128, num_gqa_groups=8, attn_mask_type="causal"),
        "16384,24576,32768,8192,28672,32768,20480,16384",
    ),
    "rl16k": (
        ModelConfig(8, 16384, 32, 128, num_gqa_groups=8, attn_mask_type="causal"),
        "4096,6144,6144,8192,8192,10240,12288,16384",
    ),
    "outlier64k": (
        ModelConfig(4, 65536, 32, 128, num_gqa_groups=8, attn_mask_type="causal"),
        "8192,8192,8192,65536",
    ),
    "bucket128k": (
        ModelConfig(3, 131072, 32, 128, num_gqa_groups=8, attn_mask_type="causal"),
        "114688,122880,131072",
    ),
    # SWA variants of mixed32k for cross-backend SWA correctness checks.
    "mixed32k_swa512": (
        ModelConfig(
            8,
            32768,
            32,
            128,
            num_gqa_groups=8,
            attn_mask_type="causal",
            window_size=(512, 0),
        ),
        "16384,24576,32768,8192,28672,32768,20480,16384",
    ),
    "mixed32k_swa1024": (
        ModelConfig(
            8,
            32768,
            32,
            128,
            num_gqa_groups=8,
            attn_mask_type="causal",
            window_size=(1024, 0),
        ),
        "16384,24576,32768,8192,28672,32768,20480,16384",
    ),
    "mixed32k_swa2048": (
        ModelConfig(
            8,
            32768,
            32,
            128,
            num_gqa_groups=8,
            attn_mask_type="causal",
            window_size=(2048, 0),
        ),
        "16384,24576,32768,8192,28672,32768,20480,16384",
    ),
}

# SWA variants of all 6 training workloads at windows 512/1024/2048.
_swa_base = {
    "bucket32k": (4, 32768, "24576,28672,30720,32768"),
    "bucket64k": (4, 65536, "57344,61440,63488,65536"),
    "mixed32k_full": (8, 32768, "16384,24576,32768,8192,28672,32768,20480,16384"),
    "rl16k": (8, 16384, "4096,6144,6144,8192,8192,10240,12288,16384"),
    "outlier64k": (4, 65536, "8192,8192,8192,65536"),
    "bucket128k": (3, 131072, "114688,122880,131072"),
}
for _name, (_b, _s, _pat) in _swa_base.items():
    for _w in (512, 1024, 2048):
        # Use "_full" suffix dropped; keep mixed32k_swa* names matching above
        if _name == "mixed32k_full":
            continue  # already added with mixed32k_swa{512,1024,2048}
        _key = f"{_name}_swa{_w}"
        _training_workloads[_key] = (
            ModelConfig(
                _b,
                _s,
                32,
                128,
                num_gqa_groups=8,
                attn_mask_type="causal",
                window_size=(_w, 0),
            ),
            _pat,
        )
for _name, (_cfg, _pat) in _training_workloads.items():
    _cfg.thd_seqlen_pattern = _pat
    model_configs_fused_attn[_name] = _cfg


# Pytest-runnable subset: skip the largest fixed-token SWA configs.
_pytest_skip_configs = {
    "uniform_1x512k_swa1024",
    "uniform_2x256k_swa1024",
    "uniform_16x32k_swa1024",
}
_pytest_configs = {
    k: v for k, v in model_configs_fused_attn.items() if k not in _pytest_skip_configs
}


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 7), reason="cuDNN 8.9.7+ is required.")
@pytest.mark.skipif(get_device_compute_capability() < (9, 0), reason="THD requires sm90+.")
@pytest.mark.parametrize("model", _pytest_configs.keys())
@pytest.mark.parametrize("qkv_format", ["thd"])
@pytest.mark.parametrize("cp_comm_type", ["p2p", "all_gather", "a2a"])
def test_cp_benchmark_configs(model, qkv_format, cp_comm_type):
    """Run benchmark/stress configs through the core CP path for correctness."""
    if 2 > torch.cuda.device_count():
        pytest.skip("Test requires 2 GPUs")

    config = _pytest_configs[model]
    config.context_parallel = True
    config.cp_comm_type = cp_comm_type

    has_swa = config.window_size != (-1, 0) and config.window_size != (-1, -1)
    if has_swa and cp_comm_type == "p2p":
        pytest.skip("p2p does not support sliding window")

    # THD uses padding mask types for backend availability check
    check_config = copy.deepcopy(config)
    if "causal" in check_config.attn_mask_type:
        check_config.attn_mask_type = "padding_causal"
    else:
        check_config.attn_mask_type = "padding"

    available, *_ = get_available_attention_backends(
        check_config,
        qkv_dtype=torch.bfloat16,
        qkv_layout="_".join([qkv_format] * 3),
    )
    _, fused_supported, _ = available
    if not fused_supported:
        pytest.skip("FusedAttention not available for this config")

    extra_kwargs = {}
    thd_pat = getattr(config, "thd_seqlen_pattern", None)
    if thd_pat is not None:
        extra_kwargs["thd_seqlen_pattern"] = thd_pat

    run_distributed(
        get_bash_arguments(
            num_gpus_per_node=2,
            dtype="bf16",
            model=model,
            qkv_format=qkv_format,
            kernel_backend="FusedAttention",
            cp_comm_type=cp_comm_type,
            log_level=pytest_logging_level,
            **extra_kwargs,
        ),
    )


# Cross-backend consistency configs: run the same input through p2p / all_gather / a2a
# and assert the outputs agree within tolerance.
model_configs_cross_backend = {
    "cp_thd_0": model_configs_fused_attn["cp_thd_0"],
    "cp_thd_1": model_configs_fused_attn["cp_thd_1"],
    "cp_thd_2": model_configs_fused_attn["cp_thd_2"],
    "cp_thd_3": model_configs_fused_attn["cp_thd_3"],
    "cp_thd_swa_0": model_configs_fused_attn["cp_thd_swa_0"],
    "cp_thd_swa_1": model_configs_fused_attn["cp_thd_swa_1"],
    "cp_thd_swa_2": model_configs_fused_attn["cp_thd_swa_2"],
    "cp_thd_swa_3": model_configs_fused_attn["cp_thd_swa_3"],
}
# Add a few training workloads (smaller ones to keep runtime reasonable)
for _name in ["rl16k", "bucket32k", "mixed32k"]:
    if _name in model_configs_fused_attn:
        model_configs_cross_backend[_name] = model_configs_fused_attn[_name]


@pytest.mark.skipif(get_cudnn_version() < (9, 3, 0), reason="cuDNN 9.3.0+ is required.")
@pytest.mark.skipif(get_device_compute_capability() < (9, 0), reason="THD requires sm90+.")
@pytest.mark.parametrize("model", model_configs_cross_backend.keys())
def test_cp_thd_cross_backend_consistency(model):
    """Compare outputs of p2p, all_gather, and a2a backends for THD format."""
    if 2 > torch.cuda.device_count():
        pytest.skip("Test requires 2 GPUs")

    config = model_configs_cross_backend[model]
    config.context_parallel = True

    has_swa = config.window_size != (-1, 0) and config.window_size != (-1, -1)
    # p2p doesn't support sliding window
    backends = ["all_gather", "a2a"] if has_swa else ["p2p", "all_gather", "a2a"]
    saved_outputs = {}

    # THD uses padding mask types for backend availability check
    check_config = copy.deepcopy(config)
    if "causal" in check_config.attn_mask_type:
        check_config.attn_mask_type = "padding_causal"
    else:
        check_config.attn_mask_type = "padding"

    with tempfile.TemporaryDirectory() as tmpdir:
        for backend in backends:
            check_config.cp_comm_type = backend
            available, *_ = get_available_attention_backends(
                check_config,
                qkv_dtype=torch.bfloat16,
                qkv_layout="thd_thd_thd",
            )
            _, fused_supported, _ = available
            if not fused_supported:
                pytest.skip(f"FusedAttention not available for {backend}")

            save_dir = os.path.join(tmpdir, backend)
            env = os.environ.copy()
            env["CP_CROSS_BACKEND_SAVE_DIR"] = save_dir
            extra_kwargs = {}
            thd_pat = getattr(config, "thd_seqlen_pattern", None)
            if thd_pat is not None:
                extra_kwargs["thd_seqlen_pattern"] = thd_pat
            run_distributed(
                get_bash_arguments(
                    num_gpus_per_node=2,
                    dtype="bf16",
                    model=model,
                    qkv_format="thd",
                    kernel_backend="FusedAttention",
                    cp_comm_type=backend,
                    log_level=pytest_logging_level,
                    **extra_kwargs,
                ),
                env=env,
            )
            saved_outputs[backend] = {
                r: torch.load(
                    os.path.join(save_dir, f"outputs_{backend}_rank{r}.pt"),
                    weights_only=True,
                )
                for r in range(2)
            }

        # Compare all backends pairwise against the first as reference.
        # Cross-backend diffs compound two independent CP implementations,
        # so use a wider tolerance than the per-backend CP-vs-nonCP tests.
        ref = backends[0]
        atol = 0.1
        for backend in backends[1:]:
            for rank in range(2):
                ref_out = saved_outputs[ref][rank]
                cmp_out = saved_outputs[backend][rank]
                for key in ["out", "dq", "dk", "dv"]:
                    if ref_out[key] is None or cmp_out[key] is None:
                        continue
                    diff = (ref_out[key] - cmp_out[key]).abs().max().item()
                    assert (
                        diff < atol
                    ), f"{backend} vs {ref} rank{rank} {key}: max_diff={diff} > {atol}"

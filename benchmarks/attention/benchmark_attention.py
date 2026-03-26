# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# Note: A "flash-attn v3" warning may appear from Transformer Engine. The script does not
# install flash-attn; it uses whatever is already installed (e.g. v2). The warning suggests
# installing v3 for Hopper+ for better support; timings are still from the active backend.

import os, sys, time
import subprocess
import pandas as pd
import numpy as np
import torch
import nvtx
import transformer_engine

# Add project root so "tests" can be imported when run from any directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))  # repo root (parent of benchmarks/)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tests.pytorch.utils import ModelConfig, get_available_attention_backends
from tests.pytorch.attention.test_attention import _run_dot_product_attention

pd.set_option("display.precision", 4)

# data type
dtype = torch.bfloat16
# number of iterations after 3 warmup iterations
num_iters = 3
# checkpointing
ckpt_attn = False
# workspace optimization path for cuDNN attention
workspace_opt = True
# QKV memory layout
qkv_layout = "bshd_bshd_bshd"
# padding between sequences for qkv_format=thd
pad_between_seqs = False
# training mode
is_training = True

# Substrings to match kernel names in nsys cuda_gpu_trace CSV (case-insensitive).
# If profiling output changes, update these (e.g. cuDNN may use "cudnn" or "cuda", flash may use "flash" or "fmha").
KERNEL_NAME_CUDNN = "cudnn"
KERNEL_NAME_FLASH = "flash"

model_configs = {
    # ModelConfig(batch_size, max_seqlen_q, num_heads, head_dim_qk, max_seqlen_kv, num_gqa_groups, ...)
    "test_0": ModelConfig(
        2, 512, 16, 64, 512, 16, dropout_p=0.0, attn_mask_type="no_mask", attn_bias_type="no_bias"
    ),  # short seq
    "test_1": ModelConfig(
        2, 2048, 16, 128, 2048, 16, dropout_p=0.0, attn_mask_type="causal", attn_bias_type="no_bias"
    ),  # longer seq, mask
    "test_2": ModelConfig(
        2,
        2048,
        16,
        128,
        2048,
        16,
        dropout_p=0.0,
        attn_mask_type="causal",
        attn_bias_type="post_scale_bias",
    ),  # bias; FlashAttention does not support post_scale_bias, so only cuDNN runs
    "test_3": ModelConfig(
        2, 8192, 32, 128, 8192, 4, dropout_p=0.0, attn_mask_type="causal", attn_bias_type="no_bias"
    ),  # GQA
}


def benchmark_dot_product_attention(
    model, fused_attn_supported, flash_attn_supported, append_csv=True
):
    config = model_configs[model]
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)
    else:
        tols = dict(atol=5e-3, rtol=5e-3)

    cudnn_times = []
    flash_times = []
    warmup_iters = 3
    for i in range(warmup_iters):
        if fused_attn_supported:
            fused_attn_fwd, _, fused_attn_bwd = _run_dot_product_attention(
                dtype,
                config,
                "FusedAttention",
                ckpt_attn,
                qkv_layout,
                workspace_opt,
                pad_between_seqs,
                is_training,
            )
        if flash_attn_supported:
            flash_attn_fwd, _, flash_attn_bwd = _run_dot_product_attention(
                dtype,
                config,
                "FlashAttention",
                ckpt_attn,
                qkv_layout,
                workspace_opt,
                pad_between_seqs,
                is_training,
            )
        if fused_attn_supported and flash_attn_supported:
            torch.testing.assert_close(fused_attn_fwd, flash_attn_fwd, **tols)
            for i, _ in enumerate(flash_attn_bwd):
                if fused_attn_bwd[i] is not None and flash_attn_bwd[i] is not None:
                    torch.testing.assert_close(fused_attn_bwd[i], flash_attn_bwd[i], **tols)

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.synchronize()
    fused_attn_start = time.time()
    if fused_attn_supported:
        for i in range(num_iters):
            _run_dot_product_attention(
                dtype,
                config,
                "FusedAttention",
                ckpt_attn,
                qkv_layout,
                workspace_opt,
                pad_between_seqs,
                is_training,
            )
    torch.cuda.synchronize()
    fused_attn_time = time.time() - fused_attn_start if fused_attn_supported else 0

    torch.cuda.synchronize()
    flash_attn_start = time.time()
    if flash_attn_supported:
        for i in range(num_iters):
            _run_dot_product_attention(
                dtype,
                config,
                "FlashAttention",
                ckpt_attn,
                qkv_layout,
                workspace_opt,
                pad_between_seqs,
                is_training,
            )
    torch.cuda.synchronize()
    flash_attn_time = time.time() - flash_attn_start if flash_attn_supported else 0

    if append_csv:
        df = pd.read_csv("times.csv")
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    [
                        [
                            fused_attn_time * 1e3 / num_iters,
                            0,
                            0,
                            0,
                            flash_attn_time * 1e3 / num_iters,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ],
                    columns=df.columns,
                ),
            ],
            ignore_index=True,
        )
        df.to_csv("times.csv", index=False)
    torch.cuda.cudart().cudaProfilerStop()


def parse_results(per_cudnn, per_flash, model):
    bench_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"prof_{model}_cuda_gpu_trace.csv"
    filepath = os.path.join(bench_dir, filename)
    if not os.path.isfile(filepath):
        return
    df = pd.read_csv(filepath)
    df_times = pd.read_csv(os.path.join(bench_dir, "times.csv"))
    row = len(df_times.index) - 1

    # Match kernel names case-insensitively; column may be "Name" or "Kernel Name" in nsys output
    name_col = "Name" if "Name" in df.columns else "Kernel Name"
    names = df[name_col].astype(str).str.lower()

    if per_cudnn > 0:
        cudnn_mask = names.str.contains(KERNEL_NAME_CUDNN.lower(), regex=False)
        if cudnn_mask.any():
            t_cudnn_all = df.loc[cudnn_mask, "Duration (ns)"].to_numpy()
            t_cudnn_all = t_cudnn_all.reshape(-1, per_cudnn)
            t_cudnn_avg = np.average(t_cudnn_all, axis=0)
            df_times.loc[row, "FusedAttention Kernels (fwd)"] = t_cudnn_avg[0] / 1e6
            df_times.loc[row, "FusedAttention Kernels (bwd)"] = t_cudnn_avg[1:4].sum() / 1e6
            df_times.loc[row, "FusedAttention Kernels (fwd+bwd)"] = t_cudnn_avg.sum() / 1e6

    if per_flash > 0:
        flash_mask = names.str.contains(KERNEL_NAME_FLASH.lower(), regex=False)
        if flash_mask.any():
            t_flash_all = df.loc[flash_mask, "Duration (ns)"].to_numpy()
            t_flash_all = t_flash_all.reshape(-1, per_flash)
            t_flash_avg = np.average(t_flash_all, axis=0)
            df_times.loc[row, "FlashAttention Kernels (fwd)"] = t_flash_avg[0] / 1e6
            df_times.loc[row, "FlashAttention Kernels (bwd)"] = t_flash_avg[1:4].sum() / 1e6
            df_times.loc[row, "FlashAttention Kernels (fwd+bwd)"] = t_flash_avg.sum() / 1e6

    if per_cudnn > 0 and per_flash > 0:
        fwd_bwd = df_times.loc[row, "FusedAttention Kernels (fwd+bwd)"]
        if fwd_bwd and fwd_bwd > 0:
            df_times.loc[row, "Fused vs Flash Kernels Speedup (fwd+bwd)"] = (
                df_times.loc[row, "FlashAttention Kernels (fwd+bwd)"] / fwd_bwd
            )
    df_times.to_csv(os.path.join(bench_dir, "times.csv"), index=False)


def main():
    times = pd.DataFrame(
        columns=[
            "FusedAttention Module",
            "FusedAttention Kernels (fwd)",
            "FusedAttention Kernels (bwd)",
            "FusedAttention Kernels (fwd+bwd)",
            "FlashAttention Module",
            "FlashAttention Kernels (fwd)",
            "FlashAttention Kernels (bwd)",
            "FlashAttention Kernels (fwd+bwd)",
            "Fused vs Flash Kernels Speedup (fwd+bwd)",
        ]
    )
    times.to_csv("times.csv", index=False)

    device_id = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device_id)
    print(
        f"Device {device_id}: "
        f"{device_properties.name} GPU, "
        f"sm{device_properties.major}{device_properties.minor} compute capability, "
        f"{device_properties.total_memory/1024**3:.1f}GB memory"
    )
    for model in model_configs.keys():
        config = model_configs[model]
        available_backends, _, fused_attn_backends = get_available_attention_backends(
            config,
            qkv_dtype=dtype,
            qkv_layout=qkv_layout,
            # window_size=config.window_size,
            pad_between_seqs=pad_between_seqs,
        )
        flash_attn_supported, fused_attn_supported, unfused_attn_supported = available_backends

        print(
            f'Running {model} with {"cuDNN attention" if fused_attn_supported else ""}'
            f'{" and flash-attention" if flash_attn_supported else ""}...'
        )

        # Run benchmark in main process so times.csv always gets a row (works without nsys)
        benchmark_dot_product_attention(
            model, fused_attn_supported, flash_attn_supported, append_csv=True
        )

        # Optional: run under nsys to get kernel-level stats; subprocess must not append again
        bench_code = (
            "import benchmark_attention; "
            "benchmark_attention.benchmark_dot_product_attention("
            f"'{model}', {fused_attn_supported}, {flash_attn_supported}, append_csv=False)"
        )
        prof_cmd = [
            "nsys",
            "profile",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop-shutdown",
            "--force-overwrite=true",
            f"--output=prof_{model}",
            "python",
            "-c",
            bench_code,
        ]
        bench_dir = os.path.dirname(os.path.abspath(__file__))
        prof_ret = subprocess.call(
            prof_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=bench_dir,
        )
        if prof_ret == 0:
            stats_cmd = [
                "nsys",
                "stats",
                "-q",
                "-r",
                "cuda_gpu_trace",
                "--format",
                "csv,column",
                "--force-overwrite=true",
                "--force-export=true",
                f"--output=prof_{model}",
                f"prof_{model}.nsys-rep",
            ]
            if fused_attn_supported:
                num_kernels_cudnn = 4
                if config.attn_bias_type == "post_scale_bias":
                    num_kernels_cudnn = num_kernels_cudnn + 1
                if config.num_heads != config.num_gqa_groups:
                    num_kernels_cudnn = num_kernels_cudnn + 2
            else:
                num_kernels_cudnn = 0
            num_kernels_flash = 4 if flash_attn_supported else 0
            subprocess.call(
                stats_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=bench_dir,
            )
            parse_code = (
                "import benchmark_attention; "
                "benchmark_attention.parse_results("
                f"{num_kernels_cudnn}, {num_kernels_flash}, '{model}')"
            )
            parse_cmd = ["python", "-c", parse_code]
            subprocess.call(
                parse_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=bench_dir,
            )

    df_times = pd.read_csv("times.csv")
    n_models = len(model_configs)
    if len(df_times) != n_models:
        raise RuntimeError(
            f"times.csv has {len(df_times)} rows but expected {n_models}. "
            "Subprocess benchmarks may have failed (check nsys availability)."
        )
    df_times.index = list(model_configs.keys())
    # Prefer module timings (from time.time(), always populated); fall back to kernel timings (from nsys)
    cudnn_col = "FusedAttention Module"
    flash_col = "FlashAttention Module"
    a = df_times[[cudnn_col, flash_col]].copy()
    a.columns = ["cuDNN fwd+bwd (ms)", "flash-attn fwd+bwd (ms)"]
    # Speedup: flash/cudnn ratio (>1 means cuDNN faster). N/A when only one backend ran (e.g. test_2 has bias, flash not used).
    cudnn_ms = df_times[cudnn_col]
    flash_ms = df_times[flash_col]
    speedup = np.where((cudnn_ms > 0) & (flash_ms > 0), flash_ms / cudnn_ms, np.nan)
    a["cuDNN vs flash speedup"] = speedup
    # Show "N/A" instead of NaN when speedup not defined (only one backend ran)
    a_display = a.copy()
    a_display["cuDNN vs flash speedup"] = [f"{x:.4f}" if not pd.isna(x) else "N/A" for x in speedup]
    print()
    print(a_display)


if __name__ == "__main__":
    main()

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os, sys, time
import subprocess
import pandas as pd
import numpy as np
import torch
import nvtx
import transformer_engine
from tests.pytorch.fused_attn.test_fused_attn import (
    ModelConfig,
    _is_flash_attention_supported,
    _is_fused_attention_supported,
    _is_unfused_attention_supported,
    _run_dot_product_attention,
)

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
# sliding window attention
swa = False
# padding between sequences for qkv_format=thd
pad_between_seqs = False
# training mode
is_training = True

model_configs = {
    #   test:             b,  h, hg,   d,   sq,  skv,   p,     mask,              bias
    "test_0": ModelConfig(2, 16, 16, 64, 512, 512, 0.0, "no_mask", "no_bias"),  # short seq
    "test_1": ModelConfig(2, 16, 16, 128, 2048, 2048, 0.0, "causal", "no_bias"),  # longer seq, mask
    "test_2": ModelConfig(2, 16, 16, 128, 2048, 2048, 0.0, "causal", "post_scale_bias"),  # bias
    "test_3": ModelConfig(2, 32, 4, 128, 8192, 8192, 0.0, "causal", "no_bias"),  # GQA
}


def benchmark_dot_product_attention(model, fused_attn_supported, flash_attn_supported):
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
            fused_attn_fwd, fused_attn_bwd = _run_dot_product_attention(
                dtype,
                config,
                "FusedAttention",
                ckpt_attn,
                qkv_layout,
                workspace_opt,
                swa,
                pad_between_seqs,
                is_training,
            )
        if flash_attn_supported:
            flash_attn_fwd, flash_attn_bwd = _run_dot_product_attention(
                dtype,
                config,
                "FlashAttention",
                ckpt_attn,
                qkv_layout,
                workspace_opt,
                swa,
                pad_between_seqs,
                is_training,
            )
        if fused_attn_supported and flash_attn_supported:
            torch.testing.assert_close(fused_attn_fwd, flash_attn_fwd, **tols)
            for i, _ in enumerate(flash_attn_bwd):
                torch.testing.assert_close(fused_attn_bwd[i], flash_attn_bwd[i], **tols)

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.synchronize()
    fused_attn_start = time.time()
    if fused_attn_supported:
        for i in range(num_iters):
            fused_attn_fwd, fused_attn_bwd = _run_dot_product_attention(
                dtype,
                config,
                "FusedAttention",
                ckpt_attn,
                qkv_layout,
                workspace_opt,
                swa,
                pad_between_seqs,
                is_training,
            )
    torch.cuda.synchronize()
    fused_attn_time = time.time() - fused_attn_start if fused_attn_supported else 0

    torch.cuda.synchronize()
    flash_attn_start = time.time()
    if flash_attn_supported:
        for i in range(num_iters):
            flash_attn_fwd, flash_attn_bwd = _run_dot_product_attention(
                dtype,
                config,
                "FlashAttention",
                ckpt_attn,
                qkv_layout,
                workspace_opt,
                swa,
                pad_between_seqs,
                is_training,
            )
    torch.cuda.synchronize()
    flash_attn_time = time.time() - flash_attn_start if flash_attn_supported else 0

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
    filename = f"prof_{model}_cuda_gpu_trace.csv"
    df = pd.read_csv(os.path.join("./", filename))
    df_times = pd.read_csv("times.csv")
    row = len(df_times.index) - 1

    if per_cudnn > 0:
        t_cudnn_all = df[df["Name"].str.contains("cudnn")]["Duration (ns)"].to_numpy()
        t_cudnn_all = t_cudnn_all.reshape(-1, per_cudnn)
        t_cudnn_avg = np.average(t_cudnn_all, axis=0)
        df_times.loc[row, "FusedAttention Kernels (fwd)"] = t_cudnn_avg[0] / 1e6
        df_times.loc[row, "FusedAttention Kernels (bwd)"] = t_cudnn_avg[1:4].sum() / 1e6
        df_times.loc[row, "FusedAttention Kernels (fwd+bwd)"] = t_cudnn_avg.sum() / 1e6

    if per_flash > 0:
        t_flash_all = df[df["Name"].str.contains("void flash")]["Duration (ns)"].to_numpy()
        t_flash_all = t_flash_all.reshape(-1, per_flash)
        t_flash_avg = np.average(t_flash_all, axis=0)
        df_times.loc[row, "FlashAttention Kernels (fwd)"] = t_flash_avg[0] / 1e6
        df_times.loc[row, "FlashAttention Kernels (bwd)"] = t_flash_avg[1:4].sum() / 1e6
        df_times.loc[row, "FlashAttention Kernels (fwd+bwd)"] = t_flash_avg.sum() / 1e6

    if per_cudnn > 0 and per_flash > 0:
        df_times.loc[row, "Fused vs Flash Kernels Speedup (fwd+bwd)"] = (
            df_times.loc[row, "FlashAttention Kernels (fwd+bwd)"]
            / df_times.loc[row, "FusedAttention Kernels (fwd+bwd)"]
        )
    df_times.to_csv("times.csv", index=False)


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
        fused_attn_supported, fused_attn_backend = _is_fused_attention_supported(
            config,
            dtype,
            qkv_layout=qkv_layout,
        )
        fused_attn_supported = fused_attn_supported and not swa
        flash_attn_supported = _is_flash_attention_supported(config)
        print(
            f'Running {model} with {"cuDNN attention" if fused_attn_supported else ""}'
            f'{" and flash-attention" if flash_attn_supported else ""}...'
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
            f""" "import benchmark_attention;""",
            f"""benchmark_attention.benchmark_dot_product_attention("""
            f"""'{model}', {fused_attn_supported}, {flash_attn_supported})" """,
        ]
        prof_cmd = " ".join(prof_cmd)
        subprocess.call(prof_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
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
        stats_cmd = " ".join(stats_cmd)
        subprocess.call(stats_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        parse_cmd = [
            "python",
            "-c",
            f""" "import benchmark_attention;""",
            f"""benchmark_attention.parse_results("""
            f"""{num_kernels_cudnn}, {num_kernels_flash}, '{model}')" """,
        ]
        parse_cmd = " ".join(parse_cmd)
        subprocess.call(parse_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)

    df_times = pd.read_csv("times.csv")
    df_times.index = list(model_configs.keys())
    a = df_times[
        [
            "FusedAttention Kernels (fwd+bwd)",
            "FlashAttention Kernels (fwd+bwd)",
            "Fused vs Flash Kernels Speedup (fwd+bwd)",
        ]
    ]
    a.columns = ["cuDNN fwd+bwd (ms)", "flash-attn fwd+bwd (ms)", "cuDNN vs flash speedup"]
    print()
    print(a)


if __name__ == "__main__":
    main()

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
import torch
import torch.utils.benchmark as benchmark
import pandas as pd

from transformer_engine.pytorch.module import Linear as TELinear
from transformer_engine.common.recipe import (
    Float8BlockScaling,
    MXFP8BlockScaling,
    NVFP4BlockScaling,
)
from transformer_engine.pytorch.quantization import autocast, FP8GlobalStateManager
from contextlib import nullcontext

"""
# Profile BF16 recipe with Nsight Systems
nsys profile \
    --output=./benchmarks/linear/b200_linear_bf16 \
    --force-overwrite true \
    --trace=cuda,nvtx,cudnn,cublas \
    python benchmarks/linear/benchmark_linear.py --profile --recipe bf16

# Profile FP8 sub-channel recipe with Nsight Systems
nsys profile \
    --output=./benchmarks/linear/b200_linear_fp8_sub_channel \
    --force-overwrite true \
    --trace=cuda,nvtx,cudnn,cublas \
    python benchmarks/linear/benchmark_linear.py --profile --recipe fp8_sub_channel

# Profile MXFP8 recipe with Nsight Systems
nsys profile \
    --output=./benchmarks/linear/b200_linear_mxfp8 \
    --force-overwrite true \
    --trace=cuda,nvtx,cudnn,cublas \
    python benchmarks/linear/benchmark_linear.py --profile --recipe mxfp8

# Profile NVFP4 recipe with Nsight Systems
nsys profile \
    --output=./benchmarks/linear/b200_linear_nvfp4_rht_cast_fusion \
    --force-overwrite true \
    --trace=cuda,nvtx,cudnn,cublas \
    python benchmarks/linear/benchmark_linear.py --profile --recipe nvfp4

# Example to look at a single kernel target with NCU, like the fused hadamard amax kernel for NVFP4 recipe
ncu -f -o ./benchmarks/linear/ncu_b200_linear_nvfp4_rht_cast_fusion \
    --set=full \
    --kernel-name "row_col_rht_gemm_device" \
    -s 5 -c 5 \
    python benchmarks/linear/benchmark_linear.py --profile --recipe nvfp4

"""

RECIPES = {
    "bf16": None,
    "fp8_sub_channel": Float8BlockScaling(),
    "mxfp8": MXFP8BlockScaling(),
    "nvfp4": NVFP4BlockScaling(),
}

mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = (
    FP8GlobalStateManager.is_fp8_block_scaling_available()
)
nvfp4_available, reason_for_no_nvfp4 = FP8GlobalStateManager.is_nvfp4_available()


def run_linear_multiple_steps(layer, x, mode, gradient, run_num_steps=1, recipe=None):
    assert mode in ["fwd_only", "fwd_bwd"]
    quantization_context = (
        autocast(enabled=True, recipe=recipe) if recipe is not None else nullcontext()
    )

    if mode == "fwd_only":
        with torch.no_grad(), quantization_context:
            for i in range(run_num_steps):
                y_q = layer.forward(
                    x,
                    is_first_microbatch=(i == 0),
                )
        return y_q
    else:
        # reset gradients
        layer.zero_grad()
        x.grad = None

        with quantization_context:
            for i in range(run_num_steps):
                label = f"step_{i}"
                torch.cuda.nvtx.range_push(label)
                y_q = layer.forward(
                    x,
                    is_first_microbatch=(i == 0),
                )
                y_q.backward(gradient)
                torch.cuda.nvtx.range_pop()

        grads_q = []
        grads_q.append(x.grad)
        # remaining derivatives are in respect to model parameters
        for p in layer.parameters():
            if p.requires_grad:
                grads_q.append(p.grad)

        return y_q, grads_q


def benchmark_linear(
    x,
    w,
    bias,
    recipe_name,
    mode,
):
    params_dtype = torch.bfloat16
    recipe = RECIPES[recipe_name]

    in_features = x.shape[1]
    out_features = w.shape[0]
    gradient = torch.ones((x.shape[0], out_features), dtype=torch.bfloat16, device=x.device)

    layer = TELinear(
        in_features,
        out_features,
        bias=bias is not None,
        params_dtype=params_dtype,
    )

    layer = layer.to("cuda")
    with torch.no_grad():
        layer.weight.copy_(w)
        if bias is not None:
            layer.bias.copy_(bias)

    num_microbatches = 32

    label = f"{recipe_name}_{'linear'}"
    torch.cuda.nvtx.range_push(label)
    timing = benchmark.Timer(
        stmt="run_linear_multiple_steps(layer, x, mode, gradient, num_microbatches, recipe)",
        globals={
            "run_linear_multiple_steps": run_linear_multiple_steps,
            "layer": layer,
            "x": x,
            "mode": mode,
            "gradient": gradient,
            "num_microbatches": num_microbatches,
            "recipe": recipe,
        },
        num_threads=1,
    ).blocked_autorange(min_run_time=10)
    print(f"{recipe_name}: {timing} \n")
    timing_ms = timing.median * 1000 / num_microbatches

    return timing_ms


def run_benchmark_linear(mkns, recipe_name, use_bias, fwd_only=False):
    data = []
    assert not use_bias, "Bias is not supported in this benchmark script"

    print(f"========== Benchmarking {recipe_name} ==========")
    for m, k, n in mkns:
        device = "cuda"
        x = torch.randn((m, k), dtype=torch.bfloat16, device=device, requires_grad=True)
        w = torch.randn((n, k), dtype=torch.bfloat16, device=device)
        bias = None

        # Run the benchmark
        print(f"fwd_m={m}, fwd_k={k}, fwd_n={n}")
        print(f"fwd_only: {fwd_only}")

        linear_fwd_bwd_timing_ms = benchmark_linear(
            x,
            w,
            bias,
            recipe_name,
            mode="fwd_only" if fwd_only else "fwd_bwd",
        )

        # Append the results
        data.append(
            [
                m,
                k,
                n,
                recipe_name,
                linear_fwd_bwd_timing_ms,
            ]
        )

    timing_notation = "linear_fwd_time_ms" if fwd_only else "linear_fwd_bwd_time_ms"

    df = pd.DataFrame(
        data=data,
        columns=[
            "m",
            "k",
            "n",
            "recipe",
            timing_notation,
        ],
    )

    print(df, "\n")
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable profiling mode")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_output/",
        help="output path for report",
    )
    # arguments for recipe, options are fp8_sub_channel, mxfp8, bf16, all
    parser.add_argument(
        "--recipe",
        type=str,
        default="bf16",
        help="Recipe to use, options are fp8_sub_channel, mxfp8, bf16, or all",
    )
    parser.add_argument(
        "--token-dim",
        type=int,
        default=None,
        help="Token dimension to use, calculated by SEQ_LEN * MBS / TP_SIZE",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension to use",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=None,
        help="Output dimension to use",
    )
    parser.add_argument(
        "--fwd-only",
        action="store_true",
        default=False,
        help="Run forward pass only, default is both forward and backward passes",
    )
    args = parser.parse_args()

    use_bias = False

    token_dim_list = [16384]
    hidden_dim_list = [4096]
    output_dim_list = [4096]

    if args.token_dim is not None:
        token_dim_list = [args.token_dim]

    if args.hidden_dim is not None:
        hidden_dim_list = [args.hidden_dim]

    if args.output_dim is not None:
        output_dim_list = [args.output_dim]

    # MKN for linear
    mkns = []
    for m in token_dim_list:
        for k in hidden_dim_list:
            for n in output_dim_list:
                mkns.append((m, k, n))

    # default recipes to run if not specified
    recipe_list = ["bf16"]

    if args.recipe == "all":
        recipe_list = ["bf16", "fp8_sub_channel", "mxfp8", "nvfp4"]
    else:
        recipe_list = [args.recipe]

    if args.profile:
        hidden_dim_to_profile = 4096 if args.hidden_dim is None else args.hidden_dim
        output_dim_to_profile = 4096 if args.output_dim is None else args.output_dim
        token_dim_to_profile = 16384 if args.token_dim is None else args.token_dim
        mkns = [(token_dim_to_profile, hidden_dim_to_profile, output_dim_to_profile)]
        # in profile mode, only run one recipe specified in args.recipe
        assert args.recipe != "all", (
            "In profile mode, only one recipe can be specified, please specify the recipe as"
            " fp8_sub_channel, mxfp8, nvfp4, or bf16"
        )
        recipe_list = [args.recipe]
        torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

    # Initialize a dataframe to store the results
    df_linears = pd.DataFrame()

    # Run the fp8 benchmarks
    for recipe_name in recipe_list:
        assert recipe_name in [
            "bf16",
            "fp8_sub_channel",
            "mxfp8",
            "nvfp4",
        ], "Recipe must be one of bf16, fp8_sub_channel, mxfp8, or nvfp4"
        if recipe_name == "mxfp8" and not mxfp8_available:
            print(f"MXFP8 is not available, skipping {recipe_name}")
            continue
        if recipe_name == "fp8_sub_channel" and not fp8_block_scaling_available:
            print(f"FP8 block scaling is not available, skipping {recipe_name}")
            continue
        if recipe_name == "nvfp4" and not nvfp4_available:
            print(f"NVFP4 is not available, skipping {recipe_name}")
            continue

        df = run_benchmark_linear(
            mkns,
            recipe_name,
            use_bias,
            fwd_only=args.fwd_only,
        )
        df_linears = pd.concat([df_linears, df])

    print(df_linears)

    if args.profile:
        torch.autograd.profiler.emit_nvtx().__exit__(None, None, None)

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
import torch
import torch.utils.benchmark as benchmark
import pandas as pd

from transformer_engine.pytorch.module import GroupedLinear
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
    --output=./benchmarks/linear/b200_numgemm_8_bf16 \
    --force-overwrite true \
    --trace=cuda,nvtx,cudnn,cublas \
    python benchmarks/linear/benchmark_grouped_linear.py --profile --recipe bf16

# Profile FP8 sub-channel recipe with Nsight Systems
nsys profile \
    --output=./benchmarks/linear/h100hbm_numgemm_8_fp8_sub_channel \
    --force-overwrite true \
    --trace=cuda,nvtx,cudnn,cublas \
    python benchmarks/linear/benchmark_grouped_linear.py --profile --recipe fp8_sub_channel

# Profile MXFP8 recipe with Nsight Systems
nsys profile \
    --output=./benchmarks/linear/b200_numgemm_8_mxfp8 \
    --force-overwrite true \
    --trace=cuda,nvtx,cudnn,cublas \
    python benchmarks/linear/benchmark_grouped_linear.py --profile --recipe mxfp8

# Profile NVFP4 recipe with Nsight Systems
nsys profile \
    --output=./benchmarks/linear/b200_numgemm_8_nvfp4 \
    --force-overwrite true \
    --trace=cuda,nvtx,cudnn,cublas \
    python benchmarks/linear/benchmark_grouped_linear.py --profile --recipe nvfp4

# Example for jagged input benchmark to simulate unbalanced token splits
python benchmarks/linear/benchmark_grouped_linear.py --recipe nvfp4 --jagged-input "15296,8960,14656,14784,11712,7936,14080,10880"

# Example to look at a single kernel target with NCU, like the fused hadamard amax kernel for NVFP4 recipe
ncu -f -o ./benchmarks/linear/ncu_b200_numgemm_8_nvfp4_rht_amax \
    --set=full \
    --kernel-name "GroupHadamardAmaxTmaKernel" \
    -s 5 -c 5 \
    python benchmarks/linear/benchmark_grouped_linear.py --profile --recipe nvfp4 --profile

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


def run_linear_multiple_steps(layer, x, m_splits, mode, gradient, run_num_steps=1, recipe=None):
    assert mode in ["fwd_only", "fwd_bwd"]
    quantization_context = (
        autocast(enabled=True, recipe=recipe) if recipe is not None else nullcontext()
    )

    if mode == "fwd_only":
        with torch.no_grad(), quantization_context:
            for i in range(run_num_steps):
                y_q = layer.forward(
                    x,
                    m_splits,
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
                    m_splits,
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
    ws,
    m_splits,
    bias,
    recipe_name,
    mode,
    num_gemms=4,
):
    params_dtype = torch.bfloat16
    recipe = RECIPES[recipe_name]

    in_features = x.shape[1]
    out_features = ws[0].shape[0]
    gradient = torch.ones((x.shape[0], out_features), dtype=torch.bfloat16, device=x.device)

    layer = GroupedLinear(
        num_gemms,
        in_features,
        out_features,
        bias=bias is not None,
        params_dtype=params_dtype,
    )

    layer = layer.to("cuda")
    with torch.no_grad():
        for i in range(num_gemms):
            weight_i = getattr(layer, f"weight{i}")
            weight_i.copy_(ws[i])
            if bias is not None:
                bias_i = getattr(layer, f"bias{i}")
                bias_i.copy_(bias)

    num_microbatches = 32

    label = f"{recipe_name}_{'grouped'}"
    torch.cuda.nvtx.range_push(label)
    timing = benchmark.Timer(
        stmt=(
            "run_linear_multiple_steps(layer, x, m_splits, mode, gradient, num_microbatches,"
            " recipe)"
        ),
        globals={
            "run_linear_multiple_steps": run_linear_multiple_steps,
            "layer": layer,
            "x": x,
            "m_splits": m_splits,
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


def run_benchmark_linear(
    mkns, recipe_name, use_bias, num_gemms=4, m_splits_provided=None, fwd_only=False
):
    data = []
    assert not use_bias, "Bias is not supported for GroupedLinear benchmark"

    print(f"========== Benchmarking {recipe_name} ==========")
    for m, k, n in mkns:
        device = "cuda"
        x = torch.randn((m, k), dtype=torch.bfloat16, device=device, requires_grad=True)
        ws = [torch.randn((n, k), dtype=torch.bfloat16, device=device) for _ in range(num_gemms)]
        m_splits = [m // num_gemms] * num_gemms if m_splits_provided is None else m_splits_provided
        # Bias is not supported for GroupedLinear benchmark
        bias = None

        # Run the benchmark
        print(f"fwd_m={m}, fwd_k={k}, fwd_n={n}")
        print(f"m_splits: {m_splits}")
        print(f"fwd_only: {fwd_only}")

        grouped_fwd_bwd_timing_ms = benchmark_linear(
            x,
            ws,
            m_splits,
            bias,
            recipe_name,
            mode="fwd_only" if fwd_only else "fwd_bwd",
            num_gemms=num_gemms,
        )

        # Append the results
        data.append(
            [
                m,
                k,
                n,
                recipe_name,
                num_gemms,
                grouped_fwd_bwd_timing_ms,
            ]
        )

    timing_notation = "grouped_fwd_time_ms" if fwd_only else "grouped_fwd_bwd_time_ms"

    df = pd.DataFrame(
        data=data,
        columns=[
            "m",
            "k",
            "n",
            "recipe",
            "num_gemms",
            timing_notation,
        ],
    )

    print(df, "\n")
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable profiling mode")
    parser.add_argument(
        "--output_dir",
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
    # add an argument for the jagged input
    # example: [15296, 8960, 14656, 14784, 11712, 7936, 14080, 10880] => sums up to 98304
    parser.add_argument(
        "--jagged-input",
        type=str,
        default=None,
        help="Jagged input to use, example: [15296, 8960, 14656, 14784, 11712, 7936, 14080, 10880]",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=7168,
        help="Hidden dimension to use, default is 7168",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=2048,
        help="Output dimension to use, default is 2048",
    )
    parser.add_argument(
        "--fwd-only",
        action="store_true",
        default=False,
        help="Run forward pass only, default is both forward and backward passes",
    )
    args = parser.parse_args()

    jagged_input_splits = None
    if args.jagged_input is not None:
        jagged_input_splits = [int(x) for x in args.jagged_input.split(",")]
        print(f"Jagged input splits: {jagged_input_splits}")
        print(f"Jagged input splits sum: {sum(jagged_input_splits)}")
        print(f"Jagged input splits num_gemms: {len(jagged_input_splits)}")

    use_bias = False
    # Set the MKN values to benchmark
    # Deepseek V3 EP64, SEQ_LEN=8192, topK8
    # 256 expert => 4 local experts
    # Avg M per expert: AvgM = SEQ_LEN * topK / localExperts = 16384
    # M = AvgM * localExperts = 65536
    # K = 7168
    # N = 2048

    # Deepseek V3 EP32, SEQ_LEN=8192, topK8
    # 256 expert => 8 local experts
    # Avg M per expert: AvgM = SEQ_LEN * topK / localExperts = 8192
    # M = AvgM * localExperts = 65536
    # K = 7168
    # N = 2048

    # 4 or 8local experts per rank
    num_gemms_list = [4, 8]

    if jagged_input_splits is not None:
        num_gemms_list = [len(jagged_input_splits)]

    token_dim_list = [16384, 32768, 65536, 98304]
    hidden_dim_list = [7168]
    output_dim_list = [2048]

    # override the default targets to benchmark if specified
    if jagged_input_splits is not None:
        token_dim_list = [sum(jagged_input_splits)]

    if args.hidden_dim is not None:
        hidden_dim_list = [args.hidden_dim]

    if args.output_dim is not None:
        output_dim_list = [args.output_dim]

    # MKN for group linear
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
        num_gemms_list = [8]
        hidden_dim_to_profile = 7168 if args.hidden_dim is None else args.hidden_dim
        output_dim_to_profile = 2048 if args.output_dim is None else args.output_dim
        token_dim_to_profile = 8192 * 8
        if jagged_input_splits is not None:
            num_gemms_list = [len(jagged_input_splits)]
            token_dim_to_profile = sum(jagged_input_splits)
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
    for num_gemms in num_gemms_list:
        print(f"========== Benchmarking with num_gemms={num_gemms} ==========")
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
                num_gemms=num_gemms,
                m_splits_provided=jagged_input_splits,
                fwd_only=args.fwd_only,
            )
            df_linears = pd.concat([df_linears, df])

    print(df_linears)

    if args.profile:
        torch.autograd.profiler.emit_nvtx().__exit__(None, None, None)

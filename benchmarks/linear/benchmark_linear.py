# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import subprocess
import pandas as pd
import argparse
import torch
import transformer_engine
import torch.utils.benchmark as benchmark
import transformer_engine_torch as tex
import pathlib

import transformer_engine.pytorch.module.linear as linear
from transformer_engine.pytorch.fp8 import fp8_autocast
from transformer_engine.common import recipe

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
pd.set_option("display.precision", 4)

#  -------------- Quantization Recipes --------------


def recipe_bf16():
    return None


def recipe_fp8_per_tensor_delayed():
    return recipe.DelayedScaling()


def recipe_fp8_per_tensor_cs():
    return recipe.Float8CurrentScaling()


def is_fp8_recipe(get_recipe):
    # check the get recipe function name, see if it contains "fp8"
    return "fp8" in get_recipe.__name__


#  -------------- Benchmark Linear --------------


def run_linear(layer, x, mode, gradient):
    assert mode in ["fwd_only", "fwd_bwd"]

    if mode == "fwd_only":
        with torch.no_grad():
            y_q = layer.forward(x, is_first_microbatch=True)
        return y_q
    else:
        # reset gradients
        layer.zero_grad()
        x.grad = None

        y_q = layer.forward(x, is_first_microbatch=True)
        y_q.backward(gradient)

        grads_q = []
        grads_q.append(x.grad)
        # remaining derivatives are in respect to model parameters
        for p in layer.parameters():
            if p.requires_grad:
                grads_q.append(p.grad)

        return y_q, grads_q


def benchmark_linear(x, w, bias, mode):
    # params_dtype=torch.float32
    params_dtype = torch.bfloat16
    activation_dtype = torch.bfloat16

    in_features = x.shape[1]
    out_features = w.shape[0]
    gradient = torch.ones((x.shape[0], out_features), dtype=torch.bfloat16, device=x.device)
    layer = linear.Linear(
        in_features, out_features, bias=bias is not None, params_dtype=params_dtype
    )

    if activation_dtype is not None:
        layer.activation_dtype = activation_dtype

    layer = layer.to("cuda")
    with torch.no_grad():
        layer.weight.copy_(w)
        if bias is not None:
            layer.bias.copy_(bias)

    timing = benchmark.Timer(
        stmt="run_linear(layer, x, mode, gradient)",
        globals={
            "run_linear": run_linear,
            "layer": layer,
            "x": x,
            "mode": mode,
            "gradient": gradient,
        },
        num_threads=1,
    ).timeit(3000)
    timing_ms = timing.median * 1000
    return timing_ms


# -------------- Run Benchmark cases --------------


def run_benchmark_linear(x_size_list, w_size_list, recipe_list, use_bias):
    print("========== Benchmark Linear ==========")
    data = []

    for x_size, w_size in zip(x_size_list, w_size_list):
        x = torch.randn(x_size, dtype=torch.bfloat16, device=device, requires_grad=True)
        w = torch.randn(w_size, dtype=torch.bfloat16, device=device)
        bias = torch.randn(w_size[0], dtype=torch.bfloat16, device=device)
        bias = bias if use_bias else None

        print(f"x.shape={x_size} and w.shape={w_size}")

        for get_recipe in recipe_list:
            if is_fp8_recipe(get_recipe):
                fp8_recipe = get_recipe()
                with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                    fwd_timing_ms = benchmark_linear(x, w, bias, mode="fwd_only")
                    fwd_bwd_timing_ms = benchmark_linear(x, w, bias, mode="fwd_bwd")
            else:
                fwd_timing_ms = benchmark_linear(x, w, bias, mode="fwd_only")
                fwd_bwd_timing_ms = benchmark_linear(x, w, bias, mode="fwd_bwd")
            data.append(
                [
                    x_size,
                    w_size,
                    get_recipe.__name__.removeprefix("recipe_"),
                    "No",
                    fwd_timing_ms,
                    fwd_bwd_timing_ms,
                ]
            )

    df = pd.DataFrame(
        data=data,
        columns=[
            "x_size",
            "w_size",
            "recipe",
            "weight_cached",
            "linear_fwd_time_ms",
            "linear_fwd_bwd_time_ms",
        ],
    )

    print(df)
    return df


def print_device_info():
    device_id = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device_id)
    print(
        f"Device {device_id}: "
        f"{device_properties.name} GPU, "
        f"sm{device_properties.major}{device_properties.minor} compute capability, "
        f"{device_properties.total_memory/1024**3:.1f}GB memory"
    )
    print("Current GPU clocks:")
    subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.current.graphics,clocks.current.sm", "--format=csv"],
        check=True,
    )


if __name__ == "__main__":
    print_device_info()
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable profiling mode")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_output/",
        help="output path for report",
    )
    args = parser.parse_args()

    x_size_list = [
        (4096, 1024),
        (4096, 4096),
        (4096, 8192),
        (4096, 16384),
        (16384, 1024),
        (16384, 4096),
        (16384, 8192),
        (16384, 16384),
    ]
    w_size_list = [
        (4096, 1024),
        (4096, 4096),
        (8192, 8192),
        (16384, 16384),
        (4096, 1024),
        (4096, 4096),
        (8192, 8192),
        (16384, 16384),
    ]

    recipe_list = [recipe_bf16, recipe_fp8_per_tensor_delayed, recipe_fp8_per_tensor_cs]
    use_bias = False

    if args.profile:
        x_size_list = [(16384, 16384)]
        w_size_list = [(16384, 16384)]
        recipe_list = [recipe_fp8_per_tensor_cs]

    df_linear = run_benchmark_linear(x_size_list, w_size_list, recipe_list, use_bias)

    print("\nFinal DataFrame\n")
    print(df_linear)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    report_file = pathlib.Path(args.output_dir) / f"{pathlib.Path(__file__).stem}_report.csv"
    # optional: save to csv
    # df_linear.to_csv(report_file, index=False)
    print(f"Report saved to {report_file}")

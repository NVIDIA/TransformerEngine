import argparse
import torch
import torch.utils.benchmark as benchmark
import pandas as pd
import pathlib

from transformer_engine.pytorch.module import GroupedLinear
from transformer_engine.common.recipe import Float8BlockScaling
from transformer_engine.pytorch.fp8 import fp8_autocast
from contextlib import nullcontext

RECIPES = {
    "bf16": None,
    "fp8_sub_channel": Float8BlockScaling(),
}


def run_linear_multiple_steps(layer, x, m_splits, mode, gradient, run_num_steps=1, recipe=None):
    assert mode in ["fwd_only", "fwd_bwd"]
    fp8_context = (
        fp8_autocast(enabled=True, fp8_recipe=recipe) if recipe is not None else nullcontext()
    )
    # print(f"fp8_context: {fp8_context} and is it nullcontext? {isinstance(fp8_context, nullcontext)}")

    if mode == "fwd_only":
        with torch.no_grad(), fp8_context:
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

        with fp8_context:
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
    ).blocked_autorange(min_run_time=5)
    print(f"{recipe_name}: {timing} \n")
    timing_ms = timing.median * 1000 / num_microbatches

    return timing_ms


def run_benchmark_linear(mkns, recipe_name, use_bias, num_gemms=4):
    data = []
    assert not use_bias, "Bias is not supported for GroupedLinear benchmark"

    print(f"========== Benchmarking {recipe_name} ==========")
    for m, k, n in mkns:
        device = "cuda"
        x = torch.randn((m, k), dtype=torch.bfloat16, device=device, requires_grad=True)
        ws = [torch.randn((n, k), dtype=torch.bfloat16, device=device) for _ in range(num_gemms)]
        assert m % num_gemms == 0
        m_splits = [m // num_gemms] * num_gemms
        # Bias is not supported for GroupedLinear benchmark
        bias = None

        # Run the benchmark
        print(f"fwd_m={m}, fwd_k={k}, fwd_n={n}")

        grouped_fwd_bwd_timing_ms = benchmark_linear(
            x,
            ws,
            m_splits,
            bias,
            recipe_name,
            mode="fwd_bwd",
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

    df = pd.DataFrame(
        data=data,
        columns=[
            "m",
            "k",
            "n",
            "recipe",
            "num_gemms",
            "grouped_fwd_bwd_time_ms",
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
    args = parser.parse_args()

    use_bias = False
    # Set the MKN values to benchmark
    mkns = []
    for m in [1024]:
        # for m in [4096, 8192, 16384]:
        # for n in [1024, 2048, 4096, 8192, 16384]:
        for n in [3072]:
            for k in [4096]:
                mkns.append((m, k, n))

    # recipe_list = [
    #     "bf16", "fp8_sub_channel",
    # ]
    recipe_list = [
        "fp8_sub_channel",
    ]

    # num_gemms_list = [16, 32]
    num_gemms_list = [4]

    if args.profile:
        # nsys profile --output=./benchmarks/linear/mkn_4096_4096_4096_numgemm_1_bf16 --trace=cuda,nvtx,cudnn,cublas python benchmarks/linear/benchmark_grouped_linear.py --profile
        # nsys profile --output=./benchmarks/linear/mkn_8192_8192_8192_numgemm_32_bf16 --trace=cuda,nvtx,cudnn,cublas python benchmarks/linear/benchmark_grouped_linear.py --profile
        # nsys profile --output=./benchmarks/linear/mkn_4096_4096_4096_numgemm_8_fp8_sub_channel --trace=cuda,nvtx,cudnn,cublas python benchmarks/linear/benchmark_grouped_linear.py --profile
        # nsys profile --output=./benchmarks/linear/mkn_8192_8192_8192_numgemm_2_fp8_sub_channel --trace=cuda,nvtx,cudnn,cublas python benchmarks/linear/benchmark_grouped_linear.py --profile
        mkns = [(4096, 4096, 4096)]
        recipe_list = ["fp8_sub_channel"]
        # recipe_list = ["bf16"]
        num_gemms_list = [8]
        torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

    # Initialize a dataframe to store the results
    df_linears = pd.DataFrame()

    # Run the fp8 benchmarks
    for num_gemms in num_gemms_list:
        print(f"========== Benchmarking with num_gemms={num_gemms} ==========")
        for recipe_name in recipe_list:
            df = run_benchmark_linear(
                mkns,
                recipe_name,
                use_bias,
                num_gemms=num_gemms,
            )
            df_linears = pd.concat([df_linears, df])

    print(df_linears)

    if args.profile:
        torch.autograd.profiler.emit_nvtx().__exit__(None, None, None)

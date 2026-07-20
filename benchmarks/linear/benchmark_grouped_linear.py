# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
import os
import torch
import torch.utils.benchmark as benchmark
import pandas as pd

import transformer_engine.pytorch as te
from transformer_engine.pytorch.module import GroupedLinear
from transformer_engine.common.recipe import (
    Float8BlockScaling,
    MXFP8BlockScaling,
    NVFP4BlockScaling,
)
from transformer_engine.pytorch.quantization import (
    autocast,
    FP8GlobalStateManager,
    get_align_size_for_quantization,
)
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
    python benchmarks/linear/benchmark_grouped_linear.py --profile --recipe nvfp4

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


# ---------------------------------------------------------------------------
# CUTLASS-vs-cuBLAS single-launch grouped GEMM comparison (NVFP4, SM100).
#
# Both backends run through the same graph-safe grouped-tensor path
# (general_grouped_gemm_for_grouped_tensor); NVTE_NVFP4_CUTLASS_GROUPED_GEMM only
# flips which single-launch kernel it dispatches to. We isolate fprop / dgrad /
# wgrad and report cuBLAS ms, CUTLASS ms and speedup, one line per config.
# wgrad is routed through CUTLASS by enabling fuse_wgrad_accumulation (fp32
# main_grad), which mirrors Megatron training.
# ---------------------------------------------------------------------------

# (num_gemms, tokens, hidden, out) MoE-shaped configs, each run balanced + imbalanced.
COMPARE_SWEEP_CONFIGS = [
    (8, 8192, 4096, 4096),
    (8, 16384, 7168, 2048),  # DeepSeek-V3-ish FC (wide K, narrow N)
    (16, 16384, 4096, 4096),
    (32, 16384, 2048, 2048),  # many small experts (launch-bound)
    (8, 65536, 7168, 2048),  # large batch
]


def _compare_set_backend(use_cutlass):
    os.environ["NVTE_NVFP4_CUTLASS_GROUPED_GEMM"] = "1" if use_cutlass else "0"


def _compare_make_m_splits(total_tokens, num_gemms, align, dist, seed=0):
    """Per-group token counts, each a multiple of ``align``, summing to total_tokens."""
    assert total_tokens % align == 0, f"tokens must be divisible by {align}"
    units = total_tokens // align
    assert units >= num_gemms, "increase tokens or lower num_gemms"
    if dist == "balanced":
        base, rem = divmod(units, num_gemms)
        u = [base + (1 if i < rem else 0) for i in range(num_gemms)]
    else:  # imbalanced: seeded skew in ~[0.25x, 1.75x] the mean (real MoE routing)
        g = torch.Generator().manual_seed(seed)
        w = (0.25 + 1.5 * torch.rand(num_gemms, generator=g)).tolist()
        s = sum(w)
        u = [max(1, int(round(units * wi / s))) for wi in w]
        i, diff = 0, units - sum(u)
        while diff != 0:
            j = i % num_gemms
            if diff > 0:
                u[j] += 1
                diff -= 1
            elif u[j] > 1:
                u[j] -= 1
                diff += 1
            i += 1
    return [x * align for x in u]


def _compare_build(num_gemms, hidden, out, dtype, fuse_wgrad=False):
    torch.manual_seed(1234)
    block = GroupedLinear(
        num_gemms,
        hidden,
        out,
        bias=False,
        params_dtype=dtype,
        device="cuda",
        fuse_wgrad_accumulation=fuse_wgrad,
    ).eval()
    if fuse_wgrad:
        for i in range(num_gemms):
            w = getattr(block, f"weight{i}")
            w.main_grad = torch.zeros_like(w, dtype=torch.float32)
    return block


def _compare_run_gemm_iter(block, x, m_splits, nvfp4_recipe, role, timed):
    """One iteration isolating a single GEMM role. Returns ms (0 if untimed)."""
    num = len(m_splits)
    if role == "wgrad":
        for i in range(num):
            getattr(block, f"weight{i}").main_grad.zero_()

    if role == "fprop":
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        if timed:
            torch.cuda.synchronize()
            start.record()
        with autocast(enabled=True, recipe=nvfp4_recipe):
            block(x, m_splits)
        if timed:
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end)
        return 0.0

    # dgrad / wgrad: forward is untimed; only the backward GEMM is measured.
    with autocast(enabled=True, recipe=nvfp4_recipe):
        out = block(x, m_splits)
    grad = torch.ones_like(out)
    if x.grad is not None:
        x.grad = None
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    if timed:
        torch.cuda.synchronize()
        start.record()
    out.backward(grad)
    if timed:
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)
    return 0.0


def _compare_measure_per_gemm(num_gemms, hidden, out, dtype, m_splits, nvfp4_recipe, iters, warmup):
    """Isolated per-GEMM timing. Returns {role: (cublas_ms, cutlass_ms)}."""
    inp = torch.randn(sum(m_splits), hidden, dtype=dtype, device="cuda")
    m_splits_t = torch.tensor(m_splits, dtype=torch.int64, device="cuda")
    res = {}
    for role in ("fprop", "dgrad", "wgrad"):
        block = _compare_build(num_gemms, hidden, out, dtype, fuse_wgrad=(role == "wgrad"))
        # dgrad needs input grad; wgrad needs weight grad; freeze the other side.
        x = inp.detach().clone().requires_grad_(role in ("fprop", "dgrad"))
        for i in range(num_gemms):
            getattr(block, f"weight{i}").requires_grad_(role == "wgrad")

        def timed_backend(use_cutlass):
            FP8GlobalStateManager.reset()
            _compare_set_backend(use_cutlass)
            for _ in range(warmup):
                _compare_run_gemm_iter(block, x, m_splits_t, nvfp4_recipe, role, timed=False)
            torch.cuda.synchronize()
            total = 0.0
            for _ in range(iters):
                total += _compare_run_gemm_iter(
                    block, x, m_splits_t, nvfp4_recipe, role, timed=True
                )
            return total / iters

        res[role] = (timed_backend(False), timed_backend(True))
    return res


def run_backend_compare_sweep(dtype, iters, warmup, seed):
    """Print one line per config with per-GEMM cuBLAS/CUTLASS ms + speedup."""
    os.environ["NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM"] = "1"
    nvfp4_recipe = NVFP4BlockScaling()
    align = get_align_size_for_quantization(nvfp4_recipe)

    print(
        "per-GEMM ms/iter -- each cell is  cuBLAS / CUTLASS / speedup  (single-launch, lower ms is"
        " better)"
    )
    gcol = 30
    hdr = (
        f"{'experts':<8}{'tokens':<8}{'hidden':<8}{'out':<7}{'dist':<12}"
        f"{'fprop(ms) cuBLAS/CUTLASS/x':>{gcol}}"
        f"{'dgrad(ms) cuBLAS/CUTLASS/x':>{gcol}}"
        f"{'wgrad(ms) cuBLAS/CUTLASS/x':>{gcol}}"
    )
    print(hdr)
    print("-" * len(hdr))
    for num_gemms, tokens, hidden, out in COMPARE_SWEEP_CONFIGS:
        for dist in ("balanced", "imbalanced"):
            m_splits = _compare_make_m_splits(tokens, num_gemms, align, dist, seed)
            res = _compare_measure_per_gemm(
                num_gemms, hidden, out, dtype, m_splits, nvfp4_recipe, iters, warmup
            )

            def cell(role):
                cub, cut = res[role]
                return f"{cub:.3f}/{cut:.3f}/{cub / cut:.2f}x"

            print(
                f"{num_gemms:<8}{tokens:<8}{hidden:<8}{out:<7}{dist:<12}"
                f"{cell('fprop'):>{gcol}}{cell('dgrad'):>{gcol}}{cell('wgrad'):>{gcol}}"
            )


def _compare_graph_step_time(
    num_gemms, hidden, out, dtype, m_splits, nvfp4_recipe, iters, warmup, use_cutlass
):
    """Time one CUDA-graphed fwd+bwd training step for a single backend.

    The whole GroupedLinear step (fprop + dgrad + wgrad) is captured with
    make_graphed_callables under the selected backend, so the kernel choice is
    baked into the graph; we then time graph replay. fuse_wgrad_accumulation
    (fp32 main_grad) routes wgrad through CUTLASS, mirroring Megatron."""
    _compare_set_backend(use_cutlass)
    FP8GlobalStateManager.reset()
    total_m = sum(m_splits)
    m_splits_t = torch.tensor(m_splits, dtype=torch.int64, device="cuda")
    block = _compare_build(num_gemms, hidden, out, dtype, fuse_wgrad=True)
    # make_graphed_callables only captures the backward graph in training mode
    # (is_training = all(c.training ...)); eval() would skip bwd capture.
    block.train()
    x = torch.randn(total_m, hidden, dtype=dtype, device="cuda", requires_grad=True)
    dy = torch.randn(total_m, out, dtype=dtype, device="cuda")

    graphed = te.make_graphed_callables(
        block,
        (x, m_splits_t),
        num_warmup_iters=3,
        enabled=True,
        recipe=nvfp4_recipe,
    )

    def step():
        for i in range(num_gemms):
            getattr(block, f"weight{i}").main_grad.zero_()
        if x.grad is not None:
            x.grad = None
        y = graphed(x, m_splits_t)
        y.backward(dy)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def run_backend_compare_graph_sweep(dtype, iters, warmup, seed):
    """Print one line per config with whole-step (fwd+bwd) CUDA-graph replay
    timing: cuBLAS vs CUTLASS single-launch."""
    os.environ["NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM"] = "1"
    nvfp4_recipe = NVFP4BlockScaling()
    align = get_align_size_for_quantization(nvfp4_recipe)

    print(
        "CUDA-graphed fwd+bwd ms/iter -- each cell is  cuBLAS / CUTLASS / speedup  "
        "(single-launch, lower ms is better)"
    )
    scol = 34
    hdr = (
        f"{'experts':<8}{'tokens':<8}{'hidden':<8}{'out':<7}{'dist':<12}"
        f"{'fwd+bwd(ms) cuBLAS/CUTLASS/speedup':>{scol}}"
    )
    print(hdr)
    print("-" * len(hdr))
    for num_gemms, tokens, hidden, out in COMPARE_SWEEP_CONFIGS:
        for dist in ("balanced", "imbalanced"):
            m_splits = _compare_make_m_splits(tokens, num_gemms, align, dist, seed)
            t_cublas = _compare_graph_step_time(
                num_gemms, hidden, out, dtype, m_splits, nvfp4_recipe, iters, warmup, False
            )
            t_cutlass = _compare_graph_step_time(
                num_gemms, hidden, out, dtype, m_splits, nvfp4_recipe, iters, warmup, True
            )
            cell = f"{t_cublas:.3f}/{t_cutlass:.3f}/{t_cublas / t_cutlass:.2f}x"
            print(f"{num_gemms:<8}{tokens:<8}{hidden:<8}{out:<7}{dist:<12}{cell:>{scol}}")


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
        if bool(int(os.getenv("NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM", "0"))):
            m_splits = torch.tensor(m_splits, dtype=torch.int64, device=device)
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
    parser.add_argument(
        "--compare-backends",
        action="store_true",
        default=False,
        help=(
            "NVFP4 only: sweep MoE shapes x {balanced, imbalanced} and report per-GEMM "
            "(fprop/dgrad/wgrad) cuBLAS vs CUTLASS single-launch timing through the "
            "graph-safe grouped-tensor path (NVTE_NVFP4_CUTLASS_GROUPED_GEMM)."
        ),
    )
    parser.add_argument("--iters", type=int, default=30, help="Timed iters for --compare-backends")
    parser.add_argument(
        "--warmup", type=int, default=10, help="Warmup iters for --compare-backends"
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for imbalanced splits")
    parser.add_argument(
        "--graph",
        action="store_true",
        default=False,
        help=(
            "With --compare-backends, time a CUDA-graphed whole-step (fwd+bwd) replay "
            "instead of eager per-GEMM timing."
        ),
    )
    args = parser.parse_args()

    if args.compare_backends:
        if not nvfp4_available:
            raise SystemExit(f"NVFP4 is not available: {reason_for_no_nvfp4}")
        if args.graph:
            run_backend_compare_graph_sweep(torch.bfloat16, args.iters, args.warmup, args.seed)
        else:
            run_backend_compare_sweep(torch.bfloat16, args.iters, args.warmup, args.seed)
        raise SystemExit(0)

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

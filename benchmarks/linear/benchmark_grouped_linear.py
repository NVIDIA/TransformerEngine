# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
import os
import statistics
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
from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
import transformer_engine_torch as tex
from contextlib import contextmanager, nullcontext
from typing import List, Optional, Tuple

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


# =============================================================================
# NVFP4 grouped GEMM backend comparison (GEMM-level): single-launch CUTLASS
# per-tensor grouped kernel vs the production multi-stream cuBLASLt per-expert
# loop. Both sit behind the SAME dispatch (nvte_multi_tensor_gemm); only the env
# NVTE_NVFP4_CUTLASS_GROUPED_GEMM selects between them, read fresh per call.
# Enabled with --compare-nvfp4-grouped-gemm. Operands are quantized ONCE
# (untimed); only the grouped GEMM is timed -- the fair backend comparison.
# Requires a Blackwell (SM100) build with the kernel/binding compiled in.
# =============================================================================
_NVFP4_GG_ENV = "NVTE_NVFP4_CUTLASS_GROUPED_GEMM"


def _has_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10


@contextmanager
def _nvfp4_gg_backend(cutlass: bool):
    """Toggle the cutlass/multi-stream env for the duration of a timing block."""
    prev = os.environ.get(_NVFP4_GG_ENV)
    os.environ[_NVFP4_GG_ENV] = "1" if cutlass else "0"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(_NVFP4_GG_ENV, None)
        else:
            os.environ[_NVFP4_GG_ENV] = prev


def _nvfp4_gg_token_counts(num_experts: int, mean_m: int, imbalanced: bool, seed: int) -> List[int]:
    """Per-expert token counts, each a multiple of 128 (the path's alignment
    contract). Balanced => all equal; imbalanced => uniform in [0.25x, 1.75x] mean."""
    if not imbalanced:
        return [mean_m] * num_experts
    g = torch.Generator().manual_seed(seed)
    lo, hi = 0.25, 1.75
    blocks_mean = max(mean_m // 128, 1)
    out = []
    for _ in range(num_experts):
        frac = lo + (hi - lo) * torch.rand(1, generator=g).item()
        blocks = max(int(round(blocks_mean * frac)), 1)
        out.append(blocks * 128)
    return out


def _nvfp4_gg_label(Ms: List[int]) -> str:
    return f"{len(Ms)}e x {'imbal' if len(set(Ms)) > 1 else Ms[0]}"


def _nvfp4_gg_quantizer() -> NVFP4Quantizer:
    """Per-tensor NVFP4 (1D, no RHT/SR/2D) so the cutlass path is eligible."""
    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
        with_random_sign_mask=False,
    )


def _nvfp4_gg_quantize(hp: torch.Tensor):
    q = _nvfp4_gg_quantizer()
    dst = q.make_empty(hp.shape, dtype=torch.bfloat16, device=hp.device)
    if hp.numel() != 0:
        tex.quantize(hp, q, dst, None)
    return dst


def _nvfp4_gg_time_us(fn, warmup: int, iters: int) -> float:
    """Median wall time of fn() in microseconds via CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: List[float] = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return statistics.median(times) * 1e3


def _nvfp4_gg_build(layout: str, Ms: List[int], N: int, K: int):
    """Returns (A, B, out, m_splits, grad, single_output, flops) for one layout.
    Operand order matches GroupedLinear: A=weight side, B=activation/grad side."""
    dev = torch.device("cuda")
    groups = len(Ms)
    m = sum(Ms)
    flops = 2.0 * m * N * K  # one logical GEMM, summed over all groups

    def qt(*sz):
        return _nvfp4_gg_quantize(torch.randn(*sz, dtype=torch.bfloat16, device=dev) * 0.5)

    if layout == "TN":  # fprop: A=W(N,K), B=X(M,K) -> out(M,N)
        A = [qt(N, K) for _ in range(groups)]
        B = [qt(mm, K) for mm in Ms]
        out = [torch.empty(m, N, dtype=torch.bfloat16, device=dev)]
        return A, B, out, Ms, False, True, flops
    if layout == "NN":  # dgrad: A=W(N,K), B=dY(M,N) -> dgrad(M,K)
        A = [qt(N, K) for _ in range(groups)]
        B = [qt(mm, N) for mm in Ms]
        out = [torch.empty(m, K, dtype=torch.bfloat16, device=dev)]
        return A, B, out, Ms, True, True, flops
    # NT wgrad: A=X(M,K), B=dY(M,N) -> wgrad(N,K), fp32 out
    A = [qt(mm, K) for mm in Ms]
    B = [qt(mm, N) for mm in Ms]
    out = [torch.empty(N, K, dtype=torch.float32, device=dev) for _ in range(groups)]
    return A, B, out, Ms, True, False, flops


def _nvfp4_gg_d_groups(layout: str, Ms: List[int], out: List[torch.Tensor]) -> List[torch.Tensor]:
    """Per-group output tensors for the direct binding. TN/NN slice a single
    packed output by tokens along dim 0; NT (wgrad) is already a per-group list."""
    if layout == "NT":
        return out
    big = out[0]
    groups, s = [], 0
    for mm in Ms:
        groups.append(big[s : s + mm])
        s += mm
    return groups


def _nvfp4_gg_bench_pure(
    layout, Ms, N, K, warmup, iters
) -> Tuple[Optional[float], Optional[float]]:
    """Fair pure-GEMM comparison. Builds FRESH operands and pre-swizzles their
    scales IN PLACE (untimed), then times BOTH backends on the SAME pre-swizzled
    operands so the per-call scale swizzle is excluded from both timers equally:
      * multi-stream-pure : general_grouped_gemm (env=0). The dispatch skips the
        (already done) swizzle; the per-expert alpha is still recomputed inside
        cuBLASLt, but on 4 streams it overlaps the GEMMs (~hidden).
      * cutlass-pure      : the direct binding with alpha precomputed (untimed) --
        the single grouped launch in isolation.
    Returns (multistream_pure_us, cutlass_pure_us). Either may be None.
    """
    if not hasattr(tex, "nvfp4_grouped_per_tensor_gemm"):
        return None, None  # binding not compiled in (stale build)
    try:
        A, B, out, m_splits, grad, single_output, _flops = _nvfp4_gg_build(layout, Ms, N, K)
        transa = layout[0] == "T"
        transb = layout[1] == "T"
        d_groups = _nvfp4_gg_d_groups(layout, Ms, out)
        # Pre-swizzle exactly as te_general_grouped_gemm does before the GEMM:
        #   A -> (rowwise=transa, columnwise=!transa); B -> (rowwise=!transb, columnwise=transb).
        tex.multi_tensor_swizzle_scales_for_gemm_(A, transa, not transa)
        tex.multi_tensor_swizzle_scales_for_gemm_(B, not transb, transb)
        alpha = tex.nvfp4_grouped_per_tensor_compute_alpha(A, transa, B, transb)
    except Exception:  # noqa: BLE001 -- bench: missing kernel -> drop the columns
        return None, None

    def ms_pure() -> Optional[float]:
        try:
            with _nvfp4_gg_backend(False):  # multi-stream, on pre-swizzled operands
                return _nvfp4_gg_time_us(
                    lambda: general_grouped_gemm(
                        A,
                        B,
                        out,
                        [None] * len(Ms),
                        out[0].dtype,
                        layout=layout,
                        m_splits=m_splits,
                        single_output=single_output,
                        grad=grad,
                    ),
                    warmup,
                    iters,
                )
        except Exception:  # noqa: BLE001
            return None

    def cu_pure() -> Optional[float]:
        try:
            return _nvfp4_gg_time_us(
                lambda: tex.nvfp4_grouped_per_tensor_gemm(
                    A, transa, B, transb, d_groups, [], alpha, False
                ),
                warmup,
                iters,
            )
        except Exception:  # noqa: BLE001
            return None

    return ms_pure(), cu_pure()


def _nvfp4_gg_bench(
    layout, Ms, N, K, warmup, iters, pure: bool = True
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], float, str]:
    """Returns (multistream_us, cutlass_us, multistream_pure_us, cutlass_pure_us, flops, note)."""
    try:
        A, B, out, m_splits, grad, single_output, flops = _nvfp4_gg_build(layout, Ms, N, K)
    except Exception as exc:  # noqa: BLE001
        return None, None, None, None, 0.0, f"ERROR(build): {type(exc).__name__}: {str(exc)[:100]}"

    def run():
        general_grouped_gemm(
            A,
            B,
            out,
            [None] * len(Ms),
            out[0].dtype,
            layout=layout,
            m_splits=m_splits,
            single_output=single_output,
            grad=grad,
        )

    def timed(cutlass: bool) -> Optional[float]:
        try:
            with _nvfp4_gg_backend(cutlass):
                return _nvfp4_gg_time_us(run, warmup, iters)
        except Exception:  # noqa: BLE001
            return None

    ms_us, cu_us = timed(False), timed(True)
    # Pure path uses FRESH operands: its in-place pre-swizzle must not strip the
    # per-call swizzle that the dispatch columns above legitimately pay.
    ms_pure_us, cu_pure_us = (
        _nvfp4_gg_bench_pure(layout, Ms, N, K, warmup, iters) if pure else (None, None)
    )
    return ms_us, cu_us, ms_pure_us, cu_pure_us, flops, ""


def run_nvfp4_grouped_gemm_comparison(layouts, configs, warmup, iters, want_pure) -> None:
    """Driver: print a dispatch row (cutlass vs multi-stream, both via dispatch) and,
    if want_pure, a fair PURE row (both pre-swizzled) for each layout x config."""
    if not _has_sm100():
        print("SKIP: NVFP4 grouped GEMM comparison requires SM100 (Blackwell)")
        return
    if not nvfp4_available:
        print(f"SKIP: NVFP4 not available ({reason_for_no_nvfp4})")
        return

    layout_label = {"TN": "TN fprop", "NN": "NN dgrad", "NT": "NT wgrad"}
    pure_hdr = (
        ""
        if not want_pure
        else (
            "  + PURE row (fair kernel-vs-kernel): both pre-swizzled, swizzle excluded from both.\n"
        )
    )
    print(
        "\nNVFP4 grouped GEMM: CUTLASS vs multi-stream cuBLASLt  "
        f"[warmup={warmup} iters={iters}]\n"
        "  DISPATCH row (real prod): multi-stream = env=0 (4-stream cuBLASLt), cutlass = env=1.\n"
        f"{pure_hdr}"
        "  speedup = multi-stream / cutlass (>1 => cutlass faster).\n"
    )

    def _ms(us: Optional[float]) -> str:
        return f"{us / 1e3:.3f}ms" if (us is not None) else "-"

    def _spd(num: Optional[float], den: Optional[float]) -> str:
        return f"{num / den:.2f}x" if (num and den and den > 0) else "-"

    header = (
        f"  {'shape':<12} {'N':>5} {'K':>5} {'tok':>6}  {'row':<9} "
        f"{'cutlass':>10}  {'multi-stream':>12}  {'speedup':>8}"
    )

    def _emit(shape, n, k, tok, row, cu_us, ms_us):
        print(
            f"  {shape:<12} {n:>5} {k:>5} {tok:>6}  {row:<9} "
            f"{_ms(cu_us):>10}  {_ms(ms_us):>12}  {_spd(ms_us, cu_us):>8}"
        )

    for layout in layouts:
        print(f"  [{layout_label[layout]}]")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for Ms, N, K in configs:
            ms_us, cu_us, ms_pure_us, cu_pure_us, _flops, note = _nvfp4_gg_bench(
                layout, Ms, N, K, warmup, iters, pure=want_pure
            )
            _emit(_nvfp4_gg_label(Ms), N, K, sum(Ms), "dispatch", cu_us, ms_us)
            if want_pure:
                _emit("", "", "", "", "pure", cu_pure_us, ms_pure_us)
            if note:
                print(f"      {note}")
        print()

    print(
        "  DISPATCH row = real prod path for both backends; both pay per-call swizzle +\n"
        "  per-expert alpha in the timer (cutlass runs the alpha kernels serially, multi-\n"
        "  stream overlaps them across 4 streams -- so it may under-sell cutlass until the\n"
        "  alpha launches are batched in a follow-up PR). PURE row = fair kernel-vs-kernel:\n"
        "  operands pre-swizzled (untimed) for BOTH; cutlass-pure also precomputes alpha\n"
        "  (untimed) and times only tex.nvfp4_grouped_per_tensor_gemm. If the PURE row is\n"
        "  blank, the kernel/binding is not compiled in -- rebuild on Blackwell. If the\n"
        "  DISPATCH speedup reads ~1.00x everywhere, env=1 is silently falling back to\n"
        "  multi-stream (also a stale build).\n"
    )


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
    # NVFP4 grouped GEMM backend comparison (GEMM-level, cutlass vs multi-stream).
    parser.add_argument(
        "--compare-nvfp4-grouped-gemm",
        action="store_true",
        help="GEMM-level NVFP4 cutlass vs multi-stream cuBLASLt comparison (then exit)",
    )
    parser.add_argument(
        "--layouts",
        nargs="+",
        default=["TN", "NN", "NT"],
        choices=["TN", "NN", "NT"],
        help="layouts for --compare-nvfp4-grouped-gemm (TN fprop, NN dgrad, NT wgrad)",
    )
    parser.add_argument(
        "--no-pure",
        action="store_true",
        help="drop the fair PURE row in --compare-nvfp4-grouped-gemm",
    )
    parser.add_argument("--gemm-warmup", type=int, default=10)
    parser.add_argument("--gemm-iters", type=int, default=100)
    args = parser.parse_args()

    jagged_input_splits = None
    if args.jagged_input is not None:
        jagged_input_splits = [int(x) for x in args.jagged_input.split(",")]
        print(f"Jagged input splits: {jagged_input_splits}")
        print(f"Jagged input splits sum: {sum(jagged_input_splits)}")
        print(f"Jagged input splits num_gemms: {len(jagged_input_splits)}")

    # GEMM-level NVFP4 cutlass-vs-multi-stream comparison (separate from the module
    # benchmark below). Honors --jagged-input / --hidden-dim / --output-dim as a
    # single custom config; otherwise uses a built-in MoE-shaped config sweep.
    if args.compare_nvfp4_grouped_gemm:
        if jagged_input_splits is not None:
            # The per-tensor cutlass path requires tokens % 128 == 0; align up so the
            # path is eligible (a non-aligned split would just fall back to cuBLAS).
            Ms = [max((s + 127) // 128, 1) * 128 for s in jagged_input_splits]
            gg_configs = [(Ms, args.output_dim, args.hidden_dim)]
        else:
            gg_configs = [
                (_nvfp4_gg_token_counts(8, 128, False, 0), 2048, 2048),  # small (launch-bound)
                (_nvfp4_gg_token_counts(8, 256, False, 0), 2048, 2048),
                (_nvfp4_gg_token_counts(8, 512, False, 0), 2048, 2048),
                (_nvfp4_gg_token_counts(8, 256, True, 1), 2048, 2048),  # imbalanced
                (_nvfp4_gg_token_counts(16, 256, False, 0), 2048, 2048),
                (_nvfp4_gg_token_counts(16, 256, True, 2), 4096, 2048),  # imbalanced, wider N
                (_nvfp4_gg_token_counts(32, 128, False, 0), 2048, 2048),  # many small experts
                (_nvfp4_gg_token_counts(32, 256, True, 3), 2048, 2048),  # many imbalanced
            ]
        run_nvfp4_grouped_gemm_comparison(
            args.layouts, gg_configs, args.gemm_warmup, args.gemm_iters, not args.no_pure
        )
        raise SystemExit(0)

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

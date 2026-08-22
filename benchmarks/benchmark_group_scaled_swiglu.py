#!/usr/bin/env python3
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark for the grouped scaled-SwiGLU MXFP8 kernel (nvte_group_scaled_swiglu).

Compares the fused kernel against the unfused path it replaces when recomputing
the MoE FC2 weight-gradient input:

    fused   : [T, 2F] bf16 --(scaled SwiGLU + columnwise MXFP8)--> [T, F] fp8 + scales
    unfused : [T, 2F] bf16 --(scaled SwiGLU)--> [T, F] bf16 --(group_quantize)--> fp8

Both paths are bandwidth bound, so the number that explains the speedup is DRAM
traffic. Per output element the unfused path moves 4 bytes reading [T, 2F], 2
writing the bf16 activation, 2 reading it back, and 1 writing FP8; the fused
kernel keeps the activation in registers and moves 4 + 1. That is 9 vs 5 bytes
per element, so ~1.8x is the ceiling for the fused kernel.

Every unfused variant ends in the same ``tex.group_quantize`` call with the same
quantizer the fused path uses (columnwise only), so the variants differ *only* in how
the activation is computed. That choice changes what the speedup means:

  unfused-eager    plain PyTorch ops, one kernel and one DRAM round trip per
                   elementwise op. Much too loose to quote. It is kept because it is
                   the only way to see that torch.compile actually fused: a silent
                   Inductor fallback would drag unfused-compiled toward this number
                   and quietly inflate the reported speedup.
  unfused-compiled the same expression through torch.compile, which fuses it into one
                   elementwise kernel. The tightest unfused implementation, so this is
                   the number to quote: against it, the fused kernel's remaining
                   advantage can only come from fusing the *quantization* in.
  unfused-te-op    the same two steps assembled from TE's existing components, i.e.
                   what a user falls back on today without this kernel. It is slower
                   than unfused-compiled for a structural reason rather than just
                   Python overhead: ScaledSwiGLU runs tex.swiglu and then a *separate*
                   kernel to apply the per-token scale, so the bf16 intermediate makes
                   one extra DRAM round trip. Operation-fuser overhead adds to that.
  fused-clamped    the clamped instantiation of the same fused kernel.
  unfused-compiled-clamped
                   the clamped expression through torch.compile, and the only fair
                   baseline for fused-clamped: unfused-compiled computes plain SwiGLU,
                   so timing the clamped kernel against it would credit the kernel for
                   arithmetic the baseline never performed.

Shapes must respect the kernel's restrictions: every expert's token count is
divisible by 128, and the GEMM-swizzled scale layout also needs F divisible by
128. The default total token count mirrors benchmark_group_quantize_current_scaling.py.

Example:
    python benchmarks/benchmark_group_scaled_swiglu.py
    python benchmarks/benchmark_group_scaled_swiglu.py --hidden 4096 --num-groups 8 64
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

# IMPORTANT: import transformer_engine before torch to avoid cublasLt symbol-resolution
# issues caused by torch's bundled CUDA libs.
import transformer_engine.pytorch  # noqa: F401  - registers extension
from transformer_engine.pytorch import MXFP8Quantizer
import transformer_engine.pytorch.ops as te_ops
import transformer_engine_torch as tex
import torch


BF16_BYTES = 2
FP8_BYTES = 1
# One e8m0 exponent per 32-row block of every column.
SCALE_BLOCK_ROWS = 32
# The kernel schedules 128-row blocks, so every expert's token count must be a multiple.
TOKEN_ALIGNMENT = 128
# The swizzled scale layout tiles the transposed scale matrix 128-wide along F.
SWIZZLE_F_ALIGNMENT = 128

VARIANTS = (
    "fused",
    "fused-clamped",
    "unfused-eager",
    "unfused-compiled",
    "unfused-compiled-clamped",
    "unfused-te-op",
)


@dataclass
class CaseResult:
    variant: str
    tokens: int
    hidden: int
    num_groups: int
    swizzled_scales: bool
    loop: str
    iters: int
    per_iter_us: float
    min_bytes: int
    bw_TBps: float
    speedup_vs_fused: Optional[float] = None


def _distribute_blocks(blocks: int, num_groups: int, imbalance: str) -> List[int]:
    """Split ``blocks`` 128-row blocks across ``num_groups`` experts, each >= 1."""
    if imbalance == "uniform":
        if blocks % num_groups != 0:
            raise SystemExit(
                f"{blocks} blocks of {TOKEN_ALIGNMENT} tokens do not divide evenly across"
                f" {num_groups} experts; pick a --tokens that is a multiple of"
                f" {TOKEN_ALIGNMENT * num_groups} or use --imbalance zipf."
            )
        return [blocks // num_groups] * num_groups

    if imbalance == "mild":
        weights = [0.8 + 0.4 * i / max(1, num_groups - 1) for i in range(num_groups)]
    elif imbalance == "zipf":
        weights = [1.0 / ((i + 1) ** 0.7) for i in range(num_groups)]
    else:
        raise SystemExit(f"unknown imbalance={imbalance}")

    total_weight = sum(weights)
    counts = [max(1, int(round(w * blocks / total_weight))) for w in weights]

    # Fix up rounding so the blocks sum back to the requested total, always keeping
    # every expert at >= 1 block.
    order = sorted(range(num_groups), key=lambda i: counts[i], reverse=True)
    idx = 0
    while sum(counts) != blocks:
        target = order[idx % num_groups]
        if sum(counts) < blocks:
            counts[target] += 1
        elif counts[target] > 1:
            counts[target] -= 1
        idx += 1
    return counts


def _make_first_dims(tokens: int, num_groups: int, imbalance: str) -> List[int]:
    if tokens % TOKEN_ALIGNMENT != 0:
        raise SystemExit(f"--tokens must be a multiple of {TOKEN_ALIGNMENT}, got {tokens}")
    blocks = _distribute_blocks(tokens // TOKEN_ALIGNMENT, num_groups, imbalance)
    return [b * TOKEN_ALIGNMENT for b in blocks]


def _make_quantizer(swizzled_scales: bool) -> MXFP8Quantizer:
    quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)
    # The FC2 wgrad GEMM only consumes the columnwise operand.
    quantizer.set_usage(rowwise=False, columnwise=True)
    quantizer.optimize_for_gemm = swizzled_scales
    return quantizer


def _scaled_swiglu_bf16(input_2f: torch.Tensor, prob: torch.Tensor, hidden: int) -> torch.Tensor:
    act = input_2f[:, :hidden]
    gate = input_2f[:, hidden:]
    return torch.nn.functional.silu(act) * gate * prob.unsqueeze(1)


def _scaled_clamped_swiglu_bf16(
    input_2f: torch.Tensor,
    prob: torch.Tensor,
    hidden: int,
    limit: float,
    alpha: float,
    glu_linear_offset: float,
) -> torch.Tensor:
    # Mirrors the kernel: the activation half is clamped from above only, the gate half
    # on both sides and then offset. Written with an explicit sigmoid because the alpha
    # inside it is what distinguishes clamped_silu from silu.
    act = input_2f[:, :hidden].clamp(max=limit)
    gate = input_2f[:, hidden:].clamp(-limit, limit) + glu_linear_offset
    return act * torch.sigmoid(alpha * act) * gate * prob.unsqueeze(1)


def _fused_bytes(tokens: int, hidden: int) -> int:
    read_input = tokens * 2 * hidden * BF16_BYTES
    write_fp8 = tokens * hidden * FP8_BYTES
    write_scales = (tokens // SCALE_BLOCK_ROWS) * hidden
    return read_input + write_fp8 + write_scales


def _unfused_bytes(tokens: int, hidden: int) -> int:
    # Activation kernel: read [T, 2F] bf16, write the [T, F] bf16 intermediate.
    activation = tokens * 2 * hidden * BF16_BYTES + tokens * hidden * BF16_BYTES
    # Quantize kernel: read that intermediate back, write FP8 plus scales.
    quantize = (
        tokens * hidden * BF16_BYTES
        + tokens * hidden * FP8_BYTES
        + (tokens // SCALE_BLOCK_ROWS) * hidden
    )
    return activation + quantize


def _compile_activation() -> Optional[Callable]:
    try:
        return torch.compile(_scaled_swiglu_bf16, dynamic=False)
    except Exception as exc:  # torch.compile is optional for this benchmark
        print(f"  (torch.compile unavailable, skipping unfused-compiled: {exc})")
        return None


def _compile_clamped_activation(clamp: Tuple[float, float, float]) -> Optional[Callable]:
    """torch.compile the clamped expression with the clamp constants closed over.

    Baking them in keeps the callable's signature identical to the plain activation's,
    so both go through the same unfused runner.
    """
    limit, alpha, glu_linear_offset = clamp

    def clamped(input_2f: torch.Tensor, prob: torch.Tensor, hidden: int) -> torch.Tensor:
        return _scaled_clamped_swiglu_bf16(input_2f, prob, hidden, limit, alpha, glu_linear_offset)

    try:
        return torch.compile(clamped, dynamic=False)
    except Exception as exc:  # torch.compile is optional for this benchmark
        print(f"  (torch.compile unavailable, skipping unfused-compiled-clamped: {exc})")
        return None


def _te_op_activation() -> Optional[Callable]:
    """TE's own ScaledSwiGLU as the activation half of the unfused path.

    ``glu_interleave_size=None`` keeps the contiguous ``[act | gate]`` layout the
    fused kernel expects; the fused grouped MLP instead runs this op with 32-wide
    interleaving, which would make the comparison a layout difference rather than a
    fusion one. The op routes through the operation fuser and autograd, so it also
    carries framework overhead that is not kernel time.
    """
    try:
        op = te_ops.ScaledSwiGLU(glu_interleave_size=None)
    except Exception as exc:
        print(f"  (te_ops.ScaledSwiGLU unavailable, skipping unfused-te-op: {exc})")
        return None

    def activation(input_2f: torch.Tensor, prob: torch.Tensor, hidden: int) -> torch.Tensor:
        del hidden  # the op infers F from the input
        with torch.no_grad():
            return op(input_2f, prob)

    return activation


def _make_runner(
    variant: str,
    quantizer: MXFP8Quantizer,
    hidden: int,
    num_groups: int,
    first_dims: Optional[torch.Tensor],
    compiled_activation: Optional[Callable],
    compiled_clamped_activation: Optional[Callable],
    te_op_activation: Optional[Callable],
    clamp: Tuple[float, float, float],
) -> Callable[[torch.Tensor, torch.Tensor], object]:
    if variant == "fused":
        return lambda x, prob: tex.group_scaled_swiglu(x, prob, quantizer, num_groups, first_dims)

    if variant == "fused-clamped":
        limit, alpha, glu_linear_offset = clamp
        return lambda x, prob: tex.group_scaled_clamped_swiglu(
            x, prob, quantizer, num_groups, limit, alpha, glu_linear_offset, first_dims
        )

    activation = {
        "unfused-eager": _scaled_swiglu_bf16,
        "unfused-compiled": compiled_activation,
        "unfused-compiled-clamped": compiled_clamped_activation,
        "unfused-te-op": te_op_activation,
    }[variant]

    def run(x: torch.Tensor, prob: torch.Tensor):
        intermediate = activation(x, prob, hidden)
        return tex.group_quantize(intermediate, quantizer, num_groups, first_dims)

    return run


def _time_eager(runner, inputs, probs, iters: int) -> float:
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for it in range(iters):
        i = it % len(inputs)
        runner(inputs[i], probs[i])
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def _time_graph(runner, inputs, probs, iters: int, calls_per_replay: int = 16):
    """Capture ``calls_per_replay`` calls into one CUDA graph and replay it.

    Removes Python and launch overhead, which is what makes a single memory-bound
    kernel look slower than it is.
    """
    static_x, static_prob = inputs[0], probs[0]

    # Warmup on a side stream before capture, per the torch CUDA-graph docs.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            runner(static_x, static_prob)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls_per_replay):
            runner(static_x, static_prob)

    replays = max(1, iters // calls_per_replay)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end), replays * calls_per_replay


def _check_fused_matches_unfused(
    quantizer: MXFP8Quantizer,
    input_2f: torch.Tensor,
    prob: torch.Tensor,
    hidden: int,
    num_groups: int,
    first_dims: Optional[torch.Tensor],
    *,
    fused_fn: Callable,
    activation: Callable,
    label: str,
) -> None:
    """Guard against timing a kernel that is not computing the right thing.

    This is a smoke test, not a numerics test: correctness lives in
    tests/cpp/operator/test_cast_mxfp8_grouped_scaled_swiglu.cu, whose CPU reference
    mirrors the kernel's arithmetic order exactly. Here the reference is plain
    PyTorch, which rounds to bf16 three times (after silu, after the gate multiply,
    after the prob multiply) where the kernel rounds once, for ~0.2% of relative
    spread on the pre-quantization values.

    MXFP8 turns that spread into whole-block disagreements: each 32-row block shares
    one e8m0 scale, and e8m0 is a pure power of two, so a block whose amax sits within
    ~0.2% of a power-of-two boundary can pick a different exponent in the two paths
    and shift all 32 of its codes at once. That affects on the order of
    0.002/ln(2) ~ 0.3% of blocks, hence a similar fraction of elements. The budget
    below sits above that but far below the ~100% a wrong formula would produce.
    """
    fused = fused_fn(input_2f, prob)
    reference = tex.group_quantize(
        activation(input_2f, prob, hidden), quantizer, num_groups, first_dims
    )
    if fused.columnwise_data.numel() != reference.columnwise_data.numel():
        raise RuntimeError(
            f"{label}: fused output has {fused.columnwise_data.numel()} elements but the"
            f" reference has {reference.columnwise_data.numel()}; the benchmark is comparing"
            " different shapes."
        )
    if int(fused.columnwise_data.view(torch.uint8).max().item()) == 0:
        raise RuntimeError(f"{label}: fused output is entirely zero; the kernel produced nothing.")

    # Coarse code-level comparison. Signs agree between the two paths in practice, so
    # treating the FP8 bytes as integers is good enough to spot a gross mismatch.
    fused_codes = fused.columnwise_data.view(torch.uint8).to(torch.int16)
    reference_codes = reference.columnwise_data.view(torch.uint8).to(torch.int16)
    mismatch = (fused_codes - reference_codes).abs() > 1
    mismatch_rate = float(mismatch.sum().item()) / max(1, mismatch.numel())
    if mismatch_rate > 2e-2:
        raise RuntimeError(
            f"{label}: fused output disagrees with the unfused reference on"
            f" {100.0 * mismatch_rate:.3f}% of FP8 codes by more than 1 ULP, which is too much"
            " to be block-scale rounding; the benchmark would be timing the wrong computation."
        )


def run_case(
    variant: str,
    *,
    tokens: int,
    hidden: int,
    num_groups: int,
    swizzled_scales: bool,
    same_shape: bool,
    imbalance: str,
    num_buffers: int,
    warmup: int,
    iters: int,
    loop: str,
    compiled_activation: Optional[Callable],
    compiled_clamped_activation: Optional[Callable],
    te_op_activation: Optional[Callable],
    clamp: Tuple[float, float, float],
) -> Optional[CaseResult]:
    quantizer = _make_quantizer(swizzled_scales)
    first_dims = None
    if not same_shape:
        first_dims = torch.tensor(
            _make_first_dims(tokens, num_groups, imbalance), dtype=torch.int64, device="cuda"
        )

    inputs = [
        torch.randn(tokens, 2 * hidden, dtype=torch.bfloat16, device="cuda")
        for _ in range(num_buffers)
    ]
    probs = [torch.rand(tokens, dtype=torch.bfloat16, device="cuda") for _ in range(num_buffers)]

    runner = _make_runner(
        variant,
        quantizer,
        hidden,
        num_groups,
        first_dims,
        compiled_activation,
        compiled_clamped_activation,
        te_op_activation,
        clamp,
    )

    for it in range(warmup):
        i = it % len(inputs)
        runner(inputs[i], probs[i])
    torch.cuda.synchronize()

    if loop == "graph":
        try:
            elapsed_ms, actual_iters = _time_graph(runner, inputs, probs, iters)
        except Exception as exc:  # capture can fail e.g. for a compiled activation
            print(f"  skipping {variant} under CUDA-graph capture: {exc}")
            return None
    else:
        elapsed_ms = _time_eager(runner, inputs, probs, iters)
        actual_iters = iters

    min_bytes = (
        _fused_bytes(tokens, hidden)
        if variant.startswith("fused")
        else _unfused_bytes(tokens, hidden)
    )
    per_iter_us = elapsed_ms * 1000.0 / actual_iters
    bw_TBps = min_bytes / (per_iter_us * 1.0e-6) / 1.0e12

    return CaseResult(
        variant=variant,
        tokens=tokens,
        hidden=hidden,
        num_groups=num_groups,
        swizzled_scales=swizzled_scales,
        loop=loop,
        iters=actual_iters,
        per_iter_us=per_iter_us,
        min_bytes=min_bytes,
        bw_TBps=bw_TBps,
    )


def _print_table(results: List[CaseResult]) -> None:
    header = (
        f"{'T x F':>14s} {'experts':>7s} {'scales':>8s} {'loop':>5s} "
        f"{'variant':16s} {'per_iter_us':>11s} {'min_GB/iter':>11s} "
        f"{'BW_TB/s':>8s} {'vs fused':>9s}"
    )
    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        shape = f"{r.tokens}x{r.hidden}"
        scales = "swizzled" if r.swizzled_scales else "compact"
        speedup = f"{r.speedup_vs_fused:.2f}x" if r.speedup_vs_fused is not None else "-"
        print(
            f"{shape:>14s} {r.num_groups:7d} {scales:>8s} {r.loop:>5s} "
            f"{r.variant:16s} {r.per_iter_us:11.2f} {r.min_bytes / 1e9:11.3f} "
            f"{r.bw_TBps:8.2f} {speedup:>9s}"
        )
    print("-" * len(header))
    print(
        "min_GB/iter = minimum DRAM traffic the path must move (reads + writes);"
        " BW = min_GB/iter / per_iter_us."
    )
    print(
        "min_GB/iter assumes one activation kernel plus one quantize kernel, so the"
        " variants that use more than that read below their real bandwidth:"
        " unfused-eager launches one kernel per elementwise op, and unfused-te-op"
        " applies the per-token scale in a separate kernel (one extra round trip of"
        " the bf16 intermediate) on top of op-fuser overhead."
    )
    print(
        "unfused-compiled is the tightest baseline; quote the speedup against it."
        " unfused-te-op is what TE's existing components give you today."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens",
        type=int,
        default=98304,
        help="Total tokens T summed over experts (multiple of 128). Default 98304.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=2048,
        help="MoE intermediate size F; the input is [T, 2F]. Default 2048.",
    )
    parser.add_argument("--num-groups", type=int, nargs="+", default=[16, 64])
    parser.add_argument(
        "--imbalance",
        choices=("uniform", "mild", "zipf"),
        default="uniform",
        help="Token distribution across experts. Default uniform.",
    )
    parser.add_argument(
        "--same-shape",
        action="store_true",
        help=(
            "Use the SAME_BOTH_DIMS layout (no first_dims). The GEMM-swizzled scale"
            " layout does not support it with more than one expert."
        ),
    )
    parser.add_argument(
        "--scales",
        choices=("compact", "swizzled", "both"),
        default="both",
        help="Scale layout to benchmark. Default both.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS),
        help=f"Subset of {VARIANTS}.",
    )
    parser.add_argument(
        "--loop",
        choices=("eager", "graph", "both"),
        default="both",
        help="Python loop, replayed CUDA graph, or both. Default both.",
    )
    parser.add_argument(
        "--clamp",
        type=float,
        nargs=3,
        metavar=("LIMIT", "ALPHA", "OFFSET"),
        default=(7.0, 1.702, 1.0),
        help="limit, alpha, glu_linear_offset for the fused-clamped variant.",
    )
    parser.add_argument("--num-buffers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    for variant in args.variants:
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant={variant}")
    if args.tokens % TOKEN_ALIGNMENT != 0:
        raise SystemExit(f"--tokens must be a multiple of {TOKEN_ALIGNMENT}, got {args.tokens}")

    scale_layouts = {
        "compact": [False],
        "swizzled": [True],
        "both": [False, True],
    }[args.scales]
    loop_modes = ("eager", "graph") if args.loop == "both" else (args.loop,)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Config: T={args.tokens}, F={args.hidden}, experts={args.num_groups},"
        f" imbalance={args.imbalance},"
        f" layout={'SAME_BOTH_DIMS' if args.same_shape else 'VARYING_FIRST_DIM'},"
        f" iters={args.iters}, warmup={args.warmup}"
    )

    compiled_activation = _compile_activation() if "unfused-compiled" in args.variants else None
    compiled_clamped_activation = (
        _compile_clamped_activation(tuple(args.clamp))
        if "unfused-compiled-clamped" in args.variants
        else None
    )
    te_op_activation = _te_op_activation() if "unfused-te-op" in args.variants else None

    # Correctness up front: a benchmark of a wrong kernel is worthless, and a speedup
    # against a baseline computing something else is worse than worthless.
    check_tokens = min(args.tokens, 4096)
    check_quantizer = _make_quantizer(False)
    check_x = torch.randn(check_tokens, 2 * args.hidden, dtype=torch.bfloat16, device="cuda")
    check_prob = torch.rand(check_tokens, dtype=torch.bfloat16, device="cuda")
    _check_fused_matches_unfused(
        check_quantizer,
        check_x,
        check_prob,
        args.hidden,
        num_groups=1,
        first_dims=None,
        fused_fn=lambda x, prob: tex.group_scaled_swiglu(x, prob, check_quantizer, 1, None),
        activation=_scaled_swiglu_bf16,
        label="plain",
    )
    if "fused-clamped" in args.variants and "unfused-compiled-clamped" in args.variants:
        limit, alpha, glu_linear_offset = tuple(args.clamp)
        _check_fused_matches_unfused(
            check_quantizer,
            check_x,
            check_prob,
            args.hidden,
            num_groups=1,
            first_dims=None,
            fused_fn=lambda x, prob: tex.group_scaled_clamped_swiglu(
                x, prob, check_quantizer, 1, limit, alpha, glu_linear_offset, None
            ),
            activation=lambda x, prob, hidden: _scaled_clamped_swiglu_bf16(
                x, prob, hidden, limit, alpha, glu_linear_offset
            ),
            label="clamped",
        )

    results: List[CaseResult] = []
    for num_groups in args.num_groups:
        for swizzled_scales in scale_layouts:
            if swizzled_scales and args.hidden % SWIZZLE_F_ALIGNMENT != 0:
                print(
                    f"  skipping swizzled scales: F={args.hidden} is not a multiple of"
                    f" {SWIZZLE_F_ALIGNMENT}"
                )
                continue
            if swizzled_scales and args.same_shape and num_groups > 1:
                print(
                    "  skipping swizzled scales with SAME_BOTH_DIMS and multiple experts"
                    " (unsupported: each expert needs its own swizzled block)"
                )
                continue
            for loop in loop_modes:
                fused_us = None
                group_results: List[CaseResult] = []
                for variant in args.variants:
                    if variant == "unfused-compiled" and compiled_activation is None:
                        continue
                    if (
                        variant == "unfused-compiled-clamped"
                        and compiled_clamped_activation is None
                    ):
                        continue
                    if variant == "unfused-te-op" and te_op_activation is None:
                        continue
                    result = run_case(
                        variant,
                        tokens=args.tokens,
                        hidden=args.hidden,
                        num_groups=num_groups,
                        swizzled_scales=swizzled_scales,
                        same_shape=args.same_shape,
                        imbalance=args.imbalance,
                        num_buffers=args.num_buffers,
                        warmup=args.warmup,
                        iters=args.iters,
                        loop=loop,
                        compiled_activation=compiled_activation,
                        compiled_clamped_activation=compiled_clamped_activation,
                        te_op_activation=te_op_activation,
                        clamp=tuple(args.clamp),
                    )
                    if result is None:
                        continue
                    if variant == "fused":
                        fused_us = result.per_iter_us
                    group_results.append(result)

                if fused_us:
                    for result in group_results:
                        result.speedup_vs_fused = result.per_iter_us / fused_us
                results.extend(group_results)

    _print_table(results)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump([r.__dict__ for r in results], f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()

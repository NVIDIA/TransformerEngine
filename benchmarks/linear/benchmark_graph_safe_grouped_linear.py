# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark MXFP8 graph-safe grouped MLP.

This mirrors ``benchmark_grouped_linear.py`` but targets the graph-safe TE ops
path used by grouped MLP:

    GroupedLinear -> ScaledSwiGLU -> GroupedLinear

The benchmark intentionally uses CUDA-device ``m_splits`` and MXFP8 only.

Example:

    python benchmarks/linear/benchmark_graph_safe_grouped_linear.py

Forward-only:

    python benchmarks/linear/benchmark_graph_safe_grouped_linear.py --fwd-only

Nsight Systems:

    (optionally: unset DEBUGINFOD_URLS)

    nsys profile \
        --output=./benchmarks/linear/graph_safe_grouped_linear_mxfp8 \
        --force-overwrite true \
        --trace=cuda,nvtx,cudnn,cublas \
        python benchmarks/linear/benchmark_graph_safe_grouped_linear.py --profile
"""

# Match the Qwen MXFP8 SFT launch toggles before importing TE.
import os

os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
os.environ.setdefault("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
os.environ.setdefault("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")

import argparse
from contextlib import nullcontext

import pandas as pd
import torch
import torch.utils.benchmark as benchmark

import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

MXFP8_AVAILABLE, REASON_FOR_NO_MXFP8 = FP8GlobalStateManager.is_mxfp8_available()


def parse_int_list(value: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(x) for x in value.split(",") if x]


def make_uniform_splits(total_tokens: int, num_groups: int) -> list[int]:
    """Split tokens uniformly across groups."""
    if total_tokens % num_groups != 0:
        raise ValueError(
            "Uniform split requires total_tokens divisible by num_groups, "
            f"got total_tokens={total_tokens}, num_groups={num_groups}"
        )
    return [total_tokens // num_groups] * num_groups


def build_grouped_mlp(
    *,
    num_groups: int,
    hidden_dim: int,
    ffn_hidden_dim: int,
    dtype: torch.dtype,
    single_grouped_weight: bool,
    accumulate_into_main_grad: bool,
    glu_interleave_size: int,
) -> te_ops.Sequential:
    """Build graph-safe grouped MLP ops sequence."""
    recipe = MXFP8BlockScaling()
    with te.quantized_model_init(enabled=True, recipe=recipe):
        fc1 = te_ops.GroupedLinear(
            num_groups,
            hidden_dim,
            2 * ffn_hidden_dim,
            bias=False,
            device="cuda",
            dtype=dtype,
            single_grouped_weight=single_grouped_weight,
            accumulate_into_main_grad=accumulate_into_main_grad,
        )
        fc2 = te_ops.GroupedLinear(
            num_groups,
            ffn_hidden_dim,
            hidden_dim,
            bias=False,
            device="cuda",
            dtype=dtype,
            single_grouped_weight=single_grouped_weight,
            accumulate_into_main_grad=accumulate_into_main_grad,
        )
        return te_ops.Sequential(
            fc1,
            te_ops.ScaledSwiGLU(glu_interleave_size=glu_interleave_size),
            fc2,
        )


def init_main_grads(module: torch.nn.Module, value: float = 0.0) -> None:
    """Initialize Megatron-style main_grad buffers for accumulate_into_main_grad."""
    with torch.no_grad():
        for param in module.parameters():
            if getattr(param, "main_grad", None) is None:
                param.main_grad = torch.empty(
                    param.size(), device=param.device, dtype=torch.float32
                )
            param.main_grad.fill_(value)


def zero_grads(module: torch.nn.Module, x: torch.Tensor, scales: torch.Tensor) -> None:
    """Reset gradients without changing allocated main_grad buffers."""
    module.zero_grad(set_to_none=True)
    x.grad = None
    scales.grad = None


def run_grouped_mlp_steps(
    module: torch.nn.Module,
    x: torch.Tensor,
    split_sizes: torch.Tensor,
    scales: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    recipe: MXFP8BlockScaling,
    fwd_only: bool,
    num_steps: int,
    accumulate_into_main_grad: bool,
) -> torch.Tensor:
    """Run eager grouped MLP for a number of synthetic microbatches."""
    quantization_context = te.autocast(enabled=True, recipe=recipe)

    if fwd_only:
        with torch.no_grad(), quantization_context:
            for _ in range(num_steps):
                out = module(x, split_sizes, scales, split_sizes)
        return out

    zero_grads(module, x, scales)
    if accumulate_into_main_grad:
        init_main_grads(module)

    with quantization_context:
        for step in range(num_steps):
            torch.cuda.nvtx.range_push(f"step_{step}")
            out = module(x, split_sizes, scales, split_sizes)
            out.backward(grad_output)
            torch.cuda.nvtx.range_pop()
    return out


def benchmark_case(
    *,
    total_tokens: int,
    hidden_dim: int,
    ffn_hidden_dim: int,
    num_groups: int,
    dtype: torch.dtype,
    fwd_only: bool,
    single_grouped_weight: bool,
    accumulate_into_main_grad: bool,
    glu_interleave_size: int,
    num_microbatches: int,
    min_run_time: float,
    profile: bool,
) -> float:
    """Benchmark one grouped MLP shape."""
    split_sizes_list = make_uniform_splits(total_tokens, num_groups)
    split_sizes = torch.tensor(split_sizes_list, dtype=torch.int64, device="cuda")
    x = torch.randn(
        (total_tokens, hidden_dim),
        dtype=dtype,
        device="cuda",
        requires_grad=not fwd_only,
    )
    scales = torch.ones(
        (total_tokens,),
        dtype=dtype,
        device="cuda",
        requires_grad=not fwd_only,
    )
    grad_output = torch.ones((total_tokens, hidden_dim), dtype=dtype, device="cuda")

    module = build_grouped_mlp(
        num_groups=num_groups,
        hidden_dim=hidden_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        dtype=dtype,
        single_grouped_weight=single_grouped_weight,
        accumulate_into_main_grad=accumulate_into_main_grad,
        glu_interleave_size=glu_interleave_size,
    )
    recipe = MXFP8BlockScaling()

    print(
        "case:",
        f"tokens={total_tokens}",
        f"hidden={hidden_dim}",
        f"ffn_hidden={ffn_hidden_dim}",
        f"num_groups={num_groups}",
        f"fwd_only={fwd_only}",
        f"single_grouped_weight={single_grouped_weight}",
        f"accumulate_into_main_grad={accumulate_into_main_grad}",
        f"glu_interleave_size={glu_interleave_size}",
    )
    print(f"m_splits: {split_sizes_list}")

    # Warmup also forces the op-fuser to materialize the expected fused ops.
    run_grouped_mlp_steps(
        module,
        x,
        split_sizes,
        scales,
        grad_output,
        recipe=recipe,
        fwd_only=fwd_only,
        num_steps=128,
        accumulate_into_main_grad=accumulate_into_main_grad,
    )
    torch.cuda.synchronize()

    forward_ops = module._module_groups[0]._forward_ops
    print("forward fused op:", type(forward_ops[0][0]).__name__ if forward_ops else "none")
    if not fwd_only:
        backward_ops = module._module_groups[0]._backward_ops
        print("backward fused op:", type(backward_ops[0][0]).__name__ if backward_ops else "none")

    label = "graph_safe_grouped_mlp_mxfp8_swiglu"
    timing_context = (
        torch.autograd.profiler.emit_nvtx(record_shapes=True) if profile else nullcontext()
    )
    with timing_context:
        torch.cuda.nvtx.range_push(label)
        timing = benchmark.Timer(
            stmt=(
                "run_grouped_mlp_steps("
                "module, x, split_sizes, scales, grad_output, "
                "recipe=recipe, fwd_only=fwd_only, num_steps=num_microbatches, "
                "accumulate_into_main_grad=accumulate_into_main_grad)"
            ),
            globals={
                "run_grouped_mlp_steps": run_grouped_mlp_steps,
                "module": module,
                "x": x,
                "split_sizes": split_sizes,
                "scales": scales,
                "grad_output": grad_output,
                "recipe": recipe,
                "fwd_only": fwd_only,
                "num_microbatches": num_microbatches,
                "accumulate_into_main_grad": accumulate_into_main_grad,
            },
            num_threads=1,
        ).blocked_autorange(min_run_time=min_run_time)
        torch.cuda.nvtx.range_pop()

    print(f"mxfp8_swiglu: {timing}\n")
    return timing.median * 1000 / num_microbatches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable NVTX profiling annotations")
    parser.add_argument(
        "--fwd-only",
        action="store_true",
        default=False,
        help="Benchmark forward only. Default benchmarks forward + backward.",
    )
    parser.add_argument(
        "--num-groups",
        type=str,
        default="8",
        help="Comma-separated local grouped GEMM/expert counts.",
    )
    parser.add_argument(
        "--token-dims",
        type=str,
        default="65536",
        help="Comma-separated total token counts to benchmark.",
    )
    parser.add_argument("--hidden-dim", type=int, default=7168)
    parser.add_argument("--ffn-hidden-dim", type=int, default=2048)
    parser.add_argument("--num-microbatches", type=int, default=32)
    parser.add_argument("--min-run-time", type=float, default=10.0)
    parser.add_argument("--glu-interleave-size", type=int, default=32)
    parser.add_argument(
        "--single-grouped-weight",
        action="store_true",
        default=False,
        help="Use one GroupedTensor parameter for each grouped linear.",
    )
    args = parser.parse_args()

    if not MXFP8_AVAILABLE:
        raise RuntimeError(f"MXFP8 is not available: {REASON_FOR_NO_MXFP8}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    dtype = torch.bfloat16
    accumulate_into_main_grad = True
    token_dims = parse_int_list(args.token_dims)
    num_groups_list = parse_int_list(args.num_groups)

    print("Environment toggles:")
    for name in (
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP",
        "CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL",
    ):
        print(f"  {name}={os.environ.get(name)}")
    print("Recipe: MXFP8BlockScaling")
    print("Activation: ScaledSwiGLU")
    print(f"Default GLU interleave size: {args.glu_interleave_size}")
    print()

    data = []
    for num_groups in num_groups_list:
        for total_tokens in token_dims:
            timing_ms = benchmark_case(
                total_tokens=total_tokens,
                hidden_dim=args.hidden_dim,
                ffn_hidden_dim=args.ffn_hidden_dim,
                num_groups=num_groups,
                dtype=dtype,
                fwd_only=args.fwd_only,
                single_grouped_weight=args.single_grouped_weight,
                accumulate_into_main_grad=accumulate_into_main_grad,
                glu_interleave_size=args.glu_interleave_size,
                num_microbatches=args.num_microbatches,
                min_run_time=args.min_run_time,
                profile=args.profile,
            )
            data.append(
                [
                    total_tokens,
                    args.hidden_dim,
                    args.ffn_hidden_dim,
                    num_groups,
                    args.glu_interleave_size,
                    args.single_grouped_weight,
                    accumulate_into_main_grad,
                    "fwd" if args.fwd_only else "fwd_bwd",
                    timing_ms,
                ]
            )

    timing_col = "time_per_microbatch_ms"
    df = pd.DataFrame(
        data=data,
        columns=[
            "tokens",
            "hidden_dim",
            "ffn_hidden_dim",
            "num_groups",
            "glu_interleave_size",
            "single_grouped_weight",
            "accumulate_into_main_grad",
            "mode",
            timing_col,
        ],
    )
    print(df)


if __name__ == "__main__":
    main()

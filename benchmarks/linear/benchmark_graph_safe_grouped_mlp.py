# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark BF16, MXFP8 and NVFP4 graph-safe grouped MLP.

This mirrors ``benchmark_grouped_linear.py`` but targets the graph-safe TE ops
path used by grouped MLP:

    GroupedLinear -> ScaledSwiGLU -> GroupedLinear

The benchmark uses CUDA-device ``m_splits`` and enforces the requested fused path.

Example:

    python benchmarks/linear/benchmark_graph_safe_grouped_mlp.py

Forward-only:

    python benchmarks/linear/benchmark_graph_safe_grouped_mlp.py --fwd-only

Compare conventional [gate | up] storage with interleave32 and unfused execution
on Qwen3.5-397B shapes (16 experts, M=2560, K=4096, F=1024):

    python benchmarks/linear/benchmark_graph_safe_grouped_mlp.py \
        --preset qwen3.5-397b --layout compare --timing both

DeepSeek-V3, context 4096, EP8, TP1 (32 local experts, M=1024):

    python benchmarks/linear/benchmark_graph_safe_grouped_mlp.py \
        --preset deepseek-v3 --expert-parallel-size 8 \
        --precision all --layout compare --timing both

All layouts use the same input and canonical weight values. Comparison mode
checks outputs and gradients against the unfused path before timing. Layout
conversion is performed only during initialization and validation. Fused modes
raise if the fuser falls back. Eager timing includes CPU dispatch and gradient
reset; graph timing replays an actual CUDA graph of the same microbatch bundle.
Use ``--comparison-rounds 2`` or more to alternate fused layout order across
rounds. Every round checks a fresh unfused reference; samples and their IQR are
reported with the round ID. These are complete-case order checks, not adjacent
kernel timing pairs.

Nsight Systems:

    (optionally: unset DEBUGINFOD_URLS)

    nsys profile \
        --output=./benchmarks/linear/graph_safe_grouped_mlp_mxfp8 \
        --force-overwrite true \
        --trace=cuda,nvtx,cudnn,cublas \
        python benchmarks/linear/benchmark_graph_safe_grouped_mlp.py --profile
"""

# Match the Qwen MXFP8 SFT launch toggles before importing TE.
import os

os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
os.environ.setdefault("CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL", "1")

import argparse
from contextlib import contextmanager, nullcontext
import functools
import inspect
from importlib.metadata import PackageNotFoundError, version
import statistics
import subprocess
from typing import Optional

import pandas as pd
import torch
import torch.utils.benchmark as benchmark

import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.common.recipe import MXFP8BlockScaling, NVFP4BlockScaling, Recipe
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.ops.fuser import OperationFuser
from transformer_engine.pytorch.ops.fused.grouped_mlp import (
    GroupedMLP_CuTeGEMMGLU,
    GroupedMLP_CuTeGEMMGLUBf16,
    fuse_glu_ops,
    fuse_bf16_ops,
)


MXFP8_AVAILABLE, REASON_FOR_NO_MXFP8 = FP8GlobalStateManager.is_mxfp8_available()


def make_recipe(precision: str) -> Optional[Recipe]:
    """Use repeatable quantization for layout comparisons."""
    if precision == "bf16":
        return None
    if precision == "mxfp8":
        return MXFP8BlockScaling()
    if precision == "nvfp4":
        return NVFP4BlockScaling(disable_stochastic_rounding=True, disable_2d_quantization=True)
    raise ValueError(f"Unsupported precision {precision}")


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
    glu_interleave_size: Optional[int],
    recipe: Optional[Recipe],
) -> te_ops.Sequential:
    """Build graph-safe grouped MLP ops sequence."""
    with te.quantized_model_init(enabled=recipe is not None, recipe=recipe):
        fc1 = te_ops.GroupedLinear(
            num_groups,
            hidden_dim,
            2 * ffn_hidden_dim,
            bias=False,
            device="cuda",
            dtype=dtype,
            single_grouped_weight=single_grouped_weight,
            accumulate_into_main_grad=accumulate_into_main_grad,
            split_alignment=256,
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
            split_alignment=256,
        )
        return te_ops.Sequential(
            fc1,
            te_ops.ScaledSwiGLU(glu_interleave_size=glu_interleave_size),
            fc2,
        )


@contextmanager
def fusion_enabled(enabled: bool):
    """Disable only grouped GLU fusion when constructing the unfused reference."""
    callbacks = OperationFuser.forward_backward_fusion_functions
    if not enabled:
        OperationFuser.forward_backward_fusion_functions = [
            callback for callback in callbacks if callback not in (fuse_glu_ops, fuse_bf16_ops)
        ]
    try:
        yield
    finally:
        OperationFuser.forward_backward_fusion_functions = callbacks


def clear_glu_wrapper_caches():
    """Drop cached function objects after an untimed tracing override."""
    for cls in (GroupedMLP_CuTeGEMMGLU, GroupedMLP_CuTeGEMMGLUBf16):
        for name in (
            "grouped_gemm_activation_kernel",
            "grouped_gemm_act_hadamard_kernel",
            "grouped_gemm_act_hadamard_quant_kernel",
            "grouped_gemm_dactivation_kernel",
        ):
            getter = getattr(cls, name, None)
            if getter is not None:
                getter.cache_clear()


@contextmanager
def trace_glu_wrappers():
    """Record actual wrapper geometry on the first, untimed microbatch only."""
    import cudnn

    originals, calls = {}, []
    for suffix in ("glu", "glu_hadamard", "glu_hadamard_quant", "dglu"):
        name = f"grouped_gemm_{suffix}_wrapper_sm100"
        original = getattr(cudnn, name, None)
        if original is None:
            continue
        originals[name] = original

        def make_traced(function, function_name):
            parameters = inspect.signature(function).parameters

            @functools.wraps(function)
            def traced(*args, **kwargs):
                calls.append(
                    {
                        "wrapper": function_name,
                        "operand_ranks": {
                            name: kwargs[name].ndim
                            for name in (
                                "a_tensor",
                                "b_tensor",
                                "c_tensor",
                                "sfa_tensor",
                                "sfb_tensor",
                                "prob_tensor",
                            )
                            if kwargs.get(name) is not None
                        },
                        **{
                            field: kwargs.get(field, parameters[field].default)
                            for field in ("mma_tiler_mn", "cluster_shape_mn", "glu_layout")
                            if field in parameters
                        },
                    }
                )
                return function(*args, **kwargs)

            return traced

        setattr(cudnn, name, make_traced(original, name))
    clear_glu_wrapper_caches()
    try:
        yield calls
    finally:
        for name, original in originals.items():
            setattr(cudnn, name, original)
        clear_glu_wrapper_caches()


@contextmanager
def canonical_fc1_tile_override(tile_m):
    """Benchmark control for regular canonical GLU; production follows the frontend hint."""
    if tile_m is None:
        yield
        return
    import cudnn

    kernel = cudnn.grouped_gemm_glu_wrapper_sm100
    missing = object()
    preference = getattr(kernel, "preferred_gate_then_up_mma_tiler", missing)
    if tile_m is not None:
        kernel.preferred_gate_then_up_mma_tiler = (tile_m, 256)
    try:
        yield
    finally:
        if preference is missing:
            if tile_m is not None:
                del kernel.preferred_gate_then_up_mma_tiler
        else:
            kernel.preferred_gate_then_up_mma_tiler = preference


@contextmanager
def canonical_rht_override(mode):
    """Measure the existing separate RHT path without changing legacy selection."""
    import cudnn
    from transformer_engine.pytorch.ops.fused import grouped_mlp

    kernel = getattr(cudnn, "grouped_gemm_glu_hadamard_wrapper_sm100", None)
    layouts = getattr(kernel, "supports_glu_layouts", None)
    preference = grouped_mlp._prefer_gate_then_up_hadamard
    if mode == "separate" and layouts is not None:
        kernel.supports_glu_layouts = tuple(x for x in layouts if x != "gate_then_up")
    if mode == "fused":
        grouped_mlp._prefer_gate_then_up_hadamard = lambda _: True
    try:
        yield
    finally:
        if layouts is not None:
            kernel.supports_glu_layouts = layouts
        grouped_mlp._prefer_gate_then_up_hadamard = preference


def initialize_weights(module, *, seed, glu_interleave_size):
    """Initialize equivalent canonical values, converting legacy storage outside timing."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for layer_index, layer in ((0, module[0]), (2, module[2])):
            weights = (
                layer.weight.split_into_quantized_tensors()
                if layer.single_grouped_weight
                else [getattr(layer, f"weight{i}") for i in range(layer.num_groups)]
            )
            for weight in weights:
                value = (
                    torch.randn(
                        weight.shape, device=weight.device, dtype=weight.dtype, generator=generator
                    )
                    * 0.02
                )
                if layer_index == 0 and glu_interleave_size is not None:
                    value = te.interleave_glu_tensor(value, glu_interleave_size)
                weight.copy_(value)


def snapshot_result(module, out, x, scales, *, fwd_only, glu_interleave_size):
    """Collect one-step results in canonical order, outside all timed regions."""
    result = {"output": out.detach().clone()}
    if fwd_only:
        return result
    result.update(dx=x.grad.detach().clone(), dscales=scales.grad.detach().clone())
    for layer_index, layer in ((0, module[0]), (2, module[2])):
        weights = (
            [layer.weight]
            if layer.single_grouped_weight
            else [getattr(layer, f"weight{i}") for i in range(layer.num_groups)]
        )
        grads = [getattr(weight, "main_grad", weight.grad).detach().clone() for weight in weights]
        grad = grads[0] if layer.single_grouped_weight else torch.stack(grads)
        if layer_index == 0 and glu_interleave_size is not None:
            grad = torch.stack([te.deinterleave_glu_tensor(g, glu_interleave_size) for g in grad])
        result[f"fc{1 if layer_index == 0 else 2}_dw"] = grad
    return result


def check_results(actual, reference, *, tolerance=0.125, reference_name="canonical unfused"):
    """Compare quantized paths by relative RMS, rejecting NaNs and layout errors."""
    errors = {}
    for name, expected in reference.items():
        value = actual[name].float()
        expected = expected.float()
        error = (value - expected).square().mean().sqrt()
        scale = expected.square().mean().sqrt().clamp_min(1e-7)
        relative_error = (error / scale).item()
        if (
            not torch.isfinite(value).all()
            or not torch.isfinite(expected).all()
            or relative_error > tolerance
        ):
            raise AssertionError(f"{name}: relative RMS error {relative_error} exceeds {tolerance}")
        errors[name] = relative_error
    print(f"Relative RMS errors vs {reference_name}:", errors)


def timing_statistics(samples: list[float]) -> dict:
    """Summarize milliseconds per microbatch without implying a confidence interval."""
    if len(samples) > 1:
        quartiles = statistics.quantiles(samples, n=4, method="inclusive")
        iqr = quartiles[2] - quartiles[0]
    else:
        iqr = 0.0
    return {"median_ms": statistics.median(samples), "iqr_ms": iqr, "samples_ms": samples}


def print_backend_versions():
    """Report source and binary versions needed to reproduce a measurement."""
    import cudnn  # pylint: disable=import-outside-toplevel
    import cutlass  # pylint: disable=import-outside-toplevel

    print(f"cuDNN frontend source: {cudnn.__file__}")
    print(f"CUTLASS DSL source: {cutlass.__file__}")
    print(f"cuDNN runtime: {torch.backends.cudnn.version()}")
    for package in ("nvidia-cudnn-frontend", "nvidia-cutlass-dsl"):
        try:
            print(f"{package}: {version(package)}")
        except PackageNotFoundError:
            print(f"{package}: no installed distribution metadata")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        print(f"NVIDIA driver: {result.stdout.splitlines()[0]}")
    except (OSError, subprocess.SubprocessError, IndexError):
        print("NVIDIA driver: unavailable from nvidia-smi")


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
    recipe: Optional[Recipe],
    fwd_only: bool,
    num_steps: int,
    accumulate_into_main_grad: bool,
) -> torch.Tensor:
    """Run eager grouped MLP for a number of synthetic microbatches."""
    quantization_context = te.autocast(enabled=recipe is not None, recipe=recipe)

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
    glu_interleave_size: Optional[int],
    num_microbatches: int,
    min_run_time: float,
    profile: bool,
    layout: str,
    timing_mode: str,
    warmup_steps: int,
    event_repeats: int,
    seed: int,
    precision: str,
    comparison_round: int,
    canonical_rht: str,
    canonical_fc1_tile_m: Optional[int] = None,
    reference: Optional[dict[str, torch.Tensor]] = None,
) -> tuple[dict[str, dict], dict[str, torch.Tensor]]:
    """Benchmark one grouped MLP shape."""
    torch.manual_seed(seed)
    split_sizes_list = make_uniform_splits(total_tokens, num_groups)
    if any(size % 256 for size in split_sizes_list):
        raise ValueError(
            "The current fused kernel requires each expert's token count divisible by 256"
        )
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
    grad_output = torch.randn((total_tokens, hidden_dim), dtype=dtype, device="cuda") * 0.1

    recipe = make_recipe(precision)

    module = build_grouped_mlp(
        num_groups=num_groups,
        hidden_dim=hidden_dim,
        ffn_hidden_dim=ffn_hidden_dim,
        dtype=dtype,
        single_grouped_weight=single_grouped_weight,
        accumulate_into_main_grad=accumulate_into_main_grad,
        glu_interleave_size=glu_interleave_size,
        recipe=recipe,
    )
    initialize_weights(module, seed=seed + 1, glu_interleave_size=glu_interleave_size)
    if accumulate_into_main_grad:
        # The unfused GroupedLinear validates this storage even under no_grad.
        init_main_grads(module)

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
        f"layout={layout}",
        f"precision={precision}",
        f"comparison_round={comparison_round}",
        f"canonical_fc1_tile_m={canonical_fc1_tile_m}",
    )
    print(f"m_splits: {split_sizes_list}")
    print(
        f"FC1 forward: {num_groups} x (M={split_sizes_list[0]}, N={2 * ffn_hidden_dim},"
        f" K={hidden_dim})"
    )
    print(
        f"FC2 forward: {num_groups} x (M={split_sizes_list[0]}, N={hidden_dim}, K={ffn_hidden_dim})"
    )
    print(
        f"FC2 dgrad + dSwiGLU: {num_groups} x (M={split_sizes_list[0]}, N={ffn_hidden_dim},"
        f" K={hidden_dim})"
    )

    # The first step materializes the selected fuser and provides untimed
    # one-microbatch results for comparison across physical weight layouts.
    with trace_glu_wrappers() as wrapper_calls:
        out = run_grouped_mlp_steps(
            module,
            x,
            split_sizes,
            scales,
            grad_output,
            recipe=recipe,
            fwd_only=fwd_only,
            num_steps=1,
            accumulate_into_main_grad=accumulate_into_main_grad,
        )
    torch.cuda.synchronize()
    print("Actual cuDNN GLU wrappers:", wrapper_calls)
    if canonical_fc1_tile_m is not None and layout == "gate_then_up" and precision != "bf16":
        for call in wrapper_calls:
            if call["wrapper"] == "grouped_gemm_glu_wrapper_sm100":
                if call["mma_tiler_mn"] != (canonical_fc1_tile_m, 256):
                    raise RuntimeError("The canonical FC1 tile control was not honored")
    if canonical_rht == "fused" and layout == "gate_then_up" and precision == "nvfp4":
        if not any("hadamard" in call["wrapper"] for call in wrapper_calls):
            raise RuntimeError(
                "Requested fused canonical RHT, but the specialized wrapper was not selected"
            )

    forward_ops = module._module_groups[0]._forward_ops
    print("forward fused op:", type(forward_ops[0][0]).__name__ if forward_ops else "none")
    if not fwd_only:
        backward_ops = module._module_groups[0]._backward_ops
        print("backward fused op:", type(backward_ops[0][0]).__name__ if backward_ops else "none")

    fused_cls = GroupedMLP_CuTeGEMMGLUBf16 if precision == "bf16" else GroupedMLP_CuTeGEMMGLU
    fused = len(forward_ops) == 1 and isinstance(forward_ops[0][0], fused_cls)
    if fused != (layout != "unfused"):
        raise RuntimeError(f"Requested {layout}, but grouped GLU fusion selected={fused}")
    if fused and not fwd_only:
        if len(backward_ops) != 1 or backward_ops[0][0] is not forward_ops[0][0]:
            raise RuntimeError("Fused grouped GLU backward was not selected")
    if layout == "gate_then_up" and not forward_ops[0][0]._gate_then_up:
        raise RuntimeError("Requested canonical GLU, but selected interleaved fused execution")

    result = snapshot_result(
        module, out, x, scales, fwd_only=fwd_only, glu_interleave_size=glu_interleave_size
    )
    if reference is not None:
        check_results(
            result, reference, tolerance={"bf16": 0.02, "mxfp8": 0.125, "nvfp4": 0.25}[precision]
        )
    del out

    run_grouped_mlp_steps(
        module,
        x,
        split_sizes,
        scales,
        grad_output,
        recipe=recipe,
        fwd_only=fwd_only,
        num_steps=warmup_steps,
        accumulate_into_main_grad=accumulate_into_main_grad,
    )
    torch.cuda.synchronize()

    def run_bundle():
        return run_grouped_mlp_steps(
            module,
            x,
            split_sizes,
            scales,
            grad_output,
            recipe=recipe,
            fwd_only=fwd_only,
            num_steps=num_microbatches,
            accumulate_into_main_grad=accumulate_into_main_grad,
        )

    graph = None
    if timing_mode in ("graph", "both"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = run_bundle()
        graph.replay()
        torch.cuda.synchronize()
        graph_result = snapshot_result(
            module,
            graph_out,
            x,
            scales,
            fwd_only=fwd_only,
            glu_interleave_size=glu_interleave_size,
        )
        for name, value in graph_result.items():
            if name != "output":
                value.div_(num_microbatches)
        check_results(graph_result, result, reference_name="one eager microbatch")
        del graph_result

    label = f"graph_safe_grouped_mlp_{precision}_swiglu"
    timing_context = (
        torch.autograd.profiler.emit_nvtx(record_shapes=True) if profile else nullcontext()
    )
    timings = {}
    if not fwd_only:
        # Time training forward and backward directly, with a fresh autograd
        # graph each step. Do not estimate backward by subtracting two medians.
        fwd_samples, bwd_samples = [], []
        start = torch.cuda.Event(enable_timing=True)
        middle = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(event_repeats):
            zero_grads(module, x, scales)
            if accumulate_into_main_grad:
                init_main_grads(module)
            with te.autocast(enabled=recipe is not None, recipe=recipe):
                start.record()
                phase_out = module(x, split_sizes, scales, split_sizes)
                middle.record()
                phase_out.backward(grad_output)
                end.record()
            end.synchronize()
            fwd_samples.append(start.elapsed_time(middle))
            bwd_samples.append(middle.elapsed_time(end))
        timings["eager_forward_phase_device"] = timing_statistics(fwd_samples)
        timings["eager_backward_phase_device"] = timing_statistics(bwd_samples)
    runners = {}
    if timing_mode in ("eager", "both"):
        runners["eager"] = run_bundle
    if graph is not None:
        runners["graph"] = graph.replay
    with timing_context:
        for mode, run in runners.items():
            with torch.cuda.nvtx.range(f"{label}_{layout}_{mode}"):
                timing = benchmark.Timer(
                    stmt="run()", globals={"run": run}, num_threads=1
                ).blocked_autorange(min_run_time=min_run_time)
            print(f"{layout} {mode}: {timing}\n")
            timings[f"{mode}_wall"] = timing_statistics(
                [value * 1000 / num_microbatches for value in timing.times]
            )

            # Device timeline includes all kernels in the bundle. Eager events
            # can include launch gaps; graph events avoid per-kernel CPU dispatch.
            samples = []
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(event_repeats):
                start.record()
                run()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / num_microbatches)
            timings[f"{mode}_device"] = timing_statistics(samples)
    for name, values in timings.items():
        print(
            f"round={comparison_round} {precision} {layout} {name}: "
            f"median_ms={values['median_ms']:.6f}, iqr_ms={values['iqr_ms']:.6f}, "
            f"samples_ms={values['samples_ms']}"
        )
    return timings, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Enable NVTX profiling annotations")
    parser.add_argument(
        "--fwd-only",
        action="store_true",
        default=False,
        help="Benchmark forward only. Default benchmarks forward + backward.",
    )
    parser.add_argument("--mode", choices=("fwd", "fwd_bwd", "both"), default="fwd_bwd")
    parser.add_argument("--precision", choices=("bf16", "nvfp4", "mxfp8", "all"), default="mxfp8")
    parser.add_argument("--canonical-rht", choices=("auto", "separate", "fused"), default="auto")
    parser.add_argument(
        "--canonical-fc1-tile-m",
        type=int,
        choices=(128, 256),
        help=(
            "Override regular canonical quantized FC1 tile M for before/after controls; default"
            " follows the frontend preference"
        ),
    )
    parser.add_argument(
        "--num-groups",
        type=str,
        default=None,
        help="Comma-separated local grouped GEMM/expert counts.",
    )
    parser.add_argument(
        "--token-dims",
        type=str,
        default=None,
        help="Comma-separated total token counts to benchmark.",
    )
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--ffn-hidden-dim", type=int)
    parser.add_argument("--num-microbatches", type=int, default=32)
    parser.add_argument("--min-run-time", type=float, default=10.0)
    parser.add_argument("--warmup-steps", type=int, default=128)
    parser.add_argument("--event-repeats", type=int, default=20)
    parser.add_argument(
        "--comparison-rounds",
        type=int,
        default=1,
        help="Repeat complete cases, alternating fused layout order in compare mode",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--glu-interleave-size", type=int, default=32)
    parser.add_argument("--preset", choices=("qwen3.5-397b", "deepseek-v3"))
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--expert-parallel-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--layout",
        choices=("interleaved_32", "gate_then_up", "unfused", "compare"),
        default="interleaved_32",
        help="compare validates both fused layouts against canonical unfused execution",
    )
    parser.add_argument("--timing", choices=("eager", "graph", "both"), default="eager")
    parser.add_argument(
        "--single-grouped-weight",
        action="store_true",
        default=False,
        help="Use one GroupedTensor parameter for each grouped linear.",
    )
    args = parser.parse_args()

    if min(args.context_length, args.expert_parallel_size, args.tensor_parallel_size) < 1:
        parser.error(
            "context-length, expert-parallel-size and tensor-parallel-size must be positive"
        )
    if args.preset:
        experts, top_k, hidden, ffn = (
            (512, 10, 4096, 1024) if args.preset == "qwen3.5-397b" else (256, 8, 7168, 2048)
        )
        if experts % args.expert_parallel_size or ffn % args.tensor_parallel_size:
            parser.error("Preset experts and FFN dimensions must divide evenly by EP and TP")
        defaults = {
            "num_groups": str(experts // args.expert_parallel_size),
            "token_dims": str(args.context_length * top_k),
            "hidden_dim": hidden,
            "ffn_hidden_dim": ffn // args.tensor_parallel_size,
        }
        print(
            f"Preset {args.preset}: context={args.context_length}, experts={experts},"
            f" top-k={top_k}, EP={args.expert_parallel_size}, TP={args.tensor_parallel_size};"
            " uniform routing"
        )
    else:
        defaults = {
            "num_groups": "8",
            "token_dims": "65536",
            "hidden_dim": 7168,
            "ffn_hidden_dim": 2048,
        }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.glu_interleave_size != 32:
        parser.error("Use --layout gate_then_up for canonical storage; legacy fusion requires 32")
    if (
        min(args.warmup_steps, args.num_microbatches, args.event_repeats, args.comparison_rounds)
        < 1
    ):
        parser.error(
            "warmup-steps, num-microbatches, event-repeats and comparison-rounds must be positive"
        )

    precisions = ("bf16", "nvfp4", "mxfp8") if args.precision == "all" else (args.precision,)
    if "mxfp8" in precisions and not MXFP8_AVAILABLE:
        raise RuntimeError(f"MXFP8 is not available: {REASON_FOR_NO_MXFP8}")
    if "nvfp4" in precisions:
        available, reason = te.is_nvfp4_available(return_reason=True)
        if not available:
            raise RuntimeError(f"NVFP4 is not available: {reason}")
        if args.single_grouped_weight:
            parser.error(
                "NVFP4 grouped MLP requires discrete weights; omit --single-grouped-weight"
            )
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
        "CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL",
    ):
        print(f"  {name}={os.environ.get(name)}")
    print(f"Precisions: {precisions}")
    print("NVFP4 recipe: RHT enabled, stochastic rounding and 2D weight quantization disabled")
    print("Activation: ScaledSwiGLU")
    print(f"Default GLU interleave size: {args.glu_interleave_size}")
    print(f"GPU: {torch.cuda.get_device_name()}, capability={torch.cuda.get_device_capability()}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"TE source: {te.__file__}")
    print_backend_versions()
    print(f"Warmup steps: {args.warmup_steps}, microbatches/bundle: {args.num_microbatches}")
    print(f"Timing: {args.timing}; median wall and CUDA event time per microbatch")
    print(f"Comparison rounds: {args.comparison_rounds}; alternate fused layout order each round")
    print("IQR describes sample spread, not a confidence interval for a performance difference.")
    print("Device time includes quantization, GEMMs, activation and gradient reset kernels.")
    print("Eager device time may include launch gaps; use graph device time to isolate GPU work.")
    print()

    data = []
    execution_modes = ("fwd", "fwd_bwd") if args.mode == "both" else (args.mode,)
    if args.fwd_only:
        execution_modes = ("fwd",)
    for num_groups in num_groups_list:
        for total_tokens in token_dims:
            for precision in precisions:
                for execution_mode in execution_modes:
                    for round_index in range(args.comparison_rounds):
                        fused_layouts = ("interleaved_32", "gate_then_up")
                        if round_index % 2:
                            fused_layouts = fused_layouts[::-1]
                        layouts = (
                            ("unfused", *fused_layouts)
                            if args.layout == "compare"
                            else (args.layout,)
                        )
                        reference = None
                        for layout in layouts:
                            interleave_size = 32 if layout == "interleaved_32" else None
                            with (
                                fusion_enabled(layout != "unfused"),
                                canonical_rht_override(args.canonical_rht),
                                canonical_fc1_tile_override(args.canonical_fc1_tile_m),
                            ):
                                timings, result = benchmark_case(
                                    total_tokens=total_tokens,
                                    hidden_dim=args.hidden_dim,
                                    ffn_hidden_dim=args.ffn_hidden_dim,
                                    num_groups=num_groups,
                                    dtype=dtype,
                                    fwd_only=execution_mode == "fwd",
                                    single_grouped_weight=args.single_grouped_weight,
                                    accumulate_into_main_grad=accumulate_into_main_grad,
                                    glu_interleave_size=interleave_size,
                                    num_microbatches=args.num_microbatches,
                                    min_run_time=args.min_run_time,
                                    profile=args.profile,
                                    layout=layout,
                                    timing_mode=args.timing,
                                    warmup_steps=args.warmup_steps,
                                    event_repeats=args.event_repeats,
                                    seed=args.seed,
                                    precision=precision,
                                    comparison_round=round_index + 1,
                                    canonical_rht=args.canonical_rht,
                                    canonical_fc1_tile_m=args.canonical_fc1_tile_m,
                                    reference=reference,
                                )
                            if layout == "unfused":
                                reference = result
                            for timing_mode, values in timings.items():
                                data.append(
                                    [
                                        total_tokens,
                                        args.hidden_dim,
                                        args.ffn_hidden_dim,
                                        num_groups,
                                        precision,
                                        layout,
                                        args.single_grouped_weight,
                                        accumulate_into_main_grad,
                                        execution_mode,
                                        round_index + 1,
                                        timing_mode,
                                        values["median_ms"],
                                        values["iqr_ms"],
                                        len(values["samples_ms"]),
                                    ]
                                )
                            del result
                        del reference

    timing_col = "time_per_microbatch_ms"
    df = pd.DataFrame(
        data=data,
        columns=[
            "tokens",
            "hidden_dim",
            "ffn_hidden_dim",
            "num_groups",
            "precision",
            "layout",
            "single_grouped_weight",
            "accumulate_into_main_grad",
            "mode",
            "comparison_round",
            "timing",
            timing_col,
            "iqr_ms",
            "sample_count",
        ],
    )
    print(df)


if __name__ == "__main__":
    main()

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Measure Linear microbatch cache memory; run each mode in a fresh process.

Example:
    python examples/pytorch/linear_microbatch_memory.py --mode eager
    python examples/pytorch/linear_microbatch_memory.py --mode eager-fresh
    python examples/pytorch/linear_microbatch_memory.py --mode compile
    python examples/pytorch/linear_microbatch_memory.py --mode compile-cg --mark-step

CUDA Graph replay does not update allocator accounting for internal allocations.
Compare reserved memory as well as allocated peaks, including capture/warmup.
"""

import argparse
import gc
import importlib
import json
from pathlib import Path

import torch

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8CurrentScaling


class LinearStack(torch.nn.Module):
    def __init__(self, hidden_size, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            te.Linear(hidden_size, hidden_size, bias=False, params_dtype=torch.bfloat16)
            for _ in range(layers)
        )

    def forward(self, inp, first):
        for layer in self.layers:
            inp = torch.nn.functional.gelu(layer(inp, is_first_microbatch=first))
        return inp


def memory():
    torch.cuda.synchronize()
    mib = 1024**2
    return {
        "allocated_mib": torch.cuda.memory_allocated() / mib,
        "reserved_mib": torch.cuda.memory_reserved() / mib,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / mib,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / mib,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("eager", "eager-fresh", "compile", "compile-cg"), required=True
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--mark-step", action="store_true", help="Mark one CG iteration per minibatch"
    )
    parser.add_argument(
        "--release-cache",
        action="store_true",
        help="Diagnostic: drop old caches before each minibatch",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(1234)
    model = LinearStack(args.hidden_size, args.layers).cuda()
    fp8_recipe = Float8CurrentScaling()
    inputs = [
        torch.randn(
            args.tokens, args.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        for _ in range(args.microbatches)
    ]

    if args.mode == "eager-fresh":
        linear = importlib.import_module("transformer_engine.pytorch.module.linear")
        original_quantize_weight = linear.quantize_weight

        def quantize_weight(**kwargs):
            if kwargs["cache"] and kwargs["update_workspace"]:
                kwargs["workspace"] = None
            return original_quantize_weight(**kwargs)

        linear.quantize_weight = quantize_weight

    def forward(inp, first):
        with te.autocast(recipe=fp8_recipe):
            return model(inp, first)

    forward(inputs[0], None).sum().backward()
    model.zero_grad(set_to_none=True)
    inputs[0].grad = None
    gc.collect()
    initial = memory()
    torch.cuda.reset_peak_memory_stats()

    replay_count = 0
    original_replay = torch.cuda.CUDAGraph.replay

    def replay(graph):
        nonlocal replay_count
        replay_count += 1
        return original_replay(graph)

    if args.mode == "compile-cg":
        torch.cuda.CUDAGraph.replay = replay
    run = forward
    if args.mode.startswith("compile"):
        run = torch.compile(
            forward,
            fullgraph=True,
            mode="reduce-overhead" if args.mode == "compile-cg" else "default",
        )

    def step(measure=False):
        if args.mark_step:
            torch.compiler.cudagraph_mark_step_begin()
        model.zero_grad(set_to_none=True)
        for inp in inputs:
            inp.grad = None
        if args.release_cache:
            for layer in model.layers:
                layer._fp8_workspaces.pop("weight", None)
        points = []
        for index, inp in enumerate(inputs):
            out = run(inp, index == 0)
            if measure:
                points.append({"phase": f"forward_{index}", **memory()})
            loss = out.float().square().mean() / args.microbatches
            loss.backward()
            del out, loss
            if measure:
                points.append({"phase": f"backward_{index}", **memory()})
        with torch.no_grad():
            for param in model.parameters():
                param.add_(param.grad, alpha=-0.001)
        return points

    for _ in range(args.warmup):
        step()
    gc.collect()
    warmup = memory()
    results = []
    from torch._dynamo.utils import counters

    graphs_before = counters["stats"]["unique_graphs"]
    replay_before = replay_count
    for index in range(args.steps):
        torch.cuda.reset_peak_memory_stats()
        points = step(measure=True)
        results.append({"step": index, "points": points, **memory()})

    if args.mode == "compile-cg":
        assert replay_count > replay_before, "No CUDA Graph replay during measured steps"
        assert counters["inductor"]["cudagraph_skips"] == 0, "CUDA Graph capture was skipped"
    for param in model.parameters():
        assert torch.isfinite(param).all(), "Nonfinite parameter"
        assert torch.isfinite(param.grad).all(), "Nonfinite gradient"
    result = {
        "config": {**vars(args), "output": str(args.output) if args.output else None},
        "torch": torch.__version__,
        "te_source": te.__file__,
        "gpu": torch.cuda.get_device_name(),
        "initial": initial,
        "warmup": warmup,
        "steps": results,
        "measured_replays": replay_count - replay_before,
        "measured_recompiles": counters["stats"]["unique_graphs"] - graphs_before,
        "cudagraph_skips": counters["inductor"]["cudagraph_skips"],
    }
    output = json.dumps(result, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()

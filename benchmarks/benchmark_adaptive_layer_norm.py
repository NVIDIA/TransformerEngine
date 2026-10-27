# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark adaptive LayerNorm and an optional diffusion-transformer MLP.

Reports CUDA graph replay latency for eager PyTorch, torch.compile, and TE.
The references accumulate normalization and modulation in float32, matching
AdaptiveLayerNorm. Shapes describe a single block, not a complete model.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import transformer_engine.pytorch as te


def reference(x, scale, shift, eps, batch_dim):
    """Normalize and modulate in float32, with one final output cast."""
    condition_shape = [1] * x.ndim
    condition_shape[batch_dim] = x.shape[batch_dim]
    condition_shape[-1] = x.shape[-1]
    scale = scale.reshape(condition_shape)
    shift = shift.reshape(condition_shape)
    normalized = F.layer_norm(x.float(), (x.shape[-1],), eps=eps)
    return (normalized * (1.0 + scale.float()) + shift.float()).to(x.dtype)


def graph_time(function, repeats):
    """Measure a fixed CUDA graph, excluding compilation and graph capture."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = function()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeats)
    # Keep the capture's output allocations live until all replays finish.
    del outputs
    return statistics.median(samples)


def host_time(function, repeats):
    """Measure amortized execution including Python dispatch and synchronization."""
    samples = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / repeats)
    return statistics.median(samples)


def peak_memory(function):
    """Measure extra PyTorch-managed allocation above persistent inputs/caches."""
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    outputs = function()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    del outputs
    return peak


def main():
    """Run correctness checks and CUDA graph benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--batch-dim", type=int, choices=(0, 1), default=1)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument(
        "--ffn", type=int, default=0, help="Add an MLP with this intermediate size."
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(1234)
    dtype = getattr(torch, args.dtype)
    shape = [args.sequence, args.sequence, args.hidden]
    shape[args.batch_dim] = args.batch
    condition_shape = [args.batch, args.hidden]
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    scale = torch.randn(condition_shape, device="cuda", requires_grad=True)
    shift = torch.randn(condition_shape, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    eps = 1e-6
    op = te.ops.AdaptiveLayerNorm(args.hidden, eps=eps, batch_dim=args.batch_dim)

    suffix = None
    if args.ffn:
        suffix = te.ops.Sequential(
            te.ops.Linear(args.hidden, args.ffn, dtype=dtype),
            te.ops.GELU(),
            te.ops.Linear(args.ffn, args.hidden, dtype=dtype),
        )

    def eager(a, b, c):
        return reference(a, b, c, eps, args.batch_dim)

    methods = {"pytorch_eager": eager, "transformer_engine": op}
    if not args.skip_compile:
        methods["pytorch_compile"] = torch.compile(eager, fullgraph=True)
    if args.reverse_order:
        methods = dict(reversed(methods.items()))

    # Compare against the same mathematical reference before measuring.
    expected = eager(x, scale, shift)
    expected_grads = torch.autograd.grad(expected, (x, scale, shift), dy)
    expected = expected.detach()
    results = []
    for name, norm in methods.items():
        actual = norm(x, scale, shift)
        gradients = torch.autograd.grad(actual, (x, scale, shift), dy)
        atol, rtol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (2e-3, 2e-3)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        for actual_grad, expected_grad in zip(gradients, expected_grads):
            # Conditions accumulate over sequence positions in float32.
            torch.testing.assert_close(actual_grad, expected_grad, atol=atol, rtol=rtol)

        # Release validation graphs before creating AccumulateGrad nodes on the
        # capture stream. Retaining them would introduce a default-stream edge.
        del actual
        parameters = () if suffix is None else tuple(suffix.parameters())
        if suffix is None:
            model = norm
        elif name == "transformer_engine":
            model = te.ops.Sequential(norm, suffix[0], suffix[1], suffix[2])
        else:
            # Identical TE MLP operations/weights isolate the normalization change.
            def model(a, b, c, normalization=norm):
                return suffix(normalization(a, b, c))

        block_gradient_checks = []
        if suffix is not None:
            block_reference = suffix(eager(x, scale, shift))
            block_actual = model(x, scale, shift)
            block_inputs = (x, scale, shift) + parameters
            block_reference_grads = torch.autograd.grad(block_reference, block_inputs, dy)
            block_actual_grads = torch.autograd.grad(block_actual, block_inputs, dy)
            torch.testing.assert_close(block_actual, block_reference, atol=atol, rtol=rtol)
            for actual_grad, expected_grad in zip(block_actual_grads, block_reference_grads):
                # Reduced MLP gradients can be much larger than activations.
                # Use one activation-dtype epsilon at the gradient RMS as the
                # absolute scale, retaining the pointwise relative criterion.
                ref_rms = expected_grad.float().square().mean().sqrt().item()
                grad_atol = max(atol, torch.finfo(dtype).eps * ref_rms)
                torch.testing.assert_close(actual_grad, expected_grad, atol=grad_atol, rtol=rtol)
                error = (actual_grad.float() - expected_grad.float()).abs()
                block_gradient_checks.append(
                    {
                        "shape": list(actual_grad.shape),
                        "reference_rms": ref_rms,
                        "error_rms": error.square().mean().sqrt().item(),
                        "max_error": error.max().item(),
                        "atol": grad_atol,
                        "rtol": rtol,
                    }
                )

            del block_reference, block_actual

        def forward(model_fn=model):
            with torch.no_grad():
                return model_fn(x, scale, shift)

        def forward_backward(model_fn=model, model_params=parameters):
            out = model_fn(x, scale, shift)
            return torch.autograd.grad(out, (x, scale, shift) + model_params, dy)

        results.append(
            {
                "method": name,
                "forward_us": graph_time(forward, args.repeats),
                "forward_backward_us": graph_time(forward_backward, args.repeats),
                "forward_host_us": host_time(forward, args.repeats),
                "forward_backward_host_us": host_time(forward_backward, args.repeats),
                "forward_backward_peak_bytes": peak_memory(forward_backward),
                "correctness": "passed",
                "block_gradient_checks": block_gradient_checks,
            }
        )
        print(json.dumps(results[-1]), flush=True)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "shape": shape,
        "condition_shape": condition_shape,
        "dtype": args.dtype,
        "condition_dtype": "float32",
        "ffn": args.ffn,
        "method_order": list(methods),
        "measurement": (
            "median of 5 batches, microseconds per iteration; *_us uses CUDA graphs, "
            "*_host_us includes Python dispatch"
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

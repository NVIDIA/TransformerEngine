# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark the FusedAdam step: regular vs. capturable kernels.

Two modes:

* ``kernel`` (default): times one optimizer step on a list of large FP32
  parameters, like the reproducer in issue #3777. The capturable step is
  captured in a CUDA graph and replayed, as Megatron does with
  ``optimizer_cuda_graph=true``.
* ``e2e``: trains a small GPT-style stack of ``te.TransformerLayer`` (BF16
  params with FP32 master weights) and times a full training step
  (forward + backward + optimizer), with the optimizer step graph-captured.
"""

import argparse
from statistics import median

import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.optimizers import FusedAdam


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("kernel", "e2e"), default="kernel")
    parser.add_argument(
        "--variant",
        choices=("regular", "capturable", "capturable-master", "all"),
        default="all",
        help="kernel mode only: which FusedAdam configuration to time",
    )
    parser.add_argument("--n-tensors", type=int, default=32)
    parser.add_argument("--numel", type=int, default=16 * 2**20)
    parser.add_argument("--layers", type=int, default=8, help="e2e mode")
    parser.add_argument("--hidden", type=int, default=2048, help="e2e mode")
    parser.add_argument("--seq", type=int, default=1024, help="e2e mode")
    parser.add_argument("--batch", type=int, default=4, help="e2e mode")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def _time_cuda(fn, warmup, iters, repeats):
    """Median over ``repeats`` of the mean time of ``iters`` calls, in microseconds."""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    results = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        results.append(start.elapsed_time(end) * 1000.0 / iters)
    return median(results), min(results), max(results)


def _graphed(fn):
    """Capture ``fn`` in a CUDA graph and return a function that replays it."""

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph.replay


def _make_optimizer(params, variant):
    kwargs = dict(lr=1.6e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, adam_w_mode=True)
    if variant == "regular":
        return FusedAdam(params, **kwargs)
    if variant == "capturable":
        return FusedAdam(params, capturable=True, **kwargs)
    return FusedAdam(params, capturable=True, master_weights=True, **kwargs)


def bench_kernel(args, variant):
    """Time one optimizer step on ``n_tensors`` parameters of ``numel`` elements."""

    dtype = torch.bfloat16 if variant == "capturable-master" else torch.float32
    params = [
        torch.nn.Parameter(torch.randn(args.numel, device="cuda", dtype=dtype))
        for _ in range(args.n_tensors)
    ]
    for p in params:
        p.grad = torch.randn_like(p) * 1e-3
    optimizer = _make_optimizer(params, variant)
    optimizer.step()  # initialize optimizer state outside the graph
    step = optimizer.step if variant == "regular" else _graphed(optimizer.step)
    t_med, t_min, t_max = _time_cuda(step, args.warmup, args.iters, args.repeats)
    print(
        f"kernel  variant={variant:<17s} tensors={args.n_tensors} numel={args.numel}"
        f"  step_us median={t_med:9.1f} min={t_min:9.1f} max={t_max:9.1f}"
    )


def bench_e2e(args):
    """Time a full training step of a small TransformerLayer stack."""

    torch.manual_seed(0)
    layers = torch.nn.Sequential(
        *[
            te.TransformerLayer(
                args.hidden,
                4 * args.hidden,
                args.hidden // 128,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                params_dtype=torch.bfloat16,
                seq_length=args.seq,
                micro_batch_size=args.batch,
            )
            for _ in range(args.layers)
        ]
    ).cuda()
    n_params = sum(p.numel() for p in layers.parameters())
    optimizer = FusedAdam(
        layers.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        capturable=True,
        master_weights=True,
    )
    x = torch.randn(args.seq, args.batch, args.hidden, device="cuda", dtype=torch.bfloat16)

    def fwd_bwd():
        # Zero in place: the captured optimizer step reads the gradients at fixed addresses.
        optimizer.zero_grad(set_to_none=False)
        out = layers(x)
        out.float().pow(2).mean().backward()

    fwd_bwd()
    optimizer.step()
    graphed_step = _graphed(optimizer.step)

    def train_step():
        fwd_bwd()
        graphed_step()

    t_med, t_min, t_max = _time_cuda(train_step, args.warmup, args.iters, args.repeats)
    o_med, o_min, o_max = _time_cuda(graphed_step, args.warmup, args.iters, args.repeats)
    print(
        f"e2e  layers={args.layers} hidden={args.hidden} seq={args.seq} batch={args.batch}"
        f" params={n_params / 1e6:.0f}M"
    )
    print(f"e2e  train_step_us   median={t_med:9.1f} min={t_min:9.1f} max={t_max:9.1f}")
    print(f"e2e  optimizer_us    median={o_med:9.1f} min={o_min:9.1f} max={o_max:9.1f}")


def main():
    """Run the selected benchmark."""

    args = parse_args()
    print(f"GPU: {torch.cuda.get_device_name()}")
    if args.mode == "e2e":
        bench_e2e(args)
        return
    variants = (
        ("regular", "capturable", "capturable-master") if args.variant == "all" else (args.variant,)
    )
    for variant in variants:
        bench_kernel(args, variant)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

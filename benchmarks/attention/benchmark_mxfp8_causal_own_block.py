# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark causal MXFP8 attention with ``fp8_dpa_causal_own_block`` against BF16 and the
ordinary MXFP8 path.

Every path runs ``DotProductAttention`` with FusedAttention (cuDNN) on identical BF16 inputs:

* ``bf16``: BF16 cuDNN causal attention;
* ``mxfp8``: ``MXFP8BlockScaling(fp8_dpa=True)``, whose causal output can read future tokens
  through shared V scales;
* ``mxfp8_own_block``: ``MXFP8BlockScaling(fp8_dpa=True, fp8_dpa_causal_own_block=True)``.

For each shape the script reports the first-call time (plan building and JIT compilation),
the steady-state median of the training forward and of forward + backward (CUDA events
around the module call, after warm-up, including quantization and Python launch overhead),
and the slowest GPU kernels of one forward + backward from ``torch.profiler``. The output relative L2 error against BF16
is printed alongside, so a timing is never read without its accuracy.

Example::

    python benchmarks/attention/benchmark_mxfp8_causal_own_block.py \\
        --batch 2 --heads 16 --kv-heads 16 --seqlens 2048 4096 8192
"""

import argparse
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile

from transformer_engine.common import recipe
from transformer_engine.pytorch import DotProductAttention, autocast
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends

HEAD_DIM = 128
PATHS = ("bf16", "mxfp8", "mxfp8_own_block")


def recipe_for(path):
    if path == "bf16":
        return None
    return recipe.MXFP8BlockScaling(
        fp8_dpa=True, fp8_dpa_causal_own_block=path == "mxfp8_own_block"
    )


def make_inputs(args, seqlen):
    generator = torch.Generator(device="cuda").manual_seed(1234)
    if args.format == "bshd":
        q_shape = (args.batch, seqlen, args.heads, HEAD_DIM)
    else:
        q_shape = (seqlen, args.batch, args.heads, HEAD_DIM)
    kv_shape = (*q_shape[:2], args.kv_heads, HEAD_DIM)
    q, k, v = (
        torch.randn(
            shape, dtype=torch.bfloat16, device="cuda", generator=generator
        ).requires_grad_()
        for shape in (q_shape, kv_shape, kv_shape)
    )
    out_grad = torch.randn(
        (*q_shape[:2], args.heads * HEAD_DIM),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    return (q, k, v), out_grad


def make_step(args, path, inputs, out_grad):
    """Return a (forward, forward_backward) pair of closures for one path."""
    module = DotProductAttention(
        args.heads,
        HEAD_DIM,
        num_gqa_groups=args.kv_heads,
        attention_type="self",
        qkv_format=args.format,
    ).cuda()
    fp8_recipe = recipe_for(path)

    def forward():
        with autocast(enabled=fp8_recipe is not None, recipe=fp8_recipe):
            return module(*inputs, qkv_format=args.format, attn_mask_type="causal")

    def forward_backward():
        for x in inputs:
            x.grad = None  # write the gradients instead of accumulating them
        out = forward()
        out.backward(out_grad)
        return out

    return forward, forward_backward


def median_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    torch.cuda.synchronize()
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    times = sorted(start.elapsed_time(end) for start, end in zip(starts, ends))
    return times[len(times) // 2]


def kernel_breakdown(fn, iters, top):
    """The ``top`` GPU kernels of ``fn`` by time, as (name, ms per call), from torch.profiler."""
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    kernels = [
        (event.key, event.device_time_total / 1e3 / iters)
        for event in prof.key_averages()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    kernels.sort(key=lambda kernel: -kernel[1])
    total = sum(ms for _, ms in kernels)
    return kernels[:top], total


def causal_tflops(args, seqlen, ms, backward):
    # Causal attention reads half of the score matrix; the backward costs 2.5x the forward.
    flops = 4 * args.batch * args.heads * seqlen * seqlen * HEAD_DIM / 2
    if backward:
        flops *= 3.5
    return flops / (ms * 1e-3) / 1e12


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=16)
    parser.add_argument("--seqlens", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    parser.add_argument("--format", choices=("bshd", "sbhd"), default="bshd")
    parser.add_argument("--paths", nargs="+", choices=PATHS, default=list(PATHS))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--profile-iters", type=int, default=10)
    parser.add_argument("--top-kernels", type=int, default=8)
    args = parser.parse_args()

    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    import cudnn  # pylint: disable=import-outside-toplevel
    import transformer_engine  # pylint: disable=import-outside-toplevel

    print(
        f"{props.name} (sm{props.major}{props.minor}), torch {torch.__version__}, "
        f"CUDA {torch.version.cuda}, cuDNN {cudnn.backend_version()}, "
        f"cuDNN Frontend {cudnn.__version__}, TE {transformer_engine.__version__}, "
        f"NVTE_ALLOW_NONDETERMINISTIC_ALGO={os.getenv('NVTE_ALLOW_NONDETERMINISTIC_ALGO', '1')}"
    )
    print(
        f"batch={args.batch} heads={args.heads} kv_heads={args.kv_heads} d={HEAD_DIM} "
        f"format={args.format} mask=causal; median of {args.iters} after {args.warmup} warm-up"
    )

    rows = []
    for seqlen in args.seqlens:
        inputs, out_grad = make_inputs(args, seqlen)
        reference = None
        for path in args.paths:
            _attention_backends["backend_selection_requires_update"] = True
            forward, forward_backward = make_step(args, path, inputs, out_grad)
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = forward_backward()
            torch.cuda.synchronize()
            first_s = time.perf_counter() - start
            if path == "bf16":
                reference = out.detach().float()
            error = (
                (out.detach().float() - reference).norm() / reference.norm()
                if reference is not None
                else float("nan")
            )
            fwd_ms = median_ms(forward, args.warmup, args.iters)
            fwd_bwd_ms = median_ms(forward_backward, args.warmup, args.iters)
            kernels, kernel_ms = kernel_breakdown(
                forward_backward, args.profile_iters, args.top_kernels
            )
            rows.append((seqlen, path, first_s, fwd_ms, fwd_bwd_ms, error))
            print(
                f"\ns={seqlen:<6} {path:<16} first call {first_s:7.2f} s | fwd {fwd_ms:7.3f} ms"
                f" ({causal_tflops(args, seqlen, fwd_ms, False):6.1f} TFLOPS) | fwd+bwd"
                f" {fwd_bwd_ms:7.3f} ms"
                f" ({causal_tflops(args, seqlen, fwd_bwd_ms, True):6.1f} TFLOPS) | out rel L2 vs"
                f" bf16 {error:.4f}"
            )
            print(f"    GPU kernel time per fwd+bwd: {kernel_ms:.3f} ms")
            for name, ms in kernels:
                print(f"    {ms:8.3f} ms  {name[:100]}")

    print("\nseqlen  path              fwd ms  fwd+bwd ms  fwd+bwd vs mxfp8  fwd+bwd vs bf16")
    by_key = {(row[0], row[1]): row for row in rows}
    for seqlen, path, _, fwd_ms, fwd_bwd_ms, _ in rows:
        mxfp8 = by_key.get((seqlen, "mxfp8"))
        bf16 = by_key.get((seqlen, "bf16"))
        vs_mxfp8 = f"{fwd_bwd_ms / mxfp8[4]:.2f}x" if mxfp8 else "-"
        vs_bf16 = f"{fwd_bwd_ms / bf16[4]:.2f}x" if bf16 else "-"
        print(
            f"{seqlen:<7} {path:<16} {fwd_ms:7.3f}  {fwd_bwd_ms:10.3f}  {vs_mxfp8:>16} "
            f" {vs_bf16:>15}"
        )


if __name__ == "__main__":
    main()

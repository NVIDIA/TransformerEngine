# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import List
import torch
import time
import argparse

import transformer_engine.pytorch as te


def run_once(
    module: torch.nn.Module,
    args: List[torch.Tensor],
    iters=2000,
    use_te: bool = True,
    is_first_microbatch: bool = True,
):
    if use_te:
        for _ in range(iters):
            module(*args, is_first_microbatch=is_first_microbatch)
    else:
        for _ in range(iters):
            module(*args)


def speedometer(
    module: torch.nn.Module,
    args: List[torch.Tensor],
    timing_iters: int = 2000,
    warmup_iters: int = 100,
    num_rounds: int = 5,
    use_te: bool = True,
) -> float:
    """Measure average run time for a PyTorch module"""
    # warm up
    run_once(module, args, iters=warmup_iters, use_te=use_te, is_first_microbatch=True)

    gpu_times = []
    cpu_times = []
    for round_idx in range(num_rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        cpu_start = time.time()
        # run timing
        run_once(module, args, iters=timing_iters, use_te=use_te, is_first_microbatch=False)
        cpu_end = time.time()
        end.record()
        torch.cuda.synchronize()
        gpu_elapsed = start.elapsed_time(end)
        cpu_elapsed = (cpu_end - cpu_start) * 1000
        gpu_times.append(gpu_elapsed)
        cpu_times.append(cpu_elapsed)
        print(
            f"Round {round_idx+1}/{num_rounds}: GPU {gpu_elapsed/timing_iters*1000:.2f} µs, CPU"
            f" {cpu_elapsed/timing_iters*1000:.2f} µs"
        )
    print(
        f"Average GPU time over {num_rounds} rounds:"
        f" {sum(gpu_times)/(num_rounds*timing_iters)*1000:.2f} µs"
    )
    print(
        f"Average CPU time over {num_rounds} rounds:"
        f" {sum(cpu_times)/(num_rounds*timing_iters)*1000:.2f} µs"
    )

    return sum(gpu_times) / num_rounds


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark torch.nn.Linear performance and CPU overhead."
    )
    parser.add_argument("--hidden_size", type=int, default=3072, help="Hidden size")
    parser.add_argument("--seq_length", type=int, default=2048, help="Sequence length")
    parser.add_argument("--warmup", type=int, default=500, help="Number of warmup iterations")
    parser.add_argument(
        "--timing_iters", type=int, default=2000, help="Number of timing iterations per round"
    )
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of timing rounds")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["torch", "te"],
        default="te",
        help="Linear backend: torch or te",
    )
    args = parser.parse_args()

    x = torch.randn(
        (args.seq_length, args.hidden_size), dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    use_te = True
    if args.backend == "torch":
        model = (
            torch.nn.Linear(args.hidden_size, args.hidden_size, bias=False)
            .to(torch.bfloat16)
            .cuda()
        )
        use_te = False
    else:
        model = te.Linear(args.hidden_size, args.hidden_size, bias=False, device="cuda").to(
            torch.bfloat16
        )
    with torch.no_grad():
        avg_gpu_time_per_round = speedometer(
            model,
            [x],
            timing_iters=args.timing_iters,
            warmup_iters=args.warmup,
            num_rounds=args.num_rounds,
            use_te=use_te,
        )

    total_ops = 2 * args.hidden_size * args.hidden_size * args.seq_length * args.timing_iters

    tflops = total_ops / avg_gpu_time_per_round / 1e9
    print(f"Estimated TFLOP/s: {tflops:.2f}")


if __name__ == "__main__":
    main()

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CPU metadata cost for restoring a grouped layer's saved tensor tuple."""

import argparse
import json
import statistics
import time

import torch

from transformer_engine.pytorch.quantized_tensor import prepare_for_saving, restore_from_saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    assert min(args.experts) > 0 and args.iterations > 0 and args.repeats > 0
    for experts in args.experts:
        captured = []
        weights = [
            torch.nn.Parameter(torch.empty(2, 2), requires_grad=False) for _ in range(experts)
        ]
        values = [None] * experts + weights + weights + [torch.empty(0) for _ in range(experts)]
        saved, metadata = prepare_for_saving(*values)

        class Capture(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(*saved)
                return x.clone()

            @staticmethod
            def backward(ctx, grad):
                captured.append(ctx.saved_tensors)
                return grad

        Capture.apply(torch.ones(1, requires_grad=True)).sum().backward()
        saved_tuple = captured[0]
        for _ in range(10):
            restore_from_saved(metadata, saved_tuple)
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter_ns()
            for _ in range(args.iterations):
                restore_from_saved(metadata, saved_tuple)
            samples.append((time.perf_counter_ns() - start) / args.iterations / 1000)
        print(
            json.dumps(
                {
                    "experts": experts,
                    "entries": len(saved_tuple),
                    "median_us": statistics.median(samples),
                    "samples_us": samples,
                    "torch": torch.__version__,
                    "device": "cpu",
                }
            )
        )


if __name__ == "__main__":
    main()

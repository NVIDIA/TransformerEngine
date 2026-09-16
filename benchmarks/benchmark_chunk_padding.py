# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compare fused chunk sort/pad/restore with TE chunk sort and fused row padding."""

import argparse
import json
import logging
import statistics

import torch

from transformer_engine.pytorch.chunk_padding import (
    moe_sort_chunks_and_pad,
    moe_unpad_and_restore_chunks,
)
from transformer_engine.pytorch.module.fp8_padding import Fp8Padding
from transformer_engine.pytorch.module.fp8_unpadding import Fp8Unpadding
from transformer_engine.pytorch.permutation import (
    moe_sort_chunks_by_index,
    moe_sort_chunks_by_index_with_probs,
)


def benchmark(num_tokens, hidden_size, ranks, experts, alignment):
    """Time the same layout roundtrip and probability padding in A-B-B-A order."""
    generator = torch.Generator().manual_seed(2026)
    counts = torch.bincount(
        torch.randint(ranks * experts, (num_tokens,), generator=generator),
        minlength=ranks * experts,
    ).tolist()
    order = [
        rank * experts + expert for expert in reversed(range(experts)) for rank in range(ranks)
    ]
    sorted_counts = [counts[index] for index in order]
    group_sizes = [
        sum(sorted_counts[start : start + ranks]) for start in range(0, len(order), ranks)
    ]
    padded_counts = sorted_counts.copy()
    for index, size in enumerate(group_sizes):
        padded_counts[(index + 1) * ranks - 1] += (-size) % alignment
    device = torch.device("cuda")
    sizes, indices, padded_sizes = [
        torch.tensor(v, device=device, dtype=torch.int32) for v in (counts, order, padded_counts)
    ]
    sorted_sizes = torch.tensor(sorted_counts, device=device, dtype=torch.int32)
    inverse = torch.argsort(indices).to(torch.int32)
    inp = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    probs = torch.rand(num_tokens, device=device)
    padding, unpadding = Fp8Padding(experts, alignment), Fp8Unpadding(experts, alignment)

    def reference():
        sorted_inp, sorted_probs = moe_sort_chunks_by_index_with_probs(inp, probs, sizes, indices)
        padded, _ = padding(sorted_inp, group_sizes)
        padded_probs, _ = padding(sorted_probs[:, None], group_sizes)
        restored = moe_sort_chunks_by_index(unpadding(padded, group_sizes), sorted_sizes, inverse)
        return restored, padded_probs.flatten()

    def candidate():
        padded, padded_probs, row_map = moe_sort_chunks_and_pad(
            inp, sizes, indices, padded_sizes, sum(padded_counts), probs
        )
        return moe_unpad_and_restore_chunks(padded, row_map, num_tokens), padded_probs

    with torch.no_grad():
        for expected, actual in zip(reference(), candidate()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        records = []
        for name, operation in (
            ("reference", reference),
            ("candidate", candidate),
            ("candidate", candidate),
            ("reference", reference),
        ):
            for _ in range(10):
                operation()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            output = operation()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - before
            del output
            samples = []
            for _ in range(50):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                operation()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            records.append(
                dict(
                    variant=name,
                    median_ms=statistics.median(samples),
                    peak_allocated_bytes=peak,
                    samples_ms=samples,
                )
            )
    return dict(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        ranks=ranks,
        experts=experts,
        alignment=alignment,
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        reference="TE chunk sort + fused row padding/unpadding",
        scope="layout roundtrip only; no GEMM or communication",
        runs=records,
    )


def main():
    """Write raw measurements as JSON for independent analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--alignment", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = [
        benchmark(count, args.hidden_size, args.ranks, args.experts, args.alignment)
        for count in args.tokens
    ]
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(results, output, indent=2)
        output.write("\n")
    logging.getLogger(__name__).info("Saved benchmark results to %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

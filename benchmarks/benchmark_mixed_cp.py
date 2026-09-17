# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compare native and mixed CP attention graph replay on one GPU group."""

import argparse
import json
import logging
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist

from transformer_engine.pytorch.attention import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention.mixed_cp import _is_supported


def main():
    """Measure both variants in ABBA order with the same tensors and process group."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=int, default=32768)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    try:
        cp_size = dist.get_world_size()
        if args.sequence % (2 * cp_size):
            raise ValueError("Sequence length must be divisible by twice the CP size")
        sequence = args.sequence // cp_size
        torch.manual_seed(2026 + dist.get_rank())
        inputs = [
            torch.randn(
                sequence, args.batch, 32, width, device="cuda", dtype=torch.bfloat16
            ).requires_grad_()
            for width in (192, 192, 128)
        ]
        gradient = torch.randn(sequence, args.batch, 32 * 128, device="cuda", dtype=torch.bfloat16)
        if not _is_supported(*inputs, cp_size):
            raise ValueError("This configuration cannot exercise mixed CP backward")
        module = (
            DotProductAttention(
                num_attention_heads=32,
                kv_channels=(192, 128),
                attention_dropout=0,
                qkv_format="sbhd",
                attn_mask_type="causal",
                softmax_scale=0.083,
                cp_group=dist.group.WORLD,
                cp_global_ranks=list(range(cp_size)),
                cp_stream=torch.cuda.Stream(),
                cp_comm_type="p2p",
            )
            .cuda()
            .train()
        )
        records = []
        reference = None
        for variant in ("native", "mixed", "mixed", "native"):
            os.environ["NVTE_FUSED_ATTN_CP_USE_FAv4_BWD"] = "1" if variant == "mixed" else "0"
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    for tensor in inputs:
                        tensor.grad = None
                    output = module(*inputs)
                    output.backward(gradient)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            values = (output.detach().clone(), [t.grad.detach().clone() for t in inputs])
            if reference is None:
                reference = values
            else:
                torch.testing.assert_close(values[0], reference[0], rtol=0, atol=0)
                for value, expected in zip(values[1], reference[1], strict=True):
                    torch.testing.assert_close(value, expected, rtol=0.04, atol=0.025)
                    assert (
                        value.float() - expected.float()
                    ).norm() / expected.float().norm() < 0.008
            del values, output
            for tensor in inputs:
                tensor.grad = None
            dist.barrier()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = module(*inputs)
                output.backward(gradient)
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() - before
            dist.barrier()
            samples = []
            for _ in range(50):
                start = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - start) * 1000)
            ranks = [None] * cp_size
            dist.all_gather_object(
                ranks, dict(samples_ms=samples, capture_incremental_peak_bytes=peak)
            )
            maxima = [max(rank["samples_ms"][i] for rank in ranks) for i in range(50)]
            records.append(dict(variant=variant, median_ms=statistics.median(maxima), ranks=ranks))
            graph.reset()
            del graph, output
            for tensor in inputs:
                tensor.grad = None
            torch.cuda.synchronize()
        if dist.get_rank() == 0:
            report = dict(
                sequence=args.sequence,
                batch=args.batch,
                cp_size=cp_size,
                scope=(
                    "Attention forward/backward CUDA graph replay only; excludes projections, MoE,"
                    " optimizer and checkpoint I/O"
                ),
                timing=(
                    "ABBA; 50 synchronized wall-clock samples per run; maximum across ranks per"
                    " sample"
                ),
                memory="Incremental allocated peak during capture and replay; not full GPU memory",
                runs=records,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            logging.getLogger(__name__).info("Saved %s", args.output)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

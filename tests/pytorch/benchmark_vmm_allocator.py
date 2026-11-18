# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Separate allocator setup/reuse costs from persistent CUDA Graph replay.

This uses simple fills, not MXFP8 quantization or an end-to-end training step.
Each cycle allocates fresh buffers of the next size, captures once, replays N
iterations, then resets and retires the buffers. TE controllers retain physical
handles between cycles. Timed phase boundaries include synchronization; do not
interpret their sum as an unsynchronized training measurement.
"""

import argparse
import json
import time


_COUNTERS = (
    "allocation_requests",
    "free_requests",
    "physical_blocks_created",
    "physical_blocks_reused",
    "physical_blocks_released",
    "map_calls",
    "unmap_calls",
)


def _delta(before, after):
    return {key: after.get(key, 0) - before.get(key, 0) for key in _COUNTERS}


def _fill(buffers):
    for buffer in buffers:
        buffer.fill_(17)


def _measure(torch, function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def _run_mode(torch, mode, args):
    from transformer_engine.pytorch.tensor.vmm import VMMRowSplitAllocator
    from transformer_engine.pytorch.tensor.vmm_mempool import VMMMemPool

    controller = None
    controller_init_ms = 0.0
    if mode != "manual":
        controller, controller_init_ms = _measure(
            torch, lambda: VMMMemPool(mode=mode, block_bytes=args.block_bytes)
        )
    capture_stream = torch.cuda.Stream()
    print(json.dumps({"event": "controller_init", "mode": mode, "wall_ms": controller_init_ms}))

    for cycle in range(args.cycles):
        size_bytes = args.sizes_mib[cycle % len(args.sizes_mib)] * 1024 * 1024
        if controller is not None:
            block = controller.stats()["block_bytes"]
            if size_bytes % (2 * block):
                raise ValueError(f"Size {size_bytes} must be aligned to two blocks ({2 * block})")
        shape = (2, size_bytes // 2)
        manual_owner = None
        stats_before = {} if controller is None else controller.stats()

        def allocate():
            nonlocal manual_owner
            if controller is None:
                manual_owner = VMMRowSplitAllocator(use_mem_pool=False)
                return [manual_owner.allocate(shape, torch.uint8) for _ in range(args.allocations)]
            return [controller.allocate(shape, torch.uint8) for _ in range(args.allocations)]

        buffers, allocation_ms = _measure(torch, allocate)
        _fill(buffers)  # Warm the fill kernel before capture; excluded from phase timing.
        torch.cuda.synchronize()

        def capture():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                _fill(buffers)
            return graph

        graph, capture_ms = _measure(torch, capture)
        before_replay = {} if controller is None else controller.stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        start = time.perf_counter()
        for _ in range(args.replays):
            graph.replay()
        submit_end = time.perf_counter()
        end_event.record()
        end_event.synchronize()
        replay_wall_ms = (time.perf_counter() - start) * 1000
        graph_interval_ms = start_event.elapsed_time(end_event)
        after_replay = {} if controller is None else controller.stats()
        replay_delta = _delta(before_replay, after_replay)
        if controller is not None and any(replay_delta.values()):
            raise AssertionError(
                f"Allocator work unexpectedly occurred during replay: {replay_delta}"
            )
        for buffer in buffers:
            torch.testing.assert_close(buffer, torch.full_like(buffer, 17))
        del buffer

        _, graph_reset_ms = _measure(torch, graph.reset)
        del graph

        def retire():
            buffers.clear()
            if controller is None:
                manual_owner.close()
            else:
                controller.collect()

        _, retire_ms = _measure(torch, retire)
        # The trailing retire fence completes TE's just-recorded events. Poll
        # again to turn completed mappings into cached, unmapped physical handles.
        _, reclaim_ms = _measure(torch, lambda: controller.collect() if controller else None)
        stats_after = {} if controller is None else controller.stats()
        phase_sum_ms = (
            allocation_ms + capture_ms + replay_wall_ms + graph_reset_ms + retire_ms + reclaim_ms
        )
        print(
            json.dumps(
                {
                    "event": "cycle",
                    "mode": mode,
                    "cycle": cycle,
                    "bytes_per_tensor": size_bytes,
                    "allocations": args.allocations,
                    "replays": args.replays,
                    "allocation_phase": "eager_before_capture",
                    "allocation_ms": allocation_ms,
                    "capture_ms": capture_ms,
                    "replay_submit_us_per_iteration": (submit_end - start) * 1e6 / args.replays,
                    "replay_wall_us_per_iteration": replay_wall_ms * 1000 / args.replays,
                    "cuda_graph_loop_interval_us_per_iteration": (
                        graph_interval_ms * 1000 / args.replays
                    ),
                    "graph_reset_ms": graph_reset_ms,
                    "retire_pools_ms": retire_ms,
                    "reclaim_mappings_ms": reclaim_ms,
                    "isolated_phase_sum_us_per_replay": phase_sum_ms * 1000 / args.replays,
                    "replay_allocator_delta": replay_delta if controller else None,
                    "cycle_allocator_delta": (
                        _delta(stats_before, stats_after) if controller else None
                    ),
                    "stats_after_reclaim": stats_after,
                }
            ),
            flush=True,
        )

    if controller is not None:
        _, drain_ms = _measure(torch, controller.empty_cache)
        print(json.dumps({"event": "final_cache_drain", "mode": mode, "wall_ms": drain_ms}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes", nargs="+", choices=("manual", "dedicated", "managed"),
        default=["manual", "dedicated"],
    )
    parser.add_argument("--sizes-mib", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--allocations", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--block-bytes", type=int, default=0)
    args = parser.parse_args()
    if min(args.allocations, args.cycles, args.replays, *args.sizes_mib) <= 0:
        parser.error("sizes, allocations, cycles, and replays must be positive")
    if args.block_bytes < 0:
        parser.error("block-bytes must be nonnegative")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; no CPU fallback is meaningful here")
    torch.cuda.init()
    print(
        json.dumps({
            "event": "environment", "torch": torch.__version__,
            "torch_git": torch.version.git_version, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "config": vars(args),
        }),
        flush=True,
    )
    for mode in args.modes:
        _run_mode(torch, mode, args)


if __name__ == "__main__":
    main()

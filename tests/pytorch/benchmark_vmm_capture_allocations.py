# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Stress allocator work during capture and time first versus steady replay.

Compare the same fill/read GPU work with buffers allocated before capture or
with A -> del A -> B -> del B repeated inside capture. Optional eager cache
seeding measures cached-handle mapping during capture; seeding/unmapping is
reported separately. Current graph pins prevent A-to-B physical sharing inside
one capture. This benchmark reports that limitation instead of claiming reuse.
"""

import argparse
import json
import time

from vmm_benchmark_utils import byte_shape, counter_delta, handles, mib_bytes, split_mapped_bytes


def _measure(torch, function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def _consume(tensor, checks, index, column):
    flat = tensor.view(-1)
    checks[index, column].copy_(flat[0])
    checks[index, column + 1].copy_(flat[-1])


def _body(torch, controller, args, checks, persistent, samples):
    for index in range(args.pairs):
        a = (
            controller.allocate(byte_shape(args.a_bytes), torch.uint8)
            if persistent is None else persistent[2 * index]
        )
        a.fill_(17)
        _consume(a, checks, index, 0)
        if index == 0:
            samples["A"] = controller.allocation_info(a)
        del a
        b = (
            controller.allocate(byte_shape(args.b_bytes), torch.uint8)
            if persistent is None else persistent[2 * index + 1]
        )
        b.fill_(31)
        _consume(b, checks, index, 2)
        if index == 0:
            samples["B"] = controller.allocation_info(b)
        del b


def _allocate_all(torch, controller, args):
    return [
        controller.allocate(byte_shape(size), torch.uint8)
        for _ in range(args.pairs)
        for size in (args.a_bytes, args.b_bytes)
    ]


def _replay_time(torch, graph, repeats):
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    start = time.perf_counter()
    for _ in range(repeats):
        graph.replay()
    submitted = time.perf_counter()
    end.record()
    end.synchronize()
    return {
        "wall_us_per_replay": (time.perf_counter() - start) * 1e6 / repeats,
        "submit_us_per_replay": (submitted - start) * 1e6 / repeats,
        "cuda_loop_interval_us_per_replay": begin.elapsed_time(end) * 1000 / repeats,
    }


def _run(torch, args, scenario, cache_state):
    from transformer_engine.pytorch.tensor.vmm_mempool import VMMMemPool

    controller, init_ms = _measure(
        torch, lambda: VMMMemPool(mode=args.mode, wait_for_reuse=True)
    )
    block = controller.stats()["block_bytes"]
    rounded_a = split_mapped_bytes(args.a_bytes, block)
    rounded_b = split_mapped_bytes(args.b_bytes, block)
    expected_capacity = args.pairs * (rounded_a + rounded_b)
    samples = {}
    persistent = None
    graph = None
    seed_allocation_ms = seed_retirement_ms = preallocation_ms = 0.0
    seed_delta = {}

    if cache_state == "warm":
        seed_before = controller.stats()
        seed, seed_allocation_ms = _measure(torch, lambda: _allocate_all(torch, controller, args))

        def retire_seed():
            seed.clear()
            controller.collect()

        _, seed_retirement_ms = _measure(torch, retire_seed)
        controller.collect()
        seed_after = controller.stats()
        seed_delta = counter_delta(seed_before, seed_after)
        assert seed_after["active_mapped_bytes"] == 0
        assert seed_after["pending_mapped_bytes"] == 0
        assert seed_after["cached_physical_bytes"] == expected_capacity
    if scenario == "preallocated":
        persistent, preallocation_ms = _measure(
            torch, lambda: _allocate_all(torch, controller, args)
        )

    checks = torch.zeros((args.pairs, 4), dtype=torch.uint8, device="cuda")
    expected = torch.tensor(
        [17, 17, 31, 31], dtype=torch.uint8, device="cuda"
    ).repeat(args.pairs, 1)
    capture_stream = torch.cuda.Stream()
    before_capture = controller.stats()
    try:
        def capture():
            result = torch.cuda.CUDAGraph()
            with torch.cuda.graph(result, stream=capture_stream):
                _body(torch, controller, args, checks, persistent, samples)
            return result

        graph, capture_ms = _measure(torch, capture)
        after_capture = controller.stats()
        capture_delta = counter_delta(before_capture, after_capture)
        expected_maps = expected_capacity // block if scenario == "capture" else 0
        assert capture_delta["map_calls"] == expected_maps
        # Neither captured graph mappings nor the persistent baseline buffers
        # are reclaimable at this point under the current allocation policy.
        assert capture_delta["unmap_calls"] == 0
        if cache_state == "warm":
            assert capture_delta["physical_blocks_created"] == 0
            assert capture_delta["physical_blocks_reused"] == expected_maps

        # Start from zero so correctness cannot pass solely due to warmup or
        # capture-time execution. Only replay writes the expected checks.
        checks.zero_()
        before_first = controller.stats()
        first_replay = _replay_time(torch, graph, 1)
        first_delta = counter_delta(before_first, controller.stats())
        assert not any(first_delta.values()), first_delta
        torch.testing.assert_close(checks, expected)
        for _ in range(args.warmup_replays):
            graph.replay()
        torch.cuda.synchronize()
        checks.zero_()
        before_steady = controller.stats()
        steady_replay = _replay_time(torch, graph, args.replays)
        replay_delta = counter_delta(before_steady, controller.stats())
        assert not any(replay_delta.values()), replay_delta
        # Correctness helpers can allocate ordinary PyTorch temporaries, so
        # keep them outside the timed replay/counter windows.
        torch.testing.assert_close(checks, expected)

        first_pair_overlap = len(handles(samples["A"]) & handles(samples["B"]))
        _, reset_ms = _measure(torch, graph.reset)
        graph = None
        if persistent is not None:
            persistent.clear()
        drain_start = time.perf_counter()
        drain_attempts = 0
        while True:
            controller.empty_cache()
            drain_attempts += 1
            if controller.stats()["physical_bytes"] == 0:
                break
            if time.perf_counter() - drain_start > args.drain_timeout:
                raise RuntimeError(f"Graph mapping drain timed out: {controller.stats()}")
            # CUDA user-object release callbacks can finish after reset/sync.
            time.sleep(0.01)
        drain_ms = (time.perf_counter() - drain_start) * 1000
        print(json.dumps({
            "event": "result", "status": "PASS", "mode": args.mode,
            "scenario": scenario, "cache_before_capture": cache_state,
            "pairs": args.pairs, "allocation_sites": 2 * args.pairs,
            "a_logical_bytes": args.a_bytes, "b_logical_bytes": args.b_bytes,
            "a_mapped_bytes": rounded_a, "b_mapped_bytes": rounded_b,
            "block_bytes": block, "controller_init_ms": init_ms,
            "cache_seed_allocation_ms": seed_allocation_ms,
            "cache_seed_retirement_ms": seed_retirement_ms,
            "cache_seed_allocator_delta": seed_delta,
            "preallocation_ms": preallocation_ms, "capture_ms": capture_ms,
            "capture_allocator_delta": capture_delta,
            "first_replay": first_replay, "steady_replay": steady_replay,
            "first_replay_allocator_delta": first_delta,
            "steady_replay_count": args.replays,
            "replay_allocator_delta": replay_delta,
            "first_pair_B_blocks_from_A": first_pair_overlap,
            "intra_capture_reuse_implemented": False,
            "physical_capacity_without_intra_capture_reuse": expected_capacity,
            "stats_after_capture": after_capture,
            "graph_reset_ms": reset_ms,
            "final_pool_and_mapping_drain_ms": drain_ms,
            "drain_attempts": drain_attempts,
        }), flush=True)
    finally:
        if graph is not None:
            graph.reset()
        if persistent is not None:
            persistent.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dedicated", "managed"), default="dedicated")
    parser.add_argument("--a-mib", dest="a_bytes", type=mib_bytes, default="16", metavar="MiB")
    parser.add_argument("--b-mib", dest="b_bytes", type=mib_bytes, default="7.5", metavar="MiB")
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--warmup-replays", type=int, default=3)
    parser.add_argument(
        "--scenarios", choices=("preallocated", "capture"), nargs="+",
        default=["preallocated", "capture"],
    )
    parser.add_argument(
        "--cache-state", choices=("cold", "warm", "both"), default="both",
        help="Physical-cache state for the allocations-during-capture scenario",
    )
    parser.add_argument("--drain-timeout", type=float, default=10.0)
    args = parser.parse_args()
    if min(args.pairs, args.replays, args.drain_timeout) <= 0 or args.warmup_replays < 0:
        parser.error("pairs/replays/drain-timeout must be positive and warmup-replays nonnegative")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    # Warm framework fill/index/copy operations on ordinary buffers, without
    # populating any TE physical cache. Cold/warm below refer to TE's cache.
    ordinary = [
        torch.empty(byte_shape(size), dtype=torch.uint8, device="cuda")
        for size in (args.a_bytes, args.b_bytes)
    ]
    scratch = torch.empty((1, 4), dtype=torch.uint8, device="cuda")
    ordinary[0].fill_(17)
    ordinary[1].fill_(31)
    _consume(ordinary[0], scratch, 0, 0)
    _consume(ordinary[1], scratch, 0, 2)
    torch.cuda.synchronize()
    del ordinary, scratch
    print(json.dumps({
        "event": "environment", "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "config": vars(args), "gpu_work": "fills_and_endpoint_copies",
    }), flush=True)
    for scenario in args.scenarios:
        states = ("cold",) if scenario == "preallocated" else (
            ("cold", "warm") if args.cache_state == "both" else (args.cache_state,)
        )
        for state in states:
            _run(torch, args, scenario, state)


if __name__ == "__main__":
    main()

# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Check eager A -> del A -> B reuse through PyTorch Storage ownership.

No CUDA Graph, explicit TE free/collect, empty_cache, or gc.collect occurs
between deleting A and allocating B. This diagnostic enables wait_for_reuse:
TE may wait for a retired allocation's completion event before unmapping it.
This establishes lifetime/reuse behavior, not nonblocking allocator performance.
"""

import argparse
import json
import time


from vmm_benchmark_utils import byte_shape, describe, handles, mib_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dedicated", "managed"), default="dedicated")
    parser.add_argument("--a-mib", dest="a_bytes", type=mib_bytes, default="16", metavar="MiB")
    parser.add_argument("--b-mib", dest="b_bytes", type=mib_bytes, default="8", metavar="MiB")
    args = parser.parse_args()
    if args.a_bytes < args.b_bytes:
        parser.error("Require a-mib >= b-mib > 0 so A can supply all of B's blocks")

    import torch
    from transformer_engine.pytorch.tensor.vmm_mempool import VMMMemPool

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    pool = VMMMemPool(mode=args.mode, wait_for_reuse=True)
    block = pool.stats()["block_bytes"]
    a_bytes, b_bytes = args.a_bytes, args.b_bytes
    print(json.dumps({
        "event": "environment", "mode": args.mode,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "block_bytes": block,
        "wait_for_reuse": True,
    }), flush=True)

    start = time.perf_counter()
    a = pool.allocate(byte_shape(a_bytes), torch.uint8)
    a_allocate_ms = (time.perf_counter() - start) * 1000
    a.fill_(17)
    torch.cuda.synchronize()  # Finish A's GPU use before testing its Storage lifetime.
    # These snapshots contain only numbers; they do not keep A/Storage alive.
    info_a = pool.allocation_info(a)
    after_a = pool.stats()
    print(json.dumps({
        "event": "allocated_A", "allocation": describe(info_a), "stats": after_a,
    }), flush=True)

    # The experiment: PyTorch owns the Storage deletion. No manual TE release.
    start = time.perf_counter()
    del a
    delete_ms = (time.perf_counter() - start) * 1000
    after_delete = pool.stats()
    # In dedicated mode, B's normal allocation performs automatic dead-pool
    # collection. In managed mode, PyTorch directly forwards completed frees.
    start = time.perf_counter()
    b = pool.allocate(byte_shape(b_bytes), torch.uint8)
    b_allocate_ms = (time.perf_counter() - start) * 1000
    torch.cuda.synchronize()
    b_ready_ms = (time.perf_counter() - start) * 1000
    info_b = pool.allocation_info(b)
    after_b = pool.stats()
    print(json.dumps({"event": "after_del_A", "stats": after_delete}), flush=True)
    print(json.dumps({
        "event": "allocated_B", "allocation": describe(info_b), "stats": after_b,
    }), flush=True)

    a_handles, b_handles = handles(info_a), handles(info_b)
    reused = len(a_handles & b_handles)
    new_blocks = after_b["physical_blocks_created"] - after_a["physical_blocks_created"]
    maps = after_b["map_calls"] - after_a["map_calls"]
    unmaps = after_b["unmap_calls"] - after_a["unmap_calls"]
    frees = after_b["free_requests"] - after_a["free_requests"]
    assert b_handles <= a_handles, "B did not reuse A's physical handles in matching domains"
    assert new_blocks == 0, f"B created {new_blocks} new physical blocks"
    assert maps == len(info_b["pages"]), "B must create fresh mappings, not reuse a mapped slot"
    assert unmaps == len(info_a["pages"]), "A's old mapping was not fully retired"
    assert frees == 1, f"Expected one completed free callback for A, got {frees}"
    assert after_b["physical_blocks_released"] == after_a["physical_blocks_released"]
    assert after_b["physical_bytes"] == after_a["physical_bytes"], "Physical capacity grew"
    assert after_b["physical_blocks_reused"] - after_a["physical_blocks_reused"] == reused
    b.fill_(31)
    torch.testing.assert_close(b, torch.full_like(b, 31))
    print(json.dumps({
        "event": "result", "status": "PASS", "mode": args.mode,
        "a_blocks": len(a_handles), "b_blocks": len(b_handles),
        "b_blocks_from_A": reused, "new_physical_blocks_for_B": new_blocks,
        "fresh_B_map_calls": maps, "A_unmap_calls": unmaps,
        "free_callbacks_on_del_A": after_delete["free_requests"] - after_a["free_requests"],
        "free_callbacks_by_B_allocation": frees,
        "event_waits": after_b["event_wait_calls"] - after_a["event_wait_calls"],
        "allocate_A_host_ms": a_allocate_ms, "del_A_host_ms": delete_ms,
        "allocate_B_host_ms": b_allocate_ms, "allocate_B_ready_ms": b_ready_ms,
        "same_numeric_virtual_address": info_a["base"] == info_b["base"],
    }), flush=True)
    # Cleanup only after the experiment and checks are complete.
    del b
    pool.empty_cache()


if __name__ == "__main__":
    main()

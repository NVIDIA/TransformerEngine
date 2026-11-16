# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformer_engine.pytorch import distributed as te_dist


def _check_cache(rank, init_file):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        device_id=torch.device("cuda", rank),
    )
    try:
        group = dist.group.WORLD
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Ordinary all-reduces with varying row counts must not retain one buffer per size.
            for rows in range(1, 33):
                tensor = torch.full((rows, 256), rank + 1, device="cuda", dtype=torch.bfloat16)
                te_dist.symmetric_all_reduce(tensor, group, all_reduce_type="two_shot")
                assert torch.equal(tensor, torch.full_like(tensor, 3))
            buffers = list(te_dist.symmetric_mem_cache.values())
            assert sum(t.numel() for t in buffers) < 8 * tensor.numel()

            # Capture two sizes sharing a capacity, then grow before replaying the old graphs.
            graphs = []
            for rows in (23, 24):
                inp = torch.empty(rows, 256, device="cuda", dtype=torch.bfloat16)
                out = torch.empty_like(inp)
                for _ in range(3):
                    inp.fill_(rank + 1)
                    te_dist.symmetric_all_reduce(inp, group, all_reduce_type="two_shot")
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    inp.fill_(rank + 1)
                    te_dist.symmetric_all_reduce(inp, group, all_reduce_type="two_shot")
                    out.copy_(inp)
                graphs.append((graph, inp, out))
            large = torch.full((128, 256), rank + 1, device="cuda", dtype=torch.bfloat16)
            te_dist.symmetric_all_reduce(large, group, all_reduce_type="two_shot")
            for _ in range(3):
                for graph, _, out in reversed(graphs):
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(out, torch.full_like(out, 3))
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two peer-accessible GPUs")
def test_symmetric_cache_capacity_and_graph_replay():
    """Exercise cache growth and graph replay in fresh distributed processes."""
    if not te_dist.HAS_TORCH_SYMMETRIC:
        pytest.skip("Requires PyTorch symmetric memory support")
    if not (torch.cuda.can_device_access_peer(0, 1) and torch.cuda.can_device_access_peer(1, 0)):
        pytest.skip("Requires peer access between GPUs 0 and 1")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_check_cache, args=(os.path.join(directory, "rendezvous"),), nprocs=2, join=True)

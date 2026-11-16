# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Hardware regression tests for the optional allocator-managed VMM backend."""

import gc

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def pool():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not hasattr(torch._C, "_cuda_customManagedPoolAllocator"):
        pytest.skip("PyTorch allocator-managed MemPool support is required")
    import transformer_engine_torch as tex

    if not hasattr(tex, "VMMMemPoolBackend"):
        pytest.skip("Rebuild TE with NVTE_WITH_VMM_MEMPOOL=1")
    get_domains = getattr(torch.cuda.memory, "get_num_locality_domains", None)
    if get_domains is None or get_domains() < 2:
        pytest.skip("Two CUDA locality domains are required")
    from transformer_engine.pytorch.tensor.vmm_mempool import VMMMemPool

    result = VMMMemPool()
    yield result
    gc.collect()
    result.empty_cache()
    from transformer_engine.pytorch.tensor.vmm import clear_vmm_mem_pool_caches

    clear_vmm_mem_pool_caches()


def test_physical_blocks_remapped_across_sizes(pool):
    block = pool.stats()["block_bytes"]
    with pool.use():
        large = torch.empty(4 * block, dtype=torch.uint8, device="cuda")
    from transformer_engine.pytorch.tensor.vmm import is_vmm_tensor

    assert is_vmm_tensor(large)
    large.fill_(7)
    torch.cuda.synchronize()
    before = pool.stats()
    del large
    with pool.use():
        small = torch.empty(2 * block, dtype=torch.uint8, device="cuda")
    after = pool.stats()
    assert after["physical_blocks_created"] == before["physical_blocks_created"]
    assert after["physical_blocks_reused"] == before["physical_blocks_reused"] + 2
    assert after["map_calls"] == before["map_calls"] + 2
    assert after["unmap_calls"] == before["unmap_calls"] + 4
    # A driver may recycle a freed VA numerically. The contract is new mappings,
    # not a permanently unique pointer value.
    small.fill_(3)
    torch.testing.assert_close(small, torch.full_like(small, 3))


def test_storage_alias_survives_root_and_allocator_close(pool):
    from transformer_engine.pytorch.tensor.vmm import VMMRowSplitAllocator, is_vmm_tensor

    allocator = VMMRowSplitAllocator(use_mem_pool=True)
    tensor = allocator.allocate((2, pool.stats()["block_bytes"]), torch.uint8)
    tensor.fill_(11)
    view = tensor[1:]
    del tensor
    allocator.close()
    gc.collect()
    assert is_vmm_tensor(view)
    torch.testing.assert_close(view, torch.full_like(view, 11))
    # Views retain the real Storage deleter, without copying a Python attribute.
    assert not hasattr(view, "_nvte_vmm_allocator")
    with pytest.raises(RuntimeError, match="closed"):
        allocator.allocate((2, 1024), torch.uint8)


def test_recorded_stream_use(pool):
    block = pool.stats()["block_bytes"]
    allocation_stream = torch.cuda.Stream()
    consumer_stream = torch.cuda.Stream()
    with torch.cuda.stream(allocation_stream), pool.use():
        tensor = torch.empty(4 * block, dtype=torch.uint8, device="cuda")
        tensor.fill_(19)
    consumer_stream.wait_stream(allocation_stream)
    with torch.cuda.stream(consumer_stream):
        # Ordinary output must remain valid while the localized source is freed.
        output = tensor.clone()
        tensor.record_stream(consumer_stream)
    del tensor
    with pool.use():
        replacement = torch.empty(2 * block, dtype=torch.uint8, device="cuda")
        replacement.fill_(31)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 19))


def test_capture_temporaries_and_shared_graph_retention(pool):
    block = pool.stats()["block_bytes"]
    source = torch.full((2 * block,), 13, dtype=torch.uint8, device="cuda")
    graphs, outputs = [], []
    # Warm up kernels before capture; these allocations use the default pool.
    source.clone().add_(1)
    torch.cuda.synchronize()
    for _ in range(2):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            temporary = torch.empty_like(source)
            temporary.copy_(source)
            output = temporary + 1
            del temporary  # Must not unmap an address recorded in the graph.
        graphs.append(graph)
        outputs.append(output)
    assert pool.stats()["captured_retired_bytes"] >= 4 * block
    for _ in range(3):
        for graph in graphs:
            graph.replay()
        for output in outputs:
            torch.testing.assert_close(output, source + 1)
    with pytest.raises(RuntimeError):
        pool.empty_cache()
    graphs[0].reset()
    with pytest.raises(RuntimeError):
        pool.empty_cache()  # The second graph still owns the pool.
    graphs[1].reset()
    pool.empty_cache()
    assert pool.stats()["captured_retired_bytes"] == 0
    for output in outputs:
        torch.testing.assert_close(output, source + 1)  # Live outputs survive drain.


@pytest.mark.parametrize("domain", [0, 1])
def test_single_domain_uses_same_storage_lifetime(pool, domain):
    from transformer_engine.pytorch.tensor.vmm import (
        VMMRowSplitAllocator,
        get_vmm_mem_pool,
        is_vmm_tensor,
    )

    allocator = VMMRowSplitAllocator(use_mem_pool=True)
    tensor = allocator.allocate_in_domain((17, 19), torch.float32, domain)
    tensor.fill_(domain + 1)
    view = tensor[:, 1:]
    del tensor
    allocator.close()
    assert is_vmm_tensor(view)
    torch.testing.assert_close(view, torch.full_like(view, domain + 1))
    del view
    get_vmm_mem_pool(domain=domain).empty_cache()


def test_pool_wrapper_can_die_before_storage(pool):
    from transformer_engine.pytorch.tensor.vmm import is_vmm_tensor
    from transformer_engine.pytorch.tensor.vmm_mempool import VMMMemPool

    other_pool = VMMMemPool()
    with other_pool.use():
        root = torch.empty(2 * pool.stats()["block_bytes"], dtype=torch.uint8, device="cuda")
    root.fill_(23)
    alias = root[1:]
    del root, other_pool
    gc.collect()
    assert is_vmm_tensor(alias)
    torch.testing.assert_close(alias, torch.full_like(alias, 23))
    del alias
    torch.cuda.empty_cache()

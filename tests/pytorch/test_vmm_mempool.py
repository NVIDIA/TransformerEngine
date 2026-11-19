# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Hardware regression tests for VMM reuse through ordinary PyTorch MemPools."""

import gc
import time

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def pool():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    import transformer_engine_torch as tex

    if not hasattr(tex, "VMMAllocationTicket"):
        pytest.skip("Rebuild TE with NVTE_WITH_VMM_MEMPOOL=1")
    from transformer_engine.pytorch.tensor.vmm_mempool import VMMMemPool

    try:
        result = VMMMemPool()
    except RuntimeError as exc:
        if (
            "requires a CUDA 13.4" in str(exc)
            or "requires at least two locality domains" in str(exc)
        ):
            pytest.skip(str(exc))
        raise
    yield result
    gc.collect()
    result.empty_cache()
    from transformer_engine.pytorch.tensor.vmm import clear_vmm_mem_pool_caches

    clear_vmm_mem_pool_caches()


def test_physical_blocks_remapped_across_sizes(pool):
    block = pool.stats()["block_bytes"]
    large = pool.allocate((4 * block,), torch.uint8)
    from transformer_engine.pytorch.tensor.vmm import is_vmm_tensor

    assert is_vmm_tensor(large)
    large.fill_(7)
    torch.cuda.synchronize()
    before = pool.stats()
    del large
    pool.collect()
    torch.cuda.synchronize()  # Includes the completion event recorded by native segment return.
    small = pool.allocate((2 * block,), torch.uint8)
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
    with torch.cuda.stream(allocation_stream):
        tensor = pool.allocate((4 * block,), torch.uint8)
        tensor.fill_(19)
    consumer_stream.wait_stream(allocation_stream)
    with torch.cuda.stream(consumer_stream):
        # Ordinary output must remain valid while the localized source is freed.
        output = tensor.clone()
        tensor.record_stream(consumer_stream)
    del tensor
    replacement = pool.allocate((2 * block,), torch.uint8)
    replacement.fill_(31)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 19))


def test_capture_temporaries_and_shared_graph_retention(pool):
    block = pool.stats()["block_bytes"]
    source = torch.full((2 * block,), 13, dtype=torch.uint8, device="cuda")
    graphs, outputs = [], []
    source.clone().add_(1)
    torch.cuda.synchronize()
    for _ in range(2):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            temporary = pool.allocate(source.shape, source.dtype)
            temporary.copy_(source)
            output = temporary + 1
            del temporary
        graphs.append(graph)
        outputs.append(output)
    physical_before = pool.stats()["physical_bytes"]
    assert physical_before >= 4 * block
    # An early cache drain must preserve captured addresses, even on PyTorch
    # builds whose use_mem_pool does not retain custom pools for the graph.
    if pool.mode == "managed":
        with pytest.raises(RuntimeError):
            pool.empty_cache()
    else:
        pool.empty_cache()
    assert pool.stats()["physical_bytes"] == physical_before
    for _ in range(3):
        for graph in graphs:
            graph.replay()
        for output in outputs:
            torch.testing.assert_close(output, source + 1)
    graphs[0].reset()
    if pool.mode == "managed":
        with pytest.raises(RuntimeError):
            pool.empty_cache()
    else:
        pool.empty_cache()
    assert pool.stats()["physical_bytes"] >= 2 * block
    graphs[1].replay()
    torch.testing.assert_close(outputs[1], source + 1)
    graphs[1].reset()
    # CUDA user-object destruction is asynchronous, including after reset/sync.
    deadline = time.monotonic() + 5
    while True:
        pool.empty_cache()
        if pool.stats()["physical_bytes"] == 0:
            break
        assert time.monotonic() < deadline, pool.stats()
        time.sleep(0.01)
    for output in outputs:
        torch.testing.assert_close(output, source + 1)


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
    root = other_pool.allocate((2 * pool.stats()["block_bytes"],), torch.uint8)
    root.fill_(23)
    alias = root[1:]
    del root, other_pool
    gc.collect()
    assert is_vmm_tensor(alias)
    torch.testing.assert_close(alias, torch.full_like(alias, 23))
    del alias
    torch.cuda.empty_cache()


def test_small_tensor_does_not_physically_back_native_segment_tail(pool):
    tiny = pool.allocate((2, 16), torch.uint8)
    stats = pool.stats()
    assert stats["physical_bytes"] == 2 * stats["block_bytes"]
    assert stats["virtual_reserved_bytes"] >= stats["physical_bytes"]
    tiny.fill_(29)
    torch.testing.assert_close(tiny, torch.full_like(tiny, 29))


def test_empty_cache_preserves_live_tensor(pool):
    tensor = pool.allocate((2, 128), torch.uint8)
    tensor.fill_(37)
    pool.empty_cache()
    assert pool.contains(tensor.data_ptr())
    torch.testing.assert_close(tensor, torch.full_like(tensor, 37))


def test_delete_a_reuses_handles_for_b_without_manual_collect(pool):
    from transformer_engine.pytorch.tensor.vmm_mempool import VMMMemPool

    eager = VMMMemPool(mode=pool.mode, wait_for_reuse=True)
    block = eager.stats()["block_bytes"]
    a = eager.allocate((2, 2 * block), torch.uint8)
    a.fill_(17)
    torch.cuda.synchronize()
    info_a = eager.allocation_info(a)
    before = eager.stats()
    del a
    # No explicit TE free, collect, empty_cache, or gc between deletion and B.
    b = eager.allocate((2, block), torch.uint8)
    info_b = eager.allocation_info(b)
    after = eager.stats()
    handles_a = {(page["domain"], page["handle"]) for page in info_a["pages"]}
    handles_b = {(page["domain"], page["handle"]) for page in info_b["pages"]}
    assert handles_b <= handles_a
    assert after["physical_blocks_created"] == before["physical_blocks_created"]
    assert after["physical_blocks_reused"] == before["physical_blocks_reused"] + 2
    assert after["free_requests"] == before["free_requests"] + 1
    assert after["map_calls"] == before["map_calls"] + 2
    assert after["unmap_calls"] == before["unmap_calls"] + 4
    assert after["physical_bytes"] == before["physical_bytes"]
    b.fill_(31)
    torch.testing.assert_close(b, torch.full_like(b, 31))


def test_live_view_prevents_reuse_until_last_storage_owner_dies(pool):
    from transformer_engine.pytorch.tensor.vmm_mempool import VMMMemPool

    eager = VMMMemPool(mode=pool.mode, wait_for_reuse=True)
    block = eager.stats()["block_bytes"]
    a = eager.allocate((2, 2 * block), torch.uint8)
    a.fill_(17)
    view = a.view(-1)
    info_a = eager.allocation_info(a)
    before = eager.stats()
    del a
    b = eager.allocate((2, block), torch.uint8)
    info_b = eager.allocation_info(b)
    after_b = eager.stats()
    handles_a = {(page["domain"], page["handle"]) for page in info_a["pages"]}
    handles_b = {(page["domain"], page["handle"]) for page in info_b["pages"]}
    assert handles_a.isdisjoint(handles_b)
    assert after_b["free_requests"] == before["free_requests"]
    torch.testing.assert_close(view, torch.full_like(view, 17))
    # B remains alive; after dropping the last alias only A can supply C's blocks.
    del view
    c = eager.allocate((2, block), torch.uint8)
    info_c = eager.allocation_info(c)
    after_c = eager.stats()
    handles_c = {(page["domain"], page["handle"]) for page in info_c["pages"]}
    assert handles_c <= handles_a
    assert handles_c.isdisjoint(handles_b)
    assert after_c["physical_blocks_created"] == after_b["physical_blocks_created"]
    assert after_c["free_requests"] == before["free_requests"] + 1
    c.fill_(23)
    b.fill_(31)
    torch.testing.assert_close(c, torch.full_like(c, 23))
    torch.testing.assert_close(b, torch.full_like(b, 31))


@pytest.mark.parametrize("b_mib", ["3", "7", "7.5"])
def test_non_aligned_tensor_size_reuses_rounded_backing(pool, b_mib):
    from transformer_engine.pytorch.tensor.vmm_mempool import VMMMemPool
    from vmm_benchmark_utils import byte_shape, handles, mib_bytes, split_mapped_bytes

    eager = VMMMemPool(mode=pool.mode, wait_for_reuse=True)
    a = eager.allocate((2, 8 * 1024 * 1024), torch.uint8)
    info_a = eager.allocation_info(a)
    before = eager.stats()
    del a
    logical = mib_bytes(b_mib)
    b = eager.allocate(byte_shape(logical), torch.uint8)
    info_b = eager.allocation_info(b)
    after = eager.stats()
    block = after["block_bytes"]
    mapped = split_mapped_bytes(logical, block)
    assert b.numel() == logical
    assert info_b["logical_bytes"] == logical
    assert len(info_b["pages"]) * block == mapped
    assert handles(info_b) <= handles(info_a)
    assert after["physical_blocks_created"] == before["physical_blocks_created"]
    assert after["map_calls"] == before["map_calls"] + mapped // block
    boundary = next(page["offset"] for page in info_b["pages"] if page["domain"] == 1)
    assert boundary == mapped // 2
    b.fill_(43)
    torch.testing.assert_close(b, torch.full_like(b, 43))

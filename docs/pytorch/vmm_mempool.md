# Localized VMM tensors with physical-block reuse

This optional prototype extends `VMMRowSplitAllocator` with allocator-managed
PyTorch MemPool ownership. Each logical allocation reserves a contiguous virtual
range and maps fixed-size physical blocks from LD0 followed by LD1. Completed
eager frees eventually unmap that range and put the physical handles into
per-domain caches. A later, differently sized tensor selects blocks from those
caches and maps them into a newly reserved virtual range. It never centers a
small tensor in a cached large mapped slot.

A fresh reservation may numerically reuse an address the CUDA driver previously
freed. The contract is fresh mappings with reusable physical handles, rather than
permanent numerical pointer uniqueness.

## Build and enable

Use a CUDA 13.4+ driver and a GPU with at least two locality domains. Build TE
against a PyTorch build exposing `MemPool(allocator_managed=True)`,
`createCustomManagedPoolAllocator` in its C++ headers, graph retention for custom
MemPools, and `torch.cuda.memory.empty_allocator_cache`. The workspace PyTorch
branch `update_for_26.09_rubin`, revision
`b0fd8533ad1928cb65d7de8b6ced663aa9721e61`, contains the interface inspected for
this implementation. Stock PyTorch builds without that interface are unsupported;
ordinary MemPool segment caching cannot enforce this per-tensor placement.

Build with `NVTE_WITH_VMM_MEMPOOL=1` added to your usual TE build environment.
Regular TE builds keep this backend disabled and need no new PyTorch interface.
At runtime set:

```sh
export NVTE_VMM_USE_MEMPOOL=1
```

Existing MXFP8/NVFP4 workspace and producer allocations through
`VMMRowSplitAllocator.allocate` and `allocate_in_domain` then use the new backend.
This does not enable localization itself: keep the existing localization flags
for the workload. There is no Megatron source change.

Explicit allocation and direct MemPool usage are also available:

```python
import torch
from transformer_engine.pytorch.tensor.vmm import (
    VMMRowSplitAllocator,
    get_vmm_mem_pool,
    clear_vmm_mem_pool_caches,
)

allocator = VMMRowSplitAllocator("cuda", use_mem_pool=True)
x = allocator.allocate((4096, 4096), torch.bfloat16)
view = x[2048:]
allocator.close()  # x and view remain valid: their Storage owns the allocation.

pool = get_vmm_mem_pool("cuda")
with pool.use():
    y = torch.empty((4096, 4096), dtype=torch.bfloat16, device="cuda")
print(pool.stats())

# Explicitly prepare each single-domain pool needed during capture:
get_vmm_mem_pool("cuda", domain=0)
get_vmm_mem_pool("cuda", domain=1)
```

Pools are shared by device and layout, so independently created TE workspaces can
reuse backing. The native backend caches one block per CUDA mapping granularity
by default. `VMMMemPool(device, domain=-1, block_bytes=...)` in `vmm_mempool.py`
constructs an independent pool with a configurable block size, which must be a
power of two and a multiple of both domains' minimum granularities. Use a CUDA device context or
`pool.use()` on the owning device; cross-device accesses and peer mappings are
not implemented.

## Ownership and capture

PyTorch tracks Tensor/Storage aliases, allocation streams, and `record_stream`.
Follow normal PyTorch stream rules: wait for producers and record side-stream
uses before dropping the tensor. The backend adds a completion event on the
allocation stream before any eager unmap or physical reuse. It polls those events
on subsequent eager allocations. This permits cross-stream reuse once completion
is established, without a device synchronization per free.

During capture the backend neither queries completion events nor reclaims
mappings. Any allocation created during capture, or freed while its allocation
stream is capturing, retains its mapping even after the Tensor dies. This
conservative quarantine supports repeated static graph replay; it does not reuse
captured temporary backing within a capture. External eager tensors referenced
by a graph must remain alive until graph reset, as with normal PyTorch graphs.

Prepare pools before capture, then either pass the pool directly or use
`pool.use()` within capture:

```python
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, pool=pool):
    temporary = torch.empty_like(y)
    temporary.copy_(y)
    output = temporary + 1
    del temporary  # Its captured address stays mapped.
graph.replay()
graph.reset()
del output
pool.empty_cache()
```

`pool.empty_cache()` delegates to PyTorch's guarded managed-pool cache drain.
PyTorch rejects it inside a pool scope, during capture, or while any graph retains
the pool. A valid drain synchronizes the owning device, unmaps retired allocations,
and releases cached physical handles. Live Tensor/Storage mappings remain valid.
`clear_vmm_mem_pool_caches()` applies this to the shared TE pools;
`clear_captured_vmm_allocations()` also calls it, preserving the existing TE and
Megatron graph-teardown entry point. Reset **every** graph using a shared pool
before that teardown. Closing a TE allocator wrapper stops further allocation
but does not drain shared caches or invalidate its live tensors.

CUDA Graph Tree checkpoint/restore and CUDA IPC are unsupported. Allocation and
free callbacks stay in C++ and never enter Python or reenter PyTorch's allocator.
On physical OOM, the backend releases unmapped cached handles and retries once;
it does not evict live/captured mappings or release PyTorch's unrelated cache.

## Placement and measurements

The physical midpoint is rounded up to a block boundary, matching the existing
TE prototype. Nonaligned logical halves can therefore place less than one block
of the second logical half in LD0. Small tensors can reside entirely in LD0.
Use aligned shapes when exact half-split locality matters. Single-domain
allocations round the total byte count up to a block boundary.

`pool.stats()` separates live mappings, pending eager mappings, retired captured
mappings, and unmapped cached physical bytes. It also reports physical block
creation/reuse/release and map/unmap counts. PyTorch statistics report logical
allocations and exclude this backend's internal cache. Physical-block reuse
reduces large-slot waste but adds mapping operations; it is not yet a measured
performance improvement. Static graphs amortize setup mappings over replay.

Run the host-only policy tests described in `tests/cpp/vmm/README.md`, then on
supported hardware run:

```sh
python -m pytest -v tests/pytorch/test_vmm_mempool.py
NVTE_VMM_USE_MEMPOOL=1 python -m pytest -v \
    tests/pytorch/mxfp8/test_mxfp8_localization.py \
    tests/pytorch/nvfp4/test_nvfp4_localization.py
```

Compare ordinary/manual-VMM/MemPool end-to-end results and latencies, physical
high-water marks, mapping counts, and capture/replay costs separately. The host
policy tests do not validate CUDA placement, PyTorch ABI linkage, kernel
correctness, or GPU performance.

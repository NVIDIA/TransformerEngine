# Localized VMM tensors with ordinary PyTorch MemPools

The default dedicated mode uses ordinary PyTorch `MemPool`, `use_mem_pool`,
and the C++ `createCustomAllocator` interface. It does not require a PyTorch
source change. An optional managed mode uses one shared MemPool per controller
when a patched PyTorch provides allocator-managed routing. Both modes share the
same TE physical-block reuse policy. Rebuild TE against the PyTorch installed
in the target container; do not build against an unrelated PyTorch checkout.

## Build and run

Requirements: PyTorch's native CUDA allocator backend, ordinary MemPool support,
a CUDA 13.4+ driver, and a GPU with at least two locality domains. Graph lifetime
pins use standard CUDA graph user objects. The alternate `cudaMallocAsync`
allocator backend is unsupported.

Add `NVTE_WITH_VMM_MEMPOOL=1` to your usual TE source-build environment. At runtime:

```sh
export NVTE_VMM_USE_MEMPOOL=1
```

Choose the integration path explicitly when comparing results:

```sh
export NVTE_VMM_MEMPOOL_MODE=dedicated  # Ordinary PyTorch; the default.
# Or, only in the patched PyTorch container:
export NVTE_VMM_MEMPOOL_MODE=managed
```

Managed mode requires `MemPool(allocator_managed=True)` and
`torch.cuda.memory.empty_allocator_cache(pool)`, including their native
allocation/free/stream/graph semantics. TE registers its allocator using the
ordinary C++ `createCustomAllocator` and `set_reset_fn` APIs, so the optional
`createCustomManagedPoolAllocator` factory and `_cuda_customManagedPoolAllocator`
Python helper are not dependencies. PyTorch does not need to implement a VMM
half-split allocator: TE owns placement, mapping, and physical handle caches.

Existing MXFP8/NVFP4 workspace allocations through `VMMRowSplitAllocator.allocate`
and `allocate_in_domain` then use this implementation. Keep the workload's
existing localization flags. No PyTorch or Megatron source change is needed.
The native extension must be rebuilt; an older extension exposing only
`VMMMemPoolBackend` does not implement the new allocation-ticket interface.

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

controller = get_vmm_mem_pool("cuda")
y = controller.allocate((4096, 4096), torch.bfloat16)
print(controller.stats())
```

`get_vmm_mem_pool` now returns a TE controller, not a `torch.cuda.MemPool`
subclass. Call `controller.allocate(shape, dtype)`. The previous experimental
`controller.use()` / `torch.cuda.graph(pool=controller)` examples no longer apply.
This API intentionally does not route arbitrary operations' allocations through
one shared split-backed MemPool.

Controllers are shared by device, integration mode, and layout (`domain=-1` for split backing,
`0` or `1` for single-domain backing). Prepare controllers before capture:

```python
get_vmm_mem_pool("cuda", domain=0)
get_vmm_mem_pool("cuda", domain=1)
```

`VMMMemPool(device, domain=-1, block_bytes=...)` in `vmm_mempool.py` creates an
independent controller. The block size defaults to the common CUDA minimum
mapping granularity; a custom value must be a power of two and a multiple of
both domains' granularities. Controllers and native cache owners are registered
for process-lifetime cleanup/lookup, so dropping a Python variable during capture
cannot destroy its pending native pools. Global cache draining also reaches
controllers the caller has dropped.

## Shared managed mode

With `mode="managed"`, each controller owns one patched native MemPool. Its
allocation method makes a normal typed `torch.empty` inside that pool. PyTorch
forwards every unrounded logical allocation to TE, bypassing its mapped-block
search, rounding, splitting, and caching. Its normal Storage/recorded-stream
bookkeeping still controls when the free callback is delivered. There are no
per-tensor tickets, dedicated-pool scans, or per-tensor native pool teardown.
TE still creates fresh virtual mappings and reuses physical blocks.

Managed cache draining goes through PyTorch's guarded `empty_allocator_cache`:
it drains deferred frees and calls the registered reset hook without touching
live allocations. It must reject capture or a pool retained by a live graph.
TE's CUDA user-object pins additionally protect captured mappings. Static CUDA
Graphs are supported by design; CUDA Graph Tree checkpoint/restore is outside
this prototype's scope.

## One ordinary MemPool per tensor

Standard MemPool callbacks operate on backing segments. PyTorch can cache and
subdivide those segments without asking TE to remap a later tensor. Sharing one
such pool across localized tensors could give a small tensor entirely LD0
backing instead of the desired LD0/LD1 split.

The revised integration prevents that by creating a fresh, dedicated ordinary
MemPool for each tensor. Only one byte backing tensor is allocated inside that
pool. TE returns a contiguous, typed view of the requested shape, detached from
Python view metadata while retaining the same Storage and native deleter.

The native allocation ticket carries the logical byte count to its custom
allocator callback. The callback receives PyTorch's segment byte count and:

1. Reserves enough contiguous virtual address space for the native segment.
2. Obtains fixed-size physical handles from TE's LD0/LD1 caches, creating only
   missing blocks.
3. Maps only the backing tensor's rounded byte count, with the split determined
   by the logical tensor size, then sets access permissions.
4. Leaves any excess native segment tail unbacked. Its dedicated pool is never
   used to allocate a second tensor, so that tail cannot be exposed for use.

For example, a 4 MiB localized backing tensor may cause PyTorch to request a
20 MiB segment. TE reserves 20 MiB of virtual space and maps only 4 MiB of
physical backing, divided into 2 MiB LD0 and 2 MiB LD1. This avoids paying
physical memory for PyTorch's segment-size rounding. Native PyTorch reserved-byte
statistics still describe its full segment; TE's physical counters describe
actual backing.

Each ticket rejects a second raw allocation and checks that the backing tensor
starts at the segment base. The controller exposes no native routing pool to
callers. These restrictions are necessary to keep the unmapped tail inaccessible.

## Storage, reuse, and streams

A ticket observes Storage through a C++ weak `StorageImpl` reference. It does
not use a Python tensor finalizer, replace the native Storage deleter, or keep
the tensor alive. All aliases, including detached views, retain that Storage.
Normal PyTorch `record_stream` bookkeeping therefore remains intact.

Before each eager allocation, `controller.collect()` checks earlier tickets.
Once Storage is dead and no native graph reference retains its dedicated pool,
it drops the native pool. Ordinary MemPool teardown and a pool-specific native
cache trim return its segment to TE's free callback. This targets the dedicated
pool, not PyTorch's default pool. Teardown can synchronize pending recorded-stream
uses; it is not guaranteed to be nonblocking.

TE's free callback records an allocation-stream event. Once that event completes,
the C++ physical-block pool unmaps each page, puts its handle in the corresponding
LD cache, and releases the old virtual reservation. A subsequent tensor maps
those handles into a freshly reserved range. No complete mapped tensor slot is
reused. CUDA can numerically recycle a freed virtual address; unique pointer
values forever are not part of the contract.

Follow normal stream rules: wait for producers and record side-stream uses
before releasing a tensor. Collecting dead pools happens on later allocations
or explicit collection/draining, rather than synchronously in Python object
finalizers. Normal pool destruction/metadata is additional overhead compared
with the previous experimental managed-pool path.

## CUDA Graph ownership

`collect()` does not destroy native pools or query completion events during
capture. Allocations made during capture also attach a CUDA user object to the
capturing CUDA graph. CUDA retains that object across graph instantiation,
cloning, and in-flight execution. Its destructor callback only marks a CPU atomic
flag; it never calls CUDA or Python.

The physical-block pool does not unmap a captured allocation until that flag
indicates the graph/exec has released its last reference and any separately
recorded eager completion event is ready. This protects mappings even on PyTorch
versions without automatic custom-pool retention inside `use_mem_pool`.

```python
controller = get_vmm_mem_pool("cuda")
graph = torch.cuda.CUDAGraph()
source = torch.ones((4096, 4096), dtype=torch.bfloat16, device="cuda")
# Warm up the kernels before capture as usual.
with torch.cuda.graph(graph):
    temporary = controller.allocate(source.shape, source.dtype)
    temporary.copy_(source)
    output = temporary + 1  # Ordinary graph-pool output allocation.
    del temporary

graph.replay()
graph.reset()
controller.empty_cache()
```

Captured temporaries retain their mappings across replay; this implementation
does not reuse their physical backing inside a capture. Eager tensors used as
external graph inputs must remain alive until all consuming graphs are done,
as with ordinary PyTorch CUDA Graph usage.

In dedicated mode, `empty_cache()` first collects eligible dedicated pools,
synchronizes the owning device, and drains reclaimable physical blocks. It preserves live Storage and
live graph mappings even when called before graph reset. CUDA user-object
callbacks can run asynchronously after reset and synchronization, so a later
allocation/collect/drain may be needed to reclaim the last captured blocks.
`clear_vmm_mem_pool_caches()` drains all registered controllers/caches;
`clear_captured_vmm_allocations()` calls it through the existing graph-teardown
entry point. `close()` on the TE allocation wrapper stops further allocations
through that wrapper without invalidating its live tensors.

CUDA Graph Tree checkpoint/restore, CUDA IPC, and cross-device/peer mappings are
unsupported. On physical OOM, TE evicts unmapped cached handles and retries once;
it never evicts live/captured mappings or calls PyTorch's global `empty_cache`
from an allocator callback.

## Placement, overhead, and validation

The physical midpoint rounds up to a block boundary, matching the original TE
prototype. Nonaligned logical halves can put less than one block of the second
logical half in LD0; small tensors can reside entirely in LD0. Use aligned
shapes for exact half-split locality. Single-domain allocations round their
entire logical byte count to a block boundary.

`controller.stats()` reports allocator-held mapped bytes, retired mapped bytes,
unmapped cached physical bytes, virtual reservation bytes, block creation/reuse,
and map/unmap counts. A mapped segment can outlive the last tensor while its
native pool or a graph still owns it. The dedicated-pool count shows pending
native pool/ticket metadata. Existing TE workspace lease caches are unchanged:
ending a workspace lease does not destroy its persistent tensors.

This design trades more native pool bookkeeping, virtual address reservations,
and mapping operations for compatibility with ordinary PyTorch and physical
reuse across tensor sizes. Pool teardown may wait for GPU work. Performance and
actual container compatibility must be measured on the target GPU; host tests
cannot establish either.

Host-only checks (no PyTorch, CUDA or pytest required):

```sh
python3 tests/pytorch/test_vmm_mempool_controller.py -v
# Physical policy tests: see tests/cpp/vmm/README.md.
```

On supported hardware with the rebuilt TE extension (add
`NVTE_VMM_MEMPOOL_MODE=managed` for the patched-container comparison):

```sh
NVTE_VMM_USE_MEMPOOL=1 python -m pytest -v -s -rs tests/pytorch/test_vmm_mempool.py
NVTE_VMM_USE_MEMPOOL=1 python -m pytest -v -s -rs \
    tests/pytorch/mxfp8/test_mxfp8_localization.py -k 'vmm and not performance'
NVTE_VMM_USE_MEMPOOL=1 RUN_BENCHMARK_TESTS=1 python -m pytest -v -s -rs \
    tests/pytorch/mxfp8/test_mxfp8_localization.py::test_mxfp8_bidirectional_swizzled_vmm_performance
```

Compare manual VMM and ordinary-MemPool correctness, physical high-water marks,
setup/capture latency, and graph replay latency separately. The implementation
has not yet been built or exercised against the user's actual 26.09 container.

## Measure graph lifetime separately from allocator reuse

A training iteration normally replays an existing graph; it does not imply graph
reset or tensor destruction. In the local Megatron full-iteration wrapper,
`FullCudaGraphWrapper.__call__` captures after warmup, then uses `.replay()` on
subsequent calls. `reset_cuda_graph()` is separate. Workspace lease release also
does not release persistent backing Storage. This source observation does not
prove the graph-reset frequency of a particular launch configuration.

The dedicated-pool costs occur on new allocations, dead-pool collection, and
reset/retirement. They do not execute inside an allocation-free graph replay.
Eager localized allocations outside the graph can still incur those costs on
every training step. If an actual workload resets/recaptures each step, the
costs cannot be amortized and this workaround may be useful only for validation.

The allocator lifecycle benchmark reports synchronized host-wall phases for
allocation, capture, replay, graph reset, native-pool retirement, mapping
reclamation, and final cache release. It alternates allocation sizes and retains
TE's physical cache between cycles. It asserts that allocator/map/unmap counters
are unchanged during replay. CUDA events report the graph-loop interval, which
can include CPU submission gaps. It uses fill kernels, not quantization, and
its separately fenced phase sum is not an end-to-end training measurement.

```sh
# Capture once and reuse for 200 iterations per cycle:
python tests/pytorch/benchmark_vmm_allocator.py \
    --modes manual dedicated --cycles 3 --replays 200

# Deliberately model a short graph lifetime (one replay per capture/reset):
python tests/pytorch/benchmark_vmm_allocator.py \
    --modes manual dedicated --cycles 3 --replays 1

# In the patched container, compare all three paths:
python tests/pytorch/benchmark_vmm_allocator.py \
    --modes manual dedicated managed --cycles 3 --replays 200
```

Inspect `physical_blocks_reused` and map/unmap deltas across changing-size
cycles to evaluate Daniel's reuse mechanism. Fixed graph replay exercises stable
mappings, so replay latency alone cannot validate that mechanism. To measure
real MXFP8 quantization replay, run the existing performance test with
`MXFP8_LOCALIZATION_USE_CUDA_GRAPH=1` in addition to the MemPool/benchmark flags.
In a real training run, compare TE allocator counters around complete steps and
record graph reset/recapture frequency; graph-only timing excludes eager work
outside capture.

..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

CUDA context recovery
=====================

``nvte_reset_cuda_state(device)`` explicitly invalidates registered common-library
state after an external owner destroys a device's primary CUDA context. It is a
recovery primitive, not a replacement for framework or training-stack cleanup.
The initial implementation supports compute streams/events, CUDA-library handle
caches, RTC modules, normalization plans, fused-attention graph caches, and active
common Workspace catalogs. It does not detect context replacement automatically.

Stop all TE submissions and destruction in every host thread before destroying
the old context. After context destruction, call ``nvte_reset_cuda_state`` with
its nonnegative device ordinal. Callers must not race recovery with TE operations,
including calls on other devices. The registry invalidates the selected device's
state in all threads; another device's resources remain valid. Calling reset on
a still-live context abandons live resources and leaks them. Multiple live CUDA
contexts per device are not supported by this protocol.

The library owns neither context teardown nor recreation. It does not explicitly
synchronize, create a context, or destroy old CUDA stream/event/module handles in
reset. Reusable host plan descriptors are released through their dependency's
APIs. Operations lazily create resources when called with a recovered context.
Compiled RTC code is retained and its modules are loaded again without compilation.

Generation and synchronization
-------------------------------

``nvte_get_cuda_context_generation(device)`` is a host-only query, initially one.
It advances after each explicit reset, even when no resources exist. A query is
serialized with reset so it cannot observe a partially reset registry. A global
atomic notification also makes driver-side TLS initialization recheck its context
on its next use after any device reset. Resource invalidation precedes release
publication; observers use acquire ordering.

Generations identify stale resources. They do not pin a context, wait for GPU
completion, stop host threads, or make concurrent reset safe. Externally cached
handles from ``nvte_get_compute_stream`` and ``nvte_get_compute_stream_event`` must
be discarded after reset. Pointer equality does not identify an allocation or
context lifetime.

Workspace recovery
--------------------

A common Workspace registers with the context on its first stream use. Reset
unbinds its storage, clears stream slots, invalidates persistent contents and
outstanding plans, and preserves the host region layout and IDs. Readiness events
held by older plan snapshots also recognize a lost context and skip CUDA destroy.
After the allocator recovers, explicitly bind new caller-owned storage and prepare
persistent regions again. Initialization callbacks are supplied by the consumer;
the catalog does not retain them or arbitrary ``user_data`` for automatic replay.
Rebuild TMA descriptors with current pointers and recapture CUDA graphs. Common
never frees or replaces the caller's device allocation. Unused bindings that have
not yet registered with a CUDA context must also be replaced by their owner.

Current integration limits
----------------------------

This API does not reset framework allocators, Python tensor caches, PyTorch
``WorkspaceOwner`` current/retired tensors, model or optimizer tensors, RNG state,
CUDA graphs, CuTeDSL/TVM-FFI modules, communication overlap objects, or EP/NCCL
communicators. Their owners must handle their lifecycle; the complete NVRx
integration remains separate work. In particular, resetting common state alone
is insufficient to recover an existing PyTorch or JAX training process.

There is also no guarantee that a dependency releases every host allocation after
context loss. On CUDA 13.5 / cuDNN 9.28, a standalone ``cudnnCreate`` →
``cudaDeviceReset`` → ``cudnnDestroy`` probe crashes, so dead cached handles are
abandoned rather than destroyed. A probe which instead recreates handles shows
increasing allocated host heap. Full leak-free recovery of those opaque resources
needs a dependency-supported cleanup strategy; suppressing destructor errors or
claiming that CUDA reset frees all host state would be incorrect.

C/C++ reference
----------------

.. doxygenfile:: cuda_context.h

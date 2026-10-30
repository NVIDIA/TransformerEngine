..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

CUDA state lifecycle
====================

TE provides two explicit operations with different lifetime requirements:

* ``nvte_release_cuda_state(device)`` releases registered resources while the
  primary CUDA context and borrowed communicators are usable. Use this for a
  context-preserving in-process restart, or before planned context destruction.
* ``nvte_reset_cuda_state(device)`` invalidates registered state **after** an
  external owner has destroyed the primary context. Dead stream/event/module
  handles are abandoned instead of passed to CUDA destroy functions. Calling
  this on a live context leaks resources and is unsupported.

Both take an explicit nonnegative CUDA device ordinal. Stop all TE submissions
and object destruction in every host thread, including calls on other devices.
For release, finish GPU work and retire captured graphs before cleanup. TE does
not own context teardown, detect context replacement, reset framework allocators,
or restore model tensors. Multiple live CUDA contexts per device are unsupported.

Compute streams/events, CUDA-library handles, RTC modules, normalization plans,
and fused-attention graphs participate in the registry.
Their next use recreates resources lazily. Compiled RTC bytes remain cached.
Thread-local library handles have process-owned slots: a finished host thread
returns its slots to an idle pool, so later cleanup can still release the handles.
Release destroys live handles in dependency order: plans, handles, modules,
then streams. It temporarily selects the specified device and restores the
previous device. A cleanup error propagates after remaining callbacks run; this
is not transactional. Finish cleanup before submitting new work after an error.

Reset performs a preflight before changing resources or generations. Active
communication/IPC/VMM owners, CuTeDSL executors and JAX pinned transfers which
cannot safely discard their dependencies reject post-loss reset. Such resources
must be released while CUDA and the transport remain usable; if that boundary
was missed, restart the worker. Reset itself does not synchronize or explicitly
create GPU resources. Host plan descriptors are released through dependency APIs.

Generation and synchronization
-------------------------------

``nvte_get_cuda_context_generation(device)`` is a host-only query, initially one.
It advances after release or reset, even when no resources exist. A query is
serialized with lifecycle operations and cannot observe partial cleanup.
``nvte_get_cuda_state_version()`` exposes a global atomic notification so TLS
owners can first perform a cheap acquire load, then recheck their device epoch
only after a change. Publication follows resource invalidation. Rejected preflight
does not advance either counter; a release with a callback error does advance
them because some resources may already have been released.

Generations identify stale resources. They do not pin a context, wait for GPU
completion, stop host threads, or make concurrent reset safe. Externally cached
handles from ``nvte_get_compute_stream`` and ``nvte_get_compute_stream_event`` must
be discarded after either operation. Pointer equality does not identify a
resource lifetime.

Bindings with their own caches can use ``nvte_register_cuda_state`` and
``nvte_unregister_cuda_state``. The registered address must remain stable until
unregistration. Preflight callbacks cannot mutate state, invalidate callbacks
cannot throw, and callbacks must not register new resources or re-enter lifecycle
APIs. Cleanup may destroy and unregister child owners. Registration is host-only
and does not create a context on GPU 0 at library import.

TMA workspace recovery
----------------------

Grouped operations use stateless caller-owned TMA storage. After context loss,
replace the caller's device allocation and use the workspace query/allocate/execute
protocol again. Each execution rebuilds descriptors from the current tensor
pointers. No workspace catalog, plans, or persistent-region registration is
involved. Retire old CUDA graphs and recapture with valid allocations.

Framework cleanup
-----------------

``transformer_engine.pytorch.release_cuda_state(device)`` synchronizes the device,
clears PyTorch's process-wide compilation cache, finalizes that device's TE EP and
Userbuffers managers, detaches FP8 reduction registrations, and releases TE-owned
tensor caches before common cleanup. TMA scratch storage is allocated by each
frontend operation and follows the framework allocator lifetime. FP8 numerical
history, model/optimizer tensors and RNG state survive.
Call outside autocast and graph capture, after completing backward. Borrowed
process groups must remain valid until cleanup finishes. Release independently
retained communication owners first. Every collective participant must execute
cleanup in the same order. Reinitialize EP/Userbuffers explicitly before reuse;
old EP routing buffers are rejected after finalization or rebootstrap.

``transformer_engine.jax.release_cuda_state(device)`` waits for effects and device
work, clears JAX's process-wide compilation cache, finalizes TE EP, and releases
per-device collective GEMM plans/communicators, pinned transfers and score-mod
graphs through registered callbacks. Explicitly reinitialize communication before
reuse. External executable/communication references must be retired beforehand.

CuTeDSL cleanup drops native TVM-FFI function references, removes TE's generated
Python registry entries and releases compiled executors while CUDA is live.
New entries include device and generation, preventing cross-device or stale
executor reuse. Third-party compiler/runtime caches are owned by those projects;
TE does not promise to clear every dependency-internal cache or user-held tensor.

For example, at a caller-controlled, context-preserving PyTorch restart boundary::

    # Finish backward, stop all callers, and retire captured graphs first.
    # All ranks clean up before destroying their borrowed process groups.
    transformer_engine.pytorch.release_cuda_state(device)
    # Reinitialize TE communication if used, then resume with existing tensors.

Context-loss limits
------------------

An initialized PyTorch CUDA allocator/stream pool and the JAX/XLA runtime cannot
be reconstructed by TE. Their public ``reset_cuda_state`` wrappers reject this
case instead of claiming recovery. Native callers with an independently recovered
allocator can use the common post-loss API for supported registered resources.
An atomic generation does not make stale framework tensors safe to free or reuse.

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

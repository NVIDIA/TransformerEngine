..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

CUDA state lifecycle
====================

TE provides two explicit, process-wide lifecycle operations:

* ``nvte_release_cuda_state()`` releases registered resources while all their
  primary CUDA contexts and borrowed communicators remain usable. Use this for a
  context-preserving restart or before planned context destruction.
* ``nvte_reset_cuda_state()`` invalidates registered state after the external
  owner has destroyed every primary context backing those resources. Dead
  stream/event/module handles are abandoned instead of passed to CUDA destroy
  functions. Using reset while any affected context is live leaks resources and
  is unsupported.

There is no device argument or selective-device recovery. Stop all TE calls,
object construction and destruction in every host thread before either operation.
For release, complete GPU work and retire captured graphs first. TE does not own
context teardown, detect context replacement, reset framework allocators, or
restore model tensors. Multiple live CUDA contexts per device are unsupported.

Compute streams/events, CUDA-library handles, RTC modules, normalization plans,
fused-attention graphs, and registered frontend owners share one registry. A
resource may own state for one or several GPUs. Operations recreate released
resources lazily; compiled RTC bytes remain cached on the CPU. Thread-local
library handles have process-owned slots, so handles from finished threads can
still be released.

Live cleanup retains device affinity: callbacks for one GPU run with its existing
primary context current, and the caller's context is restored afterward. Cleanup
does not initialize primary contexts on unused devices. Owners spanning several
GPUs select the appropriate devices themselves. Both operations use dependency
phases: plans precede library handles, modules and streams. Within a phase,
cleanup follows reverse registration order. Callback errors propagate after
remaining callbacks run; cleanup is not transactional. Finish it before resuming
work after an error.

Reset performs a process-wide preflight before mutating resources or publishing a
new generation. Active communication/IPC/VMM owners, CuTeDSL executors and JAX
pinned transfers can reject unsupported post-loss cleanup. Release such resources
while CUDA and their transport remain usable. If that boundary was missed, restart
the worker. Reset itself is host-only and does not query or synchronize CUDA.

Generation and registration
---------------------------

``nvte_get_cuda_state_version()`` returns the single process-wide generation,
initially one, through an acquire load. Release or reset publishes one increment
after cleanup, including when there are no registered resources. A rejected
preflight leaves the version unchanged. A release with a callback error advances
it because resources may already have been released.

This notification does not pin resources, wait for an in-progress lifecycle call,
or permit concurrent reset and TE work. Callers must discard cached TE stream and
event handles after either operation. Pointer equality does not identify a
resource lifetime.

The native ``Resource`` base and the C-ABI ``FrontendResource`` adapter register
objects automatically during base construction and unregister during destruction,
including constructor unwinding. The caller must exclude construction and
destruction while lifecycle callbacks run; they cannot observe partial objects.
Cleanup affinity and priority are explicit constructor parameters, rather than
virtual calls on an incompletely constructed derived object.

Bindings use ``nvte_register_cuda_state`` and ``nvte_unregister_cuda_state``.
Registration is host-only. Its device ordinal is
an affinity for live cleanup, not a reset filter; ``-1`` identifies host-only or
multi-device owners whose callbacks manage device selection. Registered addresses
remain stable until unregistration. Check callbacks cannot mutate state;
invalidation callbacks cannot throw. Callbacks must not register new resources
or re-enter lifecycle APIs, but cleanup may destroy and unregister child owners.

``nvte_set_cuda_state_device`` changes an existing registration's live-cleanup
affinity without changing its original registration order. Standard dependency
phases are named by ``NVTECudaStateReleaseOrder``; custom integer priorities remain
supported. Reset and release both process ascending priority and reverse
registration order within each priority. Unregistered children are skipped even
if they were present in the lifecycle snapshot.

Lazy initialization
-------------------

``LazyResource`` provides an optional ``ensure_valid(initializer)`` for library
handles, RTC modules and context-local kernel attributes. ``InitializationState``
serializes initialization with a mutex and a second validity check, and publishes
success with release/acquire ordering. A throwing initializer remains invalid and
can be retried. The fast path does not query CUDA or lock the resource registry.

Initializers receive their parameters from their caller; a cache container need
not implement a parameterless virtual initializer. Owners provide distinct live
release and post-loss invalidation hooks. Cleanup visits partially initialized
owners too, since an initializer may acquire resources before failing. Every
lifecycle operation invalidates successful initialization after the owner finishes
cleanup. Users must still stop all calls before teardown; ``ensure_valid`` does
not make concurrent reset or destruction safe.

TMA workspace recovery
----------------------

Grouped operations use stateless caller-owned TMA storage. After context loss,
replace the caller's device allocations and repeat query/allocate/execute. Each
execution rebuilds descriptors from the current tensor pointers. No workspace
catalog, plans, or persistent-region registration is involved. Retire old CUDA
graphs and recapture with valid allocations.

Framework cleanup
-----------------

``transformer_engine.pytorch.release_cuda_state()`` checks capture boundaries on
all active devices before mutating state, synchronizes those devices, clears
PyTorch's process-wide compilation cache, finalizes TE EP/Userbuffers, detaches
FP8 reduction registrations, and clears TE-owned tensor caches across devices.
CPU tensor cache entries and FP8 numerical history survive. Model and optimizer
tensors, RNG state and the framework runtime are retained. TMA scratch follows
the framework allocator lifetime.

Call outside FP8 autocast and graph capture, after backward completes. Release
independently retained communication owners first. Borrowed process groups must
remain valid until cleanup finishes, and every collective participant performs
cleanup in the same order. Reinitialize EP/Userbuffers explicitly before reuse;
old EP routing buffers are rejected after finalization or rebootstrap.

``transformer_engine.jax.release_cuda_state()`` waits for effects and work on all
local GPU devices, clears JAX's compilation caches, finalizes TE EP across local
devices, and releases registered collective-GEMM plans/communicators, pinned
transfers and score-mod graphs. Explicitly reinitialize communication before
reuse. External executable and communication references must be retired first.

Each frontend wrapper cleans its own Python state and then invokes global common
cleanup. Applications using multiple frontends must coordinate all their owners
and stop their work at the same boundary.

CuTeDSL cleanup drops native TVM-FFI references and TE-generated Python registry
entries while CUDA remains live. Dispatch keys retain device identity and use the
global version to avoid reusing executors from earlier lifetimes. Private caches
of third-party compiler runtimes remain owned by those runtimes.

For a caller-controlled, context-preserving PyTorch restart::

    # Complete backward, stop all TE callers and retire captured graphs.
    # Every rank cleans up before destroying borrowed process groups.
    transformer_engine.pytorch.release_cuda_state()
    # Reinitialize TE communication if used, then reuse the existing tensors.

Context-loss limits
------------------

TE does not reconstruct the initialized PyTorch allocator/stream pool or the
JAX/XLA runtime. The Python ``reset_cuda_state()`` wrappers reject unsupported
framework recovery instead of touching stale tensors. Native owners with an
independently recovered allocator can use the common API for supported resources.
The global generation does not make stale framework allocations safe to reuse.

Dependency-owned host allocations may also survive context loss. On CUDA 13.5 /
cuDNN 9.28, the saved standalone ``cudnnCreate`` → ``cudaDeviceReset`` →
``cudnnDestroy`` probe crashes, so dead handles are abandoned. Recreating handles
without destruction increased allocated host heap in that probe. Leak-free
post-loss cleanup of opaque host state requires a dependency-supported strategy.

C/C++ reference
--------------

.. doxygenfile:: cuda_context.h

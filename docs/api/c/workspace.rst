..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Workspace storage, regions, and lifetime
========================================

Ownership and representation
-----------------------------

Workspace APIs use the existing ``NVTETensor`` handle. TE lazily attaches a
host-side region catalog to that tensor; no separate workspace handle is needed.
Create and destroy it with ``nvte_create_tensor`` / ``nvte_destroy_tensor`` or
``TensorWrapper``. ``WorkspaceWrapper`` is an optional convenience wrapper around
``TensorWrapper`` and the same workspace functions. The framework or C/C++ caller
allocates, retains, and frees the device buffer.

The device buffer is byte-addressed storage. Grouped quantization reports a
one-dimensional byte tensor. DBias queries retain the backend's storage dtype;
the returned shape covers the complete layout, including alignment and any
other reservations. Region types are independent of the allocation's dtype.
The host catalog records each region's name, exact key, byte offset, reserved byte
size, alignment, and lifetime. Region IDs are stable within a workspace. Different
keys under the same name identify separate resources. Names and keys are copied;
keys must use a canonical byte encoding without uninitialized structure padding.
Include the table format version and all parameters affecting its values in its key.
Names beginning with ``te.`` are reserved for library resources.

A region address is the current base pointer plus its offset. The owner keeps the
catalog with its allocation; copying the byte buffer alone does not reconstruct
the catalog or establish table validity. Metadata is not a serialized/device ABI.
There is no GPU header to read back when planning an operation.

The caller supplies the required allocation size and base alignment. Operation
adapters validate the tensor's declared byte capacity and base alignment; they
cannot verify the actual allocation behind a supplied pointer. Low-level region
accessors resolve offsets without inspecting device memory.

Query and execution
---------------------

The existing empty-tensor protocol prepares each operation's workspace:

1. Call the operation with an ``NVTETensor`` whose rowwise data pointer is null.
   TE queries the selected backend, adds its scratch region to the internal
   catalog, and reports the required shape and dtype. Outputs are untouched.
2. Allocate the returned shape and dtype through the framework and set the
   tensor's rowwise data pointer.
3. Call the same operation with that tensor to execute. Execution resolves the
   prepared regions without growing the layout or allocating device storage.

For example:

.. code-block:: cpp

   transformer_engine::TensorWrapper workspace;
   nvte_quantize_dbias(input, output, dbias, workspace.data(), stream);
   void *buffer = framework_allocate(workspace.shape(), workspace.dtype(), 128);
   workspace.set_rowwise_data(buffer, workspace.dtype(), workspace.shape());
   nvte_quantize_dbias(input, output, dbias, workspace.data(), stream);

``framework_allocate`` is an application function. Always allocate the returned
shape and dtype. When no scratch is needed, grouped quantization requests one
byte to distinguish execution from a query, even though the region catalog is
empty. A null data pointer always selects a query, including for an existing
catalog or a committed empty plan.

Several operations can query the same empty tensor before allocation. Each query
includes the previous reservations, so the last shape/dtype covers all of them.
Repeated sufficient requests reuse the same regions. Keep that tensor handle
when the layout contains multiple regions or persistent data. A fresh handle over
a scratch-only allocation also supports isolated execution, as used by callers
that cache the queried shape/dtype separately. Independent handles over the same
allocation do not share stream mapping or table readiness: retain the same
tensor/catalog for concurrent reuse.

Optional application resources
--------------------------------

Applications reserving persistent resources can use the generic catalog API.
An explicit ``WorkspacePlanWrapper`` is optional and only describes application
reservations; operations add their own scratch during the normal size query:

.. code-block:: cpp

   transformer_engine::TensorWrapper workspace;
   const unsigned char table_key[] = {1, 2, 0};
   {
     transformer_engine::WorkspacePlanWrapper plan(workspace.data());
     plan.require("activation.gelu", table_bytes, kNVTEWorkspacePersistent,
                   128, table_key, sizeof(table_key));
     nvte_workspace_commit_plan(workspace.data(), plan.data());
   }
   nvte_quantize_dbias(input, output, dbias, workspace.data(), stream);
   void *buffer = framework_allocate(workspace.shape(), workspace.dtype(), 128);
   workspace.set_rowwise_data(buffer, workspace.dtype(), workspace.shape());
   auto table = nvte_workspace_find_region(workspace.data(), "activation.gelu",
                                            table_key, sizeof(table_key));
   // initialize_table(data, bytes, stream, context) enqueues the producer.
   nvte_workspace_initialize_region(workspace.data(), table, stream,
                                    initialize_table, nullptr);
   nvte_quantize_dbias(input, output, dbias, workspace.data(), stream);

``initialize_table`` is a host callback with type ``NVTEWorkspaceInitializer``;
it enqueues initialization and reports any enqueue failure before returning.
The workspace calls it once per region/allocation generation. Later streams
wait for an event immediately following that producer, not for its later kernels.
The callback must not synchronize the device or re-enter this workspace. After
a callback failure, drain any partially enqueued writes before retrying.
Application keys include the table format version and all value parameters.

A generic plan snapshots one workspace revision; it never changes the active
layout until committed. A commit, an operation query that grows the layout, or a
binding change makes older plans stale. Discarding a generic plan has no effect
on the active workspace. The caller serializes host access to each tensor.

Existing sufficient reservations are reused without shrinking. Growing a region
appends a replacement, preserving its ID and other regions' offsets. Retired
regions are not reclaimed or compacted automatically. A relocated persistent
region needs initialization again; unchanged regions retain their validity when
the same allocation is reused. Every nonempty region starts on at least a 128-byte
boundary. Stricter power-of-two alignment is supported. Total size includes final
padding, and the owner satisfies the maximum base alignment. Every scratch or
stream-persistent reservation includes one aligned copy per stream slot. Region
info describes slot zero and the per-slot byte size, not the replicated total.
Replicas are reserved together per region so appending another resource preserves
existing addresses; a stream's regions need not form one contiguous suballocation.

Persistent initialization and streams
--------------------------------------

There are three resource modes:

* ``kNVTEWorkspacePersistent``: shared immutable resources such as activation
  tables, initialized once per key/reservation/allocation generation.
* ``kNVTEWorkspaceStreamPersistent``: retained mutable resources such as TMA
  descriptors, with an independent copy per stream. Grouped TMA consumers rebuild
  their descriptors before every invocation.
* ``kNVTEWorkspaceScratch``: temporary mutable storage, also private per stream.

Grouped and non-grouped consumers may share an immutable table when they agree on
its name, key, format, and contents. Lazy initialization does not defer reserving
space: query the complete layout before allocating. A table is initialized once
for the program only when its device/context, key and backing allocation survive
for the program's lifetime.

Setting rowwise data through ``nvte_set_tensor_param``,
``nvte_set_tensor_param_v2``, or ``TensorWrapper::set_rowwise_data`` binds a new
allocation, just like ``nvte_workspace_bind``. Every binding starts a new
generation and invalidates all persistent regions,
including when the allocator returns the same numerical address. Do not bind again
just to reuse the same live allocation. A larger byte count or a familiar pointer
value does not establish that a table's contents are valid.

``mark_initialized`` records a readiness event after an explicitly enqueued
producer; it performs no initialization or host/GPU synchronization. It applies
only to shared immutable regions. ``initialized`` means the producer/event was
enqueued, not that the GPU has finished. Prefer ``initialize_region`` to combine
the once-only producer and readiness protocol. The stream-aware region accessor
also establishes readiness for an already initialized shared region. Drain all
consumers and destroy referencing graphs before invalidating or overwriting a
shared table.

``kMaxWorkspaceStreamSlots`` is the single C++ constant controlling the limit,
currently four; C callers can query ``nvte_workspace_max_stream_slots()``. A new
CUDA stream ID receives the next free slot. IDs are compared for equality; stream
handle values are neither indices nor assumed ordered, and recycled handles do
not acquire a previous stream's slot. Default streams use CUDA's stream identity.
All streams must belong to one CUDA device. Slots stay reserved until rebinding;
exhaustion raises an error without waiting or eviction.

``get_region_data_on_stream`` resolves the stream-private copy of mutable storage
or the common immutable address. The legacy accessor without a stream resolves
slot zero; it must not be used to share mutable regions between streams.
Host access, including the complete enqueue sequence of each operation, must be
serialized by the caller. This serialization does not add GPU dependencies:
different slots execute independently, even when submitted by one host thread.

Growing storage and CUDA Graphs
--------------------------------

To grow a retained workspace, set its rowwise data pointer to null and query
again. This binding change starts a new generation and invalidates persistent
contents. Allocate the returned shape/dtype, bind the new buffer, and initialize
persistent resources before execution. The caller can explicitly copy compatible
table contents from the old allocation and then mark the new region initialized,
ordering that copy after the old producer and before new consumers. Keep the old
allocation alive until all kernels and captured graphs referring to it finish.

Plan, allocate, and prepare immutable resources on every consuming stream before
graph capture. Ensure initialization has completed before capturing consumers:
a wait enqueued before capture is not embedded in the graph, which may later be
replayed on a different stream. Capture rejects pending initialization.
Readiness events are created/recorded outside capture; prepared streams reuse
them without adding capture dependencies. During capture, execute
using stable addresses. A new binding does not update
pointers embedded in an existing graph. Keep the old allocation for that graph or
recapture with the new binding/layout. Graph replays and ordinary executions that
share mutable storage must be ordered; the descriptor cannot observe an external
CUDA graph launch. Graphs keep the slot selected at capture, even when replayed
on another stream. Independently executing graphs require separate slots or
separate workspaces, including graphs captured on the same stream. Views and
previously obtained pointers do not follow a new binding or relocated region.

Operation adapters and compatibility
-------------------------------------

``nvte_group_quantize_v2`` prepares grouped TMA storage when called with a null
workspace data pointer and executes using that reservation once storage is bound.
Grouped MXFP8 with constant last dimensions and other supported scaling modes
need no TMA region. Varying last dimensions support more than 64 groups. Each of
the four TMA arrays and three metadata arrays is independently aligned to 128 bytes.

``nvte_quantize_dbias`` and ``nvte_group_quantize_dbias`` query their selected
backend internally, reserve the stream-private ``te.dbias`` scratch region, and report
storage through the same tensor. Execution adapts that region to the backend's
temporary tensor representation. Sequential grouped and non-grouped DBias
operations can query and reuse the same reservation. Persistent regions remain
separate from both DBias and TMA scratch.

All mutable regions are replicated per registered stream; no descriptor or scratch
storage is shared between different slots. Reuse the tensor handle to preserve
its catalog; copying the byte
buffer or creating another handle does not share that metadata. Destroying the
tensor releases its catalog, including when its handle slot is reused.

PyTorch owner
--------------

The private extension exposes an explicit, reusable owner:

.. code-block:: python

   import transformer_engine_torch as tex

   workspace = tex.Workspace()  # Keep alive across calls and captured graphs.
   result = tex.group_quantize(
       input, quantizer, num_tensors, first_dims, last_dims, workspace=workspace
   )

The owner holds an ordinary TE tensor with its internal catalog and a
one-dimensional CUDA ``uint8`` tensor.
It uses the same operation for size queries and execution, grows storage
geometrically, reuses an unchanged layout, and keeps retired
allocations until the owner is destroyed so earlier graphs retain their storage.
It belongs to one device and supports the common stream-slot limit. Its mutex
serializes host submission without waiting for GPU consumers. Layout growth
always uses fresh storage, including when spare capacity exists, so old streams
and graphs keep valid private addresses. Allocator stream recording protects
asynchronous consumers. Warm it up with all required shapes before capture;
layout growth during capture is rejected. ``size``, ``capacity``, ``generation``
and ``max_stream_slots`` expose its state. Explicit
ownership supports grouped MXFP8 quantization. Calls without this owner preserve
the existing temporary-workspace behavior.

Internal layout utility
------------------------

``common/util/workspace.h`` provides ``WorkspaceLayout``, ``WorkspaceBlock``, and
``WorkspaceView`` for laying out subregions such as grouped TMA arrays. Typed
blocks respect ``alignof(T)`` and all nonempty blocks have at least 128-byte
alignment. Empty blocks consume no storage and resolve to ``nullptr``. Arithmetic
overflow, block bounds, and alignment within the calculated layout are checked;
caller allocation size/alignment are trusted. Views own no memory and must be
rebuilt when their layout or binding changes.

C/C++ reference
----------------

.. doxygenfile:: workspace.h

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

The caller supplies the required allocation size and base alignment. Execution
trusts that allocation; it does not validate its capacity or base alignment.
Checks of resource IDs, planned reservations, arithmetic, and ordering describe
the host protocol and do not inspect the allocation.

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
a scratch-only allocation also supports the original execution protocol, as used
by callers that cache the queried shape/dtype separately.

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
   initialize_table(nvte_workspace_get_region_data(workspace.data(), table), stream);
   nvte_workspace_mark_region_initialized(workspace.data(), table, stream);
   nvte_quantize_dbias(input, output, dbias, workspace.data(), stream);

``initialize_table`` is an application function; workspace metadata does not
generate table values. Application keys include the table format version and
all parameters affecting its values.

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
padding, and the owner satisfies the maximum base alignment.

Persistent initialization and streams
--------------------------------------

Scratch may be overwritten on every invocation. Persistent resources remain valid
for the same key, region reservation, and allocation generation until explicitly
invalidated. Grouped and non-grouped consumers can share a table when they agree
on its name, key, format, and contents. Mutable scratch must not overlap those
regions.

Setting rowwise data through ``nvte_set_tensor_param``,
``nvte_set_tensor_param_v2``, or ``TensorWrapper::set_rowwise_data`` binds a new
allocation, just like ``nvte_workspace_bind``. Every binding starts a new
generation and invalidates all persistent regions,
including when the allocator returns the same numerical address. Do not bind again
just to reuse the same live allocation. A larger byte count or a familiar pointer
value does not establish that a table's contents are valid.

``mark_initialized`` records that the producer was successfully enqueued for the
current generation; it performs no initialization or synchronization. It applies
only to persistent regions. ``initialized`` means subsequent consumers ordered
in the workspace stream can use the table, rather than that the GPU has finished.
An explicit invalidation requires preparing the table before its next consumer.

The first ``begin_use`` or ``mark_initialized`` selects the CUDA stream for that
allocation generation, including the default stream. Further uses must use the
same stream. Use separate workspaces for other streams. Host metadata must be
accessed serially. No CUDA events or implicit waits are inserted by the catalog.

Growing storage and CUDA Graphs
--------------------------------

To grow a retained workspace, set its rowwise data pointer to null and query
again. This binding change starts a new generation and invalidates persistent
contents. Allocate the returned shape/dtype, bind the new buffer, and initialize
persistent resources before execution. The caller can explicitly copy compatible
table contents from the old allocation and then mark the new region initialized,
ordering that copy after the old producer and before new consumers. Keep the old
allocation alive until all kernels and captured graphs referring to it finish.

Plan, allocate, and initialize before graph capture. During capture, execute
using prepared regions and stable addresses. A new binding does not update
pointers embedded in an existing graph. Keep the old allocation for that graph or
recapture with the new binding/layout. Graph replays and ordinary executions that
share mutable storage must be ordered; the descriptor cannot observe an external
CUDA graph launch. Views and previously obtained pointers do not follow a new
binding or a relocated region automatically.

Operation adapters and compatibility
-------------------------------------

``nvte_group_quantize_v2`` prepares grouped TMA storage when called with a null
workspace data pointer and executes using that reservation once storage is bound.
Grouped MXFP8 with constant last dimensions and other supported scaling modes
need no TMA region. Varying last dimensions support more than 64 groups. Each of
the four TMA arrays and three metadata arrays is independently aligned to 128 bytes.

``nvte_quantize_dbias`` and ``nvte_group_quantize_dbias`` query their selected
backend internally, reserve the shared ``te.dbias`` scratch region, and report
storage through the same tensor. Execution adapts that region to the backend's
temporary tensor representation. Sequential grouped and non-grouped DBias
operations can query and reuse the same reservation. Persistent regions remain
separate from both DBias and TMA scratch.

Pure temporary scratch retains the original caller-controlled stream ordering.
Catalogs containing persistent data additionally enforce one stream per allocation
generation. Reuse the tensor handle to preserve its catalog; copying the byte
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
geometrically, reuses sufficient capacity, and keeps retired
allocations until the owner is destroyed so earlier graphs retain their storage.
It belongs to one device and stream and serializes host metadata access. Warm it
up with all required shapes before capture; layout growth during capture is
rejected. ``size``, ``capacity``, and ``generation`` expose its state. Explicit
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

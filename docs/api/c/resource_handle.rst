..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

resource_handle.h
=================

Grouped MXFP8 quantization can use a caller-owned resource handle for TMA
descriptors. Create a handle and reuse it across host threads and CUDA streams.
Creating a handle does not allocate device memory. The first
``nvte_group_quantize`` call that needs per-group descriptors allocates enough
storage for its group count; later calls from the same host thread and stream
reuse it. If a larger group is used, the handle grows outside CUDA graph capture.
Reserve the largest expected group count before capture with
``nvte_resource_handle_reserve``. Pass ``NVTE_RESOURCE_GROUPED_TMA_DESCRIPTORS``
as the resource type and the number of tensor groups as the capacity. The
operation updates descriptors on every call because tensor addresses and shapes
may change. Growth retains earlier allocations until handle destruction because
previously captured graphs may still reference them; reserving the expected
maximum up front avoids this extra memory use.

.. code-block:: cpp

   NVTEResourceHandle handle = nvte_create_resource_handle();
   nvte_resource_handle_reserve(handle, NVTE_RESOURCE_GROUPED_TMA_DESCRIPTORS,
                                num_tensors, stream);
   nvte_group_quantize(input, output, config, stream, handle);
   cudaStreamSynchronize(stream);
   nvte_destroy_resource_handle(handle);

In C++, the trailing handle argument defaults to ``nullptr``. Omitting it, or
passing ``nullptr`` explicitly, keeps the legacy path and its 64-descriptor
limit for groups with varying last dimensions. C callers must pass the fifth
argument explicitly. Changing the C function signature requires rebuilding
existing binary clients, even when their C++ source calls omit the handle.

The first operation or nonzero reservation binds the handle to the current
CUDA context and device. Each host thread and CUDA stream pair gets its own
storage, including when several host threads use the same stream or a default
stream. Registering a new pair is synchronized; later lookups for that pair
avoid a handle-wide lock. Each pair grows its storage independently. Reserve
on the same host thread and stream that will perform quantization during graph
capture.

Calls using different pairs may enqueue concurrently. Operations on one pair
use that pair's CUDA stream ordering. A captured graph retains the storage of
the pair used for capture; order its launches with any other use of that same
storage. Do not destroy a handle concurrently with host calls. Complete
submitted work and destroy any graphs or graph executables that use its storage
before destruction. A pair whose descriptor capacity is too small cannot grow
during graph capture; reserve before capture or warm up with the largest
expected group.

.. doxygenfile:: resource_handle.h

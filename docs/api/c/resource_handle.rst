..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

resource_handle.h
=================

The resource handle owns optional device storage for grouped MXFP8 TMA descriptors.
Creating it does not allocate device memory. The first grouped quantize or
dequantize call that needs per-group TMA descriptors allocates enough storage
for its group count; later calls reuse it. If a larger group is used, the handle grows outside CUDA
graph capture. Reserve the largest expected group count before capture with
``nvte_resource_handle_reserve_grouped_tma``. The handle keeps the allocation;
the operation still updates descriptors on every call because tensor addresses and
shapes may change.

For example, a caller can create an empty handle and pass it to the new entry
point:

.. code-block:: cpp

   NVTEResourceHandle handle = nvte_create_resource_handle();
   nvte_group_quantize_with_handle(input, output, config, stream, handle);
   cudaStreamSynchronize(stream);
   nvte_destroy_resource_handle(handle);

Calls without a handle retain the existing API and 64-descriptor limit. The
``_with_handle`` variants also accept a null handle for the legacy behavior.
Use a separate handle for concurrent streams, or explicitly order uses of a
shared handle. Use and destroy it in the CUDA context where its storage was
allocated. Complete submitted work before destroying it.

.. doxygenfile:: resource_handle.h

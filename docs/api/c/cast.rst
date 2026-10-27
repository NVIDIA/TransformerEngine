..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

cast.h
======

Grouped quantization workspace
------------------------------

``nvte_group_quantize_v2`` accepts the existing ``NVTETensor`` workspace argument.
Call it with an empty tensor to query the required shape and dtype, allocate
storage through the framework, then call the same function again to quantize.
The query prepares the region layout internally, including any persistent
reservations already attached to that tensor. No separate requirements or
operation-planning call is needed.

.. code-block:: cpp

   transformer_engine::TensorWrapper workspace;
   nvte_group_quantize_v2(input, output, workspace.data(), config, stream);
   // Allocate workspace.shape() elements of workspace.dtype() using the framework.
   // The data pointer must be 128-byte aligned for grouped MXFP8 TMA descriptors.
   workspace.set_rowwise_data(device_ptr, workspace.dtype(), workspace.shape());
   nvte_group_quantize_v2(input, output, workspace.data(), config, stream);

The query does not write quantized outputs. The workspace is a one-dimensional
byte tensor; modes and layouts without temporary storage request one byte to
distinguish execution from a query. MXFP8 groups with varying last dimensions
store their tensor-map descriptors and metadata in the workspace and support
more than 64 groups. No initialization is needed, and a sufficiently large
workspace can be reused for subsequent calls.
The caller must provide the queried allocation size and required alignment;
execution does not validate the supplied buffer or its tensor metadata.

Each descriptor and metadata array starts on a 128-byte boundary. Workspace size
and offsets are calculated with the common layout utility; callers should query
requirements rather than hard-code sizes. See :doc:`workspace` for layout,
persistent table, growth, and CUDA Graph lifetime contracts.

Allocate before CUDA graph capture. Keep the workspace alive until all uses
complete, including captured graph executions. Concurrent calls must use
separate workspaces or explicit CUDA synchronization. The original
``nvte_group_quantize`` remains available; its shared descriptor storage limits
MXFP8 groups with varying last dimensions to 64 and requires serialized use.

.. doxygenfile:: cast.h

..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

cast.h
======

Grouped quantization workspace
------------------------------

``nvte_group_quantize_v2`` accepts caller-owned temporary storage. Query the
required shape and dtype with an empty tensor, allocate device memory through
the framework allocator, then call again to quantize:

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

Allocate before CUDA graph capture. Keep the workspace alive until all uses
complete, including captured graph executions. Concurrent calls must use
separate workspaces or explicit CUDA synchronization. The original
``nvte_group_quantize`` remains available; its shared descriptor storage limits
MXFP8 groups with varying last dimensions to 64 and requires serialized use.

.. doxygenfile:: cast.h

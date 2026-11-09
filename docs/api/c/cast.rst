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
The query computes a TMA-specific layout from the group count. Query and execution
may use different tensor handles; no resource catalog or planning API is needed.

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
The query reserves one copy of the descriptor/metadata storage for the supplied
stream. Execution validates the tensor's declared byte capacity and base alignment;
the caller is responsible for the actual allocation behind the pointer.

Each descriptor and metadata array starts on a 128-byte boundary. Workspace size
and offsets are calculated with the grouped TMA layout; callers should query
requirements rather than hard-code sizes. See :doc:`workspace` for layout,
buffer ownership and CUDA Graph lifetime contracts.

Allocate before CUDA graph capture. Keep the workspace alive until all uses
complete, including captured graph executions. Use one workspace per CUDA stream;
the framework or application allocates separate memory for independently executing
streams. TE does not track streams or partition the allocation into stream slots.
Serialize complete host submissions and order all GPU uses of the same mutable
storage. Independently executing graphs also require separate workspaces.
The original ``nvte_group_quantize`` and ``nvte_group_dequantize`` reject MXFP8
layouts with varying last dimensions. Use their ``_v2`` variants with caller-owned
workspace for those layouts, including serialized use. There are no shared mutable
TMA descriptor arrays or implicit device allocations.

``nvte_group_dequantize_v2`` uses the same query/allocate/execute protocol and
TMA layout. Its descriptor updater and consumer use the same caller-owned
buffer; it supports more than 64 groups. PyTorch grouped MXFP8 quantize and
dequantize allocate workspace through the framework, including during graph
capture. The JAX grouped MXFP8 FFI retains the original API: its constant last
dimension uses launch-local descriptors and needs no descriptor arrays.

``nvte_group_requantize_v2`` uses the same workspace protocol for fused grouped
MXFP8 requantization, including varying last dimensions. The original
``nvte_group_requantize`` supports constant last dimensions and rejects layouts
requiring mutable descriptor arrays. Workspace queries validate the operation
without launching either the CUDA or CuTeDSL backend.

Only MXFP8 layouts with varying last dimensions need mutable descriptor arrays.
Constant last dimensions (uniform shapes or varying first dimensions) pass tensor
maps by value as kernel parameters. Grouped DBias, including fused activation
backward, requires a constant last dimension and retains its existing scratch
protocol. Its scratch tensor uses the existing backend-specific protocol.
Other grouped formats do not use these mutable MXFP8 descriptor arrays.

.. doxygenfile:: cast.h

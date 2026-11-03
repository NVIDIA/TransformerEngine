..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

TMA descriptor workspace
========================

Grouped MXFP8 operations with varying last dimensions require caller-owned
storage for mutable TMA descriptors and mapper metadata. This applies to grouped
quantization, activation forward/backward, and dequantization. Their ``_v2`` APIs
accept an ordinary ``NVTETensor`` representing the buffer. TE does not own device
memory or attach a resource catalog to the tensor.

Query and execution
---------------------

1. Call the operation with a workspace tensor whose rowwise data pointer is null.
   The call performs the same operation, format, and configuration checks as
   execution, then returns the required shape and dtype without launching GPU
   work or modifying outputs. This also applies to formats that need no TMA
   descriptor storage.
2. Allocate that storage through the framework or application. Descriptor storage
   requires a 128-byte aligned base pointer.
3. Set the workspace's rowwise data pointer and call the operation to execute.

.. code-block:: cpp

   transformer_engine::TensorWrapper requirement;
   nvte_group_quantize_v2(input, output, requirement.data(), config, stream);
   void *buffer = framework_allocate(requirement.shape(), requirement.dtype(), 128);
   transformer_engine::TensorWrapper workspace(buffer, {requirement.shape().data[0]},
                                               transformer_engine::DType::kByte);
   nvte_group_quantize_v2(input, output, workspace.data(), config, stream);

``framework_allocate`` is an application function. The query returns a
one-dimensional byte tensor. Layouts without mutable descriptor storage request
one byte to distinguish execution from a query; the kernel does not use that byte.
The buffer's canonical pointer is its rowwise data field, regardless of whether
the operation produces rowwise or columnwise quantized output.

Query and execution may use different tensor handles. Every call reconstructs
the layout from the operation's group count. A sufficiently large allocation can
be reused; its descriptors and metadata are rebuilt before each kernel invocation.
No initialization callback or persistent host state is required. Execution checks
the declared byte capacity and descriptor-buffer alignment. The caller provides
an actual allocation satisfying that declaration.

Layout and concurrency
------------------------

``GroupedTmaWorkspaceLayout`` in ``common/cast/core/grouped_tma_workspace.h``
contains four tensor-map arrays and three mapper arrays (rows, columns, offsets).
Each array starts on a 128-byte boundary. The same layout computes the query size
and resolves the non-owning ``GroupedTmaStorageView`` passed to GPU kernels.
Callers query sizes rather than hard-code offsets. More than 64 groups are supported.

Use separate allocations for independently executing CUDA streams. TE does not
register streams, assign slots, allocate device memory, or insert dependencies
between streams. The caller orders all uses of the same mutable buffer, including
complete descriptor-update and consumer submission sequences from host threads.

CUDA Graph lifetime
---------------------

Allocate the buffer before capturing C/C++ calls and keep it alive for all graph
executions that reference it. Capture stores the buffer address; rebinding a tensor
or changing the replay stream does not update that address. Independently
executing graphs need separate mutable storage, including graphs captured on the
same stream. Keep old allocations alive for old graphs when growing or replacing
storage, or retire those graphs and recapture.

Frameworks and other operations
--------------------------------

PyTorch grouped MXFP8 quantize/dequantize allocate temporary TMA workspace through
the framework, including graph-pool allocations during capture. Other formats
retain their existing API calls. The JAX grouped MXFP8 FFI also retains the
original API: its constant last dimension uses launch-local descriptors and
needs no descriptor arrays.

Cast+DBias and fused activation backward with DBias retain their existing
backend-specific scratch protocol. They do not use this TMA workspace layout.
Grouped MXFP8 DBias requires constant last dimensions and uses launch-local tensor
maps. Other grouped formats do not use the mutable MXFP8 descriptor arrays.

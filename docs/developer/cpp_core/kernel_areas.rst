..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _kernel-areas:

Kernel Areas
============

The C++ core organizes CUDA kernels into functional areas, each in its own subdirectory
under ``transformer_engine/common/``. Every area exposes a C API header in
``include/transformer_engine/`` and implements one or more CUDA kernels.

The major areas include GEMM (``gemm/``), normalization (``normalization/``),
quantization/cast (``cast/``), activation (``activation/``), fused attention
(``fused_attn/``), communication-GEMM overlap (``comm_gemm/``), and several others.
See the directory listing for the full set.

Rather than cataloging each area, this page describes the general architectural patterns
that a developer writing new kernels should understand.

Output-Driven Computation
--------------------------

A key design principle: the **output tensor** dictates what computation the kernel
performs. Kernels inspect the output ``NVTETensor`` to determine:

- Whether to produce rowwise data (``output.has_data()``), columnwise data
  (``output.has_columnwise_data()``), or both.
- The scaling mode (``output.scaling_mode``), which determines scale granularity.
- The output dtype, which determines the quantization target.

The input tensor is then validated to confirm it provides the fields needed for the
requested computation (e.g., the input must have data and the correct dtype). This
output-driven design makes sense because, for example, in a quantize kernel the input is
high-precision and knows nothing about rowwise/columnwise layouts — only the output
carries that information.

Scaling Mode Dispatch
----------------------

Kernels that handle multiple scaling modes use a switch on the output's
``scaling_mode`` to dispatch to the appropriate implementation. For example, in
``cast/dispatch/quantize.cuh``:

.. code-block:: text

   switch (output_tensor->scaling_mode):
     NVTE_DELAYED_TENSOR_SCALING  → fp8::quantize() or cast_transpose()
     NVTE_MXFP8_1D_SCALING       → mxfp8::quantize()
     NVTE_BLOCK_SCALING_1D/2D    → quantize_transpose_square_blockwise()
     NVTE_NVFP4_1D_SCALING       → nvfp4::quantize_transpose()

Each scaling mode may have separate code paths for rowwise-only, columnwise-only, or
both layouts, again driven by which fields are populated on the output tensor.

Architecture Dispatch
----------------------

Many kernel areas provide architecture-specific implementations for different GPU
generations (Ampere, Hopper, Blackwell). The dispatch mechanism varies by area:

- **Normalization** uses a ``KernelRegistry`` (in ``normalization/common.h``) that
  maps (hidden_size, dtype, warp_config) tuples to kernel function pointers. Kernels are
  heavily templated and the appropriate specialization is selected at runtime from the
  registry.

- **Other areas** (cast, GEMM, activation, etc.) use direct dispatch — typically switch
  statements or if/else chains on the compute capability. Some kernels only support
  certain GPU architectures (e.g., FP4 kernels require Blackwell).

There is no rigid pattern — the dispatch strategy is chosen per area based on what
makes sense for that area's complexity. The ``NVTE_CUDA_ARCHS`` CMake flag controls
which architectures are compiled. This affects which template instantiations are
generated, which in turn determines build time and binary size. See :doc:`build_system`
for details.

Fused Kernels
--------------

Many areas support fusing quantization with the primary computation:

- **Normalization + quantize**: The normalization kernel can produce FP8 output directly,
  avoiding a separate cast kernel launch.
- **Activation + quantize**: Similarly, activation functions can cast their output to FP8
  in the same kernel pass.
- **Cast + transpose**: A single kernel produces both rowwise and transposed (columnwise)
  output, critical for efficiently preparing data for the backward pass.

These fusions are driven by the same output-driven pattern: if the output tensor's dtype
is an FP8 type and has the appropriate scale fields, the kernel produces quantized output
directly.

For fused attention, see :doc:`/developer/attention/fused_attn_kernels` for dedicated
coverage.

For communication-GEMM overlap, see :doc:`/developer/distributed/comm_gemm_overlap`.

..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _fused-attn-kernels:

Fused Attention Kernels
=======================

The C++ fused attention implementation lives in ``transformer_engine/common/fused_attn/``.

C API
-----

**Header**: ``transformer_engine/common/include/transformer_engine/fused_attn.h``

Key functions:

- ``nvte_fused_attn_fwd()`` — Forward pass
- ``nvte_fused_attn_bwd()`` — Backward pass
- ``nvte_fused_attn_fwd_kvpacked()`` / ``nvte_fused_attn_bwd_kvpacked()`` — KV-packed
  variants (K and V in a single tensor)
- ``nvte_fused_attn_fwd_qkvpacked()`` / ``nvte_fused_attn_bwd_qkvpacked()`` — QKV-packed
  variants

Each function accepts:

- Q, K, V tensors (as ``NVTETensor``)
- Bias tensor (optional)
- Softmax auxiliary data (for backward — stores softmax statistics)
- Attention configuration (heads, scaling factor, dropout, mask type, etc.)
- Workspace tensor (pre-allocated scratch memory)

Directory Structure
-------------------

.. code-block:: text

   fused_attn/
   ├── fused_attn.h              # Internal C++ API
   ├── fused_attn.cpp            # C API implementation, backend dispatch
   ├── fused_attn_f16_max512_seqlen.h/.cu  # cuDNN F16 (short sequences)
   ├── fused_attn_f16_arbitrary_seqlen.h/.cu  # cuDNN F16 (any sequence length)
   ├── fused_attn_fp8.h/.cu      # cuDNN FP8
   ├── thd_utils.h               # THD layout utilities for CP
   └── utils.h                   # Shared utilities

cuDNN Integration
-----------------

The cuDNN-based backends use cuDNN's **graph API** (via the cuDNN Frontend library in
``3rdparty/cudnn-frontend``). The approach:

1. **Build a cuDNN graph** describing the attention computation (matmuls, softmax,
   dropout, scaling).
2. **Compile the graph** for the target GPU architecture.
3. **Execute the graph** with input tensors.

Graph compilation is expensive, so compiled graphs are cached for reuse across iterations.
The cache key includes: sequence lengths, head dimensions, number of heads, data types,
mask type, and dropout configuration.

FP8 Fused Attention
--------------------

The FP8 variant (``fused_attn_fp8.cu``) extends the cuDNN graph with quantization nodes:

- Q, K, V arrive in FP8 with associated scale inverses.
- Internal computation happens in higher precision (cuDNN manages this).
- Output can be produced in FP8 (with a new scale) or BF16.
- Softmax statistics are saved in FP32 for the backward pass.

Workspace Management
--------------------

Fused attention kernels require significant scratch memory. The workspace size is queried
before execution:

.. code-block:: cpp

   // Query workspace size
   nvte_fused_attn_fwd(q, k, v, ..., workspace, stream);
   // First call with workspace.data == nullptr returns required size
   // Second call with allocated workspace executes the kernel

This two-pass pattern (query size, then execute) is common across TE's C API.

Auxiliary Tensors (NVTETensorPack)
----------------------------------

The forward pass produces auxiliary tensors needed for the backward pass — softmax
statistics, RNG state, and optionally bias gradients. Because different sub-backends
produce different auxiliary tensors, they are passed through an ``NVTETensorPack``
(defined in ``transformer_engine.h``):

.. code-block:: c

   struct NVTETensorPack {
       static const int MAX_SIZE = 10;
       NVTETensor tensors[MAX_SIZE];
       size_t size = 0;
   };

Each sub-backend populates the pack differently:

- **Sub-backend 0** (F16 max512): 1 tensor — ``S`` (full softmax intermediate).
- **Sub-backend 1** (F16 arbitrary): 2+ tensors — ``Stats`` (log-sum-exp, always
  present), optionally ``Max`` (when ``return_max_logit=True``), ``rng_state``, and
  optionally ``Bias`` and ``SoftmaxOffset``.
- **Sub-backend 2** (FP8): 3 tensors — ``M`` (row max), ``ZInv`` (inverse softmax
  denominator), ``rng_state``.

On the Python/pybind11 side (``pytorch/csrc/extensions/attention.cpp``), the forward call
uses the two-pass pattern: the first call with empty tensors discovers the required shapes
and dtypes; PyTorch tensors are then allocated; the second call executes with the
memory-backed pack. The backward call receives the same pack to reuse the saved
statistics.

See Also
--------

- :doc:`backends` — How fused attention fits in the backend taxonomy
- :doc:`/developer/cpp_core/kernel_areas` — Other kernel areas

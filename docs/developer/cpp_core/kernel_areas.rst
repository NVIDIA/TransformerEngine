..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _kernel-areas:

Kernel Areas
============

The C++ core organizes CUDA kernels into functional areas, each in its own subdirectory
under ``transformer_engine/common/``. Every area exposes a C API header in
``include/transformer_engine/`` and implements one or more CUDA kernels.

GEMM (``gemm/``)
-----------------

Matrix multiplication via cuBLASLt with FP8/MXFP8/NVFP4 support.

- **Header**: ``include/transformer_engine/gemm.h``
- **Key file**: ``gemm/cublaslt_gemm.cu``
- **Entry point**: ``nvte_general_gemm()``
- Handles: scale/scale-inverse application, bias addition, pre-GeLU fusion, grouped GEMM
- Dispatches to cuBLASLt with appropriate compute types based on input precision and
  scaling mode.

Normalization (``normalization/``)
----------------------------------

LayerNorm and RMSNorm with optional FP8 output quantization.

- **Header**: ``include/transformer_engine/normalization.h``
- **Key files**: ``normalization/layernorm/``, ``normalization/rmsnorm/``
- Variants: forward, backward, fused with quantization (cast output to FP8 in the same
  kernel to save memory bandwidth)
- Architecture dispatch: separate kernel implementations for different GPU architectures
  (e.g., Hopper uses warp-specialized kernels).

Activation (``activation/``)
-----------------------------

Element-wise activation functions with optional FP8 output.

- **Header**: ``include/transformer_engine/activation.h``
- Supports: GeLU, SiLU/Swish, ReLU, QuickGeLU, SReLU, and their gated variants
- Fused quantization: activation + cast to FP8 in a single kernel pass.

Cast (``cast/``)
-----------------

Type-casting and quantization kernels.

- **Header**: ``include/transformer_engine/cast.h``
- ``nvte_quantize()`` — cast high-precision data to quantized format with scale computation
- Handles all scaling modes (tensor, block, MXFP8, NVFP4) with mode-specific kernels.

Transpose (``transpose/``)
---------------------------

Transpose operations, often fused with casting.

- **Header**: ``include/transformer_engine/transpose.h``
- ``nvte_transpose()`` — standalone transpose
- Fused variants: cast + transpose in a single kernel (critical for producing columnwise
  data efficiently during forward pass).

Fused Attention (``fused_attn/``)
---------------------------------

The largest and most complex kernel area. See :doc:`/developer/attention/fused_attn_kernels`
for detailed coverage.

- **Header**: ``include/transformer_engine/fused_attn.h``
- Multiple backends: cuDNN-based (F16 and FP8), custom CUDA kernels
- Supports: MHA, GQA, MQA, arbitrary head dims, causal/padding masks, dropout,
  sliding window, FP8 quantization.

Fused RoPE (``fused_rope/``)
-----------------------------

Rotary positional embedding applied as a fused CUDA kernel.

- **Header**: ``include/transformer_engine/fused_rope.h``
- Applies rotary embedding in-place during forward and backward passes.

Fused Softmax (``fused_softmax/``)
-----------------------------------

Scaled softmax variants optimized for attention score computation.

- **Header**: ``include/transformer_engine/softmax.h``
- Variants: scaled softmax, scaled masked softmax, scaled upper-triangular masked softmax.

Fused Router (``fused_router/``)
---------------------------------

Mixture-of-Experts (MoE) routing kernels.

- Permutation and unpermutation of tokens across experts.
- Includes its own type-switch macros (note: these need DType casting, see
  :doc:`type_system`).

Communication-GEMM Overlap (``comm_gemm/``)
--------------------------------------------

Overlapping NCCL collectives with GEMM computation using CUDA multi-stream.

- See :doc:`/developer/distributed/comm_gemm_overlap` for the design.
- Uses CUDA events and user-allocated buffers (``UserBuffers``) to pipeline communication
  and computation.

Multi-Tensor Operations (``multi_tensor/``)
-------------------------------------------

Fused operations over lists of tensors (e.g., multi-tensor scale for optimizer steps).

- **Header**: ``include/transformer_engine/multi_tensor.h``

Architecture Dispatch
---------------------

Many kernel areas provide architecture-specific implementations. The typical pattern:

.. code-block:: text

   normalization/
   ├── common.h           # Shared types and helpers
   ├── layernorm/
   │   ├── ln_api.cpp     # C API entry point, dispatches by arch
   │   ├── ln_sm80.cu     # Ampere (SM80) kernel
   │   ├── ln_sm90.cu     # Hopper (SM90) kernel
   │   └── ln_sm100.cu    # Blackwell (SM100) kernel

The C API function queries the GPU architecture at runtime and dispatches to the
appropriate kernel. See :doc:`build_system` for how ``NVTE_CUDA_ARCHS`` controls which
architectures are compiled.

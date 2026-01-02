..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _environment_variables:

Environment Variables
=====================

This document describes the environment variables used by Transformer Engine. They provide an alternate method to alter Transformer Engine's behavior during build and runtime, but are less rigorously maintained compared to the API and may be subject to change.

Build-Time Environment Variables
---------------------------------

These environment variables control the build and compilation process of Transformer Engine.

Build Configuration
^^^^^^^^^^^^^^^^^^^

.. envvar:: NVTE_BUILD_DEBUG

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Enable debug build mode. When set to ``1``, the build includes debug symbols (``-g``) and disables optimizations.

.. envvar:: NVTE_BUILD_MAX_JOBS

   :Type: ``int``
   :Default: Maximum available
   :Description: Number of parallel jobs to use during the build process. If not set, the system will use the maximum available parallel jobs. Also respects the standard ``MAX_JOBS`` environment variable.

.. envvar:: NVTE_BUILD_THREADS_PER_JOB

   :Type: ``int``
   :Default: ``1``
   :Description: Number of threads to use per parallel build job. This is passed to the CUDA compiler via the ``--threads`` flag.

.. envvar:: NVTE_FRAMEWORK

   :Type: ``str``
   :Default: Auto-detected
   :Description: Comma-separated list of frameworks to build support for (``pytorch``, ``jax``, ``all``, or ``none``). If not specified, automatically detects installed frameworks.

.. envvar:: NVTE_USE_CCACHE

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Enable ccache for faster recompilation. When set to ``1``, uses ccache as a compiler launcher for both C++ and CUDA compilation.

.. envvar:: NVTE_CCACHE_BIN

   :Type: ``str``
   :Default: ``ccache``
   :Description: Path to the ccache binary. Only used when :envvar:`NVTE_USE_CCACHE` is set to ``1``.

.. envvar:: NVTE_CMAKE_BUILD_DIR

   :Type: ``str``
   :Default: None
   :Description: Path to the CMake build directory for incremental builds. If set, CMake will use this directory for build artifacts.

.. envvar:: NVTE_RELEASE_BUILD

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Enable release build mode. When set to ``1``, prepares the build for distribution (e.g., PyPI wheel). This affects library installation paths and build tool management.

.. envvar:: NVTE_PROJECT_BUILDING

   :Type: ``int`` (0 or 1)
   :Default: Not set
   :Description: Internal flag set to ``1`` during the build process to indicate that the project is being built. Not intended for external use.

Optional Dependencies
^^^^^^^^^^^^^^^^^^^^^

.. envvar:: NVTE_UB_WITH_MPI

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Enable MPI support for userbuffers. When set to ``1``, requires ``MPI_HOME`` to be set to the MPI installation directory.

.. envvar:: NVTE_ENABLE_NVSHMEM

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Enable NVSHMEM support. When set to ``1``, requires ``NVSHMEM_HOME`` to be set to the NVSHMEM installation directory.

.. envvar:: NVTE_BUILD_ACTIVATION_WITH_FAST_MATH

   :Type: CMake option
   :Default: ``OFF``
   :Description: Compile activation kernels (GELU, ReLU, SwiGLU) with the ``--use_fast_math`` CUDA compiler flag for improved performance at the cost of some precision.

CUDA Configuration
^^^^^^^^^^^^^^^^^^

.. envvar:: NVTE_CUDA_ARCHS

   :Type: ``str``
   :Default: Auto-detected based on CUDA version
   :Description: Semicolon-separated list of CUDA compute architectures to compile for (e.g., ``"80;90"`` for A100 and H100, or ``"75;80;89;90"``). If not set, automatically determined based on the installed CUDA Toolkit version. CUDA 13.0+ defaults to ``"75;80;89;90;100;120"``, CUDA 12.8+ defaults to ``"70;80;89;90;100;120"``, and earlier versions default to ``"70;80;89;90"``. Setting this can significantly reduce build time and binary size by targeting only the GPU architectures you need.

.. envvar:: NVTE_CUDA_INCLUDE_DIR

   :Type: ``str``
   :Default: Auto-detected
   :Description: Path to CUDA include directory containing ``cuda_runtime.h``. If not set, Transformer Engine searches in common locations (``CUDA_HOME``, ``CUDA_DIR``, ``/usr/local/cuda``). This is used for NVRTC kernel compilation.

Runtime Environment Variables
------------------------------

These environment variables control the behavior of Transformer Engine during execution.

Attention Backend Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. envvar:: NVTE_FLASH_ATTN

   :Type: ``int`` (0 or 1)
   :Default: ``1``
   :Description: Enable or disable FlashAttention backend for DotProductAttention. When set to ``0``, FlashAttention will not be used.

.. envvar:: NVTE_FUSED_ATTN

   :Type: ``int`` (0 or 1)
   :Default: ``1``
   :Description: Enable or disable FusedAttention backend (cuDNN-based) for DotProductAttention. When set to ``0``, FusedAttention will not be used.

.. envvar:: NVTE_UNFUSED_ATTN

   :Type: ``int`` (0 or 1)
   :Default: ``1``
   :Description: Enable or disable UnfusedDotProductAttention backend (native PyTorch). When set to ``0``, UnfusedDotProductAttention will not be used.

.. envvar:: NVTE_FUSED_ATTN_BACKEND

   :Type: ``int`` (0, 1, or 2)
   :Default: Auto-selected
   :Description: Force a specific FusedAttention backend. ``0`` = F16_max512_seqlen (cuDNN, ≤512 seq len), ``1`` = F16_arbitrary_seqlen (cuDNN, any seq len), ``2`` = FP8 backend. If not set, the backend is automatically selected based on the input configuration.

.. envvar:: NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT

   :Type: ``int`` (0 or 1)
   :Default: Auto-determined
   :Description: Control workspace-related optimizations in FusedAttention. ``0`` disables optimizations, ``1`` enables them. These optimizations trade memory for performance. When unset, Transformer Engine determines the code path based on internal logic. For deterministic behavior with cuDNN ≥8.9.5 and <9.0.0, this is automatically set to ``1``.

.. envvar:: NVTE_FUSED_ATTN_USE_FAv2_BWD

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: When using FusedAttention, use FlashAttention-2 implementation for the backward pass instead of the cuDNN implementation. This can be useful due to performance differences between various versions of flash-attn and FusedAttention.

.. envvar:: NVTE_ALLOW_NONDETERMINISTIC_ALGO

   :Type: ``int`` (0 or 1)
   :Default: ``1``
   :Description: Allow non-deterministic algorithms for Transformer Engine execution. When set to ``0``, only deterministic algorithms are allowed. This is relevant for both PyTorch and JAX attention implementations.

.. envvar:: NVTE_FUSED_RING_ATTENTION_USE_SCAN

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: **(JAX only)** Use scan loop for ring attention implementation. When set to ``1``, the fused ring attention will use a scan-based iteration approach.

.. envvar:: NVTE_APPLY_QK_LAYER_SCALING

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Apply QK layer scaling in UnfusedDotProductAttention. This is an FP16 training trick required for certain GPT-like models. When set to ``1`` and a layer number is provided, the softmax scale is divided by the layer number, and the layer number is used as the softmax scale during the softmax operation. Only effective when using FP16 dtype and when the layer number is specified.

Context Parallelism
^^^^^^^^^^^^^^^^^^^

.. envvar:: NVTE_BATCH_MHA_P2P_COMM

   :Type: ``int`` (0 or 1)
   :Default: ``0`` (or auto-enabled for pre-Blackwell GPUs with CP size 2)
   :Description: Use batched P2P communication (``batch_isend_irecv``) for KV exchange in context parallel MultiheadAttention. When enabled, send and receive operations are batched together, which can improve communication efficiency. This is automatically enabled for devices with compute capability < 10.0 (pre-Blackwell GPUs) when context parallel size is 2. Setting this to ``1`` forces batched P2P communication regardless of device architecture.

FP8 Configuration
^^^^^^^^^^^^^^^^^

.. envvar:: NVTE_UNFUSED_FP8_UPDATE

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Use unfused kernel for FP8 amax and scale updates. When set to ``1``, amax and scale updates are computed using separate unfused kernels instead of fused operations.

.. envvar:: NVTE_FP8_DPA_BWD

   :Type: ``int`` (0 or 1)
   :Default: ``1``
   :Description: Enable FP8 in the backward pass of DotProductAttention. ``1`` = FP8 forward and backward, ``0`` = FP8 forward and FP16/BF16 backward.

.. envvar:: NVTE_DPA_FP8CS_O_in_F16

   :Type: ``int`` (0 or 1)
   :Default: ``1``
   :Description: For Float8CurrentScaling in DotProductAttention, use FP16/BF16 for the output tensor in the backward pass. ``1`` = use F16/BF16 output in backward, ``0`` = use FP8 output in backward.

.. envvar:: NVTE_DPA_FP8_RECIPE

   :Type: ``str``
   :Default: Empty (use same as linear layers)
   :Description: Override FP8 recipe for DotProductAttention layers. Valid values: ``"F16"`` (disable FP8), ``"DelayedScaling"``, or ``"Float8CurrentScaling"``. This allows using different FP8 recipes for attention vs. linear layers.

.. envvar:: NVTE_DPA_FP8_RECIPE_DPA

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Enable FP8 in DotProductAttention when using :envvar:`NVTE_DPA_FP8_RECIPE`. When set to ``1``, the DotProductAttention layer will use the FP8 recipe specified by :envvar:`NVTE_DPA_FP8_RECIPE`. This provides fine-grained control over which attention components use FP8.

.. envvar:: NVTE_DPA_FP8_RECIPE_MHA

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Enable FP8 in MultiheadAttention (MHA) when using :envvar:`NVTE_DPA_FP8_RECIPE`. When set to ``1``, the MultiheadAttention QKV and output projection layers will use the FP8 recipe specified by :envvar:`NVTE_DPA_FP8_RECIPE`. This provides fine-grained control over which attention components use FP8.

.. envvar:: NVTE_DPA_FP8_FORMAT

   :Type: ``str``
   :Default: ``"HYBRID"``
   :Description: FP8 format for DotProductAttention when switching recipes. Valid values: ``"HYBRID"``, ``"E4M3"``, ``"E5M2"``. Only used when :envvar:`NVTE_DPA_FP8_RECIPE` is set.

.. envvar:: NVTE_DPA_FP8DS_AMAX_ALGO

   :Type: ``str``
   :Default: ``"most_recent"``
   :Description: Amax computation algorithm for DelayedScaling recipe in DotProductAttention. Valid values: ``"most_recent"``, ``"max"``. Only used when :envvar:`NVTE_DPA_FP8_RECIPE` is set to ``"DelayedScaling"``.

.. envvar:: NVTE_DPA_FP8DS_AMAX_HISTLEN

   :Type: ``int``
   :Default: ``1``
   :Description: Amax history length for DelayedScaling recipe in DotProductAttention. Only used when :envvar:`NVTE_DPA_FP8_RECIPE` is set to ``"DelayedScaling"``.

.. envvar:: NVTE_DPA_FP8DS_REDUCE_AMAX

   :Type: ``int`` (0 or 1)
   :Default: ``1``
   :Description: Reduce amax across distributed ranks for DelayedScaling recipe in DotProductAttention. Only used when :envvar:`NVTE_DPA_FP8_RECIPE` is set to ``"DelayedScaling"``.

.. envvar:: NVTE_UnfusedDPA_Emulate_FP8

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Allow FP8 emulation in UnfusedDotProductAttention. When set to ``1``, UnfusedDotProductAttention can emulate FP8 operations using FP16/BF16 computation.

Kernel Configuration
^^^^^^^^^^^^^^^^^^^^

.. envvar:: NVTE_USE_FAST_MATH

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Enable fast math optimizations in runtime-compiled (NVRTC) kernels. This trades numerical accuracy for performance. These optimizations are experimental and inconsistently implemented.

.. envvar:: NVTE_DISABLE_NVRTC

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Disable NVRTC (CUDA Runtime Compilation) support. When set to ``1``, runtime kernel compilation is disabled. This can be useful in environments where NVRTC is not available or not desired.

.. envvar:: NVTE_USE_CUTLASS_GROUPED_GEMM

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Use CUTLASS implementation for grouped GEMM operations instead of cuBLAS. When set to ``1``, enables CUTLASS grouped GEMM kernels, which may provide better performance for certain workloads on Hopper (SM90) GPUs.

.. envvar:: NVTE_CUTLASS_GROUPED_GEMM_WARN_FALLBACK

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Emit a warning when falling back from CUTLASS to cuBLAS for grouped GEMM operations.

Torch Compilation and Fusion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. envvar:: NVTE_TORCH_COMPILE

   :Type: ``int`` (0 or 1)
   :Default: ``1``
   :Description: Enable PyTorch 2.x ``torch.compile`` support for compatible Transformer Engine operations. When set to ``0``, disables compilation support and uses regular PyTorch eager mode.

.. envvar:: NVTE_BIAS_GELU_NVFUSION

   :Type: ``int`` (0 or 1)
   :Default: ``1``
   :Description: Enable GELU fusion with bias using NVFusion in PyTorch. When set to ``0``, uses separate bias and GELU operations.

.. envvar:: NVTE_BIAS_DROPOUT_FUSION

   :Type: ``int`` (0 or 1)
   :Default: ``1``
   :Description: Enable fusion of bias and dropout operations. When set to ``0``, bias and dropout are computed separately.

LayerNorm/RMSNorm SM Margins
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. envvar:: NVTE_FWD_LAYERNORM_SM_MARGIN

   :Type: ``int``
   :Default: ``0``
   :Description: Number of SMs (Streaming Multiprocessors) to reserve (not use) during forward LayerNorm/RMSNorm operations. This can be used to control resource allocation and overlap computation with communication.

.. envvar:: NVTE_BWD_LAYERNORM_SM_MARGIN

   :Type: ``int``
   :Default: ``0``
   :Description: Number of SMs to reserve during backward LayerNorm/RMSNorm operations.

.. envvar:: NVTE_INF_LAYERNORM_SM_MARGIN

   :Type: ``int``
   :Default: ``0``
   :Description: Number of SMs to reserve during inference LayerNorm/RMSNorm operations.

GEMM Configuration
^^^^^^^^^^^^^^^^^^

.. envvar:: NVTE_EXT_MARGIN_SM

   :Type: ``int``
   :Default: Total SM count
   :Description: External SM margin for GEMM operations. Specifies the number of SMs to use for GEMM operations. The actual number of SMs used is ``sm_count - NVTE_EXT_MARGIN_SM``.

.. envvar:: NVTE_AG_P2P_MULTI_ATOMIC

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Enable multi-atomic mode for AllGather with atomic GEMM using P2P communication. When set to ``1``, uses ``userbuffers_sendrecv_multiatomic`` for communication during atomic GEMM overlap with AllGather operations. This disables copy engine (CE) usage and enables push mode for userbuffers. This is an advanced optimization for tensor-parallel communication-computation overlap.

CPU Offloading
^^^^^^^^^^^^^^

.. envvar:: NVTE_CPU_OFFLOAD_V1

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Enable legacy version of CPU offloading implementation. 

Debugging and Profiling
^^^^^^^^^^^^^^^^^^^^^^^^

.. envvar:: NVTE_DEBUG

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Enable debug mode. When set to ``1``, enables verbose debug output and additional checks in attention operations.

.. envvar:: NVTE_DEBUG_LEVEL

   :Type: ``int`` (0, 1, or 2)
   :Default: ``0``
   :Description: Debug verbosity level. Higher values enable more verbose debug output. Only effective when :envvar:`NVTE_DEBUG` is set to ``1``.

.. envvar:: NVTE_PRINT_LAYER_NUMBER

   :Type: ``int``
   :Default: ``1``
   :Description: Layer number to print debug information for during attention operations.

.. envvar:: NVTE_PRINT_RANK

   :Type: ``int``
   :Default: ``0``
   :Description: Distributed rank to print debug information for during attention operations.

.. envvar:: NVTE_NVTX_ENABLED

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Enable NVTX (NVIDIA Tools Extension) range profiling for Transformer Engine operations. When set to ``1``, NVTX markers are added to operations for profiling with NVIDIA Nsight Systems.

.. envvar:: NVTE_DEBUG_NUMERICS

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: **(JAX only)** Enable verbose printing of tensor numerics for debugging purposes.

Testing
^^^^^^^

.. envvar:: NVTE_TEST_NVINSPECT_ENABLED

   :Type: ``int`` (0 or 1)
   :Default: ``0``
   :Description: Enable NVInspect integration for testing. When set to ``1``, enables the NVInspect debugging API for numerical analysis during tests.

.. envvar:: NVTE_TEST_NVINSPECT_CONFIG_FILE

   :Type: ``str``
   :Default: None
   :Description: Path to NVInspect configuration file. Required when :envvar:`NVTE_TEST_NVINSPECT_ENABLED` is set to ``1``.

.. envvar:: NVTE_TEST_NVINSPECT_FEATURE_DIRS

   :Type: ``str``
   :Default: None
   :Description: Comma-separated list of directories containing NVInspect features. Required when :envvar:`NVTE_TEST_NVINSPECT_ENABLED` is set to ``1``.

.. envvar:: NVTE_TEST_ARTIFACTS_DIR

   :Type: ``str``
   :Default: System temp directory
   :Description: Directory for storing test artifacts (e.g., generated ONNX models).

ONNX Export
^^^^^^^^^^^

.. envvar:: NVTE_ONNX_KVCACHE_MAX_SEQ_LEN

   :Type: ``int``
   :Default: ``128``
   :Description: Maximum sequence length for KV cache during ONNX export. This is used for attention masking in exported ONNX models.

JAX-Specific Variables
^^^^^^^^^^^^^^^^^^^^^^

.. envvar:: NVTE_JAX_CUSTOM_CALLS

   :Type: ``str``
   :Default: None
   :Description: Control which JAX custom call primitives are enabled or disabled. Format: ``"true"`` (enable all), ``"false"`` (disable all), or comma-separated key-value pairs like ``"GemmPrimitive=false,DBiasQuantizePrimitive=true"``. This provides fine-grained control over which operations use custom CUDA kernels vs. JAX native implementations.

.. envvar:: NVTE_JAX_CUSTOM_CALLS_RE

   :Type: ``str``
   :Default: None
   :Description: **Deprecated** (use :envvar:`NVTE_JAX_CUSTOM_CALLS` instead). Regex pattern to match primitive names for enabling/disabling. Example: ``"DBiasQuantizePrimitive"`` or ``"^(?!DBiasQuantizePrimitive$).+$"``.

.. envvar:: NVTE_JAX_UNITTEST_LEVEL

   :Type: ``str``
   :Default: None
   :Description: Test level for JAX unit tests (``"L0"``, ``"L1"``, ``"L2"``). Used internally by the test suite.

Examples
--------

Building with Debug Symbols
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   export NVTE_BUILD_DEBUG=1
   export NVTE_USE_CCACHE=1
   pip install -e .

Using Specific Attention Backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Use only FlashAttention, disable FusedAttention
   export NVTE_FLASH_ATTN=1
   export NVTE_FUSED_ATTN=0
   python train.py

Configuring FP8 for Attention
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Use DelayedScaling for attention, CurrentScaling for linear layers
   export NVTE_DPA_FP8_RECIPE="DelayedScaling"
   export NVTE_DPA_FP8_FORMAT="HYBRID"
   export NVTE_DPA_FP8DS_AMAX_ALGO="most_recent"
   export NVTE_DPA_FP8DS_AMAX_HISTLEN=1024
   python train.py

Enable Profiling
^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Enable NVTX markers for profiling
   export NVTE_NVTX_ENABLED=1
   nsys profile --trace=nvtx,cuda python train.py

JAX Custom Calls Control
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Disable all custom calls
   export NVTE_JAX_CUSTOM_CALLS="false"
   python train_jax.py

   # Disable specific primitives
   export NVTE_JAX_CUSTOM_CALLS="GemmPrimitive=false,DBiasQuantizePrimitive=false"
   python train_jax.py

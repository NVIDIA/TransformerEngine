..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _scaling-modes:

Scaling Modes
=============

Transformer Engine supports multiple quantization scaling strategies, each offering
different trade-offs between precision, performance, and hardware requirements. The
scaling mode is selected via the ``NVTEScalingMode`` enum and propagated through the
``Tensor.scaling_mode`` field.

.. figure:: ./img/scaling_modes_comparison.svg
   :align: center
   :width: 80%

   Comparison of scale granularity across scaling modes.

..
   Diagram description for ``scaling_modes_comparison.svg``:
   A table/matrix with 5 rows (one per scaling mode) and columns:
   Mode Name | Enum Value | Scale Granularity | Block Size | Data Type | Min Arch.
   DELAYED_TENSOR_SCALING | 0 | 1 scale per tensor | N/A | FP8 E4M3/E5M2 | Hopper
   MXFP8_1D_SCALING | 1 | 1 scale per 32 elements | 32 | FP8 + E8M0 scales | Blackwell
   BLOCK_SCALING_1D | 2 | 1 scale per 1×N block | configurable | FP8 E4M3/E5M2 | Blackwell
   BLOCK_SCALING_2D | 3 | 1 scale per N×N block | configurable | FP8 E4M3/E5M2 | Blackwell
   NVFP4_1D_SCALING | 4 | 1 scale per 16 elements | 16 | FP4 E2M1 + E8M0 | Blackwell

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 25 10 25 15 25

   * - Mode
     - Enum
     - Scale Granularity
     - Block Size
     - Data Type
   * - Delayed Tensor Scaling
     - ``0``
     - 1 scale per tensor
     - N/A
     - FP8 E4M3 / E5M2
   * - MXFP8 1D Scaling
     - ``1``
     - 1 scale per 32 elements
     - 32
     - FP8 + E8M0 scales
   * - Block Scaling 1D
     - ``2``
     - 1 scale per 1×N block
     - Configurable
     - FP8 E4M3 / E5M2
   * - Block Scaling 2D
     - ``3``
     - 1 scale per N×N block
     - Configurable
     - FP8 E4M3 / E5M2
   * - NVFP4 1D Scaling
     - ``4``
     - 1 scale per 16 elements
     - 16
     - FP4 E2M1 + E8M0

Delayed Tensor Scaling (``NVTE_DELAYED_TENSOR_SCALING``)
--------------------------------------------------------

The original FP8 scaling mode. A single scale factor applies to the entire tensor,
computed from the *previous iteration's* amax (absolute maximum) value.

**How it works:**

1. After each forward/backward pass, the kernel records the amax of the output.
2. The ``FP8GlobalStateManager`` maintains an amax history window (typically 1024 values).
3. Before the next iteration, the scale is computed as:
   ``scale = fp8_max / amax_history.max()``
4. This scale is applied uniformly to all elements during quantization.

**Trade-offs:**

- Simple and fast (no per-element scale overhead in GEMM).
- Scale is one iteration stale — can cause overflow/underflow on loss spikes.
- Requires the amax feedback loop between iterations.

**C++ helpers:**

.. code-block:: cpp

   bool is_tensor_scaling(const NVTEScalingMode &mode);
   bool is_delayed_tensor_scaling(const NVTEScalingMode &mode);

MXFP8 1D Scaling (``NVTE_MXFP8_1D_SCALING``)
----------------------------------------------

Microscaling FP8 format per the OCP MX specification. Each block of 32 contiguous
elements shares a single E8M0 (8-bit exponent-only) scale factor.

**Key details:**

- Block size is fixed at 32 elements.
- Scales are E8M0 format (power-of-two only, no mantissa bits).
- Scales may need "GEMM swizzling" — a specific memory layout that cuBLASLt expects.
  The ``with_gemm_swizzled_scales`` flag on ``Tensor`` tracks this.
- Current scaling (no amax history needed).

**C++ helpers:**

.. code-block:: cpp

   bool is_mxfp8_scaling(const NVTEScalingMode &mode);
   bool is_mxfp_scaling(const NVTEScalingMode &mode);  // alias

Block Scaling (``NVTE_BLOCK_SCALING_1D`` / ``NVTE_BLOCK_SCALING_2D``)
----------------------------------------------------------------------

Per-block FP8 scaling with configurable block dimensions. ``1D`` uses 1×N blocks
(one scale per row-slice of N elements), while ``2D`` uses N×N blocks.

**Key details:**

- Block size is configurable via the quantizer's ``block_scaling_dim`` property.
- Scales are FP32, not E8M0 (more precise than MXFP8).
- Current scaling (computed just-in-time, no history).
- Available on Blackwell and later architectures.

**C++ helpers:**

.. code-block:: cpp

   bool is_block_scaling(const NVTEScalingMode &mode);
   // Returns true for all non-tensor-scaling modes

NVFP4 1D Scaling (``NVTE_NVFP4_1D_SCALING``)
----------------------------------------------

4-bit floating point with E8M0 block scales, targeting maximum compression.

**Key details:**

- Data is FP4 E2M1 format (2 exponent bits, 1 mantissa bit).
- Each block of 16 elements shares one E8M0 scale.
- Like MXFP8, may require GEMM-swizzled scales.
- Blackwell and later only.

**C++ helpers:**

.. code-block:: cpp

   bool is_nvfp4_scaling(const NVTEScalingMode &mode);
   bool is_nvfp_scaling(const NVTEScalingMode &mode);  // alias

Scale Storage and Layout
------------------------

For all block-scaling modes, scale tensors are stored alongside the data tensor.
The ``Tensor`` struct provides separate scale inverse fields for rowwise and columnwise
layouts:

- ``scale_inv`` — scale inverses for decoding rowwise data (used in forward GEMM).
- ``columnwise_scale_inv`` — scale inverses for decoding columnwise data (used in wgrad
  GEMM).

See :doc:`/developer/quantization/rowwise_columnwise` for how these layouts interact
with GEMM execution.

Checking Scaling Mode in Code
-----------------------------

The ``common.h`` header provides inline helpers that should be preferred over direct
enum comparison:

.. code-block:: cpp

   // Prefer this:
   if (is_mxfp8_scaling(tensor.scaling_mode)) { ... }

   // Over this:
   if (tensor.scaling_mode == NVTE_MXFP8_1D_SCALING) { ... }

This makes code resilient to future additions to the ``NVTEScalingMode`` enum.

..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Datatype and hardware support matrix
====================================

This page summarizes which low-precision quantization formats Transformer Engine
supports on which NVIDIA GPU architectures. Support is keyed by CUDA *compute
capability*, since that is what Transformer Engine checks at runtime to decide
whether a format is available.

.. note::

   The support conditions below mirror the runtime capability checks in
   ``transformer_engine.pytorch.quantization``
   (``_compute_fp8_support``, ``_compute_fp8_block_scaling_support``,
   ``_compute_mxfp8_support``, ``_compute_nvfp4_support``) and the BF16 check in
   ``transformer_engine.pytorch.utils``. If those checks change, update this
   table to match.

Compute capability reference
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Compute capability
     - Architecture
     - Representative GPUs
   * - 8.0, 8.6
     - Ampere
     - A100, A10, A40, RTX 30 series
   * - 8.9
     - Ada Lovelace
     - L4, L40S, RTX 40 series
   * - 9.0
     - Hopper
     - H100, H200
   * - 10.0, 10.3
     - Blackwell (data center)
     - B200, B300, GB300
   * - 12.0
     - Blackwell (workstation / consumer)
     - RTX PRO 6000, RTX 50 series

The architecture and GPU columns are indicative; the compute capability column is
the value Transformer Engine uses for its support decisions.

Format support by compute capability
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 13 15 17 17 16

   * - Compute capability
     - BF16
     - FP8 (per tensor)
     - FP8 block scaling
     - MXFP8
     - NVFP4
   * - 8.0, 8.6 (Ampere)
     - Yes
     - No
     - No
     - No
     - No
   * - 8.9 (Ada)
     - Yes
     - Yes [1]_
     - No
     - No
     - No
   * - 9.0 (Hopper)
     - Yes
     - Yes
     - Yes [2]_
     - No
     - No
   * - 10.0, 10.3 (Blackwell DC)
     - Yes
     - Yes
     - Yes [2]_
     - Yes
     - Yes
   * - 12.0 (Blackwell workstation)
     - Yes
     - Yes
     - Yes [2]_
     - No [3]_
     - Yes [4]_

* **BF16** requires compute capability 8.0 or higher.
* **FP8 (per tensor)** covers the :class:`~transformer_engine.common.recipe.DelayedScaling`
  and :class:`~transformer_engine.common.recipe.Float8CurrentScaling` recipes.
  It requires compute capability 8.9 or higher.
* **FP8 block scaling** is the :class:`~transformer_engine.common.recipe.Float8BlockScaling`
  recipe.
* **MXFP8** is the :class:`~transformer_engine.common.recipe.MXFP8BlockScaling` recipe.
* **NVFP4** is the :class:`~transformer_engine.common.recipe.NVFP4BlockScaling` recipe.

.. [1] On Ada (compute capability 8.9), FP8 additionally requires cuBLASLt
   version 12.1.3.x or higher and CUDA 12.1 or higher.

.. [2] FP8 block scaling additionally requires CUDA 12.9 or higher.

.. [3] MXFP8 is not yet supported on compute capability 12.0 and higher
   (support is currently limited to compute capability 10.0 through 10.x).

.. [4] The capability check reports NVFP4 as available on compute capability
   10.0 and higher. The default NVFP4 recipe additionally uses a random Hadamard
   transform and stochastic rounding, whose FP4 conversion instructions are
   architecture specific to compute capability 10.0 and 10.3. On compute
   capability 12.0, running the default recipe currently raises an
   architecture-specific error, and NVFP4 there requires round-to-nearest
   (stochastic rounding disabled).

Default recipe by architecture
-------------------------------

When no recipe is passed explicitly, Transformer Engine selects a default based on
the device (see ``get_default_fp8_recipe`` in
``transformer_engine.pytorch.quantization``):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Compute capability
     - Default recipe
   * - 8.9, 9.0 (Ada, Hopper)
     - ``DelayedScaling``
   * - 10.0, 10.3 (Blackwell DC)
     - ``MXFP8BlockScaling``
   * - 12.0 (Blackwell workstation)
     - ``Float8CurrentScaling`` (temporary, until MXFP8 supports all GEMM layouts)

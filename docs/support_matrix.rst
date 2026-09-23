..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Datatype and hardware support matrix
====================================

This page summarizes which low-precision quantization formats Transformer Engine
supports on NVIDIA GPU architectures (by compute capability).

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

The architecture and GPU columns show representative examples. For a complete list,
see NVIDIA's `CUDA GPU compute capability list <https://developer.nvidia.com/cuda/gpus>`_.

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
     - Partial [4]_

* **BF16** requires compute capability 8.0 or higher.
* **FP8 (per tensor)** covers the :class:`~transformer_engine.common.recipe.DelayedScaling`
  and :class:`~transformer_engine.common.recipe.Float8CurrentScaling` recipes.
  It requires compute capability 8.9 or higher.
* **FP8 block scaling** is the :class:`~transformer_engine.common.recipe.Float8BlockScaling`
  recipe.
* **MXFP8** is the :class:`~transformer_engine.common.recipe.MXFP8BlockScaling` recipe.
* **NVFP4** is the :class:`~transformer_engine.common.recipe.NVFP4BlockScaling` recipe.

.. [1] On Ada (compute capability 8.9), FP8 additionally requires cuBLASLt
   version 12.1.3.x or higher.

.. [2] FP8 block scaling additionally requires CUDA 12.9 or higher.

.. [3] MXFP8 is not yet supported on compute capability 12.0 and higher
   (support is currently limited to compute capability 10.0 through 10.x).

.. [4] The full NVFP4 recipe is supported on compute capability 10.x GPUs. On
   12.x GPUs, NVFP4 supports the forward pass or training with random Hadamard
   transform (RHT) and stochastic rounding disabled.

Default recipe by architecture
-------------------------------

When no recipe is passed explicitly, Transformer Engine selects a default based on
the device:

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

..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Quantization
============

This section describes the internal design of the quantization system. For user-facing
documentation on supported formats, recipes, and usage examples, see
:doc:`/features/low_precision_training/index`.

The developer documentation here focuses on the internal implementation:

- The Quantizer / QuantizedTensorStorage / QuantizedTensor class hierarchy
- How scaling recipes map to quantizer instances
- The rowwise vs. columnwise layout design for GEMM operands
- How to add a new quantization type

.. toctree::
   :maxdepth: 1

   class_hierarchy
   scaling_recipes
   rowwise_columnwise
   adding_new_type

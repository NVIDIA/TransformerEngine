..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Quantization
============

Transformer Engine supports multiple low-precision formats (FP8 E4M3/E5M2, MXFP8,
block-scaled FP8, NVFP4) through a unified quantization architecture. This section
describes the internal design of that system.

The quantization system has three main axes:

- **What** to quantize (the Quantizer/Storage/Tensor class hierarchy)
- **How** to scale (the scaling recipe and global FP8 state)
- **Where** to store (rowwise vs. columnwise layouts for GEMM)

.. toctree::
   :maxdepth: 1

   class_hierarchy
   scaling_recipes
   rowwise_columnwise
   adding_new_type

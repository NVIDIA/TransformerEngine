..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

MXFP8
=====


Hardware support for the MXFP8 (mixed FP8 format) first appeared in the Blackwell GPU.
This recipe is an improved version of FP8 blockwise scaling. With the native hardware features introduced in SM 10.0,
it is possible to efficiently use one scaling factor per 32 consecutive values.



Data Format
-----------

MXFP8 uses the same blockwise scaling principle as FP8 Blockwise Scaling, but with hardware-optimized parameters
that enable more efficient execution on Blackwell GPUs.

The representation of tensor ``A`` in MXFP8 precision is given by:

.. code-block:: python
    
    A[i] = A_fp8[i] * s[block(i)]

where

* ``A_fp8`` is an FP8 tensor in E4M3 format,
* ``s`` is an array of E8M0 scaling factors,
* ``block(i)`` determines which block index ``i`` belongs to.

**FP8 tensor**

MXFP8 uses the E4M3 FP8 format exclusively for both forward and backward passes.
The finer-grained scaling (one factor per 32 values instead of 128) provides sufficient dynamic range
without requiring the E5M2 format.


**Block size**

Block size is **32** - significantly smaller than the 128 used in FP8 Blockwise Scaling.
This provides more granular control over quantization precision.

Blocks are one-dimensional, containing 32 consecutive values. No 2D scaling is performed.

For optimal performance, tensor dimensions should be divisible by 32.


**Scaling factors**

Scaling factors use the **E8M0 format** (8 exponent bits, 0 mantissa bits), which inherently represents
powers of 2. This is different from FP8 Blockwise Scaling, which uses 32-bit floating point numbers
that are optionally constrained to powers of 2.

The scaling factor for each block is computed as follows:

1. Find the maximum absolute value (``amax``) across all 32 elements in the block
2. Compute the E8M0 exponent using the formula: ``exponent = float_to_e8m0(amax / max_norm)``
   where ``max_norm = 448`` (the maximum representable value in E4M3 format)
   
   - The ``float_to_e8m0`` conversion extracts the 8-bit exponent from the float32 representations
     and rounds toward positive infinity (round-up) when mantissa bits are non-zero, rather than truncating
   
3. The resulting scaling factor is ``s = 2^exponent``

This ensures that the largest value in each block, when divided by the scaling factor and
converted to E4M3, will fit within the FP8 range without saturation.


.. raw:: html
   :file: img/fp8_1d_scaling.svg

*Figure 1. MXFP8 uses one E8M0 scaling factor per 32 consecutive elements, providing fine-grained
quantization control with compact scaling factor representation.*


Swizzling scaling factors
-------------------------

MXFP8 GEMMs require scaling factors with a very specific data layout — see the `cuBLAS documentation <https://docs.nvidia.com/cuda/cublas/index.html#block-scaling-factors-layout>`__.
The process of converting standard scaling factors to the required layout is called *swizzling* in the context of Transformer Engine.
Swizzled scaling factors allow more efficient usage of hardware.
Transformer Engine performs swizzling before each of the GEMM operations, after optional communication, since 
the swizzled scaling factors cannot be communicated.

.. raw:: html
   :file: img/mxfp8_swizzle_both_tensors.svg

*Figure 2. MXFP8 swizzling process: standard scaling factors are rearranged into the hardware-required layout.*


Let's now look into the MXFP8 swizzling process in more detail.
Tensor Cores on Blackwell multiply two blocks of elements of size ``128x128``.
Each of these blocks has corresponding block of ``128x4`` bytes of *E8M0* scaling factors.
Note that most blocks of scaling factors are not contiguous - Blackwell hardware cannot handle this. 
Thus swizzling rearranges the elements to make each block contiguous and lay them one after another. This process is illustrated in the following figure:

.. raw:: html
   :file: img/mxfp8_tensor_scaling_layout.svg

*Figure 3. MXFP8 tensor and scaling factor layout: the tensor (left) is tiled into smaller blocks,
while the corresponding scaling factors are arranged as short and tall rectangles.*


Bytes inside each block need to be permuted to satisfy the hardware requirements.
Note that each block contains 128 x 4 bytes of scaling factors,
we will number the consecutive quadruples of bytes as 0, 1, 2, ..., 127. Then after the permutation 
these quadruples will be in order:

.. code-block:: none

   0, 32, 64, 96, 1, 33, 65, 97, ..., 0 + k, 32 + k, 64 + k, 96 + k, ..., 31, 63, 95, 127


.. raw:: html
   :file: img/mxfp8_scale_linearize_and_swizzle.svg

*Figure 4. Linearization and swizzling of scaling factors: 2D grid of scaling factors (K blocks per row)
is first flattened into a contiguous 1D array (top), then reordered from sequential order to interleaved
layout required by hardware (bottom).*

Handling transposes
-------------------

Blackwell architecture supports multiple FP8 GEMM layouts (TN, NT, NN), so explicit transposition of tensors is not required.
However, different scaling layouts are needed for using tensor and tensor transpose - 
first one uses 1x32 blocks, while the second one uses 32x1 blocks.
It means that two different quantized tensors need to be computed - one for rowwise usage and one for columnwise usage.
Moreover, they are numerically different! 

Transposing a quantized tensor is not supported, due to loss of precision. MXFP8 tensor, rowwise and/or columnwise
can be only obtained from higher precision data.

.. raw:: html
   :file: img/mxfp8_row_col.svg

*Figure 5. MXFP8 rowwise vs columnwise quantization layout.*


Distributed training
--------------------

**Scale synchronization**

The blockwise scaled tensor does not need any scale synchronization among the nodes. 
This is because each scaling factor is local to the 32 elements,
not like in FP8 Current/Delayed Scaling where the scale is global for the whole tensor, 
which may be sharded.

**Quantized all-gather**

Gather of columnwise tensor is supported - since in fact rowwise tensor layout is the same as columnwise tensor layout.


Examples
--------

Here's how to use MXFP8 recipe in PyTorch and JAX:

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: pytorch_mxfp8_example.py
         :language: python
         :start-after: # START_MXFP8_EXAMPLE
         :end-before: # END_MXFP8_EXAMPLE

   .. tab:: JAX

      .. literalinclude:: jax_mxfp8_example.py
         :language: python
         :start-after: # START_MXFP8_EXAMPLE
         :end-before: # END_MXFP8_EXAMPLE


Supported devices
-----------------

Blackwell and later (SM 10.0+)
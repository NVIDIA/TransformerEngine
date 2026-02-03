..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

MXFP8
=====


MXFP8 (Microscaling FP8) is an enhanced FP8 blockwise scaling recipe that leverages native hardware
acceleration on Blackwell GPUs (SM 10.0+). By using one scaling factor per 32 consecutive values
(rather than 128), MXFP8 delivers finer-grained quantization with improved numerical precision.



Data Format
-----------

The representation of an FP8 tensor element ``x`` in MXFP8 precision is given by:

.. code-block:: python

    x = x_fp8 * s_block

where

* ``x_fp8`` is the FP8 value in E4M3 format,
* ``s_block`` is a local **E8M0** scaling factor shared by a block of 32 elements.
  E8M0 is an 8-bit format with 8 exponent bits and 0 mantissa bits, representing only powers of 2.


**FP8 format**

Like FP8 Blockwise Scaling, E4M3 is used by default for both forward and backward passes.
The finer-grained scaling provides sufficient dynamic range without requiring the E5M2 format.
The ``fp8_format`` parameter also supports ``HYBRID`` mode (E4M3 for forward, E5M2 for backward).
Pure E5M2 training is not supported.


**Block size**

Block size is 32.
Blocks are one-dimensional, containing 32 consecutive values. No 2D scaling is performed.

There are some assumptions on the dimensions of the tensor:

* the tensor must have at least 2 dimensions,
* the last dimension must be divisible by 32,
* the product of all dimensions except the last must be divisible by 32.


**Scaling factors**

Scaling factors are stored as E8M0 (8 exponent bits, 0 mantissa bits), which inherently represents
powers of 2. This differs from FP8 Blockwise Scaling, which uses 32-bit floating point numbers
optionally constrained to powers of 2. Note that FP32 also has 8 exponent bits, so the representable
ranges are the same when the power-of-2 constraint is enabled.

Each block's scaling factor is computed through the following steps:

1. Find the maximum absolute value (``amax_block``) across all 32 elements in the block.
2. Compute the E8M0 biased exponent: ``e = float_to_e8m0(amax_block / max_fp8)``, where ``max_fp8 = 448``
   (the maximum representable value in E4M3 format).
   
   Since E8M0 and FP32 share the same exponent bias (127), ``float_to_e8m0`` simply extracts
   the 8-bit exponent from the FP32 representation, rounding up if the mantissa is non-zero.
   
3. The scaling factor is ``s_block = 2^(e - 127)``.

This ensures that the largest value in each block fits within the FP8 representable range without overflow.


.. raw:: html
   :file: img/fp8_1d_scaling.svg

*Figure 1. MXFP8 uses one E8M0 scaling factor per 32 consecutive elements, providing fine-grained
quantization and compact scaling factor representation.*


Handling transposes
-------------------

Blackwell architecture supports multiple FP8 GEMM layouts (TN, NT, NN), so columnwise usage
does not require explicit transposition. However, rowwise and columnwise quantizations are different:

- *Rowwise* - 1 scaling factor per 32 consecutive elements along a row (1×32 blocks).
- *Columnwise* - 1 scaling factor per 32 consecutive elements along a column (32×1 blocks).

Since the scaling factor blocks have different orientations, rowwise and columnwise MXFP8 tensors
are numerically different — one cannot derive one from the other. Both must be quantized
independently from the full-precision data.

.. raw:: html 
   :file: img/mxfp8_row_col.svg

*Figure 2. MXFP8 rowwise vs columnwise quantization layout.*


Distributed training
--------------------

**Scale synchronization**

The blockwise scaled tensor does not need any scale synchronization among the nodes.
This is because each scaling factor is local to its 32-element block,
unlike :doc:`FP8 Current <../fp8_current_scaling/fp8_current_scaling>`/:doc:`Delayed Scaling <../fp8_delayed_scaling/fp8_delayed_scaling>` where a single global scale applies to the entire tensor, even when sharded.

**Quantized all-gather**

MXFP8 all-gather is supported.


Examples
--------

Here's how to use MXFP8 recipe in PyTorch and JAX:

.. tabs::

   .. tab:: PyTorch

      .. raw:: html

         <div style="background: #f0f4f8; border-left: 3px solid #5c7cfa; padding: 6px 12px; font-size: 13px; color: #495057; margin-bottom: 0; border-radius: 4px 4px 0 0;">
            Requires SM100 (Blackwell) or later
         </div>

      .. literalinclude:: pytorch_mxfp8_example.py
         :language: python
         :start-after: # START_MXFP8_EXAMPLE
         :end-before: # END_MXFP8_EXAMPLE

   .. tab:: JAX

      .. raw:: html

         <div style="background: #f0f4f8; border-left: 3px solid #5c7cfa; padding: 6px 12px; font-size: 13px; color: #495057; margin-bottom: 0; border-radius: 4px 4px 0 0;">
            Requires SM100 (Blackwell) or later
         </div>

      .. literalinclude:: jax_mxfp8_example.py
         :language: python
         :start-after: # START_MXFP8_EXAMPLE
         :end-before: # END_MXFP8_EXAMPLE


Supported devices
-----------------

SM 10.0, SM 10.3


----

Developer Notes
---------------

This section contains implementation details that may be useful for developers
but are not required for using MXFP8 in practice.

Swizzling scaling factors
^^^^^^^^^^^^^^^^^^^^^^^^^

Like :doc:`FP8 Blockwise Scaling <../fp8_blockwise_scaling/fp8_blockwise_scaling>`, MXFP8 uses different data layouts for communication and computation.
MXFP8 GEMMs require scaling factors in a specific hardware layout
(see `cuBLAS documentation <https://docs.nvidia.com/cuda/cublas/index.html#block-scaling-factors-layout>`__).
The conversion to this GEMM-ready layout is called *swizzling*. When no communication is needed,
swizzling can be fused with quantization. When communication is required, swizzled scaling factors
cannot be communicated across devices, so Transformer Engine performs swizzling after communication,
just before each GEMM operation.

.. raw:: html
   :file: img/mxfp8_swizzle_both_tensors.svg

*Figure 3. MXFP8 swizzling process: standard scaling factors are rearranged into the hardware-required layout.*


Blackwell Tensor Cores compute matrix multiplications using ``128x128`` tiles.
Scaling factors are stored in row-major order, but to process a tile, we need a ``128x4`` vertical
slice of scaling factors. In row-major storage, these vertical slices are scattered in memory
with gaps between each row. The hardware requires them to be stored contiguously.

.. raw:: html
   :file: img/mxfp8_tensor_scaling_layout.svg

*Figure 4. FP8 tensor (left) is divided into 128x128 tiles. Each tile requires a 128x4 block of scaling factors (right). These vertical blocks are not contiguous in memory.*

Swizzling transforms the layout to meet hardware requirements by:

1.  **Linearizing** the ``128x4`` blocks so they are stored contiguously one after another.
2.  **Permuting** the 4-byte elements within each block.

Specifically, if we index the 128 4-byte elements in a scaling factor block as :math:`0, 1, \dots, 127`, the hardware expects them in the following interleaved order:

.. code-block:: text

   0, 32, 64, 96, 1, 33, 65, 97, ..., k, 32 + k, 64 + k, 96 + k, ..., 31, 63, 95, 127


.. raw:: html
   :file: img/mxfp8_scale_linearize_and_swizzle.svg

*Figure 5. Linearization and swizzling of scaling factors. The 2D grid of scaling factors is first flattened into a contiguous sequence of blocks (top), then the rows within each block are interleaved to match the hardware access pattern (bottom).*

For columnwise scaling factors, the process is analogous but with ``4x128`` horizontal blocks instead of ``128x4`` vertical blocks.

All-gather of columnwise tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All-gather of columnwise tensors is supported and necessary because:

- columnwise quantized tensors cannot be computed from rowwise quantized ones,
- gathering high-precision tensors is avoided in most cases for performance reasons.
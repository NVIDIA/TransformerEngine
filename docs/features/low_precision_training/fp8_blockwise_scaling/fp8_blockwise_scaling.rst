..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

FP8 Blockwise Scaling
===================================

.. warning::

   ``Float8BlockScaling`` is **currently not supported** in JAX.

FP8 Blockwise Scaling recipe is inspired by the quantization scheme used to train the `DeepSeek-v3 model <https://arxiv.org/abs/2412.19437>`__ –
the first open-source large-scale LLM trained entirely in FP8 precision.
Unlike the previous recipes, it assigns a dedicated scaling factor to each block of elements.


Data Format
--------------------------

The representation of an FP8 tensor element ``x`` in blockwise precision is given by:

.. code-block:: python

    x = x_fp8 * s_block

where

* ``x_fp8`` is the FP8 value (E4M3 or E5M2),
* ``s_block`` is a local **FP32** scaling factor shared by a block of elements.


.. raw:: html
   :file: img/combined_scaling.svg

*Figure 1. Top: Comparison of standard FP8 scaling (left) using a single scaling factor per tensor versus 
FP8 blockwise scaling in 1 dimension (right) using multiple scaling factors, one per block of 128 elements.
Bottom: FP8 blockwise scaling in 2 dimensions where each 128×128 block in the data tensor has a corresponding
scaling factor.*

**FP8 format**

Unlike FP8 Current/Delayed Scaling, E4M3 is used by default for both forward and backward passes.
Tensor-scaled recipes used E5M2 for gradients due to its higher dynamic range,
but with multiple scaling factors per tensor the dynamic range requirement is lowered, so E4M3 is usually sufficient.
The ``fp8_format`` parameter also supports ``HYBRID`` mode (E4M3 for forward, E5M2 for backward).
Pure E5M2 training is not supported.


**Block size**

Block size is 128.
Blocks can be:

* one dimensional – containing 128 consecutive values,
* two dimensional – containing tiles of 128×128 values.

By default:

* activations use 1D scaling (``x_block_scaling_dim=1``),
* weights use 2D scaling (``w_block_scaling_dim=2``),
* gradients use 1D scaling (``grad_block_scaling_dim=1``).

These can be changed in the recipe, but 2D × 2D GEMMs are not supported 
– at most one operand can use 2D scaling.

One-dimensional scaling is more granular, but 2D scaling offers two advantages:

* *Performance*: On Hopper, block-scaled GEMMs are software-emulated. GEMMs with mixed
  1D/2D scaled tensors have lower overhead than pure 1D scaled GEMMs.
* *Numerical stability*: 2D scaling behaves better when transposed (details in the next section).

There are some assumptions on the dimensions of the tensor (for both 1D and 2D scaling):

* the tensor must have at least 2 dimensions,
* the last dimension must be divisible by 128,
* the product of all dimensions except the last must be divisible by 128.

**Scaling factors**

Scaling factors are stored as 32-bit floating point numbers.
By default, they are constrained to powers of 2 (utilizing the 8 exponent bits of FP32).
On Hopper, this constraint can be relaxed by setting the environment variable ``NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1``.
On Blackwell, only powers of 2 are supported.

Each block's scaling factor is computed through the following steps:

1. Find the maximum absolute value (``amax_block``) across all elements in the block
   (128 consecutive values for 1D blocks, or 128×128 values for 2D blocks).
2. Calculate ``s_block = max_fp8 / amax_block``, where ``max_fp8`` is
   the maximum representable value in the FP8 format (448 for E4M3, 57344 for E5M2).
3. If the power-of-2 constraint is enabled, round down to the nearest power of 2
   by zeroing out the mantissa bits, retaining only the sign and exponent.
4. Multiply each element in the block by ``s_block`` before converting to FP8.

This approach ensures that the largest value in each block fits within the FP8 representable range without overflow.


Handling transposes
------------------------

On Hopper, columnwise tensor access requires data to be transposed in memory.
For 1D scaling, the block direction must align with the access pattern:

* *Rowwise access*: 1 scaling factor per 128 consecutive elements in a row.
* *Columnwise access*: 1 scaling factor per 128 consecutive elements in a row of the transposed tensor,
  corresponding to 128 consecutive elements in a column of the original tensor.

For 2D scaling, each 128×128 tile has one scaling factor regardless of access direction.

This is illustrated below:

.. raw:: html
   :file: img/transpose_handling.svg

*Figure 2. Quantization directions for original and transposed tensors.*

Note that for 1D scaling, the rowwise and columnwise quantized tensors may be numerically different,
so the gradient computation may be affected. This issue is not present for 2D scaling.


Activations and weights use the rowwise version in the forward pass and the columnwise version in the backward pass.
Experiments have shown that 2D scaling for weights is more helpful for numerical stability than for activations,
so by default 1D scaling is used for activations – as it is more granular – and 2D scaling is used for weights.


Unlike FP8 Current/Delayed Scaling, transposing a 1D quantized tensor is not supported.
Rowwise and columnwise blocks cover different sets of elements, so their scaling factors differ.
Both versions must be quantized separately from the high-precision source.

For 2D scaling, columnwise data can be created from rowwise data by transposing 
both the quantized data and the scaling factors. Each 128×128 block covers the same 
elements regardless of access direction, so the scaling factors remain valid.


Distributed training
-----------------------

**Scale synchronization**

The blockwise scaled tensor does not need any scale synchronization among the nodes.
This is because each scaling factor is local to its 128 or 128×128 element block,
unlike FP8 Current/Delayed Scaling where a single global scale applies to the entire tensor, even when sharded.

**Quantized all-gather**

FP8 Blockwise Scaling all-gather is supported.


Examples
--------

Here's how to use the FP8 Blockwise Scaling recipe in PyTorch and JAX:

.. note::

   Requires SM90 (Hopper) or later.

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: pytorch_blockwise_scaling_example.py
         :language: python
         :start-after: # START_BLOCKWISE_SCALING_EXAMPLE
         :end-before: # END_BLOCKWISE_SCALING_EXAMPLE

   .. tab:: JAX

      ``Float8BlockScaling`` is **not currently supported** in JAX.

Supported devices
-----------------

Hopper (SM 9.0)

Blackwell and later (SM >= 10.0) – the recipe is emulated with MXFP8. Note that MXFP8 is the preferred recipe on Blackwell. 
                                   Only scaling factors that are powers of 2 are supported.


----

Developer Notes
---------------

This section contains implementation details that may be useful for developers
but are not required for using FP8 Blockwise Scaling in practice.

Swizzle of scaling factors
^^^^^^^^^^^^^^^^^^^^^^^^^^

FP8 Blockwise Scaling supports all-gather of both rowwise and columnwise tensors.
To support that, it implements different data layouts for communication (all-gather)
and computation (GEMM). We refer to the conversion between these formats as *swizzling*.

A tensor of shape ``[A, B]`` can exist in two formats:

**Compact format** (used for all-gather):

The all-gather primitive only supports gathering non-transposed shards into a non-transposed full tensor,
so all tensor components in this layout are stored without transposition.
Moreover, all component tensors are stored without padding.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Component
     - Shape
   * - rowwise data
     - ``[A, B]``
   * - columnwise data
     - ``[A, B]``
   * - rowwise scales
     - ``[A, B/128]``
   * - columnwise scales
     - ``[A/128, B]``

**GEMM-ready format** (used for computation):

Tensors are transposed and padded as required by the GEMM kernel.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Component
     - Shape
   * - rowwise data
     - ``[A, B]``
   * - columnwise data
     - ``[B, A]`` (transposed)
   * - rowwise scales
     - ``[B/128, pad4(A)]`` (transposed, padded)
   * - columnwise scales
     - ``[A/128, pad4(B)]`` (padded)

Swizzling converts from compact to GEMM-ready format. This can be fused with quantization 
when no all-gather is needed, or performed separately after all-gather.

.. raw:: html
   :file: img/blockwise_swizzle_flow.svg

*Figure 3. FP8 Blockwise Scaling swizzle paths. Top: With all-gather communication – quantization produces 
compact format, then swizzle is performed separately after communication. Bottom: Without all-gather – 
quantize and swizzle are fused into a single operation, directly producing GEMM-ready format.*

All-gather of columnwise tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All-gather of columnwise tensors is supported and necessary because:

- columnwise quantized tensors cannot be computed from rowwise quantized ones,
- gathering high-precision tensors is avoided in most cases for performance reasons.

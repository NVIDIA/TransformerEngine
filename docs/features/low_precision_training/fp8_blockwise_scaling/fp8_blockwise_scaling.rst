..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

FP8 Blockwise Scaling
===================================

FP8 Blockwise Scaling is inspired by the quantization scheme used to train the `DeepSeek-v3 model <https://arxiv.org/abs/2412.19437>`__ –
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
scaling factor, providing fine-grained spatial control over quantization precision.*

**FP8 format**

Both E4M3 and E5M2 formats are supported, but unlike FP8 Current/Delayed Scaling,
E4M3 is used by default for both forward and backward passes.
Previous recipes used E5M2 for gradients due to its higher dynamic range,
but with multiple scaling factors per tensor, E4M3 is sufficient.


**Block size**

Block size is 128. 
Blocks can be:

* one dimensional – containing 128 consecutive values,
* two dimensional – containing tiles of 128x128 values.

More details when 1d and 2d scaling are used are provided later.

There are some assumptions on the dimensions of the tensor:

**Scaling factors**

Scaling factors are 32-bit floating point numbers.
By default they are constrained to powers of 2.
Note that 32-bit floats consist of 8 exponent bits.
This constraint can be relaxed via ``NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1`` environment variable.

The scaling factor for each block is computed as follows:

1. Find the maximum absolute value (``amax``) across all elements in the block
   (either 128 consecutive values for 1D blocks, or 128×128 values for 2D blocks)
2. Compute the initial scale using the formula: ``scale = max_fp8 / amax``
   where ``max_fp8`` depends on the FP8 format: 448 for E4M3, or 57344 for E5M2
3. If power-of-2 constraint is enabled (default), zero out the mantissa bits,
   keeping only the sign and exponent bits to ensure ``scale`` is a power of 2
4. Each element in the block is then multiplied by this scale factor before quantization to FP8

This ensures that the largest value in each block, when multiplied by the scale and
converted to FP8, will fit within the FP8 range without saturation.






Handling transposes
------------------------

Blockwise scaling is intended to be used on Hopper device which does need transposed tensors in backward pass.
Moreover, the Tensor Cores require that quantization direction for transpose needs to be 
different from the quantization direction for the original tensor.
This is illustrated in the picture below:

.. raw:: html
   :file: img/transpose_handling.svg



Note that for 1D scaling the rowwise and columnwise quantized tensors may be numerically different.
So the gradient computation may be affected. This issue is not present for 2D scaling.

By default TE:

* uses 1d scaling for activations,
* uses 2d scaling for weights,
* uses 1d scaling for gradients.

Activations, like weights, use the non-transposed version in the forward pass and the transposed version in the backward pass.
Experiments have shown that 2D scaling for weights is more helpful for numerical stability than for activations,
so by default 1D scaling is used for activations – as it is more granular – and 2D scaling is used for weights.


Note that – unlike in FP8 Current/Delayed Scaling – transposing 1D quantized tensor is not supported 
– since rowwise tensor has 
1 scaling factor per 128 rowwise consecutive values and columnwise tensor has 
1 scaling factor per 128 columnwise consecutive values, which are not the same.
Computing the columnwise quantized tensor from rowwise quantized one will lead to precision loss.
Thus quantized tensor can only be obtained from higher precision data.

Swizzle of scaling factors
--------------------------

Here we introduce a new concept of swizzling of data.
Sometimes the data format used for the communication is different 
from the one required by the GEMM. It was not the case for the previous recipes,
but for FP8 Blockwise Scaling it is.

For FP8 Blockwise Scaling Tensor can have 2 formats:

- compact format – used for all-gather:
  
  - rowwise data: not transposed, shape: ``[A, B]``
  - columnwise data: not transposed, shape: ``[A, B]``
  - rowwise scaling factors: not transposed, not padded, shape: ``[A, B/128]``
  - columnwise scaling factors: not transposed, not padded, shape: ``[A/128, B]``

- gemm ready format – converted from COMPACT format after all-gather and before GEMMs:
  
  - rowwise data: not transposed, shape: ``[A, B]``
  - columnwise data: transposed, shape: ``[B, A]``
  - rowwise scaling factors: transposed, padded to the multiple of 4 along the last dimension, shape: ``[B/128, pad_to_4(A)]``
  - columnwise scaling factors: not transposed, padded to the multiple of 4 along the last dimension, shape: ``[A/128, pad_to_4(B)]``

Note that data in compact format is easy to gather. Every tensor is non-transposed and no padding is needed.
This is not the case for gemm ready format.

By the **swizzling** we mean the process of converting the data from compact format to gemm ready format.
This can be fused into the quantization if no all-gather is performed, but it can also be done separately.

.. raw:: html
   :file: img/blockwise_swizzle_flow.svg

*Figure 2. FP8 Blockwise Scaling swizzle paths. Top: With all-gather communication – quantization produces 
compact format, then swizzle is performed separately after communication. Bottom: Without all-gather – 
quantize and swizzle are fused into a single operation, directly producing GEMM ready format.*



Distributed training
-----------------------

**Scale synchronization**

The blockwise scaled tensor does not need any scale synchronization among the nodes. 
This is because each scaling factor is local to the 128 or 128x128 elements,
not like in FP8 Current/Delayed Scaling where the scale is global for the whole tensor, 
which may be sharded.

**Quantized all-gather**

Gather of columnwise tensor is supported and is used since:

- as mentioned earlier, it is not supported to compute columnwise quantized tensor from rowwise quantized one,
- high precision tensor is not gathered in most cases due to performance reasons,


Examples
--------

Here's how to use FP8 Blockwise Scaling recipe in PyTorch and JAX:

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: pytorch_blockwise_scaling_example.py
         :language: python
         :start-after: # START_BLOCKWISE_SCALING_EXAMPLE
         :end-before: # END_BLOCKWISE_SCALING_EXAMPLE

   .. tab:: JAX

      .. literalinclude:: jax_blockwise_scaling_example.py
         :language: python
         :start-after: # START_BLOCKWISE_SCALING_EXAMPLE
         :end-before: # END_BLOCKWISE_SCALING_EXAMPLE

Supported devices
-----------------

Hopper (SM 9.0)

Blackwell (SM 10.0) – emulates this recipe with MXFP8
..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

NVFP4
===================================

NVFP4 is the first 4-bit recipe introduced in Transformer Engine -
please refer to the `NVFP4 paper <https://arxiv.org/abs/2509.25149>`__ for more details.
It is a more complex recipe than the previous ones - apart from the new data format,
it introduces multiple features which help training stability.

Data Format
----------------------

The NVFP4 datatype consists of 1 sign bit, 2 exponent bits, and 1 mantissa bit (E2M1).
It can represent values of magnitude up to +/- 6.
NVFP4 uses a hierarchical block scaling approach where multiple scaling factors are combined to recover the high precision value.

.. raw:: html
   :file: img/nvfp4_vs_fp8.svg

*Figure 1. Bit layout comparison between standard FP8 formats (E4M3 and E5M2) and NVFP4 (E2M1).*


The representation of an NVFP4 tensor element ``x`` is given by:

.. code-block:: python

    x = x_e2m1 * s_block * s_global

where

* ``x_e2m1`` is the 4-bit value,
* ``s_block`` is a local **FP8 E4M3** scaling factor shared by a block of 16 consecutive elements,
* ``s_global`` is a global **FP32** scaling factor applied to the entire tensor.

**Scaling Factor Computation**

The scaling factors are computed as follows:

1. Global scaling factor (``s_global``):

.. code-block:: python

    s_global = (fp8_max * fp4_max) / global_amax
    # where:
    # - global_amax: maximum absolute value across the entire tensor
    # - fp8_max: maximum representable value in FP8 E4M3 (448.0)
    # - fp4_max: maximum representable value in NVFP4 E2M1 (6.0)

2. Block scaling factor (``s_block``):

.. code-block:: python

    s_block = (block_amax / fp4_max) * s_global
    # where:
    # - block_amax: maximum absolute value within the block
    # - fp4_max: maximum representable value in NVFP4 E2M1 (6.0)
    # - s_block is stored in FP8 E4M3 format


.. raw:: html
   :file: img/nvfp4_hierarchical_scaling.svg

*Figure 2. NVFP4 hierarchical scaling structure showing the combination of block-level and global scaling factors.*

This hierarchical structure uses fine-grained block scaling 
to adapt to local magnitude variations and global scaling 
to handle the overall dynamic range.

**2D weight scaling**

NVFP4 can be:

* 1 dimensional - each block of 16 consecutive elements shares a scaling factor,
* 2 dimensional - each block of 16x16 elements shares a scaling factor.

By default, NVFP4 uses 2D scaling for weights and 1D scaling for activations and gradients.
This can be changed by setting the ``use_2d_quantization`` flag to ``True`` or ``False``.
The motivation for using 2D scaling for weights is the same as for FP8 blockwise scaling
- ensure that rowwise and columnwise quantized tensors are numerically equivalent.


Stochastic Rounding
-------------------

Stochastic rounding is applied when casting scaled values to NVFP4 format. Instead of deterministic rounding 
(always rounding to nearest even value), each scaled value is probabilistically rounded to one of the two 
nearest representable NVFP4 values. The rounding probabilities are inversely proportional to 
the distance to each representable value, which ensures that the expected value of the quantized 
tensor equals the original value, eliminating systematic quantization bias during training.
Stochastic rounding is hardware-accelerated using native GPU instructions introduced with the 
Blackwell architecture.

.. raw:: html
   :file: img/stochastic_rounding.svg

*Figure 3. Stochastic rounding illustration. Given a value* ``x`` *to be quantized, and the two nearest 
representable NVFP4 values* ``v1`` *(lower) and* ``v2`` *(higher), deterministic rounding always 
rounds to the nearest value, while stochastic rounding probabilistically rounds to either value. 
The rounding probabilities are inversely proportional to the distance: if* ``x`` *is 40% of the way from* 
``v1`` *to* ``v2``, *there is a 60% chance of rounding to* ``v1`` *and a 40% chance of rounding to* ``v2``.

By default, stochastic rounding is enabled only for gradients. It can be disabled by setting 
``disable_stochastic_rounding=True`` in the recipe configuration.


Random Hadamard Transform
--------------------------

Random Hadamard Transform (RHT) is the technique used to smooth outliers 
in the tensor distributions and make them easier to represent accurately in NVFP4.
It is used **only for wgrad GEMM**, which - according to the paper mentioned at the top - is particularly sensitive.

Let's see how RHT works for a GEMM with input matrices ``A`` and ``B``.

1. *Definition of matrix* ``H``.

   Let's consider matrix ``H`` of shape ``16 x 16`` that:

   * ``H * H^T = I``, where ``I`` is the identity matrix.
   * ``H`` is a square matrix with entries of ``+1 / sqrt(16) = 0.25`` or ``-1 / sqrt(16) = -0.25``.

   Such matrix exists:

   .. code-block:: python

      H = 0.25 * [
         [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
         [ 1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
         [ 1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1],
         [ 1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1],
         [ 1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1],
         [ 1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1],
         [ 1,  1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1],
         [ 1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1],
         [ 1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
         [ 1, -1,  1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1, -1,  1],
         [ 1,  1, -1, -1,  1,  1, -1, -1, -1, -1,  1,  1, -1, -1,  1,  1],
         [ 1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1],
         [ 1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1],
         [ 1, -1,  1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1],
         [ 1,  1, -1, -1, -1, -1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1],
         [ 1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1, -1,  1],
      ]

2. *Definition of matrix* ``S``.

   Define ``S`` as a ``16x16`` matrix with the following properties:
   * ``S`` is a diagonal matrix with entries of ``+1`` or ``-1``.
   * ``S`` is shared across all modules and is changed randomly for each iteration.


3. *Property of* ``HS`` *matrix*.

   Note that if vectors ``u`` and ``v`` are such that ``u * v^T = c``, then 

   .. code-block:: python

      (u * HS) * (v * HS)^T =  u * H * S * S^T * H^T * v^T = u * v^T = c

   thus if RHT is applied to both ``u`` and ``v``, we will get the same result of the dot product.

   We can generalize this property for tensors `A` and `B`. If we multiply 
   every block of 16 elements in `A` by ``HS`` and every block of 16 elements in `B` by ``HS``,
   then result of the ``A * B^T`` will not change.

Random Hadamard Transform is the name of the operation described above - 
applying linear transformation ``HS`` to blocks of 16 elements in both tensors before GEMM.


.. raw:: html
   :file: img/rht.svg

*Figure 4. WGRAD GEMM pipeline comparison: without RHT (left) and with RHT applied (right).*


Distributed training
--------------------

Block scaling factors (``s_block``) do not require synchronization between nodes.
However, the global scaling factor (``s_global``) requires amax synchronization for gathered tensors.

For tensors which are gathered - input and gradient in sequence parallelism,
amax reduction is performed before quantization.
If before synchronization there was ``amax_1`` on node 1,
``amax_2`` on node 2, etc., after synchronization there will be ``max(amax_1, amax_2, ...)`` on all nodes.
To make quantized all-gather possible,
all nodes must use the same ``s_global``, which is computed from the synchronized global amax.
This is automatically enabled for column-parallel and row-parallel linear layers.

.. raw:: html
   :file: img/nvfp4_all_gather.svg

*Figure 5. Quantization and all-gather flow for NVFP4 showing amax synchronization and hierarchical scaling.*

Examples
--------

Here's how to use NVFP4 recipe in PyTorch and JAX. The examples show how to configure features like 2D weight quantization and Random Hadamard Transform (RHT):

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: pytorch_nvfp4_example.py
         :language: python
         :start-after: # START_NVFP4_EXAMPLE
         :end-before: # END_NVFP4_EXAMPLE

   .. tab:: JAX

      .. literalinclude:: jax_nvfp4_example.py
         :language: python
         :start-after: # START_NVFP4_EXAMPLE
         :end-before: # END_NVFP4_EXAMPLE


Supported devices
-----------------

Blackwell and later (SM 10.0+)

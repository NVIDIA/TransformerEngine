..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

NVFP4
===================================

NVFP4 is the first 4-bit recipe introduced in Transformer Engine –
please refer to the `NVFP4 paper <https://arxiv.org/abs/2509.25149>`__ for more details.
It is a more complex recipe than the previous ones – apart from the new data format,
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

    s_global = global_amax / (fp8_max * fp4_max)
    # where:
    # - global_amax: maximum absolute value across the entire tensor
    # - fp8_max: maximum representable value in FP8 E4M3 (448.0)
    # - fp4_max: maximum representable value in NVFP4 E2M1 (6.0)

2. Block scaling factor (``s_block``):

.. code-block:: python

    s_block = (block_amax / fp4_max) / s_global
    # where:
    # - block_amax: maximum absolute value within the block
    # - fp4_max: maximum representable value in NVFP4 E2M1 (6.0)
    # - s_block is stored in FP8 E4M3 format


.. raw:: html
   :file: img/nvfp4_hierarchical_scaling.svg

*Figure 2. NVFP4 hierarchical scaling structure showing the combination of block-level and global scaling factors.*

This hierarchical structure uses fine-grained block scaling to handle the tensor's dynamic range,
while the FP4 values represent the block-level dynamic range. The global scaling factor
aligns values to the representable range of the E4M3 × E2M1 combination.

**2D weight scaling**

NVFP4 can be:

* 1 dimensional - each block of 16 consecutive elements shares a scaling factor,
* 2 dimensional - each block of 16x16 elements shares a scaling factor.

By default, NVFP4 uses 2D scaling for weights and 1D scaling for activations and gradients.
Set ``disable_2d_quantization=True`` in the recipe configuration to force 1D scaling for weights as well (activations and gradients always use 1D).
The motivation for using 2D scaling for weights is to ensure that rowwise and columnwise 
quantized tensors are numerically equivalent. 
Please refer to the `NVFP4 paper <https://arxiv.org/abs/2509.25149>`__ for more details.


Stochastic Rounding
-------------------

Stochastic rounding is applied when casting scaled values to NVFP4 format. Instead of deterministic rounding 
(always rounding to nearest even value), each scaled value is probabilistically rounded to one of the two 
nearest representable NVFP4 values. The probability of rounding to a given value is inversely proportional to 
the distance to that value, which ensures that the expected value of the quantized 
tensor equals the original value, eliminating systematic quantization bias during training.
Stochastic rounding is hardware-accelerated using native GPU instructions introduced with the 
Blackwell architecture.

.. raw:: html
   :file: img/stochastic_rounding.svg

*Figure 3. Stochastic rounding illustration. Given a value* ``x`` *to be quantized, and the two nearest 
representable NVFP4 values* ``v1`` *(lower) and* ``v2`` *(higher), deterministic rounding always 
rounds to the nearest value, while stochastic rounding probabilistically rounds to either value. 
If* ``x`` *is 40% of the way from* ``v1`` *to* ``v2``, *there is a 60% chance of rounding to* ``v1`` 
*and a 40% chance of rounding to* ``v2``.

Stochastic rounding is enabled only for gradients. It can be disabled by setting 
``disable_stochastic_rounding=True`` in the recipe configuration.


Random Hadamard Transform
--------------------------

Random Hadamard Transform (RHT) applies an orthogonal rotation to the tensor **before quantization**,
smoothing outliers in the tensor distributions and making them easier to represent accurately in NVFP4.
RHT is applied to columnwise quantization of inputs and gradients, which are operands
for the **wgrad GEMM**. This GEMM is particularly sensitive
to quantization errors, hence the additional outlier smoothing.
RHT is supported only for BF16 inputs/gradients.

The transform is defined as:

.. math::

   x' = x H

where :math:`H` is the RHT matrix defined below. The quantization scale factor is computed 
from the rotated tensor :math:`x'`.

**Hadamard matrix**

The :math:`d \times d` Hadamard matrix has elements :math:`\pm 1` and satisfies :math:`H_d H_d^T = d I`.
When normalized by :math:`1/\sqrt{d}`, the matrix becomes orthogonal and can be applied 
to both operands of a matrix multiplication:

.. math::

   C = (AH)(H^T B) = AB

where the transforms cancel within the dot-product since :math:`H H^T = I`.

**Sign matrix**

In the RHT implementation, a :math:`d`-dimensional diagonal sign matrix :math:`S_d` is applied 
together with the Hadamard matrix:

.. math::

   H = \frac{1}{\sqrt{d}} S_d H_d

where diagonal entries of :math:`S_d` are :math:`\{-1, 1\}` and flip the signs of different rows of :math:`H_d`.
As described in the paper, a single random sign vector is shared across all linear layers throughout training.
In the implementation, this vector is fixed and the RHT matrix is computed once at initialization and cached.

**Tiled implementation**

The Hadamard transform is performed in a tiled approach along the last dimension of the tensor.
For an :math:`m \times k` tensor, the data is reshaped to :math:`(mk/d) \times d` 
and multiplied by the :math:`d \times d` matrix :math:`H`. In this implementation, :math:`d = 16`.


.. raw:: html
   :file: img/rht.svg

*Figure 4. WGRAD GEMM pipeline comparison: without RHT (left) and with RHT applied (right).*

Handling transposes
-------------------

Like :doc:`MXFP8 <../mxfp8/mxfp8>`, NVFP4 requires both rowwise and columnwise quantized tensors
for different GEMM operands. Unlike MXFP8 which supports multiple layouts (TN, NT, NN),
**NVFP4 GEMM only supports the TN layout**.

NVFP4 stores columnwise data and scaling factors in a **transposed layout**:

- **Rowwise**: data ``[A, B]`` with 1×16 horizontal blocks, ``scales`` shape ``[A, B/16]``
- **Columnwise**: data ``[B, A]`` (transposed) with 1×16 horizontal blocks, ``scales`` shape ``[B, A/16]``

Scale tensors are padded for hardware alignment: first dimension to a multiple of 128,
second dimension to a multiple of 4 (e.g. rowwise: ``[roundup(A, 128), roundup(B/16, 4)]``).

.. raw:: html
   :file: img/nvfp4_row_col.svg

*Figure 5. NVFP4 rowwise vs columnwise quantization layout. Unlike MXFP8, columnwise scales are stored transposed.*


Distributed training
--------------------

**Amax reduction**

Block scaling factors (``s_block``) do not require synchronization between nodes,
as each scaling factor is local to its block of 16 elements.
However, the global scaling factor (``s_global``) requires amax synchronization for gathered tensors.
For tensors that are gathered (e.g., input and gradient in sequence parallelism),
amax reduction is performed before quantization.
If before synchronization there was ``amax_1`` on node 1,
``amax_2`` on node 2, etc., after synchronization there will be ``max(amax_1, amax_2, ...)`` on all nodes.

**Quantized all-gather**

NVFP4 all-gather is supported.

.. raw:: html
   :file: img/nvfp4_all_gather.svg

*Figure 6. Quantization and all-gather flow for NVFP4 showing amax synchronization and hierarchical scaling.*

Per-token NVFP4
---------------

The default ``NVFP4BlockScaling`` recipe computes a single per-tensor outer
``amax`` (``s_global``) for each tensor. The **per-token** variant instead
computes a per-row outer ``amax`` (length ``M``) for rowwise data and a per-col
outer ``amax`` (length ``K``) for columnwise data, giving each token/row its own
global scale. This finer outer-scale granularity can improve accuracy, and the
per-token cast feeds a dedicated fused-EVT CUTLASS GEMM that consumes the vector
outer ``amax`` directly (cuBLASLt cannot).

There are two ways to select per-token, both equivalent:

* **Explicit recipe class** ``NVFP4PerTokenBlockScaling`` (recommended for code
  that constructs its own recipe).
* **Environment variable** ``NVTE_NVFP4_PER_TOKEN=1`` on a plain
  ``NVFP4BlockScaling``. This lets frameworks that only ever build a default
  ``NVFP4BlockScaling`` (for example Megatron-Core) opt into per-token purely
  from the launch environment, with no framework-side code change.

.. code-block:: python

   from transformer_engine.common.recipe import NVFP4PerTokenBlockScaling
   import transformer_engine.pytorch as te

   # RHT and SR are OFF by default on the per-token path; opt in as needed.
   recipe = NVFP4PerTokenBlockScaling(per_token_rht=True, per_token_sr=True)
   with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
       out = model(inp)

**Differences from the per-tensor default**

* RHT and stochastic rounding are **off by default** on the per-token path (the
  per-row outer ``amax`` already mitigates the long-tail outliers RHT targets).
  Opt in with ``per_token_rht=True`` / ``per_token_sr=True`` (env vars
  :envvar:`NVTE_NVFP4_PER_TOKEN_RHT` / :envvar:`NVTE_NVFP4_PER_TOKEN_SR`).
* 2D weight quantization is disabled by default. The per-token weight-2D route
  (``per_token_weight_2d=True`` / :envvar:`NVTE_NVFP4_PER_TOKEN_WEIGHT_2D`)
  quantizes the forward weight with the transposition-invariant 2D cast emitted
  in per-token layout, removing the 1D weight-gradient bias.
* ``row_scaled_activation`` and 4over6 are forced off (mutually exclusive with
  the per-token amax layout).

**Norm forward.** The per-token forward path uses the unfused norm+amax
implementation. Its per-row/per-col outer amax is computed by the per-token
cast's first kernel (K1, the amax pass), a per-row ``(M,)`` + per-col ``(K,)``
vector that the fused norm+amax kernel (which only emits a scalar amax) cannot
produce. This is selected automatically for per-token quantizers -- no
``NVTE_NORM_FWD_USE_CUDNN=1`` needed.

**Currently unsupported on the per-token path**: forward/backward output
quantization, and communication/bulk overlap.

Running per-token NVFP4 with Megatron-Core
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Megatron-Core builds a plain ``NVFP4BlockScaling`` for ``--fp4-format e2m1`` and
has no CLI for per-token, so per-token is selected entirely through TE
environment variables. A minimal launch looks like:

.. code-block:: bash

   # Select NVFP4 per-token via TE env vars (read at recipe construction).
   export NVTE_NVFP4_PER_TOKEN=1
   # Optional per-token knobs:
   # export NVTE_NVFP4_PER_TOKEN_RHT=1
   # export NVTE_NVFP4_PER_TOKEN_SR=1
   # export NVTE_NVFP4_PER_TOKEN_WEIGHT_2D=1

   python pretrain_gpt.py \
       --transformer-impl transformer_engine \
       --fp4-format e2m1 \
       --no-gradient-accumulation-fusion \
       ...   # remaining model / data / optimizer args

Notes:

* ``--no-gradient-accumulation-fusion`` is required because the per-token kernel
  does not yet support fused wgrad accumulation.
* To keep the first/last transformer layers in BF16, use Megatron's
  ``--first-last-layers-bf16 --num-layers-at-start-in-bf16 N
  --num-layers-at-end-in-bf16 M`` flags (those layers simply skip the FP4
  autocast; the recipe is unchanged).

Examples
--------

Here's how to use NVFP4 recipe in PyTorch and JAX. The examples show how to configure features like 2D weight quantization and Random Hadamard Transform (RHT):

.. tabs::

   .. tab:: PyTorch

      .. raw:: html

         <div style="background: #f0f4f8; border-left: 3px solid #5c7cfa; padding: 6px 12px; font-size: 13px; color: #495057; margin-bottom: 0; border-radius: 4px 4px 0 0;">
            Requires SM100 (Blackwell) or later
         </div>

      .. literalinclude:: pytorch_nvfp4_example.py
         :language: python
         :start-after: # START_NVFP4_EXAMPLE
         :end-before: # END_NVFP4_EXAMPLE

   .. tab:: JAX

      .. raw:: html

         <div style="background: #f0f4f8; border-left: 3px solid #5c7cfa; padding: 6px 12px; font-size: 13px; color: #495057; margin-bottom: 0; border-radius: 4px 4px 0 0;">
            Requires SM100 (Blackwell) or later
         </div>

      .. literalinclude:: jax_nvfp4_example.py
         :language: python
         :start-after: # START_NVFP4_EXAMPLE
         :end-before: # END_NVFP4_EXAMPLE


Supported devices
-----------------

* **Training**: SM 10.0, SM 10.3
* **Inference**: SM 10.0+


----

Developer Notes
---------------

This section contains implementation details that may be useful for developers
but are not required for using NVFP4 in practice.

Swizzling scaling factors
^^^^^^^^^^^^^^^^^^^^^^^^^

NVFP4 requires swizzling of block scaling factors (``s_block``) before GEMM operations,
similar to :doc:`MXFP8 <../mxfp8/mxfp8>`. Key differences:

- Block size is 16 (vs 32 for MXFP8)
- Both rowwise and columnwise scaling factors are swizzled, but thanks to the transposed
  columnwise layout, a single rowwise swizzle kernel handles both cases.
- Scaling factors are stored as FP8 E4M3 (vs E8M0 for MXFP8)

All-gather of columnwise tensors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All-gather of columnwise tensors is supported. To enable quantized all-gather, 
all nodes must use the same ``s_global``, which is computed from the synchronized global amax.
This is automatically enabled for column-parallel and row-parallel linear layers.

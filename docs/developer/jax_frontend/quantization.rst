..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

JAX Quantization
================

The JAX frontend has its own quantization system that parallels the PyTorch design but
uses JAX idioms (immutable arrays, functional transforms).

Quantizer Hierarchy
-------------------

**Location**: ``transformer_engine/jax/quantize/quantizer.py``

The JAX quantizer hierarchy mirrors PyTorch's:

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - JAX Quantizer
     - PyTorch Equivalent
     - Scaling Mode
   * - ``DelayedScaleQuantizer``
     - ``Float8Quantizer``
     - Delayed tensor
   * - ``CurrentScaleQuantizer``
     - ``Float8CurrentScalingQuantizer``
     - Current tensor
   * - ``BlockScaleQuantizer``
     - ``MXFP8Quantizer``
     - MXFP8 block (JAX does not yet support generic block scaling 1D/2D)
   * - ``NVFP4Quantizer``
     - ``NVFP4Quantizer``
     - NVFP4
   * - ``GroupedQuantizer``
     - (no direct equivalent)
     - Group-wise FP8

ScaledTensor Types
------------------

**Location**: ``transformer_engine/jax/quantize/tensor.py``

Instead of PyTorch's ``QuantizedTensor`` (which is a ``torch.Tensor`` subclass), JAX uses
``ScaledTensor`` types registered as JAX pytrees. There are two core types:

- ``ScaledTensor1x`` — Single-layout quantized tensor (either rowwise or columnwise).
- ``ScaledTensor2x`` — Dual-layout tensor wrapping both a rowwise and columnwise
  ``ScaledTensor1x``.

A ``NoScaleTensor`` type wraps unquantized data with optional amax tracking. For grouped
operations, ``GroupedScaledTensor1x`` extends the single-layout type with support for
ragged grouping (specified via ``first_dims`` and ``last_dims`` arrays that describe
per-group sizes along each dimension), and ``GroupedNoScaleTensor`` provides an
unquantized grouped tensor for use with grouped GEMM.

Since JAX arrays are immutable, these are simple data classes (not JAX array subclasses),
registered as pytree nodes so JAX can trace through them:

.. code-block:: python

   @register_pytree_node_class
   @dataclass
   class ScaledTensor1x:
       data: jax.Array          # Quantized data
       scale_inv: jax.Array     # Scale inverse for dequantization
       scaling_mode: ScalingMode
       dq_dtype: jnp.dtype      # Target dtype for dequantization
       is_colwise: bool         # Whether this is a columnwise layout

   @register_pytree_node_class
   @dataclass
   class ScaledTensor2x:
       rowwise_tensor: ScaledTensor1x
       colwise_tensor: ScaledTensor1x

Key Differences from PyTorch
-----------------------------

1. **No ``__torch_dispatch__``**: JAX doesn't have dispatch machinery. Operations must
   explicitly handle ``ScaledTensor`` inputs.

2. **No autograd integration**: JAX uses ``custom_vjp`` on the primitive level, not on
   the tensor type.

3. **Immutable**: Quantized tensors cannot be modified in-place. Amax history for delayed
   scaling is managed through JAX's stateful mechanisms (Flax variable collections).

4. **XLA-compatible**: ``ScaledTensor`` fields are regular ``jax.Array`` values that XLA
   can trace through and compile.

Usage with Modules
------------------

Quantizers are typically created via the ``QuantizerFactory`` from a recipe:

.. code-block:: python

   from transformer_engine.jax.quantize.helper import get_quantize_config_with_recipe
   from transformer_engine.common.recipe import DelayedScaling

   recipe = DelayedScaling()
   config = get_quantize_config_with_recipe(recipe)

   # Inside a Flax module, quantizer sets are auto-created
   # via TransformerEngineBase.generate_quantizer_set()
   # Metadata (scales, amax_history) stored in Flax variable collections

See Also
--------

- :doc:`/developer/quantization/class_hierarchy` — PyTorch quantization design (same concepts)
- :doc:`/developer/quantization/scaling_recipes` — Recipe system shared between frontends

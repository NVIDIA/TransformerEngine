..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

JAX Module System
=================

The JAX frontend provides Flax ``nn.Module`` wrappers that mirror the PyTorch frontend's
module hierarchy but use JAX idioms (functional transforms, XLA compilation, sharding
annotations).

.. figure:: ./img/jax_vs_pytorch_binding.svg
   :align: center
   :width: 70%

   PyTorch and JAX frontends share the same C++ core via different binding mechanisms.

..
   Diagram description for ``jax_vs_pytorch_binding.svg``:
   Two parallel vertical paths:
   Left path: "PyTorch nn.Module" → "pybind11" → "C++ Core"
   Right path: "JAX Flax nn.Module" → "XLA FFI" → "C++ Core"
   Both paths converge at the "C++ Core" box at the bottom.
   Labels: "torch.Tensor" on left, "jax.Array" on right.

Module Location
---------------

**Location**: ``transformer_engine/jax/flax/module.py``

Flax Modules
------------

The JAX frontend provides these Flax modules. Core modules are in
``transformer_engine/jax/flax/module.py``, while attention and transformer composition
are in ``transformer_engine/jax/flax/transformer.py``.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - JAX Module
     - PyTorch Equivalent
     - Description
   * - ``DenseGeneral``
     - ``Linear``
     - General linear layer with FP8 and flexible contracting dimensions
   * - ``LayerNorm``
     - ``LayerNorm`` / ``RMSNorm``
     - Normalization (supports both ``layernorm`` and ``rmsnorm`` types)
   * - ``LayerNormDenseGeneral``
     - ``LayerNormLinear``
     - Fused LN + Linear
   * - ``LayerNormMLP``
     - ``LayerNormMLP``
     - Fused LN + MLP
   * - ``DotProductAttention``
     - ``DotProductAttention``
     - Attention (fused/unfused auto-selection)
   * - ``MultiHeadAttention``
     - ``MultiheadAttention``
     - Full MHA block with projections
   * - ``TransformerLayer``
     - ``TransformerLayer``
     - Complete Transformer block

All FP8-capable modules inherit from ``TransformerEngineBase``, which provides
``generate_quantizer_set()`` for recipe-based quantizer creation.

Key Differences from PyTorch
-----------------------------

**Functional style**: JAX modules are stateless. Parameters are passed explicitly, not
stored as attributes:

.. code-block:: python

   # PyTorch: stateful
   layer = te.Linear(768, 3072)
   output = layer(input)

   # JAX: functional
   layer = te_jax.DenseGeneral(features=3072)
   params = layer.init(rng, input)
   output = layer.apply(params, input)

**Sharding via annotations**: Instead of ``parallel_mode`` constructor args, JAX modules
use ``MeshResource`` and rely on XLA SPMD for communication:

.. code-block:: python

   layer = te_jax.DenseGeneral(
       features=3072,
       mesh_resource=MeshResource(tp_resource="tp"),
   )

**Quantization context**: Instead of ``autocast()``, JAX uses explicit quantizer
arguments:

.. code-block:: python

   output = layer.apply(
       params, input,
       quantizer=fp8_quantizer,
   )

See Also
--------

- :doc:`/developer/pytorch_frontend/module_hierarchy` — PyTorch module hierarchy for comparison
- :doc:`xla_ffi_primitives` — How JAX modules call into C++ kernels
- :doc:`sharding` — Detailed sharding configuration

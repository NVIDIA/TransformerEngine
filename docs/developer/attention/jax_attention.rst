..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

JAX Attention
=============

The JAX frontend provides attention through functional APIs that wrap the same C++ core
as the PyTorch frontend, but use JAX's XLA FFI mechanism for dispatch.

Entry Points
------------

**Location**: ``transformer_engine/jax/attention.py``

The primary functions are:

- ``fused_attn()`` — Fused attention using cuDNN (equivalent to PyTorch's cuDNN fused
  backend).
- ``fused_attn_fwd()`` / ``fused_attn_bwd()`` — Forward and backward primitives.

These are JAX primitives registered via the XLA FFI mechanism
(see :doc:`/developer/jax_frontend/xla_ffi_primitives`).

Differences from PyTorch
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - PyTorch
     - JAX
   * - API style
     - ``nn.Module`` (``DotProductAttention``)
     - Functional (``fused_attn()``)
   * - Backend selection
     - Automatic with fallback chain
     - Primarily cuDNN fused
   * - Autograd
     - ``torch.autograd.Function``
     - JAX custom VJP via primitives
   * - Context parallel
     - Ring attention with NCCL
     - Integrated with XLA sharding
   * - KV cache
     - ``InferenceParams`` class
     - Managed externally by user

Usage Example
-------------

.. code-block:: python

   from transformer_engine.jax.attention import fused_attn

   output = fused_attn(
       query, key, value,
       bias=attn_bias,
       mask=attn_mask,
       scaling_factor=1.0 / math.sqrt(head_dim),
       is_training=True,
   )

See Also
--------

- :doc:`backends` — Backend descriptions (shared with PyTorch)
- :doc:`/developer/jax_frontend/xla_ffi_primitives` — How JAX primitives are registered

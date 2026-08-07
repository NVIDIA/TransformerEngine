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

The primary public function is:

- ``fused_attn()`` — Fused attention using cuDNN (equivalent to PyTorch's cuDNN fused
  backend).

It is a ``jax.custom_vjp`` function backed by the low-level ``fused_attn_fwd()`` and
``fused_attn_bwd()`` wrappers in ``transformer_engine/jax/cpp_extensions/attention.py``.
Those wrappers invoke JAX primitives registered through XLA FFI (see
:doc:`/developer/jax_frontend/xla_ffi_primitives`).

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

   import jax.numpy as jnp
   from transformer_engine.jax.attention import (
       AttnBiasType,
       AttnMaskType,
       AttnSoftmaxType,
       QKVLayout,
       SequenceDescriptor,
       fused_attn,
   )

   batch, seqlen, heads, head_dim = 2, 128, 16, 64
   qkv = jnp.zeros((batch, seqlen, 3, heads, head_dim), dtype=jnp.bfloat16)
   sequence_descriptor = SequenceDescriptor.from_seqlens(
       seqlens=(
           jnp.full((batch,), seqlen),
           jnp.full((batch,), seqlen),
       )
   )

   output = fused_attn(
       (qkv,),
       None,  # bias
       sequence_descriptor,
       None,  # dropout seed; no dropout in this example
       AttnBiasType.NO_BIAS,
       AttnMaskType.CAUSAL_MASK,
       QKVLayout.BS3HD,
       AttnSoftmaxType.VANILLA_SOFTMAX,
       head_dim**-0.5,
       0.0,  # dropout probability
       True,  # is_training
   )

See Also
--------

- :doc:`backends` — Backend descriptions (shared with PyTorch)
- :doc:`/developer/jax_frontend/xla_ffi_primitives` — How JAX primitives are registered

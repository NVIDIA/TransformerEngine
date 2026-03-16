..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

JAX Frontend
============

.. note::

   The JAX frontend documentation is a **draft**. JAX support in Transformer Engine is
   actively evolving and some details may change.

The JAX frontend (``transformer_engine/jax/``) provides Flax ``nn.Module`` wrappers,
XLA FFI custom-call primitives, and sharding utilities for distributed training.

.. toctree::
   :maxdepth: 1

   module_system
   xla_ffi_primitives
   quantization
   sharding

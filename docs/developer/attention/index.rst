..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Attention
=========

Attention is the most complex subsystem in Transformer Engine, spanning multiple backends,
hardware-specific optimizations, and distributed strategies. This section covers the
backend taxonomy, selection logic, context parallelism, and kernel organization.

.. toctree::
   :maxdepth: 1

   backends
   backend_selection
   context_parallel
   pytorch_attention
   jax_attention
   fused_attn_kernels

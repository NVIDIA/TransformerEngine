..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

fused_attn.h
============

.. deprecated:: 2.22

   The C fused-attention execution and support-query APIs, including their v2
   variants, configuration and parameter accessors, and C++ parameter wrappers,
   will be removed in the next major release. Use
   :class:`transformer_engine.pytorch.DotProductAttention` or
   :func:`transformer_engine.jax.attention.fused_attn` instead.

.. doxygenfile:: fused_attn.h

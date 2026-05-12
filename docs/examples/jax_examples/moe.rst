..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

JAX: Mixture of Experts with TransformerEngine
==============================================

**TODO — Coming soon.**

This document will cover TE's ``MoEBlock`` layer which utilizes TE's optimized
routing, permutation and grouped GEMM:

* single-GPU ``MoEBlock`` usage vs ``jax.lax.ragged_dot``
* expert-parallel sharding considerations.

`← Back to the JAX integration overview <../te_jax_integration.html>`_

..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

JAX: Integrating TransformerEngine into an existing framework
=============================================================

This is the landing page for a series of focused documents on bringing
TransformerEngine into a JAX+Flax codebase one optimization at a time. Each
linked page isolates a single feature so you can see exactly what changes are
required and what are the performance benefits.

Pick a topic
------------

.. list-table::
   :header-rows: 1
   :widths: 25, 15, 60

   * - Document
     - Status
     - Covers
   * - `Dense GEMMs <jax_examples/dense.html>`_
     - **Available**
     - ``nn.Dense`` → quantized GEMM; single-GPU speedup; multi-GPU speedup;
       Collective GEMM
   * - `Attention <jax_examples/attention.html>`_
     - *Coming soon*
     -
   * - `Mixture of Experts <jax_examples/moe.html>`_
     - *Coming soon*
     -


Quantization recipes at a glance
--------------------------------

TE exposes its quantization choices as **recipes**. Please see
`Low-precision Training
<https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/index.html>`_
for a more detailed description of each recipe.

.. list-table::
   :header-rows: 1
   :widths: 25, 15, 30, 30

   * - Recipe
     - Hardware
     - State
     - When to use
   * - ``DelayedScaling``
     - Hopper+
     - amax history (Flax variables)
     - Per-tensor FP8 with amax history
   * - ``Float8CurrentScaling``
     - Hopper+
     - none
     - Per-tensor FP8 without an amax history
   * - ``MXFP8BlockScaling``
     - Blackwell+
     - none
     - Block-scaled FP8 (32-element blocks)
   * - ``NVFP4BlockScaling``
     - Blackwell+
     - requires a Flax RNG ``sr_rng``
     - FP4 with 2D block scaling and stochastic rounding

Import them from ``transformer_engine.common.recipe``.


Conventions used across these documents
---------------------------------------

* **Framework.** Flax Linen. (TE/JAX uses Linen; see
  `Flax NNX/Linen interop
  <https://flax.readthedocs.io/en/latest/guides/bridge_guide.html>`_ and
  `Haiku/Flax interop
  <https://dm-haiku.readthedocs.io/en/latest/notebooks/flax.html>`_ if you're on
  a different stack.)
* **Baseline dtype.** bf16 for inputs and parameters.
* **Benchmarking.** ``quickstart_jax_utils.speedometer`` runs a JIT-compiled
  fwd+bwd loop with warmup.


.. toctree::
   :hidden:

   jax_examples/dense
   jax_examples/attention
   jax_examples/moe

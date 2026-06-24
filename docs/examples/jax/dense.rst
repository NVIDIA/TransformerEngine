..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

JAX: Dense GEMMs with TransformerEngine
=======================================

This document walks through replacing a plain ``flax.linen.Dense``'s GEMM with
TransformerEngine's quantized GEMM.

**Recipe.** We use ``MXFP8BlockScaling`` in this tutorial. ``MXFP8BlockScaling`` and
``NVFP4BlockScaling`` require a Blackwell-class GPU; on Hopper, swap in
``DelayedScaling`` or ``Float8CurrentScaling``. For more information on recipes, see this :ref:`recipe overview <jax_recipe_table_overview>`.

`← Back to the JAX integration overview <../te_jax_integration.html>`_

1. Baseline: a plain Flax Dense block
-------------------------------------

We isolate the optimization to a single linear layer so it's clear what's
changing. ``dot_general_cls`` is exposed as a constructor argument so we can swap
in TE later without touching the model definition.

.. literalinclude:: dense.py
   :language: python
   :start-after: # DENSE_BASELINE_MODEL_START
   :end-before: # DENSE_BASELINE_MODEL_END

.. literalinclude:: dense.py
   :language: python
   :start-after: # DENSE_INPUTS_SETUP_START
   :end-before: # DENSE_INPUTS_SETUP_END


2. Quantized Dense via ``make_dot_general_cls``
-----------------------------------------------

TE exposes a helper, ``te_flax.make_dot_general_cls(recipe)``, that returns a Flax
module class you pass directly to ``nn.Dense(..., dot_general=...)``.

With this API, TE doesn't create the ``kernel`` params; it only wraps the GEMM.
All your initialization, sharding annotations, and optimizer state stay where
they were.

.. literalinclude:: dense.py
   :language: python
   :start-after: # DENSE_TE_SETUP_START
   :end-before: # DENSE_TE_SETUP_END

If using ``DelayedScaling``, see [#delayedscaling]_.


3. Single-GPU performance
-------------------------

``speedometer`` runs a JIT-compiled forward+backward loop with warmup, on the
same input for both models.

.. literalinclude:: dense.py
   :language: python
   :start-after: # DENSE_SINGLE_GPU_BENCH_START
   :end-before: # DENSE_SINGLE_GPU_BENCH_END

.. raw:: html

   <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
      Output:
   </div>

.. container:: program-output

   .. literalinclude:: dense.out
      :language: text
      :start-after: # SINGLE_GPU_OUTPUT_START
      :end-before: # SINGLE_GPU_OUTPUT_END

On a single GB200, that's roughly **1.6× faster** for the fwd+bwd of one large
Dense — and the only code change was passing ``dot_general=te_dot_general_cls()``
into ``nn.Dense``.

The speedup depends on shape: large GEMMs benefit most. Very small GEMMs may
not benefit at all because the cast + scale overhead can dominate.

.. warning::

   **Remat / activation checkpointing.** If your training loop uses
   ``jax.checkpoint_policies.checkpoint_dots`` (or any policy that matches
   ``jax.lax.dot_general``), swap it for
   ``transformer_engine.jax.checkpoint_policies.checkpoint_dots_and_te_gemms``.
   Otherwise TE's quantized GEMM primitives won't be checkpointed correctly
   and your performance comparison will not be accurate.


4. Multi-GPU: DP=2 / TP=2 on a single Dense
-------------------------------------------

**Prerequisite:** this section requires four GPUs.

Keeping the same ``FlaxDenseBlock`` from the rest of the document, we run it on
a 2×2 mesh with **data parallelism** on one axis and **tensor parallelism**
(column-parallel: shard the kernel's output dim) on the other.

Two pieces wire this up:

1. A ``jax.sharding.Mesh`` you build once at module scope (outside JIT).
2. TE's ``MeshResource``, set globally via ``global_shard_guard``, which tells
   TE which mesh axes are DP and TP.

.. literalinclude:: dense.py
   :language: python
   :start-after: # DENSE_MULTI_GPU_MESH_SETUP_START
   :end-before: # DENSE_MULTI_GPU_MESH_SETUP_END

**Sharding plan:**

.. csv-table::
   :header: "Tensor", "Shape", "PartitionSpec"
   :widths: 30, 40, 30

   "Kernel (column-parallel)", "``(hidden, out_features)``", "``P(None, 'tp')``"
   "Input activations", "``(batch, seq, hidden)``", "``P('dp', None, None)``"
   "Gradient on output", "``(batch, seq, out_features)``", "``P('dp', None, 'tp')``"

.. literalinclude:: dense.py
   :language: python
   :start-after: # DENSE_MULTI_GPU_SHARD_SETUP_START
   :end-before: # DENSE_MULTI_GPU_SHARD_SETUP_END

.. literalinclude:: dense.py
   :language: python
   :start-after: # DENSE_MULTI_GPU_BENCH_START
   :end-before: # DENSE_MULTI_GPU_BENCH_END

.. raw:: html

   <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
      Output:
   </div>

.. container:: program-output

   .. literalinclude:: dense.out
      :language: text
      :start-after: # MULTI_GPU_OUTPUT_START
      :end-before: # MULTI_GPU_OUTPUT_END


Next steps
----------

* `Collective GEMM <collective_gemm.html>`_: further speedups by communicating between devices inside the GEMM.
* `← Hub <../te_jax_integration.html>`_

.. rubric:: Footnotes

.. [#delayedscaling] **DelayedScaling state.** Most recipes are stateless — scaling factors are computed from each
   tensor as it flows through the GEMM, so there is nothing to persist across steps. However, if you swap in
   ``DelayedScaling`` instead, ``init`` will produce a second variable collection,
   ``_overwrite_with_gradient``, holding ``kernel_amax_history``, ``kernel_scale``,
   ``x_amax_history``, ``x_scale``, etc. These are **not** model parameters — they are Flax
   variables that TE updates each step to compute per-tensor scales from a rolling amax window.
   If you use ``DelayedScaling``, you must thread the *entire* ``var_collect`` through your
   training loop (not just ``params``) so the history persists across steps, otherwise training
   accuracy will be impacted. ``MXFP8BlockScaling``, ``NVFP4BlockScaling``, and
   ``Float8CurrentScaling`` do not require this.

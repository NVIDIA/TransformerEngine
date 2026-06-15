..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

JAX: BF16 Mixture-of-Experts with TransformerEngine
===================================================

This document walks through replacing a native JAX/Flax expert-parallel MoE
block with TransformerEngine's experimental Flax ``_MoEBlock``.

**Baseline.** The reference path is pure JAX/Flax BF16. It uses
``jax.lax.ragged_all_to_all`` for expert-parallel token exchange and
``jax.lax.ragged_dot`` for the grouped expert FFNs. The low-level ragged
all-to-all setup lives in ``moe_native.py`` so the snippets below stay focused
on model-level code.

**TransformerEngine path.** This tutorial uses ``_MoEBlock`` in BF16 with
NCCL-backed TE EP and the wrapper's current no-op quantizer sets. TE EP replaces
the tutorial's previous TE-side ragged A2A exchange with ``tex.ep_dispatch`` and
``tex.ep_combine`` over NCCL EP. Quantized MoE recipes are intentionally out of
scope here.

`<- Back to the JAX integration overview <../te_jax_integration.html>`_

The forward path below summarizes the data flow for the native baseline and the
TE replacement.

.. figure:: media/jax_moe_native_vs_te_flow.svg
   :alt: Side-by-side forward data flow for native JAX and TransformerEngine JAX MoE blocks.
   :align: center
   :width: 100%

   Forward data flow for the tutorial's BF16 MoE block. The native baseline
   keeps JAX ``ragged_all_to_all`` and ``ragged_dot``. TE keeps the same sharded
   inputs and weights, but routes through TE fused router, NCCL EP
   dispatch/combine, and grouped GEMM primitives while keeping dispatch, expert
   compute, and combine inside one MoE VJP.

1. Baseline: native JAX BF16 EP MoE
-----------------------------------

The example uses a 2x2 mesh: expert parallelism on ``ep`` and FSDP-style batch
parallelism on ``fsdp``. The batch dimension is sharded over both axes, and
expert weights are sharded over ``ep``. TE EP requires ``ep`` to be the inner
axis and currently runs in multi-process mode with one GPU per process.

.. literalinclude:: moe.py
   :language: python
   :start-after: # MOE_IMPORTS_START
   :end-before: # MOE_IMPORTS_END

.. literalinclude:: moe.py
   :language: python
   :start-after: # MOE_CONFIG_START
   :end-before: # MOE_CONFIG_END

.. literalinclude:: moe.py
   :language: python
   :start-after: # MOE_MESH_SETUP_START
   :end-before: # MOE_MESH_SETUP_END

The native baseline is exposed as a normal Flax module. Its implementation in
``moe_native.py`` performs softmax top-k routing, forward
``ragged_all_to_all`` over ``ep``, local source-major to expert-major chunk
reordering, a concatenated ``wi_0|wi_1`` ``ragged_dot`` input projection,
activation, ``wo`` ``ragged_dot`` output projection, reverse
``ragged_all_to_all``, and weighted token combine.

2. TransformerEngine ``_MoEBlock``
----------------------------------

The TE replacement registers the same gate and expert parameter names as the
baseline, then delegates routing, dispatch, grouped FFN, combine,
expert-parallel collectives, and VJP to ``transformer_engine.jax.moe.moe``.
On this branch, the TE-side expert exchange is NCCL EP: ``_MoEBlock`` calls
``tex.ep_dispatch`` before the grouped FFNs and ``tex.ep_combine`` after them.
The native baseline remains unchanged and continues to use
``jax.lax.ragged_all_to_all`` for the comparison numbers.

``_MoEBlock`` is intentionally underscore-prefixed while the API stabilizes. Use
it as an experimental integration point.

.. literalinclude:: moe.py
   :language: python
   :start-after: # MOE_MODEL_SETUP_START
   :end-before: # MOE_MODEL_SETUP_END

.. literalinclude:: moe.py
   :language: python
   :start-after: # MOE_INPUTS_SETUP_START
   :end-before: # MOE_INPUTS_SETUP_END

3. TE EP smoke check
--------------------

The direct script path initializes the TE EP communicator, creates the
``_MoEBlock`` variables, runs a BF16 forward pass, and reports the output shape
and dtype.

.. literalinclude:: moe.py
   :language: python
   :start-after: # MOE_CORRECTNESS_START
   :end-before: # MOE_CORRECTNESS_END

The native ragged A2A baseline remains in ``moe_native.py`` and is used for the
baseline timings below. Because the native ragged A2A path runs in
single-process 4-GPU mode while TE EP runs in one-process-per-GPU mode, the
benchmark sweep times the two paths separately.

4. Performance comparison
-------------------------

``run_te_benchmark`` runs a blocking JIT-compiled forward+backward loop with
warmup. Even though quantization is disabled, the benchmark passes the active
``MeshResource`` through TE's autocast context so ``_MoEBlock`` can resolve the
``ep`` axis. The TE block folds top-k weights into the per-expert FFN
intermediate with ``apply_topk_weights_early=True``; this is mathematically
equivalent for the BF16 path because the down projection is linear.

.. literalinclude:: moe.py
   :language: python
   :start-after: # MOE_BENCH_START
   :end-before: # MOE_BENCH_END

Run the full example with:

.. code-block:: bash

   for i in 0 1 2 3; do
     python docs/examples/jax/moe.py --num-process=4 --process-id=$i > proc_$i.log 2>&1 &
   done
   wait

Measured on four NVIDIA GB200 GPUs with the default tutorial shape
``batch=8``, ``seq=2048``, ``hidden=1024``, ``intermediate=4096``,
``num_experts=8``, and ``topk=2``:

.. csv-table::
   :header: "Path", "Mean fwd+bwd time", "Relative time"
   :widths: 35, 25, 25

   "Native JAX BF16 ragged A2A", "17.085 ms", "1.00x"
   "TE ``_MoEBlock`` BF16 with NCCL EP", "3.156 ms", "0.18x"

For this no-op-quantizer BF16 configuration, TE EP measured ``5.41x`` the
native ragged A2A baseline throughput on this tutorial shape.

A larger-shape sweep with the same blocking timing loop found TE EP ahead for
each shape tried. The native column uses the unchanged ragged A2A baseline; the
TE column uses NCCL EP. The default shape appears in both tables; the values
differ slightly because the standalone tutorial run and sweep were timed
separately.

.. csv-table::
   :header: "Batch", "Seq", "Hidden", "Intermediate", "Native BF16", "TE BF16", "TE speedup"
   :widths: 10, 10, 12, 16, 16, 16, 14

   "8", "1024", "1024", "4096", "8.543 ms", "2.075 ms", "4.12x"
   "8", "2048", "1024", "4096", "17.085 ms", "3.217 ms", "5.31x"
   "8", "4096", "1024", "4096", "38.811 ms", "5.349 ms", "7.26x"
   "16", "2048", "1024", "4096", "39.194 ms", "5.355 ms", "7.32x"
   "8", "1024", "2048", "8192", "19.329 ms", "4.110 ms", "4.70x"
   "8", "2048", "2048", "8192", "42.505 ms", "6.254 ms", "6.80x"
   "16", "2048", "2048", "8192", "88.134 ms", "10.542 ms", "8.36x"

The result depends on token distribution, hidden size, intermediate size, and
the target stack.

.. raw:: html

   <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
      Output:
   </div>

.. container:: program-output

   .. literalinclude:: moe.out
      :language: text
      :start-after: # MOE_OUTPUT_START
      :end-before: # MOE_OUTPUT_END

Next steps
----------

* `Dense GEMMs <dense.html>`_: quantizing a single ``flax.linen.Dense`` GEMM.
* `Collective GEMM <collective_gemm.html>`_: further speedups by communicating
  between devices inside the GEMM.
* `<- Hub <../te_jax_integration.html>`_

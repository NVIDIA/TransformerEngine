..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

JAX: BSHD Attention with TransformerEngine
==========================================

This document walks through replacing a plain JAX implementation of BSHD
attention with TransformerEngine's fused ``DotProductAttention``. The example
uses `grouped-query attention (GQA) <https://arxiv.org/abs/2305.13245>`_ and
sliding-window attention (SWA).

`← Back to the JAX integration overview <../te_jax_integration.html>`_

1. Baseline: native BSHD GQA + SWA
----------------------------------

The baseline keeps query, key, and value as separate BSHD tensors. GQA is modeled
by repeating each KV head across a group of query heads, then applying a causal
sliding-window mask before softmax.

.. literalinclude:: attention.py
   :language: python
   :start-after: # ATTENTION_IMPORTS_START
   :end-before: # ATTENTION_IMPORTS_END

.. literalinclude:: attention.py
   :language: python
   :start-after: # ATTENTION_INPUTS_START
   :end-before: # ATTENTION_INPUTS_END

.. literalinclude:: attention.py
   :language: python
   :start-after: # ATTENTION_BASELINE_MODEL_START
   :end-before: # ATTENTION_BASELINE_MODEL_END


2. Transformer Engine ``DotProductAttention``
----------------------------------------------

The Transformer Engine version keeps the same separate BSHD inputs. The important arguments are
``num_gqa_groups`` for GQA, ``attn_mask_type="causal"`` for autoregressive
attention, and ``window_size`` for SWA.

.. literalinclude:: attention.py
   :language: python
   :start-after: # ATTENTION_TE_MODEL_START
   :end-before: # ATTENTION_TE_MODEL_END


3. Single-GPU performance
-------------------------

``speedometer`` runs a JIT-compiled forward+backward loop with warmup for both
implementations.

.. literalinclude:: attention.py
   :language: python
   :start-after: # ATTENTION_SINGLE_GPU_BENCH_START
   :end-before: # ATTENTION_SINGLE_GPU_BENCH_END

.. raw:: html

   <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
      Output:
   </div>

.. container:: program-output

   .. literalinclude:: attention.out
      :language: text
      :start-after: # SINGLE_GPU_OUTPUT_START
      :end-before: # SINGLE_GPU_OUTPUT_END

On a single GB200, this run is roughly **52x faster** for the fwd+bwd of this
BSHD GQA + SWA example. This compares TE ``DotProductAttention`` against the
native JAX baseline above, which materializes attention scores with XLA ops; it
is not a comparison against ``jax.nn.dot_product_attention(...,
implementation="cudnn")``.


4. MLA-style head dimensions
----------------------------

In TE/JAX, the simple MLA-style attention case is represented by separate Q, K,
and V tensors where Q/K and V use different per-head dimensions. Keep
``qkv_layout="bshd_bshd_bshd"`` so TE can see the Q/K head dimension and the V
head dimension separately.

.. literalinclude:: attention.py
   :language: python
   :start-after: # ATTENTION_MLA_START
   :end-before: # ATTENTION_MLA_END

.. raw:: html

   <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
      Output:
   </div>

.. container:: program-output

   .. literalinclude:: attention.out
      :language: text
      :start-after: # MLA_OUTPUT_START
      :end-before: # MLA_OUTPUT_END


Other attention knobs
---------------------

The examples above intentionally stay focused. Other ``DotProductAttention``
features are enabled through the same module arguments:

* Dropout: set ``attention_dropout > 0``, call with ``deterministic=False``, and
  pass a Flax ``dropout`` RNG to ``apply``.
* Bias: pass ``bias`` and set ``attn_bias_type`` when the selected fused kernel
  supports that bias mode.
* Sink attention: use ``softmax_type="off_by_one"`` or ``"learnable"``.
* Determinism: set ``NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`` before launching the
  process if deterministic fused kernels are required.


Next steps
----------

* `Context-parallel attention <attention_context_parallel.html>`_: packed THD
  attention over a context-parallel mesh.
* `← Hub <../te_jax_integration.html>`_

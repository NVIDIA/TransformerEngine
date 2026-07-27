..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

JAX: Single-GPU Attention with TransformerEngine
================================================

This document walks through replacing a plain JAX implementation of BSHD
attention with TransformerEngine's fused ``DotProductAttention``.
The example uses
`grouped-query attention (GQA) <https://arxiv.org/abs/2305.13245>`_ and
`sliding-window attention (SWA) <https://arxiv.org/pdf/1904.10509>`_.

`← Back to the Attention overview <attention.html>`_

1. Baseline: native JAX BSHD GQA + SWA
--------------------------------------

Start with the imports shared by the native JAX and Transformer Engine
implementations.

.. literalinclude:: attention.py
   :language: python
   :start-after: # ATTENTION_IMPORTS_START
   :end-before: # ATTENTION_IMPORTS_END

Next, create reproducible BSHD Q/K/V tensors and the sequence descriptor.
The ``SequenceDescriptor`` supplies TE with sequence lengths and, for packed
inputs, segment boundaries and padding metadata.

.. literalinclude:: attention.py
   :language: python
   :start-after: # ATTENTION_INPUTS_START
   :end-before: # ATTENTION_INPUTS_END

The native JAX baseline repeats K/V heads for GQA and applies the causal
sliding-window mask explicitly.

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


4. DeepSeek-style MLA head dimensions
-------------------------------------

This example covers the attention-kernel interface used after
`DeepSeek-style MLA projections <https://arxiv.org/abs/2405.04434>`_, not the
latent projection layers themselves. At this point, separate Q, K, and V
tensors can use different per-head dimensions for Q/K and V. Keep
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

The examples above represent a subset of attention features. Other
``DotProductAttention`` features can be enabled through the same module
arguments as below:

* Dropout: set ``attention_dropout > 0``, call with ``deterministic=False``, and
  pass a Flax ``dropout`` RNG to ``apply``.
* Bias: pass ``bias`` and set ``attn_bias_type`` when the selected fused kernel
  supports that bias mode.
* Sink attention: use ``softmax_type="off_by_one"`` or ``"learnable"``.
* Score scaling: set ``scale_factor`` to override the default
  ``1 / sqrt(head_dim)`` scaling.
* Determinism: set ``NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`` before launching the
  process if deterministic fused kernels are required.
* Score modification (experimental): use ``score_mod`` for a
  FlexAttention-style cuDNN frontend callback, with runtime operands in
  ``score_mod_tensors`` and optional custom backward logic in
  ``score_mod_bprop``. This path requires fused attention and currently cannot
  be combined with masks, bias, dropout, SWA, CP, or packed/ragged sequence
  metadata.


Next steps
----------

* `Context-parallel attention <attention_context_parallel.html>`_: packed THD
  attention over a context-parallel mesh.
* `← Attention overview <attention.html>`_
* `← Hub <../te_jax_integration.html>`_

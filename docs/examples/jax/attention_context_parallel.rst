..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

JAX: Context-Parallel THD Attention with Transformer Engine
===========================================================

This document demonstrates context parallelism (CP) with packed THD attention.
CP shards the sequence dimension over a JAX mesh axis so long-context attention
can split activation memory and attention work across devices while
Transformer Engine (TE) runs the required collectives inside the fused attention call.

CP is most useful when sequence length is large enough that single-device
attention becomes memory- or latency-limited. It is usually not worth adding for
short sequences or small local batches where communication overhead can dominate
the attention work.

**Prerequisite:** this example requires four GPUs.

`← Back to the JAX integration overview <../te_jax_integration.html>`_

1. Packed THD inputs
--------------------

In the separate-QKV THD layout used here, Q/K/V are shaped
``[batch, seq, heads, dim]``, and the sequence dimension can pack several
shorter segments. The ``SequenceDescriptor`` tells TE which tokens belong to
which packed segment and which token slots are padding. This tutorial uses
sixteen padded segments per sequence and a 64k sequence length.

.. literalinclude:: attention_context_parallel.py
   :language: python
   :start-after: # ATTENTION_CP_IMPORTS_START
   :end-before: # ATTENTION_CP_IMPORTS_END

.. literalinclude:: attention_context_parallel.py
   :language: python
   :start-after: # ATTENTION_CP_INPUTS_START
   :end-before: # ATTENTION_CP_INPUTS_END


2. Context-parallel mesh
------------------------

The JAX ``Mesh`` describes the physical devices. ``MeshResource`` tells TE which
mesh axis is used for context parallelism.

.. literalinclude:: attention_context_parallel.py
   :language: python
   :start-after: # ATTENTION_CP_MESH_START
   :end-before: # ATTENTION_CP_MESH_END


3. Fused THD attention call
---------------------------

This example calls ``transformer_engine.jax.attention.fused_attn`` directly. The
Flax ``DotProductAttention`` wrapper covers the common path, but the lower-level
function exposes ``stripe_size``. This tutorial uses ``fused_attn`` directly
because stripe-size tuning matters for CP + THD striped load balancing.

.. literalinclude:: attention_context_parallel.py
   :language: python
   :start-after: # ATTENTION_CP_FUSED_ATTENTION_START
   :end-before: # ATTENTION_CP_FUSED_ATTENTION_END


4. Striped load balancing and sharding
--------------------------------------

For THD causal CP, TE uses striped load balancing. Ring attention requires
``stripe_size=1``. AllGather can use a larger stripe size; this tutorial uses
``stripe_size=4096`` for the 64k sequence shape. Ring + SWA uses the non-scan
Ring path, set in the example before the first fused attention call is compiled.

.. literalinclude:: attention_context_parallel.py
   :language: python
   :start-after: # ATTENTION_CP_REORDER_START
   :end-before: # ATTENTION_CP_REORDER_END

.. literalinclude:: attention_context_parallel.py
   :language: python
   :start-after: # ATTENTION_CP_SHARD_START
   :end-before: # ATTENTION_CP_SHARD_END


5. Ring and AllGather
---------------------

Both examples use packed THD, causal masking, SWA, and dropout-free fused
attention. The only strategy-specific difference is the CP strategy and stripe
size. CP collectives depend on the compiler seeing the intended sharding, so the
forward and forward+backward functions are compiled with explicit
``in_shardings``; the forward+backward path also pins the gradient sharding. The
timing loop follows the same forward+backward pattern as ``speedometer`` while
keeping those sharding controls visible.

.. literalinclude:: attention_context_parallel.py
   :language: python
   :start-after: # ATTENTION_CP_RUN_START
   :end-before: # ATTENTION_CP_RUN_END

.. raw:: html

   <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
      Ring output:
   </div>

.. container:: program-output

   .. literalinclude:: attention_context_parallel.out
      :language: text
      :start-after: # RING_OUTPUT_START
      :end-before: # RING_OUTPUT_END

.. raw:: html

   <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
      AllGather output:
   </div>

.. container:: program-output

   .. literalinclude:: attention_context_parallel.out
      :language: text
      :start-after: # AG_OUTPUT_START
      :end-before: # AG_OUTPUT_END


Other attention knobs
---------------------

CP supports a narrower set of feature combinations than non-CP attention. These
examples intentionally use no bias, no dropout, and vanilla softmax. Enable other
features one at a time and keep the CP strategy, layout, mask, and sequence
descriptor choices explicit when debugging unsupported combinations.


Next steps
----------

* `BSHD attention <attention.html>`_: single-GPU BSHD GQA, SWA, and MLA-style
  head dimensions.
* `← Hub <../te_jax_integration.html>`_

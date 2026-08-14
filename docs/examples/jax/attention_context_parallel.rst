..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

JAX: Context-Parallel Attention with TransformerEngine
======================================================

Transformer Engine fused attention supports context parallelism (CP) for
selected BSHD and packed THD Q/K/V layouts; see the
`JAX DotProductAttention API reference
<../../api/jax.html#transformer_engine.jax.flax.DotProductAttention>`_ for the
layout definitions and full interface. This tutorial focuses on a
representative packed THD configuration with
`grouped-query attention (GQA) <https://arxiv.org/abs/2305.13245>`_, padded
segments, causal
`sliding-window attention (SWA) <https://arxiv.org/pdf/1904.10509>`_, and both
Ring and AllGather strategies.

CP shards the sequence dimension over a JAX mesh axis so long-context attention
can split activation memory and attention work across devices while
Transformer Engine (TE) runs the required collectives inside the fused
attention call.

.. note::

   CP is most useful when attention does not fit on one GPU, or when long
   sequences and sufficiently wide attention windows provide enough
   computation to amortize communication. For instance, applications that use
   GQA may be good candidates for CP because GQA's lower K/V head count reduces
   communication across devices. Conversely, workloads with narrow SWA windows
   may be better suited to single-GPU fused attention: CP still communicates
   K/V across devices while each query attends to relatively few tokens.

**Prerequisite:** this example requires four GPUs.

`← Back to the Attention overview <attention.html>`_

1. Packed THD inputs
--------------------

In the separate-QKV THD layout used here, Q/K/V are shaped
``[batch, seq, heads, dim]``, and the sequence dimension can pack several
shorter segments. The ``SequenceDescriptor`` tells TE which tokens belong to
which packed segment and which token slots are padding. CP supports separate
Q/K/V (``THD_THD_THD``) and packed K/V (``THD_T2HD``) layouts, but not fully
packed QKV (``T3HD``); this tutorial uses separate tensors. It uses a batch of
two 64k sequences. Each sequence contains four padded, 16k-capacity segment
slots with 12,288 valid tokens and 4,096 padding tokens per slot. It also uses
GQA with 128 query heads and 8 K/V heads.

.. literalinclude:: attention_context_parallel.py
   :language: python
   :start-after: # ATTENTION_CP_IMPORTS_START
   :end-before: # ATTENTION_CP_IMPORTS_END

The tensor inputs and packed-sequence metadata are created as follows.

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


3. Fused attention call
-----------------------

This example calls ``transformer_engine.jax.attention.fused_attn`` directly. The
Flax ``DotProductAttention`` wrapper covers the common path, but the lower-level
function exposes ``stripe_size``.

.. literalinclude:: attention_context_parallel.py
   :language: python
   :start-after: # ATTENTION_CP_FUSED_ATTENTION_START
   :end-before: # ATTENTION_CP_FUSED_ATTENTION_END


4. Striped load balancing and sharding
--------------------------------------

For THD causal CP, TE uses striped load balancing. Ring attention requires
``stripe_size=1``. AllGather can use a larger stripe size; this tutorial uses
``stripe_size=512`` for the 64k sequence shape. Ring + SWA uses the non-scan
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

The single-GPU baseline and both CP examples use the same packed THD GQA shape,
causal masking, 8192-token SWA window, and dropout-free fused attention. The
only strategy-specific difference between the two CP cases is the strategy and
stripe size. CP collectives depend on the compiler seeing the intended sharding,
so the forward and forward+backward functions are compiled with explicit
``in_shardings``; the forward+backward path also pins the gradient sharding.
The timing loop follows the same forward+backward pattern as ``speedometer``
while keeping those sharding controls visible.

.. literalinclude:: attention_context_parallel.py
   :language: python
   :start-after: # ATTENTION_CP_RUN_START
   :end-before: # ATTENTION_CP_RUN_END

.. raw:: html

   <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
      Single-GPU output:
   </div>

.. container:: program-output

   .. literalinclude:: attention_context_parallel.out
      :language: text
      :start-after: # SINGLE_GPU_OUTPUT_START
      :end-before: # SINGLE_GPU_OUTPUT_END

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

On four GB200s, Ring is roughly **2.22x faster** and AllGather roughly **2.36x
faster** than the equivalent single-GPU fused-attention forward+backward pass.
These results are specific to this workload and system. Applications with long
segments and wide attention windows generally have more attention computation
relative to communication and are stronger CP candidates; workloads with short
windows may see less benefit. Performance also depends on the batch and segment
lengths, head configuration, CP strategy, stripe size, and interconnect.



Next steps
----------

* `Single-GPU attention <attention_single_gpu.html>`_: BSHD GQA, SWA, and
  DeepSeek-style MLA head dimensions.
* `← Attention overview <attention.html>`_
* `← Hub <../te_jax_integration.html>`_

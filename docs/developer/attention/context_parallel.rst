..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _context-parallel:

Context Parallelism
===================

Context Parallelism (CP) distributes the sequence dimension across multiple GPUs,
enabling training with very long sequences that don't fit in a single GPU's memory.

.. figure:: ./img/context_parallel_ring.svg
   :align: center
   :width: 70%

   Ring of 4 GPUs passing KV chunks in context parallelism.

..
   Diagram description for ``context_parallel_ring.svg``:
   Four GPU boxes arranged in a ring (square layout).
   GPU 0 has "seq[0:L/4]", GPU 1 has "seq[L/4:L/2]", GPU 2 has "seq[L/2:3L/4]",
   GPU 3 has "seq[3L/4:L]".
   Arrows form a ring: GPU0 → GPU1 → GPU2 → GPU3 → GPU0.
   Each arrow is labeled "KV chunk".
   Center text: "Each GPU computes local Q × all KV via ring passes"

Overview
--------

In CP, each GPU holds a shard of the sequence for Q, K, and V. To compute full attention,
each GPU needs access to the K and V from *all* sequence shards. CP achieves this by
passing KV chunks between GPUs in a ring or all-gather pattern.

Strategies
----------

**Ring Attention**

KV chunks are passed around a ring of GPUs. Each GPU:

1. Computes attention for its local Q against the currently-held KV chunk.
2. Sends its KV chunk to the next GPU in the ring.
3. Receives a new KV chunk from the previous GPU.
4. Repeats until all KV chunks have been seen.

This overlaps communication with computation: while computing attention on the current
KV chunk, the next chunk is being transferred.

**All-Gather KV**

All GPUs perform an all-gather to collect the full K and V tensors, then compute
attention locally. Simpler than ring attention but requires more memory (full K and V
on each GPU).

**THD (Token-Head-Dimension) Layout**

A specialized layout where tokens from different sequence positions are interleaved
across GPUs in a way that enables efficient attention computation with minimal
communication. Used with TE's custom CUDA attention kernels.

Integration with Attention Backends
------------------------------------

CP support varies by attention backend:

- **cuDNN Fused**: Full CP support via both ring and all-gather strategies.
- **FlashAttention**: Limited CP support.
- **Unfused**: No CP support.

The CP strategy is configured via ``DotProductAttention``:

.. code-block:: python

   attn = te.DotProductAttention(
       num_attention_heads=32,
       kv_channels=128,
       cp_group=cp_process_group,
       cp_stream=cuda_stream,
   )

See Also
--------

- :doc:`backends` — Backend support for context parallelism
- :doc:`/developer/distributed/sequence_parallel` — Sequence parallelism (different from CP)

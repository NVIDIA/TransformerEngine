..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _sequence-parallel:

Sequence Parallelism
====================

Sequence Parallelism (SP) extends Tensor Parallelism by partitioning the sequence
dimension for operations that don't require the full hidden dimension (LayerNorm,
dropout, residual connections). This reduces memory by distributing activations that
TP alone would replicate.

How It Works
------------

In standard TP, operations like LayerNorm process the full sequence on every GPU
(replicated). With SP:

- **LayerNorm** and **residual additions** operate on sequence-partitioned data
  (each GPU handles ``seq_len / tp_size`` tokens).
- **Column-parallel linear** begins with an **all-gather** along the sequence dimension
  to reconstruct the full sequence before the GEMM.
- **Row-parallel linear** ends with a **reduce-scatter** along the sequence dimension
  instead of all-reduce, producing sequence-partitioned output.

.. code-block:: text

   Without SP:                    With SP:
   [Full seq, Full hidden]        [Seq/TP, Full hidden]
         │                              │
    LayerNorm                      LayerNorm (local)
         │                              │
    Column-parallel               All-gather → Column-parallel
         │                              │
    Row-parallel                   Row-parallel → Reduce-scatter
         │                              │
   [Full seq, Full hidden]        [Seq/TP, Full hidden]

Memory Savings
--------------

SP reduces activation memory by a factor of ``tp_size`` for:

- LayerNorm activations
- Dropout masks
- Residual connection buffers

These savings are significant for long-sequence training where activations dominate
memory.

Enabling SP
-----------

SP is enabled alongside TP:

.. code-block:: python

   layer = te.TransformerLayer(
       hidden_size=4096,
       ffn_hidden_size=16384,
       num_attention_heads=32,
       tp_group=tp_group,
       sequence_parallel=True,
   )

**Requirements:**

- TP must be enabled (``tp_group`` must be set).
- Sequence length must be divisible by ``tp_size``.
- All ranks in the TP group process the same batch.

Implementation
--------------

**Location**: ``transformer_engine/pytorch/distributed.py``

Key functions:

- ``gather_along_first_dim(input, tp_group)``: All-gather along sequence dim
  (before column-parallel GEMM).
- ``reduce_scatter_along_first_dim(input, tp_group)``: Reduce-scatter along sequence dim
  (after row-parallel GEMM).

These are inserted automatically by the TE modules when ``sequence_parallel=True``.

See Also
--------

- :doc:`tensor_parallel` — TP fundamentals that SP builds on
- :doc:`/developer/attention/context_parallel` — Context parallelism (different strategy
  for long sequences)

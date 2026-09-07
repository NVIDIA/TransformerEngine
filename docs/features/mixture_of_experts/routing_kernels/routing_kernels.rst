..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Routing kernels
===================================

Once the router has produced a routing map, the tokens must be moved into the
expert-contiguous layout expected by the grouped GEMM (``GroupedLinear`` in
PyTorch, ``grouped_dense`` in JAX) and, afterwards, moved back. Transformer
Engine provides differentiable kernels for both directions. This page focuses on
the two core operations - token dispatch and token combine - because they
illustrate the layout transformation used by the other variants.

The snippets below show one concrete instance of this pattern: the mask-map
routing path, exposed in PyTorch as ``transformer_engine.pytorch.moe_permute``
and ``transformer_engine.pytorch.moe_unpermute``, and in JAX as
``transformer_engine.jax.permutation.token_dispatch`` and
``transformer_engine.jax.permutation.token_combine``. Other routing variants
(for example, index-map routing in PyTorch via ``map_type="index"``) are
available in both frameworks and follow the same pattern; see the
:doc:`PyTorch API reference </api/pytorch>` and
:doc:`JAX API reference </api/jax>` for the complete list and signatures.
The mask-map APIs have different framework-specific wrappers, but lower to the
same shared Triton permutation kernels, and both pairs are differentiable so they
can be used directly inside training graphs.

Token dispatch
--------------

Token dispatch is the canonical routing operation: given the original token
tensor and a routing map describing each token's destination expert, it returns
a permuted token buffer in which all rows assigned to the same expert are
stored contiguously. In PyTorch this operation is exposed as ``moe_permute``;
in JAX it is exposed as ``token_dispatch``. This is exactly the layout that
the grouped linear layer consumes via its per-expert token-count argument
(``m_splits`` in PyTorch ``GroupedLinear``, ``group_sizes`` in JAX
``grouped_dense``), so token dispatch followed by the grouped GEMM forms a
typical MoE forward block.

.. raw:: html
   :file: img/moe_permute.svg

*Figure 1. Token dispatch consumes the input token tensor together with the
routing map and produces an expert-contiguous token tensor; rows assigned to the
same expert are stored back-to-back.*

A typical call looks like:

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: moe_permute_pytorch.py
         :language: python
         :start-after: # START_MOE_PERMUTE_PYTORCH
         :end-before: # END_MOE_PERMUTE_PYTORCH

   .. tab:: JAX

      .. literalinclude:: moe_permute_jax.py
         :language: python
         :start-after: # START_MOE_PERMUTE_JAX
         :end-before: # END_MOE_PERMUTE_JAX

Both variants return the permuted token buffer of shape
``[num_out_tokens, hidden_size]`` together with a ``row_id_map`` that
carries enough information for token combine to restore the original token
order once the expert computation is done. Token dispatch and token combine
are typically used as a matched pair around the grouped GEMM call.

Token combine
-------------

Token combine is the inverse routing operation: it takes the expert-contiguous
output produced by the grouped GEMM (or any per-expert computation) and the
``row_id_map`` returned by token dispatch, and returns a single tensor of
shape ``[num_tokens, hidden_size]`` with the rows written back into the
original token order. In PyTorch this operation is exposed as
``moe_unpermute``; in JAX it is exposed as ``token_combine``.

For top-1 routing each token has exactly one expert contribution, so
``merging_probs`` is omitted. For top-k routing pass the per-token expert
weights as ``merging_probs`` and the kernel computes a weighted sum of the
per-expert contributions in the same fused pass; without it the per-expert
contributions are summed unweighted. In PyTorch, also pass
``restore_shape=(num_tokens, hidden_size)`` whenever the permuted buffer has more
rows than the original tokens (top-k routing); JAX infers the original token
count from the ``row_id_map``.

.. raw:: html
   :file: img/moe_unpermute.svg

*Figure 2. Token combine reads the expert-contiguous output tensor and the*
``row_id_map``\ *, and writes each row back to its original token slot. With*
``merging_probs``\ *, contributions from multiple experts to the same token are
combined in the same fused kernel.*

A typical call looks like:

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: moe_unpermute_pytorch.py
         :language: python
         :start-after: # START_MOE_UNPERMUTE_PYTORCH
         :end-before: # END_MOE_UNPERMUTE_PYTORCH

   .. tab:: JAX

      .. literalinclude:: moe_unpermute_jax.py
         :language: python
         :start-after: # START_MOE_UNPERMUTE_JAX
         :end-before: # END_MOE_UNPERMUTE_JAX

Token probabilities
-------------------

In top-k routing each token contributes to several experts, and those
contributions are recombined using the routing weights. There are two equivalent
places to apply the weights:

* **At combine (output side).** Pass the routing weights to token combine as
  ``merging_probs``; it forms the weighted sum of the per-expert contributions in
  the same fused pass. This is the path used in the examples above.
* **At dispatch (input side).** Scale each expert's input by its routing weight
  before the grouped GEMM. ``moe_permute_with_probs`` (PyTorch) and the ``probs``
  argument of ``token_dispatch`` (JAX) permute a probability tensor alongside the
  tokens, so the weights arrive already aligned with the expert-contiguous
  layout.

Padding and alignment
---------------------

Grouped GEMM backends are most efficient when each expert's token block starts at
an aligned offset (for example, a multiple of 128 rows). Because the number of
tokens routed to an expert is data dependent, the blocks are generally ragged.
Transformer Engine can pad each block up to a multiple of ``align_size`` as part
of the dispatch kernel, avoiding a separate padding pass.

.. raw:: html
   :file: img/moe_padding.svg

*Figure 3. Each expert's block is rounded up to a multiple of* ``align_size``\ *.
The per-expert padding offsets are returned so that token combine can drop the
padding again.*

In PyTorch this is ``moe_permute_and_pad_with_probs``; in JAX it is the
``align_size`` argument of ``token_dispatch``. Both return the padded token
buffer, the aligned per-expert token counts (used as ``m_splits`` /
``group_sizes`` for the grouped GEMM), and the per-expert ``pad_offsets`` that
token combine needs in order to remove the padding.

.. tabs::

   .. tab:: PyTorch

      .. literalinclude:: moe_permute_pad_pytorch.py
         :language: python
         :start-after: # START_MOE_PERMUTE_PAD_PYTORCH
         :end-before: # END_MOE_PERMUTE_PAD_PYTORCH

   .. tab:: JAX

      .. literalinclude:: moe_permute_pad_jax.py
         :language: python
         :start-after: # START_MOE_PERMUTE_PAD_JAX
         :end-before: # END_MOE_PERMUTE_PAD_JAX

Reordering expert chunks
------------------------

When experts are sharded across devices, the per-expert token blocks often have
to be reordered - for example, to regroup tokens by destination rank before an
all-to-all, or to restore the original grouping afterwards.
``moe_sort_chunks_by_index`` (PyTorch) and ``sort_chunks_by_index`` (JAX) permute
contiguous chunks of a token tensor according to a list of chunk sizes and a
permutation of chunk indices, without falling back to Python-level slicing and
concatenation. ``moe_sort_chunks_by_index_with_probs`` reorders an accompanying
probability tensor in the same call.

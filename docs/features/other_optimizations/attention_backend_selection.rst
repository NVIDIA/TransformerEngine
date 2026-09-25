..
    Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _attention-backend-selection:

Attention Backend Selection
===========================

.. note::

   This page covers PyTorch ``DotProductAttention``. It describes broad
   feature support by backend family. Exact cuDNN patch-level gates and
   FlashAttention package-version gates are intentionally left to the runtime
   selector in
   ``transformer_engine/pytorch/attention/dot_product_attention/utils.py`` and
   the cuDNN backend probe in
   ``transformer_engine/common/fused_attn/fused_attn.cpp``.

Transformer Engine selects among three ``DotProductAttention`` backend
families:

* ``FlashAttention``: external FlashAttention 2, 3, or 4 packages.
* ``FusedAttention``: cuDNN fused attention, either F16/BF16 or FP8.
* ``UnfusedDotProductAttention``: the native PyTorch path with fused scaled
  masked softmax.

The selector first removes backend families disabled by environment variables
(``NVTE_FLASH_ATTN=0``, ``NVTE_FUSED_ATTN=0``, or ``NVTE_UNFUSED_ATTN=0``),
then applies feature gates. If both FlashAttention and FusedAttention survive,
Transformer Engine prefers FusedAttention on Hopper and newer GPUs and
FlashAttention on older supported GPUs. Unfused attention is the fallback when
accelerated backends do not support the requested configuration.

Legend
------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Term
     - Meaning
   * - Supported
     - The backend family is intended to support the row when its package,
       hardware, shape, and runtime gates also pass.
   * - Limited
     - Some common forms are supported, but the row depends on narrower shape,
       layout, architecture, package, or cuDNN gates.
   * - No
     - The backend family is removed by the selector for this feature class.
   * - N/A
     - The row does not apply to that backend family.

Backend Family Matrix
---------------------

Use this table first to find candidate backend families. A ``Supported`` cell is
not a final backend selection; the smaller resolver tables below still apply.

.. list-table::
   :header-rows: 1
   :widths: 25 16 16 16 16 27

   * - Scenario
     - FlashAttention
     - Fused F16/BF16
     - Fused FP8
     - Unfused
     - Notes
   * - FP16/BF16 dense self-attention
     - Supported
     - Supported
     - N/A
     - Supported
     - Common ``bshd``/``sbhd`` path with standard masks.
   * - FP32 attention
     - No
     - No
     - N/A
     - Supported
     - Accelerated backends require FP16/BF16 or FP8 inputs.
   * - FP8 DPA
     - Limited
     - N/A
     - Supported
     - Limited
     - FlashAttention is restricted to selected FA3 inference paths. Unfused
       FP8 requires emulation or ONNX export mode.
   * - GQA or MQA
     - Supported
     - Supported
     - Supported
     - Supported
     - Some backends require compatible head-group divisibility.
   * - MLA or unequal QK/V head dimensions
     - Limited
     - Limited
     - Limited
     - Supported
     - FlashAttention 2 is removed. FA3, FA4, and cuDNN accept selected shapes.
   * - THD packed variable-length input
     - Limited
     - Supported
     - Limited
     - Limited
     - Padding between packed sequences narrows FlashAttention and Unfused
       support.
   * - Arbitrary attention mask
     - No
     - No
     - No
     - Supported
     - Arbitrary masks route to Unfused for non-FP8 attention.
   * - KV cache
     - Limited
     - Limited
     - No
     - Supported
     - KV cache cannot be combined with context parallelism. FA4 is removed for
       KV cache.
   * - ``score_mod`` callback
     - No
     - Limited
     - No
     - No
     - Requires a narrow cuDNN F16/BF16 path.

FlashAttention Resolver
-----------------------

Use this table when the family matrix leaves FlashAttention as a candidate.

.. list-table::
   :header-rows: 1
   :widths: 12 18 18 17 17 18

   * - Backend
     - Architecture
     - Precision
     - Context parallel
     - KV cache
     - Broad exclusions
   * - FA2
     - ``sm80+``
     - FP16/BF16
     - Limited
     - Limited
     - No FP8 DPA, MLA, arbitrary mask, or THD padding-between-sequences.
   * - FA3
     - ``sm90``
     - FP16/BF16; limited FP8 inference
     - Limited
     - Limited
     - No dropout, ALiBi, explicit bias, arbitrary mask, or FP8 training.
   * - FA4
     - ``sm80+``
     - FP16/BF16
     - No
     - No
     - No FP8 DPA, THD padding-between-sequences, dropout, bias, or ALiBi.

FusedAttention Resolver
-----------------------

Use this table when the family matrix leaves FusedAttention as a candidate. The
installed cuDNN version still determines the exact accepted shapes.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Axis
     - F16/BF16 FusedAttention
     - FP8 FusedAttention
   * - Hardware
     - Broadly ``sm80+`` with additional Blackwell and ``sm120`` gates.
     - ``sm90+`` for delayed scaling, ``sm100+`` for current scaling and
       MXFP8; disabled on ``sm120``.
   * - Precision or recipe
     - FP16 and BF16 inputs.
     - FP8 delayed scaling, current scaling, or MXFP8. Block scaling and NVFP4
       do not route to FusedAttention.
   * - Layout
     - ``bshd``, ``sbhd``, ``thd``, selected split Q/KV layouts, and selected
       paged KV layouts.
     - Primarily ``bshd``/``sbhd``/``bhsd`` depending on cuDNN support; THD is
       excluded for MXFP8 and many CP paths.
   * - Masks
     - Standard no-mask, padding, causal, padding-causal, bottom-right causal,
       and padding-bottom-right causal forms when cuDNN accepts the shape.
     - Standard no-mask, causal, padding, padding-causal, and selected
       bottom-right causal forms.
   * - Bias
     - No bias, selected post-scale bias, and selected ALiBi converted to
       post-scale bias.
     - No bias only.
   * - Sliding window attention
     - Supported for selected masks with zero dropout and compatible sequence
       lengths.
     - Limited to newer cuDNN and Blackwell-family paths.
   * - Non-vanilla softmax
     - Supported when cuDNN accepts the shape.
     - Limited to newer cuDNN and Blackwell-family paths.
   * - Determinism
     - Supported for selected shapes; training with trainable bias, older
       cuDNN, or ``sm120`` may remove it.
     - Supported only on selected training paths.

Context Parallel Resolver
-------------------------

Context parallelism removes Unfused attention. It also removes FA4 and narrows
both FlashAttention and FusedAttention before cuDNN or FlashAttention package
gates run.

.. list-table::
   :header-rows: 1
   :widths: 14 20 20 20 26

   * - CP communication
     - FlashAttention
     - Fused F16/BF16
     - Fused FP8
     - Notes
   * - ``p2p``
     - Limited
     - Limited
     - Limited
     - Allows the broadest Fused CP path, including selected post-scale bias.
   * - ``all_gather``
     - Limited
     - Limited
     - Limited
     - THD routes are FlashAttention-only when FusedAttention is removed.
   * - ``a2a``
     - Limited
     - Limited
     - Limited
     - FusedAttention requires even ``num_heads`` and ``num_gqa_groups``.
   * - ``a2a+p2p``
     - Limited
     - Limited
     - Limited
     - FusedAttention requires even head counts and does not support THD in this
       mode.

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - CP feature
     - FlashAttention
     - Fused F16/BF16
     - Fused FP8
   * - ``bshd``/``sbhd`` FP16/BF16
     - Supported with CP mask and bias limits.
     - Supported with CP mask, bias, and communication limits.
     - N/A
   * - ``thd`` FP16/BF16
     - Supported with CP mask and bias limits.
     - No for ``all_gather`` and ``a2a+p2p``; limited otherwise.
     - N/A
   * - FP8 DPA
     - No
     - N/A
     - Limited to non-THD and no-bias paths.
   * - Bottom-right causal masks
     - No
     - No
     - No
   * - Causal cross-attention
     - No
     - No
     - No
   * - Explicit bias
     - Limited to selected post-scale bias.
     - Limited to selected post-scale bias; ``p2p`` only when bias is present.
     - No
   * - Sliding window attention
     - Limited
     - No for ``p2p`` and ``a2a+p2p``; limited otherwise.
     - Limited
   * - Non-vanilla softmax
     - No
     - Limited to ``a2a``.
     - Limited to ``a2a`` and newer cuDNN paths.

Mask And Modifier Resolver
--------------------------

Use this table after the precision/layout/CP tables. It summarizes the feature
classes that most often change the selected backend.

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - Feature
     - FlashAttention
     - FusedAttention
     - Unfused
   * - ``no_mask`` and padding masks
     - Supported
     - Supported
     - Supported
   * - Top-left causal cross-attention
     - No
     - Supported
     - Supported
   * - Bottom-right causal masks
     - Supported outside CP
     - Supported outside CP
     - Supported
   * - Arbitrary mask
     - No
     - No
     - Supported
   * - ALiBi
     - FA2 only, with limits.
     - Limited through post-scale bias conversion.
     - Supported
   * - Explicit pre/post-scale bias
     - No
     - Limited post-scale bias only.
     - Supported
   * - Dropout
     - FA2 only.
     - Supported except for selected sliding-window paths.
     - Supported
   * - ``return_max_logit=True``
     - No
     - Supported for non-FP8 paths when cuDNN accepts the shape.
     - Supported for non-FP8 paths.
   * - ``num_splits != 1``
     - Limited to FA3 or FA4 SplitKV on supported hardware.
     - No
     - No

When To Check The Code
----------------------

The tables above answer whether a feature class is intended to be supported by
a backend family. Check the runtime selector when a configuration depends on any
of the following:

* exact cuDNN version or known-bad cuDNN versions,
* exact FlashAttention package version or installed FlashAttention family,
* Blackwell, ``sm120``, or architecture-specific workarounds,
* head dimensions above 128, MLA, or mixed Q/K/V dimensions,
* paged KV cache, THD padding-between-sequences, or split Q/KV layouts,
* deterministic training with bias, FP8, CUDA graphs, or non-vanilla softmax.

..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _backend-selection:

Backend Selection
=================

Transformer Engine automatically selects the best attention backend based on the
hardware, input configuration, and user preferences. This page documents the selection
logic and how to override it.

Selection Flow
--------------

The backend is selected at the start of each ``DotProductAttention.forward()`` call.
The logic (in ``transformer_engine/pytorch/attention/dot_product_attention/``) follows
a priority order:

.. code-block:: text

   1. Check user override (env vars, constructor args)
   2. Check FP8 — if FP8 enabled, try cuDNN FP8 fused
   3. Check cuDNN F16 fused — if supported by config
   4. Check FlashAttention — if installed and supported
   5. Fall back to unfused attention

At each step, the logic checks whether the backend supports the current configuration:
head dimension, sequence length, mask type, dropout, GQA groups, etc.

Environment Variables
---------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - Description
   * - ``NVTE_FUSED_ATTN``
     - ``0`` to disable cuDNN fused attention, ``1`` to enable (default: ``1``)
   * - ``NVTE_FLASH_ATTN``
     - ``0`` to disable FlashAttention, ``1`` to enable (default: ``1``)
   * - ``NVTE_FUSED_ATTN_BACKEND``
     - Force a specific cuDNN fused attention sub-backend (integer ID)
   * - ``NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT``
     - Force workspace optimization for fused attention

Constructor Override
--------------------

``DotProductAttention`` accepts a ``backend`` parameter:

.. code-block:: python

   attn = te.DotProductAttention(
       num_attention_heads=32,
       kv_channels=128,
       attention_type="self",
       backend="flash_attention",  # Force FlashAttention
   )

Feature Compatibility Matrix
-----------------------------

Not all backends support all features. Key restrictions:

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 15 15

   * - Feature
     - Unfused
     - Flash
     - cuDNN F16
     - cuDNN FP8
   * - Causal mask
     - Yes
     - Yes
     - Yes
     - Yes
   * - Padding mask
     - Yes
     - Yes
     - Yes
     - Yes
   * - Sliding window
     - No
     - Yes
     - Yes
     - Yes
   * - GQA/MQA
     - Yes
     - Yes
     - Yes
     - Yes
   * - Dropout
     - Yes
     - Yes
     - Yes
     - Yes
   * - FP8 Q/K/V
     - No
     - No
     - No
     - Yes
   * - Context parallel
     - No
     - Limited
     - Yes
     - Yes
   * - Head dim > 256
     - Yes
     - No
     - Yes
     - Limited

Debugging Selection
-------------------

To see which backend was selected, set:

.. code-block:: bash

   NVTE_DEBUG=1

This logs the backend selection decision and the reason for any fallbacks.

See Also
--------

- :doc:`backends` — Detailed description of each backend
- :doc:`context_parallel` — How CP affects backend selection

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
The selection result is **cached** — if the attention parameters haven't changed since the
last call, the cached backend is reused without re-running the selection logic (see
``_attention_backends`` in ``dot_product_attention.py``).

The selection logic lives in ``get_attention_backend()`` in
``transformer_engine/pytorch/attention/dot_product_attention/utils.py``. It filters
backends by compatibility (head dimension, sequence length, mask type, dropout, GQA
groups, etc.) and then applies a priority order:

.. code-block:: text

   On Hopper+ (sm90):  FusedAttention > FlashAttention > Unfused
   On pre-Hopper:      FlashAttention > FusedAttention > Unfused

FusedAttention is preferred on Hopper+ for performance reasons. On pre-Hopper hardware,
FlashAttention takes priority when both are available.

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
     - Force a specific cuDNN fused attention sub-backend by integer ID:
       ``0`` = F16 max512, ``1`` = F16 arbitrary seqlen, ``2`` = FP8.
       These correspond to the ``NVTE_Fused_Attn_Backend`` enum in ``fused_attn.h``.

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
     - Yes (via arbitrary mask)
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

   NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1

``NVTE_DEBUG_LEVEL`` controls verbosity: ``1`` logs the selected backend (INFO level),
``2`` logs the full selection process including each filter step and why backends were
disabled (DEBUG level). Output goes through Python's ``logging`` module under the
``DotProductAttention`` logger name.

See Also
--------

- :doc:`backends` — Detailed description of each backend
- :doc:`context_parallel` — How CP affects backend selection

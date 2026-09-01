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

SM120 (Blackwell) Architecture Notes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SM120 GPUs have additional restrictions that affect backend selection:

- **KV caching**: Both FusedAttention and FlashAttention are disabled when KV caching
  (inference mode) is active. Only the Unfused backend is available for KV-cached
  inference on SM120.
- **cuDNN version**: FusedAttention requires cuDNN >= 9.18.1 for THD layout on SM120.
  With older cuDNN versions, FusedAttention is disabled for THD.
- **Layout restrictions**: Even with cuDNN >= 9.18.1, the T3HD and TH3D layouts
  (interleaved Q/K/V) are not supported on SM120.
- **Deterministic mode**: Deterministic attention is not supported on SM120.
- **Softmax LSE packed format**: The packed softmax LSE format for THD is excluded on
  SM120 (requires cuDNN >= 9.6.0 and a non-SM120 architecture).

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
   * - ``NVTE_UNFUSED_ATTN``
     - ``0`` to disable native PyTorch attention, ``1`` to enable (default: ``1``)

Selecting a Backend
-------------------

``DotProductAttention`` does not expose a constructor argument that forces one backend.
To select a backend for debugging or testing, enable the desired backend and disable the
other two before the module is first called. For example, to select FlashAttention:

.. code-block:: python

   import os

   os.environ["NVTE_FLASH_ATTN"] = "1"
   os.environ["NVTE_FUSED_ATTN"] = "0"
   os.environ["NVTE_UNFUSED_ATTN"] = "0"

The environment variables are read during backend selection. Set them before the first
``DotProductAttention.forward()`` call because the selection result is cached.

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

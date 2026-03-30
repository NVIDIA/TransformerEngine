..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _attention-backends:

Attention Backends
==================

Transformer Engine supports multiple attention backends, each optimized for different
hardware, precision, and feature combinations.

.. figure:: ./img/attention_backends.svg
   :align: center
   :width: 80%

   Taxonomy of attention backends with architecture annotations.

..
   Diagram description for ``attention_backends.svg``:
   Tree/taxonomy diagram:
   "Attention Backends"
     ├── "Unfused" — PyTorch native ops (manual QK^TV)
     │     └── Any GPU, BF16/FP16, no cuDNN required, fallback
     ├── "FlashAttention" — Tri Dao's flash-attn package (external)
     │     └── Ampere+, BF16/FP16, no cuDNN required
     └── "FusedAttention" — cuDNN graph API
           ├── Sub-backend 0: "F16 max512" — seqlen ≤ 512, Ampere+
           ├── Sub-backend 1: "F16 arbitrary" — any seqlen, Ampere+
           └── Sub-backend 2: "FP8" — FP8 Q/K/V, Hopper+

There are exactly three backends (Unfused, FlashAttention, FusedAttention).
FusedAttention has three cuDNN sub-backends selected automatically based on precision
and sequence length.

Backend Details
---------------

Unfused Attention
^^^^^^^^^^^^^^^^^

**Class**: ``UnfusedDotProductAttention`` in
``transformer_engine/pytorch/attention/dot_product_attention/backends.py``

Computes attention as separate PyTorch operations (``torch.baddbmm`` for QK^T,
``FusedScaleMaskSoftmax`` for masking + softmax, ``torch.bmm`` for the V multiply).
Used as a fallback when no fused backend supports the requested configuration. Also
useful for debugging since it produces identical numerics to textbook attention.

Sliding window is supported by converting the window specification into an ``arbitrary``
attention mask (see ``get_full_mask()`` in ``utils.py``).

FlashAttention
^^^^^^^^^^^^^^

**Class**: ``FlashAttention`` in
``transformer_engine/pytorch/attention/dot_product_attention/backends.py``

Integrates Tri Dao's `flash-attn <https://github.com/Dao-AILab/flash-attention>`_
package as an external dependency (not bundled). Two versions are supported:

- **FA2** (``flash_attn.flash_attn_interface``): Ampere and later (sm80+).
- **FA3** (``flash_attn_3.flash_attn_interface``): Hopper only (sm90). Selected
  automatically when installed and running on sm90; disabled on other architectures.

Version constraints are managed in ``FlashAttentionUtils`` in ``utils.py``.

Code flow:

1. ``FlashAttention.forward()`` converts TE's tensor layouts as needed — sbhd tensors
   are transposed to bshd, while THD (variable-length) inputs use FlashAttention's
   native ``flash_attn_varlen_func`` without converting to dense bshd.
2. Selects between ``flash_attn_func`` (dense bshd/sbhd without padding),
   ``flash_attn_varlen_func`` (THD or padding masks), or ``flash_attn_with_kvcache``
   (inference) based on the configuration.
3. For context parallelism, wraps the call via ``attn_forward_func_with_cp()``.

To modify FlashAttention integration: start with ``FlashAttention.forward()`` in
``backends.py`` for the Python wrapper, and ``FlashAttentionUtils`` in ``utils.py`` for
version and feature compatibility checks.

cuDNN Fused Attention (F16 and FP8)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Class**: ``FusedAttention`` in
``transformer_engine/pytorch/attention/dot_product_attention/backends.py``

Uses cuDNN's **cudnn-frontend graph API** to fuse the entire attention computation
(QK^T, scaling, masking, softmax, dropout, V multiply) into a single cuDNN operation.

Code flow:

1. **Python** (``backends.py``): ``FusedAttention.forward()`` → ``FusedAttnFunc.apply()``
   (custom autograd function).
2. **pybind11** (``pytorch/csrc/extensions/attention.cpp``): ``fused_attn_fwd()`` converts
   Python tensors to ``TensorWrapper`` objects and calls the C++ layer.
3. **C++ dispatcher** (``common/fused_attn/fused_attn.cpp``): Routes to the appropriate
   sub-backend implementation based on ``nvte_get_fused_attn_backend()``.
4. **cuDNN graph builders** — one file per sub-backend:

   - ``fused_attn_f16_max512_seqlen.cu`` — sub-backend 0 (legacy; only selected when
     sub-backend 1 is unavailable due to older cuDNN or restrictive parameters)
   - ``fused_attn_f16_arbitrary_seqlen.cu`` — sub-backend 1 (preferred for all F16 cases)
   - ``fused_attn_fp8.cu`` — sub-backend 2

   Each builds a ``cudnn_frontend::graph::Graph``, configures SDPA attributes (masking,
   dropout, scaling), and executes it. **Graphs are cached** per thread in thread-local
   caches (e.g., ``sdpa_f16_fprop_cache`` keyed by ``FADescriptor_v1`` for sub-backend 1,
   ``fa_fprop_cache`` keyed by ``FADescriptor`` for FP8) to avoid rebuilding identical
   configurations. See ``utils.h`` for the descriptor definitions.

To modify cuDNN fused attention: start with the relevant ``.cu`` file for the sub-backend,
specifically the ``fused_attn_*_fwd_impl()`` / ``fused_attn_*_bwd_impl()`` functions
where the cuDNN graph is built. For backend selection logic, see
``nvte_get_fused_attn_backend()`` in ``fused_attn.cpp``.

Helper CUDA Kernels
^^^^^^^^^^^^^^^^^^^

The ``transformer_engine/common/fused_attn/`` directory also contains CUDA helper kernels
that support the attention backends:

- ``utils.cu`` — stride calculation, auxiliary tensor setup
- ``context_parallel.cu`` — KV communication for context parallelism
- ``kv_cache.cu`` — KV cache management for inference

See :doc:`fused_attn_kernels` for the full C++ kernel organization.

See Also
--------

- :doc:`backend_selection` — How the backend is chosen at runtime
- :doc:`pytorch_attention` — PyTorch DotProductAttention and MultiheadAttention modules
- :doc:`fused_attn_kernels` — C++ kernel organization and cuDNN integration
- :doc:`context_parallel` — Context parallelism strategies and implementation

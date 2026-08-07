..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

PyTorch Attention Modules
=========================

The PyTorch frontend provides two attention modules at different abstraction levels.

DotProductAttention
-------------------

**Location**: ``transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py``

The core attention computation module. Given pre-projected Q, K, V tensors, it computes
scaled dot-product attention with automatic backend selection.

**Key parameters:**

- ``num_attention_heads`` / ``num_gqa_groups`` — Head configuration
- ``kv_channels`` — Per-head dimension
- ``attention_type`` — ``"self"`` or ``"cross"``
- ``attn_mask_type`` — ``"causal"``, ``"padding"``, ``"no_mask"``, ``"arbitrary"``
- ``window_size`` — For sliding window attention
- ``cp_group`` / ``cp_stream`` — Context parallelism configuration

**Forward signature** (simplified):

.. code-block:: python

   def forward(self, query, key, value,
               attn_mask=None,
               attention_bias=None,
               inference_params=None):

The module handles:

1. Backend selection (see :doc:`backend_selection`).
2. KV caching for autoregressive inference (via ``InferenceParams``).
3. Context parallel communication if ``cp_group`` is set.
4. FP8 quantization of Q/K/V if FP8 is enabled.

MultiheadAttention
------------------

**Location**: ``transformer_engine/pytorch/attention/multi_head_attention.py``

A higher-level module that wraps QKV projection + ``DotProductAttention`` + output
projection:

.. code-block:: text

   Input
     │
     ├── QKV Projection (LayerNormLinear, column-parallel)
     │   └── Produces Q, K, V tensors
     │
     ├── DotProductAttention
     │   └── Computes attention output
     │
     └── Output Projection (Linear, row-parallel)
         └── Projects back to hidden_size

**Key parameters** (in addition to DotProductAttention params):

- ``hidden_size`` — Input/output dimension
- ``input_layernorm`` — Whether to include LayerNorm before QKV projection
- ``return_layernorm_output`` — Return LN output for residual connection

InferenceParams
---------------

**Location**: ``transformer_engine/pytorch/attention/``

Manages KV cache for autoregressive generation:

.. code-block:: python

   class InferenceParams:
       max_sequence_length: int
       max_batch_size: int
       # Pre-allocated KV cache tensors
       key_value_memory_dict: dict  # layer_idx → (K_cache, V_cache)

During inference, ``DotProductAttention`` appends new KV to the cache and attends over
the full cached sequence.

Softmax Variants
----------------

TE provides fused softmax implementations for attention scores:

- ``ScaledSoftmax`` — Standard scaled softmax
- ``ScaledMaskedSoftmax`` — Softmax with arbitrary mask
- ``ScaledUpperTriangMaskedSoftmax`` — Causal mask (upper triangular)
- ``ScaledAlignedCausalMaskedSoftmax`` — Aligned causal mask variant

These are used by the unfused attention backend. The fused backends (cuDNN, Flash)
handle softmax internally.

See Also
--------

- :doc:`backends` — Which backend executes the attention
- :doc:`/developer/pytorch_frontend/module_hierarchy` — Where attention fits in the module tree

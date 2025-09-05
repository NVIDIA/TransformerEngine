..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

pyTorch
=======

.. autoapiclass:: transformer_engine.pytorch.Linear(in_features, out_features, bias=True, **kwargs)
  :members: forward, set_tensor_parallel_group

.. autoapiclass:: transformer_engine.pytorch.GroupedLinear(in_features, out_features, bias=True, **kwargs)
  :members: forward, set_tensor_parallel_group

.. autoapiclass:: transformer_engine.pytorch.LayerNorm(hidden_size, eps=1e-5, **kwargs)

.. autoapiclass:: transformer_engine.pytorch.RMSNorm(hidden_size, eps=1e-5, **kwargs)

.. autoapiclass:: transformer_engine.pytorch.LayerNormLinear(in_features, out_features, eps=1e-5, bias=True, **kwargs)
  :members: forward, set_tensor_parallel_group

.. autoapiclass:: transformer_engine.pytorch.LayerNormMLP(hidden_size, ffn_hidden_size, eps=1e-5, bias=True, **kwargs)
  :members: forward, set_tensor_parallel_group

.. autoapiclass:: transformer_engine.pytorch.DotProductAttention(num_attention_heads, kv_channels, **kwargs)
  :members: forward, set_context_parallel_group

.. autoapiclass:: transformer_engine.pytorch.MultiheadAttention(hidden_size, num_attention_heads, **kwargs)
  :members: forward, set_context_parallel_group, set_tensor_parallel_group

.. autoapiclass:: transformer_engine.pytorch.TransformerLayer(hidden_size, ffn_hidden_size, num_attention_heads, **kwargs)
  :members: forward, set_context_parallel_group, set_tensor_parallel_group

.. autoapiclass:: transformer_engine.pytorch.dot_product_attention.inference.InferenceParams(max_batch_size, max_sequence_length)
  :members: reset, allocate_memory, pre_step, get_seqlens_pre_step, convert_paged_to_nonpaged, step

.. autoapiclass:: transformer_engine.pytorch.CudaRNGStatesTracker()
  :members: reset, get_states, set_states, add, fork

.. autoapifunction:: transformer_engine.pytorch.fp8_autocast

.. autoapifunction:: transformer_engine.pytorch.fp8_model_init

.. autoapifunction:: transformer_engine.pytorch.checkpoint

.. autoapifunction:: transformer_engine.pytorch.make_graphed_callables

.. autoapifunction:: transformer_engine.pytorch.get_cpu_offload_context

.. autoapifunction:: transformer_engine.pytorch.moe_permute

.. autoapifunction:: transformer_engine.pytorch.moe_permute_with_probs

.. autoapifunction:: transformer_engine.pytorch.moe_unpermute

.. autoapifunction:: transformer_engine.pytorch.moe_sort_chunks_by_index

.. autoapifunction:: transformer_engine.pytorch.parallel_cross_entropy

.. autoapifunction:: transformer_engine.pytorch.moe_sort_chunks_by_index_with_probs

.. autoapifunction:: transformer_engine.pytorch.initialize_ub

.. autoapifunction:: transformer_engine.pytorch.destroy_ub

.. autoapiclass:: transformer_engine.pytorch.UserBufferQuantizationMode
  :members: FP8, NONE
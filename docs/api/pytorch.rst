..
    Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

pyTorch
=======

.. autoapiclass:: transformer_engine.pytorch.Linear(in_features, out_features, bias=True, **kwargs)
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

.. autoapiclass:: transformer_engine.pytorch.InferenceParams(max_batch_size, max_sequence_length)

.. autoapiclass:: transformer_engine.pytorch.CudaRNGStatesTracker()
  :members: reset, get_states, set_states, add, fork

.. autoapifunction:: transformer_engine.pytorch.fp8_autocast

.. autoapifunction:: transformer_engine.pytorch.fp8_model_init

.. autoapifunction:: transformer_engine.pytorch.checkpoint

.. autoapifunction:: transformer_engine.pytorch.onnx_export

.. autoapifunction:: transformer_engine.pytorch.make_graphed_callables

.. autoapifunction:: transformer_engine.pytorch.get_cpu_offload_context

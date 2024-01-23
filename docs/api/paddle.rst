..
    Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

paddle
======

.. autoapiclass:: transformer_engine.paddle.Linear(in_features, out_features, **kwargs)
  :members: forward

.. autoapiclass:: transformer_engine.paddle.LayerNorm(hidden_size, eps=1e-5, **kwargs)

.. autoapiclass:: transformer_engine.paddle.LayerNormLinear(in_features, out_features, eps=1e-5, **kwargs)
  :members: forward

.. autoapiclass:: transformer_engine.paddle.LayerNormMLP(hidden_size, ffn_hidden_size, eps=1e-5, **kwargs)
  :members: forward

.. autoapiclass:: transformer_engine.paddle.FusedScaleMaskSoftmax(attn_mask_type, mask_func, **kwargs)
  :members: forward

.. autoapiclass:: transformer_engine.paddle.DotProductAttention(num_attention_heads, kv_channels, **kwargs)
  :members: forward

.. autoapiclass:: transformer_engine.paddle.MultiHeadAttention(hidden_size, num_attention_heads, **kwargs)
  :members: forward

.. autoapiclass:: transformer_engine.paddle.TransformerLayer(hidden_size, ffn_hidden_size, num_attention_heads, **kwargs)
  :members: forward

.. autoapifunction:: transformer_engine.paddle.fp8_autocast

.. autoapifunction:: transformer_engine.paddle.recompute

..
    Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

pyTorch
=======

.. autoapiclass:: transformer_engine.pytorch.Linear(in_features, out_features, bias=True, **kwargs)
  :members: forward

.. autoapiclass:: transformer_engine.pytorch.LayerNorm(hidden_size, eps=1e-5, **kwargs)

.. autoapiclass:: transformer_engine.pytorch.RMSNorm(hidden_size, eps=1e-5, **kwargs)

.. autoapiclass:: transformer_engine.pytorch.LayerNormLinear(in_features, out_features, eps=1e-5, bias=True, **kwargs)
  :members: forward

.. autoapiclass:: transformer_engine.pytorch.LayerNormMLP(hidden_size, ffn_hidden_size, eps=1e-5, bias=True, **kwargs)
  :members: forward

.. autoapiclass:: transformer_engine.pytorch.DotProductAttention(num_attention_heads, kv_channels, **kwargs)
  :members: forward

.. autoapiclass:: transformer_engine.pytorch.TransformerLayer(hidden_size, ffn_hidden_size, num_attention_heads, **kwargs)
  :members: forward

.. autoapifunction:: transformer_engine.pytorch.fp8_autocast

.. autoapifunction:: transformer_engine.pytorch.checkpoint

.. autoapifunction:: transformer_engine.pytorch.onnx_export

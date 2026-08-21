..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Models
======

DeepSeek-V3
-----------

.. autoapiclass:: transformer_engine.pytorch.models.DeepSeekV3Layer(hidden_size, num_attention_heads, **kwargs)
  :members: forward

.. autoapiclass:: transformer_engine.pytorch.models.DeepSeekV3MoE(hidden_size, moe_ffn_hidden_size, num_experts, **kwargs)
  :members: forward, update_expert_bias

.. autoapiclass:: transformer_engine.pytorch.models.MultiLatentAttention(hidden_size, num_attention_heads, **kwargs)
  :members: forward

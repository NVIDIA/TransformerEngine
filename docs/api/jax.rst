..
    Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Jax
=======

.. autoapiclass:: transformer_engine.jax.MajorShardingType
.. autoapiclass:: transformer_engine.jax.ShardingType
.. autoapiclass:: transformer_engine.jax.flax.TransformerLayerType
.. autoapiclass:: transformer_engine.jax.ShardingResource(dp_resource=None, tp_resource=None)


.. autoapifunction:: transformer_engine.jax.fp8_autocast
.. autoapifunction:: transformer_engine.jax.update_collections
.. autoapifunction:: transformer_engine.jax.update_fp8_metas


.. autoapiclass:: transformer_engine.jax.flax.LayerNorm(epsilon=1e-6, layernorm_type='layernorm', **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.DenseGeneral(features, layernorm_type='layernorm', use_bias=False, **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.LayerNormDenseGeneral(features, layernorm_type='layernorm', epsilon=1e-6, use_bias=False, **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.LayerNormMLP(intermediate_dim=2048, layernorm_type='layernorm', epsilon=1e-6, use_bias=False, **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.RelativePositionBiases(num_buckets, max_distance, num_heads, **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.DotProductAttention(head_dim, num_heads, **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.MultiHeadAttention(head_dim, num_heads, **kwargs)
  :members: __call__

.. autoapiclass:: transformer_engine.jax.flax.TransformerLayer(hidden_size=512, mlp_hidden_size=2048, num_attention_heads=8, **kwargs)
  :members: __call__

.. autoapifunction:: transformer_engine.jax.flax.extend_logical_axis_rules

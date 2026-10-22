..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Jax
===

Pre-defined Variable of Logical Axes
------------------------------------
Variables are available in `transformer_engine.jax.sharding`.

* BATCH_AXES: The logical axis of batch dimension. It is usually sharded along DP + FSDP on Mesh.
* SEQLEN_AXES: The logical axis of sequence length dimension. It is usually not sharded.
* SEQLEN_TP_AXES: The logical axis of sequence length dimension. It is usually sharded along TP on Mesh.
* HEAD_AXES: The logical axis of head dimension of MHA. It is usually sharded along TP on Mesh.
* HIDDEN_AXES: The logical axis of hidden dimension. It is usually not sharded.
* HIDDEN_TP_AXES: The logical axis of hidden dimension. It is usually sharded along TP on Mesh.
* JOINED_AXES: The logical axis of non-defined dimension. It is usually not sharded.


Checkpointing
-------------
When using checkpointing with Transformer Engine JAX, please be aware of the checkpointing policy being applied to your model. Any JAX checkpointing policy using `dot`, such as `jax.checkpoint_policies.dots_with_no_batch_dims`, may not work with GEMMs provided by Transformer Engine as they do not always use the `jax.lax.dot_general` primitive. Instead, you can use `transformer_engine.jax.checkpoint_policies.dots_and_te_gemms_with_no_batch_dims` or similar policies that are designed to work with Transformer Engine's GEMMs and `jax.lax.dot_general` GEMMs. You may also use any JAX policies that do not filter by primitive, such as `jax.checkpoint_policies.save_only_these_names` or `jax.checkpoint_policies.everything_saveable`.

Per-call dense mesh resources
-----------------------------
The functional ``dense``, ``layernorm_dense``, and ``layernorm_mlp`` APIs accept
an optional ``mesh_resource`` argument. An explicit resource is
copied into an immutable static argument and carried through the custom VJP to
GEMM, quantization, activation, and normalization partitioning, including
parameter-gradient and amax reductions. It does not
modify the enclosing global mesh resource context. Omitting the argument keeps
the existing global-context behavior.

For example, a dense region can treat both the FSDP and expert mesh axes as
FSDP, while a routed MoE region assigns them separate roles::

    from transformer_engine.jax.dense import dense
    from transformer_engine.jax.sharding import MeshResource

    y = dense(x, kernel, mesh_resource=MeshResource(
        fsdp_resource=("fsdp", "expert"),
    ))

The tuple order should match the physical axes in the tensors' partition
specifications. GEMM gathers weight shards on these FSDP axes rather than
gathering activations to accommodate the weight sharding.

Modules
-------
.. autoapiclass:: transformer_engine.jax.flax.TransformerLayerType
.. autoapiclass:: transformer_engine.jax.MeshResource()


.. autoapifunction:: transformer_engine.jax.fp8_autocast
.. autoapifunction:: transformer_engine.jax.autocast
.. autoapifunction:: transformer_engine.jax.update_collections


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

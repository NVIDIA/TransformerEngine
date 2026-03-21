..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _transformer-layer:

TransformerLayer
================

``TransformerLayer`` (``transformer_engine/pytorch/transformer.py``) composes TE modules
into a complete Transformer block. Unlike other TE modules, it does not inherit from
``TransformerEngineBaseModule`` — it is a pure composition of TE sub-modules.

For the full constructor API and parameter descriptions, see the class docstring. This
page covers implementation details that are non-obvious or differ from a standard
Transformer implementation.

QKV Weight Handling
-------------------

The ``fuse_qkv_params`` parameter (default ``True``) controls whether Q, K, and V
projections share a single fused weight tensor or use separate weights. When
``fuse_qkv_params=True``, the QKV projection is a single ``LayerNormLinear`` with output
size ``3 * hidden_size`` (or adjusted for GQA), and the weight is split into Q, K, V
slices internally.

When ``fuse_qkv_params=False``:

- The ``qkv_weight_interleaved`` parameter is silently overridden to ``False``,
  regardless of what the user passed. This is because interleaved layout only makes
  sense with a fused weight tensor.
- ``fuse_wgrad_accumulation`` is not supported and raises an error.
- Despite using separate Q, K, V weight tensors, the code still performs a single GEMM.
  This is achieved through the ``parameters_split`` feature of the ``Linear`` /
  ``LayerNormLinear`` module: the three weight matrices are registered as separate
  parameters but are concatenated (via a no-op cat that avoids actual copying when the
  tensors are already contiguous in memory) into a single weight for the GEMM. The
  output is then split back into Q, K, V slices. This means the per-parameter API is
  preserved for checkpoint compatibility while the compute path remains a single fused
  GEMM.

The ``qkv_weight_interleaved`` parameter controls the memory layout of the fused QKV
weight. When ``True``, the Q, K, V weights for each attention head are interleaved
(``[Q_head0, K_head0, V_head0, Q_head1, ...]``). When ``False``, they are contiguous
blocks (``[Q_all_heads, K_all_heads, V_all_heads]``). The interleaved layout can be
more efficient for certain GEMM configurations.

Output Structure Varies by Configuration
-----------------------------------------

The return values from ``self_attention()`` and ``layernorm_mlp()`` change structure
depending on configuration flags:

- With ``apply_residual_connection_post_layernorm=False`` (default): returns
  ``(output, bias)``
- With ``apply_residual_connection_post_layernorm=True``: returns
  ``(output, bias, residual)``
- With ``parallel_attention_mlp=True``: ``return_bias`` is forced to ``False``, so the
  attention and MLP outputs are plain tensors that get combined in a single
  bias-dropout-add call.

The unpacking logic in ``forward()`` relies on the control flow to determine the tuple
shape — there is no explicit type checking.

Bias-Dropout-Add Executor Context
-----------------------------------

The bias-dropout-add fusion uses a context manager that varies by PyTorch version.
The minimum supported PyTorch version is 2.1, so in practice the handler is
``torch.enable_grad`` for PyTorch >= 2.2 (the current code path). The
``torch.enable_grad()`` context ensures that gradients are properly tracked through the
fused bias + dropout + residual addition operation, even when called from within a
``torch.no_grad()`` region during certain compilation or warmup phases.

.. note::

   The code still contains a legacy branch for PyTorch < 2.2 that uses ``nullcontext``
   (relying on NVFuser). Since TE requires PyTorch >= 2.1, this branch is only reachable
   on PyTorch 2.1.x. The NVFuser path may be removed in a future cleanup.

Activation Checkpointing
-------------------------

``TransformerLayer`` itself contains no checkpointing logic. The
``checkpoint_core_attention`` parameter is forwarded directly to
``DotProductAttention``, which handles the actual recomputation. For full-layer
checkpointing, use the TE-provided ``checkpoint()`` function from
``transformer_engine.pytorch.distributed`` (see
:doc:`autograd_integration`).

Parallel Attention and MLP
---------------------------

When ``parallel_attention_mlp=True``, the attention and MLP blocks run on the same input
(rather than sequentially) and their outputs are summed. This changes the data flow:

- Both blocks receive the same ``hidden_states`` input.
- The ``return_bias`` flag is forced to ``False`` for both sub-modules.
- A single ``_bias_dropout_add`` call combines both outputs with the residual.

This mode corresponds to the parallel formulation used in some model architectures
(e.g., PaLM).

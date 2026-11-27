..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

JAX: Attention with TransformerEngine
=====================================

Transformer Engine's JAX attention APIs support self-attention and
cross-attention with MHA, GQA, and MQA. Inputs can use standard BSHD or packed
THD layouts, with Q/K/V supplied separately or in packed forms.

Common options include causal and padding masks, bias (including ALiBi),
dropout, sliding-window attention (SWA), attention sinks, and experimental
``score_mod`` callbacks for FlexAttention-style customization. The API also
supports different Q/K and V head dimensions, as used by the attention
operation after the projection stages in DeepSeek-style MLA.

Reusing the same dropout key repeats the mask, including during forward
recomputation.

For long contexts, selected BSHD and THD configurations support context
parallelism with Ring or AllGather collectives. Exact fused-kernel availability
depends on the input shape, dtype, GPU architecture, and feature combination;
see the `JAX DotProductAttention API reference
<../../api/jax.html#transformer_engine.jax.flax.DotProductAttention>`_ for the
full interface. Choose the tutorial that matches how the sequence dimension is
distributed in your model.

FP8 attention
--------------

The Flax ``DotProductAttention``, ``MultiHeadAttention``, and
``TransformerLayer`` modules support two ``fp8`` settings. Use ``fp8=False``
for FP16/BF16 attention or ``fp8="mxfp8"`` for block-scaled MXFP8 attention.

With MXFP8, ``fused_attn`` and Flax ``DotProductAttention`` can also take
prequantized MXFP8 operands through ``quantized_qkv`` alongside the original
FP16/BF16 Q/K/V. The output and Q/K/V gradients retain the model dtype.

:envvar:`NVTE_FP8_DPA_BWD` controls whether the backward pass also uses FP8.
Availability depends on the GPU, installed cuDNN version, and attention
configuration.

`← Back to the JAX integration overview <../te_jax_integration.html>`_

Pick a tutorial
---------------

.. list-table::
   :header-rows: 1
   :widths: 30, 70

   * - Tutorial
     - Covers
   * - `Single-GPU Attention <attention_single_gpu.html>`_
     - BSHD GQA + SWA; performance against a native JAX baseline;
       DeepSeek-style MLA head dimensions
   * - `Context-Parallel Attention <attention_context_parallel.html>`_
     - Packed THD GQA + SWA on four GPUs; Ring and AllGather CP with striped
       load balancing; performance against single-GPU fused attention

.. toctree::
   :hidden:

   attention_single_gpu
   attention_context_parallel

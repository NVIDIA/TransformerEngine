..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

JAX: Attention with TransformerEngine
=====================================

Transformer Engine's JAX attention APIs support self-attention and
cross-attention with MHA, GQA, and MQA. Inputs can use standard BSHD or packed
THD layouts, with Q/K/V supplied separately or in packed forms.

Common options include causal and padding masks, bias, dropout, sliding-window
attention (SWA), attention sinks, and experimental ``score_mod`` callbacks for
FlexAttention-style customization. The API also supports different Q/K and V
head dimensions, as used by the attention operation after the projection
stages in DeepSeek-style MLA.

FP8 attention
--------------

Set ``fp8=True`` on the Flax ``DotProductAttention``, ``MultiHeadAttention``
or ``TransformerLayer`` module to enable FP8 attention with current tensor
scaling. Set ``fp8="mxfp8"`` for block-scaled MXFP8 attention. These options
are independent of the GEMM quantization recipe. Inputs, outputs and input
gradients retain their FP16/BF16 dtype.

:envvar:`NVTE_FP8_DPA_BWD` controls whether the backward pass also uses FP8.
Availability depends on the GPU, installed cuDNN version, and attention
configuration.

``AttnBiasType.ALIBI`` (``attn_bias_type="alibi"`` in Flax) enables cuDNN's
ALiBi mask without an explicit bias tensor. Explicit dropout keys have JAX's
usual replay semantics: using the same key repeats the mask, including when
recomputing a forward pass for its backward pass.

For long contexts, selected BSHD and THD configurations support context
parallelism with Ring or AllGather collectives. Exact fused-kernel availability
depends on the input shape, dtype, GPU architecture, and feature combination;
see the `JAX DotProductAttention API reference
<../../api/jax.html#transformer_engine.jax.flax.DotProductAttention>`_ for the
full interface. Choose the tutorial that matches how the sequence dimension is
distributed in your model.

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

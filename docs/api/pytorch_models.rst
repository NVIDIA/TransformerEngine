..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Model-specific layers (te.models)
=================================

The ``transformer_engine.pytorch.models`` namespace holds full transformer
layers for specific model families, composed from Transformer Engine modules
and fused kernels. Each family lives in its own subpackage.

DeepSeek-V3
-----------

A DeepSeek-V3 transformer layer analogous to
:class:`transformer_engine.pytorch.TransformerLayer`: Multi-Latent Attention
(MLA) with low-rank q/kv latents and decoupled RoPE/NoPE heads, plus a
DeepSeek-style Mixture of Experts block (fused sigmoid router with
aux-loss-free expert bias and node-limited grouped top-k, grouped-GEMM SwiGLU
experts, optional shared expert). The same architecture is used by other
model families (e.g. GLM-5, Kimi K2), which can reuse these modules.

Basic usage (single GPU, all experts local):

.. code-block:: python

    import torch
    import transformer_engine.pytorch as te

    layer = te.models.DeepSeekV3Layer(
        hidden_size=7168,
        num_attention_heads=128,
        num_experts=64,
        moe_ffn_hidden_size=2048,
        topk=8,
        shared_expert_ffn_hidden_size=2048,
        params_dtype=torch.bfloat16,
    )
    x = torch.randn(seq_len, batch, 7168, dtype=torch.bfloat16, device="cuda")
    y = layer(x)  # sbhd layout

Expert parallelism routes tokens between GPUs with the NCCL EP backend
(``transformer_engine.pytorch.ep``). Call ``ep_bootstrap`` once per process
before the first forward; EP requires bfloat16 inputs and NCCL >= 2.30.4:

.. code-block:: python

    from transformer_engine.pytorch.ep import ep_bootstrap

    ep_bootstrap(ep_group, num_experts=64, max_tokens_per_rank=tokens,
                 hidden_dim=7168, num_topk=8, recv_capacity_per_rank=capacity)
    layer = te.models.DeepSeekV3Layer(
        ...,
        ep_group=ep_group,
        ep_max_tokens_per_rank=tokens,
    )

On SM100-class GPUs the routed experts fuse into a single CuTe grouped-GEMM
MLP when running under ``te.autocast`` with an MXFP8/NVFP4 recipe and
``NVTE_CUTEDSL_FUSED_GROUPED_MLP=1``; elsewhere the same modules run unfused
with an identical checkpoint layout.

Loading HuggingFace checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The layer follows the HF/Megatron DeepSeek-V3 conventions (interleaved rope
weights, sigmoid router bias used for selection only). Weights map from
``transformers`` ``DeepseekV3DecoderLayer`` as follows (latent RMSNorms are
fused into the up-projections):

.. list-table::
   :header-rows: 1

   * - Transformer Engine
     - HuggingFace
   * - ``input_layernorm.weight``
     - ``input_layernorm.weight``
   * - ``pre_mlp_layernorm.weight``
     - ``post_attention_layernorm.weight``
   * - ``self_attention.q_down_proj.weight``
     - ``self_attn.q_a_proj.weight``
   * - ``self_attention.q_up_proj.{layer_norm_weight, weight}``
     - ``self_attn.{q_a_layernorm, q_b_proj}.weight``
   * - ``self_attention.kv_down_proj.weight``
     - ``self_attn.kv_a_proj_with_mqa.weight``
   * - ``self_attention.kv_up_proj.{layer_norm_weight, weight}``
     - ``self_attn.{kv_a_layernorm, kv_b_proj}.weight``
   * - ``self_attention.out_proj.weight``
     - ``self_attn.o_proj.weight``
   * - ``mlp.gate.weight`` / ``mlp.expert_bias``
     - ``mlp.gate.weight`` / ``mlp.gate.e_score_correction_bias``
   * - ``mlp.experts[0].weight{i}``
     - ``interleave_glu_tensor(cat([gate_proj, up_proj]), 32)`` of expert *i*
   * - ``mlp.experts[2].weight{i}``
     - ``mlp.experts.down_proj[i]``
   * - ``mlp.shared_expert[0].weight`` / ``[2].weight``
     - ``cat([gate_proj, up_proj])`` / ``down_proj`` of ``shared_experts``

See ``tests/pytorch/test_deepseek_hf.py`` for a complete, numerically
verified mapping.

API
^^^

.. autoapiclass:: transformer_engine.pytorch.models.DeepSeekV3Layer(hidden_size, num_attention_heads, **kwargs)
  :members: forward

.. autoapiclass:: transformer_engine.pytorch.models.DeepSeekV3MoE(hidden_size, moe_ffn_hidden_size, num_experts, **kwargs)
  :members: forward, update_expert_bias

.. autoapiclass:: transformer_engine.pytorch.models.MultiLatentAttention(hidden_size, num_attention_heads, **kwargs)
  :members: forward

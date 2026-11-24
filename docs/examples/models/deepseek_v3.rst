..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

DeepSeek-V3
===========

:class:`~transformer_engine.pytorch.models.DeepSeekV3Layer` is a drop-in
analogue of :class:`~transformer_engine.pytorch.TransformerLayer` for the
DeepSeek-V3 architecture: pre-RMSNorm, multi-latent attention (MLA),
residual, pre-RMSNorm, dense MLP or DeepSeek MoE, residual. The first layers
of DeepSeek-V3 are dense (``num_experts=None``), the rest use the MoE.

Features used
-------------

**Attention** (:class:`~transformer_engine.pytorch.models.MultiLatentAttention`)

* :class:`~transformer_engine.pytorch.DotProductAttention` with asymmetric
  head sizes (``kv_channels=(192, 128)``), which selects the cuDNN fused
  attention backend for the MLA shapes.
* Fused MLA RoPE kernels (:func:`~transformer_engine.pytorch.attention.mla_rope.apply_mla_rope_q`,
  :func:`~transformer_engine.pytorch.attention.mla_rope.apply_mla_rope_kv`)
  that rotate the rope slice and assemble the key and value tensors in one
  pass, with optional YaRN context extension
  (:func:`~transformer_engine.pytorch.attention.mla_rope.build_rope_tables`).
* :class:`~transformer_engine.pytorch.LayerNormLinear` with RMSNorm, so the
  latent normalizations are fused into the up-projections, and
  :class:`~transformer_engine.pytorch.Linear` for the query and key-value
  down-projections (``q_down_proj``, ``kv_down_proj``) and output projection,
  with column and row tensor parallelism on the up and output projections.

**MoE** (:class:`~transformer_engine.pytorch.models.DeepSeekV3MoE`)

* :ref:`Fused router <moe-router>`
  (:func:`~transformer_engine.pytorch.router.fused_topk_with_score_function`):
  sigmoid scores, aux-loss-free expert bias and node-limited grouped top-k in
  one kernel; :ref:`load balancing <moe-load-balancing>` through
  ``update_expert_bias()``.
* :ref:`Token permutation <moe-token-permutation>`
  (:func:`~transformer_engine.pytorch.moe_permute_with_probs`,
  :func:`~transformer_engine.pytorch.moe_permute_and_pad_with_probs`,
  :func:`~transformer_engine.pytorch.moe_unpermute`), padding each expert's
  rows to the alignment the active quantization recipe needs.
* Experts as an :doc:`operation fuser </examples/op_fuser/op_fuser>` sequence
  :class:`~transformer_engine.pytorch.ops.GroupedLinear` ->
  :class:`~transformer_engine.pytorch.ops.ScaledSwiGLU` ->
  :class:`~transformer_engine.pytorch.ops.GroupedLinear`, with the routing
  probabilities applied inside the activation. On Blackwell GPUs with BF16,
  MXFP8 or NVFP4 the fuser replaces the three operations by the
  :ref:`fused grouped MLP <moe-grouped-mlp>` kernel
  (``NVTE_CUTEDSL_FUSED_GROUPED_MLP=1``).
* :ref:`Expert parallelism <moe-expert-parallelism>` over NCCL
  (:class:`~transformer_engine.pytorch.ops.MoeDispatch`,
  :class:`~transformer_engine.pytorch.ops.MoeCombine`,
  :func:`~transformer_engine.pytorch.ep.ep_bootstrap`), fused with the expert
  sequence so dispatch and combine feed the grouped GEMM directly.
* Shared expert as a dense ``Linear -> SwiGLU -> Linear`` operation sequence:
  the GEMMs stay separate kernels, while :class:`~transformer_engine.pytorch.ops.SwiGLU`
  computes the gated product in one kernel and writes it directly in the
  quantized format the next GEMM consumes.

Usage
-----

Without ``ep_group`` the layer runs all experts locally; for multi-GPU
training, shard the routed experts across GPUs with NCCL expert parallelism
and launch one process per GPU with ``torchrun``.

.. tabs::

   .. tab:: Single GPU

      .. code-block:: python

          import torch
          import transformer_engine.pytorch as te
          from transformer_engine.common.recipe import MXFP8BlockScaling
          from transformer_engine.pytorch.models import DeepSeekV3Layer

          layer = DeepSeekV3Layer(
              hidden_size=7168,
              num_attention_heads=128,
              num_experts=32,
              moe_ffn_hidden_size=2048,
              topk=8,
              num_groups=8,
              group_topk=4,
              shared_expert_ffn_hidden_size=2048,
              params_dtype=torch.bfloat16,
          )

          x = torch.randn(1024, 4, 7168, dtype=torch.bfloat16, device="cuda")
          with te.autocast(recipe=MXFP8BlockScaling()):
              y = layer(x)
          y.sum().backward()
          optimizer.step()
          optimizer.zero_grad()
          layer.mlp.update_expert_bias()

   .. tab:: Expert parallelism

      .. code-block:: python

          import os
          import torch
          import torch.distributed as dist
          import transformer_engine.pytorch as te
          from transformer_engine.common.recipe import MXFP8BlockScaling
          from transformer_engine.pytorch.ep import ep_bootstrap, ep_finalize
          from transformer_engine.pytorch.models import DeepSeekV3Layer, DeepSeekV3MoE

          # 1. Process group and NCCL EP initialization (once per process)
          local_rank = int(os.environ["LOCAL_RANK"])
          torch.cuda.set_device(local_rank)
          dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
          ep_group = dist.new_group(backend="nccl")
          ep_size = dist.get_world_size(ep_group)

          hidden, topk = 7168, 8
          num_local_experts = 32            # 256 experts on 8 GPUs
          num_experts = num_local_experts * ep_size
          tokens_per_rank = 4096            # upper bound on local tokens per forward

          ep_bootstrap(
              ep_group,
              num_experts=num_experts,
              max_tokens_per_rank=tokens_per_rank,
              hidden_dim=hidden,
              num_topk=topk,
              recv_capacity_per_rank=DeepSeekV3MoE.ep_recv_capacity(
                  ep_size, tokens_per_rank, topk, num_local_experts
              ),
          )

          # 2. Layer and reusable EP buffer
          layer = DeepSeekV3Layer(
              hidden_size=hidden,
              num_attention_heads=128,
              num_experts=num_experts,
              moe_ffn_hidden_size=2048,
              topk=topk,
              num_groups=8,
              group_topk=4,
              shared_expert_ffn_hidden_size=2048,
              params_dtype=torch.bfloat16,
              ep_group=ep_group,
              ep_max_tokens_per_rank=tokens_per_rank,
          )
          ep_buffer = layer.mlp.make_ep_buffer()

          # 3. Training loop with aux-loss-free load balancing
          x = torch.randn(tokens_per_rank // 4, 4, hidden, dtype=torch.bfloat16, device="cuda")
          for step in range(num_steps):
              with te.autocast(recipe=MXFP8BlockScaling()):
                  y = layer(x, ep_buffer=ep_buffer)
              y.sum().backward()
              optimizer.step()
              optimizer.zero_grad()
              dist.all_reduce(layer.mlp._last_tokens_per_expert, group=ep_group)
              layer.mlp.update_expert_bias()

          ep_finalize()
          dist.destroy_process_group()

      .. code-block:: bash

          torchrun --nproc-per-node=8 train.py

Notes:

* Every rank owns ``num_experts // ep_size`` experts; ``ep_max_tokens_per_rank``
  bounds the local tokens of one forward and sizes the receive buffers.
  Tokens are exchanged in bf16, so the input of an EP layer must be bf16.
* An ``EpBuffer`` holds one forward's routing state until its backward has
  run; reuse one buffer per layer, or let the layer create one per call.
* Tensor parallelism of the attention is independent: pass ``tp_group`` /
  ``tp_size`` and the MLA up- and output projections are column- / row-sharded.

Example profile
---------------

Single MoE layer on **1 GB200 NVL4 node** (4 x GB200, SM clock locked to
1110 MHz, EP=4, 16,384 tokens/rank; hidden 7168, 128 heads, MoE ffn 2048,
32 experts / 8 local, top-k 8, shared ffn 2048), averaged over 10 timed
iterations after 10 warm-up iterations.

.. list-table::
   :header-rows: 1

   * - Precision
     - ms / iter
     - Mtok / s
     - Speedup
   * - BF16 (fused grouped MLP)
     - 69.2
     - 0.95
     - 1.00x
   * - FP8 current scaling
     - 58.6
     - 1.12
     - 1.18x
   * - MXFP8 (fused grouped MLP)
     - 52.9
     - 1.24
     - 1.31x
   * - NVFP4 (fused grouped MLP)
     - 48.2
     - 1.36
     - 1.44x

Table timings come from runs without the profiler. Each tab below shows a
single ``nsys`` iteration on rank 0 with forward and backward side-by-side on
the same time scale; hover any block for its group, kernel name and duration.

.. tabs::

   .. tab:: BF16

      No quantization; the experts run through the BF16 cuDNN fused grouped
      MLP (**69.2 ms/iter** unprofiled).

      .. raw:: html
         :file: profile_dsv3_bf16.svg

   .. tab:: FP8 current scaling

      ``te.autocast(recipe=Float8CurrentScaling())``: per-tensor FP8 with an
      amax reduction before every cast, unfused grouped GEMMs
      (**58.6 ms/iter** unprofiled).

      .. raw:: html
         :file: profile_dsv3_fp8_cs.svg

   .. tab:: MXFP8

      ``te.autocast(recipe=MXFP8BlockScaling())`` with the cuDNN fused grouped
      MLP (**52.9 ms/iter** unprofiled).

      .. raw:: html
         :file: profile_dsv3_mxfp8.svg

   .. tab:: NVFP4

      ``te.autocast(recipe=NVFP4BlockScaling())`` with the cuDNN fused grouped
      MLP; inputs and gradients are quantized by the random Hadamard transform
      kernel (**48.2 ms/iter** unprofiled).

      .. raw:: html
         :file: profile_dsv3_nvfp4.svg

Reproduce and environment
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Setup:** 4-rank ``torchrun`` (EP=4), input ``(4096, 4, 7168)`` BF16
  (16,384 tokens/rank, 65,536 global tokens), ``DeepSeekV3Layer`` with
  ``hidden_size=7168``, ``num_attention_heads=128``, ``q_lora_rank=1536``,
  ``kv_lora_rank=512``, ``qk_nope_head_dim=128``, ``qk_rope_head_dim=64``,
  ``v_head_dim=128``, ``num_experts=32`` (8 local/rank),
  ``moe_ffn_hidden_size=2048``, ``shared_expert_ffn_hidden_size=2048``,
  ``topk=8``, reused ``EpBuffer``, 10 warm-up + 10 timed forward+backward
  iterations, ``NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`` (0 for FP8 current scaling).
* **Hardware:** 1 GB200 NVL4 node — 4 x NVIDIA GB200 (Blackwell, SM 10.0)
  with NVLink, GPUs locked to 1110 MHz
  (``sudo nvidia-smi -lgc 1110,1110 --mode 1``), Grace CPU (aarch64), one
  process per GPU.
* **Software:** Transformer Engine built with ``NVTE_CUDA_ARCHS=100a``,
  PyTorch 2.15.0a0 (NVIDIA container 26.10), CUDA 13.4,
  ``nvidia-cudnn-frontend`` 1.33.0 (fused grouped MLP requires >= 1.31),
  container ``nvidia-cutlass-dsl`` 4.8.0, cuDNN fused attention
  (FlashAttention not used).

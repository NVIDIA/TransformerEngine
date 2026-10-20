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

cuDNN execution and FP8
----------------------

JAX fused attention constructs and selects cuDNN execution plans with the
``nvidia-cudnn-frontend`` Python package. The compiled JAX FFI executor caches
those plans and its own cuDNN handles per device. Ordinary attention and
``score_mod`` share this executor; neither calls TE common's attention APIs.
The Python frontend and the headers used to build the JAX extension must have
the same version, since execution plans are serialized between the two.

Pass ``fp8=True`` to ``transformer_engine.jax.attention.fused_attn`` or to the
Flax ``DotProductAttention``, ``MultiHeadAttention`` or ``TransformerLayer``
module to enable cuDNN FP8 attention. Inputs remain FP16 or BF16. Query, key
and value are quantized to E4M3 using their current tensor maximum; backward
quantizes the output gradient to E5M2. Outputs and input gradients retain the
input dtype, and softmax statistics use FP32. Packed layouts use one scale per
packed tensor. This option is independent of the GEMM quantization recipe and
does not maintain delayed-scaling history.

``NVTE_FP8_DPA_BWD=0`` selects FP16/BF16 backward while keeping FP8 forward.
The backward pass uses dequantized forward operands, so both passes operate on
the same quantized query, key and value. The default, ``1``, uses FP8 for both.

FP8 THD attention requires cuDNN 9.23 or later. FP8 THD backward with off-by-one
or learnable softmax requires cuDNN 9.26 or later; select
``NVTE_FP8_DPA_BWD=0`` to use FP16/BF16 backward instead. For FP16/BF16 THD
learnable-softmax backward with packed statistics, cuDNN versions before 9.26
remain eligible for value-head dimensions 64, 128, and 256. Other value-head
dimensions require cuDNN 9.26 or later. Execution-plan support still depends on
the complete attention configuration.

FP8 availability depends on cuDNN, GPU architecture, layout and the selected
features. Unsupported combinations produce a diagnostic rather than silently
using a different precision. For example, cuDNN 9.26 rejects FP8 backward with
dropout on Blackwell; use ``NVTE_FP8_DPA_BWD=0`` for that combination.
Pass ``fp8="mxfp8"`` to select MXFP8 attention on SM100 or later. MXFP8 uses
E4M3 operands with E8M0 block scales along the attention contraction axes in
both passes. Inputs, outputs and gradients remain FP16/BF16. Sequence lengths
must be divisible by 128 and head dimensions by 32. This mode currently supports
separate, QKV-packed and KV-packed BSHD inputs. THD, SM12x, and cuDNN
9.23.0/9.23.1 are excluded. It requires
cuDNN 9.21 or later. ``NVTE_FP8_DPA_BWD=0`` keeps the original FP16/BF16 operands
for MXFP8 backward, matching the PyTorch MXFP8 path.
Attention bias, ALiBi, dropout and ``return_max_logit`` are excluded from this mode.

Paged KV-cache APIs and quantized module inputs/outputs require further JAX API
and frontend-binding work. ``score_mod`` currently supports FP16/BF16 only.

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

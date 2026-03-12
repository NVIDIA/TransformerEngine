..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Getting Started
===============

Overview
--------

Transformer Engine (TE) is a library for accelerating Transformer models on NVIDIA GPUs,
providing better performance with lower memory utilization in both training and inference.
It provides support for 8-bit floating point (FP8) precision on Hopper and Ada GPUs, as well as
8-bit and 4-bit floating point (NVFP4) precision on Blackwell GPUs.

TE implements a collection of highly optimized building blocks for popular Transformer
architectures and exposes an automatic-mixed-precision-like API that can be used seamlessly
with your deep learning code.


Currently two frameworks are supported: PyTorch and JAX.

.. tabs::

   .. tab:: PyTorch

      Basic knowledge of PyTorch is recommended:

      - `PyTorch Tutorials <https://pytorch.org/tutorials/>`_
      - `PyTorch Documentation <https://pytorch.org/docs/stable/index.html>`_

   .. tab:: JAX

      We recommend understanding the basics of JAX first:

      - `Thinking in JAX <https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html>`_
      - `JAX 101 <https://docs.jax.dev/en/latest/jax-101.html>`_
      - `Key concepts in JAX <https://docs.jax.dev/en/latest/key-concepts.html>`_
      - `Flax 101 <https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/index.html>`_


Baseline: Pure Framework Implementation
---------------------------------------

Let's build a Transformer decoder layer!

We'll create a basic GPT-style layer with causal masking,
which prevents each position from attending to future positions. This will be our baseline
for later comparisons with Transformer Engine.

.. raw:: html
   :file: transformer_layer.svg

.. raw:: html

   <p style="text-align: center; font-style: italic; color: #666;">Structure of a GPT decoder layer</p>

We construct the components as follows:

.. tabs::

   .. tab:: PyTorch

      * **LayerNorm**: ``torch.nn.LayerNorm``
      * **QKV Projection**: ``torch.nn.Linear`` (fused Q, K, V into single layer 3x larger)
      * **DotProductAttention**: Custom implementation using ``torch.bmm``
      * **Projection**: ``torch.nn.Linear``
      * **Dropout**: ``torch.nn.Dropout``
      * **MLP**: Two ``torch.nn.Linear`` layers with ``torch.nn.functional.gelu`` activation

   .. tab:: JAX

      * **LayerNorm**: ``nn.LayerNorm``
      * **QKV Projection**: ``nn.Dense`` (fused Q, K, V into single layer 3x larger)
      * **DotProductAttention**: ``nn.dot_product_attention``
      * **Projection**: ``nn.Dense``
      * **Dropout**: ``nn.Dropout``
      * **MLP**: Two ``nn.Dense`` layers with ``nn.gelu`` activation

Putting it all together:

.. tabs::

   .. tab:: PyTorch

      First, define the MLP block:

      .. literalinclude:: getting_started_pytorch.py
         :language: python
         :start-after: # BASELINE_MLP_START
         :end-before: # BASELINE_MLP_END

      Now, putting it all together into a GPT decoder layer:

      .. literalinclude:: getting_started_pytorch.py
         :language: python
         :start-after: # BASELINE_LAYER_START
         :end-before: # BASELINE_LAYER_END

      Benchmark the baseline implementation:

      .. literalinclude:: getting_started_pytorch.py
         :language: python
         :start-after: # BENCHMARK_BASELINE_START
         :end-before: # BENCHMARK_BASELINE_END

      .. raw:: html

         <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
            Output:
         </div>

      .. container:: program-output

         .. literalinclude:: getting_started_pytorch.out
            :language: text
            :start-after: # BENCHMARK_BASELINE_OUTPUT_START
            :end-before: # BENCHMARK_BASELINE_OUTPUT_END

   .. tab:: JAX

      First, define the MLP block:

      .. literalinclude:: getting_started_jax.py
         :language: python
         :start-after: # BASELINE_MLP_START
         :end-before: # BASELINE_MLP_END

      Now, putting it all together into a GPT decoder layer:

      .. literalinclude:: getting_started_jax.py
         :language: python
         :start-after: # BASELINE_LAYER_START
         :end-before: # BASELINE_LAYER_END

      Benchmark the baseline implementation:

      .. literalinclude:: getting_started_jax.py
         :language: python
         :start-after: # BENCHMARK_BASELINE_START
         :end-before: # BENCHMARK_BASELINE_END

      .. raw:: html

         <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
            Output:
         </div>

      .. container:: program-output

         .. literalinclude:: getting_started_jax.out
            :language: text
            :start-after: # BENCHMARK_BASELINE_OUTPUT_START
            :end-before: # BENCHMARK_BASELINE_OUTPUT_END


TE Unfused: Basic TE Modules
----------------------------

Now let's replace the standard framework modules with TE equivalents.
This is the simplest way to start using Transformer Engine.

.. tabs::

   .. tab:: PyTorch

      Replace PyTorch modules with TE equivalents:

      .. code-block:: python

         import transformer_engine.pytorch as te

      Mapping:

      * ``torch.nn.Linear`` → ``te.Linear``
      * ``torch.nn.LayerNorm`` → ``te.LayerNorm``

      .. literalinclude:: getting_started_pytorch.py
         :language: python
         :start-after: # TE_UNFUSED_MLP_START
         :end-before: # TE_UNFUSED_MLP_END

      .. literalinclude:: getting_started_pytorch.py
         :language: python
         :start-after: # TE_UNFUSED_LAYER_START
         :end-before: # TE_UNFUSED_LAYER_END

      Benchmark the TE unfused implementation:

      .. literalinclude:: getting_started_pytorch.py
         :language: python
         :start-after: # BENCHMARK_TE_UNFUSED_START
         :end-before: # BENCHMARK_TE_UNFUSED_END

      .. raw:: html

         <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
            Output:
         </div>

      .. container:: program-output

         .. literalinclude:: getting_started_pytorch.out
            :language: text
            :start-after: # BENCHMARK_TE_UNFUSED_OUTPUT_START
            :end-before: # BENCHMARK_TE_UNFUSED_OUTPUT_END

   .. tab:: JAX

      Replace Flax modules with TE equivalents:

      .. code-block:: python

         import transformer_engine.jax as te
         import transformer_engine.jax.flax as te_flax

      Mapping:

      * ``nn.Dense`` → ``te_flax.DenseGeneral``
      * ``nn.LayerNorm`` → ``te_flax.LayerNorm``

      .. literalinclude:: getting_started_jax.py
         :language: python
         :start-after: # TE_UNFUSED_MLP_START
         :end-before: # TE_UNFUSED_MLP_END

      .. literalinclude:: getting_started_jax.py
         :language: python
         :start-after: # TE_UNFUSED_LAYER_START
         :end-before: # TE_UNFUSED_LAYER_END

      Benchmark the TE unfused implementation:

      .. literalinclude:: getting_started_jax.py
         :language: python
         :start-after: # BENCHMARK_TE_UNFUSED_START
         :end-before: # BENCHMARK_TE_UNFUSED_END

      .. raw:: html

         <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
            Output:
         </div>

      .. container:: program-output

         .. literalinclude:: getting_started_jax.out
            :language: text
            :start-after: # BENCHMARK_TE_UNFUSED_OUTPUT_START
            :end-before: # BENCHMARK_TE_UNFUSED_OUTPUT_END


TE Unfused + TE Attention
-------------------------

Now let's also replace the attention mechanism with TE's optimized ``DotProductAttention``.
TE's attention automatically selects the best available backend — for example, FlashAttention or cuDNN fused attention — based on your hardware and input configuration,
delivering optimal performance without manual tuning.

.. tabs::

   .. tab:: PyTorch

      Replace the custom attention with TE's optimized implementation:

      * Custom ``DotProductAttention`` → ``te.DotProductAttention``

      .. literalinclude:: getting_started_pytorch.py
         :language: python
         :start-after: # TE_UNFUSED_ATTN_LAYER_START
         :end-before: # TE_UNFUSED_ATTN_LAYER_END

      Benchmark TE Unfused with TE Attention:

      .. literalinclude:: getting_started_pytorch.py
         :language: python
         :start-after: # BENCHMARK_TE_UNFUSED_ATTN_START
         :end-before: # BENCHMARK_TE_UNFUSED_ATTN_END

      .. raw:: html

         <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
            Output:
         </div>

      .. container:: program-output

         .. literalinclude:: getting_started_pytorch.out
            :language: text
            :start-after: # BENCHMARK_TE_UNFUSED_ATTN_OUTPUT_START
            :end-before: # BENCHMARK_TE_UNFUSED_ATTN_OUTPUT_END

   .. tab:: JAX

      Replace Flax's attention with TE's optimized implementation:

      * ``nn.dot_product_attention`` → ``te_flax.DotProductAttention``

      .. literalinclude:: getting_started_jax.py
         :language: python
         :start-after: # TE_UNFUSED_ATTN_LAYER_START
         :end-before: # TE_UNFUSED_ATTN_LAYER_END

      Benchmark TE Unfused with TE Attention:

      .. literalinclude:: getting_started_jax.py
         :language: python
         :start-after: # BENCHMARK_TE_UNFUSED_ATTN_START
         :end-before: # BENCHMARK_TE_UNFUSED_ATTN_END

      .. raw:: html

         <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
            Output:
         </div>

      .. container:: program-output

         .. literalinclude:: getting_started_jax.out
            :language: text
            :start-after: # BENCHMARK_TE_UNFUSED_ATTN_OUTPUT_START
            :end-before: # BENCHMARK_TE_UNFUSED_ATTN_OUTPUT_END


TE Unfused + TE Attention + FP8
-------------------------------

Now let's combine TE modules with TE Attention and enable FP8 precision.
Wrap your code within an ``autocast`` context manager to enable FP8.
This provides significant speedups on supported hardware (Hopper, Ada, Blackwell GPUs).

.. tabs::

   .. tab:: PyTorch

      .. code-block:: python

         from transformer_engine.common.recipe import Format, DelayedScaling

         recipe = DelayedScaling(
             fp8_format=Format.HYBRID,
             amax_history_len=16,
             amax_compute_algo="max"
         )

         with te.autocast(enabled=True, recipe=recipe):
             y = te_unfused(x, attention_mask=None)

      .. note::

         The ``autocast`` should only wrap the forward pass and must exit before
         starting a backward pass.

      Benchmark TE Unfused with FP8:

      .. literalinclude:: getting_started_pytorch.py
         :language: python
         :start-after: # BENCHMARK_TE_UNFUSED_FP8_START
         :end-before: # BENCHMARK_TE_UNFUSED_FP8_END

      .. raw:: html

         <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
            Output:
         </div>

      .. container:: program-output

         .. literalinclude:: getting_started_pytorch.out
            :language: text
            :start-after: # BENCHMARK_TE_UNFUSED_FP8_OUTPUT_START
            :end-before: # BENCHMARK_TE_UNFUSED_FP8_OUTPUT_END

   .. tab:: JAX

      .. code-block:: python

         from transformer_engine.common.recipe import Format, DelayedScaling

         recipe = DelayedScaling(
             fp8_format=Format.HYBRID,
             amax_history_len=16,
             amax_compute_algo="max"
         )

         with te.autocast(enabled=True, recipe=recipe):
             params = te_unfused.init(key, x, deterministic=False)
             y = te_unfused.apply(params, x, deterministic=True)

      .. important::

         When using FP8 in JAX, the model **must be initialized within the autocast context**
         to create the ``fp8_metas`` collection.

      Benchmark TE Unfused with FP8:

      .. literalinclude:: getting_started_jax.py
         :language: python
         :start-after: # BENCHMARK_TE_UNFUSED_FP8_START
         :end-before: # BENCHMARK_TE_UNFUSED_FP8_END

      .. raw:: html

         <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
            Output:
         </div>

      .. container:: program-output

         .. literalinclude:: getting_started_jax.out
            :language: text
            :start-after: # BENCHMARK_TE_UNFUSED_FP8_OUTPUT_START
            :end-before: # BENCHMARK_TE_UNFUSED_FP8_OUTPUT_END


TE Fused + TE Attention + FP8: Optimized Modules
------------------------------------------------

Fused modules use kernel fusion to combine multiple operations.
While speedups are modest on a single GPU, they scale better in multi-GPU setups.
Combined with TE Attention and FP8, this delivers peak performance.

.. tabs::

   .. tab:: PyTorch

      Fused modules available:

      * ``te.LayerNormLinear`` - fuses LayerNorm + Linear
      * ``te.LayerNormMLP`` - fuses LayerNorm + MLP

      .. literalinclude:: getting_started_pytorch.py
         :language: python
         :start-after: # TE_FUSED_LAYER_START
         :end-before: # TE_FUSED_LAYER_END

      Benchmark TE Fused with FP8:

      .. literalinclude:: getting_started_pytorch.py
         :language: python
         :start-after: # BENCHMARK_TE_FUSED_FP8_START
         :end-before: # BENCHMARK_TE_FUSED_FP8_END

      .. raw:: html

         <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
            Output:
         </div>

      .. container:: program-output

         .. literalinclude:: getting_started_pytorch.out
            :language: text
            :start-after: # BENCHMARK_TE_FUSED_FP8_OUTPUT_START
            :end-before: # BENCHMARK_TE_FUSED_FP8_OUTPUT_END

   .. tab:: JAX

      Fused modules available:

      * ``te_flax.LayerNormDenseGeneral`` - fuses LayerNorm + Dense
      * ``te_flax.LayerNormMLP`` - fuses LayerNorm + MLP

      .. literalinclude:: getting_started_jax.py
         :language: python
         :start-after: # TE_FUSED_LAYER_START
         :end-before: # TE_FUSED_LAYER_END

      Benchmark TE Fused with FP8:

      .. literalinclude:: getting_started_jax.py
         :language: python
         :start-after: # BENCHMARK_TE_FUSED_FP8_START
         :end-before: # BENCHMARK_TE_FUSED_FP8_END

      .. raw:: html

         <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
            Output:
         </div>

      .. container:: program-output

         .. literalinclude:: getting_started_jax.out
            :language: text
            :start-after: # BENCHMARK_TE_FUSED_FP8_OUTPUT_START
            :end-before: # BENCHMARK_TE_FUSED_FP8_OUTPUT_END


TE TransformerLayer + FP8: Ready-to-use Module
----------------------------------------------

For the simplest integration, Transformer Engine provides a ready-to-use ``TransformerLayer``
module that includes all optimizations out of the box.

.. tabs::

   .. tab:: PyTorch

      Just use ``te.TransformerLayer`` - it handles everything for you:

      .. literalinclude:: getting_started_pytorch.py
         :language: python
         :start-after: # BENCHMARK_TE_TRANSFORMER_LAYER_START
         :end-before: # BENCHMARK_TE_TRANSFORMER_LAYER_END

      .. raw:: html

         <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
            Output:
         </div>

      .. container:: program-output

         .. literalinclude:: getting_started_pytorch.out
            :language: text
            :start-after: # BENCHMARK_TE_TRANSFORMER_LAYER_OUTPUT_START
            :end-before: # BENCHMARK_TE_TRANSFORMER_LAYER_OUTPUT_END

   .. tab:: JAX

      Just use ``te_flax.TransformerLayer`` - it handles everything for you:

      .. literalinclude:: getting_started_jax.py
         :language: python
         :start-after: # BENCHMARK_TE_TRANSFORMER_LAYER_START
         :end-before: # BENCHMARK_TE_TRANSFORMER_LAYER_END

      .. raw:: html

         <div style="background: #f5f5f5; border-left: 3px solid #9ca3af; padding: 4px 12px; font-size: 12px; color: #6b7280; margin-top: -16px;">
            Output:
         </div>

      .. container:: program-output

         .. literalinclude:: getting_started_jax.out
            :language: text
            :start-after: # BENCHMARK_TE_TRANSFORMER_LAYER_OUTPUT_START
            :end-before: # BENCHMARK_TE_TRANSFORMER_LAYER_OUTPUT_END


Benchmark Summary
-----------------

The table below summarizes the performance improvements achieved with Transformer Engine
on an NVIDIA H100 GPU. Results may vary depending on hardware and configuration. While this
tutorial focuses on a simple single-GPU scenario, features like fused layers can provide
additional benefits in more complex setups such as multi-GPU training.

.. tabs::

   .. tab:: PyTorch

      .. csv-table::
         :header-rows: 1
         :widths: 40, 20, 20
         :file: getting_started_pytorch_summary.csv

   .. tab:: JAX

      .. csv-table::
         :header-rows: 1
         :widths: 40, 20, 20
         :file: getting_started_jax_summary.csv

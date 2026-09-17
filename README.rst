..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Transformer Engine
==================

|PyPI| |Documentation| |License|

`Quick start <#quick-start>`_ | `User guide <https://docs.nvidia.com/deeplearning/transformer-engine/index.html>`_ | `PyTorch API <https://docs.nvidia.com/deeplearning/transformer-engine/api/pytorch.html>`_ | `JAX API <https://docs.nvidia.com/deeplearning/transformer-engine/api/jax.html>`_ | `Examples <https://github.com/NVIDIA/TransformerEngine/tree/main/examples>`_ | `Releases <https://github.com/NVIDIA/TransformerEngine/releases>`_

What is Transformer Engine?
===========================
.. overview-begin-marker-do-not-remove

Transformer Engine (TE) is an NVIDIA library for accelerating Transformer training on NVIDIA GPUs.
It combines optimized building blocks and fused kernels with automatic mixed-precision-style APIs
for PyTorch and JAX, so low-precision training can be adopted without rewriting a training stack.

Transformer Engine manages the scaling factors, amax histories, and quantization metadata required
by low-precision recipes. Its modules cover attention, linear layers, normalization, Mixture-of-Experts
(MoE), and communication operations used in large-scale distributed training.

Highlights
==========

* FP8 training on NVIDIA Hopper, Ada, and Blackwell GPUs.
* MXFP8 and NVFP4 training on NVIDIA Blackwell GPUs.
* Optimized attention, GEMM, normalization, quantization, and fused Transformer and MoE modules.
* PyTorch and JAX APIs with autocast-style contexts and configurable low-precision recipes.
* Support for tensor, sequence, context, and expert parallelism, including communication overlap.
* FP16 and BF16 optimizations on NVIDIA Ampere architecture GPUs and later.

.. overview-end-marker-do-not-remove

News
====

* **[09/2026]** `Transformer Engine v2.19 <https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.19>`_ adds Rubin support, hybrid quantization, MXFP8 expert-parallel communication, and expanded FP8 attention support.
* **[09/2026]** `Accelerating Dropless MoE Training in JAX with NVIDIA Transformer Engine <https://developer.nvidia.com/blog/accelerating-dropless-moe-training-in-jax-with-nvidia-transform-engine/>`_ describes optimized JAX MoE training on GB200 and GB300 systems.
* **[08/2026]** `Transformer Engine v2.18 <https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.18>`_ adds FP8 block scaling in PyTorch, zero-copy expert parallelism, and CUDA Graph support for THD attention.
* **[07/2026]** `Transformer Engine v2.17 <https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.17>`_ introduces NCCL-backed expert parallelism for PyTorch and JAX, faster MoE routing, and a JAX Flax MoE block.
* **[06/2026]** `Boosting MoE Training Throughput with Advanced Fusion Kernels <https://developer.nvidia.com/blog/boosting-moe-training-throughput-with-advanced-fusion-kernels/>`_ shows how fused Transformer Engine operations accelerate MoE training.

See the `project updates archive <docs/project_updates.rst>`_ for earlier news.

Quick start
===========

Install the latest stable release for your framework:

.. code-block:: bash

    # PyTorch
    pip install --no-build-isolation "transformer_engine[pytorch]"

    # JAX
    pip install --no-build-isolation "transformer_engine[jax]"

For a ready-to-run environment, use an NVIDIA NGC framework container. Replace ``<YY.MM>`` with a
container release listed in the `NVIDIA Deep Learning Frameworks Support Matrix <https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html>`_.

.. code-block:: bash

    docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:<YY.MM>-py3
    docker run --gpus all -it --rm nvcr.io/nvidia/jax:<YY.MM>-py3

Continue with the `PyTorch and JAX getting started guide <https://docs.nvidia.com/deeplearning/transformer-engine/getting_started/index.html>`_.
For prerequisites, source builds, environment variables, and troubleshooting, see the
`installation guide <https://docs.nvidia.com/deeplearning/transformer-engine/installation.html>`_.

Integrations
============

Transformer Engine has been integrated with popular LLM frameworks such as:

* `Hugging Face Accelerate <https://huggingface.co/docs/accelerate/main/en/usage_guides/low_precision_training#configuring-transformersengine>`_
* `Lightning <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.plugins.precision.TransformerEnginePrecision.html>`_
* `NVIDIA BioNeMo Recipes <https://github.com/NVIDIA-BioNeMo/bionemo-recipes>`_
* `NVIDIA JAX Toolbox <https://github.com/NVIDIA/JAX-Toolbox>`_
* `NVIDIA Megatron-LM <https://github.com/NVIDIA/Megatron-LM>`_
* `NVIDIA NeMo Megatron Bridge <https://github.com/NVIDIA-NeMo/Megatron-Bridge>`_
* `Amazon SageMaker Model Parallel Library <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features-v2-tensor-parallelism.html>`_

See `Ecosystem and historical integrations <docs/ecosystem.rst>`_ for additional
community integrations and projects that have worked with Transformer Engine.

Contributing
============

We welcome contributions to Transformer Engine! To contribute to Transformer Engine and make pull requests,
follow the guidelines outlined in the `<CONTRIBUTING.rst>`_ guide.

Technical deep dives
====================

* `Low precision training guide <https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/index.html>`_ — FP8, MXFP8, NVFP4, scaling recipes, and performance considerations.
* `Using FP8 and FP4 with Transformer Engine <https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html>`_ — a practical introduction with code examples.
* `FP8 Formats for Deep Learning <https://arxiv.org/abs/2209.05433>`_ — the foundational paper describing the FP8 formats used for deep learning.
* `Stable and Scalable FP8 Deep Learning Training on Blackwell <https://www.nvidia.com/en-us/on-demand/session/gtc25-s72778/>`_ — a technical GTC 2025 session on training numerics and scale.

See `Resources <docs/resources.rst>`_ for the complete collection of papers and recorded talks.

.. |PyPI| image:: https://img.shields.io/pypi/v/transformer-engine.svg
   :target: https://pypi.org/project/transformer-engine/
   :alt: PyPI release
.. |Documentation| image:: https://img.shields.io/badge/docs-latest-76B900.svg
   :target: https://docs.nvidia.com/deeplearning/transformer-engine/index.html
   :alt: Documentation
.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0

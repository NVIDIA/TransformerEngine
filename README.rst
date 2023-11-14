..
    Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

|License|

Transformer Engine
==================

`Quickstart <#examples>`_ | `Installation <#installation>`_ | `User Guide <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html>`_ | `Examples <https://github.com/NVIDIA/TransformerEngine/tree/main/examples>`_ | `Model Support <#model-support>`_ | `Integrations <#integrations>`_ | `Release notes <https://docs.nvidia.com/deeplearning/transformer-engine/release-notes/index.html>`_

Latest News
==================

* [04/2023] `Benchmarking Large Language Models on NVIDIA H100 GPUs with CoreWeave (Part 1) <https://www.mosaicml.com/blog/coreweave-nvidia-h100-part-1>`_


What is Transformer Engine?
==================
.. overview-begin-marker-do-not-remove

Transformer Engine (TE) is a library for accelerating Transformer models on NVIDIA GPUs, including
using 8-bit floating point (FP8) precision on Hopper GPUs, to provide better performance with lower
memory utilization in both training and inference. TE provides a collection of highly optimized
building blocks for popular Transformer architectures and an automatic mixed precision-like API that
can be used seamlessly with your framework-specific code. TE also includes a framework agnostic
C++ API that can be integrated with other deep learning libraries to enable FP8 support for Transformers.

As the number of parameters in Transformer models continues to grow, training and inference for
architectures such as BERT, GPT and T5 become very memory and compute-intensive. Most deep learning
frameworks train with FP32 by default. This is not essential, however, to achieve full accuracy for
many deep learning models. Using mixed-precision training, which combines single-precision (FP32)
with lower precision (e.g. FP16) format when training a model, results in significant speedups with
minimal differences in accuracy as compared to FP32 training. With Hopper GPU
architecture FP8 precision was introduced, which offers improved performance over FP16 with no
degradation in accuracy. Although all major deep learning frameworks support FP16, FP8 support is
not available natively in frameworks today.

TE addresses the problem of FP8 support by providing APIs that integrate with popular Large Language
Model (LLM) libraries. It provides a Python API consisting of modules to easily build a Transformer
layer as well as a framework-agnostic library in C++ including structs and kernels needed for FP8 support.
Modules provided by TE internally maintain scaling factors and other values needed for FP8 training, greatly
simplifying mixed precision training for users.

Highlights
----------

* Easy-to-use modules for building Transformer layers with FP8 support
* Optimizations (e.g. fused kernels) for Transformer models
* Support for FP8 on NVIDIA Hopper and NVIDIA Ada GPUs
* Support for optimizations across all precisions (FP16, BF16) on NVIDIA Ampere GPU architecture generations and later

Examples
----------

PyTorch
^^^^^^^

.. code-block:: python

  import torch
  import transformer_engine.pytorch as te
  from transformer_engine.common import recipe

  # Set dimensions.
  in_features = 768
  out_features = 3072
  hidden_size = 2048

  # Initialize model and inputs.
  model = te.Linear(in_features, out_features, bias=True)
  inp = torch.randn(hidden_size, in_features, device="cuda")

  # Create an FP8 recipe. Note: All input args are optional.
  fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)

  # Enable autocasting for the forward pass
  with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
      out = model(inp)

  loss = out.sum()
  loss.backward()


JAX
^^^

Flax
~~~~

.. code-block:: python

  import jax
  import jax.numpy as jnp
  import transformer_engine.jax as te
  import transformer_engine.jax.flax as te_flax
  from transformer_engine.common import recipe

  BATCH = 32
  SEQLEN = 128
  HIDDEN = 1024

  # Initialize RNG and inputs.
  rng = jax.random.PRNGKey(0)
  init_rng, data_rng = jax.random.split(rng)
  inp = jax.random.normal(data_rng, [BATCH, SEQLEN, HIDDEN], jnp.float32)

  # Create an FP8 recipe. Note: All input args are optional.
  fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.HYBRID)

  # Enable autocasting for the forward pass
  with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
      model = te_flax.DenseGeneral(features=HIDDEN)

      def loss_fn(params, other_vars, inp):
        out = model.apply({'params':params, **other_vars}, inp)
        return jnp.mean(out)

      # Initialize models.
      variables = model.init(init_rng, inp)
      other_variables, params = variables.pop('params')

      # Construct the forward and backward function
      fwd_bwd_fn = jax.value_and_grad(loss_fn, argnums=(0, 1))

      for _ in range(10):
        loss, (param_grads, other_grads) = fwd_bwd_fn(params, other_variables, inp)

.. overview-end-marker-do-not-remove

Installation
----------
.. installation

Pre-requisites
^^^^^^^^^^^^^^^^^^^^
* Linux x86_64
* CUDA 11.8+ for Hopper and CUDA 12.1+ for Ada
* NVIDIA Driver supporting CUDA 11.8 or later
* cuDNN 8.1 or later
* For fused attention, CUDA 12.1 or later, NVIDIA Driver supporting CUDA 12.1 or later, and cuDNN 8.9 or later.

Docker
^^^^^^^^^^^^^^^^^^^^

The quickest way to get started with Transformer Engine is by using Docker images on
`NVIDIA GPU Cloud (NGC) Catalog <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`_. For example to use the NGC PyTorch container interactively,

.. code-block:: bash

    docker run --gpus all -it --rm nvcr.io/nvidia/pytorch:23.10-py3

Where 23.10 is the container version. For example, 23.10 for the October 2023 release.

pip
^^^^^^^^^^^^^^^^^^^^
To install the latest stable version of Transformer Engine,

.. code-block:: bash

    pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

This will automatically detect if any supported deep learning frameworks are installed and build Transformer Engine support for them. To explicitly specify frameworks, set the environment variable NVTE_FRAMEWORK to a comma-separated list (e.g. NVTE_FRAMEWORK=jax,pytorch).

From source
^^^^^^^^^^^
`See the installation guide <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html#installation-from-source>`_.

Compiling with FlashAttention-2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Transformer Engine release v0.11.0 adds support for FlashAttention-2 in PyTorch for improved performance. 

It is a known issue that FlashAttention-2 compilation is resource-intensive and requires a large amount of RAM (see `bug <https://github.com/Dao-AILab/flash-attention/issues/358>`_), which may lead to out of memory errors during the installation of Transformer Engine. Please try setting **MAX_JOBS=1** in the environment to circumvent the issue. If the errors persist, install a supported version of FlashAttention-1 (v1.0.6 to v1.0.9).

Note that NGC PyTorch 23.08+ containers include FlashAttention-2.

Model Support
----------

While the more granular modules in Transformer Engine allow building any Transformer architecture,
the `TransformerLayer` API of Transformer Engine is flexible enough to build multiple major
Transformer model architectures.

Transformer Engine supports the following DL frameworks: PyTorch and JAX (Flax, Praxis).

NOTE: For simplicity, we only show PyTorch examples below. For the usage of `TransformerLayer`
of all supported frameworks, refer to `examples <https://github.com/NVIDIA/TransformerEngine/tree/main/examples>`_.

GPT
^^^

`GPT` architecture has `LayerNorm` at the input side (before `QKV Gemm`) and the residual connection
is taken from the input of that `LayerNorm`. In TE this can be achieved by setting the following
arguments in the `TransformerLayer` API.

.. code-block:: python

  transformer_engine.pytorch.TransformerLayer(
          ...,
          ...,
          apply_residual_connection_post_layernorm=False,
          output_layernorm=False,
          layer_type="encoder",
  )

BERT
^^^^

`BERT` architecture has `LayerNorm` at the output side (after the final `BiasDropoutAdd`) and the
residual connection is taken from the output of that `LayerNorm`. In TE this can be achieved by
setting the following arguments in the `TransformerLayer` API.

.. code-block:: python

  transformer_engine.pytorch.TransformerLayer(
          ...,
          ...,
          apply_residual_connection_post_layernorm=True,
          output_layernorm=True,
          layer_type="encoder",
  )

T5
^^

`T5` architecture has an additional `cross-attention` + `BiasDropoutAdd` + `LayerNorm` block before
the `MLP` layer. In TE this can be added by setting the `layer_type` to `decoder` in the
`TransformerLayer` API.

.. code-block:: python

  transformer_engine.pytorch.TransformerLayer(
          ...,
          ...,
          layer_type="decoder",
  )

Integrations
==================

Transformer Engine has been integrated with popular LLM frameworks such as:

* `DeepSpeed <https://github.com/microsoft/DeepSpeed/pull/3731>`_
* `Hugging Face Accelerate <https://github.com/huggingface/accelerate/releases/tag/v0.17.0>`_
* `Lightning <https://github.com/Lightning-AI/lightning/issues/17172>`_
* `MosaicML Composer <https://github.com/mosaicml/composer/releases/tag/v0.13.1>`_
* `NVIDIA JAX Toolbox <https://github.com/NVIDIA/JAX-Toolbox>`_
* `NVIDIA Megatron-LM <https://github.com/NVIDIA/Megatron-LM>`_
* `NVIDIA NeMo <https://github.com/NVIDIA/NeMo>`_
* `Amazon SageMaker Model Parallel Library <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel.html>`_ - Coming soon!
* `Colossal-AI <https://github.com/hpcaitech/ColossalAI>`_ - Coming soon!
* `PeriFlow <https://github.com/friendliai/periflow-python-sdk>`_ - Coming soon!


Contributing
==================

We welcome contributions to Transformer Engine! To contribute to Transformer Engine and make pull requests,
follow the guidelines outlined in the `<CONTRIBUTING.rst>`_ guide.

Papers
==================

* `Attention original paper <https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`_
* `Megatron-LM tensor parallel <https://arxiv.org/pdf/1909.08053.pdf>`_
* `Megatron-LM sequence parallel <https://arxiv.org/pdf/2205.05198.pdf>`_
* `FP8 Formats for Deep Learning <https://arxiv.org/abs/2209.05433>`_

Videos
==================

* `FP8 Training with Transformer Engine <https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51393>`_
* `FP8 for Deep Learning <https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s52166/>`_
* `Inside the Hopper Architecture <https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s42663/>`_

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0

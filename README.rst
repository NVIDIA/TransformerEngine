..
    Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

|License|

Transformer Engine
==================

.. overview-begin-marker-do-not-remove

Transformer Engine (TE) is a library for accelerating Transformer models on NVIDIA GPUs, including
using 8-bit floating point (FP8) precision on Hopper GPUs, to provide better performance with lower
memory utilization in both training and inference. TE provides a collection of highly optimized
building blocks for popular Transformer architectures and an automatic mixed precision-like API that
can be used seamlessly with your PyTorch code. TE also includes a framework agnostic C++ API that
can be integrated with other deep learning libraries to enable FP8 support for Transformers.

As the number of parameters in Transformer models continues to grow, training and inference for
architectures such as BERT, GPT and T5 becomes very memory and compute intensive. Most deep learning
frameworks train with FP32 by default. This is not essential, however, to achieve full accuracy for
many deep learning models. Using mixed-precision training, which combines single-precision (FP32)
with lower precision (e.g. FP16) format when training a model, results in significant speedups with
minimal differences in accuracy as compared to FP32 training. With the introduction of Hopper GPU
architecture FP8 precision was introduced, which offers improved performance over FP16 with no
degradation in accuracy. Although all major deep learning frameworks support FP16, FP8 support is
not available today.

TE addresses the problem of FP8 support by providing APIs that integrate with popular Large Language
Model (LLM) libraries. It provides python layer (initially supporting pyTorch, with support for more
frameworks in the future) consisting of modules to easily build Transformer layer as well as
framework agnostic library in C++ including structs and kernels needed for FP8 support. Modules
provided by TE internally maintain scaling factors and other values needed for FP8 training, greatly
simplifying for the users.

Transformer Engine in action:

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

  # Create FP8 recipe. Note: All input args are optional.
  fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)

  # Enables autocasting for the forward pass
  with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
      out = model(inp)

  loss = out.sum()
  loss.backward()

Highlights
----------

* Easy-to-use pyTorch modules enabling building of the Transformer layers with FP8 support on H100
  GPUs.
* Optimizations (e.g. fused kernels) for Transformer models across all precisions and NVIDIA GPU
  architecures.

.. overview-end-marker-do-not-remove

Installation
------------

In the NGC container
^^^^^^^^^^^^^^^^^^^^

Transformer Engine comes preinstalled in the pyTorch container on
`NVIDIA GPU Cloud <https://ngc.nvidia.com>`_ (versions 22.09 and later).

From source
^^^^^^^^^^^

Clone the repository and inside it type:

.. code-block:: bash

  pip install .

User Guide
----------

For examples, tutorials and API reference please refer to the
`User Guide <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html>`_.

Transformer Architectures
-------------------------

While the more granular modules in Transformer Engine allow building any Transformer architecture,
the `TransformerLayer` API of Transformer Engine is flexible enough to build multiple major
variations of Transformers.

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

Contributing to Transformer Engine
----------------------------------

We welcome contributions to Transformer Engine. To contribute to TE and make pull requests,
follow the guidelines outlined in the `<CONTRIBUTING.rst>`_ document.

Useful Links
------------

* `Attention original paper <https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`_

* `Megatron-LM tensor parallel <https://arxiv.org/pdf/1909.08053.pdf>`_

* `Megatron-LM sequence parallel <https://arxiv.org/pdf/2205.05198.pdf>`_

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0

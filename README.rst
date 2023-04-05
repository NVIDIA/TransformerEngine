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
can be used seamlessly with your own framework-specific code. TE also includes a framework agnostic
C++ API that can be integrated with other deep learning libraries to enable FP8 support for Transformers.

As the number of parameters in Transformer models continues to grow, training and inference for
architectures such as BERT, GPT and T5 become very memory and compute intensive. Most deep learning
frameworks train with FP32 by default. This is not essential, however, to achieve full accuracy for
many deep learning models. Using mixed-precision training, which combines single-precision (FP32)
with lower precision (e.g. FP16) format when training a model, results in significant speedups with
minimal differences in accuracy as compared to FP32 training. With the introduction of Hopper GPU
architecture FP8 precision was introduced, which offers improved performance over FP16 with no
degradation in accuracy. Although all major deep learning frameworks support FP16, FP8 support is
not available today.

TE addresses the problem of FP8 support by providing APIs that integrate with popular Large Language
Model (LLM) libraries. It provides python layer consisting of modules to easily build Transformer
layer as well as framework agnostic library in C++ including structs and kernels needed for FP8 support.
Modules provided by TE internally maintain scaling factors and other values needed for FP8 training, greatly
simplifying for the users.


Examples
--------

pyTorch
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

.. code-block:: python

  import jax
  import jax.numpy as jnp
  import transformer_engine.jax as te
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
      model = te.DenseGeneral(features=HIDDEN)

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
        # Update FP8 metas
        other_variables = te.update_fp8_metas(other_grads)

TensorFlow
^^^^^^^^^^

.. code-block:: python

  import tensorflow as tf
  import transformer_engine.tensorflow as te
  from transformer_engine.common import recipe
  
  # Set dimensions.
  in_features = 768
  out_features = 3072
  hidden_size = 2048
  
  # Initialize model and inputs.
  model = te.Dense(out_features, use_bias=True)
  inp = tf.random.normal((hidden_size, in_features))
  
  optimizer = tf.keras.optimizers.Adam(0.001)
  
  # Create FP8 recipe. Note: All input args are optional.
  fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.E4M3)
  
  with tf.GradientTape(persistent=True) as tape:
      # Enables autocasting for the forward pass
      with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
          out = model(inp)
      loss = tf.reduce_sum(out)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))


Highlights
----------

* Easy-to-use modules enabling building of the Transformer layers with FP8 support
  on H100 GPUs.
* Optimizations (e.g. fused kernels) for Transformer models across all precisions and NVIDIA GPU
  architectures.

.. overview-end-marker-do-not-remove

Installation
------------

In the NGC container
^^^^^^^^^^^^^^^^^^^^

Transformer Engine comes preinstalled in the pyTorch container on
`NVIDIA GPU Cloud <https://ngc.nvidia.com>`_ (versions 22.09 and later).

From source
^^^^^^^^^^^

First, install the prequisites.

.. code-block:: bash

  apt-get install ninja-build pybind11-dev

Clone the repository and inside it type:

.. code-block:: bash

  NVTE_FRAMEWORK=all pip install .     # Building with all frameworks.
  NVTE_FRAMEWORK=pytorch pip install . # Building with pyTorch only.
  NVTE_FRAMEWORK=jax pip install .     # Building with JAX only.

You can also specify which framework bindings to build. The default is pytorch only.

.. code-block:: bash

  # Build with TensorFlow bindings
  NVTE_FRAMEWORK=tensorflow pip install .

  # Build with Jax bindings
  NVTE_FRAMEWORK=jax pip install .

  # Build with all bindings (Pytorch, TF, Jax)
  NVTE_FRAMEWORK=all pip install .

User Guide and Examples
-----------------------

For examples, tutorials and API reference please refer to:

* `User Guide <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html>`_ for the last release.
* `Development User Guide <https://nvidia.github.io/TransformerEngine/>`_ for the development version.
* `Examples <https://github.com/NVIDIA/TransformerEngine/tree/main/examples>`_.

Transformer Architectures
-------------------------

While the more granular modules in Transformer Engine allow building any Transformer architecture,
the `TransformerLayer` API of Transformer Engine is flexible enough to build multiple major
variations of Transformers.

NOTE: For simplicity, we only show pyTorch examples below. For the usage of `TransformerLayer`
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

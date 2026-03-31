..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Introduction
===================================

Transformer Engine accelerates deep learning on NVIDIA GPUs in several ways,
with low precision training being one of the most important.
This chapter introduces mixed precision training and FP8 support.


Training in BF16/FP16
---------------------

Deep learning traditionally uses 32-bit floating-point (FP32) numbers.
NVIDIA GPUs support lower precision formats—FP16 since Pascal, BF16 since Ampere—which offer higher throughput and lower memory usage.
Let's compare these formats.

.. raw:: html
   :file: img/fp_formats_comparison.svg

*Figure 1: Comparison of FP32, BF16, and FP16 floating-point formats showing bit allocation for sign, exponent, and mantissa.*

The key differences between these formats are:

* **FP32** (32 bits total): 1 sign bit + 8 exponent bits + 23 mantissa bits – standard single-precision format
* **BF16** (16 bits total): 1 sign bit + 8 exponent bits + 7 mantissa bits – maintains FP32's exponent range but has reduced precision
* **FP16** (16 bits total): 1 sign bit + 5 exponent bits + 10 mantissa bits – reduced range but higher precision than BF16

BF16's advantage is that it shares the same exponent range as FP32, 
making it easier to convert between the two formats without overflow/underflow issues. 
FP16 offers better precision for smaller values but has a limited dynamic range,
which results in the need to perform loss scaling to avoid overflow/underflow—see `this paper on loss scaling <https://arxiv.org/pdf/1710.03740>`__ for more details.

**Mixed precision**

Not all operations should be run in reduced precision to preserve accuracy.
Modern deep learning frameworks use *mixed precision training*, 
where different operations use different precisions based on their numerical properties:

* Matrix multiplications are compute-heavy and remain numerically stable at lower precision, making them ideal candidates for acceleration.
* Operations like layer normalization and softmax can work with low precision inputs and outputs, but may use high precision internally or for their weights.
* Operations like loss computation and exponentiation need high precision throughout.

**Master weights**

Another consideration in mixed precision training is how to store the model weights.
Lower precision formats like FP16 and BF16 have limited representational granularity, 
which becomes problematic during gradient updates. 
When a small gradient is added to a not so small weight stored in low precision, 
the result may round back to the original value if the update falls below the format's precision threshold.
Moreover, some elements of the gradient itself can be too small to be represented in low precision, 
especially after the accumulation from multiple GPUs in the data parallel training setting.

The solution is to maintain *master weights* in FP32. 
During training, weights are cast to lower precision for forward and backward passes, 
but the gradient updates are applied to the full-precision master copy.
This ensures that even small gradients accumulate correctly over time.

There are two common software approaches to storing master weights:

* *In the optimizer*: 
  The model holds low-precision weights, 
  while the optimizer maintains FP32 copies alongside momentum and other state. 
  During each step, 
  the optimizer updates its FP32 copy and casts the result back to the model's low-precision weights. 
  
  This approach makes it easier to shard master weights together with other optimizer state, for example in ZeRO optimizer.

  Since the casting happens only during the optimizer step, this approach is also faster when optimizer runs less frequently than the model, e.g. when performing gradient accumulation or pipeline parallel training.

* *In the model*: 
  The model stores weights directly in FP32, 
  and they are cast to lower precision on-the-fly during forward and backward passes. 
  This approach works seamlessly with any standard optimizer, requiring no special support.

.. raw:: html
   :file: img/master_weights_approaches.svg

*Figure 2: Three approaches to weight storage—low precision only (no master weights), master weights stored in the model, and master weights stored in the optimizer.*

.. tabs::

   .. tab:: PyTorch

      The PyTorch API of Transformer Engine provides several mechanisms to control precision:
      
      * **Weight precision**: Use the ``params_dtype`` argument in any TE layer constructor.
      * **Computation precision**: Use the ``torch.autocast`` context manager. When enabled, inputs are cast to the autocast dtype before computation.
      * **Input dtype**: When ``torch.autocast`` is not used, the input tensor's dtype determines the computation precision. In this case, inputs and parameters must have matching dtypes.

      .. literalinclude:: bf16_fp16_training_pytorch.py
         :language: python
         :start-after: # START_BF16_FP16_TRAINING
         :end-before: # END_BF16_FP16_TRAINING


   .. tab:: JAX

      The JAX API of Transformer Engine provides two mechanisms to control precision:
      
      * **Weight precision**: Use the ``dtype`` argument in any TE layer constructor.
      * **Computation precision**: Determined by the dtype of the input tensor.
      
      For training with master weights in FP32 and computation in BF16, 
      cast the input tensor to BF16 before passing it to the layer.

      .. literalinclude:: bf16_fp16_training_jax.py
         :language: python
         :start-after: # START_BF16_FP16_TRAINING
         :end-before: # END_BF16_FP16_TRAINING
      


Lower precisions
----------------

Transformer Engine's primary feature is supporting even lower precision than BF16/FP16, such as FP8, MXFP8, NVFP4, etc.
The logic of these precisions is more complicated than the logic of BF16/FP16 – they require scaling factors to
properly represent the full range of values in the tensor. Sometimes it is one scaling factor per tensor,
sometimes it is one scaling factor per block of values. A precision format combined with the logic for training
is called **a recipe**.

In this section we present common logic for all the recipes. Each one of them is described in more detail in a separate section later.
Let's now see how we can train in lower precisions in supported frameworks.

.. tabs::

   .. tab:: PyTorch

      The PyTorch API of Transformer Engine provides an ``autocast`` context manager to control precision.
      It's similar to the ``torch.autocast`` context manager, but tailored for low precision training.
      The most important argument is the ``recipe`` argument, which accepts objects inheriting from
      :class:`~transformer_engine.common.recipe.Recipe`.

      Forward computations need to be performed inside the ``autocast`` context manager,
      while the ``.backward()`` call should be outside of it (it inherits the setting from the
      corresponding forward pass).

      Here is a basic example:

      .. raw:: html

         <div style="background: #f0f4f8; border-left: 3px solid #5c7cfa; padding: 6px 12px; font-size: 13px; color: #495057; margin-bottom: 0; border-radius: 4px 4px 0 0;">
            Needs to be run on SM89+ (Ada or newer)
         </div>

      .. literalinclude:: autocast_pytorch.py
         :language: python
         :start-after: # START_AUTOCAST_BASIC
         :end-before: # END_AUTOCAST_BASIC

      You can use multiple recipes in the same model in the following ways:

      **Sequential contexts** – apply different recipes to different parts of your model:

      .. raw:: html

         <div style="background: #f0f4f8; border-left: 3px solid #5c7cfa; padding: 6px 12px; font-size: 13px; color: #495057; margin-bottom: 0; border-radius: 4px 4px 0 0;">
            Needs to be run on SM89+ (Ada or newer)
         </div>

      .. literalinclude:: autocast_pytorch.py
         :language: python
         :start-after: # START_AUTOCAST_SEQUENTIAL
         :end-before: # END_AUTOCAST_SEQUENTIAL

      **Nested contexts** – the inner context overrides the outer one for its scope:

      .. raw:: html

         <div style="background: #f0f4f8; border-left: 3px solid #5c7cfa; padding: 6px 12px; font-size: 13px; color: #495057; margin-bottom: 0; border-radius: 4px 4px 0 0;">
            Needs to be run on SM89+ (Ada or newer)
         </div>

      .. literalinclude:: autocast_pytorch.py
         :language: python
         :start-after: # START_AUTOCAST_NESTED
         :end-before: # END_AUTOCAST_NESTED
      

   .. tab:: JAX

      The JAX API of Transformer Engine provides an ``autocast`` context manager similar to PyTorch.
      The key difference is that in JAX, model initialization must happen inside the ``autocast`` context
      to properly capture quantization metadata in the parameter tree.

      Here is a basic example:

      .. raw:: html

         <div style="background: #f0f4f8; border-left: 3px solid #5c7cfa; padding: 6px 12px; font-size: 13px; color: #495057; margin-bottom: 0; border-radius: 4px 4px 0 0;">
            Needs to be run on SM89+ (Ada or newer)
         </div>

      .. literalinclude:: autocast_jax.py
         :language: python
         :start-after: # START_AUTOCAST_BASIC
         :end-before: # END_AUTOCAST_BASIC

      You can use multiple recipes in the same model in the following ways:

      **Sequential contexts** – apply different recipes to different parts of your model:

      .. raw:: html

         <div style="background: #f0f4f8; border-left: 3px solid #5c7cfa; padding: 6px 12px; font-size: 13px; color: #495057; margin-bottom: 0; border-radius: 4px 4px 0 0;">
            Needs to be run on SM89+ (Ada or newer)
         </div>

      .. literalinclude:: autocast_jax.py
         :language: python
         :start-after: # START_AUTOCAST_SEQUENTIAL
         :end-before: # END_AUTOCAST_SEQUENTIAL

      **Nested contexts** – the inner context overrides the outer one for its scope:

      .. raw:: html

         <div style="background: #f0f4f8; border-left: 3px solid #5c7cfa; padding: 6px 12px; font-size: 13px; color: #495057; margin-bottom: 0; border-radius: 4px 4px 0 0;">
            Needs to be run on SM89+ (Ada or newer)
         </div>

      .. literalinclude:: autocast_jax.py
         :language: python
         :start-after: # START_AUTOCAST_NESTED
         :end-before: # END_AUTOCAST_NESTED

      .. note::
         Python context managers like ``autocast`` may interact unexpectedly with JAX's JIT compilation.
         For finer-grained control, consider passing the recipe directly to TE modules instead.
         See the `TE JAX Integration notebook <https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/te_jax_integration.ipynb>`_
         for details.

**Mixed precision with 8- or 4-bit precisions**

From now on, we will refer to FP8/MXFP8/NVFP4 etc. as *low precision*
and to FP32/BF16/FP16 as *high precision*. This terminology will be
used throughout the rest of the documentation.

Not all operations run in low precision:

- **Linear operations**: run in low precision.
- **Attention computations**: run in high precision by default (some recipes allow low precision as an option).
- **Other operations** (layer normalization, softmax, etc.): run in high precision.

Within high-precision operations, there are two categories:

- **Configurable precision**: most operations run in parameter precision (FP32/BF16/FP16) or the precision specified by ``torch.autocast``.
- **Fixed FP32 precision**: some operations, or parts of operations—such as the division in layernorm—always run in FP32, regardless of other settings.

.. raw:: html
   :file: img/mixed_precision_operations.svg

*Figure 3: Default precision of operations in a TransformerLayer forward pass. Only linear operations are in lower precision. Dot product attention is shown as three separate operations (QK^T, Softmax, Scores * V), though in practice these may be fused into a single kernel.*

**Linear layer data flow**

Let's see how data flow of a linear layer works by default on a single H100 GPU with FP8 precision:

H100 (Hopper) architecture natively supports FP8 Matrix Multiplication only in **TN** layout (Transpose-NoTranspose), 
so GEMM with tensors ``A`` and ``B`` returns ``B * A^T``.

*Forward pass*

* Input is quantized to FP8 – both ``input`` and ``input^T`` quantized versions are created.
* Weights are stored in high precision and quantized to low precision before the GEMM – both ``weight`` and ``weight^T`` quantized versions are created.
* FP8 GEMM with layout **TN** is run with ``weight`` and ``input`` tensors,
* Outputs – ``input * weight^T`` tensor – are returned in high precision.

*Backward pass*

* Output gradients are quantized to FP8 – both ``output_grad`` and ``output_grad^T`` quantized versions are created.
* FP8 GEMM with layout **TN** is performed with ``weight^T`` and ``output_grad`` tensors to compute input gradients.
* FP8 GEMM with layout **TN** is performed with ``input^T`` and ``output_grad^T`` tensors to compute weight gradients.
* Input gradients – ``output_grad * weight`` tensor – are returned in high precision.
* Weight gradients – ``output_grad^T * input`` tensor – are returned in high precision.


.. raw:: html
   :file: img/fp8_linear_flow.svg

*Figure 4: Forward pass of a Linear layer with low precision data flow.*

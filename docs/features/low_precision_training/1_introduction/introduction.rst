..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Introduction
===================================

The main feature of Transformer Engine is enabling low precision training. 
While the standard floating-point format on CPUs is FP32, 
NVIDIA GPUs support lower precision formats designed to accelerate training while maintaining accuracy. 
In this chapter, we introduce the general concepts of low precision training support in Transformer Engine.


Training in BF16/FP16
---------------------

Let's remind how training in BF16/FP16 works. There are 2 components to it:


1. Choosing which operations should be performed in lower precision.
    For example matrix multiplies, convolutions and normalization layers are marked as safe, 
    while other operations like activation functions are marked as requiring FP32.
2. Dynamic loss scaling for FP16.

Transformer Engine builds upon this observations, when applying even lower precisions.

Moving to lower precisions – recipes
------------------------------------

Similarly to BF16/FP16, not all operations can be performed in 8 bit precision or lower.
Transformer Engine primarly supports low precision GEMMs and all the other operations - like layernorm or activation functions - are run in higher precision.

Let's look into default Linear layer forward pass using FP8 precision on the image below. We can see that:

.. raw:: html
   :file: img/fp8_linear_flow.svg

1. Weights are stored in higher precision and cast to FP8 before the FP8 GEMM.
2. Input is cast to FP8 before the FP8 GEMM.
3. Output is in higher precision.
4. Output gradient is casted into FP8.
5. Weight and input transposes are in FP8.
6. Gradient of weight and input are returned in higher precision.

Notice that Transformer Engine does not set BF16/FP16 as default high precision format.
One can achive it by specifying ``params_dtype`` argument in TE modules constructor.

Handling transposes
-------------------

The forward and backward passes of linear layers involve multiple matrix multiplications with different reduction dimensions. Blackwell Tensor Cores require MXFP8 data to be "consecutive" over the reduction dimension, so MXFP8 training uses non-transposed and transposed MXFP8 tensors at different points. However, while transposing FP8 data is numerically trivial, transposing MXFP8 data requires requantization.

To avoid loss of precision connected with this double quantization, Transformer Engine creates both regular and transposed copies of the tensor from the original high precision input.

.. raw:: html
   :file: img/linear_mxfp8.svg


Memory usage
------------

As we can see in the figure 1, Transformer Engine stores parameters in high precision and casts them to FP8 before the GEMM.
Optimizer step is applied to high precision parameters - applying optimizer to FP8 directly can potentially lead
to accuracy degradation, thus this is not the default behavior.

One can see that low precision training will not decrease memory usage by default - 
we need to store high precision parameters of the model anyways. Moreover, TE by default
stores quantized weight transpose for backward pass - this adds additional memory usage.
So one need not to be shocked, that fp8 training not always reduces memory - the primary 
objective is speed.

Note that for some use cases, this is not optimal. Depending on framework, Transformer Engine 
has some mechanisms of using only low precision parameters and weight caching. Refer to 
Framework API for more information.

Fused layers
------------

Quantization of input before GEMM can take non-negligible time. Assume that there is Layer Norm and 
after that there is Linear layer. Layer norm will return output in FP32, 
which will be later quantized by Linear layer.
Transformer Engine provides LayerNormLinear layer, which contains layernorm returning quantized output.

.. raw:: html
   :file: img/fused_layers.svg



Distributed training
--------------------


Transformer Engine supports distributed training with low precision. Extent and ways of this
support varies depending of the framework and precision. For example all-gather
for fp8 is supported for PyTorch, but it may not be the case for the most recently added precisions.

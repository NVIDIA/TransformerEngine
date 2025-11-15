..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

FP8 Current Scaling
===================================

FP8 current scaling is the simplest scaling mode for FP8. 
Let's start from what the FP8 data type in fact is.


FP8 data type
-------------

The FP8 datatype supported by H100 is actually 2 distinct datatypes, useful in different parts of the training of neural networks:

* E4M3 - it consists of 1 sign bit, 4 exponent bits and 3 bits of mantissa. It can store values up to +/-448 and `nan`.
* E5M2 - it consists of 1 sign bit, 5 exponent bits and 2 bits of mantissa. It can store values up to +/-57344, +/- `inf` and `nan`. The tradeoff of the increased dynamic range is lower precision of the stored values.

.. figure:: ../../examples/fp8_formats.png
   :align: center
   :width: 60%

   **Figure 1:** Structure of the floating point datatypes. All of the values shown (in FP16, BF16, FP8 E4M3 and FP8 E5M2) are the closest representations of value 0.3952.

During training neural networks both of these types may be utilized. 
Typically forward activations and weights require more precision, so E4M3 datatype is best used during forward pass. 
In the backward pass, however, gradients flowing through the network typically are less susceptible to the loss of precision, but require higher dynamic range. 
Therefore they are best stored using E5M2 data format. 


Scaling factors
---------------

8-bit precision may be not enought to capture dynamic range of some tensors. Thus they are indended to be used with scaling factors 
- proper representation of tensor A in float8 precision will be 

.. math::
    A = A_{fp8} \cdot s

where :math:`A_{fp8}` is fp8 tensor and :math:`s` is scalar 32bit float.

Let's look more closely how quantization to FP8 with scaling factor in implemented
current scaling recipe. There are two main steps:

1. Computation of absolute maximum value of the tensor.
2. Applying the scaling factor being reverse of amax to the tensor and 
then quantizing it to FP8. 

This implies that there will not be overflow.

.. raw:: html
   :file: img/fp8_cast_process.svg

Hardware support
----------------

The Ada architecture introduced FP8 support in Tensor Cores, enabling efficient low-precision computation. 
Tensor Cores support every combination of E4M3 and E5M2 formats as inputs, allowing flexible precision choices for different operands.
The inputs to an FP8 Tensor Core operation consist of chunks of FP8 tensors along with their corresponding scaling factors.
The Tensor Core performs the matrix multiplication in FP8 precision and produces output in higher precision (FP16, BF16, or FP32).

.. raw:: html
   :file: img/fp8_tensor_core.svg

The diagram above illustrates how FP8 Tensor Cores process two input tensors (A and B), each with their own scaling factors, 
and perform matrix multiplication to produce a higher-precision output.


Transpose handling
------------------

As mentioned in introducion, backward pass often needs transpose of the tensor.

For Blackwell and later architectures, Tensor Cores have hardware support 
for loading transposed data. Thus no transpose computation is needed for backward pass.

For Hopper and Ada, there are 2 ways of handling the tranpose:

1. Computation of tensor and its transpose during FP8 cast, saving only tranpose for backward.
2. Quantization of tensor computes only non-transposed tensor, which is saved for backward and tranposed in backward.

Option 1 is more optimal in most situations, but there are some cases - for example 
in distributed training, when idea 2 is implemented.

.. raw:: html
   :file: img/transpose_handling.svg

The diagram above illustrates the different transpose handling strategies across GPU architectures. 
Blackwell benefits from hardware-accelerated transpose support, while Hopper and Ada require software-based approaches with different trade-offs.

Distributed training 
--------------------

No amax synchronization is performed in most cases by defualt. Exception
is activation amax with sequence parallel in PyTorch.

Supported devices
-------------

Ada and later
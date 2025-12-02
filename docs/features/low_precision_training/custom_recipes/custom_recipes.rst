..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Custom recipes
===================================

.. warning::
    Custom recipe is an experimental feature.

Introduction
------------

The Custom Recipe feature allows users to implement their own recipes and run them together with 
the Transformer Engine layers. 
As the recipes already provided with the TE are optimized for performance,
it is expected that the custom recipes will not be as performant.
These are aimed to be used in experimental and testing purposes.
Before implementing a custom recipe, 
we show how the recipes provided by the TE are implemented.


Quantizer API
-------------

**Recipe**

The core of the custom recipe system is the interaction between the Recipe, the Quantizer, and the Quantized Tensor classes.
The ``Recipe`` class is the base class for all recipes - like for example ``Float8CurrentScaling``
or ``MXFP8BlockScaling``. The recipe contains global state - like for example recipe setting or 
amax buffers for Delayed Scaling recipe. This object is provided to the ``autocast`` context manager as an argument.
We will refer to this recipe as *active recipe* in the following text.


.. raw:: html
   :file: img/recipe_creates_quantizer.svg

*Figure 1. Recipe creates Quantizer instances for each GEMM input/output.*

**Quantizer**

Each layer consists of some number of GEMMs. For a Linear layer there is 1 GEMM and for a LayerNormMLP layer there are 2 GEMMs.
Each of the GEMMs is related to 6 tensors: input, weight, output, input gradient, weight gradient and output gradient.
For each of these tensors - except the weight gradient tensor - in each of the layers, the object of active recipe creates a ``Quantizer``.
If no recipe is active - the run is in high precision - then ``None`` is used as Quantizers.

**Quantized Tensor**

Quantizer is responsible for creating low precision tensor from the high precision one.
For example, the ``Float8Quantizer`` class is responsible for creating ``Float8Tensor``.
Sometimes Quantizer stores the data - like in Delayed Scaling recipe, where it stores the amax tensor and scaling factor.


.. raw:: html
   :file: img/quantizer_creates_tensor.svg

*Figure 2. Quantizer converts high-precision tensor to QuantizedTensor.*

**Optimization**

Transformer Engine can perform many optimizations with the Quantizer objects. Here are two examples:

* If in LayerNormLinear the GEMM uses a Quantizer, then the layer norm can use fused kernel to return proper ``QuantizedTensor`` object.
* If ``QuantizedTensor`` object is all-gathered and all gather for the recipe is implemented, then the Quantizer can all-gather the quantized tensor. 
  If this is not the case, the tensor is dequantized, all-gather is performed, and then quantized again.

Custom Recipes API 
------------------

.. tabs::

    .. tab:: PyTorch

        **Creating a Custom Recipe**

        To create a custom recipe, use the ``CustomRecipe`` class with a ``qfactory`` parameter.
        The factory is a callable that receives a ``role`` string identifying which tensor needs a quantizer,
        and returns an appropriate ``Quantizer`` instance.

        **Tensor Roles**

        The ``role`` parameter uses ``linear_*`` naming for all layer types:

        * Forward pass: ``linear_input``, ``linear_weight``, ``linear_output``
        * Backward pass: ``linear_grad_output``, ``linear_grad_input``

        **Implementing a Custom Quantizer**

        To implement a custom quantizer, inherit from the ``Quantizer`` base class 
        and implement the ``quantize_impl`` method:

        * ``__init__(self, *, rowwise: bool, columnwise: bool)``: Initialize with usage flags
        * ``quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor``: Convert high-precision tensor to quantized format

        The ``quantize_impl`` method receives a high-precision tensor and must return 
        a ``QuantizedTensor`` (or subclass) containing the quantized data and any scaling factors.

        **Implementing a Custom Tensor**

        Custom quantizers must return a ``QuantizedTensorStorage`` subclass from ``quantize_impl``.
        To trigger the custom GEMM dispatch path, the tensor must have a ``custom`` property 
        that returns ``True``.

        The ``custom_gemm`` function reads the following attributes from the tensor storage 
        and passes them to the quantizer's ``qgemm`` method:

        * ``data``, ``scale``: Rowwise quantized data and scaling factors (used in FPROP and DGRAD)
        * ``data_t``, ``scale_t``: Columnwise quantized data and scaling factors (used in DGRAD and WGRAD)
        * ``dtype``: The original high-precision dtype
        * ``original_shape``: Shape of the original tensor (used to reshape 3D outputs)

        .. code-block:: python

            @dataclasses.dataclass
            class MyCustomTensor(QuantizedTensorStorage):
                custom: bool = True  # Triggers custom GEMM dispatch
                
                data: torch.Tensor = None
                data_t: torch.Tensor = None
                scale: torch.Tensor = None
                scale_t: torch.Tensor = None
                dtype: torch.dtype = None
                original_shape: tuple = None

        **Implementing Custom GEMM**

        When tensors are marked as custom (``custom = True``), the GEMM dispatch 
        routes to the quantizer's ``qgemm`` method instead of the optimized C++ kernels.

        The ``qgemm`` method signature:

        .. code-block:: python

            def qgemm(
                self,
                qx: torch.Tensor,           # Quantized data (from A.data or A.data_t)
                qw: torch.Tensor,           # Quantized data (from B.data or B.data_t)
                m_params: MMParams,         # Matrix multiplication parameters
                out_dtype: torch.dtype,     # Output data type
                sx: torch.Tensor,           # Scale (from A.scale or A.scale_t)
                sw: torch.Tensor,           # Scale (from B.scale or B.scale_t)
                bias: torch.Tensor | None,  # Optional bias (only for FPROP)
                gemm_type: GEMMType,        # FPROP, DGRAD, or WGRAD
                qresult_x: QuantizedTensorStorage,  # Full storage object A
                qresult_w: QuantizedTensorStorage,  # Full storage object B
            ) -> torch.Tensor:

        The ``gemm_type`` parameter indicates which GEMM is being performed and which 
        tensor attributes are used:

        * ``GEMMType.FPROP``: ``A.data``, ``A.scale``, ``B.data``, ``B.scale``
        * ``GEMMType.DGRAD``: ``A.data``, ``A.scale``, ``B.data_t``, ``B.scale_t``
        * ``GEMMType.WGRAD``: ``A.data_t``, ``A.scale_t``, ``B.data_t``, ``B.scale_t``

    .. tab:: JAX

        Custom recipes are currently not supported in JAX.

**Limitations of Custom Recipes**

Please note that recipes provided by the TE support many optimizations and fusions,
and are polished for performance. We do not expose the possibility of 
most of the optimizations and fusions with custom recipes. 
Thus, custom recipes are not expected to be as performant as the ones provided by the TE.


Examples
--------

We showcase two examples of custom recipes that demonstrate different use cases.

**1. Mixed Precision Recipe (FP8 Forward / MXFP8 Backward)**

This example uses ``CustomRecipe`` to mix standard TE quantizers: FP8 current scaling for forward pass 
and MXFP8 for backward pass. 

.. note::
    Standard quantizers like ``Float8CurrentScalingQuantizer`` and ``MXFP8Quantizer`` return 
    standard TE tensor types (``Float8Tensor``, ``MXFP8Tensor``). These tensors have ``custom = False``,
    so the **optimized GEMM kernels are used** - ``custom_gemm`` is NOT invoked.
    This approach allows mixing different quantization strategies without performance penalty.

.. tabs::

    .. tab:: PyTorch

        .. literalinclude:: pytorch_mixed_precision_example.py
            :language: python
            :start-after: # START_MIXED_PRECISION_EXAMPLE
            :end-before: # END_MIXED_PRECISION_EXAMPLE

    .. tab:: JAX

        Custom recipes are currently not supported in JAX.

**2. Custom Quantization Logic with Custom GEMM (Int6)**

This example demonstrates a fully custom quantizer that implements its own quantization format 
and GEMM logic. The custom tensor storage has ``custom = True``, which triggers the custom GEMM 
dispatch path - all matrix multiplications are routed to the quantizer's ``qgemm()`` method.

.. warning::
    This approach executes GEMM in Python and is significantly slower than standard recipes.
    Use only for experimental and testing purposes.

.. tabs::

    .. tab:: PyTorch

        .. literalinclude:: pytorch_int6_example.py
            :language: python
            :start-after: # START_INT6_EXAMPLE
            :end-before: # END_INT6_EXAMPLE

    .. tab:: JAX

        Custom recipes are currently not supported in JAX.

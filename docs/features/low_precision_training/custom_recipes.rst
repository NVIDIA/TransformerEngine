..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Custom recipes
===================================

.. warning::
    **EXPERIMENTAL**: Custom recipe is experimental, still under active development,
    and the API is subject to change without notice. Use at your own risk.

Custom recipes allow you to implement your own quantization strategies while still
benefiting from Transformer Engine's infrastructure. This is useful for experimenting 
with novel quantization techniques, mixing different formats for different tensors, 
or implementing research prototypes.

Quantizer factory
----------------------

A quantizer factory is a callable that returns a quantizer based on the semantic role of a tensor.
For linear layers, the following roles are used:

**Forward pass:**

* ``"linear_input"``: Input activation tensor
* ``"linear_weight"``: Weight tensor  
* ``"linear_output"``: Output activation tensor

**Backward pass:**

* ``"linear_grad_output"``: Gradient with respect to output
* ``"linear_grad_input"``: Gradient with respect to input

The factory should return a ``Quantizer`` instance for each role, or ``None`` to skip quantization.

Basic example
^^^^^^^^^^^^^

Here's a simple factory that uses FP8 E4M3 for forward pass and E5M2 for backward pass:

.. tabs::

    .. tab:: PyTorch

        .. code-block:: python

            from transformer_engine.pytorch import Float8CurrentScalingQuantizer
            import transformer_engine_torch as tex

            def my_quantizer_factory(role):
                # Forward pass: use E4M3
                if role in ("linear_input", "linear_weight", "linear_output"):
                    return Float8CurrentScalingQuantizer(
                        tex.DType.kFloat8E4M3, device="cuda"
                    )
                
                # Backward pass: use E5M2
                if role in ("linear_grad_output", "linear_grad_input"):
                    return Float8CurrentScalingQuantizer(
                        tex.DType.kFloat8E5M2, device="cuda"
                    )
                
                return None

    .. tab:: JAX

        .. code-block:: python

            from transformer_engine.jax import Float8CurrentScalingQuantizer
            import transformer_engine.jax.cpp_extensions as tex

            def my_quantizer_factory(role):
                # Forward pass: use E4M3
                if role in ("linear_input", "linear_weight", "linear_output"):
                    return Float8CurrentScalingQuantizer(
                        tex.DType.kFloat8E4M3
                    )
                
                # Backward pass: use E5M2
                if role in ("linear_grad_output", "linear_grad_input"):
                    return Float8CurrentScalingQuantizer(
                        tex.DType.kFloat8E5M2
                    )
                
                return None

Mixed precision example
^^^^^^^^^^^^^^^^^^^^^^^

You can selectively quantize only specific tensors:

.. code-block:: python

    def mixed_precision_factory(role):
        # Quantize activations but not weights
        if role == "linear_input":
            return Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E4M3, device="cuda"
            )
        
        # Don't quantize weights
        if role == "linear_weight":
            return None
        
        if role == "linear_output":
            return Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E4M3, device="cuda"
            )
        
        if role in ("linear_grad_output", "linear_grad_input"):
            return Float8CurrentScalingQuantizer(
                tex.DType.kFloat8E5M2, device="cuda"
            )
        
        return None

Using custom recipes
-------------------

Create a :class:`~transformer_engine.common.recipe.CustomRecipe` with your factory 
and use it with the appropriate autocast context manager:

.. tabs::

    .. tab:: PyTorch

        .. code-block:: python

            import torch
            import transformer_engine.pytorch as te
            from transformer_engine.common import recipe

            # Define model
            model = te.Linear(768, 3072, bias=True).cuda()
            inp = torch.randn(32, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)

            # Create custom recipe
            custom_recipe = recipe.CustomRecipe(qfactory=my_quantizer_factory)

            # Use with autocast
            with te.autocast(enabled=True, recipe=custom_recipe):
                output = model(inp)
            
            loss = output.sum()
            loss.backward()

    .. tab:: JAX

        .. code-block:: python

            import jax
            import jax.numpy as jnp
            import transformer_engine.jax as te
            from transformer_engine.common import recipe

            # Define model
            layer = te.flax.DenseGeneral(features=3072)
            
            # Create custom recipe
            custom_recipe = recipe.CustomRecipe(qfactory=my_quantizer_factory)
            
            # Initialize parameters
            key = jax.random.PRNGKey(0)
            inp = jax.random.normal(key, (32, 768))
            variables = layer.init(key, inp)
            
            # Use with autocast
            with te.autocast(enabled=True, recipe=custom_recipe):
                output = layer.apply(variables, inp)
            
            loss = jnp.sum(output)

Performance considerations
-------------------------

Custom recipes provide flexibility but have trade-offs:

**Advantages:**

* Full control over quantization strategy
* Can mix different formats and selectively quantize tensors
* Useful for research and prototyping

**Limitations:**

* No kernel fusion with other operations
* May have additional Python overhead
* Built-in recipes have more optimized implementations

**When to use:**

* Research and prototyping of new quantization methods
* Experimenting with mixed-precision strategies
* Domain-specific requirements

**When to use built-in recipes:**

* Production training requiring maximum performance
* When DelayedScaling, Float8CurrentScaling, etc. meet your needs

Creating custom quantizers
--------------------------

To implement your own quantizer, subclass :class:`~transformer_engine.pytorch.quantized_tensor.Quantizer`
and implement the ``quantize_impl`` method. See existing quantizers in ``transformer_engine/pytorch/tensor/``
for examples, or ``transformer_engine/pytorch/custom_recipes/quantization_nvfp4.py`` for a complete 
reference implementation.

..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Operation fuser API
===================

Motivation
----------

Transformer Engine relies heavily on operation fusion to achieve high
performance. A typical training workload involves many memory-bound
operations such as activation functions and normalization, so
replacing them with fused kernels can deliver a significant
performance benefit. This is especially true for low-precision
training (e.g. FP8 and FP4) because it involves extra cast operations.

Managing these fusions can be challenging because they differ based on
operation types, communication patterns, data types, and GPU
architectures. The most straightforward solution is to provide
monolithic modules like ``Linear``, ``LayerNormLinear``, or
``TransformerLayer``. These conform to the interface of a standard
PyTorch module, but can perform arbitrary fusions internally. These
hand-tuned implementations can achieve maximum performance, but they
tend to be complicated and difficult to modify.

As an alternative to this "top-down" design, TE exposes a "bottom-up"
operation-based API. The user constructs individual operations and
passes them into a fuser, resulting in the same fused kernels as the
monolithic modules. This approach is more flexible, making it easier
to support new model architectures or to experiment with fusions.

Description and usage
---------------------

Basic usage
^^^^^^^^^^^

At the most basic level, the operation fuser API involves two classes:

- ``FusibleOperation``: An abstract base class for tensor operations.
  Examples include ``Linear``, ``LayerNorm``, and ``AllReduce``. It is
  a subclass of ``torch.nn.Module``, so it can hold trainable
  parameters and can be called to perform the operation's forward
  pass.
- ``Sequential``: A container of modules in sequential order. It has a
  very similar interface as ``torch.nn.Sequential``. If it contains
  any ``FusibleOperation`` s, then it may attempt to fuse them in the
  forward and backward passes.

Thus, using the operation fuser simply involves constructing
``FusibleOperation`` s and passing them into a ``Sequential``.

.. code-block:: python

    import torch
    import transformer_engine.pytorch as te

    # Options
    hidden_size = 4096
    ffn_size = 28672
    batch_size = 16384

    # Construct operations and fuse
    mlp = te.ops.Sequential(
        te.ops.LayerNorm(hidden_size),
        te.ops.Linear(ffn_size, hidden_size),
        te.ops.SwiGLU(),
        te.ops.Linear(hidden_size, ffn_size // 2),
    )

    # Forward pass
    x = torch.randn(batch_size, hidden_size, device="cuda")
    y = mlp(x)

.. figure:: ./layernorm_mlp.png
   :align: center

   Operations that match ``LayerNormMLP`` module. Note that different
   fusions have been applied in the forward and backward passes.

Quantization
^^^^^^^^^^^^

The operation fuser respects TE's APIs for low-precision ("quantized")
data formats like FP8 and FP4. Constructing operations within a
``quantized_model_init`` context will enable quantized weights and
performing the forward pass within an ``autocast`` context will enable
quantized compute.

.. code-block:: python

    import torch
    import transformer_engine.pytorch as te

    # Construct layer with quantized weights
    with te.quantized_model_init():
        fc1 = te.ops.Sequential(
            te.ops.LayerNorm(4096),
            te.ops.Linear(28672, 4096),
        )

    # Forward pass within autocast context
    x = torch.randn(16384, 4096, device="cuda")
    with te.autocast():
        y = fc1(x)

    # Backward pass outside of autocast context
    y.sum().backward()

.. figure:: ./fp8_layernorm_linear.png
   :align: center

   Operations that match ``LayerNormLinear`` module with FP8
   quantization.

Internally, each operation that supports quantized compute holds one
or more ``Quantizer`` s, which are builder classes for converting
high-precision tensors (e.g. in FP32 or BF16) to quantized tensors. In
order to enable fused quantization kernels, operations can access the
quantizers of neighboring operations and quantize eagerly. In some
situations, like when operations are split across multiple
``Sequential`` s, it may be helpful to encourage the fuser by manually
adding ``Quantize`` operations.

.. code-block:: python

    import torch
    import transformer_engine.pytorch as te

    # Construct layer with quantized weights
    with te.quantized_model_init():
        norm = te.ops.Sequential(
            te.ops.LayerNorm(4096),
            te.ops.Quantize(),
        )
        fc1 = te.ops.Sequential(
            te.ops.Linear(28672, 4096),
        )

    # Forward pass
    x = torch.randn(16384, 4096, device="cuda")
    with te.autocast():
        y = norm(x)  # y is a QuantizedTensor
        z = fc1(y)

.. warning::

   This is an expert technique. Quantizer configurations can be quite
   complicated, so the ``Quantize`` operation's quantizers may be
   suboptimal.

Branching operations
^^^^^^^^^^^^^^^^^^^^

The operation fuser supports very limited branching behavior. While
the operations must be in sequential order, some operations can accept
extra inputs or produce extra outputs. For example, ``AddExtraInput``
will add an extra input tensor to the intermediate tensor and
``MakeExtraOutput`` will return the intermediate tensor as an extra
output. When calling a ``Sequential`` that contains any of these
branching operations, the extra inputs should be passed in as
arguments and the extra outputs will be returned.

.. code-block:: python

    import torch
    import transformer_engine.pytorch as te

    # Construct MLP with residual connection
    fc1 = te.ops.Sequential(
        te.ops.LayerNorm(4096),
        te.ops.MakeExtraOutput(),  # Output residual
        te.ops.Linear(28672, 4096),
        te.ops.SwiGLU(),
    )
    fc2 = te.ops.Sequential(
        te.ops.Linear(4096, 14336),
        te.ops.AddExtraInput(),  # Add residual
    )

    # Forward pass
    x = torch.randn(16384, 4096, device="cuda")
    y, residual = fc1(x)
    y = fc2(x, residual)

.. figure:: ./residual_layernorm_mlp.png
   :align: center

   Operations for an MLP block with a residual connection. Note that
   the block has been split into two sections, each with one branching
   operation.

Implementation details
^^^^^^^^^^^^^^^^^^^^^^

In addition to ``FusibleOperation`` and ``Sequential``, the fuser
infrastructure relies on the following classes:

- ``BasicOperation``: The most basic type of ``FusibleOperation``.
  Examples include ``BasicLinear``, ``Bias``, and ``ReLU``. It holds
  parameters and state, and it implements both a forward and backward
  pass. The ``op_forward`` and ``op_backward`` functions have an
  interface reminiscent of ``torch.autograd.Function``, e.g. they
  accept a context object that caches state from the forward pass to
  the backward pass.
- ``FusedOperation``: A ``FusibleOperation`` that can replace one or
  more ``BasicOperation`` s. Examples include
  ``ForwardLinearBiasActivation`` and ``BackwardActivationBias``. Its
  forward and backward passes (the ``fuser_forward`` and
  ``fuser_backward`` functions) must produce equivalent results as its
  corresponding ``BasicOperation`` s. This also means that the
  ``FusedOperation`` is stateless since it can access parameters and
  state from the ``BasicOperation`` s. Note that different fusions may
  be applied in the forward and backward pass, so a ``FusedOperation``
  may be missing its forward and/or backward implementation.
- ``OperationFuser``: This is the class that manages the operation
  fusions. It launches the forward and backward passes within a
  ``torch.autograd.Function``.

The first time that a ``Sequential`` is called, it will group adjacent
``FusibleOperation`` s together into ``OperationFuser`` s. The first
time an ``OperationFuser`` is called, it will attempt to fuse
operations for the forward pass and backward pass. Subsequent calls
will reuse the same state unless it has been invalidated, e.g. by
changing the quantization recipe.

Misconceptions
--------------

- **The op fuser is not a general kernel compiler**: The op fuser API
  is simply an alternative way to access TE fused kernels, most of
  which are targeted toward common Transformer architectures. For
  generic kernel compilation, consider tools like
  `nvFuser <https://github.com/NVIDIA/Fuser>`_,
  `CuTe DSL <https://github.com/NVIDIA/cutlass>`_,
  `torch.compile <https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_,
  `Triton <https://github.com/triton-lang/triton>`_,
  or `Pallas <https://docs.jax.dev/en/latest/pallas/index.html>`_.
- **The op fuser is not a graph compiler**: The op fuser only supports
  operations in a sequential order, with very limited support for
  branching operations. Support for general graphs is not planned
  since it would massively increase complexity.
- **The op fuser is not interchangeable with the monolithic TE
  modules**: Modules like ``Linear``, ``LayerNormLinear``, and
  ``TransformerLayer`` support a wide range of features and advanced
  workflows, which makes them challenging to decompose into simple
  operations that work with the fuser. They are also carefully
  hand-tuned to achieve maximum performance.

Creating a custom fused operation
---------------------------------

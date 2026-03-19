..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _autograd-integration:

Autograd Integration
====================

Transformer Engine modules use custom ``torch.autograd.Function`` subclasses to control
what is saved for backward, how FP8 tensors flow through the autograd graph, and when
activation recomputation occurs.

The _Linear Pattern
-------------------

The canonical example is ``_Linear`` in ``transformer_engine/pytorch/module/linear.py``.
This autograd function wraps the forward GEMM and defines the corresponding backward
(dgrad + wgrad).

**Forward** (pseudocode):

.. code-block:: python

   class _Linear(torch.autograd.Function):
       @staticmethod
       def forward(ctx, input, weight, bias, ...):
           # 1. Quantize input (produces rowwise + columnwise)
           qinput = input_quantizer(input)

           # 2. Get quantized weight
           qweight = weight_quantizer(weight)

           # 3. Forward GEMM: output = qinput @ qweight^T
           output = general_gemm(qinput, qweight, bias=bias)

           # 4. Save for backward
           ctx.save_for_backward(qinput, qweight, ...)

           return output

**Backward** (pseudocode):

.. code-block:: python

       @staticmethod
       def backward(ctx, grad_output):
           qinput, qweight, ... = ctx.saved_tensors

           # 1. Quantize grad_output
           qgrad = grad_quantizer(grad_output)

           # 2. Dgrad GEMM: grad_input = qgrad @ qweight
           grad_input = general_gemm(qgrad, qweight)

           # 3. Wgrad GEMM: grad_weight = qinput^T @ qgrad
           #    Uses columnwise qinput saved from forward
           grad_weight = general_gemm(qinput.columnwise, qgrad)

           return grad_input, grad_weight, grad_bias, ...

Saved Tensor Strategy
---------------------

What gets saved for backward depends on the configuration:

**FP8 disabled**: Standard PyTorch behavior — save full-precision input and weight.

**FP8 enabled (no recompute)**: Save the ``QuantizedTensor`` objects from forward. These
contain both rowwise data (used by dgrad GEMM) and columnwise data (used by wgrad GEMM).
Memory cost: ~2× the FP8 data size (rowwise + columnwise).

Saving QuantizedTensorStorage via prepare_for_saving
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch offers two ways to pass data from forward to backward: direct attributes on
``ctx``, and ``ctx.save_for_backward()``. Direct ``ctx`` attributes are not released after
the backward pass — they persist until the *next* forward pass. Storing tensors there
would keep memory alive throughout the entire backward pass and into the next forward,
which is very wasteful. Using ``save_for_backward`` lets PyTorch release the memory
promptly, but it only accepts ``torch.Tensor`` objects.

Since ``QuantizedTensorStorage`` is not a ``torch.Tensor``, the helper function
``prepare_for_saving()`` (in ``quantized_tensor.py``) splits each storage into:

- Its **metadata** (with all tensor fields set to ``None``) — stored on ``ctx.tensor_objects``.
- A list of raw **``torch.Tensor`` objects** — passed through ``ctx.save_for_backward()``.

In the backward pass, ``restore_from_saved()`` reassembles the original
``QuantizedTensorStorage`` objects from the saved tensors and metadata.

.. code-block:: python

   # Forward: split and save
   tensors_to_save, tensor_objects = prepare_for_saving(inputmat, weightmat, weight, bias)
   ctx.save_for_backward(*tensors_to_save)
   ctx.tensor_objects = tensor_objects

   # Backward: reassemble
   inputmat, weightmat, weight, bias = restore_from_saved(
       ctx.saved_tensors, ctx.tensor_objects,
   )

**FP8 enabled + activation recompute**: Only save the high-precision input (or a stashed
copy). During backward, re-run the quantization to produce fresh FP8 data. Saves memory
at the cost of recomputation.

Activation Recomputation
------------------------

TE supports activation recomputation (gradient checkpointing) at multiple granularities:

- **Full recompute**: Re-run the entire forward pass during backward. Standard PyTorch
  ``checkpoint()`` works with TE modules.
- **Selective recompute**: Only recompute the quantization (cast) operations, keeping
  the GEMM results. This is cheaper than full recompute because casting is
  memory-bandwidth-bound while GEMM is compute-bound.

The ``TransformerLayer`` module provides a ``activation_checkpointing`` parameter that
controls this behavior.

FP8 Tensors in Autograd
------------------------

``QuantizedTensor`` (a ``torch.Tensor`` subclass) can be saved via
``ctx.save_for_backward()`` like any other tensor. Key behaviors:

- **No implicit dequantization** during save — the FP8 data is stored as-is.
- During backward, the saved ``QuantizedTensor`` is retrieved and its ``.get_storage()``
  method provides the raw FP8 data + scales for GEMM.
- If a non-TE operation receives a ``QuantizedTensor``, ``__torch_dispatch__``
  automatically dequantizes it.

Interaction with torch.compile
-------------------------------

TE modules are compatible with ``torch.compile`` but require care:

- ``NVTE_TORCH_COMPILE=1`` enables compile-friendly code paths.
- Some autograd functions use ``torch.compiler.is_compiling()`` to select between
  eager and compile-compatible implementations.
- FP8 state management (amax updates, scale refreshes) must happen outside compiled
  regions because they involve in-place mutation of module state.

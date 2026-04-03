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

Since ``QuantizedTensorStorage`` is not a ``torch.Tensor`` (see
:doc:`/developer/quantization/class_hierarchy` for the distinction between
``QuantizedTensorStorage`` and ``QuantizedTensor``), the helper function
``prepare_for_saving()`` (in ``quantized_tensor.py``) splits each storage into:

- Its **metadata** (with all tensor fields set to ``None``) — stored on ``ctx.tensor_objects``.
- A list of raw **``torch.Tensor`` objects** — passed through ``ctx.save_for_backward()``.

In the backward pass, ``restore_from_func_ctx()`` reassembles the original
``QuantizedTensorStorage`` objects from the saved tensors and metadata. It also
automatically deletes ``ctx.tensor_objects`` to avoid keeping references to the
reassembled tensors on ``ctx`` (which would defeat the purpose of using
``save_for_backward`` in the first place).

.. code-block:: python

   # Forward: split and save
   tensors_to_save, tensor_objects = prepare_for_saving(inputmat, weightmat, weight, bias)
   ctx.save_for_backward(*tensors_to_save)
   ctx.tensor_objects = tensor_objects

   # Backward: reassemble and release ctx references in one call
   inputmat, weightmat, weight, bias = restore_from_func_ctx(ctx)

Activation Recomputation
------------------------

TE provides its own ``checkpoint()`` function (in ``transformer_engine/pytorch/distributed.py``)
rather than relying on the standard PyTorch ``torch.utils.checkpoint.checkpoint()``. The TE
version handles FP8 state correctly across the recompute boundary.

TE supports activation recomputation (gradient checkpointing) at multiple granularities:

- **Full recompute**: Re-run the entire forward pass during backward. The TE checkpoint
  function re-enters the ``autocast`` context during recomputation so that quantizers
  and FP8 state are correctly restored.
- **Selective recompute**: Only recompute specific operations (e.g., the core attention
  computation) while keeping other activations saved. The ``TransformerLayer`` module
  provides an ``activation_checkpointing`` parameter that controls selective
  recomputation.

FP8 Tensors in Autograd
------------------------

``QuantizedTensor`` (a ``torch.Tensor`` subclass) can be saved via
``ctx.save_for_backward()`` like any other tensor. No implicit dequantization happens
during save — the FP8 data is stored as-is.

__torch_dispatch__ and Automatic Dequantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``QuantizedTensor`` implements ``__torch_dispatch__`` to intercept all PyTorch operations.
This is the most complex part of the ``QuantizedTensor`` class. The dispatch logic handles
three cases:

1. **TE-optimized operations** (e.g., GEMM): These recognize ``QuantizedTensor`` inputs
   and extract the raw FP8 data + scales directly via ``get_data_tensors()``, avoiding
   any dequantization.

2. **Non-mutable operations**: For standard PyTorch ops that don't modify their inputs
   (e.g., ``torch.add``, ``torch.matmul``), the dispatch automatically dequantizes all
   quantized tensor arguments, executes the operation in high precision, and returns the
   result. This makes quantized tensors "just work" with arbitrary PyTorch code, at the
   cost of a dequantize.

3. **In-place operations**: These require special care. The dispatch dequantizes the
   inputs, executes the in-place op on the dequantized data, then re-quantizes the
   result back into the original ``QuantizedTensor``'s storage. This ensures the
   quantized representation stays consistent after mutation.

Interaction with torch.compile
-------------------------------

TE modules work with ``torch.compile`` by disabling compilation for TE-specific code
(which results in graph breaks). The ``NVTE_TORCH_COMPILE=1`` environment variable
enables paths where TE uses ``torch.compile`` internally within some modules. In
CPU-limited training scenarios, the graph breaks may cause slowdowns.

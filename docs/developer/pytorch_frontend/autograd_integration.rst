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

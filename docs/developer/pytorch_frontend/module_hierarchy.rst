..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _module-hierarchy:

Module Hierarchy
================

All PyTorch modules in Transformer Engine extend a common base class that provides
FP8 state management, quantizer lifecycle, and distributed training hooks.

.. figure:: ./img/pytorch_module_hierarchy.svg
   :align: center
   :width: 80%

   Inheritance tree of TransformerEngine PyTorch modules.

..
   Diagram description for ``pytorch_module_hierarchy.svg``:
   Tree diagram with torch.nn.Module at the top.
   Below it, two branches:
   Branch 1: TransformerEngineBaseModule, with children:
     ├── Linear
     ├── LayerNorm
     ├── RMSNorm
     ├── LayerNormLinear
     ├── LayerNormMLP
     ├── GroupedLinear
     └── DotProductAttention
   Branch 2: Directly from torch.nn.Module:
     ├── MultiheadAttention (composes DotProductAttention + Linear projections)
     ├── TransformerLayer (composes the above modules)
     ├── Fp8Padding
     └── Fp8Unpadding

TransformerEngineBaseModule
---------------------------

**Location**: ``transformer_engine/pytorch/module/base.py``

``TransformerEngineBaseModule`` extends ``torch.nn.Module`` and is the foundation for all
TE modules that support FP8 quantization. Key responsibilities:

**FP8 State Management**

- ``init_fp8_metadata()``: Creates quantizer instances and amax history buffers based on
  the active recipe.
- ``pre_forward()``: Called at the start of each forward pass to update FP8 state (refresh
  scales from amax history, check if FP8 is enabled).
- ``post_forward()``: Called after forward to record amax values.
- FP8 metadata is registered as module buffers for serialization.

**Quantizer Access**

- Maintains quantizers for each quantized tensor (input, weight, output, gradient input,
  gradient output, gradient weight).
- Quantizer instances are recreated when the recipe changes.

**Distributed Hooks**

- ``set_tensor_parallel_group()``: Configure TP process group.
- ``set_sequence_parallel()``: Enable sequence parallelism.
- Manages all-reduce of amax values across distributed ranks.

**Weight Caching**

- Provides caching infrastructure for quantized weights that persist across microbatches
  during gradient accumulation.

Note that individual parameters (``weight``, ``bias``) are defined by each subclass,
not by the base module. See the module hierarchy diagram above for the full list of
subclasses and their docstrings for module-specific details.

Module Lifecycle
----------------

A typical forward pass through a TE module:

.. code-block:: text

   1. autocast() sets global FP8 state
   2. module.forward() called
      a. pre_forward() — refresh FP8 scales, create quantizers
      b. Quantize input via input_quantizer(input)
      c. Get quantized weight (cached or quantize via weight_quantizer)
      d. Call the module's custom autograd function
         i.  general_gemm(quantized_input, quantized_weight)
         ii. Save tensors for backward
      e. post_forward() — record amax values
   3. Return output

See :doc:`autograd_integration` for details on step (d).

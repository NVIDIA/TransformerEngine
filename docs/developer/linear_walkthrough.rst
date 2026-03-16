..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _linear-walkthrough:

Linear Module: End-to-End Walkthrough
======================================

This page traces a complete forward and backward pass through ``te.Linear``, the most
fundamental Transformer Engine module. It connects concepts from across the codebase:
quantization, GEMM, distributed communication, and autograd.

.. figure:: ./img/linear_e2e_flow.svg
   :align: center
   :width: 95%

   End-to-end flow through Linear forward and backward with FP8 and tensor parallelism.

..
   Diagram description for ``linear_e2e_flow.svg``:
   Two horizontal swim lanes labeled "Forward" and "Backward".
   Forward lane (left to right):
     1. "Input (BF16)" →
     2. "All-gather (if column-parallel + SP)" →
     3. "input_quantizer(input)" producing "FP8 rowwise + columnwise" →
     4. "general_gemm(weight, input)" producing "Output (BF16)" →
     5. "Reduce-scatter (if row-parallel + SP) or All-reduce (if row-parallel)"
   Backward lane (right to left):
     1. "grad_output (BF16)" →
     2. "grad_output_quantizer(grad)" →
     3. "Dgrad GEMM: general_gemm(grad, weight)" producing "grad_input" →
     4. "Wgrad GEMM: general_gemm(input_columnwise, grad)" producing "grad_weight" →
     5. "All-reduce grad_weight (if column-parallel)"
   Dotted arrows from forward boxes 3→backward box 4 labeled "saved columnwise input"
   and forward weight→backward box 3 labeled "saved weight".

Setup
-----

.. code-block:: python

   import transformer_engine.pytorch as te
   from transformer_engine.common.recipe import DelayedScaling

   # Create a Linear module with FP8
   linear = te.Linear(4096, 16384, bias=True)

   # Enable FP8
   recipe = DelayedScaling(fp8_format=te.recipe.Format.HYBRID)
   with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
       output = linear(input)  # Triggers the flow below

Phase 1: Module Forward Entry
------------------------------

**File**: ``transformer_engine/pytorch/module/linear.py``, ``Linear.forward()`` (line ~1343)

1. ``pre_forward()`` is called (inherited from ``TransformerEngineBaseModule``):

   - Checks if FP8 is enabled via ``FP8GlobalStateManager``.
   - Creates or refreshes quantizer instances from the active recipe.
   - For delayed scaling: computes new scales from amax history.

2. Quantizers are prepared:

   - ``input_quantizer`` — for the activation tensor
   - ``weight_quantizer`` — for the weight parameter
   - ``grad_output_quantizer`` — for the backward gradient

3. ``_Linear.apply()`` is called, entering the autograd function.

Phase 2: _Linear.forward() — Input Preparation
-------------------------------------------------

**File**: ``transformer_engine/pytorch/module/linear.py``, ``_Linear.forward()`` (line ~87)

The input tensor is prepared based on the parallel mode and FP8 state:

**Column-parallel with sequence parallelism** (most common for QKV/FC1):

.. code-block:: python

   # 1. Set quantizer usage: need rowwise for forward GEMM,
   #    columnwise for backward wgrad GEMM
   input_quantizer.set_usage(rowwise=True, columnwise=backward_needs_input)

   # 2. Quantize local input shard
   inputmat = input_quantizer(inputmat)  # Returns QuantizedTensorStorage

   # 3. All-gather along sequence dimension
   inputmat_total, _ = gather_along_first_dim(inputmat, tp_group)

**No parallelism** (standalone Linear):

.. code-block:: python

   input_quantizer.set_usage(rowwise=True, columnwise=backward_needs_input)
   inputmat = input_quantizer(inputmat)
   inputmat_total = inputmat

The ``input_quantizer(inputmat)`` call produces a ``QuantizedTensorStorage`` containing:

- Rowwise FP8 data + ``scale_inv`` (for forward GEMM)
- Columnwise FP8 data + ``columnwise_scale_inv`` (saved for wgrad GEMM)

See :doc:`quantization/rowwise_columnwise` for why both layouts are needed.

Phase 3: _Linear.forward() — Weight Preparation
--------------------------------------------------

.. code-block:: python

   # Configure weight quantizer
   weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)

   # Get quantized weight (may use cached version)
   weightmat = module.get_weight_workspace(
       tensor=weight,
       quantizer=weight_quantizer,
       cache_name="weight",         # Cache across microbatches
       update_workspace=is_first_microbatch,
   )

Weight caching is important: the same FP8 weight can be reused across gradient
accumulation microbatches, avoiding redundant quantization.

Phase 4: _Linear.forward() — GEMM
-----------------------------------

.. code-block:: python

   # Forward GEMM: output = input @ weight^T
   gemm_out, *_ = general_gemm(
       weightmat,           # A operand (FP8)
       inputmat_total,      # B operand (FP8)
       out_dtype=activation_dtype,
       bias=bias,
       use_split_accumulator=use_split_accumulator,
       ub=ub_obj,           # Userbuffers overlap (optional)
       ub_type=ub_type,
   )

This call chain:

1. **Python**: ``general_gemm()`` in ``transformer_engine/pytorch/cpp_extensions/gemm.py``
   extracts raw tensors and scales from the quantized inputs.
2. **pybind11**: ``te_general_gemm()`` in ``transformer_engine/pytorch/csrc/extensions/gemm.cpp``
   constructs ``NVTETensor`` handles and calls the C API.
3. **C API**: ``nvte_general_gemm()`` in ``transformer_engine/common/gemm/``
   dispatches to cuBLASLt with FP8 compute types.
4. **cuBLASLt**: Executes the FP8 matrix multiplication on the GPU.

Phase 5: _Linear.forward() — Output Communication
----------------------------------------------------

For **row-parallel** (output projection, FC2):

.. code-block:: python

   if sequence_parallel:
       out, _ = reduce_scatter_along_first_dim(out, tp_group)
   elif tensor_parallel:
       out, _ = allreduce(out, tp_group)

For **column-parallel**: no communication needed (output is naturally partitioned).

Phase 6: _Linear.forward() — Save for Backward
--------------------------------------------------

.. code-block:: python

   # Save quantized input for wgrad (columnwise data)
   # Save quantized weight for dgrad
   ctx.save_for_backward(inputmat, weightmat, weight, ...)
   ctx.weight_quantizer = weight_quantizer

Key optimization: only the *columnwise* portion of the input is kept (the rowwise
data is discarded after the forward GEMM). This halves the activation memory for FP8.

Phase 7: _Linear.backward() — Dgrad
--------------------------------------

.. code-block:: python

   # Quantize grad_output
   grad_output_quantizer.set_usage(rowwise=True, columnwise=True)
   qgrad_output = grad_output_quantizer(grad_output)

   # Dgrad GEMM: grad_input = grad_output @ weight
   dgrad, *_ = general_gemm(
       weightmat,        # Columnwise weight from forward
       qgrad_output,     # Rowwise grad_output
       out_dtype=activation_dtype,
   )

Phase 8: _Linear.backward() — Wgrad
--------------------------------------

.. code-block:: python

   # Wgrad GEMM: grad_weight = input^T @ grad_output
   # Uses columnwise input saved from forward
   wgrad, *_ = general_gemm(
       inputmat,          # Columnwise input from forward
       qgrad_output,      # Columnwise grad_output
       out_dtype=activation_dtype,
       grad=True,         # Indicates wgrad GEMM
   )

This is why the forward pass computed both rowwise and columnwise input: the columnwise
data is consumed here by the wgrad GEMM.

Phase 9: Post-Backward
------------------------

1. ``post_forward()`` records the output amax for delayed scaling.
2. For distributed training, amax values are all-reduced across TP ranks.
3. The amax history is updated for the next iteration's scale computation.

Summary of Files Touched
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - File
     - Role in Linear
   * - ``pytorch/module/linear.py``
     - ``Linear.forward()``, ``_Linear`` autograd
   * - ``pytorch/module/base.py``
     - ``pre_forward()``, FP8 state management
   * - ``pytorch/quantized_tensor.py``
     - ``Quantizer``, ``QuantizedTensorStorage`` base classes
   * - ``pytorch/tensor/float8_tensor.py``
     - ``Float8Quantizer``, FP8 cast implementation
   * - ``pytorch/cpp_extensions/gemm.py``
     - ``general_gemm()`` Python wrapper
   * - ``pytorch/csrc/extensions/gemm.cpp``
     - pybind11 GEMM binding
   * - ``common/gemm/cublaslt_gemm.cu``
     - cuBLASLt GEMM kernel dispatch
   * - ``pytorch/distributed.py``
     - ``allreduce()``, ``gather_along_first_dim()``, etc.
   * - ``pytorch/quantization.py``
     - ``FP8GlobalStateManager``, ``fp8_autocast``

See Also
--------

- :doc:`architecture_overview` — High-level system architecture
- :doc:`quantization/class_hierarchy` — Quantizer/Storage/Tensor design
- :doc:`quantization/rowwise_columnwise` — Why both layouts exist
- :doc:`pytorch_frontend/autograd_integration` — Autograd patterns
- :doc:`distributed/tensor_parallel` — TP communication in Linear

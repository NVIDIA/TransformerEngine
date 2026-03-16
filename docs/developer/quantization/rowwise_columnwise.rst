..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _rowwise-columnwise:

Rowwise vs. Columnwise Usage
=============================

Quantized tensors in Transformer Engine can be stored in two layouts: **rowwise** and
**columnwise**. This dual-layout design is driven by cuBLASLt's requirements for FP8
GEMM operands.

.. figure:: ./img/rowwise_columnwise.svg
   :align: center
   :width: 80%

   Forward GEMM uses rowwise data; backward wgrad GEMM uses columnwise data.

..
   Diagram description for ``rowwise_columnwise.svg``:
   Two side-by-side GEMM diagrams:
   Left "Forward GEMM (dgrad)":
     Matrix A (activation, rowwise) × Matrix B (weight, columnwise) = Output
     A is labeled "row-major, scale_inv per row"
     B is labeled "col-major, columnwise_scale_inv per column"
   Right "Backward GEMM (wgrad)":
     Matrix A^T (activation transposed, columnwise) × Matrix B (grad_output, rowwise) = Weight grad
     A^T is labeled "using columnwise_data from forward"
   Arrow from left A to right A^T labeled "same data, different layout view"

Why Two Layouts?
----------------

cuBLASLt's FP8 GEMM computes ``C = A × B`` where:

- ``A`` is read in **row-major** order (consecutive elements along the last dimension).
- ``B`` is read in **column-major** order (consecutive elements along the first dimension).

In the **forward pass**:

- ``output = activation × weight^T``
- Activation is the ``A`` operand → needs **rowwise** quantization.
- Weight is the ``B`` operand → needs **columnwise** quantization.

In the **backward wgrad pass**:

- ``weight_grad = activation^T × grad_output``
- Activation transposed is the ``A`` operand → needs **columnwise** data from forward.
- Grad output is the ``B`` operand → needs **columnwise** quantization.

This is why the forward pass must produce *both* rowwise and columnwise quantized data
for activations: the rowwise data is consumed immediately by the forward GEMM, while the
columnwise data is saved for the backward wgrad GEMM.

Setting Usage on Quantizers
----------------------------

The ``Quantizer.set_usage()`` method controls which layouts are produced:

.. code-block:: python

   # Forward activation quantizer: needs both layouts
   fwd_quantizer.set_usage(rowwise=True, columnwise=True)

   # Weight quantizer: only columnwise for forward GEMM
   weight_quantizer.set_usage(rowwise=False, columnwise=True)

   # Backward grad_output quantizer: needs both for dgrad and wgrad
   bwd_quantizer.set_usage(rowwise=True, columnwise=True)

When ``columnwise=True``, the quantization kernel produces an additional transposed
copy of the data (or, for block-scaling modes, computes column-oriented block scales).

Impact on the Tensor Struct
----------------------------

In the C++ ``Tensor`` struct, the layout is reflected in separate fields:

.. code-block:: cpp

   struct Tensor {
       SimpleTensor data;               // Rowwise quantized data
       SimpleTensor columnwise_data;    // Columnwise quantized data
       SimpleTensor scale_inv;          // Rowwise scale inverses
       SimpleTensor columnwise_scale_inv; // Columnwise scale inverses
       // ...
   };

For **tensor scaling** (delayed/current), ``data`` and ``columnwise_data`` point to
different memory buffers (the data is physically transposed).

For **block scaling** modes, the data may be the same buffer but with different scale
tensors (row-oriented vs. column-oriented block scales).

Performance Implications
------------------------

Producing both layouts doubles the quantization work and memory for activations. This is
a deliberate trade-off:

- **Without dual layout**: The wgrad GEMM would need to transpose and requantize at
  backward time, adding latency to the critical path.
- **With dual layout**: Extra memory and compute during forward, but the backward wgrad
  can proceed immediately with pre-computed columnwise data.

For memory-constrained scenarios, activation recomputation can be used — the forward
pass discards the saved columnwise data and recomputes it during backward. See
:doc:`/developer/pytorch_frontend/autograd_integration` for details.

See Also
--------

- :doc:`class_hierarchy` — How Storage objects hold both layouts
- :doc:`/developer/cpp_core/scaling_modes` — Scale granularity per mode

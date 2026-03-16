..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _cpp-extensions:

C++ Extensions Bridge
=====================

The PyTorch frontend communicates with the C++ core through a pybind11 extension module.
This bridge converts PyTorch tensors to ``NVTETensor`` handles and calls the C API.

Architecture
------------

.. code-block:: text

   Python (transformer_engine/pytorch/cpp_extensions/)
       │  torch.Tensor → NVTETensor conversion
       ▼
   pybind11 (transformer_engine/pytorch/csrc/extensions/)
       │  Call C API functions
       ▼
   C API (transformer_engine/common/include/transformer_engine/)
       │  Unpack NVTETensor → internal Tensor struct
       ▼
   CUDA kernels (transformer_engine/common/)

Python Side
-----------

**Location**: ``transformer_engine/pytorch/cpp_extensions/``

Each kernel area has a corresponding Python wrapper module:

- ``gemm.py`` — ``general_gemm()``, ``grouped_gemm()``
- ``normalization.py`` — ``layernorm_fwd()``, ``rmsnorm_fwd()``, etc.
- ``activation.py`` — ``gelu()``, ``silu()``, etc.
- ``cast.py`` — ``quantize()``, ``dequantize()``
- ``transpose.py`` — ``transpose()``, ``cast_transpose()``
- ``attention.py`` — ``fused_attn_fwd()``, ``fused_attn_bwd()``

These wrappers:

1. Extract raw data tensors and scales from ``QuantizedTensor`` / ``QuantizedTensorStorage``
   objects.
2. Construct ``NVTETensor`` handles via helper functions.
3. Call the pybind11-exposed C++ function.
4. Wrap outputs back into Python tensor types.

**Example** — ``general_gemm()`` (simplified):

.. code-block:: python

   def general_gemm(A, B, bias=None, ...):
       # Convert QuantizedTensor → NVTETensor components
       A_data, A_scale_inv, A_dtype = _extract_tensors(A)
       B_data, B_scale_inv, B_dtype = _extract_tensors(B)

       # Call pybind11 extension
       output = tex.general_gemm(
           A_data, A_scale_inv, A_dtype,
           B_data, B_scale_inv, B_dtype,
           bias, output_dtype, ...
       )
       return output

C++ Side (pybind11)
-------------------

**Location**: ``transformer_engine/pytorch/csrc/extensions/``

The pybind11 module (``transformer_engine/pytorch/csrc/ts_fp8_op.cpp`` or similar)
registers Python-callable functions that:

1. Accept ``torch::Tensor`` and scalar arguments from Python.
2. Construct ``NVTETensor`` handles using ``makeTransformerEngineTensor()``.
3. Call the C API function (e.g., ``nvte_general_gemm()``).
4. Return ``torch::Tensor`` outputs.

**Example** — GEMM binding (simplified):

.. code-block:: cpp

   void te_general_gemm(
       at::Tensor A, at::Tensor A_scale_inv, int A_dtype,
       at::Tensor B, at::Tensor B_scale_inv, int B_dtype,
       at::Tensor D, at::Tensor bias, ...) {

       // Create NVTETensor handles
       auto te_A = makeTransformerEngineTensor(
           A.data_ptr(), A.sizes(), A_dtype, A_scale_inv, ...);
       auto te_B = makeTransformerEngineTensor(
           B.data_ptr(), B.sizes(), B_dtype, B_scale_inv, ...);

       // Call C API
       nvte_general_gemm(te_A.data(), te_B.data(), te_D.data(),
                         te_bias.data(), stream);
   }

Tensor Conversion Helpers
-------------------------

The ``makeTransformerEngineTensor()`` family of functions (in
``transformer_engine/pytorch/csrc/common.h``) handles the conversion from PyTorch
tensors to ``NVTETensor``:

- Sets up the ``NVTEBasicTensor`` parameters (data pointer, shape, dtype).
- Attaches scaling metadata (scale, scale_inv, amax) via ``nvte_tensor_set()``.
- Sets the ``NVTEScalingMode`` on the tensor.

The reverse conversion (``NVTETensor`` → PyTorch tensor) is typically not needed because
C API functions write into pre-allocated output buffers.

Adding a New Extension
----------------------

To expose a new C API function to Python:

1. Add the C API function to the appropriate header in ``include/transformer_engine/``.
2. Add a pybind11 wrapper in ``csrc/extensions/``.
3. Register it in the pybind11 module definition.
4. Add a Python wrapper in ``cpp_extensions/``.
5. Update ``__init__.py`` exports as needed.

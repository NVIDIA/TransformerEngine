..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _cpp-extensions:

C++ Extensions Bridge
=====================

The PyTorch frontend communicates with the C++ core through a pybind11 extension module.
This bridge accepts Python objects (``QuantizedTensorStorage``, ``QuantizedTensor``, or
plain ``torch.Tensor``), converts them to ``TensorWrapper`` / ``NVTETensor`` handles, and
calls the C API.

Architecture
------------

.. code-block:: text

   Python (transformer_engine/pytorch/cpp_extensions/)
       │  Optional auxiliary processing (e.g. custom recipe handling)
       ▼
   pybind11 (transformer_engine/pytorch/csrc/extensions/)
       │  py::handle → TensorWrapper conversion, GIL release, C API call
       ▼
   C API (transformer_engine/common/include/transformer_engine/)
       │  NVTETensor handle → internal Tensor struct
       ▼
   CUDA kernels (transformer_engine/common/)

Python Side
-----------

**Location**: ``transformer_engine/pytorch/cpp_extensions/``

Only a subset of C++ extensions have dedicated Python wrapper modules:

- ``gemm.py`` — ``general_gemm()``, ``grouped_gemm()``,
  ``general_grouped_gemm_for_grouped_tensor()``
- ``fused_attn.py`` — ``fused_attn_fwd()``, ``fused_attn_bwd()``

Most other C++ extensions (normalization, activation, cast, transpose, etc.) are exposed
directly through the compiled ``transformer_engine_torch`` pybind11 module (imported
as ``tex``) and called without a Python wrapper layer. For example, normalization is
called as ``tex.layernorm_fwd(...)`` rather than through a ``normalization.py`` wrapper.

The Python wrappers that do exist provide a hook for auxiliary processing before calling
into C++. For example, ``general_gemm()`` handles
custom recipe dispatch and block-scaling detection, then passes the Python objects
through to C++:

.. code-block:: python

   def general_gemm(A, B, bias=None, quantization_params=None, ...):
       # Auxiliary logic (custom tensor dispatch, debug handling, etc.)
       if is_custom(A) or is_custom(B):
           return custom_gemm(...)

       # Pass Python objects as-is to the pybind11 layer.
       # QuantizedTensorStorage/QuantizedTensor objects are NOT unpacked here.
       out, bias_grad, gelu_input, extra_output = tex.generic_gemm(
           A, transa, B, transb, out, quantization_params, ...
       )
       return out, bias_grad, gelu_input, extra_output

C++ Side (pybind11)
-------------------

**Location**: ``transformer_engine/pytorch/csrc/extensions/``

The pybind11 module (``transformer_engine/pytorch/csrc/extensions/pybind.cpp``)
registers Python-callable functions. These functions:

1. Accept inputs as ``py::handle`` or ``py::object`` types — not ``at::Tensor``. This
   allows them to receive ``QuantizedTensorStorage``, ``QuantizedTensor``, or plain
   ``torch.Tensor`` objects without the Python layer needing to unpack them.
2. Convert inputs to ``TensorWrapper`` objects via ``makeTransformerEngineTensor()``.
3. Release the GIL and call the C API function. GIL release is important to prevent
   hangs in case of hardware failures on parallel ranks.
4. Return ``py::object`` outputs (which may be ``QuantizedTensorStorage`` or
   ``torch.Tensor`` depending on whether an output quantizer was provided).

**Example** — GEMM binding (simplified):

.. code-block:: cpp

   std::vector<py::object> gemm(
       py::handle A, bool transa,
       py::handle B, bool transb,
       py::object D,
       py::handle quantizer, ...) {

       // Convert Python objects to TensorWrapper (handles all tensor types).
       // The second argument is the quantizer — py::none() here because inputs
       // are already quantized and the quantizer is only needed for outputs.
       auto none = py::none();
       TensorWrapper A_tensor = makeTransformerEngineTensor(A, none);
       TensorWrapper B_tensor = makeTransformerEngineTensor(B, none);

       // Convert the Python quantizer handle to a C++ Quantizer object,
       // then use it to create the output buffer.
       std::unique_ptr<Quantizer> quantizer_cpp = convert_quantizer(quantizer);
       auto [out_tensor, out_py] = quantizer_cpp->create_tensor(shape, dtype);

       // Release GIL and call C API
       NVTE_SCOPED_GIL_RELEASE({
           nvte_cublas_gemm_v2(transa, transb, &alpha,
                               A_tensor.data(), B_tensor.data(),
                               &beta, out_tensor.data(), out_tensor.data(),
                               te_workspace.data(), config, main_stream);
       });

       return {out_py, bias_grad, gelu_input, extra_output};
   }

The ``NVTE_SCOPED_GIL_RELEASE`` macro checks ``PyGILState_Check()`` and releases the GIL
only if it is currently held. This ensures C API calls (which may block on CUDA operations)
do not hold the GIL, preventing deadlocks in multi-threaded or multi-rank scenarios.

TensorWrapper and NVTETensor
----------------------------

``NVTETensor`` (defined in ``transformer_engine.h``) is an opaque handle (``typedef void
*NVTETensor``). Unlike typical opaque handles in C libraries, ``NVTETensor`` is not a
direct pointer to an internal structure. Instead, it is an integer handle into a
pre-allocated pool of tensor descriptors (see ``transformer_engine.cpp``).

``TensorWrapper`` (also in ``transformer_engine.h``) is a lightweight C++ RAII wrapper
around an ``NVTETensor`` that provides builder methods (``set_rowwise_data()``,
``set_columnwise_data()``, ``set_scale()``, etc.) and calls ``nvte_destroy_tensor()`` on
destruction. Neither type owns the underlying data memory on the GPU — the Python objects
retain ownership.

Tensor Conversion: makeTransformerEngineTensor
----------------------------------------------

The key overload (in ``transformer_engine/pytorch/csrc/common.h`` and ``common.cpp``)
accepts ``py::handle`` inputs:

.. code-block:: cpp

   TensorWrapper makeTransformerEngineTensor(py::handle tensor, py::handle quantizer);

This function performs runtime type detection to handle all Python tensor types:

1. It iterates over a registry of known quantized types (defined in
   ``transformer_engine/pytorch/csrc/pybind.h`` as
   ``custom_types_converters``). Each entry maps a ``PyTypeObject*`` to an extraction
   function. This registry is populated during extension initialization
   (``init_extension()``), which is why the extensions must be initialized before any
   tensor conversion can occur.
2. When a match is found, it calls the corresponding extraction function (e.g.,
   ``NVTETensorFromFloat8Tensor``) which reads the internal data pointers and scale
   tensors from the Python object and populates a ``TensorWrapper``.
3. If no quantized type matches, it falls back to casting the input as a plain
   ``at::Tensor`` and constructing a ``TensorWrapper`` from its data pointer and shape.

This design is what allows the pybind11 layer to accept any tensor type as a
``py::handle`` without requiring the Python side to unpack anything.

Output Buffer Creation
----------------------

When an output quantizer is provided, the quantizer's ``create_tensor()`` method
(implemented in C++, e.g.,
``transformer_engine/pytorch/csrc/quantizer.cpp``) allocates both the data buffer and
the Python wrapper in one step:

.. code-block:: cpp

   // Simplified from Float8Quantizer::create_tensor()
   std::pair<TensorWrapper, py::object> Float8Quantizer::create_tensor(
       const std::vector<size_t>& shape, DType dtype) const {

       // Allocate data buffer
       at::Tensor data = at::empty(shape, at::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

       // Allocate scale inverse
       at::Tensor scale_inv = at::empty({1}, at::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

       // Create Python wrapper (Float8TensorStorage or Float8Tensor depending on `internal`)
       py::object out_py = /* construct Python object from data + scale_inv */;

       // Create C++ TensorWrapper pointing to the same buffers
       TensorWrapper out_cpp(this->get_scaling_mode());
       out_cpp.set_rowwise_data(data.data_ptr(), this->dtype, shape);
       out_cpp.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat32, {1});

       return {std::move(out_cpp), std::move(out_py)};
   }

The returned ``TensorWrapper`` and ``py::object`` point to the same underlying GPU memory.
The C API writes into the ``TensorWrapper``'s buffers, and the ``py::object`` is returned
to Python with the results already in place. Because ``TensorWrapper`` is non-owning, the
``py::object`` must be kept alive for as long as the GPU memory is needed — even if only
the ``TensorWrapper`` is passed to ``nvte_`` functions. Some functions also accept
pre-allocated output buffers from the caller instead of creating new ones via a quantizer.

Adding a New Extension
----------------------

To expose a new C API function to Python:

1. Add the C API function to the appropriate header in ``include/transformer_engine/``.
2. Add a pybind11 wrapper in ``csrc/extensions/``.
3. Register it in the pybind11 module definition.
4. (Optional) Add a Python wrapper in ``cpp_extensions/`` if auxiliary processing is
   needed before calling the ``tex`` function.

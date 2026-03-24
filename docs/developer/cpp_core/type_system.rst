..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _type-system:

Type System
===========

Transformer Engine maintains three layers of types: a C API for ABI stability, C++
wrappers for external consumers, and C++ internal types for the core implementation.

The C++ core exposes a **C API** (not C++) for ABI stability. Framework bindings interact
with opaque ``NVTETensor`` handles rather than C++ objects directly. The same header
(``transformer_engine.h``) also provides C++ wrapper types (``DType``,
``TensorWrapper``) that add type safety and convenience around the C API. These wrappers
are intended for use by framework bindings (pybind11 code, tests) — they wrap the C API
but do not expose internal implementation details.

Separately, ``common.h`` defines internal C++ types (``SimpleTensor``, ``Tensor``) used
only within the core library. These are never exposed across the API boundary.

C Types (Public API)
--------------------

Defined in ``transformer_engine/common/include/transformer_engine/transformer_engine.h``.

NVTEDType
^^^^^^^^^

A plain C ``enum`` representing data types at the API boundary:

.. code-block:: c

   enum NVTEDType {
       kNVTEByte = 0,
       kNVTEInt16 = 1,
       kNVTEInt32 = 2,
       kNVTEInt64 = 3,
       kNVTEFloat32 = 4,
       kNVTEFloat16 = 5,
       kNVTEBFloat16 = 6,
       kNVTEFloat8E4M3 = 7,
       kNVTEFloat8E5M2 = 8,
       kNVTEFloat8E8M0 = 9,
       kNVTEFloat4E2M1 = 10,
       kNVTENumTypes
   };

NVTEBasicTensor
^^^^^^^^^^^^^^^

A lightweight, non-owning tensor descriptor used to populate ``NVTETensor`` parameters:

.. code-block:: c

   struct NVTEBasicTensor {
       void *data_ptr;      // Raw pointer to data buffer
       NVTEDType dtype;     // Data type
       NVTEShape shape;     // Shape (up to 15 dimensions)
   };

NVTETensor
^^^^^^^^^^

An opaque handle (``typedef void *NVTETensor``) to the internal ``Tensor`` object.
Despite the ``void *`` typedef, ``NVTETensor`` is not a direct pointer to a ``Tensor``
struct. Instead, it is a 1-based integer index into a pre-allocated pool of ``Tensor``
objects (see ``transformer_engine.cpp``). The pool has a fixed capacity of approximately
26,000 tensors (20 MB / sizeof(Tensor)). Freed tensors are returned to a free-list for
reuse. This means ``nvte_create_tensor()`` does not allocate memory — it retrieves an
entry from the pool. If the pool is exhausted, an error is raised.

``NVTETensor`` is populated via ``nvte_tensor_set()`` with ``NVTETensorParam`` keys:

.. code-block:: c

   enum NVTETensorParam {
       kNVTERowwiseData = 0,
       kNVTEColumnwiseData = 1,
       kNVTEScale = 2,
       kNVTEAmax = 3,
       kNVTERowwiseScaleInv = 4,
       kNVTEColumnwiseScaleInv = 5,
       kNVTEColumnwiseAmax = 6,
       kNVTEWithGEMMSwizzledScales = 7,
   };

NVTEScalingMode
^^^^^^^^^^^^^^^

Controls the quantization granularity — see :doc:`scaling_modes` for details. Note that
``NVTE_DELAYED_TENSOR_SCALING`` (value 0) is used not only for delayed scaling but also
for current scaling and for high-precision (unquantized) tensors. The C++ core
distinguishes these cases by checking the data dtype (FP8 vs high-precision) and whether
the amax field is populated.

C++ Wrappers (External)
-----------------------

Defined in ``transformer_engine/common/include/transformer_engine/transformer_engine.h``,
within the ``transformer_engine`` C++ namespace. These wrap the C API with type safety and
RAII semantics, and are intended for use by framework bindings (pybind11 code) and tests.

DType
^^^^^

A C++ ``enum class`` mirroring ``NVTEDType``:

.. code-block:: cpp

   namespace transformer_engine {
   enum class DType {
       kByte, kInt16, kInt32, kInt64,
       kFloat32, kFloat16, kBFloat16,
       kFloat8E4M3, kFloat8E5M2, kFloat8E8M0,
       kFloat4E2M1, kNumTypes
   };
   }

.. warning::

   ``DType`` and ``NVTEDType`` have the same numeric values but are **not** implicitly
   convertible to each other (no implicit construction or assignment). Use ``static_cast``
   for conversions. However, comparison operators (``==``, ``!=``) are overloaded to work
   across the two types — see :ref:`dtype-conversion-patterns` below.

The header also provides inline helpers that accept ``DType``: ``is_fp8_dtype()``,
``is_fp4_dtype()``, ``is_high_precision_dtype()``.

TensorWrapper
^^^^^^^^^^^^^

A C++ RAII wrapper around the opaque ``NVTETensor`` handle. It manages the lifecycle of
the underlying C tensor (``nvte_create_tensor`` / ``nvte_destroy_tensor``) and provides
constructors that populate the tensor's fields via the C API:

.. code-block:: cpp

   class TensorWrapper {
   public:
       // Construct with data, shape, dtype, and optional scale/amax pointers
       TensorWrapper(void *dptr, const NVTEShape &shape, const DType dtype,
                     float *amax_dptr = nullptr, float *scale_dptr = nullptr,
                     float *scale_inv_dptr = nullptr,
                     NVTEShape scale_inv_shape = defaultShape,
                     const NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING);

       // Construct empty tensor
       explicit TensorWrapper(const NVTEScalingMode scaling_mode = NVTE_DELAYED_TENSOR_SCALING);

       ~TensorWrapper();  // Calls nvte_destroy_tensor

       NVTETensor data();  // Returns the underlying opaque handle
   };

``TensorWrapper`` is the primary way framework bindings create ``NVTETensor`` handles to
pass into C API functions. It does not own the data memory — it only owns the pool entry.

C++ Types (Internal)
--------------------

Defined in ``transformer_engine/common/common.h``. These types are used exclusively within
the core library and are never exposed across the API boundary.

SimpleTensor
^^^^^^^^^^^^

A lightweight internal tensor descriptor with implicit conversion to/from
``NVTEBasicTensor``:

.. code-block:: cpp

   struct SimpleTensor {
       void *dptr;
       std::vector<size_t> shape;
       DType dtype;

       // Implicit conversion from NVTEBasicTensor
       SimpleTensor(const NVTEBasicTensor &tensor);
       // Implicit conversion to NVTEBasicTensor
       operator NVTEBasicTensor() const;

       size_t numel() const;
       bool has_data() const;
       size_t buffer_size_bytes() const;
   };

``SimpleTensor`` converts freely to/from ``NVTEBasicTensor``, handling the
``NVTEDType`` ↔ ``DType`` cast internally.

Tensor
^^^^^^

The main internal tensor type, composed of multiple ``SimpleTensor`` members:

.. figure:: ./img/tensor_struct.svg
   :align: center
   :width: 60%

   UML-style diagram of the Tensor struct and its SimpleTensor members.

..
   Diagram description for ``tensor_struct.svg``:
   UML struct box labeled "Tensor":
   Fields:
     + data : SimpleTensor              [rowwise quantized data]
     + columnwise_data : SimpleTensor   [columnwise quantized data]
     + amax : SimpleTensor              [absolute maximum]
     + columnwise_amax : SimpleTensor   [columnwise amax]
     + scale : SimpleTensor             [quantization scale]
     + scale_inv : SimpleTensor         [rowwise scale inverse]
     + columnwise_scale_inv : SimpleTensor [columnwise scale inverse]
     + scaling_mode : NVTEScalingMode
     + nvte_tensor : NVTETensor (opaque handle)
     + with_gemm_swizzled_scales : bool
   Below, a smaller UML box for "SimpleTensor":
     + dptr : void*
     + shape : vector<size_t>
     + dtype : DType
   Arrow from each SimpleTensor field in Tensor to the SimpleTensor box (composition).

.. code-block:: cpp

   struct Tensor {
       SimpleTensor data;               // Rowwise data
       SimpleTensor columnwise_data;    // Columnwise data (for wgrad)
       SimpleTensor amax;               // Absolute maximum
       SimpleTensor columnwise_amax;    // Columnwise amax
       SimpleTensor scale;              // Quantization scale
       SimpleTensor scale_inv;          // Rowwise scale inverse
       SimpleTensor columnwise_scale_inv; // Columnwise scale inverse

       NVTEScalingMode scaling_mode;
       NVTETensor nvte_tensor;          // Opaque handle for C API
       bool with_gemm_swizzled_scales;  // MXFP8/NVFP4 scale format flag
   };

This structure holds all the metadata needed for a quantized tensor. Not all fields are
valid at the same time — which fields are populated depends on the ``scaling_mode`` and
the data dtype. For example, ``amax`` is only used with delayed tensor scaling and FP8
dtypes. ``scale_inv`` has shape ``[1]`` for tensor scaling but a multi-element shape for
block scaling modes. ``columnwise_data`` and ``columnwise_scale_inv`` are only populated
when the tensor was quantized with columnwise usage enabled. Validation functions in
``transformer_engine.cpp`` (e.g., ``CheckScaleTensorShape``) enforce these constraints.

.. _dtype-conversion-patterns:

DType / NVTEDType Conversion Patterns
--------------------------------------

When working across the C/C++ boundary, conversions are frequently needed. Here are the
common patterns:

**NVTEDType → DType** (e.g., reading from ``NVTEBasicTensor.dtype``):

.. code-block:: cpp

   NVTEDType nvte_type = tensor.dtype;
   DType dtype = static_cast<DType>(nvte_type);

**DType → NVTEDType** (e.g., writing to ``NVTEBasicTensor.dtype``):

.. code-block:: cpp

   DType dtype = DType::kFloat8E4M3;
   basic_tensor.dtype = static_cast<NVTEDType>(dtype);

**Comparison**: Overloaded ``==`` and ``!=`` operators exist, so direct comparison works:

.. code-block:: cpp

   if (nvte_dtype == DType::kFloat8E4M3) { ... }  // OK

**Function overloads**: Many utility functions accept both types:

.. code-block:: cpp

   // Both of these work:
   bool a = is_fp8_dtype(DType::kFloat8E4M3);
   bool b = is_fp8_dtype(kNVTEFloat8E4M3);

Overloaded functions include: ``is_fp8_dtype``, ``typeToSize``, ``typeToNumBits``,
``get_buffer_size_bytes``, ``get_cuda_dtype``, ``get_cudnn_dtype``, ``get_cudnn_fe_dtype``,
and ``to_string``.

Common Pitfalls
^^^^^^^^^^^^^^^

1. **auto deduction**: ``auto dtype = tensor.dtype;`` deduces ``NVTEDType`` for
   ``NVTEBasicTensor`` fields. If passed to a function expecting ``DType``, add an
   explicit cast.

2. **Ternary expressions**: ``condition ? nvte_val : dtype_val`` won't compile — wrap
   one side in ``static_cast``.

3. **ADL (Argument-Dependent Lookup)**: Functions in the ``transformer_engine`` namespace
   won't be found by ADL when called with ``NVTEDType`` arguments (which are in global
   scope). Use the namespace qualifier: ``transformer_engine::to_string(nvte_type)``.

4. **Domain-specific switch macros**: Some kernel areas define their own switch macros
   that restrict the set of valid types. For example, ``fused_router/utils.h`` defines
   ``TE_ROUTER_PROBS_TYPE_SWITCH_ALL`` (only FP32/FP16/BF16) and
   ``TE_ROUTER_INDEX_TYPE_SWITCH_ALL`` (only Int32/Int64/BF16/FP32). These use
   ``DType`` directly in their switch cases, so the same ``static_cast`` rules apply
   when calling them with ``NVTEDType`` values.

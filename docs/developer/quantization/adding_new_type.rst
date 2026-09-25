..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Adding a New Quantization Type
==============================

This guide walks through the steps to add a new low-precision format to Transformer
Engine, using the existing types (FP8, MXFP8, NVFP4) as templates.

Prerequisites
-------------

Before starting, you need:

- A new data type (or an existing one with a new scaling strategy).
- CUDA kernel(s) for quantization (cast) and dequantization.
- cuBLASLt support for the type in GEMM (or a custom GEMM kernel).

Step 1: Add the Data Type to the C API
---------------------------------------

If your format uses a new data type (not already in ``NVTEDType``):

1. Add an entry to the ``NVTEDType`` enum in
   ``transformer_engine/common/include/transformer_engine/transformer_engine.h``:

   .. code-block:: c

      enum NVTEDType {
          // ... existing types ...
          kNVTENewType = 11,
          kNVTENumTypes
      };

2. Add the corresponding entry to ``DType`` in ``transformer_engine/common/common.h``.

3. Add overloads for: ``typeToSize``, ``typeToNumBits``, ``is_fp8_dtype`` (if applicable),
   ``get_cuda_dtype``, ``get_cudnn_dtype``, and ``to_string``.

Step 2: Add a Scaling Mode (if needed)
---------------------------------------

If your format uses a new scaling strategy:

1. Add an entry to ``NVTEScalingMode`` in ``transformer_engine.h``.
2. Add helper functions in ``common.h`` (e.g., ``is_new_scaling()``).

Step 3: Implement Cast Kernels
-------------------------------

Create a new file or extend existing files in ``transformer_engine/common/cast/``:

1. Implement the quantization kernel (high precision → new format).
2. Implement the dequantization kernel (new format → high precision).
3. Register C API entry points in the cast header.

Step 4: Extend GEMM Support
----------------------------

In ``transformer_engine/common/gemm/``:

1. Add handling for the new type in cuBLASLt GEMM dispatch.
2. Handle scale/scale_inv for the new scaling mode.
3. Update the compute type selection logic.

Step 5: Create Python Quantizer and Tensor Classes
---------------------------------------------------

Create a new file ``transformer_engine/pytorch/tensor/new_type_tensor.py`` with:

1. **``NewTypeQuantizer(Quantizer)``**: Implements ``__call__()`` and ``quantize()``
   to produce quantized tensors in the new format.

2. **``NewTypeTensorStorage(QuantizedTensorStorage)``**: Holds the raw quantized data
   and scale inverses.

3. **``NewTypeTensor(QuantizedTensor)``**: Full ``torch.Tensor`` subclass with
   ``__torch_dispatch__`` support.

Use ``transformer_engine/pytorch/tensor/float8_tensor.py`` as a reference implementation.

Step 6: Register in the Recipe System
--------------------------------------

1. Add a recipe option that selects your new quantizer.
2. Update the recipe → quantizer mapping in the module initialization code.

Step 7: Update C++ Extensions
------------------------------

In ``transformer_engine/pytorch/cpp_extensions/``:

1. Update tensor conversion utilities to handle the new quantized tensor type.
2. Ensure ``NVTETensor`` construction properly sets scaling mode and parameters.

Step 8: Test
------------

Minimum test coverage:

- [ ] Cast round-trip: quantize → dequantize produces acceptable error.
- [ ] GEMM: FP8 GEMM with new type matches BF16 GEMM within tolerance.
- [ ] Linear forward/backward: ``te.Linear`` produces correct gradients.
- [ ] Distributed: quantized tensors survive all-gather/reduce-scatter.
- [ ] Serialization: ``torch.save`` / ``torch.load`` round-trip.

Checklist
---------

.. code-block:: text

   [ ] NVTEDType entry (if new type)
   [ ] DType entry (if new type)
   [ ] Type utility overloads (typeToSize, etc.)
   [ ] NVTEScalingMode entry (if new scaling)
   [ ] Scaling mode helpers (is_new_scaling, etc.)
   [ ] Cast kernel (quantize)
   [ ] Cast kernel (dequantize)
   [ ] GEMM support
   [ ] NewTypeQuantizer class
   [ ] NewTypeTensorStorage class
   [ ] NewTypeTensor class
   [ ] Recipe integration
   [ ] C++ extension updates
   [ ] Unit tests
   [ ] Lint passes (Black, clang-format, cpplint)

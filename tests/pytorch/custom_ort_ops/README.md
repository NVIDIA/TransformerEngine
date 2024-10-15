# Custom ONNX Runtime operations for Transformer Engine tests

This directory contains code that builds custom ONNX operators for use
in Transformer Engine tests. It includes basic, non-performant
implementations of the FP8 quantization and dequantization operators
that are used when exporting Transformer Engine models to ONNX.

For more information, see [the ONNX Runtime reference for custom
operators](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html).
Much of the code has been adapted from [an ONNX Runtime
test](https://github.com/microsoft/onnxruntime/blob/de93f40240459953a6e3bbb86b6ad83eaeab681f/onnxruntime/test/testdata/custom_op_library/custom_op_library.cc).

## Usage

* Download the ONNX Runtime source code:
```bash
$ cd TransformerEngine/tests/pytorch/custom_ort_ops
$ git clone https://github.com/microsoft/onnxruntime.git
```
* Build the library with CMake:
```bash
$ mkdir build
$ cmake -S . -B build -DCMAKE_INSTALL_PREFIX=.
$ cmake --build build
$ cmake --install build
```
* Run the ONNX test with pytest:
```bash
$ cd TransformerEngine/tests/pytorch
$ python -m pytest test_onnx_export.py
```
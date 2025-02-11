/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#define PYBIND11_DETAILED_ERROR_MESSAGES  // TODO remove

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_PYBIND_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_PYBIND_H_
#include <pybind11/detail/common.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include "common.h"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine::pytorch {

extern PyTypeObject *Float8TensorPythonClass;
extern PyTypeObject *Float8TensorBasePythonClass;
extern PyTypeObject *Float8QuantizerClass;
extern PyTypeObject *MXFP8TensorPythonClass;
extern PyTypeObject *MXFP8TensorBasePythonClass;
extern PyTypeObject *MXFP8QuantizerClass;

void init_extension();

void init_float8_extension();

void init_mxfp8_extension();

namespace detail {

inline bool IsFloat8QParams(PyObject *obj) { return Py_TYPE(obj) == Float8QuantizerClass; }

inline bool IsFloat8Tensor(PyObject *obj) {
  return Py_TYPE(obj) == Float8TensorPythonClass || Py_TYPE(obj) == Float8TensorBasePythonClass;
}

inline bool IsMXFP8QParams(PyObject *obj) { return Py_TYPE(obj) == MXFP8QuantizerClass; }

inline bool IsMXFP8Tensor(PyObject *obj) {
  return Py_TYPE(obj) == MXFP8TensorPythonClass || Py_TYPE(obj) == MXFP8TensorBasePythonClass;
}

TensorWrapper NVTETensorFromFloat8Tensor(py::handle tensor, Quantizer *quantizer);

template <typename T>
std::unique_ptr<Quantizer> CreateQuantizer(const py::handle quantizer) {
  return std::make_unique<T>(quantizer);
}

TensorWrapper NVTETensorFromMXFP8Tensor(py::handle tensor, Quantizer *quantization_params);

std::unique_ptr<Quantizer> CreateMXFP8Params(const py::handle params);

inline bool IsFloatingPointType(at::ScalarType type) {
  return type == at::kFloat || type == at::kHalf || type == at::kBFloat16;
}

constexpr std::array custom_types_converters = {
    std::make_tuple(IsFloat8Tensor, IsFloat8QParams, NVTETensorFromFloat8Tensor,
                    CreateQuantizer<Float8Quantizer>),
    std::make_tuple(IsMXFP8Tensor, IsMXFP8QParams, NVTETensorFromMXFP8Tensor,
                    CreateQuantizer<MXFP8Quantizer>)};

}  // namespace detail

}  // namespace transformer_engine::pytorch

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_PYBIND_H_

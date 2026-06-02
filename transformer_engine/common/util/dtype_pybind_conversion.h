/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_DTYPE_PYBIND_CONVERSION_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_DTYPE_PYBIND_CONVERSION_H_

#include <pybind11/pybind11.h>
#include <transformer_engine/transformer_engine.h>

#include <array>
#include <typeinfo>

namespace transformer_engine {
namespace pybind_detail {

/*! @brief Per-value cache of ``tex.DType`` enum PyObject* objects keyed by enum value. */
inline std::array<PyObject *, static_cast<size_t>(transformer_engine::DType::kNumTypes)> &
cached_dtype_objects() {
  static std::array<PyObject *, static_cast<size_t>(transformer_engine::DType::kNumTypes)> cache{};
  return cache;
}

/*! @brief Construct a ``transformer_engine_torch.DType(value)`` object once.
 * Returns a new strong reference, or ``nullptr`` (with the Python error cleared)
 * on failure.
 */
inline PyObject *construct_dtype_object(long value, PyTypeObject *type) {  // NOLINT(runtime/int)
  PyObject *arg = PyLong_FromLong(value);
  if (arg == nullptr) {
    PyErr_Clear();
    return nullptr;
  }
  PyObject *obj = PyObject_CallFunctionObjArgs(reinterpret_cast<PyObject *>(type), arg, nullptr);
  Py_DECREF(arg);
  if (obj == nullptr) {
    PyErr_Clear();
  }
  return obj;
}

/*! @brief Implicit-conversion function registered on the pybind ``DType`` enum.
 *
 * Converts a Python ``transformer_engine.pytorch.DType`` object
 * into a cached ``transformer_engine_torch.DType`` enum object.
 * This conversion function is needed since tex pybind functions generally accept
 * transformer_engine_torch.DType. This implicit conversion allows user to pass
 * transformer_engine.pytorch.DType from python and C++ functions will implicitly convert to
 * transformer_engine_torch.DType.
 */
inline PyObject *cached_int_to_dtype(PyObject *src, PyTypeObject *type) {
  // Only plain ints / IntEnum subclasses are handled here.
  // src --> transformer_engine.pytorch.DType IntEnum object
  // type --> transformer_engine_torch.DType PyTypeObject*
  if (!PyLong_Check(src)) {
    return nullptr;
  }
  const long value = PyLong_AsLong(src);  // NOLINT(runtime/int)
  if (value == -1 && PyErr_Occurred()) {
    PyErr_Clear();
    return nullptr;
  }
  if (value < 0 ||
      static_cast<size_t>(value) >= static_cast<size_t>(transformer_engine::DType::kNumTypes)) {
    return nullptr;
  }
  // cached_dtype_object --> transformer_engine_torch.DType(value) PyObject*
  PyObject *&cached_dtype_object = cached_dtype_objects()[static_cast<size_t>(value)];
  if (cached_dtype_object == nullptr) {
    cached_dtype_object = construct_dtype_object(value, type);
    if (cached_dtype_object == nullptr) {
      return nullptr;
    }
  }
  Py_INCREF(cached_dtype_object);
  return cached_dtype_object;
}

/*! @brief Register the Python -> C++ ``DType`` implicit conversion.
 * Allows a Python object of type ``transformer_engine.pytorch.DType``
 * to be passed wherever a pybind-bound ``transformer_engine::DType`` argument is expected.
 * pybind-bound ``transformer_engine::DType`` argument is expected.
 * Must be called after the pybind ``DType`` enum has been registered.
 */
inline void register_dtype_implicit_conversion() {
  auto *tinfo = pybind11::detail::get_type_info(typeid(transformer_engine::DType));
  if (tinfo != nullptr) {
    tinfo->implicit_conversions.push_back(&cached_int_to_dtype);
  }
}

}  // namespace pybind_detail
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_DTYPE_PYBIND_CONVERSION_H_

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

/*! @brief Per-value cache of ``tex.DType`` enum objects keyed by enum value. */
inline std::array<PyObject *, static_cast<size_t>(transformer_engine::DType::kNumTypes)> &
cached_dtype_objects() {
  static std::array<PyObject *, static_cast<size_t>(transformer_engine::DType::kNumTypes)> cache{};
  return cache;
}

/*! @brief Implicit-conversion function registered on the pybind ``DType`` enum.
 *
 * Converts a Python ``transformer_engine.pytorch.constants.DType`` object
 * into a cached ``transformer_engine_torch.DType`` enum object.
 * This conversion function is needed since tex pybind functions generally accept
 * transformer_engine_torch.DType. This implicit conversion allows user to pass
 * constants.DType from python and C++ functions will implicitly convert to
 * transformer_engine_torch.DType.
 */
inline PyObject *cached_int_to_dtype(PyObject *src, PyTypeObject *type) {
  // Only plain ints / IntEnum subclasses are handled here (matches the
  // behavior of pybind's ``int`` caster used by the original converter).
  if (!PyLong_Check(src)) {
    return nullptr;
  }
  const long value = PyLong_AsLong(src);  // NOLINT(runtime/int)
  if (value == -1 && PyErr_Occurred()) {
    PyErr_Clear();
    return nullptr;
  }
  if (value < 0 || value >= static_cast<long>(transformer_engine::DType::kNumTypes)) {
    return nullptr;
  }
  auto &cache = cached_dtype_objects();
  PyObject *cached = cache[static_cast<size_t>(value)];
  if (cached == nullptr) {
    // First use of this value: construct ``DType(value)`` once and keep a
    // strong reference for the lifetime of the process.
    PyObject *arg = PyLong_FromLong(value);
    if (arg == nullptr) {
      PyErr_Clear();
      return nullptr;
    }
    cached = PyObject_CallFunctionObjArgs(reinterpret_cast<PyObject *>(type), arg, nullptr);
    Py_DECREF(arg);
    if (cached == nullptr) {
      PyErr_Clear();
      return nullptr;
    }
    cache[static_cast<size_t>(value)] = cached;
  }
  Py_INCREF(cached);
  return cached;
}

/*! @brief Register the Python -> C++ ``DType`` implicit conversion.
 * Allows a Python object of type ``transformer_engine.pytorch.constants.DType``
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

/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_PYBIND_DTYPE_CASTERS_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_PYBIND_DTYPE_CASTERS_H_

#include <pybind11/pybind11.h>
#include <transformer_engine/transformer_engine.h>

#include <array>
#include <cstddef>

namespace pybind11 {
namespace detail {

/*! @brief Custom type caster for ``transformer_engine::DType``.
 *
 * pybind exposes ``transformer_engine::DType`` as the ``transformer_engine_torch.DType``
 * enum. C++ functions take ``transformer_engine::DType`` arguments, and from Python a
 * caller may pass any of the following, all of which carry the same integer dtype tag:
 *   - a ``transformer_engine.pytorch.DType`` (an ``IntEnum``; the canonical Python type), or
 *   - a plain ``int`` (the integer dtype tag), or
 *   - a ``transformer_engine_torch.DType`` (the pybind enum) --> Deprecated and supported for
 *     backward compatibility only.
 *
 * NOTE: As a compile-time specialization this must be visible in every translation unit
 * that converts ``transformer_engine::DType`` (it is pulled in via the PyTorch extension's
 * ``common.h``). Otherwise different TUs would instantiate different casters for the same
 * type, which is an ODR violation.
 */
template <>
struct type_caster<transformer_engine::DType> {
 public:
  PYBIND11_TYPE_CASTER(transformer_engine::DType, const_name("DType"));

  bool load(handle src, bool convert) {
    if (!src) {
      return false;
    }

    // Intended for ``transformer_engine.pytorch.DType``,
    // the canonical Python type. Cast it to the C++ enum value.
    if (PyLong_Check(src.ptr())) {
      const long tag = PyLong_AsLong(src.ptr());  // NOLINT(runtime/int)
      if (tag == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        return false;
      }
      if (tag < 0 ||
          static_cast<size_t>(tag) >= static_cast<size_t>(transformer_engine::DType::kNumTypes)) {
        return false;
      }
      value = static_cast<transformer_engine::DType>(tag);
      return true;
    }

    // Actual ``transformer_engine_torch.DType`` enum instances (and anything else): defer to
    // the standard caster, which reads the value out of the instance with no Python round-trip.
    type_caster_base<transformer_engine::DType> base;
    if (base.load(src, convert)) {
      value = *static_cast<transformer_engine::DType *>(base);
      return true;
    }

    return false;
  }

  static handle cast(transformer_engine::DType src, return_value_policy policy, handle parent) {
    return type_caster_base<transformer_engine::DType>::cast(src, policy, parent);
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_PYBIND_DTYPE_CASTERS_H_

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
#include <iterator>
#include <stdexcept>

namespace transformer_engine::pytorch::detail {

struct DTypeEntry {
  const char* name;
  DType value;
};

inline constexpr DTypeEntry dtype_entries[] = {
    {"kByte", DType::kByte},
    {"kInt16", DType::kInt16},
    {"kInt32", DType::kInt32},
    {"kInt64", DType::kInt64},
    {"kFloat32", DType::kFloat32},
    {"kFloat16", DType::kFloat16},
    {"kBFloat16", DType::kBFloat16},
    {"kFloat8E4M3", DType::kFloat8E4M3},
    {"kFloat8E5M2", DType::kFloat8E5M2},
    {"kFloat8E8M0", DType::kFloat8E8M0},
    {"kFloat4E2M1", DType::kFloat4E2M1},
    {"kFloat8UE5M3", DType::kFloat8UE5M3},
};

static_assert(std::size(dtype_entries) == static_cast<size_t>(DType::kNumTypes),
              "Update the Python DType and caster when adding a C++ DType");
static_assert(
    [] {
      for (size_t index = 0; index < std::size(dtype_entries); ++index) {
        if (static_cast<size_t>(dtype_entries[index].value) != index) return false;
      }
      return true;
    }(),
    "DType entries must be indexed by their C++ values");

inline std::array<PyObject*, std::size(dtype_entries)> dtype_members{};
inline PyTypeObject* dtype_type = nullptr;

inline void InitializeDTypeCaster(pybind11::module_& module) {
  namespace py = pybind11;
  const auto dtype_class = py::module_::import("transformer_engine.common.dtype").attr("DType");
  const auto members = py::dict(dtype_class.attr("__members__"));
  const auto mismatch = [] {
    throw std::runtime_error(
        "Transformer Engine Python DType is out of sync with C++; "
        "rebuild the extension with matching Python sources.");
  };
  if (members.size() != std::size(dtype_entries)) mismatch();

  py::tuple cache(std::size(dtype_entries));
  for (size_t index = 0; index < std::size(dtype_entries); ++index) {
    const auto& entry = dtype_entries[index];
    if (!members.contains(entry.name)) mismatch();
    const auto member = members[entry.name];
    if (Py_TYPE(member.ptr()) != reinterpret_cast<PyTypeObject*>(dtype_class.ptr()) ||
        !PyLong_Check(member.ptr()) || member.cast<int>() != static_cast<int>(entry.value)) {
      mismatch();
    }
    cache[index] = member;
  }
  module.attr("_dtype_members") = cache;
  for (size_t index = 0; index < dtype_members.size(); ++index) {
    dtype_members[index] = cache[index].ptr();
  }
  dtype_type = reinterpret_cast<PyTypeObject*>(dtype_class.ptr());
}

}  // namespace transformer_engine::pytorch::detail

namespace pybind11::detail {

template <>
struct type_caster<transformer_engine::DType> {
 public:
  PYBIND11_TYPE_CASTER(transformer_engine::DType, const_name("transformer_engine.pytorch.DType"));

  bool load(handle src, bool) {
    if (!src || Py_TYPE(src.ptr()) != transformer_engine::pytorch::detail::dtype_type) {
      return false;
    }
    const auto tag = PyLong_AsLong(src.ptr());
    if (tag == -1 && PyErr_Occurred()) {
      PyErr_Clear();
      return false;
    }
    if (static_cast<size_t>(tag) >= static_cast<size_t>(transformer_engine::DType::kNumTypes)) {
      return false;
    }
    value = static_cast<transformer_engine::DType>(tag);
    return true;
  }

  static handle cast(transformer_engine::DType src, return_value_policy, handle) {
    const auto index = static_cast<size_t>(src);
    const auto& members = transformer_engine::pytorch::detail::dtype_members;
    if (index >= members.size() || members[index] == nullptr) {
      throw value_error("Invalid or uninitialized Transformer Engine DType");
    }
    return handle(members[index]).inc_ref();
  }
};

}  // namespace pybind11::detail

#endif

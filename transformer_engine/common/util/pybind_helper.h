/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_PYBIND_HELPER_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_PYBIND_HELPER_H_

#include <pybind11/pybind11.h>
#include <transformer_engine/comm_gemm_overlap.h>
#include <transformer_engine/fused_attn.h>
#include <transformer_engine/transformer_engine.h>

#include <array>
#include <typeinfo>

#include "cuda_runtime.h"

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
 * Converts a Python ``int`` (including ``IntEnum`` subclasses such as
 * ``transformer_engine.pytorch.constants.DType``) into a cached
 * ``transformer_engine_torch.DType`` enum object. This is a drop-in,
 * behavior-preserving replacement for the converter registered by
 * ``pybind11::implicitly_convertible<int, transformer_engine::DType>()``,
 * except that it returns a cached singleton per value instead of allocating
 * a fresh ``DType(value)`` object on every call.
 *
 * ``tex.DType`` arguments never reach this path: they match the registered
 * enum type directly and skip implicit conversion entirely.
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

/*! @brief Register the cached int -> ``DType`` implicit conversion.
 *
 * Must be called after the pybind ``DType`` enum has been registered.
 */
inline void register_cached_dtype_implicit_conversion() {
  auto *tinfo = pybind11::detail::get_type_info(typeid(transformer_engine::DType));
  if (tinfo != nullptr) {
    tinfo->implicit_conversions.push_back(&cached_int_to_dtype);
  }
}

}  // namespace pybind_detail
}  // namespace transformer_engine

#define NVTE_DECLARE_COMMON_PYBIND11_HANDLES(m)                                                                 \
  pybind11::enum_<transformer_engine::DType>(m, "DType", pybind11::module_local())                              \
      .value("kByte", transformer_engine::DType::kByte)                                                         \
      .value("kInt32", transformer_engine::DType::kInt32)                                                       \
      .value("kFloat32", transformer_engine::DType::kFloat32)                                                   \
      .value("kFloat16", transformer_engine::DType::kFloat16)                                                   \
      .value("kBFloat16", transformer_engine::DType::kBFloat16)                                                 \
      .value("kFloat8E4M3", transformer_engine::DType::kFloat8E4M3)                                             \
      .value("kFloat8E5M2", transformer_engine::DType::kFloat8E5M2)                                             \
      .value("kFloat4E2M1", transformer_engine::DType::kFloat4E2M1)                                             \
      .def("__reduce_ex__",                                                                                     \
           [](transformer_engine::DType self, pybind11::object /*protocol*/) {                                  \
             return pybind11::make_tuple(pybind11::type::of(pybind11::cast(self)),                              \
                                         pybind11::make_tuple(static_cast<int>(self)));                         \
           })                                                                                                   \
      .def("__reduce__", [](transformer_engine::DType self) {                                                   \
        return pybind11::make_tuple(pybind11::type::of(pybind11::cast(self)),                                   \
                                    pybind11::make_tuple(static_cast<int>(self)));                              \
      }); /* Allow Python int (and IntEnum subclasses like transformer_engine.pytorch.constants.DType) to be */ \
  /* passed wherever a pybind-bound ``transformer_engine::DType`` argument is expected. Uses a    */            \
  /* cached converter (returns a singleton per value) instead of the per-call allocation that     */           \
  /* ``pybind11::implicitly_convertible<int, transformer_engine::DType>()`` would incur.           */           \
  transformer_engine::pybind_detail::register_cached_dtype_implicit_conversion();                              \
  pybind11::enum_<NVTE_Bias_Type>(m, "NVTE_Bias_Type", pybind11::module_local())                                \
      .value("NVTE_NO_BIAS", NVTE_Bias_Type::NVTE_NO_BIAS)                                                      \
      .value("NVTE_PRE_SCALE_BIAS", NVTE_Bias_Type::NVTE_PRE_SCALE_BIAS)                                        \
      .value("NVTE_POST_SCALE_BIAS", NVTE_Bias_Type::NVTE_POST_SCALE_BIAS)                                      \
      .value("NVTE_ALIBI", NVTE_Bias_Type::NVTE_ALIBI);                                                         \
  pybind11::enum_<NVTE_Mask_Type>(m, "NVTE_Mask_Type", pybind11::module_local())                                \
      .value("NVTE_NO_MASK", NVTE_Mask_Type::NVTE_NO_MASK)                                                      \
      .value("NVTE_PADDING_MASK", NVTE_Mask_Type::NVTE_PADDING_MASK)                                            \
      .value("NVTE_CAUSAL_MASK", NVTE_Mask_Type::NVTE_CAUSAL_MASK)                                              \
      .value("NVTE_PADDING_CAUSAL_MASK", NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK)                              \
      .value("NVTE_CAUSAL_BOTTOM_RIGHT_MASK", NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK)                    \
      .value("NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK",                                                           \
             NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK);                                            \
  pybind11::enum_<NVTE_Softmax_Type>(m, "NVTE_Softmax_Type", pybind11::module_local())                          \
      .value("NVTE_VANILLA_SOFTMAX", NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX)                                   \
      .value("NVTE_OFF_BY_ONE_SOFTMAX", NVTE_Softmax_Type::NVTE_OFF_BY_ONE_SOFTMAX)                             \
      .value("NVTE_LEARNABLE_SOFTMAX", NVTE_Softmax_Type::NVTE_LEARNABLE_SOFTMAX);                              \
  pybind11::enum_<NVTE_QKV_Format>(m, "NVTE_QKV_Format", pybind11::module_local())                              \
      .value("NVTE_BSHD", NVTE_QKV_Format::NVTE_BSHD)                                                           \
      .value("NVTE_SBHD", NVTE_QKV_Format::NVTE_SBHD)                                                           \
      .value("NVTE_THD", NVTE_QKV_Format::NVTE_THD)                                                             \
      .value("NVTE_SBHD_2BSHD", NVTE_QKV_Format::NVTE_SBHD_2BSHD)                                               \
      .value("NVTE_BSHD_2SBHD", NVTE_QKV_Format::NVTE_BSHD_2SBHD)                                               \
      .value("NVTE_THD_2BSHD", NVTE_QKV_Format::NVTE_THD_2BSHD)                                                 \
      .value("NVTE_THD_2SBHD", NVTE_QKV_Format::NVTE_THD_2SBHD)                                                 \
      .value("NVTE_BHSD", NVTE_QKV_Format::NVTE_BHSD)                                                           \
      .value("NVTE_QKV_Format_NOT_SET", NVTE_QKV_Format::NVTE_QKV_Format_NOT_SET);                              \
  pybind11::enum_<NVTE_QKV_Layout>(m, "NVTE_QKV_Layout", pybind11::module_local())                              \
      .value("NVTE_SB3HD", NVTE_QKV_Layout::NVTE_SB3HD)                                                         \
      .value("NVTE_SBH3D", NVTE_QKV_Layout::NVTE_SBH3D)                                                         \
      .value("NVTE_SBHD_SB2HD", NVTE_QKV_Layout::NVTE_SBHD_SB2HD)                                               \
      .value("NVTE_SBHD_SBH2D", NVTE_QKV_Layout::NVTE_SBHD_SBH2D)                                               \
      .value("NVTE_SBHD_SBHD_SBHD", NVTE_QKV_Layout::NVTE_SBHD_SBHD_SBHD)                                       \
      .value("NVTE_BS3HD", NVTE_QKV_Layout::NVTE_BS3HD)                                                         \
      .value("NVTE_BSH3D", NVTE_QKV_Layout::NVTE_BSH3D)                                                         \
      .value("NVTE_BSHD_BS2HD", NVTE_QKV_Layout::NVTE_BSHD_BS2HD)                                               \
      .value("NVTE_BSHD_BSH2D", NVTE_QKV_Layout::NVTE_BSHD_BSH2D)                                               \
      .value("NVTE_BSHD_BSHD_BSHD", NVTE_QKV_Layout::NVTE_BSHD_BSHD_BSHD)                                       \
      .value("NVTE_T3HD", NVTE_QKV_Layout::NVTE_T3HD)                                                           \
      .value("NVTE_TH3D", NVTE_QKV_Layout::NVTE_TH3D)                                                           \
      .value("NVTE_THD_T2HD", NVTE_QKV_Layout::NVTE_THD_T2HD)                                                   \
      .value("NVTE_THD_TH2D", NVTE_QKV_Layout::NVTE_THD_TH2D)                                                   \
      .value("NVTE_THD_THD_THD", NVTE_QKV_Layout::NVTE_THD_THD_THD)                                             \
      .value("NVTE_SBHD_BSHD_BSHD", NVTE_QKV_Layout::NVTE_SBHD_BSHD_BSHD)                                       \
      .value("NVTE_BSHD_SBHD_SBHD", NVTE_QKV_Layout::NVTE_BSHD_SBHD_SBHD)                                       \
      .value("NVTE_THD_BSHD_BSHD", NVTE_QKV_Layout::NVTE_THD_BSHD_BSHD)                                         \
      .value("NVTE_THD_SBHD_SBHD", NVTE_QKV_Layout::NVTE_THD_SBHD_SBHD)                                         \
      .value("NVTE_Paged_KV_BSHD_BSHD_BSHD", NVTE_QKV_Layout::NVTE_Paged_KV_BSHD_BSHD_BSHD)                     \
      .value("NVTE_Paged_KV_BSHD_SBHD_SBHD", NVTE_QKV_Layout::NVTE_Paged_KV_BSHD_SBHD_SBHD)                     \
      .value("NVTE_Paged_KV_SBHD_BSHD_BSHD", NVTE_QKV_Layout::NVTE_Paged_KV_SBHD_BSHD_BSHD)                     \
      .value("NVTE_Paged_KV_SBHD_SBHD_SBHD", NVTE_QKV_Layout::NVTE_Paged_KV_SBHD_SBHD_SBHD)                     \
      .value("NVTE_Paged_KV_THD_BSHD_BSHD", NVTE_QKV_Layout::NVTE_Paged_KV_THD_BSHD_BSHD)                       \
      .value("NVTE_Paged_KV_THD_SBHD_SBHD", NVTE_QKV_Layout::NVTE_Paged_KV_THD_SBHD_SBHD)                       \
      .value("NVTE_BHSD_BHSD_BHSD", NVTE_QKV_Layout::NVTE_BHSD_BHSD_BHSD);                                      \
  pybind11::enum_<NVTE_Fused_Attn_Backend>(m, "NVTE_Fused_Attn_Backend", pybind11::module_local())              \
      .value("NVTE_F16_arbitrary_seqlen", NVTE_Fused_Attn_Backend::NVTE_F16_arbitrary_seqlen)                   \
      .value("NVTE_FP8", NVTE_Fused_Attn_Backend::NVTE_FP8)                                                     \
      .value("NVTE_No_Backend", NVTE_Fused_Attn_Backend::NVTE_No_Backend);                                      \
  pybind11::enum_<transformer_engine::Float8BlockScaleTensorFormat>(                                            \
      m, "Float8BlockScaleTensorFormat", pybind11::module_local())                                              \
      .value("GEMM_READY", transformer_engine::Float8BlockScaleTensorFormat::GEMM_READY)                        \
      .value("COMPACT", transformer_engine::Float8BlockScaleTensorFormat::COMPACT)                              \
      .value("INVALID", transformer_engine::Float8BlockScaleTensorFormat::INVALID);                             \
  pybind11::enum_<transformer_engine::CommOverlapType>(m, "CommOverlapType",                                    \
                                                       pybind11::module_local())                                \
      .value("RS", transformer_engine::CommOverlapType::RS)                                                     \
      .value("AG", transformer_engine::CommOverlapType::AG);                                                    \
  pybind11::enum_<transformer_engine::CommOverlapAlgo>(m, "CommOverlapAlgo",                                    \
                                                       pybind11::module_local())                                \
      .value("BULK_OVERLAP_AG", transformer_engine::CommOverlapAlgo::BULK_OVERLAP_AG)                           \
      .value("BULK_OVERLAP_RS", transformer_engine::CommOverlapAlgo::BULK_OVERLAP_RS)                           \
      .value("SPLIT_PIPELINED_AG_P2P",                                                                          \
             transformer_engine::CommOverlapAlgo::SPLIT_PIPELINED_AG_P2P)                                       \
      .value("SPLIT_PIPELINED_RS", transformer_engine::CommOverlapAlgo::SPLIT_PIPELINED_RS)                     \
      .value("SPLIT_PIPELINED_RS_P2P",                                                                          \
             transformer_engine::CommOverlapAlgo::SPLIT_PIPELINED_RS_P2P)                                       \
      .value("ATOMIC_GEMM_RS", transformer_engine::CommOverlapAlgo::ATOMIC_GEMM_RS)                             \
      .value("ATOMIC_GEMM_AG_P2P", transformer_engine::CommOverlapAlgo::ATOMIC_GEMM_AG_P2P)                     \
      .value("ATOMIC_GEMM_RS_P2P", transformer_engine::CommOverlapAlgo::ATOMIC_GEMM_RS_P2P)                     \
      .value("EXTERNAL_BULK_OVERLAP_AG",                                                                        \
             transformer_engine::CommOverlapAlgo::EXTERNAL_BULK_OVERLAP_AG);                                    \
  py::class_<transformer_engine::CommOverlapCore,                                                               \
             std::shared_ptr<transformer_engine::CommOverlapCore>>(m, "CommOverlapCore",                        \
                                                                   pybind11::module_local())                    \
      .def(py::init([]() { return new transformer_engine::CommOverlapCore(); }),                                \
           py::call_guard<py::gil_scoped_release>())                                                            \
      .def("is_atomic_gemm", &transformer_engine::CommOverlapCore::is_atomic_gemm,                              \
           py::call_guard<py::gil_scoped_release>())                                                            \
      .def("is_p2p_overlap", &transformer_engine::CommOverlapCore::is_p2p_overlap,                              \
           py::call_guard<py::gil_scoped_release>())                                                            \
      .def("is_fp8_ubuf", &transformer_engine::CommOverlapCore::is_fp8_ubuf,                                    \
           py::call_guard<py::gil_scoped_release>());                                                           \
  py::class_<transformer_engine::CommOverlapBase,                                                               \
             std::shared_ptr<transformer_engine::CommOverlapBase>,                                              \
             transformer_engine::CommOverlapCore>(m, "CommOverlapBase", pybind11::module_local())               \
      .def(py::init([]() { return new transformer_engine::CommOverlapBase(); }),                                \
           py::call_guard<py::gil_scoped_release>());                                                           \
  py::class_<transformer_engine::CommOverlapP2PBase,                                                            \
             std::shared_ptr<transformer_engine::CommOverlapP2PBase>,                                           \
             transformer_engine::CommOverlapCore>(m, "CommOverlapP2PBase",                                      \
                                                  pybind11::module_local())                                     \
      .def(py::init([]() { return new transformer_engine::CommOverlapP2PBase(); }),                             \
           py::call_guard<py::gil_scoped_release>());                                                           \
  m.def("device_supports_multicast", &transformer_engine::cuda::supports_multicast,                             \
        py::call_guard<py::gil_scoped_release>(), py::arg("device_id") = -1);                                   \
  m.def(                                                                                                        \
      "get_stream_priority_range",                                                                              \
      [](int device_id = -1) {                                                                                  \
        int low_pri, high_pri;                                                                                  \
        transformer_engine::cuda::stream_priority_range(&low_pri, &high_pri, device_id);                        \
        return std::make_pair(low_pri, high_pri);                                                               \
      },                                                                                                        \
      py::call_guard<py::gil_scoped_release>(), py::arg("device_id") = -1);                                     \
  m.def("ubuf_built_with_mpi", &transformer_engine::ubuf_built_with_mpi,                                        \
        py::call_guard<py::gil_scoped_release>());

#endif

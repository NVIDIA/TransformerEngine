/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_STABLE_TENSOR_CASTER_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_STABLE_TENSOR_CASTER_H_

/* In the default (non-stable) build torch_compat::Tensor is at::Tensor and
 * torch's own pybind caster applies; this caster exists only in stable mode. */
#ifdef NVTE_WITH_TORCH_STABLE

#include <Python.h>
#include <pybind11/pybind11.h>

#include "../torch_compat.h"
#include "common/util/logging.h"

namespace pybind11 {
namespace detail {

/*! @brief Custom type caster for ``torch::stable::Tensor``.
 *
 * Lets pybind-bound functions take/return ``torch::stable::Tensor`` directly:
 * a ``torch.Tensor`` argument is unwrapped into a stable tensor sharing the
 * same TensorImpl, and a returned stable tensor is wrapped back into a
 * ``torch.Tensor``.
 *
 * NOTE: As a compile-time specialization this must be visible in every
 * translation unit that converts ``torch::stable::Tensor`` (it is pulled in
 * via the PyTorch extension's ``common.h``), otherwise different TUs would
 * instantiate different casters for the same type (ODR violation).
 */
template <>
struct type_caster<torch::stable::Tensor> {
 public:
  PYBIND11_TYPE_CASTER(torch::stable::Tensor, const_name("torch.Tensor"));

  static bool is_tensor(PyObject *obj) {
    static PyObject *tensor_type = [] {
      PyObject *mod = PyImport_ImportModule("torch");
      NVTE_CHECK(mod != nullptr, "Could not import torch");
      PyObject *type = PyObject_GetAttrString(mod, "Tensor");
      Py_DECREF(mod);
      NVTE_CHECK(type != nullptr, "Could not get torch.Tensor");
      return type;
    }();
    return PyObject_IsInstance(obj, tensor_type) == 1;
  }

  bool load(handle src, bool) {
    if (!src || !is_tensor(src.ptr())) {
      return false;
    }
    value = torch::stable::tensor_from_pyobject(src.ptr());
    return true;
  }

  static handle cast(const torch::stable::Tensor &src, return_value_policy, handle) {
    return handle(static_cast<PyObject *>(torch::stable::tensor_to_pyobject(src)));
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // NVTE_WITH_TORCH_STABLE

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_STABLE_TENSOR_CASTER_H_

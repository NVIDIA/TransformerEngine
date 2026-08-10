/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_STABLE_TENSOR_CASTER_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_EXTENSIONS_STABLE_TENSOR_CASTER_H_

/* In the default (non-stable) build torch_stable::Tensor is at::Tensor and
 * torch's own pybind caster applies; this caster exists only in stable mode. */
#ifdef NVTE_WITH_TORCH_STABLE

#include <pybind11/pybind11.h>

#include "../torch_stable.h"

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

  bool load(handle src, bool) {
    if (!src || !torch::stable::is_tensor_pyobject(src.ptr())) {
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

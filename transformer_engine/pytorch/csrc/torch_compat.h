/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_TORCH_COMPAT_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_TORCH_COMPAT_H_

#include <Python.h>
#include <cuda_runtime.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#ifdef NVTE_WITH_TORCH_STABLE
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/pyobject.h>
#else
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#endif

#include "common/util/logging.h"

/* Compatibility layer for the incremental migration to the torch stable ABI.
 * Code is written against the torch::stable surface. With NVTE_WITH_TORCH_STABLE
 * these helpers forward to the stable shims (torch >= 2.14); without it the same
 * functionality is polyfilled with the full torch ABI, so the non-stable build
 * keeps supporting older torch versions. */
namespace transformer_engine::pytorch::torch_compat {

inline bool is_tensor_pyobject(PyObject *obj) {
#ifdef NVTE_WITH_TORCH_STABLE
  static PyObject *tensor_type = [] {
    PyObject *mod = PyImport_ImportModule("torch");
    NVTE_CHECK(mod != nullptr, "Could not import torch");
    PyObject *type = PyObject_GetAttrString(mod, "Tensor");
    Py_DECREF(mod);
    NVTE_CHECK(type != nullptr, "Could not get torch.Tensor");
    return type;
  }();
  return PyObject_IsInstance(obj, tensor_type) == 1;
#else
  return THPVariable_Check(obj);
#endif
}

/* Borrowed torch.Tensor PyObject -> stable Tensor sharing the TensorImpl.
 * The GIL must be held. */
inline torch::stable::Tensor tensor_from_pyobject(PyObject *obj) {
#ifdef NVTE_WITH_TORCH_STABLE
  return torch::stable::tensor_from_pyobject(obj);
#else
  return torch::stable::Tensor(
      torch::aot_inductor::new_tensor_handle(at::Tensor(THPVariable_Unpack(obj))));
#endif
}

/* Stable Tensor -> new-reference torch.Tensor PyObject. The GIL must be held. */
inline PyObject *tensor_to_pyobject(const torch::stable::Tensor &tensor) {
#ifdef NVTE_WITH_TORCH_STABLE
  return static_cast<PyObject *>(torch::stable::tensor_to_pyobject(tensor));
#else
  return THPVariable_Wrap(*torch::aot_inductor::tensor_handle_to_tensor_pointer(tensor.get()));
#endif
}

inline cudaStream_t getCurrentCUDAStream() {
#ifdef NVTE_WITH_TORCH_STABLE
  return static_cast<cudaStream_t>(
      torch::stable::accelerator::getCurrentStream(
          torch::stable::accelerator::getCurrentDeviceIndex())
          .nativeHandle());
#else
  return at::cuda::getCurrentCUDAStream();
#endif
}

}  // namespace transformer_engine::pytorch::torch_compat

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_TORCH_COMPAT_H_

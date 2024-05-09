/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_USERBUFFERS_DLPACK_HELPER_H_
#define TRANSFORMER_ENGINE_COMMON_USERBUFFERS_DLPACK_HELPER_H_

#include <cassert>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <dlpack/dlpack.h>
#include <pybind11/pybind11.h>

#include <transformer_engine/transformer_engine.h>
#include "logging.h"
#include "../common.h"

namespace py = pybind11;

namespace transformer_engine {

DType get_te_dtype(DLDataType dtype) {
  NVTE_CHECK(dtype.lanes == 1, "Unsupported number of data lanes: ", dtype.lanes);
  if (dtype.code == kDLFloat) {
    if (dtype.bits == 16) {
      return DType::kFloat16;
    } else if (dtype.bits == 32) {
      return DType::kFloat32;
    } else {
      NVTE_ERROR("Unsupported ", dtype.bits, "-bit float type.");
    }
  } else if (dtype.code == kDLBfloat) {
    NVTE_CHECK(dtype.bits == 16, "BFloat16 type must be 16-bits.");
    return DType::kBFloat16;
  } else if (dtype.code == kDLInt) {
    if (dtype.bits == 32) {
      return DType::kInt32;
    } else if (dtype.bits == 64) {
      return DType::kInt64;
    } else {
      NVTE_ERROR("Unsupported ", dtype.bits, "-bit int type.");
    }
  } else {
    NVTE_CHECK(dtype.bits == 8, "Unsupported ", dtype.bits, "-bit data type.");
    return DType::kByte;
  }
}

template <typename T>
DLDataType get_dlpack_dtype() {
  DLDataType dl_dtype{};
  dl_dtype.lanes = 1;
  dl_dtype.bits = sizeof(T) * 8;
  if (std::is_same<T, float>() || std::is_same<T, half>()) {
    dl_dtype.code = kDLFloat;
  } else if (std::is_same<T, nv_bfloat16>()) {
    dl_dtype.code = kDLBfloat;
  } else if (std::is_same<T, int32_t>() || std::is_same<T, int64_t>()) {
    dl_dtype.code = kDLInt;
  } else {
    NVTE_CHECK(dl_dtype.bits == 8, "Unsupported ", dl_dtype.bits, "-bit data type.");
    dl_dtype.code = kDLUInt;
  }
  return dl_dtype;
}

DLDataType get_dlpack_dtype(DType dtype) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(dtype, D,
    return get_dlpack_dtype<D>();
  )
}

static void dlpack_capsule_deleter(PyObject *self) {
  if (PyCapsule_IsValid(self, "used_dltensor")) {
    return;   // data in capsule is in-use so we cannot delete it
  }

  // grab any exceptions that may be in-flight
  PyObject *type, *value, *traceback;
  PyErr_Fetch(&type, &value, &traceback);

  DLManagedTensor *managed = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(self, "dltensor"));
  if (managed == NULL) {
    // tensor manager has no deleter so just move on
    PyErr_WriteUnraisable(self);
    goto done;
  }
  if (managed->deleter) {
    // attempt to call the deleter set by the framework
    managed->deleter(managed);
    assert(!PyErr_Occurred());
  }

done:
  // restore any in-flight exceptions we might have grabbed earlier
  PyErr_Restore(type, value, traceback);
}

template <typename T = char>
DLManagedTensor * buffer_to_dlpack(void *data, int64_t bytes, int device_id = -1) {
  // Convert data type and compute number of tensor elements
  DLDataType dl_dtype = get_dlpack_dtype<T>();
  assert(bytes % dl_dtype.bits == 0);
  int64_t numel = bytes / (dl_dtype.bits / 8);

  // Create and return the dlpack tensor structure
  DLManagedTensor *dlmt = new DLManagedTensor{};
  DLTensor dl_tensor = dlmt->dl_tensor;
  dl_tensor.data = data;
  dl_tensor.dtype = dl_dtype;
  dl_tensor.shape = &numel;
  dl_tensor.ndim = 1;
  if (device_id < 0) {
    dl_tensor.device = {kDLCPU, 0};
  } else {
    dl_tensor.device = {kDLCUDA, device_id};
  }
  return dlmt;
}

DLManagedTensor * buffer_to_dlpack(void *data, int64_t bytes, DType dtype, int device_id = -1) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(dtype, D,
    return buffer_to_dlpack<D>(data, bytes, device_id);
  )
}

py::capsule dlpack_to_capsule(DLManagedTensor *dlmt) {
  return py::capsule(dlmt, "dltensor", &dlpack_capsule_deleter);
}

template <typename T = char>
py::capsule buffer_to_capsule(void *data, int64_t bytes, int device_id = -1) {
  return dlpack_to_capsule(buffer_to_dlpack<T>(data, bytes, device_id));
}

py::capsule buffer_to_capsule(void *data, int64_t bytes, DType dtype, int device_id = -1) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(dtype, D,
    return buffer_to_capsule<D>(data, bytes, device_id);
  )
}

DLManagedTensor * capsule_to_dlpack(py::capsule *capsule) {
  if (strcmp(capsule->name(), "used_dltensor") != 0) {
    // something else is already using the data in the capsule
    return nullptr;
  }
  capsule->set_name("used_dltensor");
  return capsule->get_pointer<DLManagedTensor>();
}

int64_t dlpack_to_buffer(DLManagedTensor *dlmt, void **buffer) {
  *buffer = dlmt->dl_tensor.data;
  int64_t numel = std::reduce(dlmt->dl_tensor.shape, dlmt->dl_tensor.shape + dlmt->dl_tensor.ndim,
                              1, std::multiplies<int64_t>{});
  int64_t element_size = dlmt->dl_tensor.dtype.bits / 8;
  int64_t bytes = numel * element_size;
  return bytes;
}

int64_t capsule_to_buffer(py::capsule *capsule, void **buffer) {
  return dlpack_to_buffer(capsule_to_dlpack(capsule), buffer);
}

}  // namespace transformer_engine

#endif

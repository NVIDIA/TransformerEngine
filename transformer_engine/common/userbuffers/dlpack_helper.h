/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_USERBUFFERS_DLPACK_HELPER_H_
#define TRANSFORMER_ENGINE_COMMON_USERBUFFERS_DLPACK_HELPER_H_

#include <cassert>
#include <typeinfo>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <dlpack/dlpack.h>
#include <pybind11/pybind11.h>

#include "transformer_engine/transformer_engine.h"
#include "../common.h"
#include "../util/logging.h"

namespace py = pybind11;

namespace transformer_engine {

namespace userbuffers {

typedef enum _RawBufferType {
  HOST = 0,
  DEVICE = 1,
  PINNED = 2
}  RawBufferType;

template <typename T>
DLDataType get_dlpack_dtype(int dtype = -1) {
  if (typeid(T) == typeid(DType)) {
    NVTE_CHECK((dtype >= 0) && (dtype < static_cast<int>(DType::kNumTypes)),
      "Missing/invalid NVTEDtype in arguments when templating type conversion with <NVTEDtype>.");
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(static_cast<DType>(dtype), D,
      return get_dlpack_dtype<D>();
    )
  }
  DLDataType dl_dtype{};
  dl_dtype.lanes = 1;
  dl_type.bits = sizeof(T) * 8;
  switch (typeid(static_cast<T>(1))) {
    case typeid(static_cast<double>(1)):
    case typeid(static_cast<float>(1)):
    case typeid(static_cast<half>(1)):
      dl_dtype.code = kDLFloat;
      break;

    case typeid(static_cast<nv_bfloat16>(1)):
      dl_dtype.code = kDLBfloat;
      break;

    case typeid(static_cast<bool>(1)):
      dl_dtype.code = kDLBool;
      break;

    case typeid(static_cast<int8_t>(1)):
    case typeid(static_cast<int16_t>(1)):
    case typeid(static_cast<int32_t>(1)):
    case typeid(static_cast<int64_t>(1)):
      dl_dtype.code = kDLInt;
      break;

    // Everything else is treated as an unsigned integer
    // NOTE: This includes fp8 dtypes because not all frameworks natively support fp8
    default:
      dl_dtype.code = kDLUInt;
      break;
  }
  return dl_dtype;
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
py::capsule buffer_to_dlpack(void *data, int64_t bytes, int device_id = -1, int dtype = -1) {
  // Convert data type and compute number of tensor elements
  DLDataType dl_dtype = get_dlpack_dtype<T>(dtype);
  assert(bytes % dl_dtype.bits == 0);
  int64_t numel = bytes / (dl_dtype.bits / 8);

  // Create and return the dlpack tensor structure
  DLManagedTensor dlmt{};
  DLTensor dl_tensor = dlmt.dl_tensor;
  dl_tensor.data = data;
  dl_tensor.dtype = dl_dtype;
  dl_tensor.shape = &numel;
  dl_tensor.ndim = 1;
  if (device_id < 0) {
    dl_tensor.device = {kDLCPU, 0};
  } else {
    dl_tensor.device = {kDLCUDA, device_id};
  }
  return py::capsule(&dlmt, "dltensor", &dlpack_capsule_deleter);
}

int64_t dlpack_to_buffer(const py::capsule &capsule, void **buffer) {
  if (strcmp(capsule.name(), "used_dltensor") != 0) {
    // something else is already using the data in the capsule
    return -1;
  }
  DLManagedTensor *dlmt = capsule.get_pointer<DLManagedTensor>();
  capsule.set_name("used_dltensor");
  *buffer = dlmt->dl_tensor.data;
  return dlmt->dl_tensor.shape[0] * (dlmt->dl_tensor.dtype.bits / 8);
}

}  // namespace userbuffers

}  // namespace transformer_engine

#endif

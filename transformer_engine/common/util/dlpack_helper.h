/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_DLPACK_HELPER_H
#define TRANSFORMER_ENGINE_COMMON_UTIL_DLPACK_HELPER_H

#include <dlpack/dlpack.h>
#include <pybind11/pybind11.h>
#include <transformer_engine/transformer_engine.h>

#include "cuda_runtime.h"
#include "logging.h"

namespace transformer_engine {

DLDataType nvte_dtype_to_dldtype(DType dtype) {
  DLDataType dldtype;
  dldtype.lanes = 1;
  switch (dtype) {
    case DType::kInt64:
      dldtype.bits = 64;
      dldtype.code = DLDataTypeCode::kDLInt;
      break;

    case DType::kInt32:
      dldtype.bits = 32;
      dldtype.code = DLDataTypeCode::kDLInt;
      break;

    case DType::kByte:
      dldtype.bits = 8;
      dldtype.code = DLDataTypeCode::kDLUInt;
      break;

    case DType::kFloat32:
      dldtype.bits = 32;
      dldtype.code = DLDataTypeCode::kDLFloat;
      break;

    case DType::kFloat16:
      dldtype.bits = 16;
      dldtype.code = DLDataTypeCode::kDLFloat;
      break;

    case DType::kBFloat16:
      dldtype.bits = 16;
      dldtype.code = DLDataTypeCode::kDLBfloat;
      break;

    case DType::kFloat8E4M3:
      dldtype.bits = 8;
      dldtype.code = DLDataTypeCode::kDLFloat;
      break;

    case DType::kFloat8E5M2:
      dldtype.bits = 8;
      dldtype.code = DLDataTypeCode::kDLFloat;
      break;

    default:
      NVTE_ERROR("Unrecognized transformer_engine::DType.");
  }
  return dldtype;
}

DType dldtype_to_nvte_dtype(const DLDataType &dldtype, bool grad) {
  NVTE_CHECK(dldtype.lanes == 1, "Unsupported number of lanes in DLDataType: ", dldtype.lanes);

  switch (dldtype.code) {
    case DLDataTypeCode::kDLInt:
      switch (dldtype.bits) {
        case 64:
          return DType::kInt64;

        case 32:
          return DType::kInt32;

        default:
          NVTE_ERROR("Unsupported bits in integer DLDataType: ", dldtype.bits);
      }

    case DLDataTypeCode::kDLFloat:
      switch (dldtype.bits) {
        case 32:
          return DType::kFloat32;

        case 16:
          return DType::kFloat16;

        case 8:
          if (grad) {
            return DType::kFloat8E5M2;
          } else {
            return DType::kFloat8E4M3;
          }

        default:
          NVTE_ERROR("Unsupported bits in float DLDataType: ", dldtype.bits);
      }

    case DLDataTypeCode::kDLBfloat:
      if (dldtype.bits == 16) {
        return DType::kBFloat16;
      } else {
        NVTE_ERROR("Unsupported bits in bfloat DLDataType: ", dldtype.bits);
      }

    case DLDataTypeCode::kDLBool:
    case DLDataTypeCode::kDLUInt:
      if (dldtype.bits == 8) {
        return DType::kByte;
      } else {
        NVTE_ERROR("Unsupported bits in unsigned int DLDataType: ", dldtype.bits);
      }

    default:
      NVTE_ERROR("Unsupported DLDataType.");
  }
}

class DLPackWrapper : public TensorWrapper {
 protected:
  DLManagedTensor managed_tensor;

 public:
  // Inherit TensorWrapper constructors
  using TensorWrapper::TensorWrapper;

  // Construct a new DLPackWrapper from existing TensorWrapper
  DLPackWrapper(TensorWrapper &&other) : TensorWrapper(std::move(other)) {}

  // New constructor from PyObject
  DLPackWrapper(pybind11::object obj, bool grad = false) {
    NVTE_CHECK(PyCapsule_CheckExact(obj.ptr()), "Expected DLPack capsule");

    DLManagedTensor *dlMTensor = (DLManagedTensor *)PyCapsule_GetPointer(obj.ptr(), "dltensor");
    NVTE_CHECK(dlMTensor, "Invalid DLPack capsule.");

    DLTensor *dlTensor = &dlMTensor->dl_tensor;
    NVTE_CHECK(dlTensor->device.device_type == DLDeviceType::kDLCUDA,
               "DLPack tensor is not on a CUDA device.");
    NVTE_CHECK(dlTensor->device.device_id == cuda::current_device(),
               "DLPack tensor resides on a different device.");

    if (dlTensor->strides) {
      for (int idx = dlTensor->ndim - 1; idx >= 0; ++idx) {
        NVTE_CHECK(dlTensor->strides[idx] == 1,
                   "DLPack tensors with non-standard strides are not supported.");
      }
    }

    NVTEShape shape;
    shape.data = reinterpret_cast<size_t *>(dlTensor->shape);
    shape.ndim = static_cast<size_t>(dlTensor->ndim);
    this->tensor_ = nvte_create_tensor(
        dlTensor->data, shape, static_cast<NVTEDType>(dldtype_to_nvte_dtype(dlTensor->dtype, grad)),
        nullptr, nullptr, nullptr);
  }

  pybind11::object capsule() {
    DLDevice tensor_context;
    tensor_context.device_type = DLDeviceType::kDLCUDA;
    tensor_context.device_id = cuda::current_device();

    DLTensor dlTensor;
    dlTensor.data = dptr();
    dlTensor.device = tensor_context;
    dlTensor.ndim = ndim();
    dlTensor.dtype = nvte_dtype_to_dldtype(dtype());
    dlTensor.shape = reinterpret_cast<int64_t *>(const_cast<size_t *>(shape().data));
    dlTensor.strides = nullptr;
    dlTensor.byte_offset = 0;

    managed_tensor.dl_tensor = dlTensor;
    managed_tensor.manager_ctx = nullptr;
    managed_tensor.deleter = [](DLManagedTensor *) {};

    return pybind11::reinterpret_steal<pybind11::object>(
        PyCapsule_New(&managed_tensor, "dltensor", nullptr));
  }
};

}  // namespace transformer_engine

#endif

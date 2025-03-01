/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "extensions/ffi.h"

#include <iostream>

namespace transformer_engine {
namespace jax {

// For XLA_FFI_DataType Enum Reference: https://github.com/openxla/xla/blob/d054e8366c4e8807726961feeb28b1cdba681888/xla/ffi/api/c_api.h#L163-L186
DType convert_ffi_datatype_to_te_dtype(const xla::ffi::DataType &type) {
  switch (type) {
    case xla::ffi::DataType::U8:
      return DType::kByte;
      break;
    case xla::ffi::DataType::S32:
      return DType::kInt32;
      break;
    case xla::ffi::DataType::S64:
      return DType::kInt64;
      break;
    case xla::ffi::DataType::F32:
      return DType::kFloat32;
      break;
    case xla::ffi::DataType::F16:
      return DType::kFloat16;
      break;
    case xla::ffi::DataType::BF16:
      return DType::kBFloat16;
      break;
    case xla::ffi::DataType::F8E5M2:
      return DType::kFloat8E5M2;
      break;
    case xla::ffi::DataType::F8E4M3FN:
      return DType::kFloat8E4M3;
      break;
    default:
      auto type_num = static_cast<XLA_FFI_DataType>(type);
      NVTE_ERROR("TE does not support conversion of XLA_FFI_DataType %d",
                 static_cast<int>(type_num));
      break;
  }
}

Error_Type ffi_with_cuda_error_check() {
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    return Error_Type(XLA_FFI_Error_Code_INTERNAL,
                      std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return Error_Type::Success();
}

}  // namespace jax
}  // namespace transformer_engine

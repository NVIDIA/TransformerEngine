/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "common/util/logging.h"
#include "jax/csrc/extensions/ffi.h"

namespace transformer_engine {
namespace jax {

DType convert_ffi_datatype_to_te_dtype(xla::ffi::DataType type){
  switch(type){
    case xla::ffi::DataType::F32:
      return DType::kFloat32;
    break;
    case xla::ffi::DataType::F16:
      return DType::kFloat16;
    break;
    case xla::ffi::DataType::BF16:
      return DType::kBFloat16;
    break;
    default:
      NVTE_ERROR("Invalid FFI DataType");
  }
}
    // case xla::ffi::DataType::F8E5M2:
    //   return DType::kBFloat8E5M2;
    // break;
    // case xla::ffi::DataType::F8E4M3FN:
    //   return DType::kBFloat8E4M3;
    // break;

}  // namespace jax
}  // namespace transformer_engine

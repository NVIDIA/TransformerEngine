/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>
#include "xla/ffi/api/ffi.h"

namespace transformer_engine {
namespace jax {

using Buffer_Type = xla::ffi::AnyBuffer;
using Result_Type = xla::ffi::Result<xla::ffi::AnyBuffer>;
using Error_Type = xla::ffi::Error;
using FFI_Bind = xla::ffi::Ffi::Bind();
using FFI_Stream_Type = xla::ffi::PlatformStream<cudaStream_t>>;

DType FFI_DataType_To_TE_DType(xla::ffi::DataType type){
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
    case xla::ffi::DataType::F8E5M2:
      return DType::kBFloat8E5M2;
    break;
    case xla::ffi::DataType::F8E4M3FN:
      return DType::kBFloat8E4M3;
    break;
    default:
      NVTE_ERROR("Invalid FFI DataType");
  }
}

}  // namespace jax
}  // namespace transformer_engine

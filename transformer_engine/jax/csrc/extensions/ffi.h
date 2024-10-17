/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>
#include <xla/ffi/api/ffi.h>

#include <numeric>

namespace transformer_engine {
namespace jax {

using Buffer_Type = xla::ffi::AnyBuffer;
using Result_Type = xla::ffi::Result<xla::ffi::AnyBuffer>;
using Error_Type = xla::ffi::Error;
using FFI = xla::ffi::Ffi;
using FFI_Stream_Type = xla::ffi::PlatformStream<cudaStream_t>;
constexpr auto FFI_CudaGraph_Traits = {xla::ffi::Traits::kCmdBufferCompatible};

DType convert_ffi_datatype_to_te_dtype(const xla::ffi::DataType &type);
Error_Type ffi_with_cuda_error_check();

}  // namespace jax
}  // namespace transformer_engine

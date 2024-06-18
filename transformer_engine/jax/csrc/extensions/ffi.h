/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <xla/ffi/api/ffi.h>

#include <transformer_engine/transformer_engine.h>

namespace transformer_engine {
namespace jax {

using Buffer_Type = xla::ffi::AnyBuffer;
using Result_Type = xla::ffi::Result<xla::ffi::AnyBuffer>;
using Error_Type = xla::ffi::Error;
using FFI = xla::ffi::Ffi;
using FFI_Stream_Type = xla::ffi::PlatformStream<cudaStream_t>;

DType convert_ffi_datatype_to_te_dtype(xla::ffi::DataType type);

}  // namespace jax
}  // namespace transformer_engine

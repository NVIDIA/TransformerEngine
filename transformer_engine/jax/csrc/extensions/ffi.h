/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>
#include <xla/ffi/api/ffi.h>

#include <numeric>

#include "common/util/logging.h"

namespace transformer_engine {
namespace jax {

using Buffer_Type = xla::ffi::AnyBuffer;
using Result_Type = xla::ffi::Result<xla::ffi::AnyBuffer>;
using Variadic_Buffer_Type = xla::ffi::RemainingArgs;
using Variadic_Result_Type = xla::ffi::RemainingRets;
using Error_Type = xla::ffi::Error;
using FFI = xla::ffi::Ffi;
using FFI_Stream_Type = xla::ffi::PlatformStream<cudaStream_t>;
using Dictionary = xla::ffi::Dictionary;

constexpr auto FFI_Prepare = xla::ffi::ExecutionStage::kPrepare;
constexpr auto FFI_CudaGraph_Traits = {xla::ffi::Traits::kCmdBufferCompatible};

DType convert_ffi_datatype_to_te_dtype(const xla::ffi::DataType& type);

Error_Type ffi_with_cuda_error_check();

// source_location is not available in C++17, so we implement it ourselves
#if defined(__GNUC__) || defined(__clang__)
#define CURRENT_FILE __builtin_FILE()
#define CURRENT_LINE __builtin_LINE()
#define CURRENT_FUNCTION __builtin_FUNCTION()
#else
#define CURRENT_FILE __FILE__
#define CURRENT_LINE __LINE__
#define CURRENT_FUNCTION __func__
#endif

class source_location {
 public:
  static source_location current(const char* file = CURRENT_FILE, int line = CURRENT_LINE,
                                 const char* function = CURRENT_FUNCTION) {
    return source_location(file, line, function);
  }

  constexpr const char* file_name() const { return file_; }
  constexpr int line() const { return line_; }
  constexpr const char* function_name() const { return function_; }

 private:
  constexpr source_location(const char* file, int line, const char* function)
      : file_(file), line_(line), function_(function) {}

  const char* file_;
  int line_;
  const char* function_;
};

template <typename T>
T get_attr_value(Dictionary& attrs, std::string attr_name,
                 const source_location& loc = source_location::current()) {
  auto attr = attrs.get<T>(attr_name);
  if (attr.has_error()) {
    NVTE_ERROR("Failure in getting attribute value of '", attr_name, "'\n",
               "Called from: ", loc.file_name(), ":", loc.line(), "\n",
               "In function: ", loc.function_name(), "\n",
               "Please ensure the attribute name and datatype match between C++ and Python APIs.");
  }
  return attr.value();
}

inline size_t product(const xla::ffi::Span<const int64_t>& data, size_t start_idx = 0,
                      size_t end_idx = 0) {
  end_idx = (end_idx == 0) ? data.size() : end_idx;
  return std::accumulate(data.begin() + start_idx, data.begin() + end_idx, size_t(1),
                         std::multiplies<size_t>());
}

}  // namespace jax
}  // namespace transformer_engine

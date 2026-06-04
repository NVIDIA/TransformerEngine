/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include <cuda_runtime.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#include "../extensions.h"
#include "xla/ffi/api/c_api.h"

namespace transformer_engine {
namespace jax {

// Sanitize a probe name for use as a filename component: replace any
// character that's not [A-Za-z0-9._-] with '_'. Probe names like
// "fwd/sparse_probs_after_fused_topk" therefore become legal POSIX
// filenames ("fwd_sparse_probs_after_fused_topk") without losing the
// trailing semantic suffix.
static std::string SanitizeProbeName(std::string_view name) {
  std::string out;
  out.reserve(name.size());
  for (char c : name) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' ||
        c == '_' || c == '-') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  if (out.empty()) {
    out = "anon";
  }
  return out;
}

Error_Type InspectFFI(cudaStream_t stream, Buffer_Type input_buf, Buffer_Type min_buf,
                      Buffer_Type max_buf, Buffer_Type mean_buf, Buffer_Type std_buf,
                      Result_Type output_buf, std::string_view name) {
  NVTE_CHECK(input_buf.untyped_data() != nullptr, "Input must be provided for inspect operation");
  NVTE_CHECK(output_buf->untyped_data() != nullptr,
             "Output must be provided for inspect operation");
  NVTE_CHECK(input_buf.untyped_data() == output_buf->untyped_data(),
             "Input and output must point to the same buffer for inspect operation");

  std::vector<uint8_t> input_data(input_buf.size_bytes());
  NVTE_CHECK_CUDA(cudaMemcpyAsync(input_data.data(), input_buf.untyped_data(),
                                  input_buf.size_bytes(), cudaMemcpyDeviceToHost, stream));

  float min_val{}, max_val{}, mean_val{}, std_val{};
  NVTE_CHECK_CUDA(cudaMemcpyAsync(&min_val, min_buf.untyped_data(), sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));
  NVTE_CHECK_CUDA(cudaMemcpyAsync(&max_val, max_buf.untyped_data(), sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));
  NVTE_CHECK_CUDA(cudaMemcpyAsync(&mean_val, mean_buf.untyped_data(), sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));
  NVTE_CHECK_CUDA(cudaMemcpyAsync(&std_val, std_buf.untyped_data(), sizeof(float),
                                  cudaMemcpyDeviceToHost, stream));

  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));

  int device;
  NVTE_CHECK_CUDA(cudaGetDevice(&device));

  // Per-probe filenames: my_tensor_gpu{device}_{sanitized_name}.bin /
  // ..._meta.json. With distinct names, the on-disk dumps survive across
  // probes instead of being overwritten on every call, so a single test
  // run produces one .bin per probe per rank ready for offline analysis.
  std::string safe_name = SanitizeProbeName(name);
  std::string device_str = std::to_string(device);
  std::string filename = "my_tensor_gpu" + device_str + "_" + safe_name + ".bin";
  std::ofstream file(filename, std::ios::binary);
  NVTE_CHECK(file.is_open(), "Failed to create file: ", filename);
  file.write(reinterpret_cast<const char *>(input_data.data()), input_data.size());
  file.close();

  std::string meta_filename = "my_tensor_gpu" + device_str + "_" + safe_name + "_meta.json";
  std::ofstream meta_file(meta_filename);
  NVTE_CHECK(meta_file.is_open(), "Failed to create file: ", meta_filename);
  meta_file << "{";
  // Echo the original (un-sanitized) probe name so analysis tools can
  // recover the semantic label even when the filename had to mangle it.
  meta_file << "\"name\": \"" << name << "\", ";
  meta_file << "\"shape\": [";
  for (size_t i = 0; i < input_buf.dimensions().size(); ++i) {
    meta_file << input_buf.dimensions()[i];
    if (i < input_buf.dimensions().size() - 1) {
      meta_file << ", ";
    }
  }
  meta_file << "], ";
  meta_file << "\"dtype\": " << static_cast<int>(input_buf.element_type());
  meta_file << ", \"min\": " << min_val;
  meta_file << ", \"max\": " << max_val;
  meta_file << ", \"mean\": " << mean_val;
  meta_file << ", \"std\": " << std_val;
  meta_file << "}";
  meta_file.close();

  // Surface the probe name in the live log alongside the file path, so
  // analysing a multi-probe trace doesn't require correlating by
  // shape/dtype guesswork.
  printf("[gpu%d %.*s]: written to %s (shape: [", device, static_cast<int>(name.size()),
         name.data(), filename.c_str());
  for (size_t i = 0; i < input_buf.dimensions().size(); ++i) {
    printf("%zu", static_cast<size_t>(input_buf.dimensions()[i]));
    if (i < input_buf.dimensions().size() - 1) {
      printf(", ");
    }
  }
  printf("], dtype: %d", static_cast<int>(input_buf.element_type()));
  printf(", min: %f, max: %f, mean: %f, std: %f)\n", min_val, max_val, mean_val, std_val);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(InspectHandler, InspectFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()          // stream
                                  .Arg<Buffer_Type>()              // input
                                  .Arg<Buffer_Type>()              // min
                                  .Arg<Buffer_Type>()              // max
                                  .Arg<Buffer_Type>()              // mean
                                  .Arg<Buffer_Type>()              // std
                                  .Ret<Buffer_Type>()              // output
                                  .Attr<std::string_view>("name")  // probe name
);

}  // namespace jax
}  // namespace transformer_engine

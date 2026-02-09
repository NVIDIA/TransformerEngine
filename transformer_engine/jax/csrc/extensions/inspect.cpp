/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>

#include "../extensions.h"
#include "xla/ffi/api/c_api.h"

namespace transformer_engine {
namespace jax {

Error_Type InspectFFI(cudaStream_t stream, Buffer_Type input_buf, Buffer_Type min_buf,
                      Buffer_Type max_buf, Buffer_Type mean_buf, Buffer_Type std_buf,
                      Result_Type output_buf) {
  NVTE_CHECK(input_buf.untyped_data() != nullptr, "Input must be provided for inspect operation");
  NVTE_CHECK(output_buf->untyped_data() != nullptr,
             "Output must be provided for inspect operation");
  NVTE_CHECK(input_buf.untyped_data() == output_buf->untyped_data(),
             "Input and output must point to the same buffer for inspect operation");

  std::vector<uint8_t> input_data(input_buf.size_bytes());
  cudaMemcpyAsync(input_data.data(), input_buf.untyped_data(), input_buf.size_bytes(),
                  cudaMemcpyDeviceToHost, stream);

  float min_val{}, max_val{}, mean_val{}, std_val{};
  cudaMemcpyAsync(&min_val, min_buf.untyped_data(), sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&max_val, max_buf.untyped_data(), sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&mean_val, mean_buf.untyped_data(), sizeof(float), cudaMemcpyDeviceToHost,
                  stream);
  cudaMemcpyAsync(&std_val, std_buf.untyped_data(), sizeof(float), cudaMemcpyDeviceToHost, stream);

  cudaStreamSynchronize(stream);

  int device;
  cudaGetDevice(&device);

  // Write the tensor data to a file as a binary blob
  std::string filename = "my_tensor_gpu" + std::to_string(device) + ".bin";
  std::ofstream file(filename, std::ios::binary);
  if (file.is_open()) {
    file.write(reinterpret_cast<const char *>(input_data.data()), input_data.size());
    file.close();
  }

  // Write out a metadata file
  std::string meta_filename = "my_tensor_gpu" + std::to_string(device) + "_meta.json";
  std::ofstream meta_file(meta_filename);
  if (meta_file.is_open()) {
    meta_file << "{";
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
  }

  // Log the tensor metadata to the console
  printf("Tensor data written to %s (shape: [", filename.c_str());
  for (size_t i = 0; i < input_buf.dimensions().size(); ++i) {
    printf("%ld", static_cast<long>(input_buf.dimensions()[i]));
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
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // input
                                  .Arg<Buffer_Type>()      // min
                                  .Arg<Buffer_Type>()      // max
                                  .Arg<Buffer_Type>()      // mean
                                  .Arg<Buffer_Type>()      // std
                                  .Ret<Buffer_Type>()      // output
);

}  // namespace jax
}  // namespace transformer_engine

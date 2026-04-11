/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include <cuda_runtime.h>

#include <vector>

#include "../extensions.h"

namespace transformer_engine {
namespace jax {

Error_Type ValidateGroupSizesFFI(cudaStream_t stream, Buffer_Type group_sizes_buf,
                                 Result_Type output_buf, ValidateGroupSizesConfig config) {
  NVTE_CHECK(group_sizes_buf.untyped_data() != nullptr,
             "group_sizes input must be provided for validate_group_sizes operation");
  NVTE_CHECK(output_buf->untyped_data() != nullptr,
             "output must be provided for validate_group_sizes operation");
  NVTE_CHECK(group_sizes_buf.untyped_data() == output_buf->untyped_data(),
             "Input and output must point to the same buffer for validate_group_sizes operation");

  const int64_t align_size = config.align_size;
  const int64_t num_experts = static_cast<int64_t>(product(group_sizes_buf.dimensions()));

  std::vector<int32_t> group_sizes_host(num_experts);
  NVTE_CHECK_CUDA(cudaMemcpyAsync(group_sizes_host.data(), group_sizes_buf.untyped_data(),
                                  num_experts * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));

  for (int64_t i = 0; i < num_experts; ++i) {
    NVTE_CHECK(group_sizes_host[i] % align_size == 0,
               "group_sizes alignment check failed: group_sizes[", i, "] = ", group_sizes_host[i],
               " is not divisible by align_size = ", align_size);
  }

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(ValidateGroupSizesHandler, ValidateGroupSizesFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // group_sizes
                                  .Ret<Buffer_Type>()      // output (aliased to input)
                                  .Attr<ValidateGroupSizesConfig>("config"));
// Note: no FFI_CudaGraph_Traits — not CUDA-graph compatible

}  // namespace jax
}  // namespace transformer_engine

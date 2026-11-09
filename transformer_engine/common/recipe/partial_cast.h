/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_RECIPE_PARTIAL_CAST_H_
#define TRANSFORMER_ENGINE_RECIPE_PARTIAL_CAST_H_

#include "common/common.h"

namespace transformer_engine {
namespace detail {

// The legacy C APIs do not accept caller-provided workspace. Their adapters
// exceptionally allocate stream-ordered scratch space to normalize old scale
// layouts and datatypes. The unified API uses only caller-owned device memory.
class PartialCastWorkspace {
 public:
  PartialCastWorkspace(size_t bytes, cudaStream_t stream) : stream_(stream) {
    if (bytes != 0) NVTE_CHECK_CUDA(cudaMallocAsync(&data_, bytes, stream_));
  }
  PartialCastWorkspace(const PartialCastWorkspace &) = delete;
  PartialCastWorkspace &operator=(const PartialCastWorkspace &) = delete;
  ~PartialCastWorkspace() {
    if (data_ != nullptr) cudaFreeAsync(data_, stream_);
  }
  void *data() const { return data_; }

 private:
  void *data_ = nullptr;
  cudaStream_t stream_;
};

}  // namespace detail

namespace fp8_block_scaling_recipe {
void partial_cast(const Tensor &inp, Tensor &out, size_t h, size_t w, size_t start_offset,
                  cudaStream_t stream);
}  // namespace fp8_block_scaling_recipe
namespace mxfp8_scaling_recipe {
void partial_cast(const Tensor &inp, Tensor &out, size_t h, size_t w, size_t start_offset,
                  cudaStream_t stream);
}  // namespace mxfp8_scaling_recipe
namespace nvfp4_recipe {
void partial_cast(const Tensor &inp, Tensor &out, size_t h, size_t w, size_t start_offset,
                  cudaStream_t stream);
}  // namespace nvfp4_recipe
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_RECIPE_PARTIAL_CAST_H_

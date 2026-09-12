/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../extensions.h"
#include "common.h"

namespace transformer_engine::pytorch {

at::Tensor get_cudnn_attention_rng_state(const std::optional<at::Generator> rng_gen,
                                         size_t increment) {
  auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
      rng_gen, at::cuda::detail::getDefaultCUDAGenerator());
  at::PhiloxCudaState philox_args = init_philox_state(gen, increment);
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  auto rng_state = torch::empty({2}, options);
  // PhiloxCudaState contains device pointers while CUDA graph capture is
  // active. Unpacking on the current stream therefore preserves PyTorch's
  // graph-safe intragraph offset semantics.
  philox_unpack(philox_args, static_cast<int64_t *>(rng_state.data_ptr()));
  return rng_state;
}

}  // namespace transformer_engine::pytorch

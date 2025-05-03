/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#pragma once

#include <assert.h>
#include <cuda_runtime.h>
#include <transformer_engine/multi_tensor.h>
#include <transformer_engine/transformer_engine.h>

#include "../common.h"

// This header is the one-stop shop for all your multi-tensor apply needs.

// Change device if needed.
class OptionalCUDAGuard {
 public:
  explicit OptionalCUDAGuard(int new_device) {
    if (new_device < 0) return;

    int current_device;
    NVTE_CHECK_CUDA(cudaGetDevice(&current_device));

    if (new_device != current_device) {
      NVTE_CHECK_CUDA(cudaSetDevice(new_device));
      device_changed_ = true;
      prev_device_ = current_device;
    }
  }

  OptionalCUDAGuard(const OptionalCUDAGuard &) = delete;
  OptionalCUDAGuard &operator=(const OptionalCUDAGuard &) = delete;

  OptionalCUDAGuard(OptionalCUDAGuard &&other) noexcept
      : prev_device_(other.prev_device_), device_changed_(other.device_changed_) {
    other.device_changed_ = false;
  }

  OptionalCUDAGuard &operator=(OptionalCUDAGuard &&other) noexcept {
    if (this != &other) {
      if (device_changed_) {
        cudaSetDevice(prev_device_);
      }
      prev_device_ = other.prev_device_;
      device_changed_ = other.device_changed_;
      other.device_changed_ = false;
    }
    return *this;
  }

  ~OptionalCUDAGuard() {
    if (device_changed_) {
      NVTE_CHECK_CUDA(cudaSetDevice(prev_device_));
    }
  }

 private:
  int prev_device_;
  bool device_changed_ = false;
};

// TODO:  Kernel arg size limit may be <4KB for some other cards (ie Jetson)
constexpr int depth_to_max_tensors[6] = {110, 64, 48, 36, 30, 24};
constexpr int depth_to_max_blocks[6] = {320, 320, 320, 320, 320, 320};

template <int n, bool USE_FP8 = false>
struct TensorListMetadataBase {
  void *addresses[n][depth_to_max_tensors[n - 1]];
  int sizes[depth_to_max_tensors[n - 1]];
  unsigned char block_to_tensor[depth_to_max_blocks[n - 1]];
  int block_to_chunk[depth_to_max_blocks[n - 1]];
  int start_tensor_this_launch;
};

template <int n, bool USE_FP8 = false>
struct TensorListMetadata : public TensorListMetadataBase<n, USE_FP8> {};

template <int n>
struct TensorListMetadata<n, true> : public TensorListMetadataBase<n, true> {
  void *fp8_meta_addresses[3][depth_to_max_tensors[n - 1]];
};

template <typename T, typename U, typename... ArgTypes>
__global__ void multi_tensor_apply_kernel(int64_t chunk_size, volatile int *noop_flag, T tl,
                                          U callable, ArgTypes... args) {
  // Hand the chunk information to the user-supplied functor to process however
  // it likes.
  callable(chunk_size, noop_flag, tl, args...);
}

template <int depth, bool USE_FP8 = false, typename T, typename... ArgTypes>
void multi_tensor_apply(int64_t block_size, int64_t chunk_size,
                        const transformer_engine::Tensor &noop_flag,
                        std::vector<std::vector<transformer_engine::Tensor *>> tensor_lists,
                        T callable, const int device_id, cudaStream_t stream, ArgTypes... args) {
  const size_t num_tensor_lists = tensor_lists.size();
  const size_t num_tensors_per_list = tensor_lists[0].size();

  if constexpr (USE_FP8) {
    NVTE_CHECK(num_tensor_lists == depth + 3,
               "tensor_lists.size() != depth + 3, tensor_lists should have 3 more tensors (scale, "
               "amax, scale_inv) for fp8");
  } else {
    NVTE_CHECK(num_tensor_lists == depth, "tensor_lists.size() != depth");
  }

  TensorListMetadata<depth, USE_FP8> tl;

  const OptionalCUDAGuard device_guard(device_id);

  tl.start_tensor_this_launch = 0;
  int loc_block_info = 0;
  int loc_tensor_info = 0;
  for (int t = 0; t < num_tensors_per_list; t++) {
    tl.sizes[loc_tensor_info] = tensor_lists[0][t]->numel();
    for (int d = 0; d < depth; d++)
      tl.addresses[d][loc_tensor_info] = tensor_lists[d][t]->data.dptr;
    if constexpr (USE_FP8) {
      for (int i = 0; i < 3; i++)
        tl.fp8_meta_addresses[i][loc_tensor_info] = tensor_lists[depth + i][t]->data.dptr;
    }
    loc_tensor_info++;

    auto chunks_this_tensor = (tensor_lists[0][t]->numel() + chunk_size - 1) / chunk_size;

    for (auto chunk = 0; chunk < chunks_this_tensor; chunk++) {
      tl.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
      tl.block_to_chunk[loc_block_info] = chunk;
      loc_block_info++;

      bool tensors_full =
          (loc_tensor_info == depth_to_max_tensors[depth - 1] && chunk == chunks_this_tensor - 1);
      bool blocks_full = (loc_block_info == depth_to_max_blocks[depth - 1]);
      bool last_chunk = (t == num_tensors_per_list - 1 && chunk == chunks_this_tensor - 1);
      if (tensors_full || blocks_full || last_chunk) {
        multi_tensor_apply_kernel<<<loc_block_info, block_size, 0, stream>>>(
            chunk_size, reinterpret_cast<int *>(noop_flag.data.dptr), tl, callable, args...);

        NVTE_CHECK_CUDA(cudaGetLastError());

        // Reset.  The control flow possibilities here make my brain hurt.
        loc_block_info = 0;
        if (chunk == chunks_this_tensor - 1) {
          loc_tensor_info = 0;
          tl.start_tensor_this_launch = t + 1;
        } else {
          tl.sizes[0] = tl.sizes[loc_tensor_info - 1];
          for (int d = 0; d < depth; d++) {
            tl.addresses[d][0] = tl.addresses[d][loc_tensor_info - 1];
          }
          if constexpr (USE_FP8) {
            for (int i = 0; i < 3; i++) {
              tl.fp8_meta_addresses[i][0] = tl.fp8_meta_addresses[i][loc_tensor_info - 1];
            }
          }
          loc_tensor_info = 1;
          tl.start_tensor_this_launch = t;
        }
      }
    }
  }
}

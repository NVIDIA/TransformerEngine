/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ROUTER_UTILS_H_
#define TRANSFORMER_ENGINE_FUSED_ROUTER_UTILS_H_

#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {

constexpr size_t kThreadsPerWarp = 32;
constexpr int kThreadsPerBlock =
    128;  // Using 4 warps in 1 CTA, Each warp is responsible for 1 token.
constexpr float epsilon = 1e-20;

template <typename T>
__device__ inline T max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
__device__ inline T sum(T a, T b) {
  return a + b;
}

template <typename T>
__device__ inline T warp_reduce_on_shmem(T *data_ptr, int data_size, T (*reduce_func)(T, T),
                                         int lane_id) {
  // Some value is hanlded in local thread
  // Thread 0 is responsible for the: 0-th, 32-th, 64-th, 96-th ...
  // Reduce the value in local thread
  volatile double val =
      lane_id < data_size ? static_cast<double>(data_ptr[lane_id]) : static_cast<double>(0);
  for (int i = lane_id + kThreadsPerWarp; i < data_size; i += kThreadsPerWarp) {
    val = reduce_func(val, data_ptr[i]);
  }

  // Warp shuffle between threads
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 16));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 8));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 4));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 2));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 1));
  __syncwarp();
  return T(val);
}

template <typename DataType>
__device__ inline void apply_sigmoid_on_float(DataType *scores, int data_size, int lane_id) {
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    scores[i] = static_cast<float>(1.0f / (1.0f + exp(-static_cast<float>(scores[i]))));
  }
}

template <typename T>
__device__ inline T masked_warp_reduce_on_shmem(T *data_ptr, bool *mask, int data_size,
                                                T (*reduce_func)(T, T), int lane_id) {
  // Some value is hanlded in local thread
  // Thread 0 is responsible for the: 0-th, 32-th, 64-th, 96-th ...
  // Reduce the value in local thread
  volatile double val = lane_id < data_size && mask[lane_id]
                            ? static_cast<double>(data_ptr[lane_id])
                            : static_cast<double>(0);
  for (int i = lane_id + kThreadsPerWarp; i < data_size; i += kThreadsPerWarp) {
    if (mask[i]) {
      val = reduce_func(val, data_ptr[i]);
    }
  }

  // Warp shuffle between threads
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 16));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 8));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 4));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 2));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 1));
  __syncwarp();
  return T(val);
}

template <typename DataType>
__device__ inline void apply_sigmoid_bwd_on_float(DataType *grad, DataType *fwd_output,
                                                  int data_size, int lane_id) {
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    grad[i] = static_cast<double>(grad[i]) * static_cast<double>(fwd_output[i]) *
              (1 - static_cast<double>(fwd_output[i]));
  }
}

template <typename DataType>
__device__ inline void apply_softmax_bwd_on_float(DataType *grad, DataType *fwd_output,
                                                  DataType *comp_buf, bool *mask, int data_size,
                                                  int lane_id) {
  // Put the result of output * grad to the comp_buf
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    if (mask) {
      if (mask[i])
        comp_buf[i] = static_cast<float>(grad[i]) * static_cast<float>(fwd_output[i]);
      else
        comp_buf[i] = 0.0f;
    } else {
      comp_buf[i] = static_cast<float>(grad[i]) * static_cast<float>(fwd_output[i]);
    }
  }
  __syncwarp();
  float sum_Output_x_Grad = warp_reduce_on_shmem(
      /*data ptr = */ comp_buf,
      /*data size = */ data_size,
      /*reduce func = */ sum, lane_id);
  // In-place update
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    if (mask) {
      if (mask[i])
        grad[i] =
            static_cast<float>(fwd_output[i]) * (static_cast<float>(grad[i]) - sum_Output_x_Grad);
      else
        grad[i] = 0.0f;
    } else {
      grad[i] =
          static_cast<float>(fwd_output[i]) * (static_cast<float>(grad[i]) - sum_Output_x_Grad);
    }
  }
}

template <typename DataType>
__device__ inline void apply_softmax_on_float(DataType *scores, int data_size, int lane_id) {
  // 1. compute the max of value
  float max_val = static_cast<float>(warp_reduce_on_shmem(scores, data_size, max, lane_id));
  // 2. value -> exp_value
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    scores[i] = static_cast<float>(exp(static_cast<float>(scores[i]) - max_val));
  }
  __syncwarp();
  // 3. compute the sum of exp_value
  float sum_val = static_cast<float>(warp_reduce_on_shmem(scores, data_size, sum, lane_id));
  // 4. update the softmax value
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    scores[i] = static_cast<float>(scores[i]) / sum_val;
  }
  __syncwarp();
}

template <typename T>
__device__ inline void naive_topk_and_mask(T *scores, int data_size, int topk, int *topk_indices,
                                           T *topk_scores, int lane_id) {
  // Topk Times: Find the max value and its index
  // Then mask it, and record the index in the topk_indices
  // After looping topk times, the topk_indices will be the topk indices
  for (int k = 0; k < topk; k++) {
    // Find the max value and its index
    volatile double val =
        (lane_id < data_size) ? static_cast<double>(scores[lane_id]) : static_cast<double>(0);
    volatile int index = (lane_id < data_size) ? lane_id : 0;
    // Some value is hanlded in local thread
    // Thread 0 is responsible for the: 0-th, 32-th, 64-th, 96-th ...
    // Reduce the value in local thread
    for (int i = lane_id + kThreadsPerWarp; i < data_size; i += kThreadsPerWarp) {
      volatile double cur_val = scores[i];
      if (cur_val > val) {
        val = cur_val;
        index = i;
      }
    }
    // Warp shuffle between threads
    for (int s = 16; s > 0; s /= 2) {
      volatile auto shuffled_val = __shfl_xor_sync(0xffffffff, val, s);
      volatile auto shuffled_index = __shfl_xor_sync(0xffffffff, index, s);
      if (shuffled_val > val) {
        val = shuffled_val;
        index = shuffled_index;
      }
    }
    if (lane_id == 0) {
      topk_indices[k] = index;
      topk_scores[k] = val;
      scores[index] =
          static_cast<double>(-1.0) - val;  // make the selected experts using val = - 1 - val
    }
    __syncwarp();
  }

  // Reset the scores to the original value
  for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
    scores[topk_indices[i]] =
        static_cast<double>(-1.0) - static_cast<double>(scores[topk_indices[i]]);
  }
}

// Current TE only support float32/bf16/fp16, float64 probs should be considered in the future
#define TE_ROUTER_PROBS_TYPE_SWITCH_ALL(dtype, type, ...) \
  switch (dtype) {                                        \
    using namespace transformer_engine;                   \
    case DType::kFloat32: {                               \
      using type = float;                                 \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    case DType::kFloat16: {                               \
      using type = fp16;                                  \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    case DType::kBFloat16: {                              \
      using type = bf16;                                  \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    default:                                              \
      NVTE_ERROR("Invalid type.");                        \
  }

#define TE_ROUTER_INDEX_TYPE_SWITCH_ALL(dtype, type, ...) \
  switch (dtype) {                                        \
    using namespace transformer_engine;                   \
    case DType::kInt32: {                                 \
      using type = int32_t;                               \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    case DType::kInt64: {                                 \
      using type = int64_t;                               \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    default:                                              \
      NVTE_ERROR("Invalid type.");                        \
  }
}  // namespace transformer_engine
#endif

/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ROUTER_UTILS_H_
#define TRANSFORMER_ENGINE_FUSED_ROUTER_UTILS_H_

#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
namespace fused_router {

// Using FP32 to handle all the calculations.
// Currently, only FP32 is supported because
//   1. The score functions (sigmoid, softmax, sqrtsoftplus) are implemented in FP32.
//   2. The intermediate buffer is initialized in FP32.
using CompType = float;

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

enum ReduceFuncType {
  SUM,
  MAX,
};

template <typename T>
__device__ inline T warp_reduce_on_shmem(T *data_ptr, int data_size, ReduceFuncType type,
                                         int lane_id) {
  T (*reduce_func)(T, T);
  CompType default_val = 0.0;
  if (type == ReduceFuncType::SUM) {
    reduce_func = sum;
    default_val = 0.0;
  } else if (type == ReduceFuncType::MAX) {
    reduce_func = max;
    default_val = -std::numeric_limits<CompType>::infinity();
  }

  // Some value is hanlded in local thread
  // Thread 0 is responsible for the: 0-th, 32-th, 64-th, 96-th ...
  // Reduce the value in local thread
  CompType val = lane_id < data_size ? data_ptr[lane_id] : default_val;
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

template <typename T>
__device__ inline T masked_warp_reduce_on_shmem(T *data_ptr, bool *mask, int data_size,
                                                ReduceFuncType type, int lane_id) {
  T (*reduce_func)(T, T);
  CompType default_val = 0.0;
  if (type == ReduceFuncType::SUM) {
    reduce_func = sum;
    default_val = 0.0;
  } else if (type == ReduceFuncType::MAX) {
    reduce_func = max;
    default_val = -std::numeric_limits<CompType>::infinity();
  }

  // Some value is hanlded in local thread
  // Thread 0 is responsible for the: 0-th, 32-th, 64-th, 96-th ...
  // Reduce the value in local thread
  CompType val = lane_id < data_size && mask[lane_id] ? data_ptr[lane_id] : default_val;
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

__device__ inline void apply_sigmoid_on_float(float *scores, int data_size, int lane_id) {
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    scores[i] = 1.0f / (1.0f + expf(-scores[i]));
  }
}

__device__ inline void apply_sigmoid_bwd_on_float(float *grad, float *fwd_output, int data_size,
                                                  int lane_id) {
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    grad[i] = grad[i] * fwd_output[i] * (1.0f - fwd_output[i]);
  }
}

// sqrtsoftplus: y = sqrt(softplus(x)) = sqrt(log(1 + exp(x)))
__device__ inline void apply_sqrtsoftplus_on_float(float *scores, int data_size, int lane_id) {
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    float x = scores[i];
    // softplus(x) = log(1 + exp(x)), numerically stable version
    // Matches PyTorch's Softplus(beta=1.0, threshold=20.0)
    float softplus_val;
    if (x > 20.0f) {
      softplus_val = x;  // for large x, softplus(x) ≈ x
    } else {
      softplus_val = log1pf(expf(x));
    }
    scores[i] = sqrtf(softplus_val);
  }
}

// sqrtsoftplus backward:
// y = sqrt(softplus(x))
// Matches PyTorch's Softplus(beta=1.0, threshold=20.0)
// We need the original logits (x) to compute the gradient
__device__ inline void apply_sqrtsoftplus_bwd_on_float(float *grad, float *fwd_output,
                                                       float *logits_buf, int data_size,
                                                       int lane_id) {
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    float x = logits_buf[i];  // original logit
    float y = fwd_output[i];  // sqrtsoftplus output
    float dy_dx;
    if (x > 20.0f) {
      // When softplus(x) = x, y = sqrt(x), dy/dx = 1/(2*y)
      dy_dx = 1.0f / (2.0f * y + epsilon);
    } else {
      // When softplus(x) = log(1+exp(x)), dy/dx = sigmoid(x) / (2*y)
      // where sigmoid(x) = 1 / (1 + exp(-x))
      float sigmoid_x = 1.0f / (1.0f + expf(-x));
      dy_dx = sigmoid_x / (2.0f * y + epsilon);
    }
    grad[i] = grad[i] * dy_dx;
  }
}

__device__ inline void apply_softmax_bwd_on_float(float *grad, float *fwd_output, float *comp_buf,
                                                  bool *mask, int data_size, int lane_id) {
  // Put the result of output * grad to the comp_buf
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    if (mask) {
      if (mask[i])
        comp_buf[i] = grad[i] * fwd_output[i];
      else
        comp_buf[i] = 0.0f;
    } else {
      comp_buf[i] = grad[i] * fwd_output[i];
    }
  }
  __syncwarp();
  float sum_Output_x_Grad = warp_reduce_on_shmem(
      /*data ptr = */ comp_buf,
      /*data size = */ data_size,
      /*reduce func = */ ReduceFuncType::SUM, lane_id);
  // In-place update
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    if (mask) {
      if (mask[i])
        grad[i] = fwd_output[i] * (grad[i] - sum_Output_x_Grad);
      else
        grad[i] = 0.0f;
    } else {
      grad[i] = fwd_output[i] * (grad[i] - sum_Output_x_Grad);
    }
  }
}

__device__ inline void apply_softmax_on_float(float *scores, int data_size, int lane_id) {
  // 1. compute the max of value
  float max_val = warp_reduce_on_shmem(scores, data_size, ReduceFuncType::MAX, lane_id);
  // 2. value -> exp_value
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    scores[i] = expf(scores[i] - max_val);
  }
  __syncwarp();
  // 3. compute the sum of exp_value
  float sum_val = warp_reduce_on_shmem(scores, data_size, ReduceFuncType::SUM, lane_id);
  // 4. update the softmax value
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    scores[i] = scores[i] / sum_val;
  }
  __syncwarp();
}

template <typename T>
__device__ inline void naive_topk_and_mask(T *scores, int data_size, int topk, int *topk_indices,
                                          T *topk_scores, int lane_id) {
  // Bit i indicates whether the i-th local element (lane_id + i * warp_size) was selected.
  uint32_t local_mask = 0;

  for (int k = 0; k < topk; k++) {
    CompType local_max_val = -std::numeric_limits<CompType>::infinity();
    int local_max_idx = -1;

    // 1) Per-lane local max on unmasked elements.
    int bit_idx = 0;
    for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
      CompType cur_val = 0.0f;
      if constexpr (std::is_same_v<CompType, double>) {
        uint64_t mask = -(uint64_t)((local_mask >> bit_idx) & 1u);
        uint64_t x_bits = __double_as_longlong(static_cast<CompType>(scores[i]));
        uint64_t result_bits =
          (~mask & x_bits) | (mask & 0xFFF0000000000000ULL);
        cur_val = __longlong_as_double(result_bits);  
      } else {
        uint32_t full_mask = -(uint32_t)((local_mask >> bit_idx) & 1u);
        uint32_t x_bits = __float_as_uint(static_cast<CompType>(scores[i]));
        uint32_t result_bits =
            (~full_mask & x_bits) | (full_mask & 0xFF800000u);
        cur_val = __uint_as_float(result_bits);
      }
      if (cur_val > local_max_val) {
        local_max_val = cur_val;
        local_max_idx = i;
      }
      bit_idx++;
    }

    // 2) Warp reduction to find global max and index.
    CompType global_max_val = local_max_val;
    int global_max_idx = local_max_idx;
    for (int s = kThreadsPerWarp / 2; s > 0; s /= 2) {
      CompType shuffled_val = __shfl_down_sync(0xffffffff, global_max_val, s);
      int shuffled_idx = __shfl_down_sync(0xffffffff, global_max_idx, s);
      if (shuffled_val > global_max_val) {
        global_max_val = shuffled_val;
        global_max_idx = shuffled_idx;
      }
    }
    global_max_idx = __shfl_sync(0xffffffff, global_max_idx, 0);
    global_max_val = __shfl_sync(0xffffffff, global_max_val, 0);

    // 3) Write top-k result.
    if (lane_id == 0) {
      topk_indices[k] = global_max_idx;
      topk_scores[k] = static_cast<T>(global_max_val);
    }

    // 4) Mark selected element in owning lane's local mask.
    if (global_max_idx >= 0 && (global_max_idx % kThreadsPerWarp) == lane_id) {
      int local_bit_pos = global_max_idx / kThreadsPerWarp;
      if (local_bit_pos < 32) {
        local_mask |= (1u << local_bit_pos);
      }
    }
  }
  __syncwarp();
}

// Current TE only support float32/bf16/fp16, float64 probs should be considered in the future
#define TE_ROUTER_PROBS_TYPE_SWITCH_ALL(dtype, type, ...)                                 \
  switch (dtype) {                                                                        \
    using namespace transformer_engine;                                                   \
    case DType::kFloat32: {                                                               \
      using type = float;                                                                 \
      { __VA_ARGS__ }                                                                     \
    } break;                                                                              \
    case DType::kFloat16: {                                                               \
      using type = fp16;                                                                  \
      { __VA_ARGS__ }                                                                     \
    } break;                                                                              \
    case DType::kBFloat16: {                                                              \
      using type = bf16;                                                                  \
      { __VA_ARGS__ }                                                                     \
    } break;                                                                              \
    default:                                                                              \
      NVTE_ERROR("Unsupported router probs dtype ", to_string(static_cast<DType>(dtype)), \
                 ". Expected one of: Float32, Float16, BFloat16.");                       \
  }

#define TE_ROUTER_INDEX_TYPE_SWITCH_ALL(dtype, type, ...)                                 \
  switch (dtype) {                                                                        \
    using namespace transformer_engine;                                                   \
    case DType::kInt32: {                                                                 \
      using type = int32_t;                                                               \
      { __VA_ARGS__ }                                                                     \
    } break;                                                                              \
    case DType::kInt64: {                                                                 \
      using type = int64_t;                                                               \
      { __VA_ARGS__ }                                                                     \
    } break;                                                                              \
    case DType::kBFloat16: {                                                              \
      using type = bf16;                                                                  \
      { __VA_ARGS__ }                                                                     \
    } break;                                                                              \
    case DType::kFloat32: {                                                               \
      using type = float;                                                                 \
      { __VA_ARGS__ }                                                                     \
    } break;                                                                              \
    default:                                                                              \
      NVTE_ERROR("Unsupported router index dtype ", to_string(static_cast<DType>(dtype)), \
                 ". Expected one of: Int32, Int64, BFloat16, "                            \
                 "Float32.");                                                             \
  }
}  // namespace fused_router
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_FUSED_ROUTER_UTILS_H_

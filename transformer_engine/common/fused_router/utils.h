/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ROUTER_UTILS_H_
#define TRANSFORMER_ENGINE_FUSED_ROUTER_UTILS_H_

#include "../util/logging.h"
#include "../utils.cuh"
#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {
namespace fused_router {

// Check if requested shared memory size exceeds device capacity.
// Throws an error with num_experts info to help users diagnose the issue.
inline void check_shared_memory_capacity_num_experts(size_t shared_memory_size, int num_experts) {
  int device_id;
  NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
  int max_smem_per_block;
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&max_smem_per_block,
                                         cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));
  NVTE_CHECK(shared_memory_size <= static_cast<size_t>(max_smem_per_block), "Shared memory size (",
             shared_memory_size, " bytes) exceeds device capacity (", max_smem_per_block,
             " bytes). Try reducing num_experts (currently ", num_experts, ").");
}

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

  // Some value is handled in local thread
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

  // Some value is handled in local thread
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
  // --- Pass 1: Online accumulation of max and sum_exp ---
  float local_max = -std::numeric_limits<float>::infinity();
  float local_sum = 0.0f;

  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    float val = scores[i];
    if (val > local_max) {
      // Rescale accumulated sum for the new max
      local_sum *= expf(local_max - val);
      local_max = val;
    }
    local_sum += expf(val - local_max);
  }

  // Warp-level reduction of (max, sum_exp) across 32 lanes.
  // When merging two lanes with (max_a, sum_a) and (max_b, sum_b):
  //   merged_max = max(max_a, max_b)
  //   merged_sum = sum_a * exp(max_a - merged_max) + sum_b * exp(max_b - merged_max)
  //
  // NaN guard: when data_size < 32, some lanes have (max=-inf, sum=0).
  // Merging two such lanes computes expf(-inf - (-inf)) = expf(NaN) = NaN,
  // and 0.0 * NaN = NaN in IEEE 754, contaminating valid lanes.
  // Fix: treat -inf max as "no data" and skip the expf computation.
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other_max = warp_shuffle_xor(local_max, offset);
    float other_sum = warp_shuffle_xor(local_sum, offset);
    float new_max = fmaxf(local_max, other_max);
    if (new_max > -std::numeric_limits<float>::infinity()) {
      // At least one side has real data; safe to compute expf differences
      float my_scale =
          (local_max > -std::numeric_limits<float>::infinity()) ? expf(local_max - new_max) : 0.0f;
      float other_scale =
          (other_max > -std::numeric_limits<float>::infinity()) ? expf(other_max - new_max) : 0.0f;
      local_sum = local_sum * my_scale + other_sum * other_scale;
    }
    // else: both sides are -inf (no data), keep local_sum = 0
    local_max = new_max;
  }
  // After reduction, all lanes have the same (local_max, local_sum)

  // --- Pass 2: Normalize in-place ---
  float inv_sum = 1.0f / local_sum;
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    scores[i] = expf(scores[i] - local_max) * inv_sum;
  }
  __syncwarp();
}

enum class TopkFuncType {
  Naive = 0,
  Radix = 1,
};

/*******************************************************************************
 * radix_topk_and_mask — Warp-level radix-selection based top-K
 *
 * O(E) algorithm independent of K, adapted from PyTorch's radix selection.
 * Uses 4-bit radix (16 buckets) → 8 passes for float32.
 *
 * Algorithm:
 *   Phase 1 — Radix selection (8 passes):
 *     Convert float scores to "order-preserving" uint32 (flip sign bit for
 *     positives, flip all bits for negatives).  Then iterate 4 bits at a time
 *     from the MSB.  Each pass:
 *       1. Each of 32 threads counts elements per radix bucket that match the
 *          "desired" bit pattern found so far.
 *       2. Warp-reduce the per-thread histograms (16 sums).
 *       3. Scan buckets from largest to smallest to locate which bucket
 *          contains the K-th largest element.
 *       4. Narrow the desired pattern by 4 bits.
 *     After 8 passes: the exact uint32 bit pattern of the K-th value is known.
 *
 *   Phase 2 — Gather (single pass over E):
 *     Collect elements strictly greater than the K-th value (same uint order),
 *     then fill remaining slots with elements equal to the K-th value (ties
 *     broken by ascending index for determinism matching torch.topk).
 *     Write indices and scores to the output arrays.
 *
 * Tie-breaking: (value DESC, index ASC) — matches torch.topk behavior.
 *
 * Constraints:
 *   - 0 < topk <= data_size
 *   - No upper limit on topk or data_size (unlike v1's 128 cap)
 *   - scores must be in shared memory accessible by the warp
 *
 * Complexity: 9 × O(E/32) = O(E) per warp, independent of K.
 ******************************************************************************/

__device__ inline void radix_topk_and_mask(CompType *scores, int data_size, int topk,
                                           int *topk_indices, CompType *topk_scores, int lane_id) {
  // assert(topk > 0 && "naive_topk_and_mask_v2: topk must be positive");
  // assert(topk <= data_size && "naive_topk_and_mask_v2: topk exceeds data_size");

  constexpr int RADIX_BITS = 4;
  constexpr int RADIX_SIZE = 1 << RADIX_BITS;  // 16 buckets
  constexpr int RADIX_MASK = RADIX_SIZE - 1;   // 0xF
  constexpr int NUM_PASSES = 32 / RADIX_BITS;  // 8 passes for float32

  // =========================================================================
  // Phase 1: Radix selection — find the bit pattern of the K-th largest value
  // =========================================================================
  unsigned int desired = 0;       // accumulated bit pattern of the K-th value
  unsigned int desired_mask = 0;  // bits determined so far
  int k_remaining = topk;         // how many more elements we need to skip

  for (int pass = NUM_PASSES - 1; pass >= 0; pass--) {
    int digit_pos = pass * RADIX_BITS;

    // Each thread counts elements per bucket that match the desired pattern
    unsigned int counts[RADIX_SIZE];
#pragma unroll
    for (int b = 0; b < RADIX_SIZE; b++) {
      counts[b] = 0;
    }

    for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
      unsigned int u = float_to_ordered_uint(scores[i]);
      // Check if this element matches the desired pattern on already-decided bits
      if ((u & desired_mask) == desired) {
        int bucket = (u >> digit_pos) & RADIX_MASK;
        counts[bucket]++;
      }
    }

    // Warp-reduce each bucket count across all 32 lanes
    unsigned int total_counts[RADIX_SIZE];
#pragma unroll
    for (int b = 0; b < RADIX_SIZE; b++) {
      unsigned int c = warp_allreduce_sum(counts[b]);
      total_counts[b] = c;  // same value on all lanes after full reduction
    }

    // Scan buckets from LARGEST digit value (15) to smallest (0).
    // We're looking for the top-K largest, so we want the highest-valued
    // bucket first.  Accumulate counts until we find the bucket containing
    // the k_remaining-th element.
    int target_bucket = 0;
    for (int b = RADIX_SIZE - 1; b >= 0; b--) {
      unsigned int bc = total_counts[b];
      if (bc < static_cast<unsigned int>(k_remaining)) {
        // All elements in this bucket are in the top set; skip them
        k_remaining -= bc;
      } else {
        // The K-th element is in this bucket
        target_bucket = b;
        break;
      }
    }

    // Update the desired pattern and mask
    desired |= (static_cast<unsigned int>(target_bucket) << digit_pos);
    desired_mask |= (static_cast<unsigned int>(RADIX_MASK) << digit_pos);
  }

  // After all passes, `desired` holds the exact ordered-uint bit pattern of
  // the K-th largest value, and `k_remaining` is the number of elements with
  // that exact value that should be included in the top-K set.
  // (k_remaining >= 1 unless all elements equal the K-th value boundary)

  // =========================================================================
  // Phase 2: Gather — collect top-K elements into output arrays
  // =========================================================================
  // Two sub-passes over the data:
  //   Pass A: Collect all elements strictly greater than the K-th value.
  //   Pass B: Collect elements equal to the K-th value (up to k_remaining),
  //           in ascending index order for deterministic tie-breaking.
  //
  // Since the warp processes indices in strided order, we need a warp-level
  // prefix sum to assign output positions without conflicts.

  // --- Pass A: elements strictly greater than K-th value ---
  // Use a warp-wide running counter for output position.
  int write_pos = 0;  // shared across warp via __shfl_sync

  for (int base = 0; base < data_size; base += kThreadsPerWarp) {
    int i = base + lane_id;
    bool valid = (i < data_size);

    unsigned int u = valid ? float_to_ordered_uint(scores[i]) : 0;
    bool is_greater = valid && (u > desired);

    // Warp ballot to count how many lanes have a qualifying element
    unsigned int ballot = __ballot_sync(0xffffffff, is_greater);
    int lane_prefix = __popc(ballot & ((1u << lane_id) - 1));  // exclusive prefix
    int total_qualifying = __popc(ballot);

    if (is_greater) {
      int out_idx = write_pos + lane_prefix;
      if (out_idx < topk) {
        topk_indices[out_idx] = i;
        topk_scores[out_idx] = scores[i];
      }
    }
    write_pos += total_qualifying;
  }

  // --- Pass B: elements equal to K-th value (up to k_remaining) ---
  int tie_remaining = k_remaining;  // broadcast same value to all lanes

  for (int base = 0; base < data_size && tie_remaining > 0; base += kThreadsPerWarp) {
    int i = base + lane_id;
    bool valid = (i < data_size);

    unsigned int u = valid ? float_to_ordered_uint(scores[i]) : 0;
    bool is_equal = valid && (u == desired);

    unsigned int ballot = __ballot_sync(0xffffffff, is_equal);
    int lane_prefix = __popc(ballot & ((1u << lane_id) - 1));
    int total_equal = __popc(ballot);

    if (is_equal && lane_prefix < tie_remaining) {
      int out_idx = write_pos + lane_prefix;
      if (out_idx < topk) {
        topk_indices[out_idx] = i;
        topk_scores[out_idx] = scores[i];
      }
    }

    int consumed = (total_equal < tie_remaining) ? total_equal : tie_remaining;
    write_pos += consumed;
    tie_remaining -= consumed;
  }

  __syncwarp();
}

__device__ inline void naive_topk_and_mask(CompType *scores, int data_size, int topk,
                                           int *topk_indices, CompType *topk_scores, int lane_id) {
  // Check if the index is masked by the later iteration
  auto is_masked = [&topk_indices](int k, int index) {
    if (k == 0) return false;
    for (int i = 0; i < k; i++) {
      if (topk_indices[i] == index) return true;
    }
    return false;
  };
  // Topk Times: Find the max value and its index
  // Then mask it, and record the index in the topk_indices
  // After looping topk times, the topk_indices will be the topk indices
  for (int k = 0; k < topk; k++) {
    // Find the max value and its index
    CompType val = (lane_id < data_size && !is_masked(k, lane_id))
                       ? scores[lane_id]
                       : -std::numeric_limits<CompType>::infinity();
    int index = (lane_id < data_size) ? lane_id : 0;
    // Some value is hanlded in local thread
    // Thread 0 is responsible for the: 0-th, 32-th, 64-th, 96-th ...
    // Reduce the value in local thread
    for (int i = lane_id + kThreadsPerWarp; i < data_size; i += kThreadsPerWarp) {
      CompType cur_val = (is_masked(k, i)) ? -std::numeric_limits<CompType>::infinity() : scores[i];
      if (cur_val > val) {
        val = cur_val;
        index = i;
      }
    }
    // Warp shuffle between threads
    for (int s = 16; s > 0; s /= 2) {
      auto shuffled_val = __shfl_xor_sync(0xffffffff, val, s);
      auto shuffled_index = __shfl_xor_sync(0xffffffff, index, s);
      if (shuffled_val > val) {
        val = shuffled_val;
        index = shuffled_index;
      }
    }
    if (lane_id == 0) {
      topk_indices[k] = index;
      topk_scores[k] = val;
    }
    __syncwarp();
  }
}

template <TopkFuncType TopkFunc>
__device__ __forceinline__ void topk_and_mask(CompType *scores, int data_size, int topk,
                                              int *topk_indices, CompType *topk_scores,
                                              int lane_id) {
  if constexpr (TopkFunc == TopkFuncType::Radix)
    return radix_topk_and_mask(scores, data_size, topk, topk_indices, topk_scores, lane_id);
  else
    return naive_topk_and_mask(scores, data_size, topk, topk_indices, topk_scores, lane_id);
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

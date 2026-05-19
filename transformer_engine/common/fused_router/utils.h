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

// ============================================================================
// Scalar (per-element) score functions — for fused paths
// ============================================================================

// Forward: y = sigmoid(x) = 1 / (1 + exp(-x))
__device__ __forceinline__ float sigmoid_scalar(float x) { return 1.0f / (1.0f + expf(-x)); }

// Forward: y = sqrt(softplus(x)) = sqrt(log(1 + exp(x)))
__device__ __forceinline__ float sqrtsoftplus_scalar(float x) {
  float sp = (x > 20.0f) ? x : log1pf(expf(x));
  return sqrtf(sp);
}

// Backward: sigmoid — given sigmoid output y, dy/dx = y * (1 - y)
__device__ __forceinline__ float sigmoid_bwd_scalar(float grad, float y) {
  return grad * y * (1.0f - y);
}

// Backward: sqrtsoftplus — given original logit x and sqrtsoftplus output y = sqrt(softplus(x)),
// dy/dx = sigmoid(x) / (2 * y).  For large x (>20), softplus(x) ≈ x so dy/dx ≈ 1/(2*y).
__device__ __forceinline__ float sqrtsoftplus_bwd_scalar(float grad, float x, float y) {
  float dy_dx = (x > 20.0f) ? (1.0f / (2.0f * y + epsilon))
                             : (sigmoid_scalar(x) / (2.0f * y + epsilon));
  return grad * dy_dx;
}

// Backward: normalization — given grad, routed flag, and pre-computed sums.
// Used by sigmoid/sqrtsoftplus with topk > 1.
// sum_act = sum of activation outputs over routed experts.
// sum_grad_act = sum of grad * act over routed experts.
__device__ __forceinline__ float normalize_bwd_scalar(float grad, bool routed, float sum_act,
                                                      float sum_grad_act) {
  if (!routed) return 0.0f;
  float denom = sum_act + epsilon;
  return grad / denom - sum_grad_act / (denom * denom);
}

// Backward: softmax element — given grad, softmax output, and sum(output * grad).
__device__ __forceinline__ float softmax_bwd_scalar(float grad, float act, float dot) {
  return act * (grad - dot);
}

// ============================================================================
// Array (in-place on shmem) softmax — still used by forward kernels
// ============================================================================

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

// Maximum num_experts supported by the packed 8-bit radix topk histogram.
// Each thread processes ceil(data_size/32) elements per bucket.  With 8-bit
// counters the max per-thread count is 255, so data_size <= 255 * 32 = 8160.
constexpr int kMaxExpertsRadixTopk = 255 * 32;  // 8160

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
 * Register pressure optimization: pack 16 bucket counts into 4 registers
 * using 8-bit fields (4 counters per u32).  The original counts[16] +
 * total_counts[16] required 32 registers, causing massive spill to local
 * memory on large kernels (81% of L1 traffic on E=2304, K=36).
 *
 * Tie-breaking: (value DESC, index ASC) — matches torch.topk behavior.
 *
 * Constraints:
 *   - 0 < topk <= data_size
 *   - data_size <= kMaxExpertsRadixTopk (8160) to avoid 8-bit overflow
 *   - scores must be in shared memory accessible by the warp
 *
 * Complexity: 9 × O(E/32) = O(E) per warp, independent of K.
 ******************************************************************************/

__device__ inline void radix_topk_and_mask(CompType *scores, int data_size, int topk,
                                           int *topk_indices, CompType *topk_scores, int lane_id) {
  constexpr int RADIX_BITS = 4;
  constexpr int RADIX_SIZE = 1 << RADIX_BITS;  // 16 buckets
  constexpr int RADIX_MASK = RADIX_SIZE - 1;   // 0xF
  constexpr int NUM_PASSES = 32 / RADIX_BITS;  // 8 passes for float32

  // =========================================================================
  // Phase 1: Radix selection — find the bit pattern of the K-th largest value
  //
  // Packed counters: 16 bucket counts are stored in 4 × u32 registers using
  // 8-bit fields (4 counters per register).  Bucket b is in byte (b % 4) of
  // packed[b / 4].  This reduces register usage from 32 (counts[16] +
  // total_counts[16]) to 4 registers.
  //
  // Max per-thread count per bucket = ceil(data_size / 32).
  // For E=2304: max 72 — fits in 8 bits (max 255).
  // Constraint: data_size must be <= kMaxExpertsRadixTopk (8160).
  // =========================================================================
  unsigned int desired = 0;       // accumulated bit pattern of the K-th value
  unsigned int desired_mask = 0;  // bits determined so far
  int k_remaining = topk;         // how many more elements we need to skip

#pragma unroll 1
  for (int pass = NUM_PASSES - 1; pass >= 0; pass--) {
    int digit_pos = pass * RADIX_BITS;

    // Packed counters: packed[i] holds 4 × 8-bit counts for buckets [4i..4i+3].
    // Bucket b is in byte (b % 4) of packed[b / 4].
    unsigned int packed[4] = {0, 0, 0, 0};

    for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
      unsigned int u = float_to_ordered_uint(scores[i]);
      if ((u & desired_mask) == desired) {
        int bucket = (u >> digit_pos) & RADIX_MASK;
        int pack_idx = bucket >> 2;         // bucket / 4
        int byte_idx = bucket & 3;          // bucket % 4
        int shift = byte_idx << 3;          // byte_idx * 8
        packed[pack_idx] += (1u << shift);  // increment the 8-bit field
      }
    }

    // Warp-reduce each bucket, then scan to find the target bucket.
    int target_bucket = 0;
    int k_remaining_copy = k_remaining;
#pragma unroll
    for (int b = RADIX_SIZE - 1; b >= 0; b--) {
      // Unpack: extract 8-bit count for bucket b from packed[b/4]
      int pack_idx = b >> 2;
      int shift = (b & 3) << 3;
      unsigned int my_count = (packed[pack_idx] >> shift) & 0xFFu;
      // Warp-reduce to get total count across all 32 lanes
      unsigned int bc = warp_allreduce_sum(my_count);
      if (bc < static_cast<unsigned int>(k_remaining_copy)) {
        k_remaining_copy -= bc;
      } else {
        target_bucket = b;
        break;
      }
    }
    k_remaining = k_remaining_copy;

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

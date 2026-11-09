/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/* Single-pass sigmoid router for 256 experts with top-8 routing (DeepSeek-V3 style, optionally
 * group-limited with 8 groups / top-4 groups).
 *
 * One forward kernel computes, from one read of the logits, what the generic kernels compute in
 * three launches plus a reduction:
 *   - fused_topk_with_score_function with dense indices (sigmoid, expert bias, group-limited
 *     top-k, normalisation, scaling), with the probabilities written compactly as [T, topk] in
 *     the order of topk_indices instead of as a dense [T, E] tensor;
 *   - fused_compute_score_for_moe_aux_loss (sigmoid scores normalised over all experts) and
 *     the column sums of its routing map, i.e. the local tokens per expert of the aux loss;
 *   - optionally, the column sums of the selected routes added to a caller-owned counter
 *     (the token counts used to update the expert bias).
 * One backward kernel produces the gradient of the logits that the two generic backward kernels
 * (top-k with dense indices, and score-for-aux-loss) produce, summed as autograd sums the two
 * logits-dtype gradients.
 *
 * Every floating-point operation, reduction order and tie-break follows the generic kernels
 * (fused_topk_forward_simple_kernel with the naive top-k, the simple aux-score kernel,
 * fused_topk_backward_selected_indices_kernel and fused_score_for_moe_aux_loss_backward_kernel),
 * so the outputs are bitwise identical to them for finite logits.
 *
 * Tie-breaking of the naive warp top-k: lane i scans experts i, i + 32, ... with a strict '>',
 * and the xor 16/8/4/2/1 butterfly keeps its own candidate on equal values, so among equal values
 * the expert with the smaller 5-bit reversal of (e % 32) wins, then the smaller e / 32.
 * tie_rank() encodes that order (smaller wins), so each lane can keep a sorted list of its own
 * experts and the warp can merge them with a single (value, rank) argmax per selected expert.
 */

#include <cuda_runtime.h>
#include <transformer_engine/fused_router.h>

#include <cstdint>
#include <limits>

#include "../common.h"
#include "../util/logging.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_router {
namespace {

constexpr int kNumExperts = 256;
constexpr int kTopk = 8;
constexpr int kNumGroups = 8;
constexpr int kGroupTopk = 4;
constexpr int kExpertsPerLane = kNumExperts / kThreadsPerWarp;  // lane l owns experts [8l, 8l+8)
constexpr int kLanesPerGroup = kNumExperts / kNumGroups / kExpertsPerLane;
constexpr int kWarpsPerBlock = 8;  // one token per warp
constexpr int kBwdThreadsPerBlock = 256;
static_assert(kExpertsPerLane == 8 && kLanesPerGroup == 4, "layout assumes 256 experts, 8 groups");
static_assert(kTopk / kGroupTopk == 2, "group score is the sum of the group's top-2");

__device__ __forceinline__ int bitrev5(int x) {
  return static_cast<int>(__brev(static_cast<unsigned>(x)) >> 27);
}
__device__ __forceinline__ int tie_rank(int e) { return (bitrev5(e & 31) << 3) | (e >> 5); }
__device__ __forceinline__ int expert_of_rank(int r) { return bitrev5(r >> 3) | ((r & 7) << 5); }
constexpr int kNoRank = 0x7fffffff;

// (va, ra) precedes (vb, rb) in the naive top-k order
__device__ __forceinline__ bool precedes(float va, int ra, float vb, int rb) {
  return va > vb || (va == vb && ra < rb);
}

__device__ __forceinline__ void compare_swap(float &va, int &ra, float &vb, int &rb) {
  const bool swap = precedes(vb, rb, va, ra);
  const float tv = swap ? vb : va;
  const int tr = swap ? rb : ra;
  vb = swap ? va : vb;
  rb = swap ? ra : rb;
  va = tv;
  ra = tr;
}

// Sorts a lane's 8 candidates into the naive top-k order (19-comparator network).
__device__ __forceinline__ void sort8(float (&v)[8], int (&r)[8]) {
  compare_swap(v[0], r[0], v[1], r[1]);
  compare_swap(v[2], r[2], v[3], r[3]);
  compare_swap(v[4], r[4], v[5], r[5]);
  compare_swap(v[6], r[6], v[7], r[7]);
  compare_swap(v[0], r[0], v[2], r[2]);
  compare_swap(v[1], r[1], v[3], r[3]);
  compare_swap(v[4], r[4], v[6], r[6]);
  compare_swap(v[5], r[5], v[7], r[7]);
  compare_swap(v[1], r[1], v[2], r[2]);
  compare_swap(v[5], r[5], v[6], r[6]);
  compare_swap(v[0], r[0], v[4], r[4]);
  compare_swap(v[3], r[3], v[7], r[7]);
  compare_swap(v[1], r[1], v[5], r[5]);
  compare_swap(v[2], r[2], v[6], r[6]);
  compare_swap(v[1], r[1], v[4], r[4]);
  compare_swap(v[3], r[3], v[6], r[6]);
  compare_swap(v[2], r[2], v[4], r[4]);
  compare_swap(v[3], r[3], v[5], r[5]);
  compare_swap(v[3], r[3], v[4], r[4]);
}

__device__ __forceinline__ void pop_front(float (&v)[8], int (&r)[8]) {
#pragma unroll
  for (int i = 0; i < 7; ++i) {
    v[i] = v[i + 1];
    r[i] = r[i + 1];
  }
  v[7] = -std::numeric_limits<float>::infinity();
  r[7] = kNoRank;
}

// Argmax of (value, rank) over aligned groups of `width` lanes (width is a power of two <= 32).
template <int width>
__device__ __forceinline__ void lane_group_best(float &v, int &r) {
#pragma unroll
  for (int m = 1; m < width; m <<= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, m);
    const int orank = __shfl_xor_sync(0xffffffffu, r, m);
    if (precedes(ov, orank, v, r)) {
      v = ov;
      r = orank;
    }
  }
}

template <typename T>
__device__ __forceinline__ void load8_as_float(const T *src, float (&dst)[8]) {
  // 16-byte aligned (checked on the host): one or two 16-byte loads per lane
  constexpr int kVecs = 8 * sizeof(T) / sizeof(uint4);
  static_assert(kVecs >= 1, "unsupported element size");
  uint4 raw[kVecs];
#pragma unroll
  for (int i = 0; i < kVecs; ++i) raw[i] = __ldg(reinterpret_cast<const uint4 *>(src) + i);
  const T *vals = reinterpret_cast<const T *>(raw);
#pragma unroll
  for (int i = 0; i < 8; ++i) dst[i] = static_cast<CompType>(vals[i]);
}

template <typename T>
__device__ __forceinline__ void store8(T *dst, const T (&src)[8]) {
  constexpr int kVecs = 8 * sizeof(T) / sizeof(uint4);
  static_assert(kVecs >= 1, "unsupported element size");
#pragma unroll
  for (int i = 0; i < kVecs; ++i) {
    reinterpret_cast<uint4 *>(dst)[i] = reinterpret_cast<const uint4 *>(src)[i];
  }
}

__device__ __forceinline__ float load_bias(const void *bias, DType dtype, int e) {
  switch (dtype) {
    case DType::kFloat32:
      return reinterpret_cast<const float *>(bias)[e];
    case DType::kBFloat16:
      return static_cast<CompType>(reinterpret_cast<const bf16 *>(bias)[e]);
    case DType::kFloat16:
      return static_cast<CompType>(reinterpret_cast<const fp16 *>(bias)[e]);
    default:
      return 0.0f;
  }
}

template <typename DataType, typename ProbType, typename IndexType, bool kGrouped>
__global__ void __launch_bounds__(kWarpsPerBlock *kThreadsPerWarp)
    fused_topk_with_aux_loss_scores_forward_kernel(
        const DataType *__restrict__ logits, const void *__restrict__ expert_bias,
        DType expert_bias_dtype, int num_tokens, float scaling_factor, ProbType *__restrict__ probs,
        IndexType *__restrict__ topk_indices, CompType *__restrict__ aux_scores,
        int64_t *__restrict__ aux_tokens_per_expert, int64_t *__restrict__ tokens_per_expert) {
  __shared__ CompType scores_smem[kWarpsPerBlock][kNumExperts];
  __shared__ int aux_count_smem[kNumExperts];
  __shared__ int route_count_smem[kNumExperts];
  for (int i = threadIdx.x; i < kNumExperts; i += blockDim.x) {
    aux_count_smem[i] = 0;
    route_count_smem[i] = 0;
  }
  __syncthreads();

  const int lane = threadIdx.x % kThreadsPerWarp;
  const int warp = threadIdx.x / kThreadsPerWarp;
  const int e0 = lane * kExpertsPerLane;
  const int token = blockIdx.x * kWarpsPerBlock + warp;
  CompType *scores = scores_smem[warp];

  if (token < num_tokens) {  // warp-uniform; no early return because of the block reduction below
    const size_t row = static_cast<size_t>(token) * kNumExperts;
    // Sigmoid scores; scores[] in shared memory is the generic kernels' intermediate_output.
    float s[kExpertsPerLane];
    load8_as_float(logits + row + e0, s);
#pragma unroll
    for (int i = 0; i < kExpertsPerLane; ++i) s[i] = sigmoid_scalar(s[i]);
#pragma unroll
    for (int i = 0; i < kExpertsPerLane; ++i) scores[e0 + i] = s[i];
    __syncwarp();

    int rank[kExpertsPerLane];
#pragma unroll
    for (int i = 0; i < kExpertsPerLane; ++i) rank[i] = tie_rank(e0 + i);

    // ---- Aux-loss scores: s / (sum(s) + eps), with warp_reduce_on_shmem's summation order
    {
      CompType sum = scores[lane];
#pragma unroll
      for (int j = 1; j < kNumExperts / kThreadsPerWarp; ++j) sum += scores[lane + j * 32];
      sum += __shfl_xor_sync(0xffffffffu, sum, 16);
      sum += __shfl_xor_sync(0xffffffffu, sum, 8);
      sum += __shfl_xor_sync(0xffffffffu, sum, 4);
      sum += __shfl_xor_sync(0xffffffffu, sum, 2);
      sum += __shfl_xor_sync(0xffffffffu, sum, 1);
      float a[kExpertsPerLane];
      int ar[kExpertsPerLane];
#pragma unroll
      for (int i = 0; i < kExpertsPerLane; ++i) {
        a[i] = s[i] / (sum + epsilon);
        ar[i] = rank[i];
      }
      float4 *dst = reinterpret_cast<float4 *>(aux_scores + row + e0);
      dst[0] = make_float4(a[0], a[1], a[2], a[3]);
      dst[1] = make_float4(a[4], a[5], a[6], a[7]);
      // Aux-loss routing map (top-k of the normalised scores) -> local tokens per expert
      sort8(a, ar);
#pragma unroll
      for (int k = 0; k < kTopk; ++k) {
        float v = a[0];
        int r = ar[0];
        lane_group_best<kThreadsPerWarp>(v, r);
        if (r == ar[0]) {  // this lane owns the winner
          pop_front(a, ar);
          atomicAdd(&aux_count_smem[expert_of_rank(r)], 1);
        }
      }
    }

    // ---- Routing: top-k of s + bias, optionally limited to the best groups
    int selected[kTopk];
    {
      float v[kExpertsPerLane];
      int vr[kExpertsPerLane];
#pragma unroll
      for (int i = 0; i < kExpertsPerLane; ++i) {
        v[i] = s[i];
        if (expert_bias != nullptr) v[i] += load_bias(expert_bias, expert_bias_dtype, e0 + i);
        vr[i] = rank[i];
      }
      sort8(v, vr);
      if constexpr (kGrouped) {
        // Group score = (0 + top1) + top2 over the group's 4 lanes.
        float top1 = v[0];
        int top1_rank = vr[0];
        lane_group_best<kLanesPerGroup>(top1, top1_rank);
        const bool own_top1 = vr[0] == top1_rank;
        float top2 = own_top1 ? v[1] : v[0];
        int top2_rank = own_top1 ? vr[1] : vr[0];
        lane_group_best<kLanesPerGroup>(top2, top2_rank);
        CompType group_score = 0.0f;
        group_score = group_score + top1;
        group_score = group_score + top2;
        // Keep the group if fewer than group_topk groups precede it (group h competes as lane h
        // of the generic kernel's top-k over the group scores).
        const int group = lane / kLanesPerGroup;
        int preceding = 0;
#pragma unroll
        for (int h = 0; h < kNumGroups; ++h) {
          const float other = __shfl_sync(0xffffffffu, group_score, h * kLanesPerGroup);
          preceding += precedes(other, bitrev5(h), group_score, bitrev5(group)) ? 1 : 0;
        }
        if (preceding >= kGroupTopk) {
#pragma unroll
          for (int i = 0; i < kExpertsPerLane; ++i) {
            v[i] = -std::numeric_limits<float>::infinity();
            vr[i] = kNoRank;
          }
        }
      }
#pragma unroll
      for (int k = 0; k < kTopk; ++k) {
        float best = v[0];
        int best_rank = vr[0];
        lane_group_best<kThreadsPerWarp>(best, best_rank);
        const int e = expert_of_rank(best_rank);
        if (best_rank == vr[0]) {
          pop_front(v, vr);
          if (tokens_per_expert != nullptr) atomicAdd(&route_count_smem[e], 1);
        }
        selected[k] = e;
      }
    }

    if (lane == 0) {
      // Normalise the selected pre-bias sigmoid scores (warp_reduce_on_shmem order for 8
      // values: lanes 0-7 hold them, xor 16 and 8 add zeros), scale, round to the logits dtype.
      float t[kTopk];
#pragma unroll
      for (int k = 0; k < kTopk; ++k) t[k] = (scores[selected[k]] + 0.0f) + 0.0f;
      const CompType sum = ((t[0] + t[4]) + (t[2] + t[6])) + ((t[1] + t[5]) + (t[3] + t[7]));
      alignas(16) ProbType p[kTopk];
      alignas(16) IndexType idx[kTopk];
#pragma unroll
      for (int k = 0; k < kTopk; ++k) {
        const CompType normalized = scores[selected[k]] / (sum + epsilon);
        const DataType prob = static_cast<DataType>(scaling_factor * normalized);
        p[k] = static_cast<ProbType>(static_cast<CompType>(prob));
        idx[k] = static_cast<IndexType>(selected[k]);
      }
      store8(probs + static_cast<size_t>(token) * kTopk, p);
      store8(topk_indices + static_cast<size_t>(token) * kTopk, idx);
    }
  }

  __syncthreads();
  for (int i = threadIdx.x; i < kNumExperts; i += blockDim.x) {
    const int aux_count = aux_count_smem[i];
    if (aux_count != 0) {
      atomicAdd(reinterpret_cast<unsigned long long *>(aux_tokens_per_expert) + i,  // NOLINT(*)
                static_cast<unsigned long long>(aux_count));                        // NOLINT(*)
    }
    const int route_count = route_count_smem[i];
    if (route_count != 0) {
      atomicAdd(reinterpret_cast<unsigned long long *>(tokens_per_expert) + i,  // NOLINT(*)
                static_cast<unsigned long long>(route_count));                  // NOLINT(*)
    }
  }
}

// One warp per token; lane l owns experts l + 32 j (the generic kernels' layout).
template <typename DataType, typename GradProbType, typename IndexType, bool kHasAuxGrad>
__global__ void __launch_bounds__(kBwdThreadsPerBlock)
    fused_topk_with_aux_loss_scores_backward_kernel(const DataType *__restrict__ logits,
                                                    const IndexType *__restrict__ topk_indices,
                                                    const GradProbType *__restrict__ grad_probs,
                                                    const CompType *__restrict__ grad_aux_scores,
                                                    int num_tokens, float scaling_factor,
                                                    DataType *__restrict__ grad_logits) {
  constexpr int kPerLane = kNumExperts / kThreadsPerWarp;
  const int lane = threadIdx.x % kThreadsPerWarp;
  const int token = static_cast<int>((static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) /
                                     kThreadsPerWarp);
  if (token >= num_tokens) return;  // warp-uniform
  const size_t row = static_cast<size_t>(token) * kNumExperts;

  float act[kPerLane];
#pragma unroll
  for (int j = 0; j < kPerLane; ++j) {
    act[j] = sigmoid_scalar(static_cast<CompType>(logits[row + lane + j * kThreadsPerWarp]));
  }

  // Top-k part (fused_topk_backward_selected_indices_kernel): lane k handles route k.
  int route_expert = -1;
  CompType route_grad = 0.0f;
  {
    CompType sum_act = 0.0f;
    CompType sum_grad_act = 0.0f;
    CompType g = 0.0f;
    CompType a = 0.0f;
    for (int k = lane; k < kTopk; k += kThreadsPerWarp) {
      route_expert = static_cast<int>(topk_indices[static_cast<size_t>(token) * kTopk + k]);
      // A float32 gradient of logits-dtype probabilities is first rounded to the logits dtype,
      // as the autograd cast back to the probabilities' dtype would do.
      const DataType grad =
          static_cast<DataType>(grad_probs[static_cast<size_t>(token) * kTopk + k]);
      g = static_cast<CompType>(grad) * scaling_factor;
      a = sigmoid_scalar(static_cast<CompType>(logits[row + route_expert]));
      sum_act += a;
      sum_grad_act += g * a;
    }
    sum_act = warp_allreduce_sum(sum_act);
    sum_grad_act = warp_allreduce_sum(sum_grad_act);
    if (lane < kTopk) {
      g = normalize_bwd_scalar(g, true, sum_act, sum_grad_act);
      route_grad = sigmoid_bwd_scalar(g, a);
    }
  }

  // Aux-score part (fused_score_for_moe_aux_loss_backward_kernel).
  float aux_grad[kPerLane];
  if constexpr (kHasAuxGrad) {
    float grad[kPerLane];
    CompType sum_act = 0.0f;
    CompType sum_grad_act = 0.0f;
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
      grad[j] = grad_aux_scores[row + lane + j * kThreadsPerWarp];
      sum_act += act[j];
      sum_grad_act += grad[j] * act[j];
    }
    sum_act = warp_allreduce_sum(sum_act);
    sum_grad_act = warp_allreduce_sum(sum_grad_act);
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
      const CompType g = normalize_bwd_scalar(grad[j], true, sum_act, sum_grad_act);
      aux_grad[j] = sigmoid_bwd_scalar(g, act[j]);
    }
  }

  // Move each route's gradient to the lane that owns its expert.
  float topk_grad[kPerLane];
  bool routed[kPerLane];
#pragma unroll
  for (int j = 0; j < kPerLane; ++j) {
    topk_grad[j] = 0.0f;
    routed[j] = false;
  }
#pragma unroll
  for (int k = 0; k < kTopk; ++k) {
    const int e = __shfl_sync(0xffffffffu, route_expert, k);
    const float g = __shfl_sync(0xffffffffu, route_grad, k);
#pragma unroll
    for (int j = 0; j < kPerLane; ++j) {
      if (e == lane + j * kThreadsPerWarp) {
        topk_grad[j] = g;
        routed[j] = true;
      }
    }
  }

#pragma unroll
  for (int j = 0; j < kPerLane; ++j) {
    DataType out;
    if constexpr (kHasAuxGrad) {
      // autograd's sum of the two logits-dtype gradients (0 where the expert is not routed)
      const CompType topk_part = static_cast<CompType>(static_cast<DataType>(topk_grad[j]));
      const CompType aux_part = static_cast<CompType>(static_cast<DataType>(aux_grad[j]));
      // __fadd_rn: a separate rounded add (no contraction into the products above), as
      // autograd adds the two materialised gradients.
      out = static_cast<DataType>(__fadd_rn(topk_part, aux_part));
    } else {
      out = static_cast<DataType>(routed[j] ? topk_grad[j] : 0.0f);
    }
    grad_logits[row + lane + j * kThreadsPerWarp] = out;
  }
}

bool is_supported_config(int num_experts, int topk, int num_groups, int group_topk,
                         int score_function) {
  const bool grouped = group_topk > 0;
  return score_function == 0 && num_experts == kNumExperts && topk == kTopk &&
         (!grouped || (num_groups == kNumGroups && group_topk == kGroupTopk));
}

void check_config(int num_experts, int topk, int num_groups, int group_topk, int score_function) {
  NVTE_CHECK(is_supported_config(num_experts, topk, num_groups, group_topk, score_function),
             "fused top-k with aux-loss scores supports score_function=sigmoid, num_experts=",
             kNumExperts, ", topk=", kTopk, " and either no group limit or num_groups=", kNumGroups,
             " with group_topk=", kGroupTopk, "; got score_function=", score_function,
             ", num_experts=", num_experts, ", topk=", topk, ", num_groups=", num_groups,
             ", group_topk=", group_topk);
}

void check_aligned(const void *ptr, const char *name) {
  NVTE_CHECK(reinterpret_cast<uintptr_t>(ptr) % 16 == 0, name, " must be 16-byte aligned");
}

#define TE_ROUTER_INDEX_32_64_SWITCH(dtype, type, ...)                                \
  switch (dtype) {                                                                    \
    case DType::kInt32: {                                                             \
      using type = int32_t;                                                           \
      { __VA_ARGS__ }                                                                 \
    } break;                                                                          \
    case DType::kInt64: {                                                             \
      using type = int64_t;                                                           \
      { __VA_ARGS__ }                                                                 \
    } break;                                                                          \
    default:                                                                          \
      NVTE_ERROR("topk_indices must be int32 or int64, got ", to_string(dtype), "."); \
  }

}  // namespace

void fused_topk_with_aux_loss_scores_forward(const Tensor &logits, int num_tokens, int num_experts,
                                             int topk, int num_groups, int group_topk,
                                             float scaling_factor, int score_function,
                                             const Tensor &expert_bias, Tensor &probs,
                                             Tensor &topk_indices, Tensor &aux_scores,
                                             Tensor &aux_tokens_per_expert,
                                             Tensor &tokens_per_expert, cudaStream_t stream) {
  check_config(num_experts, topk, num_groups, group_topk, score_function);
  NVTE_CHECK(num_tokens >= 0, "num_tokens must be non-negative, got ", num_tokens);
  const std::vector<size_t> dense_shape{static_cast<size_t>(num_tokens),
                                        static_cast<size_t>(num_experts)};
  const std::vector<size_t> topk_shape{static_cast<size_t>(num_tokens), static_cast<size_t>(topk)};
  const std::vector<size_t> expert_shape{static_cast<size_t>(num_experts)};
  NVTE_CHECK(logits.data.shape == dense_shape, "logits shape must be ", dense_shape, ", got ",
             logits.data.shape);
  NVTE_CHECK(probs.data.shape == topk_shape, "probs shape must be ", topk_shape, ", got ",
             probs.data.shape);
  NVTE_CHECK(topk_indices.data.shape == topk_shape, "topk_indices shape must be ", topk_shape,
             ", got ", topk_indices.data.shape);
  NVTE_CHECK(aux_scores.data.shape == dense_shape, "aux_scores shape must be ", dense_shape,
             ", got ", aux_scores.data.shape);
  NVTE_CHECK(aux_scores.data.dtype == DType::kFloat32, "aux_scores must be float32");
  NVTE_CHECK(aux_tokens_per_expert.data.shape == expert_shape &&
                 aux_tokens_per_expert.data.dtype == DType::kInt64,
             "aux_tokens_per_expert must be int64 with shape ", expert_shape);
  NVTE_CHECK(probs.data.dtype == logits.data.dtype || probs.data.dtype == DType::kFloat32,
             "probs must have the logits dtype or float32");
  const bool has_bias = expert_bias.has_data();
  if (has_bias) {
    NVTE_CHECK(expert_bias.data.shape == expert_shape, "expert_bias shape must be ", expert_shape,
               ", got ", expert_bias.data.shape);
    NVTE_CHECK(expert_bias.data.dtype == DType::kFloat32 ||
                   expert_bias.data.dtype == DType::kBFloat16 ||
                   expert_bias.data.dtype == DType::kFloat16,
               "expert_bias must be float32, bfloat16 or float16");
  }
  const bool count_routes = tokens_per_expert.has_data();
  if (count_routes) {
    NVTE_CHECK(tokens_per_expert.data.shape == expert_shape &&
                   tokens_per_expert.data.dtype == DType::kInt64,
               "tokens_per_expert must be int64 with shape ", expert_shape);
  }
  NVTE_CHECK_CUDA(
      cudaMemsetAsync(aux_tokens_per_expert.data.dptr, 0, num_experts * sizeof(int64_t), stream));
  if (num_tokens == 0) return;
  check_aligned(logits.data.dptr, "logits");
  check_aligned(probs.data.dptr, "probs");
  check_aligned(topk_indices.data.dptr, "topk_indices");
  check_aligned(aux_scores.data.dptr, "aux_scores");

  const int blocks = (num_tokens + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const bool grouped = group_topk > 0;
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      logits.data.dtype, DataType,
      TE_ROUTER_INDEX_32_64_SWITCH(
          topk_indices.data.dtype, IndexType,
          auto launch =
              [&](auto kernel, auto *probs_ptr) {
                kernel<<<blocks, kWarpsPerBlock * kThreadsPerWarp, 0, stream>>>(
                    reinterpret_cast<const DataType *>(logits.data.dptr),
                    has_bias ? expert_bias.data.dptr : nullptr,
                    has_bias ? expert_bias.data.dtype : DType::kFloat32, num_tokens, scaling_factor,
                    probs_ptr, reinterpret_cast<IndexType *>(topk_indices.data.dptr),
                    reinterpret_cast<CompType *>(aux_scores.data.dptr),
                    reinterpret_cast<int64_t *>(aux_tokens_per_expert.data.dptr),
                    count_routes ? reinterpret_cast<int64_t *>(tokens_per_expert.data.dptr)
                                 : nullptr);
                NVTE_CHECK_CUDA(cudaGetLastError());
              };
          if (probs.data.dtype == logits.data.dtype) {
            auto *probs_ptr = reinterpret_cast<DataType *>(probs.data.dptr);
            if (grouped) {
              launch(fused_topk_with_aux_loss_scores_forward_kernel<DataType, DataType, IndexType,
                                                                    true>,
                     probs_ptr);
            } else {
              launch(fused_topk_with_aux_loss_scores_forward_kernel<DataType, DataType, IndexType,
                                                                    false>,
                     probs_ptr);
            }
          } else {
            auto *probs_ptr = reinterpret_cast<float *>(probs.data.dptr);
            if (grouped) {
              launch(
                  fused_topk_with_aux_loss_scores_forward_kernel<DataType, float, IndexType, true>,
                  probs_ptr);
            } else {
              launch(
                  fused_topk_with_aux_loss_scores_forward_kernel<DataType, float, IndexType, false>,
                  probs_ptr);
            }
          }););
}

void fused_topk_with_aux_loss_scores_backward(const Tensor &logits, const Tensor &topk_indices,
                                              const Tensor &grad_probs,
                                              const Tensor &grad_aux_scores, int num_tokens,
                                              int num_experts, int topk, float scaling_factor,
                                              int score_function, Tensor &grad_logits,
                                              cudaStream_t stream) {
  check_config(num_experts, topk, /*num_groups=*/0, /*group_topk=*/0, score_function);
  NVTE_CHECK(num_tokens >= 0, "num_tokens must be non-negative, got ", num_tokens);
  const std::vector<size_t> dense_shape{static_cast<size_t>(num_tokens),
                                        static_cast<size_t>(num_experts)};
  const std::vector<size_t> topk_shape{static_cast<size_t>(num_tokens), static_cast<size_t>(topk)};
  NVTE_CHECK(logits.data.shape == dense_shape, "logits shape must be ", dense_shape, ", got ",
             logits.data.shape);
  NVTE_CHECK(grad_logits.data.shape == dense_shape && grad_logits.data.dtype == logits.data.dtype,
             "grad_logits must match logits in shape and dtype");
  NVTE_CHECK(topk_indices.data.shape == topk_shape, "topk_indices shape must be ", topk_shape,
             ", got ", topk_indices.data.shape);
  NVTE_CHECK(grad_probs.data.shape == topk_shape, "grad_probs shape must be ", topk_shape, ", got ",
             grad_probs.data.shape);
  NVTE_CHECK(grad_probs.data.dtype == logits.data.dtype || grad_probs.data.dtype == DType::kFloat32,
             "grad_probs must have the logits dtype or float32");
  const bool has_aux_grad = grad_aux_scores.has_data();
  if (has_aux_grad) {
    NVTE_CHECK(
        grad_aux_scores.data.shape == dense_shape && grad_aux_scores.data.dtype == DType::kFloat32,
        "grad_aux_scores must be float32 with shape ", dense_shape);
  }
  if (num_tokens == 0) return;
  const int64_t threads = static_cast<int64_t>(num_tokens) * kThreadsPerWarp;
  const int blocks = static_cast<int>((threads + kBwdThreadsPerBlock - 1) / kBwdThreadsPerBlock);
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      logits.data.dtype, DataType,
      TE_ROUTER_INDEX_32_64_SWITCH(
          topk_indices.data.dtype, IndexType,
          auto launch =
              [&](auto kernel, auto *grad_ptr) {
                kernel<<<blocks, kBwdThreadsPerBlock, 0, stream>>>(
                    reinterpret_cast<const DataType *>(logits.data.dptr),
                    reinterpret_cast<const IndexType *>(topk_indices.data.dptr), grad_ptr,
                    has_aux_grad ? reinterpret_cast<const CompType *>(grad_aux_scores.data.dptr)
                                 : nullptr,
                    num_tokens, scaling_factor,
                    reinterpret_cast<DataType *>(grad_logits.data.dptr));
                NVTE_CHECK_CUDA(cudaGetLastError());
              };
          if (grad_probs.data.dtype == logits.data.dtype) {
            const auto *grad_ptr = reinterpret_cast<const DataType *>(grad_probs.data.dptr);
            if (has_aux_grad) {
              launch(fused_topk_with_aux_loss_scores_backward_kernel<DataType, DataType, IndexType,
                                                                     true>,
                     grad_ptr);
            } else {
              launch(fused_topk_with_aux_loss_scores_backward_kernel<DataType, DataType, IndexType,
                                                                     false>,
                     grad_ptr);
            }
          } else {
            const auto *grad_ptr = reinterpret_cast<const float *>(grad_probs.data.dptr);
            if (has_aux_grad) {
              launch(
                  fused_topk_with_aux_loss_scores_backward_kernel<DataType, float, IndexType, true>,
                  grad_ptr);
            } else {
              launch(fused_topk_with_aux_loss_scores_backward_kernel<DataType, float, IndexType,
                                                                     false>,
                     grad_ptr);
            }
          }););
}

#undef TE_ROUTER_INDEX_32_64_SWITCH

}  // namespace fused_router
}  // namespace transformer_engine

int nvte_fused_topk_with_aux_loss_scores_is_supported(int num_experts, int topk, int num_groups,
                                                      int group_topk, int score_function) {
  return transformer_engine::fused_router::is_supported_config(num_experts, topk, num_groups,
                                                               group_topk, score_function)
             ? 1
             : 0;
}

void nvte_fused_topk_with_aux_loss_scores_forward(
    const NVTETensor logits, int num_tokens, int num_experts, int topk, int num_groups,
    int group_topk, float scaling_factor, int score_function, const NVTETensor expert_bias,
    NVTETensor probs, NVTETensor topk_indices, NVTETensor aux_scores,
    NVTETensor aux_tokens_per_expert, NVTETensor tokens_per_expert, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_aux_loss_scores_forward);
  using namespace transformer_engine;
  fused_router::fused_topk_with_aux_loss_scores_forward(
      *convertNVTETensorCheck(logits), num_tokens, num_experts, topk, num_groups, group_topk,
      scaling_factor, score_function, *convertNVTETensorCheck(expert_bias),
      *convertNVTETensorCheck(probs), *convertNVTETensorCheck(topk_indices),
      *convertNVTETensorCheck(aux_scores), *convertNVTETensorCheck(aux_tokens_per_expert),
      *convertNVTETensorCheck(tokens_per_expert), stream);
}

void nvte_fused_topk_with_aux_loss_scores_backward(
    const NVTETensor logits, const NVTETensor topk_indices, const NVTETensor grad_probs,
    const NVTETensor grad_aux_scores, int num_tokens, int num_experts, int topk,
    float scaling_factor, int score_function, NVTETensor grad_logits, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_aux_loss_scores_backward);
  using namespace transformer_engine;
  fused_router::fused_topk_with_aux_loss_scores_backward(
      *convertNVTETensorCheck(logits), *convertNVTETensorCheck(topk_indices),
      *convertNVTETensorCheck(grad_probs), *convertNVTETensorCheck(grad_aux_scores), num_tokens,
      num_experts, topk, scaling_factor, score_function, *convertNVTETensorCheck(grad_logits),
      stream);
}

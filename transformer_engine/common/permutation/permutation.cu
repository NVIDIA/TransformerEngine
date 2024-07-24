/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/permutation.h>

#include "../common.h"

static __global__ void moe_permute_row_map(const int *sorted_row_id, int *row_id_map,
                                           const int num_rows, const int num_topK,
                                           const int num_out_tokens) {
  // Each block corresponds to one source token
  // row_id_map[num_topK][num_rows]
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int idx = bid * blockDim.x + tid;

  if (idx >= num_rows * num_topK) return;

  int source_row = sorted_row_id[idx];
  int source_token_id = source_row / num_topK;
  int source_topK_id = source_row % num_topK;

  if (idx >= num_out_tokens) {
    row_id_map[source_topK_id * num_rows + source_token_id] = -1;
  } else {
    row_id_map[source_topK_id * num_rows + source_token_id] = idx;
  }
}

template <typename T, typename TCompute, bool hasProb>
__global__ void moe_unpermute_kernel(const T *input, T *unpermuted_output, const int *row_id_map,
                                     const float *prob, const int num_rows, const int num_topK,
                                     const int num_cols) {
  extern __shared__ int8_t s_mem[];
  TCompute *s_prob = reinterpret_cast<TCompute *>(s_mem);

  // each block corresponds to one source token
  const int source_token = blockIdx.x;
  const int tid = threadIdx.x;

  if (hasProb) {
    for (int i = tid; i < num_topK; i += blockDim.x * blockDim.y) {
      s_prob[i] = TCompute(prob[source_token * num_topK + i]);
    }
    __syncthreads();
  }

  float4 frag_load_store;
  T *frag_load_store_ptr = reinterpret_cast<T *>(&frag_load_store);

  static constexpr int kElementsPerAccess = 16 / sizeof(T);

  for (int i = tid * kElementsPerAccess; i < num_cols; i += blockDim.x * kElementsPerAccess) {
    TCompute frag_elem[kElementsPerAccess];
    TCompute frag_sum[kElementsPerAccess];

    int source_row = row_id_map[source_token];

    if (source_row != -1) {
      const T *source_row_ptr = input + source_row * num_cols;

      frag_load_store = __ldlu(reinterpret_cast<const float4 *>(source_row_ptr + i));

      for (int e = 0; e < kElementsPerAccess; e++) {
        frag_sum[e] = TCompute(frag_load_store_ptr[e]);
      }

      if (hasProb) {
        for (int e = 0; e < kElementsPerAccess; e++) {
          frag_sum[e] = frag_sum[e] * s_prob[0];
        }
      }
    } else {
      for (int e = 0; e < kElementsPerAccess; e++) {
        frag_sum[e] = TCompute(0.0f);
      }
    }

    for (int k = 1; k < num_topK; k++) {
      source_row = row_id_map[k * num_rows + source_token];

      if (source_row == -1) continue;

      const T *source_row_ptr = input + source_row * num_cols;

      frag_load_store = __ldlu(reinterpret_cast<const float4 *>(source_row_ptr + i));

      for (int e = 0; e < kElementsPerAccess; e++) {
        frag_elem[e] = TCompute(frag_load_store_ptr[e]);
      }

      if (hasProb) {
        for (int e = 0; e < kElementsPerAccess; e++) {
          frag_elem[e] = frag_elem[e] * s_prob[k];
        }
      }

      for (int e = 0; e < kElementsPerAccess; e++) {
        frag_sum[e] = frag_sum[e] + frag_elem[e];
      }
    }

    T *dest_row_ptr = unpermuted_output + source_token * num_cols;

    for (int e = 0; e < kElementsPerAccess; e++) {
      frag_load_store_ptr[e] = T(frag_sum[e]);
    }

    *(float4 *)(dest_row_ptr + i) = frag_load_store;
  }
}

template <typename T, typename TCompute, int topKTile, bool hasProb>
__global__ void moe_permute_kernel(const T *input_bwd, const T *input_fwd, T *act_grad,
                                   const float *prob, float *prob_grad, const int *row_id_map,
                                   const int num_rows, const int num_topK, const int num_cols) {
  extern __shared__ int8_t s_mem[];
  TCompute *s_prob = reinterpret_cast<TCompute *>(s_mem);

  const int source_token = blockIdx.x;
  const int tid = threadIdx.x;

  if (hasProb) {
    for (int i = tid; i < num_topK; i += blockDim.x) {
      s_prob[i] = TCompute(prob[source_token * num_topK + i]);
    }
    __syncthreads();
  }

  float accum[topKTile] = {0.0f};

  float4 frag_load_store;
  T *frag_load_store_ptr = reinterpret_cast<T *>(&frag_load_store);

  static constexpr int kElementsPerAccess = 16 / sizeof(T);

  const T *source_row_ptr = input_bwd + source_token * num_cols;
  for (int i = tid * kElementsPerAccess; i < num_cols; i += blockDim.x * kElementsPerAccess) {
    TCompute frag_src[kElementsPerAccess];

    frag_load_store = __ldlu(reinterpret_cast<const float4 *>(source_row_ptr + i));

    for (int e = 0; e < kElementsPerAccess; e++) frag_src[e] = TCompute(frag_load_store_ptr[e]);

    int index = source_token;

    for (int k = 0; k < topKTile; k++) {
      if (k == num_topK) break;

      int dest_row = row_id_map[index];
      index += num_rows;

      if (dest_row != -1) {
        if (hasProb) {
          for (int e = 0; e < kElementsPerAccess; e++)
            frag_load_store_ptr[e] = T(frag_src[e] * s_prob[k]);
        } else {
          for (int e = 0; e < kElementsPerAccess; e++) frag_load_store_ptr[e] = T(frag_src[e]);
        }

        T *dest_row_ptr = act_grad + dest_row * num_cols;
        *(float4 *)(dest_row_ptr + i) = frag_load_store;

        if (hasProb) {
          const T *input_fwd_ptr = input_fwd + dest_row * num_cols;

          frag_load_store = __ldlu(reinterpret_cast<const float4 *>(input_fwd_ptr + i));

          TCompute frag_input_fwd[kElementsPerAccess];
          for (int e = 0; e < kElementsPerAccess; e++)
            frag_input_fwd[e] = TCompute(frag_load_store_ptr[e]);

          for (int e = 0; e < kElementsPerAccess; e++) {
            accum[k] += float(frag_src[e] * frag_input_fwd[e]);
          }
        }
      }
    }
  }

  if (hasProb) {
    for (int k = 0; k < topKTile; k++) {
      if (k == num_topK) break;

      for (int mask = 16; mask > 0; mask /= 2) {
        accum[k] = accum[k] + __shfl_xor_sync(0xffffffff, accum[k], mask, 32);
      }
    }

    if (tid == 0) {
      for (int k = 0; k < topKTile; k++) {
        if (k == num_topK) break;
        prob_grad[source_token * num_topK + k] = accum[k];
      }
    }
  }
}

template <typename T>
void nvte_permute_launcher(const T *input, T *output, const int *sorted_row_id, int *row_id_map,
                           const float *prob, const int num_rows, const int num_topK,
                           const int num_cols, const int num_out_tokens, float *prob_grad,
                           const T *input_fwd, cudaStream_t stream) {
  using TCompute = typename std::conditional<(std::is_same<T, __nv_fp8_e5m2>::value ||
                                              std::is_same<T, __nv_fp8_e4m3>::value),
                                             half, T>::type;

  static constexpr int kElementsPerAccess = 16 / sizeof(T);

  if (prob == nullptr) {
    if (input_fwd == nullptr) {
      // Permute fwd
      int threads = 64;
      int blocks = (num_rows * num_topK + threads - 1) / threads;
      moe_permute_row_map<<<blocks, threads, 0, stream>>>(sorted_row_id, row_id_map, num_rows,
                                                          num_topK, num_out_tokens);

      blocks = num_rows;
      threads = std::min(num_cols / kElementsPerAccess, 1024);
      moe_permute_kernel<T, TCompute, 128, false><<<blocks, threads, 0, stream>>>(
          input, nullptr, output, nullptr, nullptr, row_id_map, num_rows, num_topK, num_cols);
    } else {
      // Unpermute bwd without probs for topK == 1
      int blocks = num_rows;
      int threads = 32;

      moe_permute_kernel<T, TCompute, 1, false><<<blocks, threads, 0, stream>>>(
          input, input_fwd, output, prob, prob_grad, row_id_map, num_rows, num_topK, num_cols);
    }
  } else {
    // Unpermute bwd with probs
    int blocks = num_rows;
    int threads = 32;
    size_t smem_bytes = num_topK * sizeof(TCompute);

    if (num_topK <= 8) {
      moe_permute_kernel<T, TCompute, 8, true><<<blocks, threads, smem_bytes, stream>>>(
          input, input_fwd, output, prob, prob_grad, row_id_map, num_rows, num_topK, num_cols);
    } else if (num_topK <= 16) {
      moe_permute_kernel<T, TCompute, 16, true><<<blocks, threads, smem_bytes, stream>>>(
          input, input_fwd, output, prob, prob_grad, row_id_map, num_rows, num_topK, num_cols);
    } else if (num_topK <= 32) {
      moe_permute_kernel<T, TCompute, 32, true><<<blocks, threads, smem_bytes, stream>>>(
          input, input_fwd, output, prob, prob_grad, row_id_map, num_rows, num_topK, num_cols);
    } else if (num_topK <= 64) {
      moe_permute_kernel<T, TCompute, 64, true><<<blocks, threads, smem_bytes, stream>>>(
          input, input_fwd, output, prob, prob_grad, row_id_map, num_rows, num_topK, num_cols);
    } else if (num_topK <= 128) {
      moe_permute_kernel<T, TCompute, 128, true><<<blocks, threads, smem_bytes, stream>>>(
          input, input_fwd, output, prob, prob_grad, row_id_map, num_rows, num_topK, num_cols);
    } else {
      NVTE_ERROR("num_topK cannot exceed 128.");
    }
  }
}

template <typename T>
void nvte_unpermute_launcher(const T *input, T *output, int *row_id_map, const float *prob,
                             const int num_rows, const int num_topK, const int num_cols,
                             cudaStream_t stream) {
  using TCompute = typename std::conditional<(std::is_same<T, __nv_fp8_e5m2>::value ||
                                              std::is_same<T, __nv_fp8_e4m3>::value),
                                             half, T>::type;

  static constexpr int kElementsPerAccess = 16 / sizeof(T);

  int blocks = num_rows;
  int threads = std::min(num_cols / kElementsPerAccess, 1024);
  size_t smem_bytes = num_topK * sizeof(TCompute);

  if (prob == nullptr) {
    // Permute bwd
    // Unpermute fwd without probs
    moe_unpermute_kernel<T, TCompute, false><<<blocks, threads, smem_bytes, stream>>>(
        input, output, row_id_map, prob, num_rows, num_topK, num_cols);
  } else {
    // Unpermute fwd with probs
    moe_unpermute_kernel<T, TCompute, true><<<blocks, threads, smem_bytes, stream>>>(
        input, output, row_id_map, prob, num_rows, num_topK, num_cols);
  }
}

void nvte_permute(const void *input, void *output, const transformer_engine::DType dtype,
                  const int *sorted_row_id, int *row_id_map, const float *prob, const int num_rows,
                  const int num_topK, const int num_cols, const int num_out_tokens,
                  float *prob_grad, const void *input_fwd, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      dtype, T,
      nvte_permute_launcher(reinterpret_cast<const T *>(input), reinterpret_cast<T *>(output),
                            sorted_row_id, row_id_map, prob, num_rows, num_topK, num_cols,
                            num_out_tokens, prob_grad, reinterpret_cast<const T *>(input_fwd),
                            stream););
}

void nvte_unpermute(const void *input, void *output, const transformer_engine::DType dtype,
                    int *row_id_map, const float *prob, const int num_rows, const int num_topK,
                    const int num_cols, cudaStream_t stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      dtype, T,
      nvte_unpermute_launcher(reinterpret_cast<const T *>(input), reinterpret_cast<T *>(output),
                              row_id_map, prob, num_rows, num_topK, num_cols, stream););
}

/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/permutation.h>

#include <cub/cub.cuh>

#include "../common.h"

static __global__ void moe_permute_row_map(const int *sorted_row_id, int *row_id_map,
                                           const int num_rows, const int topK,
                                           const int num_out_tokens) {
  // Each block corresponds to one source token
  // row_id_map[topK][num_rows]
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int idx = bid * blockDim.x + tid;

  if (idx >= num_rows * topK) return;

  int source_row = sorted_row_id[idx];
  int source_token_id = source_row / topK;
  int source_topK_id = source_row % topK;

  if (idx >= num_out_tokens) {
    // Set the indices of dropped tokens to -1
    row_id_map[source_topK_id * num_rows + source_token_id] = -1;
  } else {
    // Create a row id map for subsequent unpermute operation
    row_id_map[source_topK_id * num_rows + source_token_id] = idx;
  }
}

template <typename T, typename TCompute, bool hasProb>
__global__ void moe_unpermute_kernel(const T *input, T *unpermuted_output, const int *row_id_map,
                                     const float *prob, const int num_rows, const int topK,
                                     const int num_cols) {
  extern __shared__ int8_t s_mem[];
  TCompute *s_prob = reinterpret_cast<TCompute *>(s_mem);

  // Each block corresponds to one dest token
  const int source_token = blockIdx.x;
  const int tid = threadIdx.x;

  if (hasProb) {
    for (int i = tid; i < topK; i += blockDim.x * blockDim.y) {
      // Load all the topK probs related to the source row into smem
      s_prob[i] = TCompute(prob[source_token * topK + i]);
    }
    __syncthreads();
  }

  // Register buffers for vector type (float4) memory access
  float4 frag_load_store;
  T *frag_load_store_ptr = reinterpret_cast<T *>(&frag_load_store);

  // Number of elemments in frag_load_store
  static constexpr int kElementsPerAccess = 16 / sizeof(T);

  // Traverse along the hidden dimention
  for (int i = tid * kElementsPerAccess; i < num_cols; i += blockDim.x * kElementsPerAccess) {
    TCompute frag_elem[kElementsPerAccess];
    TCompute frag_sum[kElementsPerAccess];

    int source_row = row_id_map[source_token];

    // source_row == -1 represents a dropped token
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

    for (int k = 1; k < topK; k++) {
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
      if constexpr ((std::is_same_v<T, __nv_fp8_e4m3> || std::is_same_v<T, __nv_fp8_e5m2>) &&
                    (!hasProb)) {
        frag_sum[e] = frag_sum[e] / TCompute(topK);
      }
      frag_load_store_ptr[e] = T(frag_sum[e]);
    }

    *reinterpret_cast<float4 *>(dest_row_ptr + i) = frag_load_store;
  }
}

template <typename T, typename TCompute, int topKTile, bool hasProb>
__global__ void moe_permute_kernel(const T *input_bwd, const T *input_fwd, T *act_grad,
                                   const float *prob, float *prob_grad, const int *row_id_map,
                                   const int num_rows, const int topK, const int num_cols) {
  extern __shared__ int8_t s_mem[];
  TCompute *s_prob = reinterpret_cast<TCompute *>(s_mem);

  // Each block corresponds to one source token
  const int source_token = blockIdx.x;
  const int tid = threadIdx.x;

  if (hasProb) {
    for (int i = tid; i < topK; i += blockDim.x) {
      // Load all the topK probs related to the source row into smem
      s_prob[i] = TCompute(prob[source_token * topK + i]);
    }
    __syncthreads();
  }

  // Accumulators for the calculation of prob_grad
  float accum[topKTile] = {0.0f};

  // Register buffers for vector type (float4) memory access
  float4 frag_load_store;
  T *frag_load_store_ptr = reinterpret_cast<T *>(&frag_load_store);

  // Number of elemments in frag_load_store
  static constexpr int kElementsPerAccess = 16 / sizeof(T);

  // The starting address of each source row
  const T *source_row_ptr = input_bwd + source_token * num_cols;

  // Traverse along the hidden dimention
  for (int i = tid * kElementsPerAccess; i < num_cols; i += blockDim.x * kElementsPerAccess) {
    TCompute frag_src[kElementsPerAccess];

    frag_load_store = __ldlu(reinterpret_cast<const float4 *>(source_row_ptr + i));

    for (int e = 0; e < kElementsPerAccess; e++) frag_src[e] = TCompute(frag_load_store_ptr[e]);

    int index = source_token;

    // Process each row in the corresponding topK rows
    for (int k = 0; k < topKTile; k++) {
      if (k == topK) break;

      int dest_row = row_id_map[index];
      index += num_rows;

      if (dest_row != -1) {
        if (hasProb) {
          // Calculate act_grad in unpermute bwd
          for (int e = 0; e < kElementsPerAccess; e++)
            frag_load_store_ptr[e] = T(frag_src[e] * s_prob[k]);
        } else {
          // permute fwd
          for (int e = 0; e < kElementsPerAccess; e++) frag_load_store_ptr[e] = T(frag_src[e]);
        }

        T *dest_row_ptr = act_grad + dest_row * num_cols;
        *reinterpret_cast<float4 *>(dest_row_ptr + i) = frag_load_store;

        if (hasProb) {
          // Inner product calculation for prob_grad in unpermute bwd
          const T *input_fwd_ptr = input_fwd + dest_row * num_cols;

          frag_load_store = __ldlu(reinterpret_cast<const float4 *>(input_fwd_ptr + i));

          TCompute frag_input_fwd[kElementsPerAccess];
          for (int e = 0; e < kElementsPerAccess; e++)
            frag_input_fwd[e] = TCompute(frag_load_store_ptr[e]);

          for (int e = 0; e < kElementsPerAccess; e++) {
            accum[k] += static_cast<float>(frag_src[e] * frag_input_fwd[e]);
          }
        }
      }
    }
  }

  if (hasProb) {
    for (int k = 0; k < topKTile; k++) {
      if (k == topK) break;
      // Warp-level reduction
      for (int mask = 16; mask > 0; mask /= 2) {
        accum[k] = accum[k] + __shfl_xor_sync(0xffffffff, accum[k], mask, 32);
      }
    }

    if (tid == 0) {
      for (int k = 0; k < topKTile; k++) {
        if (k == topK) break;
        prob_grad[source_token * topK + k] = accum[k];
      }
    }
  }
}

template <typename T>
void nvte_permute_launcher(const T *input, T *output, const int *sorted_row_id, int *row_id_map,
                           const float *prob, float *prob_grad, const T *input_fwd,
                           const int num_rows, const int topK, const int num_cols,
                           const int num_out_tokens, cudaStream_t stream) {
  using TCompute = typename std::conditional<(std::is_same<T, __nv_fp8_e5m2>::value ||
                                              std::is_same<T, __nv_fp8_e4m3>::value),
                                             half, T>::type;

  static constexpr int kElementsPerAccess = 16 / sizeof(T);

  if (input_fwd == nullptr) {
    // moe_permute_fwd

    int threads = 64;
    int blocks = (num_rows * topK + threads - 1) / threads;

    moe_permute_row_map<<<blocks, threads, 0, stream>>>(sorted_row_id, row_id_map, num_rows, topK,
                                                        num_out_tokens);
    NVTE_CHECK_CUDA(cudaGetLastError());

    blocks = num_rows;
    threads = std::min(num_cols / kElementsPerAccess, 1024);
    moe_permute_kernel<T, TCompute, 128, false><<<blocks, threads, 0, stream>>>(
        input, nullptr, output, nullptr, nullptr, row_id_map, num_rows, topK, num_cols);
    NVTE_CHECK_CUDA(cudaGetLastError());
  } else {
    // moe_unpermute_bwd

    int threads = 32;
    int blocks = num_rows;

    if (prob == nullptr) {
      // moe_unpermute_bwd without probs

      moe_permute_kernel<T, TCompute, 1, false><<<blocks, threads, 0, stream>>>(
          input, input_fwd, output, nullptr, nullptr, row_id_map, num_rows, topK, num_cols);
      NVTE_CHECK_CUDA(cudaGetLastError());
    } else {
      // moe_unpermute_bwd with probs

      size_t smem_bytes = topK * sizeof(TCompute);

      if (topK <= 8) {
        moe_permute_kernel<T, TCompute, 8, true><<<blocks, threads, smem_bytes, stream>>>(
            input, input_fwd, output, prob, prob_grad, row_id_map, num_rows, topK, num_cols);
      } else if (topK <= 16) {
        moe_permute_kernel<T, TCompute, 16, true><<<blocks, threads, smem_bytes, stream>>>(
            input, input_fwd, output, prob, prob_grad, row_id_map, num_rows, topK, num_cols);
      } else if (topK <= 32) {
        moe_permute_kernel<T, TCompute, 32, true><<<blocks, threads, smem_bytes, stream>>>(
            input, input_fwd, output, prob, prob_grad, row_id_map, num_rows, topK, num_cols);
      } else if (topK <= 64) {
        moe_permute_kernel<T, TCompute, 64, true><<<blocks, threads, smem_bytes, stream>>>(
            input, input_fwd, output, prob, prob_grad, row_id_map, num_rows, topK, num_cols);
      } else if (topK <= 128) {
        moe_permute_kernel<T, TCompute, 128, true><<<blocks, threads, smem_bytes, stream>>>(
            input, input_fwd, output, prob, prob_grad, row_id_map, num_rows, topK, num_cols);
      } else {
        NVTE_ERROR("topK cannot exceed 128.");
      }
      NVTE_CHECK_CUDA(cudaGetLastError());
    }
  }
}

template <typename T>
void nvte_unpermute_launcher(const T *input, T *output, int *row_id_map, const float *prob,
                             const int num_rows, const int topK, const int num_cols,
                             cudaStream_t stream) {
  using TCompute = typename std::conditional<(std::is_same<T, __nv_fp8_e5m2>::value ||
                                              std::is_same<T, __nv_fp8_e4m3>::value),
                                             half, T>::type;

  static constexpr int kElementsPerAccess = 16 / sizeof(T);

  int blocks = num_rows;
  int threads = std::min(num_cols / kElementsPerAccess, 1024);
  size_t smem_bytes = topK * sizeof(TCompute);

  if (prob == nullptr) {
    // moe_permute_bwd
    // moe_unpermute_fwd without probs

    moe_unpermute_kernel<T, TCompute, false><<<blocks, threads, smem_bytes, stream>>>(
        input, output, row_id_map, nullptr, num_rows, topK, num_cols);
    NVTE_CHECK_CUDA(cudaGetLastError());
  } else {
    // moe_unpermute_fwd with probs

    moe_unpermute_kernel<T, TCompute, true><<<blocks, threads, smem_bytes, stream>>>(
        input, output, row_id_map, prob, num_rows, topK, num_cols);
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
}

void nvte_permute(const NVTETensor input, NVTETensor output, const NVTETensor sorted_row_id,
                  NVTETensor row_id_map, const NVTETensor prob, NVTETensor prob_grad,
                  const NVTETensor input_fwd, const int num_rows, const int topK,
                  const int num_cols, const int num_out_tokens, cudaStream_t stream) {
  using namespace transformer_engine;
  NVTE_API_CALL(nvte_permute);

  const Tensor *input_cu = convertNVTETensorCheck(input);
  const Tensor *output_cu = convertNVTETensorCheck(output);
  const Tensor *sorted_row_id_cu = convertNVTETensorCheck(sorted_row_id);
  const Tensor *row_id_map_cu = convertNVTETensorCheck(row_id_map);
  const Tensor *prob_cu = convertNVTETensorCheck(prob);
  const Tensor *prob_grad_cu = convertNVTETensorCheck(prob_grad);
  const Tensor *input_fwd_cu = convertNVTETensorCheck(input_fwd);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input_cu->data.dtype, T,
      nvte_permute_launcher(reinterpret_cast<const T *>(input_cu->data.dptr),
                            reinterpret_cast<T *>(output_cu->data.dptr),
                            reinterpret_cast<const int *>(sorted_row_id_cu->data.dptr),
                            reinterpret_cast<int *>(row_id_map_cu->data.dptr),
                            reinterpret_cast<const float *>(prob_cu->data.dptr),
                            reinterpret_cast<float *>(prob_grad_cu->data.dptr),
                            reinterpret_cast<const T *>(input_fwd_cu->data.dptr), num_rows, topK,
                            num_cols, num_out_tokens, stream););
}

void nvte_unpermute(const NVTETensor input, NVTETensor output, NVTETensor row_id_map,
                    const NVTETensor prob, const int num_rows, const int topK, const int num_cols,
                    cudaStream_t stream) {
  using namespace transformer_engine;
  NVTE_API_CALL(nvte_unpermute);

  const Tensor *input_cu = convertNVTETensorCheck(input);
  const Tensor *output_cu = convertNVTETensorCheck(output);
  const Tensor *row_id_map_cu = convertNVTETensorCheck(row_id_map);
  const Tensor *prob_cu = convertNVTETensorCheck(prob);

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
      input_cu->data.dtype, T,
      nvte_unpermute_launcher(reinterpret_cast<const T *>(input_cu->data.dptr),
                              reinterpret_cast<T *>(output_cu->data.dptr),
                              reinterpret_cast<int *>(row_id_map_cu->data.dptr),
                              reinterpret_cast<const float *>(prob_cu->data.dptr), num_rows, topK,
                              num_cols, stream););
}

void nvte_device_radix_sort_pairs(void *temp_storage, size_t *temp_storage_bytes, int *keys_in,
                                  int *keys_out, int *values_in, int *values_out,
                                  size_t num_items) {
  NVTE_API_CALL(nvte_device_radix_sort_pairs);
  cub::DeviceRadixSort::SortPairs(temp_storage, *temp_storage_bytes, keys_in, keys_out, values_in,
                                  values_out, num_items);
}

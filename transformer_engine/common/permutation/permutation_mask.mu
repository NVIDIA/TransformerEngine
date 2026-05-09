#include <transformer_engine/permutation.h>

#include <cstdio>

#include "../common.h"
#include "../util/mtfp8_utils.muh"
#include "../utils.cuh"

namespace {

using  v2i32_t = int32_t __attribute__((vector_size(8)));

}

// input: [num_tokens, hidden_size] @ [stride_input_token, stride_input_hidden]
// row_id_map: [num_experts, num_tokens]
// output: [num_out_tokens, hidden_size] @ [stride_output_token, stride_output_hidden]
// probs: [num_tokens, num_experts]
// permuted_probs: [num_out_tokens]
template <typename Dtype, typename P_Dtype, typename IdxDtype, bool with_permuted_probs = false>
__global__ void permute_with_mask_map_trans(
    MUtensorDescriptor out_dev_tensorDesc, MUtensorDescriptor in_dev_tensorDesc,
    IdxDtype *row_id_map_ptr, const P_Dtype *probs_ptr, P_Dtype *permuted_probs_ptr,
    const int num_tokens, const int num_experts, const int hidden_size,
    const int stride_input_token, const int stride_input_hidden, const int stride_output_token,
    const int stride_output_hidden, const int stride_probs_token, const int stride_probs_expert,
    const int stride_permuted_probs_token, const v2i32_t ld_st_token_dim) {
  using IdxVec = transformer_engine::Vec<IdxDtype, 4>;
  int tidx = threadIdx.x;
  int token_id = blockIdx.x;

  int trans_count = hidden_size * sizeof(Dtype);
  extern __shared__ __align__(128) char shared_array[];
  Dtype *smem = reinterpret_cast<Dtype *>(shared_array);
  __musa::async_barrier bar(1);
  bar.init_arrival(1);
  __syncthreads();

  v2i32_t ld_pos{0, static_cast<int>(token_id)};
  __musa::memcpy_async(bar, smem, &in_dev_tensorDesc, ld_st_token_dim, ld_pos, trans_count, 0, 3, 1);

  unsigned phase_id = bar.arrive();
  bar.wait(phase_id);

  IdxDtype dst_row;
  for (int expert_id = 0; expert_id < num_experts; expert_id += 1) {
    dst_row = row_id_map_ptr[expert_id * num_tokens + token_id];
    if (dst_row != -1) {
      v2i32_t st_pos{0, static_cast<int>(dst_row)};
      __musa::memcpy(smem, &out_dev_tensorDesc, ld_st_token_dim, st_pos);
      if constexpr (with_permuted_probs) {
        if (tidx == 0) {
          int prob_offset = token_id * stride_probs_token + expert_id * stride_probs_expert;
          P_Dtype prob_val = probs_ptr[prob_offset];
          int permuted_prob_offset = dst_row * stride_permuted_probs_token;
          permuted_probs_ptr[permuted_prob_offset] = prob_val;
        }
      }
    }
  }
  __musa::memcpy_idf_l2();
}

template <typename Dtype, typename P_Dtype, typename IdxDtype, bool with_permuted_probs = false>
__global__ void permute_with_mask_map(MUtensorDescriptor out_dev_tensorDesc,
                                      MUtensorDescriptor in_dev_tensorDesc,
                                      MUtensorDescriptor map_dev_tensorDesc, const P_Dtype *probs_ptr,
                                      P_Dtype *permuted_probs_ptr, const int num_tokens,
                                      const int num_experts, const int hidden_size,
                                      const int stride_input_token, const int stride_input_hidden,
                                      const int stride_output_token, const int stride_output_hidden,
                                      const int stride_probs_token, const int stride_probs_expert,
                                      const int stride_permuted_probs_token, const v2i32_t ld_st_token_dim) {
  using IdxVec = transformer_engine::Vec<IdxDtype, 4>;
  int tidx = threadIdx.x;
  int token_id = blockIdx.x;

  int trans_count = hidden_size * sizeof(Dtype);
  int trans_count_map = num_experts * sizeof(IdxDtype);
  extern __shared__ __align__(128) char shared_array[];
  Dtype *smem = reinterpret_cast<Dtype *>(shared_array);
  const size_t hidden_size_aligned =
      (hidden_size * sizeof(Dtype) + 127) / 128 * 128 / sizeof(Dtype);
  IdxDtype *smem_map = reinterpret_cast<IdxDtype *>(smem + hidden_size_aligned);
  __musa::async_barrier bar(1);
  __musa::async_barrier bar_map(2);
  bar.init_arrival(1);
  bar_map.init_arrival(1);
  __syncthreads();

  int ld_dim_map = num_experts;
  int ld_pos_map = token_id * num_experts;
  __musa::memcpy_async(bar_map, smem_map, &map_dev_tensorDesc, ld_dim_map, ld_pos_map,
                       trans_count_map, 0, 3, 1);
  unsigned phase_id_map = bar_map.arrive();

  v2i32_t ld_pos{0, static_cast<int>(token_id)};
  __musa::memcpy_async(bar, smem, &in_dev_tensorDesc, ld_st_token_dim, ld_pos, trans_count, 0, 3, 1);
  unsigned phase_id = bar.arrive();

  bar_map.wait(phase_id_map);
  bar.wait(phase_id);

  IdxVec dst_row_vec;
  for (int expert_id = 0; expert_id < num_experts; expert_id += 4) {
    dst_row_vec.load_from(smem_map + expert_id);
#pragma unroll
    for (int i = 0; i < 4; i++) {
      if (dst_row_vec.data.elt[i] != -1) {

        v2i32_t st_pos{0, static_cast<int>(dst_row_vec.data.elt[i])};
        __musa::memcpy(smem, &out_dev_tensorDesc, ld_st_token_dim, st_pos);

        if constexpr (with_permuted_probs) {
          if (tidx == 0) {
            int prob_offset = token_id * stride_probs_token + (expert_id + i) * stride_probs_expert;
            P_Dtype prob_val = probs_ptr[prob_offset];
            int permuted_prob_offset = dst_row_vec.data.elt[i] * stride_permuted_probs_token;
            permuted_probs_ptr[permuted_prob_offset] = prob_val;
          }
        }
      }
    }
  }
  __musa::memcpy_idf_l2();
}


// input: [num_out_tokens, hidden_size]
// row_id_map: [num_experts, num_tokens]
// output: [num_tokens, hidden_size]
// merging_probs: [num_tokens, num_experts]
// permuted_probs: [num_out_tokens]
// unpermuted_probs: [num_tokens, num_experts]
template <typename Dtype, typename P_Dtype, typename IdxDtype, bool with_merging_probs, bool with_permuted_probs,
          bool trans_row_id_map, int vlen>
__global__ void moe_unpermute_mask(
    const Dtype *in_ptr, Dtype *out_ptr, IdxDtype *row_id_map_ptr, const P_Dtype *merging_probs_ptr,
    const P_Dtype *permuted_probs_ptr, P_Dtype *unpermuted_probs_ptr, const int num_tokens,
    const int num_experts, const int hidden_size, const int stride_input_token,
    const int stride_input_hidden, const int stride_output_token, const int stride_output_hidden,
    const int stride_merging_probs_token, const int stride_merging_probs_expert,
    const int stride_permuted_probs_token, const int stride_unpermuted_probs_token,
    const int stride_unpermuted_probs_expert) {
  using DtypeVec = transformer_engine::Vec<Dtype, vlen>;
  using ComputeVec = transformer_engine::Vec<float, vlen>;
  constexpr int idx_vlen = 4;
  using IdxVec = transformer_engine::Vec<IdxDtype, idx_vlen>;
  int token_id = blockIdx.y;
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidx_vlen = (blockIdx.x * blockDim.x + threadIdx.x) * vlen;
  extern __shared__ IdxDtype smem[];

  if constexpr (trans_row_id_map) {
    for (int expert_id = threadIdx.x; expert_id < num_experts; expert_id += blockDim.x) {
      memcpy_global2shared(smem + expert_id, row_id_map_ptr + expert_id * num_tokens + token_id, 1);
    }
  } else {
    for (int expert_id = threadIdx.x * idx_vlen; expert_id < num_experts;
         expert_id += blockDim.x * idx_vlen) {
      memcpy_global2shared(smem + expert_id, row_id_map_ptr + token_id * num_experts + expert_id,
                           idx_vlen);
    }
  }
  __syncthreads_lm();

  ComputeVec acc_vec = 0.0f;
  if (tidx_vlen < hidden_size) {
    IdxVec src_row_vec;
    for (int expert_id = 0; expert_id < num_experts; expert_id += idx_vlen) {
      src_row_vec.load_from(smem + expert_id);
#pragma unroll
      for (int i = 0; i < idx_vlen; i++) {
        int unpermuted_offset = token_id * stride_unpermuted_probs_token +
                                (expert_id + i) * stride_unpermuted_probs_expert;
        if (src_row_vec.data.elt[i] != -1) {
          int src_offset =
              src_row_vec.data.elt[i] * stride_input_token + tidx_vlen * stride_input_hidden;
          DtypeVec src_val_vec = (Dtype)(0.0f);
          src_val_vec.load_from(in_ptr + src_offset);

          P_Dtype merging_probs_val = (P_Dtype)(1.0f);
          if constexpr (with_merging_probs) {
            int merging_probs_offset = token_id * stride_merging_probs_token +
                                       (expert_id + i) * stride_merging_probs_expert;
            merging_probs_val = merging_probs_ptr[merging_probs_offset];
          }

#pragma unroll
          for (int j = 0; j < vlen; j++) {
            acc_vec.data.elt[j] += (float)src_val_vec.data.elt[j] * (float)merging_probs_val;
          }

          if constexpr (with_permuted_probs) {
            if (tidx == 0) {
              unpermuted_probs_ptr[unpermuted_offset] =
                  permuted_probs_ptr[src_row_vec.data.elt[i] * stride_permuted_probs_token];
            }
          }
        } else {
          if constexpr (with_permuted_probs) {
            if (tidx == 0) {
              unpermuted_probs_ptr[unpermuted_offset] = (P_Dtype)(0.0f);
            }
          }
          continue;
        }
      }
    }
    int dst_offset = token_id * stride_output_token + tidx_vlen * stride_output_hidden;
#pragma unroll
    for (int i = 0; i < vlen; i++) {
      out_ptr[dst_offset + i] = (Dtype)acc_vec.data.elt[i];
    }
  }
}

// fwd_input_grad,      [num_out_tokens, hidden_size]
// merging_probs_grad,  [num_tokens, num_experts]
// fwd_output_grad,     [num_tokens, hidden_size]
// fwd_input,           [num_out_tokens, hidden_size]
// merging_probs,       [num_tokens, num_experts]
// row_id_map,          [num_experts, num_tokens]
template <typename Dtype, typename IdxDtype, bool trans_row_id_map, int vlen>
__global__ void moe_unpermute_mask_bwd_with_merging_probs(
    const Dtype *fwd_output_grad_ptr, Dtype *fwd_input_grad_ptr, const Dtype *fwd_input_ptr,
    const Dtype *merging_probs_ptr, Dtype *merging_probs_grad_ptr, IdxDtype *row_id_map_ptr,
    const int num_tokens, const int num_experts, const int hidden_size,
    const int stride_fwd_output_grad_token, const int stride_fwd_output_grad_hidden,
    const int stride_fwd_input_grad_token, const int stride_fwd_input_grad_hidden,
    const int stride_fwd_input_token, const int stride_fwd_input_hidden,
    const int stride_merging_probs_token, const int stride_merging_probs_expert,
    const int stride_merging_probs_grad_token, const int stride_merging_probs_grad_expert) {
  constexpr int idx_vlen = 4;
  using ComputeDtype = float;
  using DtypeVec = transformer_engine::Vec<Dtype, vlen>;
  using IdxVec = transformer_engine::Vec<IdxDtype, idx_vlen>;
  int token_id = blockIdx.x;
  int tidx = threadIdx.x;
  int tidx_vlen = (threadIdx.x) * vlen;
  extern __shared__ IdxDtype smem[];
  int warp_id = tidx >> 5;
  int lane_id = tidx & 31;
  ComputeDtype *warpLevelVal = reinterpret_cast<ComputeDtype *>(smem + num_experts);

  if constexpr (trans_row_id_map) {
    for (int expert_id = threadIdx.x; expert_id < num_experts; expert_id += blockDim.x) {
      memcpy_global2shared(smem + expert_id, row_id_map_ptr + expert_id * num_tokens + token_id, 1);
    }
  } else {
    for (int expert_id = threadIdx.x * idx_vlen; expert_id < num_experts;
         expert_id += blockDim.x * idx_vlen) {
      memcpy_global2shared(smem + expert_id, row_id_map_ptr + token_id * num_experts + expert_id,
                           idx_vlen);
    }
  }
  __syncthreads_lm();

  IdxVec dst_row_vec;
  for (int expert_id = 0; expert_id < num_experts; expert_id += idx_vlen) {
    dst_row_vec.load_from(smem + expert_id);
#pragma unroll
    for (int i = 0; i < idx_vlen; i++) {
      int probs_grad_offset = token_id * stride_merging_probs_grad_token +
                              (expert_id + i) * stride_merging_probs_grad_expert;
      if (dst_row_vec.data.elt[i] != -1) {
        ComputeDtype prob_grad_acc = 0.0f;
        for (int hidden_offset = tidx_vlen; hidden_offset < hidden_size;
             hidden_offset += blockDim.x * vlen) {
          int input_offset = token_id * stride_fwd_output_grad_token +
                             hidden_offset * stride_fwd_output_grad_hidden;
          DtypeVec src_val_vec = (Dtype)(0.0f);
          src_val_vec.load_from(fwd_output_grad_ptr + input_offset);

          int merging_prob_offset =
              token_id * stride_merging_probs_token + (expert_id + i) * stride_merging_probs_expert;
          float merging_prob = (float)(merging_probs_ptr[merging_prob_offset]);

          DtypeVec dst_val_vec = (Dtype)(0.0f);
          int output_offset = dst_row_vec.data.elt[i] * stride_fwd_input_grad_token +
                              hidden_offset * stride_fwd_input_grad_hidden;
#pragma unroll
          for (int j = 0; j < vlen; j++) {
            dst_val_vec.data.elt[j] = (float)src_val_vec.data.elt[j] * merging_prob;
          }
          dst_val_vec.store_to(fwd_input_grad_ptr + output_offset);

          int fwd_input_offset = dst_row_vec.data.elt[i] * stride_fwd_input_token +
                                 hidden_offset * stride_fwd_input_hidden;
          DtypeVec fwd_input_vec = (Dtype)(0.0f);
          fwd_input_vec.load_from(fwd_input_ptr + fwd_input_offset);
#pragma unroll
          for (int j = 0; j < vlen; j++) {
            prob_grad_acc =
                prob_grad_acc + (float)fwd_input_vec.data.elt[j] * (float)src_val_vec.data.elt[j];
          }
        }
        ComputeDtype sum = prob_grad_acc;
        for (int delta = 16; delta > 0; delta >>= 1) {
          sum += __shfl_down_sync(0xffffffff, sum, delta);
        }
        if (lane_id == 0) {
          warpLevelVal[warp_id] = sum;
        }
        __syncthreads_lm();
        if (warp_id == 0) {
          sum = (lane_id < 4) ? warpLevelVal[lane_id] : 0;
          for (int delta = 2; delta > 0; delta >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, delta);
          }
          if (tidx == 0) {
            merging_probs_grad_ptr[probs_grad_offset] = (Dtype)sum;
          }
        }
      } else {
        merging_probs_grad_ptr[probs_grad_offset] = (Dtype)(0.0f);
      }
    }
  }
}

template <typename Dtype, typename P_Dtype, typename IdxDtype, bool with_permuted_probs = false,
          bool trans_row_id_map = true>
void nvte_permute_mask_launcher(const Dtype *input, Dtype *output, IdxDtype *row_id_map,
                                const P_Dtype *probs, P_Dtype *permuted_probs, const int num_tokens,
                                const int num_experts, const int num_out_tokens,
                                const int hidden_size, musaStream_t stream) {
  NVTE_CHECK((hidden_size * sizeof(Dtype)) % 4 == 0, "bytes of hidden_size must be divisible by 4");
  if constexpr (!trans_row_id_map) {
    NVTE_CHECK((num_experts * sizeof(IdxDtype)) % 4 == 0,
               "bytes of num_experts must be divisible by 4");
  }
  constexpr int data_size = sizeof(Dtype);

  MUtensorDescriptor intensorDesc;
  MUtensorDescriptor outtensorDesc;
  MUtensorDescriptor maptensorDesc;
  MUtensorDescriptorDataType tensorDataType = MU_TENSOR_DESCRIPTOR_DATA_TYPE_BFLOAT16;
  MUtensorDescriptorDataType mapDataType = MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT64;
  MUtensorDescriptorInterleave interleave = MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE;
  uint64_t oobConstantFill = 0;

  uint32_t tensorRank = 1;
  const uint64_t map_globalDim[5] = {static_cast<uint64_t>(num_experts) * num_tokens, 1, 1, 1, 1};
  const uint64_t map_globalStrides[4] = {0, 0, 0, 0};
  NVTE_CHECK_MU(muTensorDescriptorEncode(&maptensorDesc, mapDataType, tensorRank,
                                         (void *)row_id_map, map_globalDim, map_globalStrides,
                                         interleave, oobConstantFill));

  uint32_t tensorRankIn = 2;
  const uint64_t in_globalDim[5] = {static_cast<uint64_t>(hidden_size), static_cast<uint64_t>(num_tokens), 1, 1, 1};
  const uint64_t in_globalStrides[4] = {static_cast<uint64_t>(hidden_size)*data_size, 0, 0, 0};
  NVTE_CHECK_MU(muTensorDescriptorEncode(&intensorDesc, tensorDataType, tensorRankIn, (void *)input,
                                         in_globalDim, in_globalStrides, interleave,
                                         oobConstantFill));

  uint32_t tensorRankOut = 2;
  const uint64_t out_globalDim[5] = {static_cast<uint64_t>(hidden_size), static_cast<uint64_t>(num_out_tokens), 1, 1, 1};
  const uint64_t out_globalStrides[4] = {static_cast<uint64_t>(hidden_size)*data_size, 0, 0, 0};
  NVTE_CHECK_MU(muTensorDescriptorEncode(&outtensorDesc, tensorDataType, tensorRankOut, (void *)output,
                                         out_globalDim, out_globalStrides, interleave,
                                         oobConstantFill));

  const v2i32_t ld_st_token_dim{hidden_size, 1};

  const int block_x = 32;
  const int grid_x = num_tokens;
  dim3 block(block_x, 1);
  dim3 grid(grid_x, 1);

  if constexpr (trans_row_id_map) {
    int smem_size = hidden_size * sizeof(Dtype);
    permute_with_mask_map_trans<Dtype, P_Dtype, IdxDtype, with_permuted_probs><<<grid, block, smem_size, stream>>>(
        outtensorDesc, intensorDesc, row_id_map, probs, permuted_probs, num_tokens, num_experts,
        hidden_size, hidden_size, 1, hidden_size, 1, num_experts, 1, 1, ld_st_token_dim);
  } else {
    int smem_size = hidden_size * sizeof(Dtype) + num_experts * sizeof(IdxDtype);
    permute_with_mask_map<Dtype, P_Dtype, IdxDtype, with_permuted_probs><<<grid, block, smem_size, stream>>>(
        outtensorDesc, intensorDesc, maptensorDesc, probs, permuted_probs, num_tokens, num_experts,
        hidden_size, hidden_size, 1, hidden_size, 1, num_experts, 1, 1, ld_st_token_dim);
  }
}

template <typename Dtype, typename P_Dtype, typename IdxDtype, bool with_merging_probs = false,
          bool with_permuted_probs = false, bool trans_row_id_map = true>
void nvte_unpermute_mask_launcher(const Dtype *input, Dtype *output, IdxDtype *row_id_map,
                                  const P_Dtype *merging_probs, const P_Dtype *permuted_probs,
                                  P_Dtype *unpermuted_probs, const int num_tokens,
                                  const int num_experts, const int hidden_size,
                                  musaStream_t stream) {
  constexpr int vlen = 16 / sizeof(Dtype);
  int block_x = 128;
  int grid_x = transformer_engine::mtfp8::ceil_div(hidden_size, block_x * vlen);
  int grid_y = num_tokens;
  dim3 block(block_x, 1);
  dim3 grid(grid_x, grid_y);
  int smem_size = num_experts * sizeof(IdxDtype);

  moe_unpermute_mask<Dtype, P_Dtype, IdxDtype, with_merging_probs, with_permuted_probs, trans_row_id_map,
                     vlen><<<grid, block, smem_size, stream>>>(
      input, output, row_id_map, merging_probs, permuted_probs, unpermuted_probs, num_tokens,
      num_experts, hidden_size, hidden_size, 1, hidden_size, 1, num_experts, 1, 1, num_experts, 1);
}

template <typename Dtype, typename IdxDtype, bool trans_row_id_map = true>
void nvte_unpermute_mask_bwd_with_merging_probs_launcher(
    const Dtype *fwd_output_grad, Dtype *fwd_input_grad, const Dtype *fwd_input,
    const Dtype *merging_probs, Dtype *merging_probs_grad, IdxDtype *row_id_map,
    const int num_tokens, const int num_experts, const int hidden_size, musaStream_t stream) {
  NVTE_CHECK(num_experts % 4 == 0, "num_experts must be divisible by 4");

  constexpr int vlen = 16 / sizeof(Dtype);
  int block_x = 128;
  int grid_x = num_tokens;
  dim3 block(block_x, 1, 1);
  dim3 grid(grid_x, 1, 1);

  int smem_size = num_experts * sizeof(IdxDtype) + block_x * sizeof(float);
  moe_unpermute_mask_bwd_with_merging_probs<Dtype, IdxDtype, trans_row_id_map, vlen>
      <<<grid, block, smem_size, stream>>>(
          fwd_output_grad, fwd_input_grad, fwd_input, merging_probs, merging_probs_grad, row_id_map,
          num_tokens, num_experts, hidden_size, hidden_size, 1, hidden_size, 1, hidden_size, 1,
          num_experts, 1, num_experts, 1);
}

#define CALL_PERMUTE_MASK_LAUNCHER(_PERMUTED_PROBS, _TRANS_ROW_ID_MAP)                  \
  TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(                                                 \
      input_cu->data.dtype, T,                                                          \
      nvte_permute_mask_launcher<T, T, int64_t, _PERMUTED_PROBS, _TRANS_ROW_ID_MAP>(       \
          reinterpret_cast<const T *>(input_cu->data.dptr),                             \
          reinterpret_cast<T *>(output_cu->data.dptr),                                  \
          reinterpret_cast<int64_t *>(row_id_map_cu->data.dptr),                        \
          reinterpret_cast<const T *>(probs_cu->data.dptr),                             \
          reinterpret_cast<T *>(permuted_probs_cu->data.dptr), num_tokens, num_experts, \
          num_out_tokens, hidden_size, stream););

void nvte_permute_mask(const NVTETensor input, NVTETensor output, NVTETensor row_id_map,
                       const NVTETensor probs, NVTETensor permuted_probs, const int num_tokens,
                       const int num_experts, const int num_out_tokens, const int hidden_size,
                       musaStream_t stream) {
  NVTE_API_CALL(nvte_permute_mask);

  const transformer_engine::Tensor *input_cu =
      transformer_engine::convertNVTETensorCheck(input);
  const transformer_engine::Tensor *output_cu =
      transformer_engine::convertNVTETensorCheck(output);
  const transformer_engine::Tensor *row_id_map_cu =
      transformer_engine::convertNVTETensorCheck(row_id_map);
  const transformer_engine::Tensor *probs_cu =
      transformer_engine::convertNVTETensorCheck(probs);
  const transformer_engine::Tensor *permuted_probs_cu =
      transformer_engine::convertNVTETensorCheck(permuted_probs);

  if (probs_cu->data.dptr != nullptr) {
    if (row_id_map_cu->data.shape[0] == num_experts) {
      CALL_PERMUTE_MASK_LAUNCHER(true, true);
    } else {
      CALL_PERMUTE_MASK_LAUNCHER(true, false);
    }
  } else {
    if (row_id_map_cu->data.shape[0] == num_experts) {
      CALL_PERMUTE_MASK_LAUNCHER(false, true);
    } else {
      CALL_PERMUTE_MASK_LAUNCHER(false, false);
    }
  }
}
#undef CALL_PERMUTE_MASK_LAUNCHER

#define CALL_UNPERMUTE_MASK_LAUNCHER(_MERGING_PROBS, _PERMUTED_PROBS, _TRANS_ROW_ID_MAP)  \
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(                                                     \
      input_cu->data.dtype, T,                                                            \
      nvte_unpermute_mask_launcher<T, T, int64_t, _MERGING_PROBS, _PERMUTED_PROBS,           \
                                   _TRANS_ROW_ID_MAP>(                                    \
          reinterpret_cast<const T *>(input_cu->data.dptr),                               \
          reinterpret_cast<T *>(output_cu->data.dptr),                                    \
          reinterpret_cast<int64_t *>(row_id_map_cu->data.dptr),                          \
          reinterpret_cast<const T *>(merging_probs_cu->data.dptr),                       \
          reinterpret_cast<const T *>(permuted_probs_cu->data.dptr),                      \
          reinterpret_cast<T *>(unpermuted_probs_cu->data.dptr), num_tokens, num_experts, \
          hidden_size, stream););

#define CALL_UNPERMUTE_MASK_TRANS_LAUNCHER(_TRANS_ROW_ID_MAP)        \
  if (merging_probs_cu->data.dptr != nullptr) {                      \
    if (permuted_probs_cu->data.dptr != nullptr) {                   \
      CALL_UNPERMUTE_MASK_LAUNCHER(true, true, _TRANS_ROW_ID_MAP);   \
    } else {                                                         \
      CALL_UNPERMUTE_MASK_LAUNCHER(true, false, _TRANS_ROW_ID_MAP);  \
    }                                                                \
  } else {                                                           \
    if (permuted_probs_cu->data.dptr != nullptr) {                   \
      CALL_UNPERMUTE_MASK_LAUNCHER(false, true, _TRANS_ROW_ID_MAP);  \
    } else {                                                         \
      CALL_UNPERMUTE_MASK_LAUNCHER(false, false, _TRANS_ROW_ID_MAP); \
    }                                                                \
  }

void nvte_unpermute_mask(const NVTETensor input, NVTETensor output, NVTETensor row_id_map,
                         const NVTETensor merging_probs, const NVTETensor permuted_probs,
                         NVTETensor unpermuted_probs, const int num_tokens, const int num_experts,
                         const int hidden_size, musaStream_t stream) {
  NVTE_API_CALL(nvte_unpermute_mask);

  const transformer_engine::Tensor *input_cu =
      transformer_engine::convertNVTETensorCheck(input);
  const transformer_engine::Tensor *output_cu =
      transformer_engine::convertNVTETensorCheck(output);
  const transformer_engine::Tensor *row_id_map_cu =
      transformer_engine::convertNVTETensorCheck(row_id_map);
  const transformer_engine::Tensor *merging_probs_cu =
      transformer_engine::convertNVTETensorCheck(merging_probs);
  const transformer_engine::Tensor *permuted_probs_cu =
      transformer_engine::convertNVTETensorCheck(permuted_probs);
  const transformer_engine::Tensor *unpermuted_probs_cu =
      transformer_engine::convertNVTETensorCheck(unpermuted_probs);

  if (row_id_map_cu->data.shape[0] == num_experts) {
    CALL_UNPERMUTE_MASK_TRANS_LAUNCHER(true);
  } else {
    CALL_UNPERMUTE_MASK_TRANS_LAUNCHER(false);
  }
}

#undef CALL_UNPERMUTE_MASK_LAUNCHER
#undef CALL_UNPERMUTE_MASK_TRANS_LAUNCHER

#define PROBS_TYPE_SWITCH(probs_dtype, probs_type, ...)        \
  switch (probs_dtype) {                                       \
    using namespace transformer_engine;                        \
    case DType::kFloat16: {                                    \
      using probs_type = fp16;                                 \
      __VA_ARGS__;                                             \
      break;                                                   \
    }                                                          \
    case DType::kBFloat16: {                                   \
      using probs_type = bf16;                                 \
      __VA_ARGS__;                                             \
      break;                                                   \
    }                                                          \
    case DType::kFloat32: {                                    \
      using probs_type = fp32;                                 \
      __VA_ARGS__;                                             \
      break;                                                   \
    }                                                          \
    default:                                                   \
      NVTE_ERROR("Invalid probs type.");                       \
  }
#define TRANSFORMER_ENGINE_PROBS_PERMUTE_TYPE_SWITCH(dtype, type, probs_dtype, probs_type,...) \
  switch (dtype) {                                             \
    using namespace transformer_engine;                        \
    case DType::kFloat16: {                                    \
      using type = fp16;                                       \
      PROBS_TYPE_SWITCH(probs_dtype, probs_type, __VA_ARGS__); \
      break;                                                   \
    }                                                          \
    case DType::kBFloat16: {                                   \
      using type = bf16;                                       \
      PROBS_TYPE_SWITCH(probs_dtype, probs_type, __VA_ARGS__); \
      break;                                                   \
    }                                                          \
    default:                                                   \
      NVTE_ERROR("Invalid type for 16 bit.");                  \
  }

#define CALL_HIGH_PRECISION_PROBS_PERMUTE_MASK_LAUNCHER(_PERMUTED_PROBS, _TRANS_ROW_ID_MAP)                  \
  TRANSFORMER_ENGINE_PROBS_PERMUTE_TYPE_SWITCH(                                                 \
      input_cu->data.dtype, T, probs_cu->data.dtype, T_P,                                                         \
      nvte_permute_mask_launcher<T, T_P, int64_t, _PERMUTED_PROBS, _TRANS_ROW_ID_MAP>(       \
          reinterpret_cast<const T *>(input_cu->data.dptr),                             \
          reinterpret_cast<T *>(output_cu->data.dptr),                                  \
          reinterpret_cast<int64_t *>(row_id_map_cu->data.dptr),                        \
          reinterpret_cast<const T_P *>(probs_cu->data.dptr),                             \
          reinterpret_cast<T_P *>(permuted_probs_cu->data.dptr), num_tokens, num_experts, \
          num_out_tokens, hidden_size, stream););

void nvte_permute_mask_high_precision_probs(const NVTETensor input, NVTETensor output, NVTETensor row_id_map,
                       const NVTETensor probs, NVTETensor permuted_probs, const int num_tokens,
                       const int num_experts, const int num_out_tokens, const int hidden_size,
                       musaStream_t stream) {
  NVTE_API_CALL(nvte_permute_mask_high_precision_probs);

  const transformer_engine::Tensor *input_cu =
      transformer_engine::convertNVTETensorCheck(input);
  const transformer_engine::Tensor *output_cu =
      transformer_engine::convertNVTETensorCheck(output);
  const transformer_engine::Tensor *row_id_map_cu =
      transformer_engine::convertNVTETensorCheck(row_id_map);
  const transformer_engine::Tensor *probs_cu =
      transformer_engine::convertNVTETensorCheck(probs);
  const transformer_engine::Tensor *permuted_probs_cu =
      transformer_engine::convertNVTETensorCheck(permuted_probs);

  if (probs_cu->data.dptr != nullptr) {
    if (row_id_map_cu->data.shape[0] == num_experts) {
      CALL_HIGH_PRECISION_PROBS_PERMUTE_MASK_LAUNCHER(true, true);
    } else {
      CALL_HIGH_PRECISION_PROBS_PERMUTE_MASK_LAUNCHER(true, false);
    }
  } else {
    if (row_id_map_cu->data.shape[0] == num_experts) {
      CALL_HIGH_PRECISION_PROBS_PERMUTE_MASK_LAUNCHER(false, true);
    } else {
      CALL_HIGH_PRECISION_PROBS_PERMUTE_MASK_LAUNCHER(false, false);
    }
  }
}
#undef CALL_HIGH_PRECISION_PROBS_PERMUTE_MASK_LAUNCHER

#define CALL_HIGH_PRECISION_PROBS_UNPERMUTE_MASK_LAUNCHER(_MERGING_PROBS, _PERMUTED_PROBS, _TRANS_ROW_ID_MAP)  \
  TRANSFORMER_ENGINE_PROBS_PERMUTE_TYPE_SWITCH(                                                     \
      input_cu->data.dtype, T,permuted_probs_cu->data.dtype, T_P,                                                            \
      nvte_unpermute_mask_launcher<T, T_P, int64_t, _MERGING_PROBS, _PERMUTED_PROBS,           \
                                   _TRANS_ROW_ID_MAP>(                                    \
          reinterpret_cast<const T *>(input_cu->data.dptr),                               \
          reinterpret_cast<T *>(output_cu->data.dptr),                                    \
          reinterpret_cast<int64_t *>(row_id_map_cu->data.dptr),                          \
          reinterpret_cast<const T_P *>(merging_probs_cu->data.dptr),                       \
          reinterpret_cast<const T_P *>(permuted_probs_cu->data.dptr),                      \
          reinterpret_cast<T_P *>(unpermuted_probs_cu->data.dptr), num_tokens, num_experts, \
          hidden_size, stream););

#define CALL_HIGH_PRECISION_PROBS_UNPERMUTE_MASK_TRANS_LAUNCHER(_TRANS_ROW_ID_MAP)        \
  if (merging_probs_cu->data.dptr != nullptr) {                      \
    if (permuted_probs_cu->data.dptr != nullptr) {                   \
      CALL_HIGH_PRECISION_PROBS_UNPERMUTE_MASK_LAUNCHER(true, true, _TRANS_ROW_ID_MAP);   \
    } else {                                                         \
      CALL_HIGH_PRECISION_PROBS_UNPERMUTE_MASK_LAUNCHER(true, false, _TRANS_ROW_ID_MAP);  \
    }                                                                \
  } else {                                                           \
    if (permuted_probs_cu->data.dptr != nullptr) {                   \
      CALL_HIGH_PRECISION_PROBS_UNPERMUTE_MASK_LAUNCHER(false, true, _TRANS_ROW_ID_MAP);  \
    } else {                                                         \
      CALL_HIGH_PRECISION_PROBS_UNPERMUTE_MASK_LAUNCHER(false, false, _TRANS_ROW_ID_MAP); \
    }                                                                \
  }

void nvte_unpermute_mask_high_precision_probs(const NVTETensor input, NVTETensor output, NVTETensor row_id_map,
                         const NVTETensor merging_probs, const NVTETensor permuted_probs,
                         NVTETensor unpermuted_probs, const int num_tokens, const int num_experts,
                         const int hidden_size, musaStream_t stream) {
  NVTE_API_CALL(nvte_unpermute_mask_high_precision_probs);

  const transformer_engine::Tensor *input_cu =
      transformer_engine::convertNVTETensorCheck(input);
  const transformer_engine::Tensor *output_cu =
      transformer_engine::convertNVTETensorCheck(output);
  const transformer_engine::Tensor *row_id_map_cu =
      transformer_engine::convertNVTETensorCheck(row_id_map);
  const transformer_engine::Tensor *merging_probs_cu =
      transformer_engine::convertNVTETensorCheck(merging_probs);
  const transformer_engine::Tensor *permuted_probs_cu =
      transformer_engine::convertNVTETensorCheck(permuted_probs);
  const transformer_engine::Tensor *unpermuted_probs_cu =
      transformer_engine::convertNVTETensorCheck(unpermuted_probs);

  if (row_id_map_cu->data.shape[0] == num_experts) {
    CALL_HIGH_PRECISION_PROBS_UNPERMUTE_MASK_TRANS_LAUNCHER(true);
  } else {
    CALL_HIGH_PRECISION_PROBS_UNPERMUTE_MASK_TRANS_LAUNCHER(false);
  }
}

#undef CALL_HIGH_PRECISION_PROBS_UNPERMUTE_MASK_LAUNCHER
#undef CALL_HIGH_PRECISION_PROBS_UNPERMUTE_MASK_TRANS_LAUNCHER

#undef PROBS_TYPE_SWITCH
#undef TRANSFORMER_ENGINE_PROBS_PERMUTE_TYPE_SWITCH


#define CALL_UNPERMUTE_MASK_BWD_WITH_MERGING_PROBS_LAUNCHER(_TRANS_ROW_ID_MAP)            \
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(                                             \
      fwd_output_grad_cu->data.dtype, T,                                                  \
      nvte_unpermute_mask_bwd_with_merging_probs_launcher<T, int64_t, _TRANS_ROW_ID_MAP>( \
          reinterpret_cast<const T *>(fwd_output_grad_cu->data.dptr),                     \
          reinterpret_cast<T *>(fwd_input_grad_cu->data.dptr),                            \
          reinterpret_cast<const T *>(fwd_input_cu->data.dptr),                           \
          reinterpret_cast<const T *>(merging_probs_cu->data.dptr),                       \
          reinterpret_cast<T *>(merging_probs_grad_cu->data.dptr),                        \
          reinterpret_cast<int64_t *>(row_id_map_cu->data.dptr), num_tokens, num_experts, \
          hidden_size, stream););

void nvte_unpermute_mask_bwd_with_merging_probs(
    const NVTETensor fwd_output_grad, NVTETensor fwd_input_grad, const NVTETensor fwd_input,
    const NVTETensor merging_probs, NVTETensor merging_probs_grad, NVTETensor row_id_map,
    const int num_tokens, const int num_experts, const int hidden_size, musaStream_t stream) {
  NVTE_API_CALL(nvte_unpermute_mask_bwd_with_merging_probs);

  const transformer_engine::Tensor *fwd_output_grad_cu =
      transformer_engine::convertNVTETensorCheck(fwd_output_grad);
  const transformer_engine::Tensor *fwd_input_grad_cu =
      transformer_engine::convertNVTETensorCheck(fwd_input_grad);
  const transformer_engine::Tensor *fwd_input_cu =
      transformer_engine::convertNVTETensorCheck(fwd_input);
  const transformer_engine::Tensor *merging_probs_cu =
      transformer_engine::convertNVTETensorCheck(merging_probs);
  const transformer_engine::Tensor *merging_probs_grad_cu =
      transformer_engine::convertNVTETensorCheck(merging_probs_grad);
  const transformer_engine::Tensor *row_id_map_cu =
      transformer_engine::convertNVTETensorCheck(row_id_map);

  if (row_id_map_cu->data.shape[0] == num_experts) {
    CALL_UNPERMUTE_MASK_BWD_WITH_MERGING_PROBS_LAUNCHER(true);
  } else {
    CALL_UNPERMUTE_MASK_BWD_WITH_MERGING_PROBS_LAUNCHER(false);
  }
}

#undef CALL_UNPERMUTE_MASK_BWD_WITH_MERGING_PROBS_LAUNCHER
/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <transformer_engine/hadamard_transform.h>
#include <transformer_engine/multi_tensor.h>

#include <cuda/barrier>

#include "common/common.h"
#include "common/util/ptx.cuh"
#include "common/utils.cuh"
#include "hadamard_transform_utils.cuh"

namespace transformer_engine {
namespace {

constexpr int kMaxTensorsPerKernel = 64;
constexpr int kThreadsPerWarp = 32;

enum ShapeRepresentation {
  SAME_BOTH_DIMS = 0,
  VARYING_FIRST_DIM = 1,
  VARYING_LAST_DIM = 2,
  VARYING_BOTH_DIMS = 3
};

__device__ __forceinline__ size_t get_current_tensor_id(
    const ShapeRepresentation shape_rep, const size_t num_tensors, const size_t current_offset,
    const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t* const __restrict__ offsets_ptr) {
  if (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS) {
    const size_t current_row = current_offset / last_logical_dim;
    const size_t rows_per_tensor = first_logical_dim / num_tensors;
    return current_row / rows_per_tensor;
  } else {
    // upper_bound(offsets, current_offset) - 1 in range i in [0..num_tensors)
    size_t low = 0;
    size_t hi = num_tensors;  // half-open [low, hi)

    while (low < hi) {
      const size_t mid = low + (hi - low) / 2;
      const size_t mid_offset = static_cast<size_t>(offsets_ptr[mid]);

      if (mid_offset <= current_offset) {
        low = mid + 1;
      } else {
        hi = mid;
      }
    }

    // low = first index where offsets[low] > current_offset (or low == num_tensors)
    // id = low - 1, but need to evaluate if current_offset < offsets[0]
    return (low == 0) ? 0 : (low - 1);
  }
}

template <typename IType, int kHadamardDimension, int BUFF_DIM_Y, int BUFF_DIM_X,
          bool kReturnPreRhtAmax, bool kReturnIdentityAmax, bool kReturnTransposedAmax>
__device__ __forceinline__ void ComputeKernel(uint32_t b_frag_i[4], uint32_t b_frag_t[4],
                                              IType* in_sh_ptr, uint32_t& local_pre_rht_amax_reg,
                                              uint32_t& local_amax_reg,
                                              uint32_t& local_amax_t_reg) {
  uint32_t a_frag[4];  // A matrix fragment
  uint32_t c_frag[4];  // Result fragment

  int warp_id = threadIdx.x / kThreadsPerWarp;
  int local_rank = (threadIdx.x % kThreadsPerWarp);

  int ld_row_idx = local_rank % kHadamardDimension;
  int ld_col_idx = local_rank / kHadamardDimension + warp_id * 2;
  int swizzle_idx = swizzle_128B_atom_32B(ld_row_idx, ld_col_idx);

  uint32_t temp_amax_reg;
  uint32_t temp_amax_t_reg;

  if (kReturnIdentityAmax) {
    ldmatrix_x4_m8n8_shared_b16<false>(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                                       reinterpret_cast<uint4*>(in_sh_ptr) + swizzle_idx);

    mma_m16_n16_k16_b16_b16_b16_noacc<kReturnIdentityAmax>(
        a_frag[0], a_frag[1], a_frag[2], a_frag[3], b_frag_i[0], b_frag_i[1], b_frag_i[2],
        b_frag_i[3], c_frag[0], c_frag[1], c_frag[2], c_frag[3], temp_amax_reg);
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                 : "=r"(local_amax_reg)
                 : "r"(local_amax_reg), "r"(temp_amax_reg));
  }

  if (kReturnTransposedAmax) {
    // TODO(Frank): This is not efficient, since we could directly load the
    // matrix in transposed layout.
    if (!kReturnIdentityAmax) {
      ldmatrix_x4_m8n8_shared_b16<false>(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                                         reinterpret_cast<uint4*>(in_sh_ptr) + swizzle_idx);
    }

    matrix_transpose_m8_n8_b16_inplace(a_frag[0]);
    matrix_transpose_m8_n8_b16_inplace(a_frag[1]);
    matrix_transpose_m8_n8_b16_inplace(a_frag[2]);
    matrix_transpose_m8_n8_b16_inplace(a_frag[3]);

    mma_m16_n16_k16_b16_b16_b16_noacc<kReturnTransposedAmax>(
        a_frag[0], a_frag[2], a_frag[1], a_frag[3], b_frag_t[0], b_frag_t[1], b_frag_t[2],
        b_frag_t[3], c_frag[0], c_frag[1], c_frag[2], c_frag[3], temp_amax_t_reg);
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                 : "=r"(local_amax_t_reg)
                 : "r"(local_amax_t_reg), "r"(temp_amax_t_reg));
  }

  if (kReturnPreRhtAmax) {
    if (!kReturnIdentityAmax && !kReturnTransposedAmax) {
      ldmatrix_x4_m8n8_shared_b16<false>(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                                         reinterpret_cast<uint4*>(in_sh_ptr) + swizzle_idx);
    }

    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                 : "=r"(a_frag[0])
                 : "r"(a_frag[0]), "r"(a_frag[1]));
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                 : "=r"(a_frag[2])
                 : "r"(a_frag[2]), "r"(a_frag[3]));
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                 : "=r"(a_frag[0])
                 : "r"(a_frag[0]), "r"(a_frag[2]));
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                 : "=r"(local_pre_rht_amax_reg)
                 : "r"(a_frag[0]), "r"(local_pre_rht_amax_reg));
  }
}

template <int kN>
__device__ __host__ constexpr int NextPowerOf2() {
  static_assert(kN > 0, "kN must be > 0");
  // Round up to the next power of 2 by counting leading zeros.
  return 1 << (32 - __builtin_clz(kN - 1));
}

template <int kNumWarps, bool kReturnPreRhtAmax, bool kReturnIdentityAmax,
          bool kReturnTransposedAmax>
__device__ __forceinline__ void ReduceMax(const float pre_rht_amax, const float identity_amax,
                                          const float transpose_amax, float* staging_for_pre_rht,
                                          float* staging_for_identity, float* staging_for_transpose,
                                          float* output_pre_rht_amax_ptr,
                                          float* output_identity_amax_ptr,
                                          float* output_transpose_amax_ptr, const int warpid) {
  // intra-warp reduction
  constexpr int kWarpSize = 32;
  int local_rank = threadIdx.x % 32;
  float warp_pre_rht_amax = kReturnPreRhtAmax ? warp_reduce_max<kWarpSize>(pre_rht_amax) : 0.0f;
  float warp_identity_amax = kReturnIdentityAmax ? warp_reduce_max<kWarpSize>(identity_amax) : 0.0f;
  float warp_transpose_amax =
      kReturnTransposedAmax ? warp_reduce_max<kWarpSize>(transpose_amax) : 0.0f;

  // inter-warp reduction
  if (threadIdx.x % 32 == 0) {
    if (kReturnPreRhtAmax) {
      staging_for_pre_rht[warpid] = warp_pre_rht_amax;
    }
    if (kReturnIdentityAmax) {
      staging_for_identity[warpid] = warp_identity_amax;
    }
    if (kReturnTransposedAmax) {
      staging_for_transpose[warpid] = warp_transpose_amax;
    }
  }
  __syncthreads();
  constexpr int kNumWarpsPow2 = NextPowerOf2<kNumWarps>();
  if (warpid == 0) {
    if (kReturnIdentityAmax) {
      float identity_accum = local_rank < kNumWarps ? staging_for_identity[local_rank] : 0.0f;
      identity_accum = warp_reduce_max<kNumWarpsPow2>(identity_accum);
      if (local_rank == 0) {
        atomicMaxFloat(output_identity_amax_ptr, identity_accum);
      }
    }
  }
  if (warpid == 1) {
    if (kReturnTransposedAmax) {
      float transpose_accum = local_rank < kNumWarps ? staging_for_transpose[local_rank] : 0.0f;
      transpose_accum = warp_reduce_max<kNumWarpsPow2>(transpose_accum);
      if (local_rank == 0) {
        atomicMaxFloat(output_transpose_amax_ptr, transpose_accum);
      }
    }
  }
  if (warpid == 2) {
    if (kReturnPreRhtAmax) {
      float pre_rht_accum = local_rank < kNumWarps ? staging_for_pre_rht[local_rank] : 0.0f;
      pre_rht_accum = warp_reduce_max<kNumWarpsPow2>(pre_rht_accum);
      if (local_rank == 0) {
        atomicMaxFloat(output_pre_rht_amax_ptr, pre_rht_accum);
      }
    }
  }
}

__global__ void GraphSafeMultiZeroAmaxKernel(const size_t num_tensors, float* amax_rowwise_ptr,
                                             float* amax_colwise_ptr) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; tid < num_tensors; tid += stride) {
    amax_rowwise_ptr[tid] = 0;
    amax_colwise_ptr[tid] = 0;
  }
}

__global__ void GraphSafeMultiAmaxMemcpyD2DKernelPreRHT(const size_t num_tensors,
                                                        float* amax_rowwise_ptr,
                                                        float* amax_colwise_ptr) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; tid < num_tensors; tid += stride) {
    float* output_pre_rht_amax_ptr = amax_rowwise_ptr + tid;
    float* output_transpose_amax_ptr = amax_colwise_ptr + tid;
    if (output_pre_rht_amax_ptr != nullptr) {
      float pre_rht_amax = *output_pre_rht_amax_ptr;
      if (output_transpose_amax_ptr != nullptr) {
        *output_transpose_amax_ptr = pre_rht_amax;
      }
    }
  }
}

template <typename IType, int kHadamardDimension, int CHUNK_DIM_Y, int CHUNK_DIM_X, int BUFF_DIM_Y,
          int BUFF_DIM_X, int THREADS_PER_CHUNK, int THREADS_PER_Y, bool kReturnPreRhtAmax,
          bool kReturnIdentityAmax, bool kReturnTransposedAmax>
__global__ void GraphSafeGroupHadamardAmaxTmaKernel(
    const __grid_constant__ CUtensorMap tensor_map_input, uint16_t random_sign_mask,
    uint16_t random_sign_mask_t, const ShapeRepresentation shape_rep, const size_t num_tensors,
    const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t* const __restrict__ offsets_ptr, const int64_t* const __restrict__ first_dims_ptr,
    float* const __restrict__ amax_rowwise_ptr, float* const __restrict__ amax_colwise_ptr) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

  float* output_pre_rht_amax_ptr;
  float* output_identity_amax_ptr = nullptr;
  float* output_transpose_amax_ptr;

  // calculate the global offset to get tensor id
  size_t global_offset = blockIdx.y * CHUNK_DIM_Y * last_logical_dim;
  int tensor_id = get_current_tensor_id(shape_rep, num_tensors, global_offset, first_logical_dim,
                                        last_logical_dim, offsets_ptr);
  output_pre_rht_amax_ptr = static_cast<float*>(amax_rowwise_ptr) + tensor_id;
  output_transpose_amax_ptr = static_cast<float*>(amax_colwise_ptr) + tensor_id;

  static_assert(CHUNK_DIM_Y >= BUFF_DIM_Y && CHUNK_DIM_Y % BUFF_DIM_Y == 0);
  static_assert(CHUNK_DIM_X >= BUFF_DIM_X && CHUNK_DIM_X % BUFF_DIM_X == 0);

  constexpr size_t STAGES_Y = CHUNK_DIM_Y / BUFF_DIM_Y;
  constexpr size_t STAGES_X = CHUNK_DIM_X / BUFF_DIM_X;

  constexpr int kNumWarps = (THREADS_PER_CHUNK * THREADS_PER_Y) / kThreadsPerWarp;

  const int input_block_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const int input_block_offset_X = blockIdx.x * CHUNK_DIM_X;

  extern __shared__ __align__(128) char dynamic_shmem[];
  uintptr_t base_shmem_ptr = reinterpret_cast<uintptr_t>(dynamic_shmem);
  // Manually align dynamic SHMEM per TMA requirements using padding
  // __align__(128) Does not guarantee the pointer to be aligned!
  uint8_t* dshmem = reinterpret_cast<uint8_t*>((base_shmem_ptr + 127) & ~127ULL);

  // The destination shared memory buffer of a bulk tensor operation should be 16-byte aligned
  constexpr size_t in_buff_size = BUFF_DIM_X * BUFF_DIM_Y * sizeof(IType);
  IType* in_sh_0 = reinterpret_cast<IType*>(dshmem);
  dshmem += in_buff_size;
  IType* in_sh_1 = reinterpret_cast<IType*>(dshmem);
  dshmem += in_buff_size;

  IType* in_shs[2] = {in_sh_0, in_sh_1};

  constexpr int shmem_buff_size = BUFF_DIM_X * BUFF_DIM_Y * sizeof(IType);

  const bool is_master_thread = (threadIdx.x == 0 && threadIdx.y == 0);

  // Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  uint64_t* mbar = reinterpret_cast<uint64_t*>(dshmem);
  dshmem += sizeof(uint64_t) * (STAGES_X * STAGES_Y);

  float* max_staging_identity = reinterpret_cast<float*>(dshmem);
  dshmem += sizeof(float) * kNumWarps;
  float* max_staging_transpose = reinterpret_cast<float*>(dshmem);
  dshmem += sizeof(float) * kNumWarps;
  float* max_staging_pre_rht = reinterpret_cast<float*>(dshmem);
  dshmem += sizeof(float) * kNumWarps;

  initialize_barriers<STAGES_X * STAGES_Y, THREADS_PER_CHUNK * THREADS_PER_Y>(mbar,
                                                                              is_master_thread);

  copy_2d_to_shared(in_shs[0], reinterpret_cast<const void*>(&tensor_map_input),
                    input_block_offset_X, input_block_offset_Y, shmem_buff_size, &mbar[0],
                    is_master_thread);

  uint32_t had_frag_i[4];
  uint32_t had_frag_t[4];
  get_hadamard_matrix_fragment<kReturnIdentityAmax, kReturnTransposedAmax, false, false>(
      had_frag_i, random_sign_mask, had_frag_t, random_sign_mask_t);

  float local_pre_rht_amax = 0.0;
  float local_amax = 0.0;
  float local_amax_t = 0.0;
  uint32_t local_pre_rht_amax_reg = *reinterpret_cast<uint32_t*>(&local_pre_rht_amax);
  uint32_t local_amax_reg = *reinterpret_cast<uint32_t*>(&local_amax);
  uint32_t local_amax_t_reg = *reinterpret_cast<uint32_t*>(&local_amax_t);

  for (int stage_y = 0; stage_y < STAGES_Y; ++stage_y) {
    for (int stage_x = 0; stage_x < STAGES_X; ++stage_x) {
      int stage = STAGES_X * stage_y + stage_x;

      const int next_stage = stage + 1;
      const int next_stage_x = stage_x + 1 == STAGES_X ? 0 : stage_x + 1;
      const int next_stage_y = stage_x + 1 == STAGES_X ? stage_y + 1 : stage_y;

      if (next_stage < STAGES_X * STAGES_Y) {
        const int input_global_offset_Y = input_block_offset_Y + next_stage_y * BUFF_DIM_Y;
        const int input_global_offset_X = input_block_offset_X + next_stage_x * BUFF_DIM_X;

        copy_2d_to_shared(in_shs[next_stage % 2],  // ping-pong
                          reinterpret_cast<const void*>(&tensor_map_input), input_global_offset_X,
                          input_global_offset_Y, shmem_buff_size, &mbar[next_stage],
                          is_master_thread);
      }

      ptx::fence_proxy_async_shared_cta();

      // Wait for the data to have arrived
      ptx::mbarrier_wait_parity(&mbar[stage], 0);

      const size_t compute_stage_x_num =
          BUFF_DIM_X / (kHadamardDimension * (THREADS_PER_CHUNK / kThreadsPerWarp));
      const size_t compute_stage_y_num = BUFF_DIM_Y / (kHadamardDimension * THREADS_PER_Y);

      const size_t in_row_stride = BUFF_DIM_X;

      IType* in_sh_ptr = in_shs[stage % 2];

#pragma unroll
      for (size_t compute_stage_y = 0; compute_stage_y < compute_stage_y_num; compute_stage_y++) {
        const int row_idx_offset = (compute_stage_y * kHadamardDimension * THREADS_PER_Y +
                                    threadIdx.y * kHadamardDimension);
        const int in_row_offset = row_idx_offset * in_row_stride;

#pragma unroll
        for (size_t compute_stage_x = 0; compute_stage_x < compute_stage_x_num; compute_stage_x++) {
          ComputeKernel<IType, kHadamardDimension, BUFF_DIM_Y, BUFF_DIM_X, kReturnPreRhtAmax,
                        kReturnIdentityAmax, kReturnTransposedAmax>(
              had_frag_i, had_frag_t,
              in_sh_ptr + in_row_offset +
                  (compute_stage_x * kHadamardDimension * (THREADS_PER_CHUNK / kThreadsPerWarp)),
              local_pre_rht_amax_reg, local_amax_reg, local_amax_t_reg);
        }

        // Ensure all threads have finished their computation before new data over-writes the shared
        // memory.
        __syncthreads();
      }
    }
  }

  const int warpid = (threadIdx.x + threadIdx.y * blockDim.x) / kThreadsPerWarp;

  if constexpr (kReturnPreRhtAmax) {
    unpack_max_of_packed_bf16(local_pre_rht_amax_reg, local_pre_rht_amax);
  }
  if constexpr (kReturnIdentityAmax) {
    unpack_max_of_packed_bf16(local_amax_reg, local_amax);
  }
  if constexpr (kReturnTransposedAmax) {
    unpack_max_of_packed_bf16(local_amax_t_reg, local_amax_t);
  }

  ReduceMax<kNumWarps, kReturnPreRhtAmax, kReturnIdentityAmax, kReturnTransposedAmax>(
      local_pre_rht_amax, local_amax, local_amax_t, max_staging_pre_rht, max_staging_identity,
      max_staging_transpose, output_pre_rht_amax_ptr, output_identity_amax_ptr,
      output_transpose_amax_ptr, warpid);

  destroy_barriers<STAGES_X * STAGES_Y>(mbar, is_master_thread);
#else
  NVTE_DEVICE_ERROR("Kernel is only supported on SM 10.0+.");
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

}  // namespace

// broadcast_pre_rht_amax: when it's true, hadamard transform will be disabled
// if at this time, the amax buffers for output expects both amax_rowwise and amax_colwise
// then call MultiAmaxMemcpyD2DKernelPreRHT to D2D copy the amax values
void group_hadamard_transform_amax_graph_safe(const GroupedTensor* input, GroupedTensor* output,
                                              uint16_t random_sign_mask,
                                              uint16_t random_sign_mask_t,
                                              bool broadcast_pre_rht_amax, cudaStream_t stream) {
  NVTE_API_CALL(group_hadamard_transform_amax_graph_safe);
#if CUDA_VERSION >= 12080

  NVTE_CHECK(input->num_tensors == output->num_tensors,
             "Number of input and output tensors must be same.");
  NVTE_CHECK(input->has_data(), "Cannot quantize tensor without rowwise data.");

  checkCuDriverContext(stream);

  bool all_return_pre_rht_amax = output->has_data();
  // there is no rowwise RHT transform in current recipe
  bool all_return_identity_amax = false;
  bool all_return_transposed_amax = output->has_columnwise_data();

  NVTE_CHECK(all_return_pre_rht_amax || all_return_identity_amax || all_return_transposed_amax,
             "At least one of return_pre_rht_amax, return_identity_amax, or return_transposed_amax "
             "must be true");

  if (broadcast_pre_rht_amax) {
    NVTE_CHECK(all_return_pre_rht_amax,
               "broadcast_pre_rht_amax is only supported when we compute pre-RHT amax");
    // if all_return_identity_amax and all_return_transposed_amax both are false, there is no need to broadcast anything
    broadcast_pre_rht_amax &= (all_return_identity_amax || all_return_transposed_amax);
  }

  const size_t num_tensors = input->num_tensors;
  const size_t first_logical_dim = input->logical_shape.data[0];
  const size_t last_logical_dim = input->logical_shape.data[1];
  // const size_t elts_total = first_logical_dim * last_logical_dim;
  NVTE_CHECK(first_logical_dim % 128 == 0,
             "First dimension of a grouped tensor should be divisible by 128.");
  NVTE_CHECK(last_logical_dim % 128 == 0,
             "Last dimension of a grouped tensor should be divisible by 128.");

  float* const amax_rowwise_ptr = reinterpret_cast<float*>(output->amax.dptr);
  float* const amax_colwise_ptr = reinterpret_cast<float*>(output->columnwise_amax.dptr);

  const int64_t* const offsets_ptr = reinterpret_cast<const int64_t*>(input->tensor_offsets.dptr);
  const int64_t* const first_dims_ptr = reinterpret_cast<const int64_t*>(input->first_dims.dptr);
  // const int64_t *const last_dims_ptr = reinterpret_cast<const int64_t *>(input->last_dims.dptr);

  // some sanity checks
  if (all_return_pre_rht_amax) {
    NVTE_CHECK(amax_rowwise_ptr != nullptr, "Amax rowwise pointer should not be nullptr.");
  }
  if (all_return_transposed_amax) {
    NVTE_CHECK(amax_colwise_ptr != nullptr, "Amax columnwise pointer should not be nullptr.");
  }

  // Multi zero out multiple amaxes if needed
  dim3 block_setup_amax(kMaxTensorsPerKernel);
  dim3 grid_setup_amax(1);
  GraphSafeMultiZeroAmaxKernel<<<grid_setup_amax, block_setup_amax, 0, stream>>>(
      num_tensors, amax_rowwise_ptr, amax_colwise_ptr);
  NVTE_CHECK_CUDA(cudaGetLastError());

  using IType = bf16;
  constexpr int kHadamardDimension = 16;

  // four (1x4) 64x64 sub-tiles for ping-pong overlap
  constexpr uint64_t kChunkBlockXSmall = 256;
  constexpr uint64_t kChunkBlockYSmall = 64;
  constexpr uint64_t kBuffDimX = 64;
  constexpr uint64_t kBuffDimY = 64;

  alignas(64) CUtensorMap tensor_map_input{};

  create_2D_tensor_map(
      /*tensorMap=*/tensor_map_input,
      /*tensor=*/input->data,
      /*globalY=*/first_logical_dim,
      /*globalX=*/last_logical_dim,
      /*shmemY=*/kBuffDimY,
      /*shmemX=*/kBuffDimX,
      /*stride_elems=*/last_logical_dim,
      /*offset_elems=*/0,
      /*type_num_bits=*/sizeof(IType) * 8,
      /*swizzle=*/CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B);

  constexpr uint64_t kThreadBlockX = 4;
  constexpr uint64_t kThreadBlockY = 1;
  constexpr uint64_t kNumWarps = kThreadBlockX * kThreadBlockY;

  dim3 block(kThreadBlockX * kThreadsPerWarp, kThreadBlockY);
  dim3 grid(DIVUP(last_logical_dim, kChunkBlockXSmall),
            DIVUP(first_logical_dim, kChunkBlockYSmall));

  ShapeRepresentation shape_rep = ShapeRepresentation::VARYING_FIRST_DIM;
  if (output->all_same_shape()) {
    shape_rep = ShapeRepresentation::SAME_BOTH_DIMS;
  } else if (output->all_same_first_dim()) {
    shape_rep = ShapeRepresentation::VARYING_LAST_DIM;
  } else if (output->all_same_last_dim()) {
    shape_rep = ShapeRepresentation::VARYING_FIRST_DIM;
  } else if (output->varying_both_dims()) {
    shape_rep = ShapeRepresentation::VARYING_BOTH_DIMS;
  }

  const bool is_const_last_dim = (shape_rep == ShapeRepresentation::SAME_BOTH_DIMS ||
                                  shape_rep == ShapeRepresentation::VARYING_FIRST_DIM);

  NVTE_CHECK(is_const_last_dim,
             "Currently we only support const last dimension for graph safe hadamard transform.");

  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      (all_return_transposed_amax && !broadcast_pre_rht_amax), kReturnTransposedAmax,

      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          (all_return_identity_amax && !broadcast_pre_rht_amax), kReturnIdentityAmax,

          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              all_return_pre_rht_amax, kReturnPreRhtAmax,

              // *2 for ping-pong
              size_t in_sh_size = kBuffDimX * kBuffDimY * 2 * sizeof(IType);
              size_t mbar_size = sizeof(uint64_t) * (kChunkBlockXSmall / kBuffDimX) *
                                 (kChunkBlockYSmall / kBuffDimY);
              size_t shmem_bytes = in_sh_size + mbar_size + kNumWarps * sizeof(float) * 3;
              // Add padding in case shmem ptr is not aligned to 128 bytes.
              shmem_bytes = (shmem_bytes + 128);

              auto kernel = GraphSafeGroupHadamardAmaxTmaKernel<
                  IType, kHadamardDimension, kChunkBlockYSmall, kChunkBlockXSmall, kBuffDimY,
                  kBuffDimX, kThreadBlockX * kThreadsPerWarp, kThreadBlockY, kReturnPreRhtAmax,
                  kReturnIdentityAmax, kReturnTransposedAmax>;
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   shmem_bytes);

              kernel<<<grid, block, shmem_bytes, stream>>>(
                  tensor_map_input, random_sign_mask, random_sign_mask_t, shape_rep, num_tensors,
                  first_logical_dim, last_logical_dim, offsets_ptr, first_dims_ptr,
                  amax_rowwise_ptr, amax_colwise_ptr);
              if (broadcast_pre_rht_amax) {
                GraphSafeMultiAmaxMemcpyD2DKernelPreRHT<<<grid_setup_amax, block_setup_amax, 0,
                                                          stream>>>(num_tensors, amax_rowwise_ptr,
                                                                    amax_colwise_ptr);
              })));

  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("Hadamard transform requires CUDA 12.8+, but compile-time CUDA version is ",
             CUDA_VERSION);
#endif  // CUDA_VERSION >= 12080
}

}  // namespace transformer_engine

void nvte_group_hadamard_transform_amax_graph_safe(const NVTEGroupedTensor input,
                                                   NVTEGroupedTensor output, int random_sign_mask,
                                                   int random_sign_mask_t, cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_hadamard_transform_amax_graph_safe);
  using namespace transformer_engine;

  GroupedTensor* input_tensor = convertNVTEGroupedTensorCheck(input);
  GroupedTensor* output_tensor = convertNVTEGroupedTensorCheck(output);

  if (input_tensor->num_tensors == 0) {
    return;
  }

  // Call the group tensor Hadamard transform amax implementation.
  group_hadamard_transform_amax_graph_safe(
      input_tensor, output_tensor, static_cast<uint16_t>(random_sign_mask),
      static_cast<uint16_t>(random_sign_mask_t), false, stream);
}

// Grouped-tensor amax without doing hadamard transform
void nvte_group_amax_graph_safe(const NVTEGroupedTensor input, NVTEGroupedTensor output,
                                cudaStream_t stream) {
  NVTE_API_CALL(nvte_group_amax_graph_safe);
  using namespace transformer_engine;

  GroupedTensor* input_tensor = convertNVTEGroupedTensorCheck(input);
  GroupedTensor* output_tensor = convertNVTEGroupedTensorCheck(output);

  if (input_tensor->num_tensors == 0) {
    return;
  }

  group_hadamard_transform_amax_graph_safe(input_tensor, output_tensor, 0, 0, true, stream);
}

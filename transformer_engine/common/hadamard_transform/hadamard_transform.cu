/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <transformer_engine/hadamard_transform.h>

#include <cuda/barrier>

#include "common/common.h"
#include "common/util/ptx.cuh"
#include "common/utils.cuh"

namespace transformer_engine {
namespace {

constexpr int kThreadsPerWarp = 32;
constexpr float k16x16HadamardScale = 0.25f;

template <bool kTranspose>
__device__ __forceinline__ void ldmatrix_x4_m8n8_shared_b16(uint32_t& a0, uint32_t& a1,
                                                            uint32_t& a2, uint32_t& a3,
                                                            void* addr) {
  auto smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
  if constexpr (kTranspose) {
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                 : "r"(smem_addr));
  } else {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                 : "r"(smem_addr));
  }
}

template <bool kTranspose>
__device__ __forceinline__ void load_matrix_16x16_from_shared(uint32_t& a0, uint32_t& a1,
                                                              uint32_t& a2, uint32_t& a3,
                                                              void* addr, uint32_t stride) {
  if constexpr (kTranspose) {
    asm volatile(
        "wmma.load.a.sync.aligned.col.m16n16k16.shared::cta.bf16 "
        "{%0,%1,%2,%3}, [%4], %5;\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        : "l"(addr), "r"(stride));
  } else {
    asm volatile(
        "wmma.load.a.sync.aligned.row.m16n16k16.shared::cta.bf16 "
        "{%0,%1,%2,%3}, [%4], %5;\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        : "l"(addr), "r"(stride));
  }
}

template <bool kTranspose>
__device__ __forceinline__ void store_matrix_16x16_to_global(uint32_t& a0, uint32_t& a1,
                                                             uint32_t& a2, uint32_t& a3, void* addr,
                                                             uint32_t stride) {
  if constexpr (kTranspose) {
    asm volatile("wmma.store.d.sync.aligned.col.m16n16k16.global.f16 [%0], {%1, %2, %3, %4}, %5;\n"
                 :
                 : "l"(addr), "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(stride));
  } else {
    asm volatile("wmma.store.d.sync.aligned.row.m16n16k16.global.f16 [%0], {%1, %2, %3, %4}, %5;\n"
                 :
                 : "l"(addr), "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(stride));
  }
}

__device__ __forceinline__ void matrix_transpose_m8_n8_b16_inplace(uint32_t& a0) {
  asm volatile(
      "movmatrix.sync.aligned.m8n8.trans.b16 "
      "%0, %1;\n\t"
      : "=r"(a0)
      : "r"(a0));
}

__device__ __forceinline__ void unpack_max_of_packed_bf16(uint32_t& packed_bf16, float& float_dst) {
  __nv_bfloat162 bf16x2 = *reinterpret_cast<__nv_bfloat162*>(&packed_bf16);
  float f_a = __bfloat162float(bf16x2.x);
  float f_b = __bfloat162float(bf16x2.y);
  asm volatile("max.xorsign.abs.f32 %0, %1, %2;\n\t" : "=f"(float_dst) : "f"(f_a), "f"(f_b));
  float_dst = fabsf(float_dst);
}

template <bool kCalculateAmax>
__device__ __forceinline__ void mma_m16_n16_k16_b16_b16_b16_noacc(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3, uint32_t& b0, uint32_t& b1,
    uint32_t& b2, uint32_t& b3, uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3,
    uint32_t& amax_result) {
  uint32_t zero = 0;
  uint32_t temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7;
  asm volatile(
      "wmma.mma.sync.aligned.row.row.m16n16k16.f32.bf16.bf16.f32 \n"
      "{%0, %1, %2, %3, %4, %5, %6, %7}, \n"
      "{%8, %9, %10, %11}, \n"
      "{%12, %13, %14, %15}, \n"
      "{%16, %17, %18, %19, %20, %21, %22, %23};\n\t"
      : "=r"(temp0), "=r"(temp1), "=r"(temp2), "=r"(temp3), "=r"(temp4), "=r"(temp5), "=r"(temp6),
        "=r"(temp7)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(zero),
        "r"(zero), "r"(zero), "r"(zero), "r"(zero), "r"(zero), "r"(zero), "r"(zero));
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c0) : "r"(temp1), "r"(temp0));
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c1) : "r"(temp3), "r"(temp2));
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c2) : "r"(temp5), "r"(temp4));
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t" : "=r"(c3) : "r"(temp7), "r"(temp6));
  if constexpr (kCalculateAmax) {
    uint32_t max_even;
    uint32_t max_odd;
    // Reduction tree to amax(abs(result)) into bf16x2 reg outparam.
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t" : "=r"(max_even) : "r"(c0), "r"(c2));
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t" : "=r"(max_odd) : "r"(c1), "r"(c3));
    // N.B. mma is only called up to once per thread for identity and transpose respectively, so
    // we don't have to accumulate into amax_result and can directly store into it.
    asm volatile("max.xorsign.abs.bf16x2 %0, %1, %2;\n\t"
                 : "=r"(amax_result)
                 : "r"(max_even), "r"(max_odd));
  }
}

template <bool kReturnIdentity, bool kReturnTransposed, bool kInverseHadamardIdentity,
          bool kInverseHadamardTransposed>
__device__ __forceinline__ void get_hadamard_matrix_fragment(uint32_t* had_frag_i,
                                                             uint16_t random_sign_mask,
                                                             uint32_t* had_frag_t,
                                                             uint16_t random_sign_mask_t) {
  int32_t tid = threadIdx.x % 32;  // Local tid
  float temp_i[2];
  float temp_t[2];
#pragma unroll
  for (int i = 0; i < 2; i++) {
    // i is the vertical fragment index.
    // For a 16x16 matrix matrix fragment, 4 threads fill a fragment of 8 BF16 vals.
    uint32_t r = i * 8 + tid / 4;

#pragma unroll
    for (int j = 0; j < 2; j++) {
#pragma unroll
      for (int k = 0; k < 2; k++) {
        // k is column position [0, 1] within a quad of 2 BF16s  stored together in 32 bits.
        // j is the column fragment idx selecting between even and odd fragments.
        // j increments 8 columns by switching fragments.
        uint32_t c = j * 8 + k + tid % 4 * 2;
        // 1 -> -1.0f, 0 -> 1.0f
        int32_t base_sign = __popc(r & c);
        if constexpr (kReturnIdentity) {
          int32_t sign_i;
          // Because tensor cores want the dot product dimension,
          // contiguous, the regular, non-inverse hadamard swaps
          // signs of columns and rows for inverse. In a simple reference,
          // x.reshape(-1, 16) @ sign @ H16, this would be opposite but
          // (sign @ H16) is transposed in this fragment.
          if constexpr (kInverseHadamardIdentity) {
            sign_i = ((random_sign_mask >> r) ^ base_sign);
          } else {
            sign_i = ((random_sign_mask >> c) ^ base_sign);
          }
          temp_i[k] = copysignf(k16x16HadamardScale, __int_as_float(sign_i << 31));
        }
        if constexpr (kReturnTransposed) {
          int32_t sign_t;
          if constexpr (kInverseHadamardTransposed) {
            sign_t = ((random_sign_mask_t >> r) ^ base_sign);
          } else {
            sign_t = ((random_sign_mask_t >> c) ^ base_sign);
          }
          temp_t[k] = copysignf(k16x16HadamardScale, __int_as_float(sign_t << 31));
        }
      }

      if constexpr (kReturnIdentity) {
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t"
                     : "=r"(had_frag_i[i * 2 + j])
                     : "f"(temp_i[1]), "f"(temp_i[0]));
      }
      if constexpr (kReturnTransposed) {
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;\n\t"
                     : "=r"(had_frag_t[i * 2 + j])
                     : "f"(temp_t[1]), "f"(temp_t[0]));
      }
    }
  }
}

__device__ __forceinline__ uint32_t swizzle_128B_atom_32B(uint32_t gmem_row_idx,
                                                          uint32_t gmem_col_idx) {
  uint32_t smem_row_idx = gmem_row_idx;
  uint32_t xor_factor = (smem_row_idx * 2) % 8;
  uint32_t smem_col_idx = gmem_col_idx ^ xor_factor;
  return smem_row_idx * 8 + smem_col_idx;
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

__launch_bounds__(1) __global__ void ZeroAmaxKernel(float* __restrict__ output_pre_rht_amax_ptr,
                                                    float* __restrict__ output_identity_amax_ptr,
                                                    float* __restrict__ output_transpose_amax_ptr) {
  if (output_pre_rht_amax_ptr != nullptr) {
    *output_pre_rht_amax_ptr = 0;
  }
  if (output_identity_amax_ptr != nullptr) {
    *output_identity_amax_ptr = 0;
  }
  if (output_transpose_amax_ptr != nullptr) {
    *output_transpose_amax_ptr = 0;
  }
}

template <typename IType, int kHadamardDimension, int CHUNK_DIM_Y, int CHUNK_DIM_X, int BUFF_DIM_Y,
          int BUFF_DIM_X, int THREADS_PER_CHUNK, int THREADS_PER_Y, bool kReturnPreRhtAmax,
          bool kReturnIdentityAmax, bool kReturnTransposedAmax>
__global__ void HadamardAmaxTmaKernel(const __grid_constant__ CUtensorMap tensor_map_input,
                                      float* __restrict__ output_pre_rht_amax_ptr,
                                      float* __restrict__ output_identity_amax_ptr,
                                      float* __restrict__ output_transpose_amax_ptr,
                                      uint16_t random_sign_mask, uint16_t random_sign_mask_t,
                                      uint64_t num_rows, uint64_t row_length) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

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

template <typename T, int kHadamardDimension, bool kComputeIdentity, bool kComputeTransposed,
          bool kReturnIdentity, bool kReturnTransposed, bool kUpdateIdentityAmax,
          bool kUpdateTransposeAmax, bool kOutputTrueTransposed>
__global__ void HadamardTransformKernel(const T* __restrict__ input, T* __restrict__ output,
                                        T* __restrict__ output_t, uint16_t random_sign_mask,
                                        uint16_t random_sign_mask_t, uint64_t num_input_rows,
                                        uint64_t num_input_cols, float* __restrict__ amax,
                                        float* __restrict__ amax_t, bool inverse_hadamard) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  static_assert(kHadamardDimension == 16, "Currently only hadamard dimension 16 is supported.");

  // The whole threadblock will share the same smem.
  extern __shared__ __align__(16) T smem[];

  // Each 32 threads process a 16x16 matrix. There is a (y, z) grid of 16x16.
  // If y = 4, z = 4, then each threadblock is processing a 4x4 grid of 16x16 matrices.
  int32_t tid = threadIdx.x;
  int32_t warp_id = threadIdx.y * blockDim.z + threadIdx.z;
  int32_t local_bx = threadIdx.y;
  int32_t local_by = threadIdx.z;

  // Define the register fragments
  uint32_t a_frag[4];    // A matrix fragment
  uint32_t b_frag_i[4];  // Transposed Hadamard matrix fragment, used for A @ B(col major)
  uint32_t b_frag_t[4];  // Hadamard matrix fragment, used for A.T @ B.T(col major)
  uint32_t c_frag[4];    // Result fragment

  // row and col for each thread. 32 threads will work together in 128 chunk to
  // load the data from global memory to shared memory.
  uint32_t row = tid / (kHadamardDimension * sizeof(T) / sizeof(uint4));
  uint32_t col = tid % (kHadamardDimension * sizeof(T) / sizeof(uint4));

  uint32_t smem_index = tid;

  uint32_t input_start_col = (blockIdx.x * blockDim.y + local_bx) * kHadamardDimension;
  uint32_t input_start_row = (blockIdx.y * blockDim.z + local_by) * kHadamardDimension;

  bool load = (input_start_col < num_input_cols) && (input_start_row < num_input_rows);
  if (!load) {
    // Out of bound, we are returning early. No thread divergence since the whole warp
    // will return early.
    return;
  }

  uint64_t global_offset = input_start_col + input_start_row * num_input_cols;
  uint64_t global_offset_t =
      kOutputTrueTransposed ? (input_start_row + input_start_col * num_input_rows) : global_offset;

  T* base_smem = smem + kHadamardDimension * kHadamardDimension * warp_id;

  uint32_t* smem_b32 = reinterpret_cast<uint32_t*>(base_smem);
  uint4* smem_b128 = reinterpret_cast<uint4*>(base_smem);

  // Asynchronously load the data from global memory to shared memory.
  const uint4* input_b128 = reinterpret_cast<const uint4*>(input + global_offset);
  // Each 16x16 chunk is divided into 4 8x8 matrices, we are trying to load each
  // 8x8 chunks consecutively into the smem, so we could leverage ldmatrix m8n8x4
  // to load the data in the tensor core swizzled format.
  __pipeline_memcpy_async(&smem_b128[smem_index],
                          &input_b128[row * num_input_cols / (sizeof(uint4) / sizeof(T)) + col],
                          sizeof(uint4));
  __pipeline_commit();  // Commit the memcpy. Wait when we are in the computation.

  if (inverse_hadamard) {
    get_hadamard_matrix_fragment<kComputeIdentity, kComputeTransposed,
                                 /*kInverseHadamard=*/true,
                                 /*kInverseHadamardTransposed=*/true>(b_frag_i, random_sign_mask,
                                                                      b_frag_t, random_sign_mask_t);
  } else {
    get_hadamard_matrix_fragment<kComputeIdentity, kComputeTransposed,
                                 /*kInverseHadamard=*/false,
                                 /*kInverseHadamardTransposed=*/false>(
        b_frag_i, random_sign_mask, b_frag_t, random_sign_mask_t);
  }

  float local_amax = 0.0;
  float local_amax_t = 0.0;
  uint32_t local_amax_reg = *reinterpret_cast<uint32_t*>(&local_amax);
  uint32_t local_amax_t_reg = *reinterpret_cast<uint32_t*>(&local_amax_t);
  __pipeline_wait_prior(0);

  __syncwarp();  // ensure all lanes finished their cp.async before reading smem

  // Load the A to a_frag.
  if constexpr (kComputeIdentity) {
    load_matrix_16x16_from_shared<false>(a_frag[0], a_frag[1], a_frag[2], a_frag[3], smem_b32,
                                         kHadamardDimension);

    // 16x16 @ 16x16 leveraging all threads in the warp.
    mma_m16_n16_k16_b16_b16_b16_noacc<kUpdateIdentityAmax>(
        a_frag[0], a_frag[1], a_frag[2], a_frag[3], b_frag_i[0], b_frag_i[1], b_frag_i[2],
        b_frag_i[3], c_frag[0], c_frag[1], c_frag[2], c_frag[3], local_amax_reg);

    // Store the result to the shared memory in non-transposed order.
    if constexpr (kReturnIdentity) {
      uint4* output_b128 = reinterpret_cast<uint4*>(output + global_offset);
      store_matrix_16x16_to_global<false>(c_frag[0], c_frag[1], c_frag[2], c_frag[3], output_b128,
                                          num_input_cols);
    }
  }

  if constexpr (kComputeTransposed) {
    if (kComputeIdentity) {
      matrix_transpose_m8_n8_b16_inplace(a_frag[0]);
      matrix_transpose_m8_n8_b16_inplace(a_frag[1]);
      matrix_transpose_m8_n8_b16_inplace(a_frag[2]);
      matrix_transpose_m8_n8_b16_inplace(a_frag[3]);
    } else {
      load_matrix_16x16_from_shared<true>(a_frag[0],
                                          a_frag[2],  // NOTE: intentional index swapping
                                          a_frag[1],  // NOTE: intentional index swapping
                                          a_frag[3], smem_b32, kHadamardDimension);
    }

    mma_m16_n16_k16_b16_b16_b16_noacc<kUpdateTransposeAmax>(
        a_frag[0],
        // 2,1 is used if we are using movmatrix instruction.
        // Thus loading the matrix in 2,1 order will just be normal.
        // This is to be compatible with the movmatrix instruction.
        a_frag[2],  // NOTE: intentional index swapping for transpose purpose.
        a_frag[1],  // NOTE: intentional index swapping for transpose purpose.
        a_frag[3], b_frag_t[0], b_frag_t[1], b_frag_t[2], b_frag_t[3], c_frag[0], c_frag[1],
        c_frag[2], c_frag[3], local_amax_t_reg);

    // Store the result to the shared memory in non-transposed order.
    if constexpr (kReturnTransposed) {
      uint4* output_t_b128 = reinterpret_cast<uint4*>(output_t + global_offset_t);
      store_matrix_16x16_to_global<!kOutputTrueTransposed>(
          c_frag[0], c_frag[1], c_frag[2], c_frag[3], output_t_b128,
          kOutputTrueTransposed ? num_input_rows : num_input_cols);
    }
  }

  if constexpr (kUpdateIdentityAmax) {
    unpack_max_of_packed_bf16(local_amax_reg, local_amax);
    local_amax = warp_reduce_max<kThreadsPerWarp>(local_amax);
    // broadcast the amax to all threads in a warp from the lane 0
    constexpr int lane_zero = 0;
    local_amax = __shfl_sync(0xFFFFFFFF, local_amax, lane_zero);
    // atomic CAS to output memory.
    if (tid % kThreadsPerWarp == 0) {
      atomicMaxFloat(amax, local_amax);
    }
  }
  if constexpr (kUpdateTransposeAmax) {
    unpack_max_of_packed_bf16(local_amax_t_reg, local_amax_t);
    local_amax_t = warp_reduce_max<kThreadsPerWarp>(local_amax_t);
    // broadcast the amax to all threads in a warp from the lane 0
    constexpr int lane_zero = 0;
    local_amax_t = __shfl_sync(0xFFFFFFFF, local_amax_t, lane_zero);
    // atomic CAS to output memory.
    if (tid % kThreadsPerWarp == 0) {
      atomicMaxFloat(amax_t, local_amax_t);
    }
  }
#else
  NVTE_DEVICE_ERROR("Kernel is only supported on SM 9.0+.");
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
}

}  // namespace

void hadamard_transform(const Tensor& input_, Tensor& output_, uint16_t random_sign_mask,
                        uint16_t random_sign_mask_t, cudaStream_t stream) {
  NVTE_API_CALL(hadamard_transform);

  // Check tensors
  // NOTE (frsun): This is non-intuitive, we are writing the result of
  // transposed RHT to the output of rowwise.
  NVTE_CHECK(input_.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Input tensor must be BF16 tensor, but scaling mode is ",
             to_string(input_.scaling_mode), ".");
  NVTE_CHECK(input_.dtype() == transformer_engine::DType::kBFloat16,
             "Input tensor must be BF16 tensor, but dtype is ", to_string(input_.dtype()), ".");
  NVTE_CHECK(input_.dim() >= 2, "Input must be a 2D tensor.");
  NVTE_CHECK(output_.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Output tensor must be simple tensor, but scaling mode is ",
             to_string(output_.scaling_mode), ".");
  const SimpleTensor& input = input_.data;
  SimpleTensor output;
  SimpleTensor& output_t = output_.data;

  // Check requested outputs
  const bool return_identity = output.dptr != nullptr;
  const bool return_transposed = output_t.dptr != nullptr;
  if (!return_identity && !return_transposed) {  // Nothing to do/ill-defined behavior.
    return;
  }

  checkCuDriverContext(stream);

  const size_t ndim = input.shape.size();
  const size_t row_length = input.shape[ndim - 1];
  size_t num_rows = 1;
  for (size_t i = 0; i < ndim - 1; ++i) {
    num_rows *= input.shape[i];
  }

  using IType = bf16;

  constexpr int kHadamardDimension = 16;
  NVTE_CHECK(row_length % kHadamardDimension == 0,
             "row_length must be divisible by hadamard_dimension.");
  NVTE_CHECK(num_rows % kHadamardDimension == 0,
             "num_rows must be divisible by hadamard_dimension");

  constexpr uint64_t kThreadBlockX = 4;
  // Configure 4 is used for Hopper, 8 is used for Blackwell for extra memory bandwidth.
  constexpr uint64_t kThreadBlockY = 4;

  uint64_t kNumWarpsPerSM = kThreadBlockX * kThreadBlockY;

  // The shared memory number of bytes required for **the whole threadblock**.
  size_t shmem_bytes = kHadamardDimension * kHadamardDimension * sizeof(IType) * kNumWarpsPerSM;

  dim3 block(kThreadsPerWarp, kThreadBlockX, kThreadBlockY);

  dim3 grid(DIVUP(row_length / kHadamardDimension, kThreadBlockX),
            DIVUP(num_rows / kHadamardDimension, kThreadBlockY));

  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      return_transposed, kReturnTransposed,

      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          return_identity, kReturnIdentity,

          auto kernel =
              HadamardTransformKernel<IType, kHadamardDimension, kReturnIdentity, kReturnTransposed,
                                      kReturnIdentity, kReturnTransposed, false, false, true>;

          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);

          kernel<<<grid, block, shmem_bytes, stream>>>(
              reinterpret_cast<const IType*>(input.dptr), reinterpret_cast<IType*>(output.dptr),
              reinterpret_cast<IType*>(output_t.dptr), random_sign_mask, random_sign_mask_t,
              num_rows, row_length, nullptr, nullptr, false);););

  NVTE_CHECK_CUDA(cudaGetLastError());
}

// Kernel that will apply the 16x16 hadamard transform the input and input.T, and then
// get the absolute max value of the result.
void hadamard_transform_amax(const Tensor& input_, Tensor& output_, uint16_t random_sign_mask,
                             uint16_t random_sign_mask_t, cudaStream_t stream) {
  NVTE_API_CALL(hadamard_transform_amax);
#if CUDA_VERSION >= 12080

  // Check input tensor
  NVTE_CHECK(input_.scaling_mode == NVTE_DELAYED_TENSOR_SCALING,
             "Input tensor must be BF16 tensor, but scaling mode is ",
             to_string(input_.scaling_mode), ".");
  NVTE_CHECK(input_.dtype() == transformer_engine::DType::kBFloat16,
             "Input tensor must be BF16 tensor, but dtype is ", to_string(input_.dtype()), ".");
  NVTE_CHECK(input_.dim() >= 2, "Input must be a 2D tensor.");
  const SimpleTensor& input = input_.data;

  // Check amax tensors
  SimpleTensor& output_pre_rht_amax = output_.amax;
  SimpleTensor output_identity_amax;
  SimpleTensor& output_transpose_amax = output_.columnwise_amax;

  // Check requested outputs
  const bool return_pre_rht_amax = output_pre_rht_amax.dptr != nullptr;
  const bool return_identity_amax = output_identity_amax.dptr != nullptr;
  const bool return_transposed_amax = output_transpose_amax.dptr != nullptr;
  if (!return_identity_amax && !return_transposed_amax &&
      !return_pre_rht_amax) {  // Nothing to do/ill-defined behavior.
    return;
  }

  // Zero out amaxes if needed
  ZeroAmaxKernel<<<1, 1, 0, stream>>>(reinterpret_cast<float*>(output_pre_rht_amax.dptr),
                                      reinterpret_cast<float*>(output_identity_amax.dptr),
                                      reinterpret_cast<float*>(output_transpose_amax.dptr));
  NVTE_CHECK_CUDA(cudaGetLastError());

  checkCuDriverContext(stream);

  using IType = bf16;

  const size_t ndim = input.shape.size();
  const size_t row_length = input.shape[ndim - 1];
  size_t num_rows = 1;
  for (size_t i = 0; i < ndim - 1; ++i) {
    num_rows *= input.shape[i];
  }

  constexpr int kHadamardDimension = 16;
  NVTE_CHECK(row_length % kHadamardDimension == 0,
             "row_length must be divisible by hadamard_dimension.");
  NVTE_CHECK(num_rows % kHadamardDimension == 0,
             "num_rows must be divisible by hadamard_dimension");

  constexpr uint64_t kChunkBlockXSmall = 128;
  constexpr uint64_t kChunkBlockYSmall = 128;
  constexpr uint64_t kBuffDimX = 64;
  constexpr uint64_t kBuffDimY = 64;

  alignas(64) CUtensorMap tensor_map_input{};

  create_2D_tensor_map(
      /*tensorMap=*/tensor_map_input,
      /*tensor=*/input,
      /*globalY=*/num_rows,
      /*globalX=*/row_length,
      /*shmemY=*/kBuffDimY,
      /*shmemX=*/kBuffDimX,
      /*stride_elems=*/row_length,
      /*offset_elems=*/0,
      /*type_num_bits=*/sizeof(IType) * 8,
      /*swizzle=*/CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B);

  constexpr uint64_t kThreadBlockX = 4;
  constexpr uint64_t kThreadBlockY = 1;
  constexpr uint64_t kNumWarps = kThreadBlockX * kThreadBlockY;

  dim3 block(kThreadBlockX * kThreadsPerWarp, kThreadBlockY);

  dim3 grid(DIVUP(row_length, kChunkBlockXSmall), DIVUP(num_rows, kChunkBlockYSmall));

  TRANSFORMER_ENGINE_SWITCH_CONDITION(
      return_transposed_amax, kReturnTransposedAmax,

      TRANSFORMER_ENGINE_SWITCH_CONDITION(
          return_identity_amax, kReturnIdentityAmax,

          TRANSFORMER_ENGINE_SWITCH_CONDITION(
              return_pre_rht_amax, kReturnPreRhtAmax,

              // *2 for ping-pong
              size_t in_sh_size = kBuffDimX * kBuffDimY * 2 * sizeof(IType);
              size_t mbar_size = sizeof(uint64_t) * (kChunkBlockXSmall / kBuffDimX) *
                                 (kChunkBlockYSmall / kBuffDimY);
              size_t shmem_bytes = in_sh_size + mbar_size + kNumWarps * sizeof(float) * 3;
              // Add padding in case shmem ptr is not aligned to 128 bytes.
              shmem_bytes = (shmem_bytes + 128);

              auto kernel = HadamardAmaxTmaKernel<
                  IType, kHadamardDimension, kChunkBlockYSmall, kChunkBlockXSmall, kBuffDimY,
                  kBuffDimX, kThreadBlockX * kThreadsPerWarp, kThreadBlockY, kReturnPreRhtAmax,
                  kReturnIdentityAmax, kReturnTransposedAmax>;
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   shmem_bytes);

              kernel<<<grid, block, shmem_bytes, stream>>>(
                  tensor_map_input, reinterpret_cast<float*>(output_pre_rht_amax.dptr),
                  reinterpret_cast<float*>(output_identity_amax.dptr),
                  reinterpret_cast<float*>(output_transpose_amax.dptr), random_sign_mask,
                  random_sign_mask_t, num_rows, row_length);)));

  NVTE_CHECK_CUDA(cudaGetLastError());
#else
  NVTE_ERROR("Hadamard transform requires CUDA 12.8+, but compile-time CUDA version is ",
             CUDA_VERSION);
#endif  // CUDA_VERSION >= 12080
}

}  // namespace transformer_engine

void nvte_hadamard_transform(const NVTETensor input, NVTETensor output, int random_sign_mask,
                             int random_sign_mask_t, cudaStream_t stream) {
  NVTE_API_CALL(nvte_hadamard_transform);
  using namespace transformer_engine;
  hadamard_transform(*convertNVTETensorCheck(input), *convertNVTETensorCheck(output),
                     static_cast<uint16_t>(random_sign_mask),
                     static_cast<uint16_t>(random_sign_mask_t), stream);
}

void nvte_hadamard_transform_amax(const NVTETensor input, NVTETensor output, int random_sign_mask,
                                  int random_sign_mask_t, cudaStream_t stream) {
  NVTE_API_CALL(nvte_hadamard_transform_amax);
  using namespace transformer_engine;
  hadamard_transform_amax(*convertNVTETensorCheck(input), *convertNVTETensorCheck(output),
                          static_cast<uint16_t>(random_sign_mask),
                          static_cast<uint16_t>(random_sign_mask_t), stream);
}

/*************************************************************************
* Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
*
* See LICENSE for license information.
************************************************************************/

#ifndef TRANSFORMER_ENGINE_HADAMARD_TRANSFORM_UTILS_CUH_
#define TRANSFORMER_ENGINE_HADAMARD_TRANSFORM_UTILS_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/common.h"
#include "common/util/ptx.cuh"
#include "common/utils.cuh"

namespace transformer_engine {

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

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_HADAMARD_TRANSFORM_UTILS_CUH_

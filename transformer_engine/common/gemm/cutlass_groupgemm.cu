/***************************************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#include "../common.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass_groupgemm.cuh"

namespace grouped_gemm {

// Explicit template instantiation to match the template declarations in the .cuh
template void CutlassGroupedGemm<false, false, cutlass::half_t>(const NVTETensor*,
                                                                const NVTETensor*, NVTETensor*,
                                                                NVTETensor*, float, float, int,
                                                                cudaStream_t, int, int);
template void CutlassGroupedGemm<true, false, cutlass::half_t>(const NVTETensor*, const NVTETensor*,
                                                               NVTETensor*, NVTETensor*, float,
                                                               float, int, cudaStream_t, int, int);
template void CutlassGroupedGemm<false, true, cutlass::half_t>(const NVTETensor*, const NVTETensor*,
                                                               NVTETensor*, NVTETensor*, float,
                                                               float, int, cudaStream_t, int, int);

template void CutlassGroupedGemm<false, false, cutlass::bfloat16_t>(const NVTETensor*,
                                                                    const NVTETensor*, NVTETensor*,
                                                                    NVTETensor*, float, float, int,
                                                                    cudaStream_t, int, int);
template void CutlassGroupedGemm<true, false, cutlass::bfloat16_t>(const NVTETensor*,
                                                                   const NVTETensor*, NVTETensor*,
                                                                   NVTETensor*, float, float, int,
                                                                   cudaStream_t, int, int);
template void CutlassGroupedGemm<false, true, cutlass::bfloat16_t>(const NVTETensor*,
                                                                   const NVTETensor*, NVTETensor*,
                                                                   NVTETensor*, float, float, int,
                                                                   cudaStream_t, int, int);

}  // namespace grouped_gemm

void cutlass_grouped_gemm(const NVTETensor* A, const NVTETensor* B, NVTETensor* D, int num_gemms,
                          bool transa, bool transb, bool grad, NVTETensor* workspace,
                          bool accumulate, int device, int math_sm_count, cudaStream_t stream) {
  auto* inputA = transformer_engine::convertNVTETensorCheck(A[0]);
  auto* inputB = transformer_engine::convertNVTETensorCheck(B[0]);

  auto A_type = get_cuda_dtype(inputA->data.dtype);
  auto B_type = get_cuda_dtype(inputB->data.dtype);

  float one = 1.0;
  float zero = 0.0;
  float alpha = one;
  float beta = (accumulate) ? one : zero;

  auto dispatch = [&](auto tag) {
    using T = decltype(tag);
    if (!transa && !transb) {
      grouped_gemm::CutlassGroupedGemm<false, false, T>(B, A, D, workspace, alpha, beta, num_gemms,
                                                        stream, device, math_sm_count);
    } else if (!transb && transa) {
      grouped_gemm::CutlassGroupedGemm<false, true, T>(B, A, D, workspace, alpha, beta, num_gemms,
                                                       stream, device, math_sm_count);
    } else if (transb && !transa) {
      grouped_gemm::CutlassGroupedGemm<true, false, T>(B, A, D, workspace, alpha, beta, num_gemms,
                                                       stream, device, math_sm_count);
    } else {
      NVTE_ERROR("Layout 'TT' is not supported by cutlass_grouped_gemm.");
    }
  };

  if (A_type == CUDA_R_16BF) {
    dispatch(cutlass::bfloat16_t{});
  } else if (A_type == CUDA_R_16F) {
    dispatch(cutlass::half_t{});
  } else {
    NVTE_ERROR("Unsupported dtype: only BF16(FP16) are supported.");
  }
}

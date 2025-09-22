/***************************************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass_grouped_gemm.cuh"

namespace transformer_engine {
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
}  // namespace transformer_engine

void cutlass_grouped_gemm(const NVTETensor* A, const NVTETensor* B, NVTETensor* D, int num_gemms,
                          bool transa, bool transb, bool grad, NVTETensor* workspace,
                          bool accumulate, int device, int math_sm_count, cudaStream_t stream) {
  using namespace transformer_engine;
  auto* inputA = convertNVTETensorCheck(A[0]);
  auto* inputB = convertNVTETensorCheck(B[0]);

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

  if (inputA->data.dtype == DType::kBFloat16) {
    dispatch(cutlass::bfloat16_t{});
  } else if (inputA->data.dtype == DType::kFloat16) {
    dispatch(cutlass::half_t{});
  } else {
    NVTE_ERROR("Unsupported dtype: only BF16(FP16) are supported.");
  }
}

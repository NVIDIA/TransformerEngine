/***************************************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

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

// ---- SM100 (Blackwell) forward grouped-GEMM instantiations (kSm100=true) ----
template void CutlassGroupedGemm<false, false, cutlass::half_t, true>(const NVTETensor*,
                                                                      const NVTETensor*,
                                                                      NVTETensor*, NVTETensor*,
                                                                      float, float, int,
                                                                      cudaStream_t, int, int);
template void CutlassGroupedGemm<true, false, cutlass::half_t, true>(const NVTETensor*,
                                                                     const NVTETensor*, NVTETensor*,
                                                                     NVTETensor*, float, float, int,
                                                                     cudaStream_t, int, int);
template void CutlassGroupedGemm<false, true, cutlass::half_t, true>(const NVTETensor*,
                                                                     const NVTETensor*, NVTETensor*,
                                                                     NVTETensor*, float, float, int,
                                                                     cudaStream_t, int, int);
template void CutlassGroupedGemm<false, false, cutlass::bfloat16_t, true>(const NVTETensor*,
                                                                          const NVTETensor*,
                                                                          NVTETensor*, NVTETensor*,
                                                                          float, float, int,
                                                                          cudaStream_t, int, int);
template void CutlassGroupedGemm<true, false, cutlass::bfloat16_t, true>(const NVTETensor*,
                                                                         const NVTETensor*,
                                                                         NVTETensor*, NVTETensor*,
                                                                         float, float, int,
                                                                         cudaStream_t, int, int);
template void CutlassGroupedGemm<false, true, cutlass::bfloat16_t, true>(const NVTETensor*,
                                                                         const NVTETensor*,
                                                                         NVTETensor*, NVTETensor*,
                                                                         float, float, int,
                                                                         cudaStream_t, int, int);

// Explicit instantiation: BF16-in / FP32-out (default) wgrad path.
template void CutlassGroupedGemmWgrad<true, false, float>(const NVTETensor*, const NVTETensor*,
                                                          NVTETensor*, NVTETensor*, float, float,
                                                          int, cudaStream_t, int, int);

// Explicit instantiation: BF16-in / BF16-out wgrad path.
template void CutlassGroupedGemmWgrad<true, false, cutlass::bfloat16_t>(const NVTETensor*,
                                                                        const NVTETensor*,
                                                                        NVTETensor*, NVTETensor*,
                                                                        float, float, int,
                                                                        cudaStream_t, int, int);

// ---- SM100 (Blackwell) wgrad instantiations (kSm100=true), both N-tile variants (kBigN) ----
template void CutlassGroupedGemmWgrad<true, false, float, true, false>(const NVTETensor*,
                                                                       const NVTETensor*,
                                                                       NVTETensor*, NVTETensor*,
                                                                       float, float, int,
                                                                       cudaStream_t, int, int);
template void CutlassGroupedGemmWgrad<true, false, cutlass::bfloat16_t, true, false>(
    const NVTETensor*, const NVTETensor*, NVTETensor*, NVTETensor*, float, float, int, cudaStream_t,
    int, int);
template void CutlassGroupedGemmWgrad<true, false, float, true, true>(const NVTETensor*,
                                                                      const NVTETensor*,
                                                                      NVTETensor*, NVTETensor*,
                                                                      float, float, int,
                                                                      cudaStream_t, int, int);
template void CutlassGroupedGemmWgrad<true, false, cutlass::bfloat16_t, true, true>(
    const NVTETensor*, const NVTETensor*, NVTETensor*, NVTETensor*, float, float, int, cudaStream_t,
    int, int);

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

  // Select the CUTLASS collective by device arch: SM100 (Blackwell, CC 10.x) uses the tcgen05
  // Ptr-Array schedule (kSm100=true); SM90 (Hopper) uses the original wgmma Ptr-Array schedule.
  int sm_major = 0;
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
  const bool sm100 = (sm_major == 10);

  auto run = [&](auto tag, auto sm100_tag) {
    using T = decltype(tag);
    constexpr bool S = decltype(sm100_tag)::value;
    if (!transa && !transb) {
      grouped_gemm::CutlassGroupedGemm<false, false, T, S>(
          B, A, D, workspace, alpha, beta, num_gemms, stream, device, math_sm_count);
    } else if (!transb && transa) {
      grouped_gemm::CutlassGroupedGemm<false, true, T, S>(B, A, D, workspace, alpha, beta,
                                                          num_gemms, stream, device, math_sm_count);
    } else if (transb && !transa) {
      grouped_gemm::CutlassGroupedGemm<true, false, T, S>(B, A, D, workspace, alpha, beta,
                                                          num_gemms, stream, device, math_sm_count);
    } else {
      NVTE_ERROR("Layout 'TT' is not supported by cutlass_grouped_gemm.");
    }
  };
  auto dispatch = [&](auto tag) {
    if (sm100) {
      run(tag, std::true_type{});
    } else {
      run(tag, std::false_type{});
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

namespace {

// Zero-initialize empty (K=0) groups (when not accumulating) and forward the non-empty groups to
// CUTLASS. Precondition: the dispatcher (nvte_multi_tensor_gemm) has already validated the BF16 NT
// wgrad contract -- 2D, matching ragged K, uniform hidden/expert, BF16-in / (FP32|BF16)-out -- via
// is_bf16_wgrad_dtype() + is_bf16_wgrad_shape(), so it is not re-checked here.
void collect_bf16_wgrad_nt_groups(const NVTETensor* A, const NVTETensor* B, NVTETensor* D,
                                  int num_gemms, bool accumulate, cudaStream_t stream,
                                  std::vector<NVTETensor>* A_nz, std::vector<NVTETensor>* B_nz,
                                  std::vector<NVTETensor>* D_nz,
                                  transformer_engine::DType* out_dtype) {
  using namespace transformer_engine;
  // hidden/expert/output-dtype are uniform across groups; read them once from group 0.
  const int64_t hidden = convertNVTETensorCheck(A[0])->data.shape[1];
  const int64_t expert = convertNVTETensorCheck(B[0])->data.shape[1];
  *out_dtype = convertNVTETensorCheck(D[0])->data.dtype;
  const size_t elem = (*out_dtype == DType::kFloat32) ? sizeof(float) : sizeof(__nv_bfloat16);

  for (int i = 0; i < num_gemms; ++i) {
    if (convertNVTETensorCheck(A[i])->data.shape[0] == 0) {
      // Empty group: its null A/B pointers would crash TMA descriptor construction, so zero the
      // output (when not accumulating) and exclude it from the launch.
      auto* out = convertNVTETensorCheck(D[i]);
      if (!accumulate && out->data.dptr != nullptr) {
        NVTE_CHECK_CUDA(cudaMemsetAsync(out->data.dptr, 0,
                                        static_cast<size_t>(expert) * hidden * elem, stream));
      }
    } else {
      A_nz->push_back(A[i]);
      B_nz->push_back(B[i]);
      D_nz->push_back(D[i]);
    }
  }
}

}  // namespace

void cutlass_grouped_gemm_varlen_k(const NVTETensor* A, const NVTETensor* B, NVTETensor* D,
                                   int num_gemms, bool transa, bool transb, bool grad,
                                   NVTETensor* workspace, bool accumulate, int device,
                                   int math_sm_count, cudaStream_t stream) {
  using namespace transformer_engine;
  // The kernel hard-codes the NT layout, so assert it: a wrong-layout caller would otherwise
  // mis-compute silently. (Arch / no-epilogue / group-0 dtype are already gated by the
  // dispatcher, and a wrong arch would fail loudly inside the CUTLASS kernel anyway.)
  NVTE_CHECK(!transa && transb && grad,
             "cutlass_grouped_gemm_varlen_k requires NT wgrad layout "
             "(transa=false, transb=true, grad=true).");
  NVTE_CHECK(workspace != nullptr, "cutlass_grouped_gemm_varlen_k requires a non-null workspace.");

  std::vector<NVTETensor> A_nz, B_nz, D_nz;
  A_nz.reserve(num_gemms);
  B_nz.reserve(num_gemms);
  D_nz.reserve(num_gemms);
  DType out_dtype = DType::kFloat32;
  collect_bf16_wgrad_nt_groups(A, B, D, num_gemms, accumulate, stream, &A_nz, &B_nz, &D_nz,
                               &out_dtype);

  // All groups have K=0: outputs are already zero-initialized above, nothing to launch.
  if (A_nz.empty()) return;

  const int n_nz = static_cast<int>(A_nz.size());
  float one = 1.0;
  float zero = 0.0;
  float alpha = one;
  float beta = (accumulate) ? one : zero;

  // NT wgrad: D_i = B_i^T @ A_i. Pass grad_output (outer B) as CUTLASS A (trans_a=true)
  // and input (outer A) as CUTLASS B (trans_b=false). CutlassGroupedGemmWgrad validates
  // the workspace size internally.
  int sm_major = 0;
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
  const bool sm100 = (sm_major == 10);
  // Per-shape SM100 N-tile selection: large average-K -> 256x256 (kBigN=true), else 256x128.
  // 256x256 wins ~+10% at large K but regresses small-K (latency-bound) ~-30%; threshold K>=1536.
  int64_t total_k = 0;
  for (int i = 0; i < n_nz; ++i) {
    total_k += transformer_engine::convertNVTETensorCheck(A_nz[i])->data.shape[0];
  }
  const bool big_n = sm100 && n_nz > 0 && (total_k / n_nz) >= 1536;
  auto dispatch = [&](auto tag) {
    using T = decltype(tag);
    auto launch = [&](auto sm100_tag, auto bign_tag) {
      grouped_gemm::CutlassGroupedGemmWgrad<true, false, T, decltype(sm100_tag)::value,
                                            decltype(bign_tag)::value>(
          B_nz.data(), A_nz.data(), D_nz.data(), workspace, alpha, beta, n_nz, stream, device,
          math_sm_count);
    };
    if (!sm100) {
      launch(std::false_type{}, std::false_type{});
    } else if (big_n) {
      launch(std::true_type{}, std::true_type{});
    } else {
      launch(std::true_type{}, std::false_type{});
    }
  };

  if (out_dtype == DType::kFloat32) {
    dispatch(float{});
  } else {
    dispatch(cutlass::bfloat16_t{});
  }
}

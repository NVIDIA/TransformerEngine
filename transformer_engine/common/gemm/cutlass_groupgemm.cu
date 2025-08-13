#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass_groupgemm.cuh"

namespace grouped_gemm {

cudaDataType_t get_cuda_dtype(const transformer_engine::DType t) {
  using namespace transformer_engine;
  switch (t) {
    case DType::kFloat16:
      return CUDA_R_16F;
    case DType::kFloat32:
      return CUDA_R_32F;
    case DType::kBFloat16:
      return CUDA_R_16BF;
    case DType::kFloat8E4M3:
      return CUDA_R_8F_E4M3;
    case DType::kFloat8E5M2:
      return CUDA_R_8F_E5M2;
    default:
      NVTE_ERROR("Invalid type");
  }
}

// 显式实例化，配合 .cuh 中的 extern template
template void CutlassGroupedGemm<false,false, cutlass::half_t>(bool,bool,const NVTETensor*,const NVTETensor*,NVTETensor*,NVTETensor*,float,float,int,cudaStream_t,int,int);
template void CutlassGroupedGemm<true ,false, cutlass::half_t>(bool,bool,const NVTETensor*,const NVTETensor*,NVTETensor*,NVTETensor*,float,float,int,cudaStream_t,int,int);
template void CutlassGroupedGemm<false,true , cutlass::half_t>(bool,bool,const NVTETensor*,const NVTETensor*,NVTETensor*,NVTETensor*,float,float,int,cudaStream_t,int,int);

template void CutlassGroupedGemm<false,false, cutlass::bfloat16_t>(bool,bool,const NVTETensor*,const NVTETensor*,NVTETensor*,NVTETensor*,float,float,int,cudaStream_t,int,int);
template void CutlassGroupedGemm<true ,false, cutlass::bfloat16_t>(bool,bool,const NVTETensor*,const NVTETensor*,NVTETensor*,NVTETensor*,float,float,int,cudaStream_t,int,int);
template void CutlassGroupedGemm<false,true , cutlass::bfloat16_t>(bool,bool,const NVTETensor*,const NVTETensor*,NVTETensor*,NVTETensor*,float,float,int,cudaStream_t,int,int);

} // namespace grouped_gemm

void nvte_cutlass_grouped_gemm(const NVTETensor* A, const NVTETensor* B, NVTETensor* D,
                               int num_gemms, bool transa, bool transb, bool grad,
                               NVTETensor* workspace, bool accumulate, int device,
                               int math_sm_count, cudaStream_t stream) {
  auto* inputA = transformer_engine::convertNVTETensorCheck(A[0]);
  auto* inputB = transformer_engine::convertNVTETensorCheck(B[0]);

  auto A_type = grouped_gemm::get_cuda_dtype(inputA->data.dtype);
  auto B_type = grouped_gemm::get_cuda_dtype(inputB->data.dtype);
  NVTE_CHECK(A_type == B_type, "A/B dtype mismatch in cutlass_grouped_gemm.");

  float one = 1.0;
  float zero = 0.0;
  float alpha = one;
  float beta = (accumulate) ? one : zero;

  auto dispatch = [&](auto tag) {
    using T = decltype(tag);
    if (!transa && !transb) {
      grouped_gemm::CutlassGroupedGemm<false, false, T>(transb, transa, B, A, D, workspace, alpha,
                                                        beta, num_gemms, stream, device,
                                                        math_sm_count);
    } else if (!transb && transa) {
      grouped_gemm::CutlassGroupedGemm<false, true, T>(transb, transa, B, A, D, workspace, alpha,
                                                       beta, num_gemms, stream, device,
                                                       math_sm_count);
    } else if (transb && !transa) {
      grouped_gemm::CutlassGroupedGemm<true, false, T>(transb, transa, B, A, D, workspace, alpha,
                                                       beta, num_gemms, stream, device,
                                                       math_sm_count);
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
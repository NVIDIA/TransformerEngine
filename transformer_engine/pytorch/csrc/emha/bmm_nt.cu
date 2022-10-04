/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
///////////////////////////////////////////////////////////////////////////////////////////////////

#include "relu_gemm.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
using ReluGemmKernel = typename ReluGemmUniversal<
    T, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone,
    8,  // transposed B operand
    T, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone,
    8,  // transposed A operand
    T, cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<T, 8, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 5,
    cutlass::arch::OpMultiplyAdd>::GemmKernel;

/*
using cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_nt_align8_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    cutlass::half_t, cutlass::layout::ColumnMajor,
cutlass::ComplexTransform::kNone, 8,    // transposed B operand cutlass::half_t,
cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed
A operand cutlass::half_t, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      float,
      float
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    5,
    cutlass::arch::OpMultiplyAdd
>::GemmKernel;

static_assert(std::is_same<ReluGemmKernel,
cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_nt_align8_base>::value);
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

// Define named type
struct cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_nt_align8
    : public ReluGemmKernel<cutlass::half_t> {};
// public cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_nt_align8_base { };

struct cutlass_tensorop_bf16_s16816gemm_bf16_128x128_32x5_nt_align8
    : public ReluGemmKernel<cutlass::bfloat16_t> {};

///////////////////////////////////////////////////////////////////////////////////////////////////

void cuda_bmm_nt(int m, int n, int k, float alpha, half const *A, int lda,
                 int64_t batch_stride_A, half const *B, int ldb,
                 int64_t batch_stride_B, half *C, int ldc,
                 int64_t batch_stride_C, float beta, int batch_count,
                 cudaStream_t stream) {
  using Gemm_nt = cutlass::gemm::device::GemmUniversalAdapter<
      cutlass_tensorop_f16_s16816gemm_f16_128x128_32x5_nt_align8>;
  using Arguments = typename Gemm_nt::Arguments;

  Arguments args{// GemmUniversalMode mode,
                 cutlass::gemm::GemmUniversalMode::kBatched,
                 // GemmCoord problem_size,
                 {m, n, k},
                 // int batch_count,
                 batch_count,
                 // typename EpilogueOutputOp::Params epilogue,
                 {alpha, beta},
                 // void const * ptr_A,
                 A,
                 // void const * ptr_B,
                 B,
                 // void const * ptr_C,
                 C,
                 // void * ptr_D,
                 C,
                 // int64_t batch_stride_A,
                 batch_stride_A,
                 // int64_t batch_stride_B,
                 batch_stride_B,
                 // int64_t batch_stride_C,
                 batch_stride_C,
                 // int64_t batch_stride_D,
                 batch_stride_C,
                 // typename LayoutA::Stride stride_a,
                 lda,
                 // typename LayoutB::Stride stride_b,
                 ldb,
                 // typename LayoutC::Stride stride_c,
                 ldc,
                 // typename LayoutC::Stride stride_d
                 ldc};

  Gemm_nt bmm;
  bmm(args, /*workspace*/ nullptr, stream);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void cuda_bmm_nt(int m, int n, int k, float alpha, nv_bfloat16 const *A,
                 int lda, int64_t batch_stride_A, nv_bfloat16 const *B,
                 int ldb, int64_t batch_stride_B, nv_bfloat16 *C, int ldc,
                 int64_t batch_stride_C, float beta, int batch_count,
                 cudaStream_t stream) {
  using Gemm_nt = cutlass::gemm::device::GemmUniversalAdapter<
      cutlass_tensorop_bf16_s16816gemm_bf16_128x128_32x5_nt_align8>;
  using Arguments = typename Gemm_nt::Arguments;

  Arguments args{// GemmUniversalMode mode,
                 cutlass::gemm::GemmUniversalMode::kBatched,
                 // GemmCoord problem_size,
                 {m, n, k},
                 // int batch_count,
                 batch_count,
                 // typename EpilogueOutputOp::Params epilogue,
                 {alpha, beta},
                 // void const * ptr_A,
                 A,
                 // void const * ptr_B,
                 B,
                 // void const * ptr_C,
                 C,
                 // void * ptr_D,
                 C,
                 // int64_t batch_stride_A,
                 batch_stride_A,
                 // int64_t batch_stride_B,
                 batch_stride_B,
                 // int64_t batch_stride_C,
                 batch_stride_C,
                 // int64_t batch_stride_D,
                 batch_stride_C,
                 // typename LayoutA::Stride stride_a,
                 lda,
                 // typename LayoutB::Stride stride_b,
                 ldb,
                 // typename LayoutC::Stride stride_c,
                 ldc,
                 // typename LayoutC::Stride stride_d
                 ldc};

  Gemm_nt bmm;
  bmm(args, /*workspace*/ nullptr, stream);
}

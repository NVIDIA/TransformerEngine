/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/native/Resize.h>
#include <c10/util/MaybeOwned.h>
#include <torch/extension.h>

void cuda_bmm_nn(int m, int n, int k, float alpha, half const *A, int lda,
                 int64_t batch_stride_A, half const *B, int ldb,
                 int64_t batch_stride_B, half *C, int ldc,
                 int64_t batch_stride_C, float beta, int batch_count,
                 cudaStream_t stream);

void cuda_bmm_nt(int m, int n, int k, float alpha, half const *A, int lda,
                 int64_t batch_stride_A, half const *B, int ldb,
                 int64_t batch_stride_B, half *C, int ldc,
                 int64_t batch_stride_C, float beta, int batch_count,
                 cudaStream_t stream);

void cuda_bmm_nn(int m, int n, int k, float alpha, nv_bfloat16 const *A,
                 int lda, int64_t batch_stride_A, nv_bfloat16 const *B,
                 int ldb, int64_t batch_stride_B, nv_bfloat16 *C, int ldc,
                 int64_t batch_stride_C, float beta, int batch_count,
                 cudaStream_t stream);

void cuda_bmm_nt(int m, int n, int k, float alpha, nv_bfloat16 const *A,
                 int lda, int64_t batch_stride_A, nv_bfloat16 const *B,
                 int ldb, int64_t batch_stride_B, nv_bfloat16 *C, int ldc,
                 int64_t batch_stride_C, float beta, int batch_count,
                 cudaStream_t stream);

void bmm_nn(const at::Tensor &A, const at::Tensor &B, at::Tensor C) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda());
  const auto sizeA = A.sizes();
  const auto sizeB = B.sizes();
  const auto sizeC = C.sizes();
  const auto strideA = A.strides();
  const auto strideB = B.strides();
  const auto strideC = C.strides();
  int batch_count = sizeA[0];
  TORCH_CHECK(batch_count = sizeB[0]);
  TORCH_CHECK(sizeA.size() == 3);
  TORCH_CHECK(sizeB.size() == 3);
  TORCH_CHECK(strideA[2] == 1);
  TORCH_CHECK(strideB[2] == 1);
  TORCH_CHECK(strideC[2] == 1);
  TORCH_CHECK(A.scalar_type() == B.scalar_type());
  TORCH_CHECK(A.scalar_type() == C.scalar_type());

  // C[i]  = A[i]  * B[i]  where A ,B ,C  are RM
  // C[i]' = B[i]' * A[i]' where A',B',C' are CM

  const int m = sizeB[2];
  const int n = sizeA[1];
  const int k = sizeA[2];
  const size_t batch_stride_A = strideB[0];
  const size_t batch_stride_B = strideA[0];
  const size_t batch_stride_C = strideC[0];
  const int lda = strideB[1];
  const int ldb = strideA[1];
  const int ldc = strideC[1];
  TORCH_CHECK(k == sizeB[1]);
  TORCH_CHECK(m == sizeC[2]);
  TORCH_CHECK(n == sizeC[1]);
  TORCH_CHECK(batch_count == sizeC[0]);
  auto opts = A.options();

  auto stream = at::cuda::getCurrentCUDAStream();

  /*
  printf("m=%d\n n=%d\n k=%d\n lda=%d\n batch_stride_A=%d\n ldb=%d\n
  batch_stride_B=%d\n ldc=%d\n batch_stride_C=%d\n batch_count=%d\n", (int)m,
         (int)n,
         (int)k,
         (int)lda,
         (int)batch_stride_A,
         (int)ldb,
         (int)batch_stride_B,
         (int)ldc,
         (int)batch_stride_C,
         (int)batch_count
        );
        */

  if (B.scalar_type() == torch::kFloat16) {
    const half *ptr_A = static_cast<const half *>(B.data_ptr());
    const half *ptr_B = static_cast<const half *>(A.data_ptr());
    half *ptr_C = static_cast<half *>(C.data_ptr());

    cuda_bmm_nn(m, n, k, 1.f, ptr_A, lda, batch_stride_A, ptr_B, ldb,
                batch_stride_B, ptr_C, ldc, batch_stride_C, 0.f, batch_count,
                stream);

  } else if (B.scalar_type() == torch::kBFloat16) {
    const nv_bfloat16 *ptr_A = static_cast<const nv_bfloat16 *>(B.data_ptr());
    const nv_bfloat16 *ptr_B = static_cast<const nv_bfloat16 *>(A.data_ptr());
    nv_bfloat16 *ptr_C = static_cast<nv_bfloat16 *>(C.data_ptr());

    cuda_bmm_nn(m, n, k, 1.f, ptr_A, lda, batch_stride_A, ptr_B, ldb,
                batch_stride_B, ptr_C, ldc, batch_stride_C, 0.f, batch_count,
                stream);
  }
}

void bmm_nt(const at::Tensor &A, const at::Tensor &B, at::Tensor C) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda());
  const auto sizeA = A.sizes();
  const auto sizeB = B.sizes();
  const auto sizeC = C.sizes();
  const auto strideA = A.strides();
  const auto strideB = B.strides();
  const auto strideC = C.strides();
  int batch_count = sizeA[0];
  TORCH_CHECK(batch_count = sizeB[0]);
  TORCH_CHECK(batch_count == sizeC[0]);
  TORCH_CHECK(sizeA.size() == 3);
  TORCH_CHECK(sizeB.size() == 3);
  TORCH_CHECK(strideA[1] == 1);
  TORCH_CHECK(strideB[2] == 1);
  TORCH_CHECK(strideC[2] == 1);

  TORCH_CHECK(A.scalar_type() == B.scalar_type());
  TORCH_CHECK(A.scalar_type() == C.scalar_type());

  // C[i]  = A[i]  * B[i]  where A ,B ,C  are RM
  // C[i]' = B[i]' * A[i]' where A',B',C' are CM

  const int m = sizeB[2];
  const int n = sizeA[1];
  const int k = sizeA[2];
  const size_t batch_stride_A = strideB[0];
  const size_t batch_stride_B = strideA[0];
  const size_t batch_stride_C = strideC[0];
  const int lda = strideB[1];
  const int ldb = strideA[2];
  const int ldc = strideC[1];

  TORCH_CHECK(k == sizeB[1]);
  TORCH_CHECK(m == sizeC[2]);
  TORCH_CHECK(n == sizeC[1]);
  /*
  printf("m=%d\n n=%d\n k=%d\n lda=%d\n batch_stride_A=%d\n ldb=%d\n
  batch_stride_B=%d\n ldc=%d\n batch_stride_C=%d\n batch_count=%d\n", (int)m,
         (int)n,
         (int)k,
         (int)lda,
         (int)batch_stride_A,
         (int)ldb,
         (int)batch_stride_B,
         (int)ldc,
         (int)batch_stride_C,
         (int)batch_count
        );
        */

  auto stream = at::cuda::getCurrentCUDAStream();

  if (B.scalar_type() == torch::kFloat16) {
    const half *ptr_A = static_cast<const half *>(B.data_ptr());
    const half *ptr_B = static_cast<const half *>(A.data_ptr());
    half *ptr_C = static_cast<half *>(C.data_ptr());

    cuda_bmm_nt(m, n, k, 1.f, ptr_A, lda, batch_stride_A, ptr_B, ldb,
                batch_stride_B, ptr_C, ldc, batch_stride_C, 0.f, batch_count,
                stream);

  } else if (B.scalar_type() == torch::kBFloat16) {
    const nv_bfloat16 *ptr_A = static_cast<const nv_bfloat16 *>(B.data_ptr());
    const nv_bfloat16 *ptr_B = static_cast<const nv_bfloat16 *>(A.data_ptr());
    nv_bfloat16 *ptr_C = static_cast<nv_bfloat16 *>(C.data_ptr());
    cuda_bmm_nt(m, n, k, 1.f, ptr_A, lda, batch_stride_A, ptr_B, ldb,
                batch_stride_B, ptr_C, ldc, batch_stride_C, 0.f, batch_count,
                stream);
  }
}

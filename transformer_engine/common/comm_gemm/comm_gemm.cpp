/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/comm_gemm.h"

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <cal.h>
#include <cublasmp.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nvshmem.h>

#include "../common.h"
#include "../util/logging.h"

using namespace transformer_engine;

namespace {

// TODO: log warnings on failures of the *Destroy calls below, once TE has such ability.
// For now, just silently ignoring the errors, since the only diag available in TE is throwing
// exceptions, but these calls will typically be made from destructors, so cannot throw.

using CalComm = std::unique_ptr<std::remove_pointer_t<cal_comm_t>, decltype(&cal_comm_destroy)>;
using CudaStream =
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)>;
using CudaEvent =
    std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, decltype(&cudaEventDestroy)>;
using CublasMp =
    std::unique_ptr<std::remove_pointer_t<cublasMpHandle_t>, decltype(&cublasMpDestroy)>;
using CublasMpGrid =
    std::unique_ptr<std::remove_pointer_t<cublasMpGrid_t>, decltype(&cublasMpGridDestroy)>;
using CublasMpMatrixDesc = std::unique_ptr<std::remove_pointer_t<cublasMpMatrixDescriptor_t>,
                                           decltype(&cublasMpMatrixDescriptorDestroy)>;
using CublasMpMatmulDesc = std::unique_ptr<std::remove_pointer_t<cublasMpMatmulDescriptor_t>,
                                           decltype(&cublasMpMatmulDescriptorDestroy)>;

calError_t allgather(void* src_buf, void* recv_buf, size_t size, void* /* data */, void** request) {
  MPI_Request req{};
  int err = MPI_Iallgather(src_buf, size, MPI_BYTE, recv_buf, size, MPI_BYTE, MPI_COMM_WORLD, &req);
  if (err != MPI_SUCCESS) {
    return CAL_ERROR;
  }
  *request = (void*)req;
  return CAL_OK;
}

calError_t request_test(void* request) {
  MPI_Request req = (MPI_Request)request;
  int completed;
  int err = MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
  if (err != MPI_SUCCESS) {
    return CAL_ERROR;
  }
  return completed ? CAL_OK : CAL_ERROR_INPROGRESS;
}

calError_t request_free(void* request) { return CAL_OK; }
}  // namespace

struct CommGemmCtx {
  int64_t nranks;
  int64_t rank;
  CalComm cal_comm;
  CudaStream stream;
  CudaEvent event;
  CublasMp cublas_mp;
  CublasMpGrid grid_col_major;
  CublasMpGrid grid_row_major;
  CublasMpMatrixDesc a_desc;
  CublasMpMatrixDesc b_desc;
  CublasMpMatrixDesc d_desc;
  CublasMpMatmulDesc matmul_desc;
  void* workspace;
  size_t workspace_size;
};

namespace {

int64_t block_size(CommGemmCtx* ctx, int64_t global_size) {
    // Use non-cyclic layout to maximize opportunity for comm overlap.
    return (global_size + ctx->nranks - 1) / ctx->nranks;
}

void cublasmp_gemm(CommGemmCtx* ctx, cublasMpMatmulAlgoType_t algo, int64_t m, int64_t n, int64_t k,
                   const Tensor* a, const Tensor* b, const Tensor* d, const Tensor* bias,
                   const Tensor* pre_act_out, bool transa, bool transb, bool grad, bool accumulate,
                   int comm_sm_count, cudaStream_t main_stream) {
  const auto a0 = a->flat_first_dim();
  const auto a1 = a->flat_last_dim();
  const auto b0 = b->flat_first_dim();
  const auto b1 = b->flat_last_dim();
  const auto d0 = d->flat_first_dim();
  const auto d1 = d->flat_last_dim();

  if (transa) {
    NVTE_CHECK(a1 == k);
    NVTE_CHECK_CUBLASMP(cublasMpMatrixDescriptorInit(k, m, k, block_size(ctx, m), 0, 0, k,
                                                     get_cuda_dtype(a->dtype()),
                                                     ctx->grid_row_major.get(), ctx->a_desc.get()));
  } else {
    NVTE_CHECK(false);
    NVTE_CHECK_CUBLASMP(cublasMpMatrixDescriptorInit(m, k, a1, a0, 0, 0, a1,
                                                     get_cuda_dtype(a->dtype()),
                                                     ctx->grid_row_major.get(), ctx->a_desc.get()));
  }
  if (transb) {
    NVTE_CHECK(false);
    NVTE_CHECK_CUBLASMP(cublasMpMatrixDescriptorInit(n, k, b0, b1, 0, 0, b0,
                                                     get_cuda_dtype(b->dtype()),
                                                     ctx->grid_row_major.get(), ctx->b_desc.get()));
  } else {
    NVTE_CHECK(b1 == k);
    NVTE_CHECK_CUBLASMP(cublasMpMatrixDescriptorInit(k, n, k, block_size(ctx, n), 0, 0, k,
                                                     get_cuda_dtype(b->dtype()),
                                                     ctx->grid_row_major.get(), ctx->b_desc.get()));
  }
  NVTE_CHECK(d0 == n);
  NVTE_CHECK_CUBLASMP(cublasMpMatrixDescriptorInit(
      m, n, block_size(ctx, m), block_size(ctx, n), 0, 0, block_size(ctx, m),
      get_cuda_dtype(d->dtype()), ctx->grid_col_major.get(), ctx->d_desc.get()));

  const cublasOperation_t trans_a = transa ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t trans_b = transb ? CUBLAS_OP_T : CUBLAS_OP_N;
  NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
      ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSA, &trans_a, sizeof trans_a));
  NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
      ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_TRANSB, &trans_b, sizeof trans_b));
  NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorAttributeSet(
      ctx->matmul_desc.get(), CUBLASMP_MATMUL_DESCRIPTOR_ATTRIBUTE_ALGO_TYPE, &algo, sizeof algo));

  NVTE_CHECK_CUBLASMP(cublasMpStreamSet(ctx->cublas_mp.get(), main_stream));

  size_t wrksp_size_device{};
  size_t wrksp_size_host{};

  float alpha = 1.0;
  float beta = accumulate ? 1.0 : 0.0;
  std::tuple args{ctx->cublas_mp.get(),
                  ctx->matmul_desc.get(),
                  m,
                  n,
                  k,
                  &alpha,
                  a->data.dptr,
                  1,
                  1,
                  ctx->a_desc.get(),
                  b->data.dptr,
                  1,
                  1,
                  ctx->b_desc.get(),
                  &beta,
                  d->data.dptr,
                  1,
                  1,
                  ctx->d_desc.get(),
                  d->data.dptr,
                  1,
                  1,
                  ctx->d_desc.get()};
  NVTE_CHECK_CUBLASMP(
      std::apply(cublasMpMatmul_bufferSize,
                 std::tuple_cat(args, std::tuple{&wrksp_size_device, &wrksp_size_host})));

  std::vector<uint8_t> workspace_host(wrksp_size_host);
  if (ctx->workspace_size < wrksp_size_device) {
    nvshmem_free(ctx->workspace);
    ctx->workspace = nvshmem_malloc(wrksp_size_device);
    ctx->workspace_size = wrksp_size_device;
  }

  NVTE_CHECK_CUBLASMP(
      std::apply(cublasMpMatmul,
                 std::tuple_cat(args, std::tuple{ctx->workspace, ctx->workspace_size,
                                                 workspace_host.data(), workspace_host.size()})));

  NVTE_CHECK_CUDA(cudaEventRecord(ctx->event.get(), main_stream));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(ctx->stream.get(), ctx->event.get(), 0));
}

}  // namespace

CommGemmCtx* nvte_comm_gemm_ctx_create(int nranks, int rank, int local_device) {
  cal_comm_create_params_t params{
      .allgather = allgather,
      .req_test = request_test,
      .req_free = request_free,
      .nranks = nranks,
      .rank = rank,
      .local_device = local_device,
  };
  cal_comm_t cal_comm_raw{};
  NVTE_CHECK_CAL(cal_comm_create(params, &cal_comm_raw));
  CalComm cal_comm(cal_comm_raw, cal_comm_destroy);

  cudaStream_t stream_raw{};
  NVTE_CHECK_CUDA(cudaStreamCreate(&stream_raw));
  CudaStream stream(stream_raw, cudaStreamDestroy);
  cudaEvent_t event_raw{};
  NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&event_raw, cudaEventDisableTiming));
  CudaEvent event(event_raw, cudaEventDestroy);
  cublasMpHandle_t cublasmp_raw{};
  NVTE_CHECK_CUBLASMP(cublasMpCreate(&cublasmp_raw, stream.get()));
  CublasMp cublas_mp(cublasmp_raw, cublasMpDestroy);

  cublasMpGrid_t col_major_raw{};
  NVTE_CHECK_CUBLASMP(cublasMpGridCreate(params.nranks, 1, CUBLASMP_GRID_LAYOUT_COL_MAJOR,
                                         cal_comm.get(), &col_major_raw));
  CublasMpGrid col_major(col_major_raw, cublasMpGridDestroy);
  cublasMpGrid_t row_major_raw{};
  NVTE_CHECK_CUBLASMP(cublasMpGridCreate(1, params.nranks, CUBLASMP_GRID_LAYOUT_ROW_MAJOR,
                                         cal_comm.get(), &row_major_raw));
  CublasMpGrid row_major(row_major_raw, cublasMpGridDestroy);

  cublasMpMatrixDescriptor_t raw{};
  NVTE_CHECK_CUBLASMP(
      cublasMpMatrixDescriptorCreate(1, 1, 1, 1, 0, 0, 1, CUDA_R_16F, row_major.get(), &raw));
  CublasMpMatrixDesc a_desc(raw, cublasMpMatrixDescriptorDestroy);
  NVTE_CHECK_CUBLASMP(
      cublasMpMatrixDescriptorCreate(1, 1, 1, 1, 0, 0, 1, CUDA_R_16F, row_major.get(), &raw));
  CublasMpMatrixDesc b_desc(raw, cublasMpMatrixDescriptorDestroy);
  NVTE_CHECK_CUBLASMP(
      cublasMpMatrixDescriptorCreate(1, 1, 1, 1, 0, 0, 1, CUDA_R_16F, row_major.get(), &raw));
  CublasMpMatrixDesc d_desc(raw, cublasMpMatrixDescriptorDestroy);

  cublasMpMatmulDescriptor_t matmul_raw{};
  NVTE_CHECK_CUBLASMP(cublasMpMatmulDescriptorCreate(&matmul_raw, CUBLAS_COMPUTE_32F));
  CublasMpMatmulDesc matmul_desc(matmul_raw, cublasMpMatmulDescriptorDestroy);

  return new CommGemmCtx{
      .nranks = nranks,
      .rank = rank,
      .cal_comm = std::move(cal_comm),
      .stream = std::move(stream),
      .event = std::move(event),
      .cublas_mp = std::move(cublas_mp),
      .grid_col_major = std::move(col_major),
      .grid_row_major = std::move(row_major),
      .a_desc = std::move(a_desc),
      .b_desc = std::move(b_desc),
      .d_desc = std::move(d_desc),
      .matmul_desc = std::move(matmul_desc),
  };
}

void nvte_comm_gemm_ctx_destroy(CommGemmCtx* ctx) {
    nvshmemx_sync_all_on_stream(ctx->stream.get());
    NVTE_CHECK_CAL(cal_comm_barrier(ctx->cal_comm.get(), ctx->stream.get()));
    delete ctx;
}

void nvte_comm_gemm(CommGemmCtx* ctx, int64_t m, int64_t n, int64_t k, const NVTETensor a,
                    const NVTETensor b, const NVTETensor d, const NVTETensor bias,
                    const NVTETensor pre_act_out, bool transa, bool transb, bool grad,
                    bool accumulate, int comm_sm_count, cudaStream_t main_stream) {
  auto ta = static_cast<const Tensor*>(a);
  auto tb = static_cast<const Tensor*>(b);
  auto td = static_cast<const Tensor*>(d);
  auto tbias = static_cast<const Tensor*>(bias);
  auto tpre_act_out = static_cast<const Tensor*>(pre_act_out);
  cublasmp_gemm(ctx, CUBLASMP_MATMUL_ALGO_TYPE_SPLIT_P2P, m, n, k, ta, tb, td, tbias, tpre_act_out,
                transa, transb, grad, accumulate, comm_sm_count, main_stream);
}

int64_t nvte_comm_gemm_numroc(CommGemmCtx* ctx, int64_t global_size) {
  return cublasMpNumroc(global_size, block_size(ctx, global_size), ctx->rank, 0, ctx->nranks);
}

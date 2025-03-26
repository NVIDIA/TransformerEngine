/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/comm_gemm.h"

#include <memory>
#include <type_traits>
#include <utility>

#include <cal.h>
#include <cublasmp.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include "../util/logging.h"

using CalComm = std::unique_ptr<std::remove_pointer_t<cal_comm_t>, decltype(&cal_comm_destroy)>;
using CudaStream = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)>;
using CublasMp = std::unique_ptr<std::remove_pointer_t<cublasMpHandle_t>, decltype(&cublasMpDestroy)>;
using CublasMpGrid = std::unique_ptr<std::remove_pointer_t<cublasMpGrid_t>, decltype(&cublasMpGridDestroy)>;

struct CommGemmCtx {
  CalComm cal_comm;
  CudaStream stream;
  CublasMp cublas_mp;
  CublasMpGrid grid_col_major;
  CublasMpGrid grid_row_major;
};

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

  return new CommGemmCtx{
    .cal_comm = std::move(cal_comm),
    .stream = std::move(stream),
    .cublas_mp = std::move(cublas_mp),
    .grid_col_major = std::move(col_major),
    .grid_row_major = std::move(row_major),
  };
}

void nvte_comm_gemm_ctx_destroy(CommGemmCtx* ctx) noexcept { delete ctx; }

void nvte_comm_gemm(CommGemmCtx* ctx, const NVTETensor a, const NVTETensor b, NVTETensor d,
                    const NVTETensor bias, NVTETensor pre_gelu_out, bool transa, bool transb,
                    bool grad, bool accumulate, int comm_sm_count) {
  // TODO:
}

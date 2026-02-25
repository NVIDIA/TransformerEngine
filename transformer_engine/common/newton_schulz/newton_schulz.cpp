/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/newton_schulz.h"

#include <cusolverMp.h>
#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "../common.h"
#include "../util/logging.h"

using namespace transformer_engine;

// RAII wrapper types for cuSolverMp handles (outside anonymous namespace because
// CusolverMpHandle and CusolverMpGrid are used in the NVTECusolverMpCtx struct)

struct CusolverMpHandleDeleter {
  void operator()(cusolverMpHandle_t handle) const { cusolverMpDestroy(handle); }
};
using CusolverMpHandle = std::unique_ptr<std::remove_pointer_t<cusolverMpHandle_t>,
                                         CusolverMpHandleDeleter>;

struct CusolverMpGridDeleter {
  void operator()(cusolverMpGrid_t grid) const { cusolverMpDestroyGrid(grid); }
};
using CusolverMpGrid = std::unique_ptr<std::remove_pointer_t<cusolverMpGrid_t>,
                                       CusolverMpGridDeleter>;

namespace {

struct CusolverMpMatrixDescDeleter {
  void operator()(cusolverMpMatrixDescriptor_t desc) const { cusolverMpDestroyMatrixDesc(desc); }
};
using CusolverMpMatrixDesc = std::unique_ptr<std::remove_pointer_t<cusolverMpMatrixDescriptor_t>,
                                             CusolverMpMatrixDescDeleter>;

struct CusolverMpNSDescDeleter {
  void operator()(cusolverMpNewtonSchulzDescriptor_t desc) const {
    cusolverMpNewtonSchulzDescriptorDestroy(desc);
  }
};
using CusolverMpNSDesc = std::unique_ptr<std::remove_pointer_t<cusolverMpNewtonSchulzDescriptor_t>,
                                         CusolverMpNSDescDeleter>;

CusolverMpHandle MakeCusolverMpHandle(int device_id, cudaStream_t stream) {
  cusolverMpHandle_t raw{};
  NVTE_CHECK_CUSOLVERMP(cusolverMpCreate(&raw, device_id, stream));
  return CusolverMpHandle(raw);
}

CusolverMpGrid MakeCusolverMpGrid(cusolverMpHandle_t handle, ncclComm_t comm,
                                   int32_t nprow, int32_t npcol,
                                   cusolverMpGridMapping_t mapping) {
  cusolverMpGrid_t raw{};
  NVTE_CHECK_CUSOLVERMP(cusolverMpCreateDeviceGrid(handle, &raw, comm, nprow, npcol, mapping));
  return CusolverMpGrid(raw);
}

CusolverMpMatrixDesc MakeCusolverMpMatrixDesc(cusolverMpGrid_t grid, cudaDataType_t dtype,
                                               int64_t m, int64_t n, int64_t mb, int64_t nb,
                                               uint32_t rsrc, uint32_t csrc, int64_t lld) {
  cusolverMpMatrixDescriptor_t raw{};
  NVTE_CHECK_CUSOLVERMP(
      cusolverMpCreateMatrixDesc(&raw, grid, dtype, m, n, mb, nb, rsrc, csrc, lld));
  return CusolverMpMatrixDesc(raw);
}

CusolverMpNSDesc MakeCusolverMpNSDesc() {
  cusolverMpNewtonSchulzDescriptor_t raw{};
  NVTE_CHECK_CUSOLVERMP(cusolverMpNewtonSchulzDescriptorCreate(&raw));
  return CusolverMpNSDesc(raw);
}

}  // namespace

struct NVTECusolverMpCtx {
  int64_t nranks;
  int64_t rank;
  cudaStream_t stream;
  cudaEvent_t in_ready;
  cudaEvent_t out_ready;
  CusolverMpHandle handle;
  CusolverMpGrid grid;
  void* workspace;
  size_t workspace_size;
};

NVTECusolverMpCtx* nvte_cusolvermp_ctx_create(ncclComm_t comm, int nranks, int rank) {
  NVTE_API_CALL(nvte_cusolvermp_ctx_create);
  int device_id{};
  NVTE_CHECK_CUDA(cudaGetDevice(&device_id));

  cudaStream_t stream{};
  NVTE_CHECK_CUDA(cudaStreamCreate(&stream));

  cudaEvent_t in_ready{};
  NVTE_CHECK_CUDA(cudaEventCreate(&in_ready));
  cudaEvent_t out_ready{};
  NVTE_CHECK_CUDA(cudaEventCreate(&out_ready));

  auto handle = MakeCusolverMpHandle(device_id, stream);
  auto grid = MakeCusolverMpGrid(handle.get(), comm, nranks, 1,
                                  CUSOLVERMP_GRID_MAPPING_COL_MAJOR);

  return new NVTECusolverMpCtx{
      .nranks = nranks,
      .rank = rank,
      .stream = stream,
      .in_ready = in_ready,
      .out_ready = out_ready,
      .handle = std::move(handle),
      .grid = std::move(grid),
      .workspace = nullptr,
      .workspace_size = 0,
  };
}

void nvte_cusolvermp_ctx_destroy(NVTECusolverMpCtx* ctx) {
  NVTE_API_CALL(nvte_cusolvermp_ctx_destroy);
  if (ctx->workspace) {
    cudaFree(ctx->workspace);
  }
  // Destroy handle and grid before the stream they depend on
  ctx->handle.reset();
  ctx->grid.reset();
  cudaEventDestroy(ctx->in_ready);
  cudaEventDestroy(ctx->out_ready);
  cudaStreamDestroy(ctx->stream);
  delete ctx;
}

void nvte_newton_schulz(NVTECusolverMpCtx* ctx, int64_t m, int64_t n, NVTETensor x,
                        int64_t num_iterations, const float* coefficients,
                        int64_t num_coefficients, cudaStream_t caller_stream) {
  NVTE_API_CALL(nvte_newton_schulz);
  const auto* t = convertNVTETensorCheck(x);

  // Make the internal stream wait for the caller's stream so that
  // the input tensor is ready before cuSolverMp reads it.
  NVTE_CHECK_CUDA(cudaEventRecord(ctx->in_ready, caller_stream));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(ctx->stream, ctx->in_ready));

  // Block size for ScaLAPACK-style distribution
  const int64_t mb = (m + ctx->nranks - 1) / ctx->nranks;
  const int64_t nb = n;

  // Compute local leading dimension
  const int64_t local_rows = cusolverMpNUMROC(m, mb, ctx->rank, 0, ctx->nranks);
  const int64_t lld = std::max(local_rows, static_cast<int64_t>(1));

  const cudaDataType_t cuda_dtype = get_cuda_dtype(t->dtype());

  // Create matrix descriptor
  auto mat_desc = MakeCusolverMpMatrixDesc(ctx->grid.get(), cuda_dtype, m, n, mb, nb, 0, 0, lld);

  // Create Newton-Schulz descriptor
  auto ns_desc = MakeCusolverMpNSDesc();

  // Query workspace sizes
  size_t wrksp_size_device = 0;
  size_t wrksp_size_host = 0;
  NVTE_CHECK_CUSOLVERMP(cusolverMpNewtonSchulz_bufferSize(
      ctx->handle.get(), ns_desc.get(), m, n, t->data.dptr, 1, 1, mat_desc.get(), num_iterations,
      coefficients, CUDA_R_32F, &wrksp_size_device, &wrksp_size_host));

  // Allocate/grow device workspace
  if (ctx->workspace_size < wrksp_size_device) {
    if (ctx->workspace) {
      NVTE_CHECK_CUDA(cudaFree(ctx->workspace));
    }
    NVTE_CHECK_CUDA(cudaMalloc(&ctx->workspace, wrksp_size_device));
    ctx->workspace_size = wrksp_size_device;
  }

  // Allocate host workspace
  std::vector<uint8_t> workspace_host(wrksp_size_host);

  // Execute Newton-Schulz
  NVTE_CHECK_CUSOLVERMP(cusolverMpNewtonSchulz(
      ctx->handle.get(), ns_desc.get(), m, n, t->data.dptr, 1, 1, mat_desc.get(), num_iterations,
      coefficients, CUDA_R_32F, ctx->workspace, ctx->workspace_size, workspace_host.data(),
      workspace_host.size(), nullptr));

  // Make the caller's stream wait for the internal stream so that
  // the output tensor is ready before the caller uses it.
  NVTE_CHECK_CUDA(cudaEventRecord(ctx->out_ready, ctx->stream));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(caller_stream, ctx->out_ready));
}

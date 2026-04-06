/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/newton_schulz.h"

#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "../common.h"
#include "../util/logging.h"

#ifdef NVTE_WITH_CUSOLVERMP

#include <cusolverMp.h>

using namespace transformer_engine;

namespace {

struct CudaStreamDeleter {
  void operator()(std::remove_pointer_t<cudaStream_t>* stream) const { cudaStreamDestroy(stream); }
};
using CudaStream = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, CudaStreamDeleter>;

struct CudaEventDeleter {
  void operator()(std::remove_pointer_t<cudaEvent_t>* event) const { cudaEventDestroy(event); }
};
using CudaEvent = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, CudaEventDeleter>;

struct CusolverMpHandleDeleter {
  void operator()(cusolverMpHandle_t handle) const { cusolverMpDestroy(handle); }
};
using CusolverMpHandle =
    std::unique_ptr<std::remove_pointer_t<cusolverMpHandle_t>, CusolverMpHandleDeleter>;

struct CusolverMpGridDeleter {
  void operator()(cusolverMpGrid_t grid) const { cusolverMpDestroyGrid(grid); }
};
using CusolverMpGrid =
    std::unique_ptr<std::remove_pointer_t<cusolverMpGrid_t>, CusolverMpGridDeleter>;

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

CusolverMpGrid MakeCusolverMpGrid(cusolverMpHandle_t handle, ncclComm_t comm, int32_t nprow,
                                  int32_t npcol, cusolverMpGridMapping_t mapping) {
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

CudaStream MakeCudaStream() {
  cudaStream_t raw{};
  NVTE_CHECK_CUDA(cudaStreamCreate(&raw));
  return CudaStream(raw);
}

CudaEvent MakeCudaEvent() {
  cudaEvent_t raw{};
  NVTE_CHECK_CUDA(cudaEventCreate(&raw));
  return CudaEvent(raw);
}

}  // namespace

struct NVTECusolverMpCtx {
  int64_t nranks;
  int64_t rank;
  CudaStream stream;
  CudaEvent in_ready;
  CudaEvent out_ready;
  CusolverMpHandle handle;
  CusolverMpGrid grid;
  void* workspace;
  size_t workspace_size;
  bool workspace_registered;
};

namespace {

void FreeWorkspace(NVTECusolverMpCtx* ctx) {
  if (ctx->workspace == nullptr) {
    return;
  }
  if (ctx->workspace_registered) {
    NVTE_CHECK_CUSOLVERMP(cusolverMpBufferDeregister(ctx->grid.get(), ctx->workspace));
    NVTE_CHECK_NCCL(ncclMemFree(ctx->workspace));
  } else {
    NVTE_CHECK_CUDA(cudaFree(ctx->workspace));
  }
  ctx->workspace = nullptr;
  ctx->workspace_size = 0;
  ctx->workspace_registered = false;
}

}  // namespace

NVTECusolverMpCtx* nvte_cusolvermp_ctx_create(ncclComm_t comm, int nranks, int rank) {
  NVTE_API_CALL(nvte_cusolvermp_ctx_create);
  int device_id{};
  NVTE_CHECK_CUDA(cudaGetDevice(&device_id));

  auto stream = MakeCudaStream();
  auto in_ready = MakeCudaEvent();
  auto out_ready = MakeCudaEvent();

  auto handle = MakeCusolverMpHandle(device_id, stream.get());
  auto grid = MakeCusolverMpGrid(handle.get(), comm, nranks, 1, CUSOLVERMP_GRID_MAPPING_COL_MAJOR);

  return new NVTECusolverMpCtx{
      nranks,
      rank,
      std::move(stream),
      std::move(in_ready),
      std::move(out_ready),
      std::move(handle),
      std::move(grid),
      nullptr,
      0,
      false,
  };
}

void nvte_cusolvermp_ctx_destroy(NVTECusolverMpCtx* ctx) {
  NVTE_API_CALL(nvte_cusolvermp_ctx_destroy);
  FreeWorkspace(ctx);
  // Destroy handle and grid before the stream they depend on
  ctx->grid.reset();
  ctx->handle.reset();
  delete ctx;
}

void nvte_newton_schulz(NVTECusolverMpCtx* ctx, int64_t m, int64_t n, NVTETensor x,
                        int64_t num_iterations, const float* coefficients, int64_t num_coefficients,
                        cudaStream_t caller_stream) {
  NVTE_API_CALL(nvte_newton_schulz);
  NVTE_CHECK(num_coefficients == num_iterations * 3, num_iterations, " iterations require ",
             num_iterations * 3, " coefficients, but ", num_coefficients, " are passed");
  const auto* t = convertNVTETensorCheck(x);

  // Make the internal stream wait for the caller's stream so that
  // the input tensor is ready before cuSolverMp reads it.
  NVTE_CHECK_CUDA(cudaEventRecord(ctx->in_ready.get(), caller_stream));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(ctx->stream.get(), ctx->in_ready.get()));

  // Block size for ScaLAPACK-style distribution
  const int64_t mb = m;
  const int64_t nb = (n + ctx->nranks - 1) / ctx->nranks;

  // Compute local leading dimension
  const int64_t local_cols = cusolverMpNUMROC(n, nb, ctx->rank, 0, ctx->nranks);
  NVTE_CHECK(t->shape().size() == 2, "Shape size:", t->shape().size());
  NVTE_CHECK(t->shape()[1] == local_cols, "Tensor cols:", t->shape()[1], "Local cols:", local_cols);
  const int64_t lld = std::max(local_cols, static_cast<int64_t>(1));

  const cudaDataType_t cuda_dtype = get_cuda_dtype(t->dtype());

  // Create matrix descriptor
  auto mat_desc = MakeCusolverMpMatrixDesc(ctx->grid.get(), cuda_dtype, n, m, nb, mb, 0, 0, lld);

  // Create Newton-Schulz descriptor
  auto ns_desc = MakeCusolverMpNSDesc();

  // Query workspace sizes
  size_t wrksp_size_device = 0;
  size_t wrksp_size_host = 0;
  NVTE_CHECK_CUSOLVERMP(cusolverMpNewtonSchulz_bufferSize(
      ctx->handle.get(), ns_desc.get(), n, m, t->data.dptr, 1, 1, mat_desc.get(), num_iterations,
      coefficients, CUDA_R_32F, &wrksp_size_device, &wrksp_size_host));

  // Allocate/grow device workspace
  if (ctx->workspace_size < wrksp_size_device) {
    FreeWorkspace(ctx);

    void* workspace = nullptr;
    bool workspace_registered = false;

    if (ncclMemAlloc(&workspace, wrksp_size_device) == ncclSuccess) {
      if (cusolverMpBufferRegister(ctx->grid.get(), workspace, wrksp_size_device) ==
          CUSOLVER_STATUS_SUCCESS) {
        workspace_registered = true;
      } else {
        NVTE_CHECK_NCCL(ncclMemFree(workspace));
        workspace = nullptr;
      }
    }

    if (workspace == nullptr) {
      NVTE_CHECK_CUDA(cudaMalloc(&workspace, wrksp_size_device));
    }

    ctx->workspace = workspace;
    ctx->workspace_size = wrksp_size_device;
    ctx->workspace_registered = workspace_registered;
  }

  // Allocate host workspace
  std::vector<uint8_t> workspace_host(wrksp_size_host);

  // Execute Newton-Schulz
  NVTE_CHECK_CUSOLVERMP(cusolverMpNewtonSchulz(
      ctx->handle.get(), ns_desc.get(), n, m, t->data.dptr, 1, 1, mat_desc.get(), num_iterations,
      coefficients, CUDA_R_32F, ctx->workspace, ctx->workspace_size, workspace_host.data(),
      workspace_host.size(), nullptr));

  // Make the caller's stream wait for the internal stream so that
  // the output tensor is ready before the caller uses it.
  NVTE_CHECK_CUDA(cudaEventRecord(ctx->out_ready.get(), ctx->stream.get()));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(caller_stream, ctx->out_ready.get()));
}

#else  // NVTE_WITH_CUSOLVERMP

struct NVTECusolverMpCtx {};

NVTECusolverMpCtx* nvte_cusolvermp_ctx_create(ncclComm_t comm, int nranks, int rank) {
  NVTE_ERROR("Transformer Engine has not been built with cuSolverMp support.");
}

void nvte_cusolvermp_ctx_destroy(NVTECusolverMpCtx* ctx) {
  NVTE_ERROR("Transformer Engine has not been built with cuSolverMp support.");
}

void nvte_newton_schulz(NVTECusolverMpCtx* ctx, int64_t m, int64_t n, NVTETensor x,
                        int64_t num_iterations, const float* coefficients, int64_t num_coefficients,
                        cudaStream_t caller_stream) {
  NVTE_ERROR("Transformer Engine has not been built with cuSolverMp support.");
}

#endif  // NVTE_WITH_CUSOLVERMP

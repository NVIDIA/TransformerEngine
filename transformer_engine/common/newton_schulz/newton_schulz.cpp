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

namespace {

template <typename HandlePtr, typename CreateFn, typename DestroyFn, typename... Args>
auto CreateWithCudaCheck(CreateFn create_fn, DestroyFn destroy_fn, Args&&... args) {
  using Handle = std::remove_pointer_t<HandlePtr>;
  HandlePtr raw{};
  NVTE_CHECK_CUDA(create_fn(&raw, std::forward<Args>(args)...));
  return std::unique_ptr<Handle, DestroyFn>(raw, destroy_fn);
}

using CudaStream =
    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, decltype(&cudaStreamDestroy)>;

CudaStream CudaStreamCreate() {
  return CreateWithCudaCheck<cudaStream_t>(cudaStreamCreate, cudaStreamDestroy);
}

template <bool raw_last, typename HandlePtr, typename CreateFn, typename DestroyFn,
          typename... Args>
auto CreateWithCusolverMpCheck(CreateFn create_fn, DestroyFn destroy_fn, Args&&... args) {
  using Handle = std::remove_pointer_t<HandlePtr>;
  HandlePtr raw{};
  if constexpr (raw_last) {
    NVTE_CHECK_CUSOLVERMP(create_fn(std::forward<Args>(args)..., &raw));
  } else {
    NVTE_CHECK_CUSOLVERMP(create_fn(&raw, std::forward<Args>(args)...));
  }
  return std::unique_ptr<Handle, DestroyFn>(raw, destroy_fn);
}

using CusolverMp =
    std::unique_ptr<std::remove_pointer_t<cusolverMpHandle_t>, decltype(&cusolverMpDestroy)>;

CusolverMp CusolverMpCreate(cudaStream_t stream) {
  return CreateWithCusolverMpCheck<false, cusolverMpHandle_t>(cusolverMpCreate, cusolverMpDestroy,
                                                              stream);
}

using CusolverMpGrid =
    std::unique_ptr<std::remove_pointer_t<cusolverMpGrid_t>, decltype(&cusolverMpDestroyGrid)>;

CusolverMpGrid CusolverMpGridCreate(int64_t nprow, int64_t npcol,
                                     cusolverMpGridLayout_t layout, ncclComm_t comm) {
  return CreateWithCusolverMpCheck<true, cusolverMpGrid_t>(
      cusolverMpCreateDeviceGrid, cusolverMpDestroyGrid, nprow, npcol, layout, comm);
}

using CusolverMpMatrixDesc =
    std::unique_ptr<std::remove_pointer_t<cusolverMpMatrixDescriptor_t>,
                    decltype(&cusolverMpDestroyMatrixDesc)>;

CusolverMpMatrixDesc CusolverMpMatrixDescCreate(int64_t m, int64_t n, int64_t mb, int64_t nb,
                                                 int64_t rsrc, int64_t csrc, int64_t lld,
                                                 cudaDataType_t type, cusolverMpGrid_t grid) {
  return CreateWithCusolverMpCheck<true, cusolverMpMatrixDescriptor_t>(
      cusolverMpCreateMatrixDesc, cusolverMpDestroyMatrixDesc, m, n, mb, nb, rsrc, csrc, lld, type,
      grid);
}

using CusolverMpNSDesc =
    std::unique_ptr<std::remove_pointer_t<cusolverMpNewtonSchulzDescriptor_t>,
                    decltype(&cusolverMpNewtonSchulzDescriptorDestroy)>;

CusolverMpNSDesc CusolverMpNSDescCreate(int64_t num_iterations, const float* coefficients,
                                         int64_t num_coefficients) {
  return CreateWithCusolverMpCheck<false, cusolverMpNewtonSchulzDescriptor_t>(
      cusolverMpNewtonSchulzDescriptorCreate, cusolverMpNewtonSchulzDescriptorDestroy,
      num_iterations, coefficients, num_coefficients);
}

}  // namespace

struct NVTECusolverMpCtx {
  int64_t nranks;
  int64_t rank;
  ncclComm_t comm;
  CudaStream stream;
  CusolverMp cusolver_mp;
  CusolverMpGrid grid;
  void* workspace;
  size_t workspace_size;
};

NVTECusolverMpCtx* nvte_cusolvermp_ctx_create(ncclComm_t comm, int nranks, int rank) {
  NVTE_API_CALL(nvte_cusolvermp_ctx_create);
  auto stream = CudaStreamCreate();
  auto cusolver_mp = CusolverMpCreate(stream.get());

  // 1D row partition: nranks x 1, column-major
  auto grid =
      CusolverMpGridCreate(nranks, 1, CUSOLVERMP_GRID_LAYOUT_COL_MAJOR, comm);

  return new NVTECusolverMpCtx{
      .nranks = nranks,
      .rank = rank,
      .comm = comm,
      .stream = std::move(stream),
      .cusolver_mp = std::move(cusolver_mp),
      .grid = std::move(grid),
      .workspace = nullptr,
      .workspace_size = 0,
  };
}

void nvte_cusolvermp_ctx_destroy(NVTECusolverMpCtx* ctx) {
  NVTE_API_CALL(nvte_cusolvermp_ctx_destroy);
  if (ctx->workspace) {
    NVTE_CHECK_CUDA(cudaFree(ctx->workspace));
  }
  delete ctx;
}

void nvte_newton_schulz(NVTECusolverMpCtx* ctx, int64_t m, int64_t n, NVTETensor x,
                        int64_t num_iterations, const float* coefficients,
                        int64_t num_coefficients, cudaStream_t stream) {
  NVTE_API_CALL(nvte_newton_schulz);
  const auto* t = convertNVTETensorCheck(x);

  // Block size for ScaLAPACK-style distribution
  const int64_t mb = (m + ctx->nranks - 1) / ctx->nranks;
  const int64_t nb = n;

  // Compute local leading dimension
  const int64_t local_rows = cusolverMpNUMROC(m, mb, ctx->rank, 0, ctx->nranks);
  const int64_t lld = std::max(local_rows, static_cast<int64_t>(1));

  const cudaDataType_t cuda_dtype = get_cuda_dtype(t->dtype());

  // Create matrix descriptor
  auto mat_desc = CusolverMpMatrixDescCreate(m, n, mb, nb, 0, 0, lld, cuda_dtype, ctx->grid.get());

  // Create Newton-Schulz descriptor
  auto ns_desc = CusolverMpNSDescCreate(num_iterations, coefficients, num_coefficients);

  // Set stream on the cuSolverMp handle
  NVTE_CHECK_CUSOLVERMP(cusolverMpStreamSet(ctx->cusolver_mp.get(), stream));

  // Query workspace sizes
  size_t wrksp_size_device = 0;
  size_t wrksp_size_host = 0;
  NVTE_CHECK_CUSOLVERMP(cusolverMpNewtonSchulz_bufferSize(
      ctx->cusolver_mp.get(), ns_desc.get(), m, n, t->data.dptr, 1, 1, mat_desc.get(),
      &wrksp_size_device, &wrksp_size_host));

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
      ctx->cusolver_mp.get(), ns_desc.get(), m, n, t->data.dptr, 1, 1, mat_desc.get(),
      ctx->workspace, ctx->workspace_size, workspace_host.data(), workspace_host.size()));

  // Synchronize
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
}

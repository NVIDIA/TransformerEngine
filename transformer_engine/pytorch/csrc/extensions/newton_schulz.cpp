/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifdef NVTE_WITH_CUSOLVERMP

#include "../extensions.h"

#include "transformer_engine/newton_schulz.h"

namespace transformer_engine::pytorch {

int64_t cusolvermp_ctx_create(int64_t nccl_comm_ptr, int nranks, int rank) {
  auto comm = reinterpret_cast<ncclComm_t>(nccl_comm_ptr);
  auto* ctx = nvte_cusolvermp_ctx_create(comm, nranks, rank);
  return reinterpret_cast<int64_t>(ctx);
}

void cusolvermp_ctx_destroy(int64_t ctx_ptr) {
  auto* ctx = reinterpret_cast<NVTECusolverMpCtx*>(ctx_ptr);
  nvte_cusolvermp_ctx_destroy(ctx);
}

void newton_schulz(int64_t ctx_ptr, int64_t m, int64_t n, at::Tensor x, int64_t num_iterations,
                   std::vector<float> coefficients) {
  auto* ctx = reinterpret_cast<NVTECusolverMpCtx*>(ctx_ptr);

  // Build NVTETensor from PyTorch tensor
  auto x_sizes = x.sizes().vec();
  std::vector<size_t> shape(x_sizes.begin(), x_sizes.end());

  auto te_dtype = GetTransformerEngineDType(x.scalar_type());
  TensorWrapper x_tensor(x.data_ptr(), shape, te_dtype);

  auto caller_stream = at::cuda::getCurrentCUDAStream().stream();
  nvte_newton_schulz(ctx, m, n, x_tensor.data(), num_iterations, coefficients.data(),
                     static_cast<int64_t>(coefficients.size()), caller_stream);
}

}  // namespace transformer_engine::pytorch

#endif  // NVTE_WITH_CUSOLVERMP

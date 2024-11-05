/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include "../../common.h"
#include "../common.h"
#include "transformer_engine/rmsnorm.h"

namespace transformer_engine {

using namespace normalization;

void rmsnorm_fwd(const Tensor &x, const Tensor &gamma, const float epsilon, Tensor *z,
                 Tensor *rsigma, Tensor *workspace, const int multiprocessorCount,
                 const bool zero_centered_gamma, cudaStream_t stream) {
  NVTE_CHECK(x.data.shape.size() == 2);

  NVTE_CHECK(gamma.data.shape[0] == x.data.shape[1]);
  NVTE_CHECK(epsilon >= 0.f);

  NVTE_CHECK(z->data.shape == x.data.shape);

  NVTE_CHECK(rsigma->data.shape == std::vector<size_t>{x.data.shape[0]});
  NVTE_CHECK(rsigma->data.dtype == DType::kFloat32);

  if (workspace->data.dptr != nullptr) {
    CheckInputTensor(x, "x");
    CheckInputTensor(gamma, "gamma");

    CheckOutputTensor(*z, "z");
    CheckOutputTensor(*rsigma, "rsigma");
  }

  Tensor empty;

  NVTE_Norm_Backend norm_backend;
  bool is_aligned = true;
  if (std::getenv("NVTE_FWD_RMSNORM_USE_CUDNN")) {
    // TODO: add check for GPU ARCH
    norm_backend = NVTE_Norm_Backend::Cudnn;
  } else {
    norm_backend = NVTE_Norm_Backend::Te;
    is_aligned = is_ptr_aligned(z->data.dptr, x.data.dptr, gamma.data.dptr, rsigma->data.dptr);
  }
  auto plan = NormalizationPlanRegistry::getInstance().getNormalizationPlan(
      norm_backend, NVTE_Norm_Type::RMSNorm, NVTE_Norm_Stage::Forward,
      gamma.data.dtype,  // wtype
      x.data.dtype,      // itype
      gamma.data.dtype,  // otype
      x.data.shape[0],   // batch_size
      x.data.shape[1],   // hidden_size
      multiprocessorCount, zero_centered_gamma, is_aligned);

  if (workspace->data.dptr == nullptr) {
    workspace->data.shape = plan->getWorkspaceShape();
    workspace->data.dtype = DType::kByte;
    return;
  } else {
    NVTE_CHECK(workspace->data.shape == plan->getWorkspaceShape());
    plan->execute(z, x.data.dptr, gamma.data.dptr, nullptr, nullptr,
                  reinterpret_cast<void *>(const_cast<float *>(&epsilon)), rsigma->data.dptr,
                  workspace->data.dptr, stream);
  }

  return;
}

void rmsnorm_bwd(const Tensor &dz, const Tensor &x, const Tensor &rsigma, const Tensor &gamma,
                 Tensor *dx, Tensor *dgamma, Tensor *workspace, const int multiprocessorCount,
                 const bool zero_centered_gamma, cudaStream_t stream) {
  using namespace transformer_engine;

  NVTE_CHECK(dz.data.dtype == gamma.data.dtype);
  NVTE_CHECK(rsigma.data.dtype == DType::kFloat32);

  NVTE_CHECK(x.data.shape.size() == 2);
  NVTE_CHECK(dz.data.shape == x.data.shape);

  NVTE_CHECK(gamma.data.shape[0] == x.data.shape[1]);

  NVTE_CHECK(dx->data.shape == x.data.shape);
  NVTE_CHECK(dx->data.dtype == x.data.dtype);

  NVTE_CHECK(dgamma->data.shape == gamma.data.shape);
  NVTE_CHECK(dgamma->data.dtype == gamma.data.dtype);

  if (workspace->data.dptr != nullptr) {
    CheckInputTensor(dz, "dz");
    CheckInputTensor(x, "x");
    CheckInputTensor(rsigma, "rsigma");
    CheckInputTensor(gamma, "gamma");
    CheckOutputTensor(*dx, "dx");
    CheckOutputTensor(*dgamma, "dgamma");
  }

  Tensor empty;

  NVTE_Norm_Backend norm_backend;
  bool is_aligned = true;
  if (std::getenv("NVTE_BWD_RMSNORM_USE_CUDNN")) {
    // TODO: add check for GPU ARCH
    norm_backend = NVTE_Norm_Backend::Cudnn;
  } else {
    norm_backend = NVTE_Norm_Backend::Te;
    is_aligned = is_ptr_aligned(x.data.dptr, gamma.data.dptr, rsigma.data.dptr, dx->data.dptr,
                                dz.data.dptr, dgamma->data.dptr);
  }
  auto plan = NormalizationPlanRegistry::getInstance().getNormalizationPlan(
      norm_backend, NVTE_Norm_Type::RMSNorm, NVTE_Norm_Stage::Backward,
      gamma.data.dtype,  // wtype
      x.data.dtype,      // itype
      gamma.data.dtype,  // otype
      x.data.shape[0],   // batch_size
      x.data.shape[1],   // hidden_size
      multiprocessorCount, zero_centered_gamma, is_aligned);

  if (workspace->data.dptr == nullptr) {
    workspace->data.shape = plan->getWorkspaceShape();
    workspace->data.dtype = DType::kByte;
    return;
  } else {
    NVTE_CHECK(workspace->data.shape == plan->getWorkspaceShape());
    plan->execute(x.data.dptr, gamma.data.dptr, nullptr, rsigma.data.dptr, dx->data.dptr,
                  dz.data.dptr, nullptr, dgamma->data.dptr, workspace->data.dptr, stream);
  }
  return;
}

}  // namespace transformer_engine

void nvte_rmsnorm_fwd(const NVTETensor x,      // Nxhidden_size
                      const NVTETensor gamma,  // hidden_size
                      const float epsilon, NVTETensor z, NVTETensor rsigma, NVTETensor workspace,
                      const int multiprocessorCount, const bool zero_centered_gamma,
                      cudaStream_t stream) {
  NVTE_API_CALL(nvte_rmsnorm_fwd);
  using namespace transformer_engine;
  rmsnorm_fwd(*reinterpret_cast<const Tensor *>(x), *reinterpret_cast<const Tensor *>(gamma),
              epsilon, reinterpret_cast<Tensor *>(z), reinterpret_cast<Tensor *>(rsigma),
              reinterpret_cast<Tensor *>(workspace), multiprocessorCount, zero_centered_gamma,
              stream);
}

void nvte_rmsnorm_bwd(const NVTETensor dz,      // Nxhidden_size
                      const NVTETensor x,       // Nxhidden_size
                      const NVTETensor rsigma,  // N, FP32!
                      const NVTETensor gamma,   // hidden_size
                      NVTETensor dx, NVTETensor dgamma, NVTETensor workspace,
                      const int multiprocessorCount, const bool zero_centered_gamma,
                      cudaStream_t stream) {
  NVTE_API_CALL(nvte_rmsnorm_bwd);
  using namespace transformer_engine;
  rmsnorm_bwd(*reinterpret_cast<const Tensor *>(dz), *reinterpret_cast<const Tensor *>(x),
              *reinterpret_cast<const Tensor *>(rsigma), *reinterpret_cast<const Tensor *>(gamma),
              reinterpret_cast<Tensor *>(dx), reinterpret_cast<Tensor *>(dgamma),
              reinterpret_cast<Tensor *>(workspace), multiprocessorCount, zero_centered_gamma,
              stream);
}

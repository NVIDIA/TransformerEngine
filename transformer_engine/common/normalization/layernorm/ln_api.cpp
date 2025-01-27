/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/normalization.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include "../../common.h"
#include "../common.h"

namespace transformer_engine {

using namespace normalization;

void layernorm_fwd(const Tensor& x,      // BxSxhidden_size
                   const Tensor& gamma,  // hidden_size
                   const Tensor& beta,   // hidden_size
                   const float epsilon, Tensor* z, Tensor* mu, Tensor* rsigma, Tensor* workspace,
                   const int multiprocessorCount, const bool zero_centered_gamma,
                   cudaStream_t stream) {
  if (is_fp8_dtype(z->data.dtype) && !is_delayed_tensor_scaling(z->scaling_mode) &&
      !is_block_scaling(z->scaling_mode)) {
    NVTE_ERROR("Not implemented scaling mode: " + to_string(z->scaling_mode) + ".");
  }

  NVTE_CHECK(x.data.shape.size() == 2);
  NVTE_CHECK(gamma.data.shape == beta.data.shape);
  NVTE_CHECK(x.data.shape[1] == gamma.data.shape[0]);

  NVTE_CHECK(epsilon >= 0.f);

  NVTE_CHECK(z->data.shape == x.data.shape);

  NVTE_CHECK(mu->data.shape == std::vector<size_t>{x.data.shape[0]});
  NVTE_CHECK(mu->data.dtype == DType::kFloat32);

  NVTE_CHECK(rsigma->data.shape == std::vector<size_t>{x.data.shape[0]});
  NVTE_CHECK(rsigma->data.dtype == DType::kFloat32);

  if (!workspace->data.shape.empty()) {
    CheckInputTensor(x, "x");
    CheckInputTensor(gamma, "gamma");
    CheckInputTensor(beta, "beta");

    CheckOutputTensor(*z, "z");
    CheckOutputTensor(*mu, "mu");
    CheckOutputTensor(*rsigma, "rsigma");
  }

  NVTE_Norm_Backend norm_backend;
  bool is_aligned = true;
  bool cudnn_backend = use_cudnn_norm_fwd() || is_block_scaling(z->scaling_mode);

  if (cudnn_backend) {
    // TODO: add check for GPU ARCH
    norm_backend = NVTE_Norm_Backend::Cudnn;
  } else {
    norm_backend = NVTE_Norm_Backend::Te;
    is_aligned = is_ptr_aligned(z->data.dptr, x.data.dptr, gamma.data.dptr, beta.data.dptr,
                                mu->data.dptr, rsigma->data.dptr);
  }

  bool training =
      is_delayed_tensor_scaling(z->scaling_mode) || (z->columnwise_data).dptr != nullptr;

  auto plan = NormalizationPlanRegistry::getInstance().getNormalizationPlan(
      norm_backend, NVTE_Norm_Type::LayerNorm, NVTE_Norm_Stage::Forward,
      gamma.data.dtype,  // wtype
      x.data.dtype,      // itype
      z->data.dtype,     // otype
      x.data.shape[0],   // batch_size
      x.data.shape[1],   // hidden_size
      multiprocessorCount, zero_centered_gamma, is_aligned, z->scaling_mode, training);

  if (workspace->data.shape.empty()) {
    workspace->data.shape = plan->getWorkspaceShape();
    workspace->data.dtype = DType::kByte;
    return;
  }

  NVTE_CHECK(workspace->data.shape == plan->getWorkspaceShape());
  NVTE_CHECK(
      !is_block_scaling(z->scaling_mode) || (!training || z->columnwise_scale_inv.dptr != nullptr),
      "Columnwise scale_inv must be allocated for NormFwdTraining!");
  plan->execute(z, x.data.dptr, gamma.data.dptr, beta.data.dptr, mu->data.dptr,
                reinterpret_cast<void*>(const_cast<float*>(&epsilon)), rsigma->data.dptr,
                workspace->data.dptr, stream);
  return;
}

void layernorm_bwd(const Tensor& dz, const Tensor& x, const Tensor& mu, const Tensor& rsigma,
                   const Tensor& gamma, Tensor* dx, Tensor* dgamma, Tensor* dbeta,
                   Tensor* workspace, const int multiprocessorCount, const bool zero_centered_gamma,
                   cudaStream_t stream) {
  using namespace transformer_engine;
  NVTE_CHECK(dz.data.dtype == gamma.data.dtype);
  NVTE_CHECK(mu.data.dtype == DType::kFloat32);
  NVTE_CHECK(rsigma.data.dtype == mu.data.dtype);

  NVTE_CHECK(x.data.shape.size() == 2);
  NVTE_CHECK(dz.data.shape == x.data.shape);

  NVTE_CHECK(mu.data.shape[0] == x.data.shape[0]);
  NVTE_CHECK(mu.data.shape == rsigma.data.shape);

  NVTE_CHECK(gamma.data.shape[0] == x.data.shape[1]);

  NVTE_CHECK(dx->data.shape == x.data.shape);
  NVTE_CHECK(dx->data.dtype == x.data.dtype);

  NVTE_CHECK(dgamma->data.shape == gamma.data.shape);
  NVTE_CHECK(dgamma->data.dtype == gamma.data.dtype);

  NVTE_CHECK(dbeta->data.shape == gamma.data.shape);
  NVTE_CHECK(dbeta->data.dtype == gamma.data.dtype);

  if (!workspace->data.shape.empty()) {
    CheckInputTensor(dz, "dz");
    CheckInputTensor(x, "x");
    CheckInputTensor(mu, "mu");
    CheckInputTensor(rsigma, "rsigma");
    CheckInputTensor(gamma, "gamma");
    CheckOutputTensor(*dx, "dx");
    CheckOutputTensor(*dgamma, "dgamma");
    CheckOutputTensor(*dbeta, "dbeta");
  }

  NVTE_Norm_Backend norm_backend;
  bool is_aligned = true;
  if (use_cudnn_norm_bwd()) {
    // TODO: add check for GPU ARCH
    norm_backend = NVTE_Norm_Backend::Cudnn;
  } else {
    norm_backend = NVTE_Norm_Backend::Te;
    is_aligned = is_ptr_aligned(x.data.dptr, gamma.data.dptr, mu.data.dptr, rsigma.data.dptr,
                                dx->data.dptr, dz.data.dptr, dbeta->data.dptr, dgamma->data.dptr);
  }
  auto plan = NormalizationPlanRegistry::getInstance().getNormalizationPlan(
      norm_backend, NVTE_Norm_Type::LayerNorm, NVTE_Norm_Stage::Backward,
      gamma.data.dtype,  // wtype
      x.data.dtype,      // itype
      gamma.data.dtype,  // otype
      x.data.shape[0],   // batch_size
      x.data.shape[1],   // hidden_size
      multiprocessorCount, zero_centered_gamma, is_aligned);

  if (workspace->data.shape.empty()) {
    workspace->data.shape = plan->getWorkspaceShape();
    workspace->data.dtype = DType::kByte;
    return;
  } else {
    NVTE_CHECK(workspace->data.shape == plan->getWorkspaceShape());
    plan->execute(x.data.dptr, gamma.data.dptr, mu.data.dptr, rsigma.data.dptr, dx->data.dptr,
                  dz.data.dptr, dbeta->data.dptr, dgamma->data.dptr, workspace->data.dptr, stream);
  }
  return;
}
}  // namespace transformer_engine

void nvte_layernorm_fwd(const NVTETensor x,      // BxSxhidden_size
                        const NVTETensor gamma,  // hidden_size
                        const NVTETensor beta,   // hidden_size
                        const float epsilon, NVTETensor z, NVTETensor mu, NVTETensor rsigma,
                        NVTETensor workspace, const int multiprocessorCount,
                        const bool zero_centered_gamma, cudaStream_t stream) {
  NVTE_API_CALL(nvte_layernorm_fwd);
  using namespace transformer_engine;
  layernorm_fwd(*reinterpret_cast<const Tensor*>(x), *reinterpret_cast<const Tensor*>(gamma),
                *reinterpret_cast<const Tensor*>(beta), epsilon, reinterpret_cast<Tensor*>(z),
                reinterpret_cast<Tensor*>(mu), reinterpret_cast<Tensor*>(rsigma),
                reinterpret_cast<Tensor*>(workspace), multiprocessorCount, zero_centered_gamma,
                stream);
}

void nvte_layernorm_bwd(const NVTETensor dz,      // BxSxhidden_size
                        const NVTETensor x,       // BxSxhidden_size
                        const NVTETensor mu,      // BxS, FP32!
                        const NVTETensor rsigma,  // BxS, FP32!
                        const NVTETensor gamma,   // hidden_size
                        NVTETensor dx, NVTETensor dgamma, NVTETensor dbeta, NVTETensor workspace,
                        const int multiprocessorCount, const bool zero_centered_gamma,
                        cudaStream_t stream) {
  NVTE_API_CALL(nvte_layernorm_bwd);
  using namespace transformer_engine;
  layernorm_bwd(*reinterpret_cast<const Tensor*>(dz), *reinterpret_cast<const Tensor*>(x),
                *reinterpret_cast<const Tensor*>(mu), *reinterpret_cast<const Tensor*>(rsigma),
                *reinterpret_cast<const Tensor*>(gamma), reinterpret_cast<Tensor*>(dx),
                reinterpret_cast<Tensor*>(dgamma), reinterpret_cast<Tensor*>(dbeta),
                reinterpret_cast<Tensor*>(workspace), multiprocessorCount, zero_centered_gamma,
                stream);
}

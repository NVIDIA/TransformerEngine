/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/layer_norm.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include "../common.h"
#include "norms.h"


namespace transformer_engine {

void layernorm_fwd(const Tensor& x,      // BxSxhidden_size
                   const Tensor& gamma,  // hidden_size
                   const Tensor& beta,   // hidden_size
                   const float epsilon, Tensor* z, Tensor* mu, Tensor* rsigma, cudaStream_t stream,
                   const int multiprocessorCount, Tensor* workspace, Tensor* barrier,
                   const bool zero_centered_gamma) {
  using namespace transformer_engine;

  NVTE_CHECK(x.data.shape.size() == 2);
  NVTE_CHECK(gamma.data.shape == beta.data.shape);
  NVTE_CHECK(x.data.shape[1] == gamma.data.shape[0]);

  NVTE_CHECK(epsilon >= 0.f);

  NVTE_CHECK(z->data.shape == x.data.shape);

  NVTE_CHECK(mu->data.shape == std::vector<size_t>{x.data.shape[0]});
  NVTE_CHECK(mu->data.dtype == DType::kFloat32);

  NVTE_CHECK(rsigma->data.shape == std::vector<size_t>{x.data.shape[0]});
  NVTE_CHECK(rsigma->data.dtype == DType::kFloat32);

  if (workspace->data.dptr != nullptr) {
    CheckInputTensor(x, "x");
    CheckInputTensor(gamma, "gamma");
    CheckInputTensor(beta, "beta");

    CheckOutputTensor(*z, "z");
    CheckOutputTensor(*mu, "mu");
    CheckOutputTensor(*rsigma, "rsigma");
  }

  // TODO: add check for GPU ARCH

  if (std::getenv("NVTE_FWD_LAYERNORM_USE_CUDNN")) {
    printf("TE/cuDNN LayerNorm is WIP. There are known bugs. Use it at your own risk!\n");
    NormFwdCudnn<NVTE_NORM_TYPE::LN_FWD_CUDNN> NormFwd(x, gamma, beta, epsilon, z, mu, rsigma,
                                                       stream, multiprocessorCount, workspace,
                                                       zero_centered_gamma);
    norms_launcher<NVTE_NORM_TYPE::LN_FWD_CUDNN>(NormFwd, workspace);
  } else {
    NormFwdTe<NVTE_NORM_TYPE::LN_FWD_TE> NormFwd(x, gamma, beta, epsilon, z, mu, rsigma, stream,
                                                 multiprocessorCount, workspace, barrier,
                                                 zero_centered_gamma);
    norms_launcher<NVTE_NORM_TYPE::LN_FWD_TE>(NormFwd, workspace, barrier);
  }
  return;
}

void layernorm_bwd(const Tensor& dz, const Tensor& x, const Tensor& mu, const Tensor& rsigma,
                   const Tensor& gamma, Tensor* dx, Tensor* dgamma, Tensor* dbeta,
                   Tensor* dgamma_part, Tensor* dbeta_part, cudaStream_t stream,
                   const int multiprocessorCount, Tensor* workspace, Tensor* barrier,
                   const bool zero_centered_gamma) {
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

  if (workspace->data.dptr) {
    CheckInputTensor(dz, "dz");
    CheckInputTensor(x, "x");
    CheckInputTensor(mu, "mu");
    CheckInputTensor(rsigma, "rsigma");
    CheckInputTensor(gamma, "gamma");
    CheckOutputTensor(*dx, "dx");
    CheckOutputTensor(*dgamma, "dgamma");
    CheckOutputTensor(*dbeta, "dbeta");
  }

  if (std::getenv("NVTE_BWD_LAYERNORM_USE_CUDNN")) {
    printf("TE/cuDNN LayerNorm is WIP. There are known bugs. Use it at your own risk!\n");
    NormBwdCudnn<NVTE_NORM_TYPE::LN_BWD_CUDNN> BwdNorm(dz, x, mu, rsigma, gamma, dx, dgamma, dbeta,
                                                       stream, multiprocessorCount, workspace,
                                                       zero_centered_gamma);
    norms_launcher<NVTE_NORM_TYPE::LN_BWD_CUDNN>(BwdNorm, workspace);
  } else {
    NormBwdTe<NVTE_NORM_TYPE::LN_BWD_TE> BwdNorm(
        dz, x, mu, rsigma, gamma, dx, dgamma, dbeta, dgamma_part, dbeta_part, stream,
        multiprocessorCount, workspace, barrier, zero_centered_gamma);
    norms_launcher<NVTE_NORM_TYPE::LN_BWD_TE>(BwdNorm, workspace, barrier, dgamma_part, dbeta_part);
  }
}
}  // namespace transformer_engine

void nvte_layernorm_fwd(const NVTETensor x,      // BxSxhidden_size
                        const NVTETensor gamma,  // hidden_size
                        const NVTETensor beta,   // hidden_size
                        const float epsilon, NVTETensor z, NVTETensor mu, NVTETensor rsigma,
                        cudaStream_t stream, const int multiprocessorCount, NVTETensor workspace,
                        NVTETensor barrier) {
  NVTE_API_CALL(nvte_layernorm_fwd);
  using namespace transformer_engine;
  layernorm_fwd(*reinterpret_cast<const Tensor*>(x), *reinterpret_cast<const Tensor*>(gamma),
                *reinterpret_cast<const Tensor*>(beta), epsilon, reinterpret_cast<Tensor*>(z),
                reinterpret_cast<Tensor*>(mu), reinterpret_cast<Tensor*>(rsigma), stream,
                multiprocessorCount, reinterpret_cast<Tensor*>(workspace),
                reinterpret_cast<Tensor*>(barrier), false);
}

void nvte_layernorm_bwd(const NVTETensor dz,      // BxSxhidden_size
                        const NVTETensor x,       // BxSxhidden_size
                        const NVTETensor mu,      // BxS, FP32!
                        const NVTETensor rsigma,  // BxS, FP32!
                        const NVTETensor gamma,   // hidden_size
                        NVTETensor dx, NVTETensor dgamma, NVTETensor dbeta, NVTETensor dgamma_part,
                        NVTETensor dbeta_part, cudaStream_t stream, const int multiprocessorCount,
                        NVTETensor workspace, NVTETensor barrier) {
  NVTE_API_CALL(nvte_layernorm_bwd);
  using namespace transformer_engine;
  layernorm_bwd(*reinterpret_cast<const Tensor*>(dz), *reinterpret_cast<const Tensor*>(x),
                *reinterpret_cast<const Tensor*>(mu), *reinterpret_cast<const Tensor*>(rsigma),
                *reinterpret_cast<const Tensor*>(gamma), reinterpret_cast<Tensor*>(dx),
                reinterpret_cast<Tensor*>(dgamma), reinterpret_cast<Tensor*>(dbeta),
                reinterpret_cast<Tensor*>(dgamma_part), reinterpret_cast<Tensor*>(dbeta_part),
                stream, multiprocessorCount, reinterpret_cast<Tensor*>(workspace),
                reinterpret_cast<Tensor*>(barrier), false);
}

void nvte_layernorm1p_fwd(const NVTETensor x,      // BxSxhidden_size
                          const NVTETensor gamma,  // hidden_size
                          const NVTETensor beta,   // hidden_size
                          const float epsilon, NVTETensor z, NVTETensor mu, NVTETensor rsigma,
                          cudaStream_t stream, const int multiprocessorCount, NVTETensor workspace,
                          NVTETensor barrier) {
  NVTE_API_CALL(nvte_layernorm1p_fwd);
  using namespace transformer_engine;
  layernorm_fwd(*reinterpret_cast<const Tensor*>(x), *reinterpret_cast<const Tensor*>(gamma),
                *reinterpret_cast<const Tensor*>(beta), epsilon, reinterpret_cast<Tensor*>(z),
                reinterpret_cast<Tensor*>(mu), reinterpret_cast<Tensor*>(rsigma), stream,
                multiprocessorCount, reinterpret_cast<Tensor*>(workspace),
                reinterpret_cast<Tensor*>(barrier), true);
}

void nvte_layernorm1p_bwd(const NVTETensor dz,      // BxSxhidden_size
                          const NVTETensor x,       // BxSxhidden_size
                          const NVTETensor mu,      // BxS, FP32!
                          const NVTETensor rsigma,  // BxS, FP32!
                          const NVTETensor gamma,   // hidden_size
                          NVTETensor dx, NVTETensor dgamma, NVTETensor dbeta,
                          NVTETensor dgamma_part, NVTETensor dbeta_part, cudaStream_t stream,
                          const int multiprocessorCount, NVTETensor workspace, NVTETensor barrier) {
  NVTE_API_CALL(nvte_layernorm1p_bwd);
  using namespace transformer_engine;
  layernorm_bwd(*reinterpret_cast<const Tensor*>(dz), *reinterpret_cast<const Tensor*>(x),
                *reinterpret_cast<const Tensor*>(mu), *reinterpret_cast<const Tensor*>(rsigma),
                *reinterpret_cast<const Tensor*>(gamma), reinterpret_cast<Tensor*>(dx),
                reinterpret_cast<Tensor*>(dgamma), reinterpret_cast<Tensor*>(dbeta),
                reinterpret_cast<Tensor*>(dgamma_part), reinterpret_cast<Tensor*>(dbeta_part),
                stream, multiprocessorCount, reinterpret_cast<Tensor*>(workspace),
                reinterpret_cast<Tensor*>(barrier), true);
}

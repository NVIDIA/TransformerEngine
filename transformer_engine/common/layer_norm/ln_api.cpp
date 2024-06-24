/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/layer_norm.h>

#include <cstdint>
#include <vector>

#include "../common.h"
#include "ln.h"

/*

Supported Type combinations:

input    compute   weights   output
=======================================
fp32     fp32      fp32      fp32
fp16     fp32      fp16      fp16
bf16     fp32      bf16      bf16
fp32     fp32      fp16      fp16
fp32     fp32      bf16      bf16
bf16     fp32      bf16      fp8

Remarks:
Output type = Weight type
Compute always in FP32

*/

namespace transformer_engine {
namespace layer_norm {

using namespace transformer_engine;

// Create registries and provide runtime versions of config hash functions.

FwdTunedRegistry FWD_TUNED_FUNCS;
BwdTunedRegistry BWD_TUNED_FUNCS;
FwdGeneralRegistry FWD_GENERAL_FUNCS;
BwdGeneralRegistry BWD_GENERAL_FUNCS;

////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t get_type_id(DType dtype) {
  if (dtype == DType::kFloat16) {
    return TypeId<fp16>::Value;
  } else if (dtype == DType::kBFloat16) {
    return TypeId<bf16>::Value;
  } else if (dtype == DType::kFloat32) {
    return TypeId<fp32>::Value;
  } else if (dtype == DType::kFloat8E4M3) {
    return TypeId<fp8e4m3>::Value;
  } else {
    NVTE_ERROR("Type not supported.");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

uint64_t get_key(DType wtype, DType itype, DType otype, DType ctype, uint64_t hidden_size) {
  using namespace layer_norm;
  uint64_t type_key = get_type_id(wtype) | (get_type_id(itype) << 2) | (get_type_id(otype) << 4) |
                      (get_type_id(ctype) << 6);
  uint64_t launcher_key = (type_key << 32) | hidden_size;
  return launcher_key;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

layer_norm::FwdFunction& get_fwd_launcher(DType wtype, DType itype, DType otype, DType ctype,
                                          const layer_norm::FwdParams& params) {
  // Look for tuned kernel
  auto tuned_key = layer_norm::get_key(wtype, itype, otype, ctype, params.cols);
  auto is_aligned = [](const void* ptr) -> bool {
    // Assume vectorized memory accesses are <=16B
    return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
  };
  if (params.rows % 4 == 0 && is_aligned(params.x) && is_aligned(params.mu) &&
      is_aligned(params.rs) && is_aligned(params.gamma) && is_aligned(params.beta) &&
      is_aligned(params.z) && layer_norm::FWD_TUNED_FUNCS.count(tuned_key) > 0) {
    return layer_norm::FWD_TUNED_FUNCS.at(tuned_key);
  }

  // Pick general kernel
  auto general_key = layer_norm::get_key(wtype, itype, otype, ctype, 0);
  if (layer_norm::FWD_GENERAL_FUNCS.count(general_key) == 0) {
    NVTE_ERROR("FWD: Unsupported types.");
  }
  auto& general_func_map = layer_norm::FWD_GENERAL_FUNCS.at(general_key);
  auto func_iter = general_func_map.lower_bound(params.cols);
  if (func_iter == general_func_map.end()) {
    // Hidden size is too big, need to use multi-CTA
    return general_func_map.rbegin()->second;
  } else {
    return func_iter->second;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

layer_norm::BwdFunction& get_bwd_launcher(DType wtype, DType itype, DType otype, DType ctype,
                                          const layer_norm::BwdParams& params) {
  // Look for tuned kernel
  auto tuned_key = layer_norm::get_key(wtype, itype, otype, ctype, params.cols);
  auto is_aligned = [](const void* ptr) -> bool {
    // Assume vectorized memory accesses are <=16B
    return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
  };
  if (params.rows % 4 == 0 && is_aligned(params.x) && is_aligned(params.mu) &&
      is_aligned(params.rs) && is_aligned(params.gamma) && is_aligned(params.dz) &&
      is_aligned(params.dx) && is_aligned(params.dbeta) && is_aligned(params.dgamma) &&
      is_aligned(params.dbeta_part) && is_aligned(params.dgamma_part) &&
      layer_norm::BWD_TUNED_FUNCS.count(tuned_key) > 0) {
    return layer_norm::BWD_TUNED_FUNCS.at(tuned_key);
  }

  // Pick general kernel
  auto general_key = layer_norm::get_key(wtype, itype, otype, ctype, 0);
  if (layer_norm::BWD_GENERAL_FUNCS.count(general_key) == 0) {
    NVTE_ERROR("BWD: Unsupported types.");
  }
  auto& general_func_map = layer_norm::BWD_GENERAL_FUNCS.at(general_key);
  auto func_iter = general_func_map.lower_bound(params.cols);
  if (func_iter == general_func_map.end()) {
    // Hidden size is too big, need to use multi-CTA
    return general_func_map.rbegin()->second;
  } else {
    return func_iter->second;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

size_t product(const std::vector<size_t>& shape) {
  size_t ret = 1;
  for (auto s : shape) {
    ret *= s;
  }
  return ret;
}

}  // namespace layer_norm

////////////////////////////////////////////////////////////////////////////////////////////////////

void layernorm_fwd(const Tensor& x,      // BxSxhidden_size
                   const Tensor& gamma,  // hidden_size
                   const Tensor& beta,   // hidden_size
                   const float epsilon, Tensor* z, Tensor* mu, Tensor* rsigma, cudaStream_t stream,
                   const int multiprocessorCount, Tensor* workspace, Tensor* barrier,
                   const bool zero_centered_gamma) {
  const auto itype = x.data.dtype;
  const auto wtype = gamma.data.dtype;
  const auto otype = z->data.dtype;
  const bool fp8_out = is_fp8_dtype(otype);
  const auto ctype = layer_norm::DType::kFloat32;

  NVTE_CHECK(x.data.shape.size() == 2);

  const size_t rows = x.data.shape[0];
  const size_t cols = x.data.shape[1];
  const auto hidden_size = gamma.data.shape[0];

  NVTE_CHECK(gamma.data.shape == beta.data.shape);
  NVTE_CHECK(hidden_size == cols);

  NVTE_CHECK(epsilon >= 0.f);

  NVTE_CHECK(z->data.shape == x.data.shape);

  NVTE_CHECK(mu->data.shape == std::vector<size_t>{rows});
  NVTE_CHECK(mu->data.dtype == ctype);

  NVTE_CHECK(rsigma->data.shape == std::vector<size_t>{rows});
  NVTE_CHECK(rsigma->data.dtype == ctype);

  layer_norm::LaunchParams<layer_norm::FwdParams> launch_params;

  launch_params.multiprocessorCount = multiprocessorCount;
  launch_params.stream = stream;

  // Set the kernel runtime parameters.
  layer_norm::FwdParams& params = launch_params.params;
  params.rows = rows;
  params.cols = cols;
  params.x = x.data.dptr;
  params.mu = mu->data.dptr;
  params.rs = rsigma->data.dptr;
  params.gamma = gamma.data.dptr;
  params.beta = beta.data.dptr;
  params.z = z->data.dptr;
  params.epsilon = epsilon;
  params.amax = z->amax.dptr;
  params.scale = z->scale.dptr;
  params.fp8_out = fp8_out;
  params.zero_centered_gamma = zero_centered_gamma;

  // Request the kernel launcher.
  auto launcher = layer_norm::get_fwd_launcher(wtype, itype, otype, ctype, params);

  // Query the kernel-specific launch parameters.
  launcher(launch_params, true);
  if (launch_params.workspace_bytes == 0) {
    launch_params.workspace_bytes = 1;
  }

  if (workspace->data.dptr == nullptr) {
    NVTE_CHECK(barrier->data.dptr == nullptr);

    workspace->data.dtype = layer_norm::DType::kByte;
    workspace->data.shape = {launch_params.workspace_bytes};

    barrier->data.dtype = layer_norm::DType::kInt32;
    barrier->data.shape = {launch_params.barrier_size};

    return;
  } else {
    NVTE_CHECK(workspace->data.dtype == layer_norm::DType::kByte);
    NVTE_CHECK(workspace->data.shape == std::vector<size_t>{launch_params.workspace_bytes});
  }

  if (launch_params.barrier_size > 0) {
    NVTE_CHECK(barrier->data.dptr != nullptr);
    NVTE_CHECK(barrier->data.dtype == layer_norm::DType::kInt32);
    NVTE_CHECK(barrier->data.shape == std::vector<size_t>{launch_params.barrier_size});
  }

  // Tensor checks are delayed here in order to recover workspace sizes with null data
  CheckInputTensor(x, "x");
  CheckInputTensor(gamma, "gamma");
  CheckInputTensor(beta, "beta");

  CheckOutputTensor(*z, "z");
  CheckOutputTensor(*mu, "mu");
  CheckOutputTensor(*rsigma, "rsigma");

  if (launch_params.barrier_size > 0) {
    params.workspace = workspace->data.dptr;
    params.barrier = reinterpret_cast<int*>(barrier->data.dptr);
  }

  // Clear buffers
  if (params.fp8_out) {
    cudaMemsetAsync(params.amax, 0, layer_norm::product(z->amax.shape) * typeToSize(z->amax.dtype),
                    stream);
  }
  if (launch_params.barrier_size > 0) {
    cudaMemsetAsync(params.barrier, 0,
                    layer_norm::product(barrier->data.shape) * typeToSize(barrier->data.dtype),
                    stream);
  }

  // Launch the kernel.
  launcher(launch_params, false);

  return;
}

void layernorm_bwd(const Tensor& dz, const Tensor& x, const Tensor& mu, const Tensor& rsigma,
                   const Tensor& gamma, Tensor* dx, Tensor* dgamma, Tensor* dbeta,
                   Tensor* dgamma_part, Tensor* dbeta_part, cudaStream_t stream,
                   const int multiprocessorCount, Tensor* workspace, Tensor* barrier,
                   const bool zero_centered_gamma) {
  using namespace transformer_engine;

  auto itype = x.data.dtype;
  auto wtype = gamma.data.dtype;
  auto otype = wtype;
  auto ctype = DType::kFloat32;

  NVTE_CHECK(dz.data.dtype == otype);
  NVTE_CHECK(mu.data.dtype == ctype);
  NVTE_CHECK(rsigma.data.dtype == ctype);

  NVTE_CHECK(x.data.shape.size() == 2);
  NVTE_CHECK(dz.data.shape == x.data.shape);
  auto rows = x.data.shape[0];
  auto cols = x.data.shape[1];

  auto hidden_size = gamma.data.shape[0];

  NVTE_CHECK(mu.data.shape[0] == rows);
  NVTE_CHECK(mu.data.shape == rsigma.data.shape);

  NVTE_CHECK(gamma.data.shape[0] == cols);

  NVTE_CHECK(dx->data.shape == x.data.shape);
  NVTE_CHECK(dx->data.dtype == x.data.dtype);

  NVTE_CHECK(dgamma->data.shape == gamma.data.shape);
  NVTE_CHECK(dgamma->data.dtype == gamma.data.dtype);

  NVTE_CHECK(dbeta->data.shape == gamma.data.shape);
  NVTE_CHECK(dbeta->data.dtype == gamma.data.dtype);

  layer_norm::LaunchParams<layer_norm::BwdParams> launch_params;
  launch_params.stream = stream;
  launch_params.multiprocessorCount = multiprocessorCount;

  // Set the kernel runtime parameters.
  layer_norm::BwdParams& params = launch_params.params;
  params.rows = rows;
  params.cols = cols;
  params.x = x.data.dptr;
  params.mu = mu.data.dptr;
  params.rs = rsigma.data.dptr;
  params.gamma = gamma.data.dptr;
  params.dz = dz.data.dptr;
  params.dx = dx->data.dptr;
  params.dbeta = dbeta->data.dptr;
  params.dgamma = dgamma->data.dptr;
  params.dbeta_part = dbeta_part->data.dptr;
  params.dgamma_part = dgamma_part->data.dptr;
  params.zero_centered_gamma = zero_centered_gamma;

  auto launcher = layer_norm::get_bwd_launcher(wtype, itype, otype, ctype, params);

  // Query the kernel-specific launch parameters.
  launcher(launch_params, true);

  // Populate shape and dtypes for FW to allocate memory
  if (dgamma_part->data.dptr == nullptr) {
    NVTE_CHECK(dbeta_part->data.dptr == nullptr);

    dgamma_part->data.dtype = ctype;
    dgamma_part->data.shape = {static_cast<uint64_t>(launch_params.params.ctas_per_col),
                               hidden_size};

    dbeta_part->data.dtype = ctype;
    dbeta_part->data.shape = {static_cast<uint64_t>(launch_params.params.ctas_per_col),
                              hidden_size};

    workspace->data.dtype = layer_norm::DType::kByte;
    workspace->data.shape = {launch_params.workspace_bytes};

    barrier->data.dtype = layer_norm::DType::kInt32;
    barrier->data.shape = {launch_params.barrier_size};

    return;
  } else {
    NVTE_CHECK(dbeta_part->data.dptr != nullptr);
    auto pdw_shape =
        std::vector<size_t>{static_cast<uint64_t>(launch_params.params.ctas_per_col), hidden_size};

    NVTE_CHECK(dgamma_part->data.dtype == ctype);
    NVTE_CHECK(dgamma_part->data.shape == pdw_shape);
    NVTE_CHECK(dbeta_part->data.dtype == ctype);
    NVTE_CHECK(dbeta_part->data.shape == pdw_shape);
  }

  if (launch_params.barrier_size > 0) {
    NVTE_CHECK(barrier->data.dptr != nullptr);
    NVTE_CHECK(barrier->data.dtype == layer_norm::DType::kInt32);
    NVTE_CHECK(barrier->data.shape == std::vector<size_t>{launch_params.barrier_size});
  }

  if (launch_params.workspace_bytes > 0) {
    NVTE_CHECK(workspace->data.dptr != nullptr);
    NVTE_CHECK(workspace->data.dtype == layer_norm::DType::kByte);
    NVTE_CHECK(workspace->data.shape == std::vector<size_t>{launch_params.workspace_bytes});
  }

  // Tensor checks are delayed here in order to recover workspace sizes with null data
  CheckInputTensor(dz, "dz");
  CheckInputTensor(x, "x");
  CheckInputTensor(mu, "mu");
  CheckInputTensor(rsigma, "rsigma");
  CheckInputTensor(gamma, "gamma");
  CheckOutputTensor(*dx, "dx");
  CheckOutputTensor(*dgamma, "dgamma");
  CheckOutputTensor(*dbeta, "dbeta");

  if (launch_params.barrier_size > 0) {
    params.workspace = workspace->data.dptr;
    params.barrier = reinterpret_cast<int*>(barrier->data.dptr);
    cudaMemsetAsync(params.barrier, 0,
                    layer_norm::product(barrier->data.shape) * typeToSize(barrier->data.dtype),
                    stream);
  }

  // Launch the kernel.
  launcher(launch_params, false);
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

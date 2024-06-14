/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cstdint>
#include <numeric>
#include <vector>

#include "../common.h"
#include "rmsnorm.h"
#include "transformer_engine/rmsnorm.h"

/*

Supported Type combinations:

input    compute   weights   output
=======================================
fp32     fp32      fp32      fp32
fp16     fp32      fp16      fp16
bf16     fp32      bf16      bf16
fp32     fp32      fp32      fp16
fp32     fp32      fp32      bf16
fp32     fp32      fp32      fp8
fp16     fp32      fp16      fp8
bf16     fp32      bf16      fp8

Remarks:
Input type = Weight type
Compute always in FP32

*/

namespace transformer_engine {

namespace layer_norm {
uint64_t get_key(DType wtype, DType itype, DType otype, DType ctype, uint64_t hidden_size);
}

namespace rmsnorm {

using namespace transformer_engine;

FwdTunedRegistry FWD_TUNED_FUNCS;
BwdTunedRegistry BWD_TUNED_FUNCS;
FwdGeneralRegistry FWD_GENERAL_FUNCS;
BwdGeneralRegistry BWD_GENERAL_FUNCS;

FwdFunction &get_fwd_launcher(DType wtype, DType itype, DType otype, DType ctype,
                              const layer_norm::FwdParams &params) {
  // Look for tuned kernel
  auto tuned_key = layer_norm::get_key(wtype, itype, otype, ctype, params.cols);
  auto is_aligned = [](const void *ptr) -> bool {
    // Assume vectorized memory accesses are <=16B
    return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
  };
  if (params.rows % 4 == 0 && is_aligned(params.x) && is_aligned(params.rs) &&
      is_aligned(params.gamma) && is_aligned(params.z) && FWD_TUNED_FUNCS.count(tuned_key) > 0) {
    return FWD_TUNED_FUNCS.at(tuned_key);
  }

  // Pick general kernel
  auto general_key = layer_norm::get_key(wtype, itype, otype, ctype, 0);
  if (FWD_GENERAL_FUNCS.count(general_key) == 0) {
    NVTE_ERROR("FWD: Unsupported types.");
  }
  auto &general_func_map = FWD_GENERAL_FUNCS.at(general_key);
  auto func_iter = general_func_map.lower_bound(params.cols);
  if (func_iter == general_func_map.end()) {
    // Hidden size is too big, need to use multi-CTA
    return general_func_map.rbegin()->second;
  } else {
    return func_iter->second;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

BwdFunction &get_bwd_launcher(DType wtype, DType itype, DType otype, DType ctype,
                              const layer_norm::BwdParams &params) {
  // Look for tuned kernel
  auto tuned_key = layer_norm::get_key(wtype, itype, otype, ctype, params.cols);
  auto is_aligned = [](const void *ptr) -> bool {
    // Assume vectorized memory accesses are <=16B
    return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
  };
  if (params.rows % 4 == 0 && is_aligned(params.x) && is_aligned(params.rs) &&
      is_aligned(params.gamma) && is_aligned(params.dz) && is_aligned(params.dx) &&
      is_aligned(params.dgamma) && is_aligned(params.dgamma_part) &&
      layer_norm::BWD_TUNED_FUNCS.count(tuned_key) > 0) {
    return BWD_TUNED_FUNCS.at(tuned_key);
  }

  // Pick general kernel
  auto general_key = layer_norm::get_key(wtype, itype, otype, ctype, 0);
  if (BWD_GENERAL_FUNCS.count(general_key) == 0) {
    NVTE_ERROR("BWD: Unsupported types.");
  }
  auto &general_func_map = BWD_GENERAL_FUNCS.at(general_key);
  auto func_iter = general_func_map.lower_bound(params.cols);
  if (func_iter == general_func_map.end()) {
    // Hidden size is too big, need to use multi-CTA
    return general_func_map.rbegin()->second;
  } else {
    return func_iter->second;
  }
}

// ////////////////////////////////////////////////////////////////////////////////////////////////////

inline size_t product(const std::vector<size_t> &shape) {
  return std::accumulate(shape.cbegin(), shape.cend(), size_t{1}, std::multiplies<>());
}

}  // namespace rmsnorm

////////////////////////////////////////////////////////////////////////////////////////////////////

void rmsnorm_fwd(const Tensor &x, const Tensor &gamma, const float epsilon, Tensor *z,
                 Tensor *rsigma, cudaStream_t stream, const int multiprocessorCount,
                 Tensor *workspace, Tensor *barrier, const bool zero_centered_gamma) {
  auto itype = x.data.dtype;
  auto wtype = gamma.data.dtype;
  auto otype = z->data.dtype;
  const bool fp8_out = is_fp8_dtype(otype);
  auto ctype = DType::kFloat32;

  NVTE_CHECK(x.data.shape.size() == 2);

  const size_t rows = x.data.shape[0];
  const size_t cols = x.data.shape[1];
  const auto hidden_size = gamma.data.shape[0];

  NVTE_CHECK(hidden_size == cols);
  NVTE_CHECK(epsilon >= 0.f);

  NVTE_CHECK(z->data.shape == x.data.shape);

  NVTE_CHECK(rsigma->data.shape == std::vector<size_t>{rows});
  NVTE_CHECK(rsigma->data.dtype == ctype);

  rmsnorm::LaunchParams<rmsnorm::FwdParams> launch_params;

  launch_params.multiprocessorCount = multiprocessorCount;
  launch_params.stream = stream;

  // Set the kernel runtime parameters.
  rmsnorm::FwdParams &params = launch_params.params;
  params.rows = rows;
  params.cols = cols;
  params.x = x.data.dptr;
  params.mu = nullptr;
  params.rs = rsigma->data.dptr;
  params.gamma = gamma.data.dptr;
  params.beta = nullptr;
  params.z = z->data.dptr;
  params.epsilon = epsilon;
  params.amax = z->amax.dptr;
  params.scale = z->scale.dptr;
  params.fp8_out = fp8_out;
  params.zero_centered_gamma = zero_centered_gamma;

  // Request the kernel launcher.
  auto launcher = rmsnorm::get_fwd_launcher(wtype, itype, otype, ctype, params);

  // Query the kernel-specific launch parameters.
  launcher(launch_params, true);
  if (launch_params.workspace_bytes == 0) {
    launch_params.workspace_bytes = 1;
  }

  if (workspace->data.dptr == nullptr) {
    NVTE_CHECK(barrier->data.dptr == nullptr);

    workspace->data.dtype = DType::kByte;
    workspace->data.shape = {launch_params.workspace_bytes};

    barrier->data.dtype = DType::kInt32;
    barrier->data.shape = {launch_params.barrier_size};

    return;
  } else {
    NVTE_CHECK(workspace->data.dtype == DType::kByte);
    NVTE_CHECK(workspace->data.shape == std::vector<size_t>{launch_params.workspace_bytes});
  }

  if (launch_params.barrier_size > 0) {
    NVTE_CHECK(barrier->data.dptr != nullptr);
    NVTE_CHECK(barrier->data.dtype == DType::kInt32);
    NVTE_CHECK(barrier->data.shape == std::vector<size_t>{launch_params.barrier_size});
  }

  // Tensor checks are delayed here in order to recover workspace sizes with null data
  CheckInputTensor(x, "x");
  CheckInputTensor(gamma, "gamma");

  CheckOutputTensor(*z, "z");
  CheckOutputTensor(*rsigma, "rsigma");

  if (launch_params.barrier_size > 0) {
    params.workspace = workspace->data.dptr;
    params.barrier = reinterpret_cast<int *>(barrier->data.dptr);
  }

  // Clear buffers
  if (params.fp8_out) {
    cudaMemsetAsync(params.amax, 0, rmsnorm::product(z->amax.shape) * typeToSize(z->amax.dtype),
                    stream);
  }
  if (launch_params.barrier_size > 0) {
    cudaMemsetAsync(params.barrier, 0,
                    rmsnorm::product(barrier->data.shape) * typeToSize(barrier->data.dtype),
                    stream);
  }

  // Launch the kernel.
  launcher(launch_params, false);

  return;
}

void rmsnorm_bwd(const Tensor &dz, const Tensor &x, const Tensor &rsigma, const Tensor &gamma,
                 Tensor *dx, Tensor *dgamma, Tensor *dgamma_part, cudaStream_t stream,
                 const int multiprocessorCount, Tensor *workspace, Tensor *barrier,
                 const bool zero_centered_gamma) {
  using namespace transformer_engine;

  auto itype = x.data.dtype;
  auto wtype = gamma.data.dtype;
  auto otype = wtype;
  auto ctype = DType::kFloat32;

  NVTE_CHECK(dz.data.dtype == otype);
  NVTE_CHECK(rsigma.data.dtype == ctype);

  NVTE_CHECK(x.data.shape.size() == 2);
  NVTE_CHECK(dz.data.shape == x.data.shape);

  const auto rows = x.data.shape[0];
  const auto cols = x.data.shape[1];
  const auto hidden_size = gamma.data.shape[0];

  NVTE_CHECK(gamma.data.shape[0] == cols);

  NVTE_CHECK(dx->data.shape == x.data.shape);
  NVTE_CHECK(dx->data.dtype == x.data.dtype);

  NVTE_CHECK(dgamma->data.shape == gamma.data.shape);
  NVTE_CHECK(dgamma->data.dtype == gamma.data.dtype);

  rmsnorm::LaunchParams<rmsnorm::BwdParams> launch_params;
  launch_params.stream = stream;
  launch_params.multiprocessorCount = multiprocessorCount;

  // Set the kernel runtime parameters.
  rmsnorm::BwdParams &params = launch_params.params;
  params.rows = rows;
  params.cols = cols;
  params.x = x.data.dptr;
  params.mu = nullptr;
  params.rs = rsigma.data.dptr;
  params.gamma = gamma.data.dptr;
  params.dz = dz.data.dptr;
  params.dx = dx->data.dptr;
  params.dbeta = nullptr;
  params.dgamma = dgamma->data.dptr;
  params.dbeta_part = nullptr;
  params.dgamma_part = dgamma_part->data.dptr;
  params.zero_centered_gamma = zero_centered_gamma;

  // Request the kernel launcher.
  auto launcher = rmsnorm::get_bwd_launcher(wtype, itype, otype, ctype, params);

  // Query the kernel-specific launch parameters.
  launcher(launch_params, true);

  // Populate shape and dtypes for FW to allocate memory
  if (dgamma_part->data.dptr == nullptr) {
    dgamma_part->data.dtype = ctype;
    dgamma_part->data.shape = {static_cast<uint64_t>(launch_params.params.ctas_per_col),
                               hidden_size};

    workspace->data.dtype = DType::kByte;
    workspace->data.shape = {launch_params.workspace_bytes};

    barrier->data.dtype = DType::kInt32;
    barrier->data.shape = {launch_params.barrier_size};

    return;
  } else {
    auto pdw_shape =
        std::vector<size_t>{static_cast<uint64_t>(launch_params.params.ctas_per_col), hidden_size};
    NVTE_CHECK(dgamma_part->data.dtype == ctype);
    NVTE_CHECK(dgamma_part->data.shape == pdw_shape);
  }

  if (launch_params.barrier_size > 0) {
    NVTE_CHECK(barrier->data.dptr != nullptr);
    NVTE_CHECK(barrier->data.dtype == DType::kInt32);
    NVTE_CHECK(barrier->data.shape == std::vector<size_t>{launch_params.barrier_size});
  }

  if (launch_params.workspace_bytes > 0) {
    NVTE_CHECK(workspace->data.dptr != nullptr);
    NVTE_CHECK(workspace->data.dtype == DType::kByte);
    NVTE_CHECK(workspace->data.shape == std::vector<size_t>{launch_params.workspace_bytes});
  }

  // Tensor checks are delayed here in order to recover workspace sizes with null data
  CheckInputTensor(dz, "dz");
  CheckInputTensor(x, "x");
  CheckInputTensor(rsigma, "rsigma");
  CheckInputTensor(gamma, "gamma");
  CheckOutputTensor(*dx, "dx");
  CheckOutputTensor(*dgamma, "dgamma");

  if (launch_params.barrier_size > 0) {
    params.workspace = workspace->data.dptr;
    params.barrier = reinterpret_cast<int *>(barrier->data.dptr);
    cudaMemsetAsync(params.barrier, 0,
                    rmsnorm::product(barrier->data.shape) * typeToSize(barrier->data.dtype),
                    stream);
  }

  // Launch the kernel.
  launcher(launch_params, false);
}

}  // namespace transformer_engine

void nvte_rmsnorm_fwd(const NVTETensor x,      // Nxhidden_size
                      const NVTETensor gamma,  // hidden_size
                      const float epsilon, NVTETensor z, NVTETensor rsigma, cudaStream_t stream,
                      const int multiprocessorCount, NVTETensor workspace, NVTETensor barrier) {
  NVTE_API_CALL(nvte_rmsnorm_fwd);
  using namespace transformer_engine;
  rmsnorm_fwd(*reinterpret_cast<const Tensor *>(x), *reinterpret_cast<const Tensor *>(gamma),
              epsilon, reinterpret_cast<Tensor *>(z), reinterpret_cast<Tensor *>(rsigma), stream,
              multiprocessorCount, reinterpret_cast<Tensor *>(workspace),
              reinterpret_cast<Tensor *>(barrier), false);
}

void nvte_rmsnorm_bwd(const NVTETensor dz,      // Nxhidden_size
                      const NVTETensor x,       // Nxhidden_size
                      const NVTETensor rsigma,  // N, FP32!
                      const NVTETensor gamma,   // hidden_size
                      NVTETensor dx, NVTETensor dgamma, NVTETensor dgamma_part, cudaStream_t stream,
                      const int multiprocessorCount, NVTETensor workspace, NVTETensor barrier) {
  NVTE_API_CALL(nvte_rmsnorm_bwd);
  using namespace transformer_engine;
  rmsnorm_bwd(*reinterpret_cast<const Tensor *>(dz), *reinterpret_cast<const Tensor *>(x),
              *reinterpret_cast<const Tensor *>(rsigma), *reinterpret_cast<const Tensor *>(gamma),
              reinterpret_cast<Tensor *>(dx), reinterpret_cast<Tensor *>(dgamma),
              reinterpret_cast<Tensor *>(dgamma_part), stream, multiprocessorCount,
              reinterpret_cast<Tensor *>(workspace), reinterpret_cast<Tensor *>(barrier), false);
}

void nvte_rmsnorm1p_fwd(const NVTETensor x,      // Nxhidden_size
                        const NVTETensor gamma,  // hidden_size
                        const float epsilon, NVTETensor z, NVTETensor rsigma, cudaStream_t stream,
                        const int multiprocessorCount, NVTETensor workspace, NVTETensor barrier) {
  NVTE_API_CALL(nvte_rmsnorm1p_fwd);
  using namespace transformer_engine;
  rmsnorm_fwd(*reinterpret_cast<const Tensor *>(x), *reinterpret_cast<const Tensor *>(gamma),
              epsilon, reinterpret_cast<Tensor *>(z), reinterpret_cast<Tensor *>(rsigma), stream,
              multiprocessorCount, reinterpret_cast<Tensor *>(workspace),
              reinterpret_cast<Tensor *>(barrier), true);
}

void nvte_rmsnorm1p_bwd(const NVTETensor dz,      // Nxhidden_size
                        const NVTETensor x,       // Nxhidden_size
                        const NVTETensor rsigma,  // N, FP32!
                        const NVTETensor gamma,   // hidden_size
                        NVTETensor dx, NVTETensor dgamma, NVTETensor dgamma_part,
                        cudaStream_t stream, const int multiprocessorCount, NVTETensor workspace,
                        NVTETensor barrier) {
  NVTE_API_CALL(nvte_rmsnorm1p_bwd);
  using namespace transformer_engine;
  rmsnorm_bwd(*reinterpret_cast<const Tensor *>(dz), *reinterpret_cast<const Tensor *>(x),
              *reinterpret_cast<const Tensor *>(rsigma), *reinterpret_cast<const Tensor *>(gamma),
              reinterpret_cast<Tensor *>(dx), reinterpret_cast<Tensor *>(dgamma),
              reinterpret_cast<Tensor *>(dgamma_part), stream, multiprocessorCount,
              reinterpret_cast<Tensor *>(workspace), reinterpret_cast<Tensor *>(barrier), true);
}

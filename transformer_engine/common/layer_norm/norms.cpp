/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/* #include <transformer_engine/layer_norm.h> */

#include "norms.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>

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

// Create registries and provide runtime versions of config hash functions.
FwdTunedRegistry LN_FWD_TUNED_FUNCS;
BwdTunedRegistry LN_BWD_TUNED_FUNCS;
FwdGeneralRegistry LN_FWD_GENERAL_FUNCS;
BwdGeneralRegistry LN_BWD_GENERAL_FUNCS;

FwdTunedRegistry RMS_FWD_TUNED_FUNCS;
BwdTunedRegistry RMS_BWD_TUNED_FUNCS;
FwdGeneralRegistry RMS_FWD_GENERAL_FUNCS;
BwdGeneralRegistry RMS_BWD_GENERAL_FUNCS;

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

uint64_t get_key(DType wtype, DType itype, DType otype, DType ctype, uint64_t hidden_size) {
  uint64_t type_key = get_type_id(wtype) | (get_type_id(itype) << 2) | (get_type_id(otype) << 4) |
                      (get_type_id(ctype) << 6);
  uint64_t launcher_key = (type_key << 32) | hidden_size;
  return launcher_key;
}

template <NVTE_NORM_TYPE NormEnum>
FwdFunction& get_fwd_launcher(DType wtype, DType itype, DType otype, DType ctype,
                              const FwdParams& params) {
  if constexpr (!IF_TE_FWD_NORMS<NormEnum>()) NVTE_ERROR("Unexpected NVTE_NORM_TYPE!");

  auto& FWD_TUNED_FUNCS = GET_REGISTRY<NormEnum, true>();
  auto& FWD_GENERAL_FUNCS = GET_REGISTRY<NormEnum, false>();

  // Look for tuned kernel
  auto tuned_key = get_key(wtype, itype, otype, ctype, params.cols);
  auto is_aligned = [](const void* ptr) -> bool {
    // Assume vectorized memory accesses are <=16B
    return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
  };
  if (params.rows % 4 == 0 && is_aligned(params.x) && is_aligned(params.rs) &&
      is_aligned(params.gamma) && is_aligned(params.z) && FWD_TUNED_FUNCS.count(tuned_key) > 0) {
    if constexpr (NormEnum == NVTE_NORM_TYPE::LN_FWD_TE) {
      if (is_aligned(params.mu) && is_aligned(params.beta)) return FWD_TUNED_FUNCS.at(tuned_key);
    } else
      return FWD_TUNED_FUNCS.at(tuned_key);
  }

  // Pick general kernel
  auto general_key = get_key(wtype, itype, otype, ctype, 0);
  if (FWD_GENERAL_FUNCS.count(general_key) == 0) {
    NVTE_ERROR("FWD: Unsupported types.");
  }
  auto& general_func_map = FWD_GENERAL_FUNCS.at(general_key);
  auto func_iter = general_func_map.lower_bound(params.cols);
  if (func_iter == general_func_map.end()) {
    // Hidden size is too big, need to use multi-CTA
    return general_func_map.rbegin()->second;
  } else {
    return func_iter->second;
  }
}

template <NVTE_NORM_TYPE NormEnum>
BwdFunction& get_bwd_launcher(DType wtype, DType itype, DType otype, DType ctype,
                              const BwdParams& params) {
  if constexpr (!IF_TE_BWD_NORMS<NormEnum>()) NVTE_ERROR("Unexpected NVTE_NORM_TYPE!");

  auto& BWD_TUNED_FUNCS = GET_REGISTRY<NormEnum, true>();
  auto& BWD_GENERAL_FUNCS = GET_REGISTRY<NormEnum, false>();

  // Look for tuned kernel
  auto tuned_key = get_key(wtype, itype, otype, ctype, params.cols);
  auto is_aligned = [](const void* ptr) -> bool {
    // Assume vectorized memory accesses are <=16B
    return reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
  };
  if (params.rows % 4 == 0 && is_aligned(params.x) && is_aligned(params.rs) &&
      is_aligned(params.gamma) && is_aligned(params.dz) && is_aligned(params.dx) &&
      is_aligned(params.dgamma) && is_aligned(params.dgamma_part) &&
      BWD_TUNED_FUNCS.count(tuned_key) > 0) {
    if constexpr (NormEnum == NVTE_NORM_TYPE::LN_BWD_TE) {
      if (is_aligned(params.mu) && is_aligned(params.dbeta) && is_aligned(params.dbeta_part))
        return BWD_TUNED_FUNCS.at(tuned_key);

    } else
      return BWD_TUNED_FUNCS.at(tuned_key);
  }

  // Pick general kernel
  auto general_key = get_key(wtype, itype, otype, ctype, 0);
  if (BWD_GENERAL_FUNCS.count(general_key) == 0) {
    NVTE_ERROR("BWD: Unsupported types.");
  }
  auto& general_func_map = BWD_GENERAL_FUNCS.at(general_key);
  auto func_iter = general_func_map.lower_bound(params.cols);
  if (func_iter == general_func_map.end()) {
    // Hidden size is too big, need to use multi-CTA
    return general_func_map.rbegin()->second;
  } else {
    return func_iter->second;
  }
}

static cudnn_frontend::DataType_t te2cudnnDtype(DType teDtype) {
  namespace fe = cudnn_frontend;
  const std::unordered_map<DType, fe::DataType_t> typeMap = {
      {DType::kFloat32, fe::DataType_t::FLOAT},
      {DType::kFloat16, fe::DataType_t::HALF},
      {DType::kBFloat16, fe::DataType_t::BFLOAT16},
      {DType::kByte, fe::DataType_t::INT8},
      {DType::kInt32, fe::DataType_t::INT32},
      {DType::kInt64, fe::DataType_t::INT64},
      {DType::kFloat8E4M3, fe::DataType_t::FP8_E4M3},
      {DType::kFloat8E5M2, fe::DataType_t::FP8_E5M2},
  };

  const auto it = typeMap.find(teDtype);
  if (it != typeMap.end()) {
    return it->second;
  } else {
    return fe::DataType_t::NOT_SET;
  }
}

template <NVTE_NORM_TYPE NormEnum>
NormFwdTe<NormEnum>::NormFwdTe() {
  if constexpr (NormEnum == NVTE_NORM_TYPE::LN_FWD_TE) {
    NVTE_ERROR("NormFwdTe default constructor is only for its inherited classes!");
  }
}

template <NVTE_NORM_TYPE NormEnum>
NormFwdTe<NormEnum>::NormFwdTe(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                               const float epsilon, Tensor* z, Tensor* mu, Tensor* rsigma,
                               cudaStream_t stream, const int multiprocessorCount,
                               Tensor* workspace, Tensor* barrier, const bool zero_centered_gamma) {
  if constexpr (!IF_TE_FWD_NORMS<NormEnum>()) {
    NVTE_ERROR("Unexpected NVTE_NORM_TYPE!");
  }
  _launch_params.multiprocessorCount = multiprocessorCount;
  _launch_params.stream = stream;

  // Set the kernel runtime parameters.
  auto& params = _launch_params.params;
  params.rows = x.data.shape[0];
  params.cols = x.data.shape[1];
  params.x = x.data.dptr;
  params.rs = rsigma->data.dptr;
  params.gamma = gamma.data.dptr;
  params.z = z->data.dptr;
  params.epsilon = epsilon;
  params.amax = z->amax.dptr;
  params.amax_byte_size = product(z->amax.shape) * typeToSize(z->amax.dtype);
  params.scale = z->scale.dptr;
  params.scale_inv = z->scale_inv.dptr;
  params.scale_byte_size = product(z->scale.shape) * typeToSize(z->scale.dtype);
  params.fp8_out = is_fp8_dtype(z->data.dtype);
  params.zero_centered_gamma = zero_centered_gamma;
  if constexpr (NormEnum == NVTE_NORM_TYPE::LN_FWD_TE) {
    params.mu = mu->data.dptr;
    params.beta = beta.data.dptr;
  }

  // Request the kernel launcher.
  _launcher = get_fwd_launcher<NormEnum>(gamma.data.dtype,  // wtype
                                         x.data.dtype,      // itype,
                                         z->data.dtype,     // otype,
                                         DType::kFloat32,   // ctype,
                                         params);
  if (params.fp8_out) set_amax();
}

/*** BWD TE ***/
template <NVTE_NORM_TYPE NormEnum>
NormBwdTe<NormEnum>::NormBwdTe(const Tensor& dz, const Tensor& x, const Tensor& mu,
                               const Tensor& rsigma, const Tensor& gamma, Tensor* dx,
                               Tensor* dgamma, Tensor* dbeta, Tensor* dgamma_part,
                               Tensor* dbeta_part, cudaStream_t stream,
                               const int multiprocessorCount, Tensor* workspace, Tensor* barrier,
                               const bool zero_centered_gamma)
    : NormFwdTe<NormEnum>::NormFwdTe() {
  if constexpr (!IF_TE_BWD_NORMS<NormEnum>()) NVTE_ERROR("Unexpected NVTE_NORM_TYPE!");

  auto& _launch_params = NormFwdTe<NormEnum>::_launch_params;
  _launch_params.stream = stream;
  _launch_params.multiprocessorCount = multiprocessorCount;

  // Set the kernel runtime parameters.
  auto& params = _launch_params.params;
  params.rows = x.data.shape[0];
  params.cols = x.data.shape[1];
  params.x = x.data.dptr;
  params.rs = rsigma.data.dptr;
  params.gamma = gamma.data.dptr;
  params.dz = dz.data.dptr;
  params.dx = dx->data.dptr;
  params.dgamma = dgamma->data.dptr;
  params.dgamma_part = dgamma_part->data.dptr;
  params.zero_centered_gamma = zero_centered_gamma;

  if constexpr (NormEnum == NVTE_NORM_TYPE::LN_BWD_TE) {
    params.mu = mu.data.dptr;
    params.dbeta = dbeta->data.dptr;
    params.dbeta_part = dbeta_part->data.dptr;
  }

  NormFwdTe<NormEnum>::_launcher = get_bwd_launcher<NormEnum>(gamma.data.dtype,  // wtype,
                                                              x.data.dtype,      // itype,
                                                              gamma.data.dtype,  // otype,
                                                              DType::kFloat32,   // ctype,
                                                              params);
}

template <NVTE_NORM_TYPE NormEnum>
void NormFwdTe<NormEnum>::initialize() {
  // Query the kernel-specific launch parameters.
  _launcher(_launch_params, true);
  if (_launch_params.workspace_bytes == 0) {
    _launch_params.workspace_bytes = 1;
  }
}

template <NVTE_NORM_TYPE NormEnum>
void NormFwdTe<NormEnum>::set_workspace_and_barrier(void* workspace_ptr, void* barrier_ptr) {
  NVTE_CHECK(_launch_params.workspace_bytes);
  _launch_params.params.workspace = workspace_ptr;

  if (_launch_params.barrier_size > 0) {
    _launch_params.params.barrier = reinterpret_cast<int*>(barrier_ptr);
    cudaMemsetAsync(_launch_params.params.barrier, 0,
                    _launch_params.barrier_size * typeToSize(DType::kFloat32),
                    _launch_params.stream);
  }
}

template <NVTE_NORM_TYPE NormEnum>
void NormFwdTe<NormEnum>::set_amax() {
  cudaMemsetAsync(_launch_params.params.amax, 0, _launch_params.params.amax_byte_size,
                  _launch_params.stream);
}

template <NVTE_NORM_TYPE NormEnum>
void NormFwdTe<NormEnum>::execute() {
  _launcher(_launch_params, false);
}

template <NVTE_NORM_TYPE NormEnum>
std::vector<size_t> NormFwdTe<NormEnum>::get_workspace_shape() {
  return {_launch_params.workspace_bytes};
}

template <NVTE_NORM_TYPE NormEnum>
std::vector<size_t> NormFwdTe<NormEnum>::get_barrier_shape() {
  return {_launch_params.barrier_size};
}

template <NVTE_NORM_TYPE NormEnum>
std::vector<size_t> NormBwdTe<NormEnum>::get_dgamma_shape() {
  if constexpr (!IF_TE_BWD_NORMS<NormEnum>()) NVTE_ERROR("Unexpected NVTE_NORM_TYPE!");
  return {static_cast<size_t>(NormBwdTe<NormEnum>::_launch_params.params.ctas_per_col),
          static_cast<size_t>(NormBwdTe<NormEnum>::_launch_params.params.cols)};
}

template <NVTE_NORM_TYPE NormEnum>
NormFwdCudnn<NormEnum>::NormFwdCudnn() {
  if constexpr (NormEnum == NVTE_NORM_TYPE::LN_FWD_CUDNN) {
    NVTE_ERROR("NormFwdCudnn default constructor is only for its inherited classes!");
  }
  NVTE_CHECK_CUDNN(cudnnCreate(&_handle));
}

template <NVTE_NORM_TYPE NormEnum>
NormFwdCudnn<NormEnum>::NormFwdCudnn(const Tensor& x, const Tensor& gamma, const Tensor& beta,
                                     const float epsilon, Tensor* z, Tensor* mu, Tensor* rsigma,
                                     cudaStream_t stream, const int multiprocessorCount,
                                     Tensor* workspace, const bool zero_centered_gamma) {
  static_assert(CUDNN_FRONTEND_VERSION >= 10601,
                "CUDNN_FRONTEND_VERSION should be at least 1.6.1!");
  if constexpr (!IF_CUDNN_FWD_NORMS<NormEnum>()) NVTE_ERROR("Unexpected NVTE_NORM_TYPE!");
  NVTE_CHECK_CUDNN(cudnnCreate(&_handle));

  namespace fe = cudnn_frontend;
  const auto itype = x.data.dtype;
  const auto wtype = gamma.data.dtype;
  const auto otype = z->data.dtype;
  const bool fp8_out = is_fp8_dtype(otype);
  const auto ctype = DType::kFloat32;

  _epsilon = epsilon;
  _scalar_offset = 1.0f;

  _graph.set_io_data_type(te2cudnnDtype(itype))
      .set_intermediate_data_type(te2cudnnDtype(ctype))
      .set_compute_data_type(te2cudnnDtype(ctype));

  if (cudnnGetVersion() >= 90400) _graph.set_sm_count(multiprocessorCount);

  const auto batch_dim = static_cast<int32_t>(x.data.shape[0]);
  const auto hidden_dim = static_cast<int32_t>(x.data.shape[1]);

  auto x_tensor = _graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("X")
                                    .set_dim({batch_dim, hidden_dim, 1, 1})
                                    .set_stride({hidden_dim, 1, hidden_dim, hidden_dim}));
  auto gamma_zero_tensor = _graph.tensor(fe::graph::Tensor_attributes()
                                             .set_name("gamma_zero")
                                             .set_dim({1, hidden_dim, 1, 1})
                                             .set_stride({hidden_dim, 1, hidden_dim, hidden_dim})
                                             .set_data_type(te2cudnnDtype(wtype)));
  auto beta_tensor = _graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("bias")
                                       .set_dim({1, hidden_dim, 1, 1})
                                       .set_stride({hidden_dim, 1, hidden_dim, hidden_dim})
                                       .set_data_type(te2cudnnDtype(wtype)));
  auto eps_tensor = _graph.tensor(fe::graph::Tensor_attributes()
                                      .set_name("epsilon")
                                      .set_dim({1, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(te2cudnnDtype(ctype)));

  std::shared_ptr<fe::graph::Tensor_attributes> gamma_tensor, one_tensor;
  if (zero_centered_gamma) {
    one_tensor = _graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("one")
                                   .set_dim({1, 1, 1, 1})
                                   .set_stride({1, 1, 1, 1})
                                   .set_data_type(te2cudnnDtype(wtype))
                                   .set_is_pass_by_value(true));
    auto centered_options = fe::graph::Pointwise_attributes()
                                .set_mode(fe::PointwiseMode_t::ADD)
                                .set_compute_data_type(te2cudnnDtype(ctype));
    gamma_tensor = _graph.pointwise(gamma_zero_tensor, one_tensor, centered_options);
    gamma_tensor->set_output(false).set_data_type(te2cudnnDtype(wtype));
  } else
    gamma_tensor = gamma_zero_tensor;

  std::shared_ptr<fe::graph::Tensor_attributes> z_tensor, mean_tensor, inv_var_tensor;
  if constexpr (NormEnum == NVTE_NORM_TYPE::LN_FWD_CUDNN) {
    auto norm_options = fe::graph::Layernorm_attributes()
                            .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
                            .set_epsilon(eps_tensor);
    auto ret = _graph.layernorm(x_tensor, gamma_tensor, beta_tensor, norm_options);
    std::tie(z_tensor, mean_tensor, inv_var_tensor) = std::make_tuple(ret[0], ret[1], ret[2]);
    mean_tensor->set_output(true).set_data_type(te2cudnnDtype(ctype));
  } else {
    auto norm_options = fe::graph::Rmsnorm_attributes()
                            .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
                            .set_epsilon(eps_tensor);
    auto ret = _graph.rmsnorm(x_tensor, gamma_tensor, norm_options);
    std::tie(z_tensor, inv_var_tensor) = std::make_tuple(ret[0], ret[1]);
  }

  inv_var_tensor->set_output(true).set_data_type(te2cudnnDtype(ctype));

  const auto ZDtype = fp8_out ? ctype : otype;
  z_tensor->set_output(!fp8_out).set_data_type(te2cudnnDtype(ZDtype));

  std::shared_ptr<fe::graph::Tensor_attributes> z_fp8_tensor, amax_tensor, z_scale_tensor;
  fe::graph::Pointwise_attributes z_scale_options;
  if (fp8_out) {
    // create a scale node
    z_scale_tensor = _graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("z_scale")
                                       .set_dim({1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(te2cudnnDtype(ctype)));
    z_scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
    z_fp8_tensor = _graph.pointwise(z_tensor, z_scale_tensor, z_scale_options);

    z_fp8_tensor->set_output(true).set_data_type(te2cudnnDtype(otype));

    // create an amax reduction node
    amax_tensor = _graph.reduction(z_tensor, fe::graph::Reduction_attributes()
                                                 .set_mode(fe::ReductionMode_t::AMAX)
                                                 .set_compute_data_type(te2cudnnDtype(ctype)));
    amax_tensor->set_output(true).set_data_type(te2cudnnDtype(ctype)).set_dim({1, 1, 1, 1});
  }

  _variant_pack = {
      {x_tensor, x.data.dptr},
      {inv_var_tensor, rsigma->data.dptr},
      {eps_tensor, reinterpret_cast<void*>(&_epsilon)},
  };

  if (zero_centered_gamma) {
    _variant_pack.insert({{one_tensor, reinterpret_cast<void*>(&_scalar_offset)},
                          {gamma_zero_tensor, gamma.data.dptr}});
  } else {
    _variant_pack.insert({{gamma_tensor, gamma.data.dptr}});
  }

  if constexpr (NormEnum == NVTE_NORM_TYPE::LN_FWD_CUDNN)
    _variant_pack.insert({
        {mean_tensor, mu->data.dptr},
        {beta_tensor, beta.data.dptr},
    });

  if (fp8_out)
    _variant_pack.insert({
        {z_scale_tensor, z->scale.dptr},
        {amax_tensor, z->amax.dptr},
        {z_fp8_tensor, z->data.dptr},
    });
  else
    _variant_pack.insert({{z_tensor, z->data.dptr}});
}

template <NVTE_NORM_TYPE NormEnum>
NormBwdCudnn<NormEnum>::NormBwdCudnn(const Tensor& dz, const Tensor& x, const Tensor& mu,
                                     const Tensor& rsigma, const Tensor& gamma, Tensor* dx,
                                     Tensor* dgamma, Tensor* dbeta, cudaStream_t stream,
                                     const int multiprocessorCount, Tensor* workspace,
                                     const bool zero_centered_gamma)
    : NormFwdCudnn<NormEnum>::NormFwdCudnn() {
  static_assert(CUDNN_FRONTEND_VERSION >= 10601,
                "CUDNN_FRONTEND_VERSION should be at least 1.6.1!");
  if constexpr (!IF_CUDNN_BWD_NORMS<NormEnum>()) NVTE_ERROR("Unexpected NVTE_NORM_TYPE!");

  auto& _graph = NormBwdCudnn<NormEnum>::_graph;
  namespace fe = cudnn_frontend;
  const auto otype = gamma.data.dtype;
  const auto wtype = gamma.data.dtype;
  const auto ctype = DType::kFloat32;

  this->_scalar_offset = 1.0f;

  _graph.set_io_data_type(te2cudnnDtype(x.data.dtype))
      .set_intermediate_data_type(te2cudnnDtype(ctype))
      .set_compute_data_type(te2cudnnDtype(ctype));

  if (cudnnGetVersion() >= 90400) _graph.set_sm_count(multiprocessorCount);

  const auto batch_dim = static_cast<int32_t>(x.data.shape[0]);
  const auto hidden_dim = static_cast<int32_t>(gamma.data.shape[0]);

  auto x_tensor = _graph.tensor(fe::graph::Tensor_attributes()
                                    .set_name("x")
                                    .set_dim({batch_dim, hidden_dim, 1, 1})
                                    .set_stride({hidden_dim, 1, hidden_dim, hidden_dim}));
  auto dz_tensor = _graph.tensor(fe::graph::Tensor_attributes()
                                     .set_name("dz")
                                     .set_dim({batch_dim, hidden_dim, 1, 1})
                                     .set_stride({hidden_dim, 1, hidden_dim, hidden_dim}));
  auto mean_tensor = _graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("mean")
                                       .set_dim({batch_dim, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(te2cudnnDtype(ctype)));
  auto gamma_zero_tensor = _graph.tensor(fe::graph::Tensor_attributes()
                                             .set_name("gamma")
                                             .set_dim({1, hidden_dim, 1, 1})
                                             .set_stride({hidden_dim, 1, hidden_dim, hidden_dim})
                                             .set_data_type(te2cudnnDtype(wtype)));
  auto inv_var_tensor = _graph.tensor(fe::graph::Tensor_attributes()
                                          .set_name("inv_var")
                                          .set_dim({batch_dim, 1, 1, 1})
                                          .set_stride({1, 1, 1, 1})
                                          .set_data_type(te2cudnnDtype(ctype)));
  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> gamma_tensor, one_tensor;
  if (zero_centered_gamma) {
    one_tensor = _graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("one")
                                   .set_dim({1, 1, 1, 1})
                                   .set_stride({1, 1, 1, 1})
                                   .set_data_type(te2cudnnDtype(wtype))
                                   .set_is_pass_by_value(true));
    auto centered_options = fe::graph::Pointwise_attributes()
                                .set_mode(fe::PointwiseMode_t::ADD)
                                .set_compute_data_type(te2cudnnDtype(ctype));
    gamma_tensor = _graph.pointwise(gamma_zero_tensor, one_tensor, centered_options);
    gamma_tensor->set_output(false).set_data_type(te2cudnnDtype(wtype));
  } else
    gamma_tensor = gamma_zero_tensor;

  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> dx_tensor, dgamma_tensor, dbeta_tensor;
  if constexpr (NormEnum == NVTE_NORM_TYPE::LN_BWD_CUDNN) {
    auto norm_options = fe::graph::Layernorm_backward_attributes().set_saved_mean_and_inv_variance(
        mean_tensor, inv_var_tensor);
    auto ret = _graph.layernorm_backward(dz_tensor, x_tensor, gamma_tensor, norm_options);
    std::tie(dx_tensor, dgamma_tensor, dbeta_tensor) = std::make_tuple(ret[0], ret[1], ret[2]);
    dbeta_tensor->set_output(true).set_data_type(te2cudnnDtype(otype));
  } else {
    auto norm_options = fe::graph::Rmsnorm_backward_attributes().has_dbias(false);
    auto ret =
        _graph.rmsnorm_backward(dz_tensor, x_tensor, gamma_tensor, inv_var_tensor, norm_options);
    std::tie(dx_tensor, dgamma_tensor, dbeta_tensor) = std::make_tuple(ret[0], ret[1], ret[2]);
    if (dgamma_tensor != nullptr) NVTE_ERROR("cuDNN rmsnorm dbias incorrectly returned.");
  }
  dx_tensor->set_output(true).set_data_type(te2cudnnDtype(otype));
  dgamma_tensor->set_output(true).set_data_type(te2cudnnDtype(otype));

  NormBwdCudnn<NormEnum>::_variant_pack = {{x_tensor, x.data.dptr},
                                           {dz_tensor, dz.data.dptr},
                                           {inv_var_tensor, rsigma.data.dptr},
                                           {dgamma_tensor, dgamma->data.dptr},
                                           {dx_tensor, dx->data.dptr}};

  float offset_scalar = 1.0f;
  if (zero_centered_gamma) {
    NormBwdCudnn<NormEnum>::_variant_pack.insert(
        {{one_tensor, reinterpret_cast<void*>(&this->_scalar_offset)},
         {gamma_zero_tensor, gamma.data.dptr}});
  } else {
    NormBwdCudnn<NormEnum>::_variant_pack.insert({{gamma_tensor, gamma.data.dptr}});
  }

  if constexpr (NormEnum == NVTE_NORM_TYPE::LN_BWD_CUDNN)
    NormBwdCudnn<NormEnum>::_variant_pack.insert({
        {mean_tensor, mu.data.dptr},
        {dbeta_tensor, dbeta->data.dptr},
    });
}

template <NVTE_NORM_TYPE NormEnum>
void NormFwdCudnn<NormEnum>::initialize() {
  NVTE_CHECK(_graph.validate().is_good());
  NVTE_CHECK(_graph.build_operation_graph(_handle).is_good());
  NVTE_CHECK(_graph.create_execution_plans({cudnn_frontend::HeurMode_t::FALLBACK}).is_good());
  NVTE_CHECK(_graph.check_support(_handle).is_good());
  NVTE_CHECK(_graph.build_plans(_handle).is_good());
}

template <NVTE_NORM_TYPE NormEnum>
void NormFwdCudnn<NormEnum>::set_workspace_and_barrier(void* workspace_ptr, void* barrier_ptr) {
  _workspace = workspace_ptr;
}

template <NVTE_NORM_TYPE NormEnum>
void NormFwdCudnn<NormEnum>::execute() {
  NVTE_CHECK(_graph.execute(_handle, _variant_pack, _workspace).is_good());
}

template <NVTE_NORM_TYPE NormEnum>
std::vector<size_t> NormFwdCudnn<NormEnum>::get_workspace_shape() {
  return {static_cast<size_t>(_graph.get_workspace_size())};
}

template <NVTE_NORM_TYPE NormEnum, typename NormType>
void norms_launcher(NormType& Norm, Tensor* workspace, Tensor* barrier, Tensor* dgamma_part,
                    Tensor* dbeta_part) {
  Norm.initialize();

  // Populate shape and dtypes for FW to allocate memory
  void* test_ptr = IF_TE_BWD_NORMS<NormEnum>() ? dgamma_part->data.dptr : workspace->data.dptr;
  if (test_ptr == nullptr) {
    if constexpr (IF_TE_BWD_NORMS<NormEnum>()) {
      NVTE_CHECK(dgamma_part->data.dptr == nullptr);
      dgamma_part->data.dtype = DType::kFloat32;
      dgamma_part->data.shape = Norm.get_dgamma_shape();
    }
    if constexpr (NormEnum == NVTE_NORM_TYPE::LN_BWD_TE) {
      NVTE_CHECK(dbeta_part->data.dptr == nullptr);
      dbeta_part->data.dtype = DType::kFloat32;
      dbeta_part->data.shape = Norm.get_dgamma_shape();
    }
    if constexpr (IF_TE_NORMS<NormEnum>()) {
      barrier->data.dtype = DType::kInt32;
      barrier->data.shape = Norm.get_barrier_shape();
    }
    workspace->data.dtype = DType::kByte;
    workspace->data.shape = Norm.get_workspace_shape();

    return;
  } else {
    if constexpr (IF_TE_BWD_NORMS<NormEnum>()) {
      NVTE_CHECK(dgamma_part->data.dtype == DType::kFloat32);
      NVTE_CHECK(dgamma_part->data.shape == Norm.get_dgamma_shape());
    }
    if constexpr (NormEnum == NVTE_NORM_TYPE::LN_BWD_TE) {
      NVTE_CHECK(dbeta_part->data.dptr != nullptr);
      NVTE_CHECK(dbeta_part->data.dtype == DType::kFloat32);
      NVTE_CHECK(dbeta_part->data.shape == Norm.get_dgamma_shape());
    }
    if constexpr (IF_TE_NORMS<NormEnum>()) {
      NVTE_CHECK(barrier->data.dtype == DType::kInt32);
      NVTE_CHECK(barrier->data.shape == Norm.get_barrier_shape());
    }
    NVTE_CHECK(workspace->data.dtype == DType::kByte);
    NVTE_CHECK(workspace->data.shape == Norm.get_workspace_shape());
  }

  auto barrier_ptr = barrier != nullptr ? barrier->data.dptr : nullptr;
  Norm.set_workspace_and_barrier(workspace->data.dptr, barrier_ptr);

  Norm.execute();
}

template class NormFwdTe<NVTE_NORM_TYPE::LN_FWD_TE>;
template class NormFwdTe<NVTE_NORM_TYPE::RMS_FWD_TE>;
template class NormBwdTe<NVTE_NORM_TYPE::LN_BWD_TE>;
template class NormBwdTe<NVTE_NORM_TYPE::RMS_BWD_TE>;

template class NormFwdCudnn<NVTE_NORM_TYPE::LN_FWD_CUDNN>;
template class NormFwdCudnn<NVTE_NORM_TYPE::RMS_FWD_CUDNN>;
template class NormBwdCudnn<NVTE_NORM_TYPE::LN_BWD_CUDNN>;
template class NormBwdCudnn<NVTE_NORM_TYPE::RMS_BWD_CUDNN>;

template void norms_launcher<NVTE_NORM_TYPE::LN_FWD_TE, NormFwdTe<NVTE_NORM_TYPE::LN_FWD_TE>>(
    NormFwdTe<NVTE_NORM_TYPE::LN_FWD_TE>&, Tensor*, Tensor*, Tensor*, Tensor*);
template void norms_launcher<NVTE_NORM_TYPE::LN_BWD_TE, NormBwdTe<NVTE_NORM_TYPE::LN_BWD_TE>>(
    NormBwdTe<NVTE_NORM_TYPE::LN_BWD_TE>&, Tensor*, Tensor*, Tensor*, Tensor*);
template void norms_launcher<NVTE_NORM_TYPE::RMS_FWD_TE, NormFwdTe<NVTE_NORM_TYPE::RMS_FWD_TE>>(
    NormFwdTe<NVTE_NORM_TYPE::RMS_FWD_TE>&, Tensor*, Tensor*, Tensor*, Tensor*);
template void norms_launcher<NVTE_NORM_TYPE::RMS_BWD_TE, NormBwdTe<NVTE_NORM_TYPE::RMS_BWD_TE>>(
    NormBwdTe<NVTE_NORM_TYPE::RMS_BWD_TE>&, Tensor*, Tensor*, Tensor*, Tensor*);

template void
norms_launcher<NVTE_NORM_TYPE::LN_FWD_CUDNN, NormFwdCudnn<NVTE_NORM_TYPE::LN_FWD_CUDNN>>(
    NormFwdCudnn<NVTE_NORM_TYPE::LN_FWD_CUDNN>&, Tensor*, Tensor*, Tensor*, Tensor*);
template void
norms_launcher<NVTE_NORM_TYPE::LN_BWD_CUDNN, NormBwdCudnn<NVTE_NORM_TYPE::LN_BWD_CUDNN>>(
    NormBwdCudnn<NVTE_NORM_TYPE::LN_BWD_CUDNN>&, Tensor*, Tensor*, Tensor*, Tensor*);
template void
norms_launcher<NVTE_NORM_TYPE::RMS_FWD_CUDNN, NormFwdCudnn<NVTE_NORM_TYPE::RMS_FWD_CUDNN>>(
    NormFwdCudnn<NVTE_NORM_TYPE::RMS_FWD_CUDNN>&, Tensor*, Tensor*, Tensor*, Tensor*);
template void
norms_launcher<NVTE_NORM_TYPE::RMS_BWD_CUDNN, NormBwdCudnn<NVTE_NORM_TYPE::RMS_BWD_CUDNN>>(
    NormBwdCudnn<NVTE_NORM_TYPE::RMS_BWD_CUDNN>&, Tensor*, Tensor*, Tensor*, Tensor*);

}  // namespace transformer_engine

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

uint64_t get_key(NVTE_Norm_Type NormType, NVTE_Norm_Stage NormStage, DType wtype, DType itype,
                 DType otype, DType ctype, uint64_t batch_size, uint64_t hidden_size,
                 bool zero_centered_gamma) {
  uint64_t type_key = get_type_id(wtype) | (get_type_id(itype) << 2) | (get_type_id(otype) << 4) |
                      (get_type_id(ctype) << 6 | uint32_t(NormType) << 8 |
                       uint32_t(NormStage) << 9 | uint32_t(zero_centered_gamma) << 10);
  // We have 26 bits to hash batch_size or hidden_size. Undefined behavior when these sizes > 2^26
  uint64_t launcher_key = hidden_size | (batch_size << 26) | (type_key << 52);
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

template <NVTE_NORM_TYPE NormEnum, typename NormType>
void norms_launcher(NormType& Norm, Tensor* workspace, Tensor* barrier, Tensor* dgamma_part,
                    Tensor* dbeta_part) {
  Norm.initialize();

  // Populate shape and dtypes for FW to allocate memory
  if (workspace->data.dptr == nullptr) {
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
    barrier->data.dtype = DType::kInt32;
    barrier->data.shape = Norm.get_barrier_shape();
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
    NVTE_CHECK(barrier->data.dtype == DType::kInt32);
    NVTE_CHECK(barrier->data.shape == Norm.get_barrier_shape());
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

template void norms_launcher<NVTE_NORM_TYPE::LN_FWD_TE, NormFwdTe<NVTE_NORM_TYPE::LN_FWD_TE>>(
    NormFwdTe<NVTE_NORM_TYPE::LN_FWD_TE>&, Tensor*, Tensor*, Tensor*, Tensor*);
template void norms_launcher<NVTE_NORM_TYPE::LN_BWD_TE, NormBwdTe<NVTE_NORM_TYPE::LN_BWD_TE>>(
    NormBwdTe<NVTE_NORM_TYPE::LN_BWD_TE>&, Tensor*, Tensor*, Tensor*, Tensor*);
template void norms_launcher<NVTE_NORM_TYPE::RMS_FWD_TE, NormFwdTe<NVTE_NORM_TYPE::RMS_FWD_TE>>(
    NormFwdTe<NVTE_NORM_TYPE::RMS_FWD_TE>&, Tensor*, Tensor*, Tensor*, Tensor*);
template void norms_launcher<NVTE_NORM_TYPE::RMS_BWD_TE, NormBwdTe<NVTE_NORM_TYPE::RMS_BWD_TE>>(
    NormBwdTe<NVTE_NORM_TYPE::RMS_BWD_TE>&, Tensor*, Tensor*, Tensor*, Tensor*);

NormalizationPlan::NormalizationPlan(NVTE_Norm_Type NormType, NVTE_Norm_Stage NormStage,
                                     DType wtype, DType itype, DType otype, DType ctype,
                                     const size_t batch_size, const size_t hidden_size,
                                     const bool zero_centered_gamma, const size_t sm_count)
    : _fp8_out(is_fp8_dtype(otype)), _zero_centered(zero_centered_gamma) {
  static_assert(CUDNN_FRONTEND_VERSION >= 10601,
                "CUDNN_FRONTEND_VERSION should be at least 1.6.1!");

  namespace fe = cudnn_frontend;

  _scalar_dptr = std::make_unique<char[]>(typeToSize(wtype));
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      wtype, cpp_dtype, *(reinterpret_cast<cpp_dtype*>(_scalar_dptr.get())) = (cpp_dtype)1.0f;);

  _handle = cudnnExecutionPlanManager::Instance().GetCudnnHandle();

  _graph.set_io_data_type(get_cudnn_fe_dtype(itype))
      .set_intermediate_data_type(get_cudnn_fe_dtype(ctype))
      .set_compute_data_type(get_cudnn_fe_dtype(ctype));

  if (cudnnGetVersion() >= 90400) _graph.set_sm_count(sm_count);

  const auto batch_dim = static_cast<int32_t>(batch_size);
  const auto hidden_dim = static_cast<int32_t>(hidden_size);

  // Create graph tensors
  _x = _graph.tensor(fe::graph::Tensor_attributes()
                         .set_name("X")
                         .set_dim({batch_dim, hidden_dim, 1, 1})
                         .set_stride({hidden_dim, 1, hidden_dim, hidden_dim})
                         .set_data_type(get_cudnn_fe_dtype(itype)));

  _gamma_zero = _graph.tensor(fe::graph::Tensor_attributes()
                                  .set_name("gamma_zero")
                                  .set_dim({1, hidden_dim, 1, 1})
                                  .set_stride({hidden_dim, 1, hidden_dim, hidden_dim})
                                  .set_data_type(get_cudnn_fe_dtype(wtype)));
  if (zero_centered_gamma) {
    _scalar_offset = _graph.tensor(fe::graph::Tensor_attributes()
                                       .set_name("one")
                                       .set_dim({1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(get_cudnn_fe_dtype(wtype))
                                       .set_is_pass_by_value(true));
    auto centered_options = fe::graph::Pointwise_attributes()
                                .set_mode(fe::PointwiseMode_t::ADD)
                                .set_compute_data_type(get_cudnn_fe_dtype(ctype))
                                .set_compute_data_type(get_cudnn_fe_dtype(ctype));
    _gamma = _graph.pointwise(_gamma_zero, _scalar_offset, centered_options);
    _gamma->set_output(false).set_data_type(get_cudnn_fe_dtype(wtype));
  } else {
    _gamma = _gamma_zero;
  }

  // Create graph computation nodes
  if (NormStage == NVTE_Norm_Stage::Forward) {
    _eps = _graph.tensor(fe::graph::Tensor_attributes()
                             .set_name("epsilon")
                             .set_dim({1, 1, 1, 1})
                             .set_stride({1, 1, 1, 1})
                             .set_data_type(get_cudnn_fe_dtype(ctype))
                             .set_is_pass_by_value(true));
    if (NormType == NVTE_Norm_Type::LayerNorm) {
      _beta = _graph.tensor(fe::graph::Tensor_attributes()
                                .set_name("bias")
                                .set_dim({1, hidden_dim, 1, 1})
                                .set_stride({hidden_dim, 1, hidden_dim, hidden_dim})
                                .set_data_type(get_cudnn_fe_dtype(wtype)));
      auto norm_options = fe::graph::Layernorm_attributes()
                              .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
                              .set_epsilon(_eps);
      auto ret = _graph.layernorm(_x, _gamma, _beta, norm_options);
      std::tie(_z, _mean, _rsigma) = std::make_tuple(ret[0], ret[1], ret[2]);
      _mean->set_output(true).set_data_type(get_cudnn_fe_dtype(ctype));
    } else if (NormType == NVTE_Norm_Type::RMSNorm) {
      auto norm_options = fe::graph::Rmsnorm_attributes()
                              .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
                              .set_epsilon(_eps);
      auto ret = _graph.rmsnorm(_x, _gamma, norm_options);
      std::tie(_z, _rsigma) = std::make_tuple(ret[0], ret[1]);
    }

    _rsigma->set_output(true).set_data_type(get_cudnn_fe_dtype(ctype));

    const auto ZDtype = _fp8_out ? ctype : otype;
    _z->set_output(!_fp8_out).set_data_type(get_cudnn_fe_dtype(ZDtype));

    if (_fp8_out) {
      // create a scale node
      _z_scale = _graph.tensor(fe::graph::Tensor_attributes()
                                   .set_name("z_scale")
                                   .set_dim({1, 1, 1, 1})
                                   .set_stride({1, 1, 1, 1})
                                   .set_data_type(get_cudnn_fe_dtype(ctype)));
      auto z_scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
      _z_fp8 = _graph.pointwise(_z, _z_scale, z_scale_options);

      _z_fp8->set_output(true).set_data_type(get_cudnn_fe_dtype(otype));

      // create an amax reduction node
      _amax = _graph.reduction(_z, fe::graph::Reduction_attributes()
                                       .set_mode(fe::ReductionMode_t::AMAX)
                                       .set_compute_data_type(get_cudnn_fe_dtype(ctype)));
      _amax->set_output(true).set_data_type(get_cudnn_fe_dtype(ctype)).set_dim({1, 1, 1, 1});
    }
  } else {
    _dz = _graph.tensor(fe::graph::Tensor_attributes()
                            .set_name("dz")
                            .set_dim({batch_dim, hidden_dim, 1, 1})
                            .set_stride({hidden_dim, 1, hidden_dim, hidden_dim}));
    _rsigma = _graph.tensor(fe::graph::Tensor_attributes()
                                .set_name("inv_var")
                                .set_dim({batch_dim, 1, 1, 1})
                                .set_stride({1, 1, 1, 1})
                                .set_data_type(get_cudnn_fe_dtype(ctype)));
    _mean = _graph.tensor(fe::graph::Tensor_attributes()
                              .set_name("mean")
                              .set_dim({batch_dim, 1, 1, 1})
                              .set_stride({1, 1, 1, 1})
                              .set_data_type(get_cudnn_fe_dtype(ctype)));
    if (NormType == NVTE_Norm_Type::LayerNorm) {
      auto norm_options =
          fe::graph::Layernorm_backward_attributes().set_saved_mean_and_inv_variance(_mean,
                                                                                     _rsigma);
      auto ret = _graph.layernorm_backward(_dz, _x, _gamma, norm_options);
      std::tie(_dx, _dgamma, _dbeta) = std::make_tuple(ret[0], ret[1], ret[2]);
      _dbeta->set_output(true).set_data_type(get_cudnn_fe_dtype(otype));
    } else {
      auto norm_options = fe::graph::Rmsnorm_backward_attributes().has_dbias(false);
      auto ret = _graph.rmsnorm_backward(_dz, _x, _gamma, _rsigma, norm_options);
      std::tie(_dx, _dgamma, _dbeta) = std::make_tuple(ret[0], ret[1], ret[2]);
      if (_dbeta != nullptr) NVTE_ERROR("cuDNN rmsnorm dbias incorrectly returned.");
    }
    _dx->set_output(true).set_data_type(get_cudnn_fe_dtype(otype));
    _dgamma->set_output(true).set_data_type(get_cudnn_fe_dtype(otype));
  }
  // Build the graph
  this->build();
}

void NormalizationPlan::build() {
  NVTE_CHECK(_graph.validate().is_good());
  NVTE_CHECK(_graph.build_operation_graph(_handle).is_good());
  NVTE_CHECK(_graph
                 .create_execution_plans(
                     {cudnn_frontend::HeurMode_t::A, cudnn_frontend::HeurMode_t::FALLBACK})
                 .is_good());
  NVTE_CHECK(_graph.check_support(_handle).is_good());
  NVTE_CHECK(
      _graph.build_plans(_handle, cudnn_frontend::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());
}

std::vector<size_t> NormalizationPlan::getWorkspaceShape() const {
  return {static_cast<size_t>(_graph.get_workspace_size())};
}

void NormalizationPlan::execute(Tensor* z, void* x_dptr, void* gamma_dptr, void* beta_dptr,
                                void* mean_dptr, void* eps_dptr, void* rsigma_dptr,
                                void* workspace_dptr) {
  // Binding data pointers to graph tensors
  _variant_pack = {{_x, x_dptr}, {_rsigma, rsigma_dptr}, {_eps, eps_dptr}};

  // layernorm should have valid mean_dptr and beta_dptr
  if (mean_dptr && beta_dptr) _variant_pack.insert({{_mean, mean_dptr}, {_beta, beta_dptr}});

  if (_zero_centered)
    _variant_pack.insert(
        {{_scalar_offset, reinterpret_cast<void*>(_scalar_dptr.get())}, {_gamma_zero, gamma_dptr}});
  else
    _variant_pack.insert({{_gamma, gamma_dptr}});

  if (_fp8_out)
    _variant_pack.insert(
        {{_z_scale, z->scale.dptr}, {_amax, z->amax.dptr}, {_z_fp8, z->data.dptr}});
  else
    _variant_pack.insert({{_z, z->data.dptr}});

  // Execute the computation
  NVTE_CHECK(_graph.execute(_handle, _variant_pack, workspace_dptr).is_good());
  update_tensor_scale_inv(z, 0);
}

void NormalizationPlan::execute(void* x_dptr, void* gamma_dptr, void* mean_dptr, void* rsigma_dptr,
                                void* dx_dptr, void* dz_dptr, void* dbeta_dptr, void* dgamma_dptr,
                                void* workspace_dptr) {
  // Binding data pointers to graph tensors
  _variant_pack = {
      {_x, x_dptr}, {_rsigma, rsigma_dptr}, {_dz, dz_dptr}, {_dgamma, dgamma_dptr}, {_dx, dx_dptr}};

  if (_zero_centered)
    _variant_pack.insert({{_scalar_offset, reinterpret_cast<void*>(this->_scalar_dptr.get())},
                          {_gamma_zero, gamma_dptr}});
  else
    _variant_pack.insert({{_gamma, gamma_dptr}});

  // layernorm should have valid mean_dptr and beta_dptr
  if (mean_dptr && dbeta_dptr) _variant_pack.insert({{_mean, mean_dptr}, {_dbeta, dbeta_dptr}});

  // Execute the computation
  NVTE_CHECK(_graph.execute(_handle, _variant_pack, workspace_dptr).is_good());
}

NormalizationPlan* NormalizationPlanRegistry::getNormalizationPlan(
    NVTE_Norm_Type NormType, NVTE_Norm_Stage NormStage, DType wtype, DType itype, DType otype,
    const size_t batch_size, const size_t hidden_size, const bool zero_centered_gamma,
    const size_t sm_count) {
  const DType ctype = DType::kFloat32;
  auto key = get_key(NormType, NormStage, wtype, itype, otype, ctype, batch_size, hidden_size,
                     zero_centered_gamma);

  auto it = normalizationPlanMap.find(key);
  if (it != normalizationPlanMap.end()) {
    return it->second.get();
  }
  auto plan =
      std::make_unique<NormalizationPlan>(NormType, NormStage, wtype, itype, otype, ctype,
                                          batch_size, hidden_size, zero_centered_gamma, sm_count);
  normalizationPlanMap.insert({key, std::move(plan)});
  return normalizationPlanMap[key].get();
}
}  // namespace transformer_engine

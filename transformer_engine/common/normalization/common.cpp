/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/* #include <transformer_engine/layer_norm.h> */

#include "common.h"

#include <bitset>
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
namespace normalization {

TupleKeyType get_key(NVTE_Norm_Type NormType, NVTE_Norm_Stage NormStage, DType wtype, DType itype,
                     DType otype, DType ctype, uint64_t batch_size, uint64_t hidden_size,
                     bool zero_centered_gamma, bool is_tuned) {
  uint64_t general_key = static_cast<uint32_t>(itype) | (static_cast<uint32_t>(otype) << 3) |
                         (static_cast<uint32_t>(ctype) << 6) | (static_cast<uint32_t>(wtype) << 9) |
                         (uint32_t(NormType) << 12) | (uint32_t(NormStage)) << 14 |
                         (uint32_t(zero_centered_gamma) << 16);
  return std::make_tuple(general_key, batch_size, hidden_size, is_tuned);
}

template <typename KernelParamsType>
TeNormalizationPlan<KernelParamsType>::TeNormalizationPlan(
    NVTE_Norm_Type NormType, NVTE_Norm_Stage NormStage, DType wtype, DType itype, DType otype,
    DType ctype, const size_t batch_size, const size_t hidden_size, const size_t sm_count,
    const bool zero_centered_gamma, const bool is_tuned)
    : _is_layernorm(NormType == NVTE_Norm_Type::LayerNorm) {
  _launch_params.multiprocessorCount = sm_count;

  auto& kernel_params = _launch_params.params;
  kernel_params.rows = batch_size;
  kernel_params.cols = hidden_size;
  kernel_params.zero_centered_gamma = zero_centered_gamma;
  if constexpr (std::is_same_v<KernelParamsType, ForwardKernelParams>) {
    kernel_params.fp8_out = is_fp8_dtype(otype);
  }
  // TE kernels have no template for batch_size and zero_centered_gamma, thus zero out those
  auto key =
      get_key(NormType, NormStage, wtype, itype, otype, ctype, 0, hidden_size, false, is_tuned);
  _kernel = KernelRegistry::getKernel(key);

  this->_build();
}

template <>
void TeNormalizationPlan<ForwardKernelParams>::execute(Tensor* z, void* x_dptr, void* gamma_dptr,
                                                       void* beta_dptr, void* mean_dptr,
                                                       void* eps_dptr, void* rsigma_dptr,
                                                       void* workspace_dptr, cudaStream_t stream) {
  _launch_params.stream = stream;

  auto& kernel_params = _launch_params.params;
  kernel_params.workspace = workspace_dptr;
  kernel_params.x = x_dptr;
  kernel_params.rs = rsigma_dptr;
  kernel_params.gamma = gamma_dptr;
  kernel_params.z = z->data.dptr;
  kernel_params.epsilon = *reinterpret_cast<float*>(eps_dptr);
  kernel_params.amax = z->amax.dptr;
  kernel_params.scale = z->scale.dptr;
  kernel_params.scale_inv = z->scale_inv.dptr;

  if (_is_layernorm) {
    kernel_params.mu = mean_dptr;
    kernel_params.beta = beta_dptr;
  }

  _set_workspace();
  _kernel(_launch_params, false);
}

template <>
void TeNormalizationPlan<BackwardKernelParams>::execute(Tensor* z, void* x_dptr, void* gamma_dptr,
                                                        void* beta_dptr, void* mean_dptr,
                                                        void* eps_dptr, void* rsigma_dptr,
                                                        void* workspace_dptr, cudaStream_t stream) {
  NVTE_ERROR("Backward normalization should not call the forward execute function!");
}

template <typename KernelParamsType>
void TeNormalizationPlan<KernelParamsType>::_build() {
  _kernel(_launch_params, true);
  _launch_params.alignWorkspace();
}

template <typename KernelParamsType>
std::vector<size_t> TeNormalizationPlan<KernelParamsType>::getWorkspaceShape() const {
  return {_launch_params.getTotalWorkspaceBytes(_is_layernorm)};
}

template <typename KernelParamsType>
void TeNormalizationPlan<KernelParamsType>::_set_workspace() {
  if (_launch_params.getTotalWorkspaceBytes() > 0) {
    auto workspace_dptr = reinterpret_cast<byte*>(_launch_params.params.workspace);

    if (_launch_params.barrier_bytes > 0) {
      _launch_params.params.barrier =
          reinterpret_cast<int*>(workspace_dptr + _launch_params.workspace_bytes);
      cudaMemsetAsync(_launch_params.params.barrier, 0, _launch_params.barrier_bytes,
                      _launch_params.stream);
    }
    if constexpr (std::is_same_v<KernelParamsType, BackwardKernelParams>) {
      _launch_params.params.dgamma_part =
          workspace_dptr + _launch_params.workspace_bytes + _launch_params.barrier_bytes;
      if (_is_layernorm) {
        _launch_params.params.dbeta_part =
            reinterpret_cast<byte*>(_launch_params.params.dgamma_part) +
            _launch_params.dgamma_part_bytes;
      }
    }
  }
}

template <>
void TeNormalizationPlan<ForwardKernelParams>::execute(void* x_dptr, void* gamma_dptr,
                                                       void* mean_dptr, void* rsigma_dptr,
                                                       void* dx_dptr, void* dz_dptr,
                                                       void* dbeta_dptr, void* dgamma_dptr,
                                                       void* workspace_dptr, cudaStream_t stream) {
  NVTE_ERROR("Forward normalization should not call the backward execute function!");
}

template <>
void TeNormalizationPlan<BackwardKernelParams>::execute(void* x_dptr, void* gamma_dptr,
                                                        void* mean_dptr, void* rsigma_dptr,
                                                        void* dx_dptr, void* dz_dptr,
                                                        void* dbeta_dptr, void* dgamma_dptr,
                                                        void* workspace_dptr, cudaStream_t stream) {
  _launch_params.stream = stream;

  auto& kernel_params = _launch_params.params;
  kernel_params.workspace = workspace_dptr;
  kernel_params.x = x_dptr;
  kernel_params.gamma = gamma_dptr;
  kernel_params.rs = rsigma_dptr;
  kernel_params.dx = dx_dptr;
  kernel_params.dz = dz_dptr;
  kernel_params.dgamma = dgamma_dptr;

  if (_is_layernorm) {
    kernel_params.mu = mean_dptr;
    kernel_params.dbeta = dbeta_dptr;
  }

  _set_workspace();
  _kernel(_launch_params, false);
}

CudnnNormalizationPlan::CudnnNormalizationPlan(NVTE_Norm_Type NormType, NVTE_Norm_Stage NormStage,
                                               DType wtype, DType itype, DType otype, DType ctype,
                                               const size_t batch_size, const size_t hidden_size,
                                               const size_t sm_count,
                                               const bool zero_centered_gamma)
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
                              .set_epsilon(_eps)
                              .set_compute_data_type(get_cudnn_fe_dtype(ctype));
      auto ret = _graph.layernorm(_x, _gamma, _beta, norm_options);
      std::tie(_z, _mean, _rsigma) = std::make_tuple(ret[0], ret[1], ret[2]);
      _mean->set_output(true).set_data_type(get_cudnn_fe_dtype(ctype));
    } else if (NormType == NVTE_Norm_Type::RMSNorm) {
      auto norm_options = fe::graph::Rmsnorm_attributes()
                              .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
                              .set_epsilon(_eps)
                              .set_compute_data_type(get_cudnn_fe_dtype(ctype));
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
      auto z_scale_options = fe::graph::Pointwise_attributes()
                                 .set_mode(fe::PointwiseMode_t::MUL)
                                 .set_compute_data_type(get_cudnn_fe_dtype(ctype));
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
      auto norm_options = fe::graph::Layernorm_backward_attributes()
                              .set_saved_mean_and_inv_variance(_mean, _rsigma)
                              .set_compute_data_type(get_cudnn_fe_dtype(ctype));
      auto ret = _graph.layernorm_backward(_dz, _x, _gamma, norm_options);
      std::tie(_dx, _dgamma, _dbeta) = std::make_tuple(ret[0], ret[1], ret[2]);
      _dbeta->set_output(true).set_data_type(get_cudnn_fe_dtype(otype));
    } else {
      auto norm_options =
          fe::graph::Rmsnorm_backward_attributes().has_dbias(false).set_compute_data_type(
              get_cudnn_fe_dtype(ctype));
      auto ret = _graph.rmsnorm_backward(_dz, _x, _gamma, _rsigma, norm_options);
      std::tie(_dx, _dgamma, _dbeta) = std::make_tuple(ret[0], ret[1], ret[2]);
      if (_dbeta != nullptr) NVTE_ERROR("cuDNN rmsnorm dbias incorrectly returned.");
    }
    _dx->set_output(true).set_data_type(get_cudnn_fe_dtype(otype));
    _dgamma->set_output(true).set_data_type(get_cudnn_fe_dtype(otype));
  }
  // Build the graph
  this->_build();
}

void CudnnNormalizationPlan::_build() {
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

std::vector<size_t> CudnnNormalizationPlan::getWorkspaceShape() const {
  return {static_cast<size_t>(_graph.get_workspace_size())};
}

void CudnnNormalizationPlan::execute(Tensor* z, void* x_dptr, void* gamma_dptr, void* beta_dptr,
                                     void* mean_dptr, void* eps_dptr, void* rsigma_dptr,
                                     void* workspace_dptr, cudaStream_t stream) {
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
  NVTE_CHECK_CUDNN(cudnnSetStream(_handle, stream));
  NVTE_CHECK(_graph.execute(_handle, _variant_pack, workspace_dptr).is_good());
  if (_fp8_out) update_tensor_scale_inv(z, stream);
}

void CudnnNormalizationPlan::execute(void* x_dptr, void* gamma_dptr, void* mean_dptr,
                                     void* rsigma_dptr, void* dx_dptr, void* dz_dptr,
                                     void* dbeta_dptr, void* dgamma_dptr, void* workspace_dptr,
                                     cudaStream_t stream) {
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
  NVTE_CHECK_CUDNN(cudnnSetStream(_handle, stream));
  NVTE_CHECK(_graph.execute(_handle, _variant_pack, workspace_dptr).is_good());
}

NormalizationPlanBase* NormalizationPlanRegistry::getNormalizationPlan(
    NVTE_Norm_Backend NormBackend, NVTE_Norm_Type NormType, NVTE_Norm_Stage NormStage, DType wtype,
    DType itype, DType otype, const size_t batch_size, const size_t hidden_size,
    const size_t sm_count, const bool zero_centered_gamma, const bool is_aligned) {
  const DType ctype = DType::kFloat32;
  bool is_tuned = is_aligned && (batch_size % 4 == 0);
  auto key = get_key(NormType, NormStage, wtype, itype, otype, ctype, batch_size, hidden_size,
                     zero_centered_gamma, is_tuned);

  auto it = normalizationPlanMap.find(key);
  if (it != normalizationPlanMap.end()) {
    return it->second.get();
  }

  std::unique_ptr<NormalizationPlanBase> plan;
  if (NormBackend == NVTE_Norm_Backend::Cudnn) {
    plan = std::make_unique<CudnnNormalizationPlan>(NormType, NormStage, wtype, itype, otype, ctype,
                                                    batch_size, hidden_size, sm_count,
                                                    zero_centered_gamma);
  } else if (NormStage == NVTE_Norm_Stage::Forward) {
    plan = std::make_unique<TeNormalizationPlan<ForwardKernelParams>>(
        NormType, NormStage, wtype, itype, otype, ctype, batch_size, hidden_size, sm_count,
        zero_centered_gamma, is_tuned);
  } else {
    plan = std::make_unique<TeNormalizationPlan<BackwardKernelParams>>(
        NormType, NormStage, wtype, itype, otype, ctype, batch_size, hidden_size, sm_count,
        zero_centered_gamma, is_tuned);
  }
  normalizationPlanMap.insert({key, std::move(plan)});
  return normalizationPlanMap[key].get();
}

bool use_cudnn_norm_fwd() { return transformer_engine::getenv<bool>("NVTE_NORM_FWD_USE_CUDNN"); }
bool use_cudnn_norm_bwd() { return transformer_engine::getenv<bool>("NVTE_NORM_BWD_USE_CUDNN"); }

}  //  namespace normalization
}  // namespace transformer_engine

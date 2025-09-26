/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "transformer_engine/normalization.h"

#include <cuda_runtime.h>

#include "../extensions.h"

namespace transformer_engine {
namespace jax {

pybind11::tuple GetNormForwardWorkspaceSizes(size_t batch_size, size_t hidden_size, DType in_dtype,
                                             DType w_dtype, DType out_dtype,
                                             NVTE_Norm_Type norm_type,
                                             JAXX_Scaling_Mode scaling_mode,
                                             bool zero_centered_gamma, float epsilon, int sm_margin,
                                             bool is_training) {
  auto input_shape = std::vector<size_t>{batch_size, hidden_size};
  auto weight_shape = std::vector<size_t>{hidden_size};
  auto intermediates_shape = std::vector<size_t>{batch_size};

  // empty tensor wrappers are okay just to get workspace size
  auto input_tensor = TensorWrapper(nullptr, input_shape, in_dtype);
  auto gamma_tensor = TensorWrapper(nullptr, weight_shape, w_dtype);
  auto rsigma_tensor = TensorWrapper(nullptr, intermediates_shape, DType::kFloat32);

  auto output_tensor = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
  output_tensor.set_rowwise_data(nullptr, out_dtype, input_shape);

  // WAR: NVTE Norms query the is_training from whereas columwise_data is allocated
  if (is_training && scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING) {
    int temp = 1;
    output_tensor.set_columnwise_data(static_cast<void *>(&temp), out_dtype, input_shape);
  }

  // dummy tensor wrappers that will carry workspace size info later
  TensorWrapper dummy_work_tensor;
  auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount() - sm_margin;
  if (norm_type == NVTE_Norm_Type::LayerNorm) {
    auto beta_tensor = TensorWrapper(nullptr, weight_shape, w_dtype);
    auto mu_tensor = TensorWrapper(nullptr, intermediates_shape, DType::kFloat32);

    nvte_layernorm_fwd(input_tensor.data(), gamma_tensor.data(), beta_tensor.data(), epsilon,
                       output_tensor.data(), mu_tensor.data(), rsigma_tensor.data(),
                       dummy_work_tensor.data(), num_sm, zero_centered_gamma, nullptr);
  } else {
    NVTE_CHECK(scaling_mode != JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING || !zero_centered_gamma,
               "rmsnorm doesn't support zero_centered_gamma.");
    nvte_rmsnorm_fwd(input_tensor.data(), gamma_tensor.data(), epsilon, output_tensor.data(),
                     rsigma_tensor.data(), dummy_work_tensor.data(), num_sm, zero_centered_gamma,
                     nullptr);
  }

  auto work_shape = MakeShapeVector(dummy_work_tensor.shape());
  return pybind11::make_tuple(std::make_pair(work_shape, dummy_work_tensor.dtype()));
}

Error_Type NormForwardFFI(cudaStream_t stream, Buffer_Type x_buf, Buffer_Type scale_buf,
                          Buffer_Type gamma_buf, Buffer_Type beta_buf, Result_Type output_buf,
                          Result_Type colwise_output_buf, Result_Type scale_inv_buf,
                          Result_Type colwise_scale_inv_buf, Result_Type amax_buf,
                          Result_Type mu_buf, Result_Type rsigma_buf, Result_Type wkspace_buf,
                          int norm_type, bool zero_centered_gamma, double epsilon,
                          int64_t sm_margin, JAXX_Scaling_Mode scaling_mode, bool is_2x) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(x_buf.element_type());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output_buf->element_type());
  auto w_dtype = convert_ffi_datatype_to_te_dtype(gamma_buf.element_type());
  auto wkspace_dtype = convert_ffi_datatype_to_te_dtype(wkspace_buf->element_type());

  auto *input = x_buf.untyped_data();
  auto *scale = reinterpret_cast<float *>(scale_buf.untyped_data());
  auto *gamma = gamma_buf.untyped_data();
  auto *beta = beta_buf.untyped_data();
  auto *output = output_buf->untyped_data();
  auto *rsigma = rsigma_buf->untyped_data();
  auto *mu = mu_buf->untyped_data();
  auto *amax = reinterpret_cast<float *>(amax_buf->untyped_data());
  auto *workspace = wkspace_buf->untyped_data();

  auto _norm_type = static_cast<NVTE_Norm_Type>(norm_type);
  auto _is_2x = static_cast<bool>(is_2x);

  auto x_size = product(x_buf.dimensions());
  auto gamma_size = product(gamma_buf.dimensions());
  auto workspace_size = product(wkspace_buf->dimensions());
  auto hidden_size = gamma_size;
  auto batch_size = x_size / gamma_size;

  float _epsilon = static_cast<float>(epsilon);
  int _sm_margin = static_cast<int>(sm_margin);

  auto input_shape = std::vector<size_t>{batch_size, hidden_size};
  auto gamma_shape = std::vector<size_t>{hidden_size};
  auto intermediates_shape = std::vector<size_t>{batch_size};
  auto workspace_shape = std::vector<size_t>{workspace_size};

  auto input_tensor = TensorWrapper(input, input_shape, in_dtype);
  auto gamma_tensor = TensorWrapper(gamma, gamma_shape, w_dtype);

  auto rsigma_tensor = TensorWrapper(rsigma, intermediates_shape, DType::kFloat32);
  auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount() - _sm_margin;
  auto workspace_tensor = TensorWrapper(workspace, workspace_shape, wkspace_dtype);

  auto output_tensor = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
  output_tensor.set_rowwise_data(output, static_cast<DType>(out_dtype), input_shape);

  NVTE_CHECK(
      scaling_mode != JAXX_Scaling_Mode::CURRENT_TENSOR_SCALING,
      "Current tensor scaling does not support fused operations yet. Please call this primitive "
      "in higher-precision then quantize with current scaling.");

  if (is_fp8_dtype(out_dtype)) {
    output_tensor.set_rowwise_scale_inv(
        scale_inv_buf->untyped_data(),
        convert_ffi_datatype_to_te_dtype(scale_inv_buf->element_type()),
        std::vector<size_t>{
            product(scale_inv_buf->dimensions(), 0, scale_inv_buf->dimensions().size() - 1),
            static_cast<size_t>(scale_inv_buf->dimensions().back())});
  }

  if (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING && is_fp8_dtype(out_dtype)) {
    output_tensor.set_scale(scale, DType::kFloat32, std::vector<size_t>{1});
    nvte_memset(amax, 0, sizeof(float), stream);
    output_tensor.set_amax(amax, DType::kFloat32, std::vector<size_t>{1});
  }

  if (_is_2x) {
    output_tensor.set_columnwise_data(colwise_output_buf->untyped_data(),
                                      static_cast<DType>(out_dtype), input_shape);
    output_tensor.set_columnwise_scale_inv(
        colwise_scale_inv_buf->untyped_data(),
        convert_ffi_datatype_to_te_dtype(colwise_scale_inv_buf->element_type()),
        std::vector<size_t>{product(colwise_scale_inv_buf->dimensions(), 0,
                                    colwise_scale_inv_buf->dimensions().size() - 1),
                            static_cast<size_t>(colwise_scale_inv_buf->dimensions().back())});
  }

  if (_norm_type == NVTE_Norm_Type::LayerNorm) {
    NVTE_CHECK(w_dtype == convert_ffi_datatype_to_te_dtype(beta_buf.element_type()),
               "gamma and beta must have the same data type.");
    auto beta_tensor = TensorWrapper(beta, gamma_shape, w_dtype);
    auto mu_tensor = TensorWrapper(mu, intermediates_shape, DType::kFloat32);

    nvte_layernorm_fwd(input_tensor.data(), gamma_tensor.data(), beta_tensor.data(), _epsilon,
                       output_tensor.data(), mu_tensor.data(), rsigma_tensor.data(),
                       workspace_tensor.data(), num_sm, zero_centered_gamma, stream);
  } else {
    NVTE_CHECK(scaling_mode != JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING || !zero_centered_gamma,
               "rmsnorm doesn't support zero_centered_gamma.");
    nvte_rmsnorm_fwd(input_tensor.data(), gamma_tensor.data(), _epsilon, output_tensor.data(),
                     rsigma_tensor.data(), workspace_tensor.data(), num_sm, zero_centered_gamma,
                     stream);
  }
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(NormForwardHandler, NormForwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // x
                                  .Arg<Buffer_Type>()      // scale
                                  .Arg<Buffer_Type>()      // gamma
                                  .Arg<Buffer_Type>()      // beta
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // colwise_output
                                  .Ret<Buffer_Type>()      // scale_inv
                                  .Ret<Buffer_Type>()      // colwise_scale_inv
                                  .Ret<Buffer_Type>()      // amax
                                  .Ret<Buffer_Type>()      // mu
                                  .Ret<Buffer_Type>()      // rsigma
                                  .Ret<Buffer_Type>()      // wkspace
                                  .Attr<int64_t>("norm_type")
                                  .Attr<bool>("zero_centered_gamma")
                                  .Attr<double>("epsilon")
                                  .Attr<int64_t>("sm_margin")
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<bool>("is_2x"),
                              FFI_CudaGraph_Traits);

pybind11::tuple GetNormBackwardWorkspaceSizes(size_t batch_size, size_t hidden_size, DType in_dtype,
                                              DType w_dtype, NVTE_Norm_Type norm_type,
                                              bool zero_centered_gamma, int sm_margin) {
  auto input_shape = std::vector<size_t>{batch_size, hidden_size};
  auto weight_shape = std::vector<size_t>{hidden_size};
  auto intermediates_shape = std::vector<size_t>{batch_size};
  auto intermediates_dtype = DType::kFloat32;

  // empty tensor wrappers are okay just to get workspace size
  auto dz_tensor = TensorWrapper(nullptr, input_shape, in_dtype);
  auto rsigma_tensor = TensorWrapper(nullptr, intermediates_shape, intermediates_dtype);
  auto x_tensor = TensorWrapper(nullptr, input_shape, in_dtype);
  auto gamma_tensor = TensorWrapper(nullptr, weight_shape, w_dtype);
  auto xgrad_tensor = TensorWrapper(nullptr, input_shape, in_dtype);
  auto wgrad_tensor = TensorWrapper(nullptr, weight_shape, w_dtype);

  // dummy tensor wrappers that will carry workspace size info later
  TensorWrapper dummy_work_tensor;
  auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount() - sm_margin;

  if (norm_type == NVTE_Norm_Type::LayerNorm) {
    auto mu_tensor = TensorWrapper(nullptr, intermediates_shape, intermediates_dtype);
    auto dbeta_tensor = TensorWrapper(nullptr, weight_shape, w_dtype);

    nvte_layernorm_bwd(dz_tensor.data(), x_tensor.data(), mu_tensor.data(), rsigma_tensor.data(),
                       gamma_tensor.data(), xgrad_tensor.data(), wgrad_tensor.data(),
                       dbeta_tensor.data(), dummy_work_tensor.data(), num_sm, zero_centered_gamma,
                       nullptr);

  } else {
    NVTE_CHECK(!zero_centered_gamma, "rmsnorm doesn't support zero_centered_gamma.");
    nvte_rmsnorm_bwd(dz_tensor.data(), x_tensor.data(), rsigma_tensor.data(), gamma_tensor.data(),
                     xgrad_tensor.data(), wgrad_tensor.data(), dummy_work_tensor.data(), num_sm,
                     zero_centered_gamma, nullptr);
  }

  auto work_shape = MakeShapeVector(dummy_work_tensor.shape());
  return pybind11::make_tuple(std::make_pair(work_shape, dummy_work_tensor.dtype()));
}

Error_Type NormBackwardFFI(cudaStream_t stream, Buffer_Type dz_buf, Buffer_Type x_buf,
                           Buffer_Type mu_buf, Buffer_Type rsigma_buf, Buffer_Type gamma_buf,
                           Result_Type xgrad_buf, Result_Type wgrad_buf, Result_Type dbeta_buf,
                           Result_Type wkspace_buf, int64_t norm_type, bool zero_centered_gamma,
                           int64_t sm_margin) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(x_buf.element_type());
  auto w_dtype = convert_ffi_datatype_to_te_dtype(gamma_buf.element_type());
  auto wkspace_dtype = convert_ffi_datatype_to_te_dtype(wkspace_buf->element_type());

  auto *ograd = dz_buf.untyped_data();
  auto *input = x_buf.untyped_data();
  void *mu = mu_buf.untyped_data();
  auto *rsigma = rsigma_buf.untyped_data();
  auto *gamma = gamma_buf.untyped_data();
  auto *xgrad = xgrad_buf->untyped_data();
  auto *wgrad = wgrad_buf->untyped_data();
  void *dbeta = dbeta_buf->untyped_data();
  auto *workspace = wkspace_buf->untyped_data();

  auto x_size = product(x_buf.dimensions());
  auto gamma_size = product(gamma_buf.dimensions());
  auto wkspace_size = product(wkspace_buf->dimensions());
  auto hidden_size = gamma_size;
  auto batch_size = x_size / gamma_size;

  int _sm_margin = static_cast<int>(sm_margin);

  auto input_shape = std::vector<size_t>{batch_size, hidden_size};
  auto weight_shape = std::vector<size_t>{hidden_size};
  auto intermediates_shape = std::vector<size_t>{batch_size};
  auto intermediates_dtype = DType::kFloat32;

  // assume input type = output type
  auto *grad_output = ograd;
  auto x_dtype = in_dtype;
  auto dz_tensor = TensorWrapper(grad_output, input_shape, x_dtype);

  auto rsigma_tensor = TensorWrapper(rsigma, intermediates_shape, intermediates_dtype);

  auto x_tensor = TensorWrapper(input, input_shape, x_dtype);

  auto gamma_tensor = TensorWrapper(gamma, weight_shape, w_dtype);
  auto xgrad_tensor = TensorWrapper(xgrad, input_shape, x_dtype);
  auto wgrad_tensor = TensorWrapper(wgrad, weight_shape, w_dtype);

  auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount() - _sm_margin;

  auto workspace_shape = std::vector<size_t>{wkspace_size};
  auto workspace_tensor = TensorWrapper(workspace, workspace_shape, wkspace_dtype);

  if (static_cast<NVTE_Norm_Type>(norm_type) == NVTE_Norm_Type::LayerNorm) {
    auto mu_tensor = TensorWrapper(mu, intermediates_shape, intermediates_dtype);
    auto dbeta_tensor = TensorWrapper(dbeta, weight_shape, w_dtype);

    nvte_layernorm_bwd(dz_tensor.data(), x_tensor.data(), mu_tensor.data(), rsigma_tensor.data(),
                       gamma_tensor.data(), xgrad_tensor.data(), wgrad_tensor.data(),
                       dbeta_tensor.data(), workspace_tensor.data(), num_sm, zero_centered_gamma,
                       stream);
  } else {
    NVTE_CHECK(!zero_centered_gamma, "rmsnorm doesn't support zero_centered_gamma.");
    nvte_rmsnorm_bwd(dz_tensor.data(), x_tensor.data(), rsigma_tensor.data(), gamma_tensor.data(),
                     xgrad_tensor.data(), wgrad_tensor.data(), workspace_tensor.data(), num_sm,
                     zero_centered_gamma, stream);
  }

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(NormBackwardHandler, NormBackwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // dz
                                  .Arg<Buffer_Type>()      // x
                                  .Arg<Buffer_Type>()      // mu
                                  .Arg<Buffer_Type>()      // rsigma
                                  .Arg<Buffer_Type>()      // gamma
                                  .Ret<Buffer_Type>()      // xgrad
                                  .Ret<Buffer_Type>()      // wgrad
                                  .Ret<Buffer_Type>()      // dbeta
                                  .Ret<Buffer_Type>()      // wkspace
                                  .Attr<int64_t>("norm_type")
                                  .Attr<bool>("zero_centered_gamma")
                                  .Attr<int64_t>("sm_margin"),
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine

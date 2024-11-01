/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "extensions.h"
#include "transformer_engine/layer_norm.h"
#include "transformer_engine/rmsnorm.h"

namespace transformer_engine {
namespace jax {

pybind11::tuple GetLayerNormForwardWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                  DType in_dtype, DType w_dtype, DType out_dtype,
                                                  bool is_layer_norm, bool zero_centered_gamma,
                                                  float eps, int sm_margin) {
  auto input_shape = std::vector<size_t>{batch_size, hidden_size};
  auto weight_shape = std::vector<size_t>{hidden_size};
  auto intermediates_shape = std::vector<size_t>{batch_size};

  // empty tensor wrappers are okay just to get workspace size
  auto input_tensor = TensorWrapper(nullptr, input_shape, in_dtype);
  auto gamma_tensor = TensorWrapper(nullptr, weight_shape, in_dtype);
  auto output_tensor = TensorWrapper(nullptr, input_shape, out_dtype);
  auto rsigma_tensor = TensorWrapper(nullptr, intermediates_shape, DType::kFloat32);

  // dummy tensor wrappers that will carry workspace size info later
  TensorWrapper dummy_work_tensor, dummy_barrier_tensor;
  auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount() - sm_margin;
  auto layernorm_fwd_func = zero_centered_gamma ? nvte_layernorm1p_fwd : nvte_layernorm_fwd;
  if (is_layer_norm) {
    auto beta_tensor = TensorWrapper(nullptr, weight_shape, w_dtype);
    auto mu_tensor = TensorWrapper(nullptr, intermediates_shape, DType::kFloat32);

    layernorm_fwd_func(input_tensor.data(), gamma_tensor.data(), beta_tensor.data(), eps,
                       output_tensor.data(), mu_tensor.data(), rsigma_tensor.data(), nullptr,
                       num_sm, dummy_work_tensor.data(), dummy_barrier_tensor.data());
  } else {
    NVTE_CHECK(!zero_centered_gamma, "rmsnorm doesn't support zero_centered_gamma.");
    nvte_rmsnorm_fwd(input_tensor.data(), gamma_tensor.data(), eps, output_tensor.data(),
                     rsigma_tensor.data(), nullptr, num_sm, dummy_work_tensor.data(),
                     dummy_barrier_tensor.data());
  }

  auto work_shape = MakeShapeVector(dummy_work_tensor.shape());
  auto barrier_shape = MakeShapeVector(dummy_barrier_tensor.shape());
  return pybind11::make_tuple(std::make_pair(work_shape, dummy_work_tensor.dtype()),
                              std::make_pair(barrier_shape, dummy_barrier_tensor.dtype()));
}

void LayerNormForwardImpl(size_t batch_size, size_t hidden_size, size_t workspace_size,
                          size_t barrier_size, bool zero_centered_gamma, float eps, void *input,
                          DType in_dtype, void *weight, DType w_dtype, void *bias, void *output,
                          DType out_dtype, void *workspace, DType work_dtype, void *barrier,
                          DType barrier_dtype, void *mu, void *rsigma, float *amax, float *scale,
                          float *scale_inv, int sm_margin, cudaStream_t stream) {
  auto input_shape = std::vector<size_t>{batch_size, hidden_size};
  auto weight_shape = std::vector<size_t>{hidden_size};
  auto intermediates_shape = std::vector<size_t>{batch_size};
  auto workspace_shape = std::vector<size_t>{workspace_size};
  auto barrier_shape = std::vector<size_t>{barrier_size};
  auto is_layer_norm = (bias) ? true : false;

  auto input_tensor = TensorWrapper(input, input_shape, in_dtype);
  auto gamma_tensor = TensorWrapper(weight, weight_shape, in_dtype);

  // assume output dtype = input dtype
  // If we need mixed I/O precision in the future, we need an additional
  // parameter for output type
  auto output_tensor = TensorWrapper(output, input_shape, out_dtype, amax, scale, scale_inv);
  auto rsigma_tensor = TensorWrapper(rsigma, intermediates_shape, DType::kFloat32);

  auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount() - sm_margin;
  auto layernorm_fwd_func = zero_centered_gamma ? nvte_layernorm1p_fwd : nvte_layernorm_fwd;

  auto workspace_tensor = TensorWrapper(workspace, workspace_shape, work_dtype);
  auto barrier_tensor = TensorWrapper(barrier, barrier_shape, barrier_dtype);

  if (is_layer_norm) {
    auto beta_tensor = TensorWrapper(bias, weight_shape, w_dtype);
    auto mu_tensor = TensorWrapper(mu, intermediates_shape, DType::kFloat32);

    layernorm_fwd_func(input_tensor.data(), gamma_tensor.data(), beta_tensor.data(), eps,
                       output_tensor.data(), mu_tensor.data(), rsigma_tensor.data(), stream, num_sm,
                       workspace_tensor.data(), barrier_tensor.data());
  } else {
    NVTE_CHECK(!zero_centered_gamma, "rmsnorm doesn't support zero_centered_gamma.");
    nvte_rmsnorm_fwd(input_tensor.data(), gamma_tensor.data(), eps, output_tensor.data(),
                     rsigma_tensor.data(), stream, num_sm, workspace_tensor.data(),
                     barrier_tensor.data());
  }
}

Error_Type LayerNormForwardImplFFI(cudaStream_t stream, Buffer_Type *x_buf, Buffer_Type *gamma_buf,
                                   Buffer_Type *beta_buf, Buffer_Type *amax_buf,
                                   Buffer_Type *scale_buf, Buffer_Type *scale_inv_buf,
                                   Result_Type *output_buf, Result_Type *mu_buf,
                                   Result_Type *rsigma_buf, Result_Type *amax_out_buf,
                                   Result_Type *wkspace_buf, Result_Type *barrier_buf,
                                   bool zero_centered_gamma, double eps_, int64_t sm_margin_,
                                   bool is_layer_norm, bool is_fp8) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype((*x_buf).element_type());
  auto w_dtype = convert_ffi_datatype_to_te_dtype((*gamma_buf).element_type());
  auto wkspace_dtype = convert_ffi_datatype_to_te_dtype((*wkspace_buf)->element_type());
  auto barrier_dtype = convert_ffi_datatype_to_te_dtype((*barrier_buf)->element_type());

  auto *input = x_buf->untyped_data();
  auto *weight = gamma_buf->untyped_data();
  auto *output = (*output_buf)->untyped_data();
  auto *rsigma = (*rsigma_buf)->untyped_data();
  auto *workspace = (*wkspace_buf)->untyped_data();
  auto *barrier = (*barrier_buf)->untyped_data();

  void *bias = nullptr;
  void *mu = nullptr;
  if (is_layer_norm) {
    bias = beta_buf->untyped_data();
    mu = (*mu_buf)->untyped_data();
  }

  float *amax = nullptr;
  float *scale = nullptr;
  float *scale_inv = nullptr;
  void *amax_out = nullptr;
  auto out_dtype = in_dtype;
  if (is_fp8) {
    amax = reinterpret_cast<float *>(amax_buf->untyped_data());
    scale = reinterpret_cast<float *>(scale_buf->untyped_data());
    scale_inv = reinterpret_cast<float *>(scale_inv_buf->untyped_data());
    amax_out = (*amax_out_buf)->untyped_data();
    NVTE_CHECK(amax_out == amax, "amax not bound to amax_out in TE/JAX LayerNormForward primitive");
    out_dtype = DType::kFloat8E4M3;
  }

  auto x_size = product(x_buf->dimensions());
  auto gamma_size = product(gamma_buf->dimensions());
  auto wkspace_size = product((*wkspace_buf)->dimensions());
  auto barrier_size = product((*barrier_buf)->dimensions());
  auto hidden_size = gamma_size;
  auto batch_size = x_size / gamma_size;

  float eps = static_cast<float>(eps_);
  int sm_margin = static_cast<int>(sm_margin_);

  LayerNormForwardImpl(batch_size, hidden_size, wkspace_size, barrier_size, zero_centered_gamma,
                       eps, input, in_dtype, weight, w_dtype, bias, output, out_dtype, workspace,
                       wkspace_dtype, barrier, barrier_dtype, mu, rsigma, amax, scale, scale_inv,
                       sm_margin, stream);
  return ffi_with_cuda_error_check();
}

Error_Type LayerNormForwardFP8FFI(cudaStream_t stream, Buffer_Type x_buf, Buffer_Type gamma_buf,
                                  Buffer_Type beta_buf, Buffer_Type amax_buf, Buffer_Type scale_buf,
                                  Buffer_Type scale_inv_buf, Result_Type output_buf,
                                  Result_Type mu_buf, Result_Type rsigma_buf,
                                  Result_Type amax_out_buf, Result_Type wkspace_buf,
                                  Result_Type barrier_buf, bool zero_centered_gamma, double eps_,
                                  int64_t sm_margin_) {
  return LayerNormForwardImplFFI(stream, &x_buf, &gamma_buf, &beta_buf, &amax_buf, &scale_buf,
                                 &scale_inv_buf, &output_buf, &mu_buf, &rsigma_buf, &amax_out_buf,
                                 &wkspace_buf, &barrier_buf, zero_centered_gamma, eps_, sm_margin_,
                                 true,  // is_layer_norm
                                 true   // is_fp8
  );
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(LayerNormForwardFP8Handler, LayerNormForwardFP8FFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // x
                                  .Arg<Buffer_Type>()      // gamma
                                  .Arg<Buffer_Type>()      // beta
                                  .Arg<Buffer_Type>()      // amax
                                  .Arg<Buffer_Type>()      // scale
                                  .Arg<Buffer_Type>()      // scale_inv
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // mu
                                  .Ret<Buffer_Type>()      // rsigma
                                  .Ret<Buffer_Type>()      // amax_out
                                  .Ret<Buffer_Type>()      // wkspace
                                  .Ret<Buffer_Type>()      // barrier
                                  .Attr<bool>("zero_centered_gamma")
                                  .Attr<double>("eps")
                                  .Attr<int64_t>("sm_margin"),
                              FFI_CudaGraph_Traits);

Error_Type LayerNormForwardFFI(cudaStream_t stream, Buffer_Type x_buf, Buffer_Type gamma_buf,
                               Buffer_Type beta_buf, Result_Type output_buf, Result_Type mu_buf,
                               Result_Type rsigma_buf, Result_Type wkspace_buf,
                               Result_Type barrier_buf, bool zero_centered_gamma, double eps_,
                               int64_t sm_margin_) {
  return LayerNormForwardImplFFI(stream, &x_buf, &gamma_buf, &beta_buf,
                                 nullptr,  // amax_buf
                                 nullptr,  // scale_buf,
                                 nullptr,  // scale_inv_buf,
                                 &output_buf, &mu_buf, &rsigma_buf,
                                 nullptr,  // amax_out_buf,
                                 &wkspace_buf, &barrier_buf, zero_centered_gamma, eps_, sm_margin_,
                                 true,  // is_layer_norm
                                 false  // is_fp8
  );
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(LayerNormForwardHandler, LayerNormForwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // x
                                  .Arg<Buffer_Type>()      // gamma
                                  .Arg<Buffer_Type>()      // beta
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // mu
                                  .Ret<Buffer_Type>()      // rsigma
                                  .Ret<Buffer_Type>()      // wkspace
                                  .Ret<Buffer_Type>()      // barrier
                                  .Attr<bool>("zero_centered_gamma")
                                  .Attr<double>("eps")
                                  .Attr<int64_t>("sm_margin"),
                              FFI_CudaGraph_Traits);

Error_Type RMSNormForwardFP8FFI(cudaStream_t stream, Buffer_Type x_buf, Buffer_Type gamma_buf,
                                Buffer_Type amax_buf, Buffer_Type scale_buf,
                                Buffer_Type scale_inv_buf, Result_Type output_buf,
                                Result_Type rsigma_buf, Result_Type amax_out_buf,
                                Result_Type wkspace_buf, Result_Type barrier_buf,
                                bool zero_centered_gamma, double eps_, int64_t sm_margin_) {
  return LayerNormForwardImplFFI(stream, &x_buf, &gamma_buf,
                                 nullptr,  // beta_buf,
                                 &amax_buf, &scale_buf, &scale_inv_buf, &output_buf,
                                 nullptr,  // mu_buf,
                                 &rsigma_buf, &amax_out_buf, &wkspace_buf, &barrier_buf,
                                 zero_centered_gamma, eps_, sm_margin_,
                                 false,  // is_layer_norm
                                 true    // is_fp8
  );
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(RMSNormForwardFP8Handler, RMSNormForwardFP8FFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // x
                                  .Arg<Buffer_Type>()      // gamma
                                  .Arg<Buffer_Type>()      // amax
                                  .Arg<Buffer_Type>()      // scale
                                  .Arg<Buffer_Type>()      // scale_inv
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // rsigma
                                  .Ret<Buffer_Type>()      // amax_out
                                  .Ret<Buffer_Type>()      // wkspace
                                  .Ret<Buffer_Type>()      // barrier
                                  .Attr<bool>("zero_centered_gamma")
                                  .Attr<double>("eps")
                                  .Attr<int64_t>("sm_margin"),
                              FFI_CudaGraph_Traits);

Error_Type RMSNormForwardFFI(cudaStream_t stream, Buffer_Type x_buf, Buffer_Type gamma_buf,
                             Result_Type output_buf, Result_Type rsigma_buf,
                             Result_Type wkspace_buf, Result_Type barrier_buf,
                             bool zero_centered_gamma, double eps_, int64_t sm_margin_) {
  return LayerNormForwardImplFFI(stream, &x_buf, &gamma_buf,
                                 nullptr,  // beta_buf,
                                 nullptr,  // amax_buf,
                                 nullptr,  // scale_buf,
                                 nullptr,  // scale_inv_buf,
                                 &output_buf,
                                 nullptr,  // mu_buf,
                                 &rsigma_buf,
                                 nullptr,  // amax_out_buf,
                                 &wkspace_buf, &barrier_buf, zero_centered_gamma, eps_, sm_margin_,
                                 false,  // is_layer_norm
                                 false   // is_fp8
  );
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(RMSNormForwardHandler, RMSNormForwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // x
                                  .Arg<Buffer_Type>()      // gamma
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // rsigma
                                  .Ret<Buffer_Type>()      // wkspace
                                  .Ret<Buffer_Type>()      // barrier
                                  .Attr<bool>("zero_centered_gamma")
                                  .Attr<double>("eps")
                                  .Attr<int64_t>("sm_margin"),
                              FFI_CudaGraph_Traits);

pybind11::tuple GetLayerNormBackwardWorkspaceSizes(size_t batch_size, size_t hidden_size,
                                                   DType in_dtype, DType w_dtype,
                                                   bool is_layer_norm, bool zero_centered_gamma,
                                                   float eps, int sm_margin) {
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
  TensorWrapper dummy_work_tensor, dummy_barrier_tensor;
  TensorWrapper dummy_dgamma_part_tensor, dummy_dbeta_part_tensor;
  auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount() - sm_margin;
  auto layernorm_bwd_func = zero_centered_gamma ? nvte_layernorm1p_bwd : nvte_layernorm_bwd;

  // initialize dBeta information here -- layernorm will modify but RMSnorm will not
  std::vector<size_t> dbeta_part_shape;
  if (is_layer_norm) {
    auto mu_tensor = TensorWrapper(nullptr, intermediates_shape, intermediates_dtype);
    auto dbeta_tensor = TensorWrapper(nullptr, weight_shape, w_dtype);

    layernorm_bwd_func(dz_tensor.data(), x_tensor.data(), mu_tensor.data(), rsigma_tensor.data(),
                       gamma_tensor.data(), xgrad_tensor.data(), wgrad_tensor.data(),
                       dbeta_tensor.data(), dummy_dgamma_part_tensor.data(),
                       dummy_dbeta_part_tensor.data(), nullptr, num_sm, dummy_work_tensor.data(),
                       dummy_barrier_tensor.data());

    dbeta_part_shape = MakeShapeVector(dummy_dbeta_part_tensor.shape());
  } else {
    NVTE_CHECK(!zero_centered_gamma, "rmsnorm doesn't support zero_centered_gamma.");
    nvte_rmsnorm_bwd(dz_tensor.data(), x_tensor.data(), rsigma_tensor.data(), gamma_tensor.data(),
                     xgrad_tensor.data(), wgrad_tensor.data(), dummy_dgamma_part_tensor.data(),
                     nullptr, num_sm, dummy_work_tensor.data(), dummy_barrier_tensor.data());

    dbeta_part_shape = std::vector<size_t>{0, 0};
  }

  auto work_shape = MakeShapeVector(dummy_work_tensor.shape());
  auto barrier_shape = MakeShapeVector(dummy_barrier_tensor.shape());
  auto dgamma_part_shape = MakeShapeVector(dummy_dgamma_part_tensor.shape());
  return pybind11::make_tuple(std::make_pair(work_shape, dummy_work_tensor.dtype()),
                              std::make_pair(barrier_shape, dummy_barrier_tensor.dtype()),
                              std::make_pair(dgamma_part_shape, dummy_dgamma_part_tensor.dtype()),
                              std::make_pair(dbeta_part_shape, dummy_dbeta_part_tensor.dtype()));
}

void LayerNormBackwardImpl(size_t batch_size, size_t hidden_size, size_t wkspace_size,
                           size_t barrier_size, Shape dgamma_part_shape, Shape dbeta_part_shape,
                           bool zero_centered_gamma, float eps, void *input, DType in_dtype,
                           void *weight, DType w_dtype, void *ograd, void *workspace,
                           DType wkspace_dtype, void *barrier, DType barrier_dtype, void *mu,
                           void *rsigma, void *xgrad, void *wgrad, void *dbeta, void *dgamma_part,
                           DType dgamma_dtype, void *dbeta_part, DType dbeta_dtype, int sm_margin,
                           cudaStream_t stream) {
  auto input_shape = std::vector<size_t>{batch_size, hidden_size};
  auto weight_shape = std::vector<size_t>{hidden_size};
  auto intermediates_shape = std::vector<size_t>{batch_size};
  auto intermediates_dtype = DType::kFloat32;
  auto is_layer_norm = (dbeta) ? true : false;

  // assume input type = output type
  auto *grad_output = ograd;
  auto x_dtype = in_dtype;
  auto dz_tensor = TensorWrapper(grad_output, input_shape, x_dtype);

  auto rsigma_tensor = TensorWrapper(rsigma, intermediates_shape, intermediates_dtype);

  auto *x = input;
  auto x_tensor = TensorWrapper(x, input_shape, x_dtype);

  auto gamma_tensor = TensorWrapper(weight, weight_shape, w_dtype);
  auto xgrad_tensor = TensorWrapper(xgrad, input_shape, x_dtype);
  auto wgrad_tensor = TensorWrapper(wgrad, weight_shape, w_dtype);

  auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount() - sm_margin;
  auto layernorm_bwd_func = zero_centered_gamma ? nvte_layernorm1p_bwd : nvte_layernorm_bwd;

  auto workspace_shape = std::vector<size_t>{wkspace_size};
  auto workspace_tensor = TensorWrapper(workspace, workspace_shape, wkspace_dtype);
  auto barrier_shape = std::vector<size_t>{barrier_size};
  auto barrier_tensor = TensorWrapper(barrier, barrier_shape, barrier_dtype);
  auto dgamma_part_tensor = TensorWrapper(dgamma_part, dgamma_part_shape.to_vector(), dgamma_dtype);

  if (is_layer_norm) {
    auto mu_tensor = TensorWrapper(mu, intermediates_shape, intermediates_dtype);
    auto dbeta_tensor = TensorWrapper(dbeta, weight_shape, w_dtype);
    auto dbeta_part_tensor = TensorWrapper(dbeta_part, dbeta_part_shape.to_vector(), dbeta_dtype);

    layernorm_bwd_func(dz_tensor.data(), x_tensor.data(), mu_tensor.data(), rsigma_tensor.data(),
                       gamma_tensor.data(), xgrad_tensor.data(), wgrad_tensor.data(),
                       dbeta_tensor.data(), dgamma_part_tensor.data(), dbeta_part_tensor.data(),
                       stream, num_sm, workspace_tensor.data(), barrier_tensor.data());
  } else {
    NVTE_CHECK(!zero_centered_gamma, "rmsnorm doesn't support zero_centered_gamma.");
    nvte_rmsnorm_bwd(dz_tensor.data(), x_tensor.data(), rsigma_tensor.data(), gamma_tensor.data(),
                     xgrad_tensor.data(), wgrad_tensor.data(), dgamma_part_tensor.data(), stream,
                     num_sm, workspace_tensor.data(), barrier_tensor.data());
  }
}

Error_Type LayerNormBackwardImplFFI(cudaStream_t stream, Buffer_Type *dz_buf, Buffer_Type *x_buf,
                                    Buffer_Type *mu_buf, Buffer_Type *rsigma_buf,
                                    Buffer_Type *gamma_buf, Result_Type *xgrad_buf,
                                    Result_Type *wgrad_buf, Result_Type *dbeta_buf,
                                    Result_Type *wkspace_buf, Result_Type *barrier_buf,
                                    Result_Type *dgamma_part_buf, Result_Type *dbeta_part_buf,
                                    bool zero_centered_gamma, double eps_, int64_t sm_margin_,
                                    bool is_layer_norm) {
  auto in_dtype = convert_ffi_datatype_to_te_dtype(x_buf->element_type());
  auto w_dtype = convert_ffi_datatype_to_te_dtype(gamma_buf->element_type());
  auto wkspace_dtype = convert_ffi_datatype_to_te_dtype((*wkspace_buf)->element_type());
  auto barrier_dtype = convert_ffi_datatype_to_te_dtype((*barrier_buf)->element_type());
  auto dgamma_part_dtype = convert_ffi_datatype_to_te_dtype((*dgamma_part_buf)->element_type());

  auto *ograd = dz_buf->untyped_data();
  auto *rsigma = rsigma_buf->untyped_data();
  auto *input = x_buf->untyped_data();
  auto *weight = gamma_buf->untyped_data();
  auto *xgrad = (*xgrad_buf)->untyped_data();
  auto *wgrad = (*wgrad_buf)->untyped_data();
  auto *workspace = (*wkspace_buf)->untyped_data();
  auto *barrier = (*barrier_buf)->untyped_data();
  auto *dgamma_part = (*dgamma_part_buf)->untyped_data();

  void *mu = nullptr;
  void *dbeta = nullptr;
  void *dbeta_part = nullptr;
  auto dbeta_part_dtype = DType::kByte;
  if (is_layer_norm) {
    mu = (*mu_buf).untyped_data();
    dbeta = (*dbeta_buf)->untyped_data();
    dbeta_part = (*dbeta_part_buf)->untyped_data();
    dbeta_part_dtype = convert_ffi_datatype_to_te_dtype((*dbeta_part_buf)->element_type());
  }

  auto x_size = product(x_buf->dimensions());
  auto gamma_size = product(gamma_buf->dimensions());
  auto wkspace_size = product((*wkspace_buf)->dimensions());
  auto barrier_size = product((*barrier_buf)->dimensions());
  auto hidden_size = gamma_size;
  auto batch_size = x_size / gamma_size;

  Shape dgamma_part_shape;
  auto dgamma_part_dims = (*dgamma_part_buf)->dimensions();
  std::vector<size_t> dgamma_parts_dims_vector(dgamma_part_dims.begin(), dgamma_part_dims.end());
  dgamma_part_shape.from_vector(dgamma_parts_dims_vector);

  Shape dbeta_part_shape;
  if (is_layer_norm) {
    auto dbeta_part_dims = (*dbeta_part_buf)->dimensions();
    std::vector<size_t> dbeta_parts_dims_vector(dbeta_part_dims.begin(), dbeta_part_dims.end());
    dbeta_part_shape.from_vector(dbeta_parts_dims_vector);
  } else {
    dbeta_part_shape.from_vector({0, 0});
  }

  float eps = static_cast<float>(eps_);
  int sm_margin = static_cast<int>(sm_margin_);

  LayerNormBackwardImpl(batch_size, hidden_size, wkspace_size, barrier_size, dgamma_part_shape,
                        dbeta_part_shape, zero_centered_gamma, eps, input, in_dtype, weight,
                        w_dtype, ograd, workspace, wkspace_dtype, barrier, barrier_dtype, mu,
                        rsigma, xgrad, wgrad, dbeta, dgamma_part, dgamma_part_dtype, dbeta_part,
                        dbeta_part_dtype, sm_margin, stream);
  return ffi_with_cuda_error_check();
}

Error_Type LayerNormBackwardFFI(cudaStream_t stream, Buffer_Type dz_buf, Buffer_Type x_buf,
                                Buffer_Type mu_buf, Buffer_Type rsigma_buf, Buffer_Type gamma_buf,
                                Result_Type xgrad_buf, Result_Type wgrad_buf, Result_Type dbeta_buf,
                                Result_Type wkspace_buf, Result_Type barrier_buf,
                                Result_Type dgamma_part_buf, Result_Type dbeta_part_buf,
                                bool zero_centered_gamma, double eps_, int64_t sm_margin_) {
  return LayerNormBackwardImplFFI(stream, &dz_buf, &x_buf, &mu_buf, &rsigma_buf, &gamma_buf,
                                  &xgrad_buf, &wgrad_buf, &dbeta_buf, &wkspace_buf, &barrier_buf,
                                  &dgamma_part_buf, &dbeta_part_buf, zero_centered_gamma, eps_,
                                  sm_margin_,
                                  true  // is_layer_norm
  );
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(LayerNormBackwardHandler, LayerNormBackwardFFI,
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
                                  .Ret<Buffer_Type>()      // barrier
                                  .Ret<Buffer_Type>()      // dgamma_part
                                  .Ret<Buffer_Type>()      // dbeta_part
                                  .Attr<bool>("zero_centered_gamma")
                                  .Attr<double>("eps")
                                  .Attr<int64_t>("sm_margin"),
                              FFI_CudaGraph_Traits);

Error_Type RMSNormBackwardFFI(cudaStream_t stream, Buffer_Type dz_buf, Buffer_Type x_buf,
                              Buffer_Type rsigma_buf, Buffer_Type gamma_buf, Result_Type xgrad_buf,
                              Result_Type wgrad_buf, Result_Type wkspace_buf,
                              Result_Type barrier_buf, Result_Type dgamma_part_buf,
                              bool zero_centered_gamma, double eps_, int64_t sm_margin_) {
  return LayerNormBackwardImplFFI(stream, &dz_buf, &x_buf,
                                  nullptr,  // mu_buf
                                  &rsigma_buf, &gamma_buf, &xgrad_buf, &wgrad_buf,
                                  nullptr,  // dbeta_buf,
                                  &wkspace_buf, &barrier_buf, &dgamma_part_buf,
                                  nullptr,  // dbeta_part_buf,
                                  zero_centered_gamma, eps_, sm_margin_,
                                  false  // is_layer_norm
  );
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(RMSNormBackwardHandler, RMSNormBackwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // dz
                                  .Arg<Buffer_Type>()      // x
                                  .Arg<Buffer_Type>()      // rsigma
                                  .Arg<Buffer_Type>()      // gamma
                                  .Ret<Buffer_Type>()      // xgrad
                                  .Ret<Buffer_Type>()      // wgrad
                                  .Ret<Buffer_Type>()      // wkspace
                                  .Ret<Buffer_Type>()      // barrier
                                  .Ret<Buffer_Type>()      // dgamma_part
                                  .Attr<bool>("zero_centered_gamma")
                                  .Attr<double>("eps")
                                  .Attr<int64_t>("sm_margin"),
                              FFI_CudaGraph_Traits);

void LayerNormForwardFP8(cudaStream_t stream, void **buffers, const char *opaque,
                         size_t opaque_len) {
  auto *input = buffers[0];
  auto *weight = buffers[1];
  auto *bias = buffers[2];
  auto *amax = reinterpret_cast<float *>(buffers[3]);
  auto *scale = reinterpret_cast<float *>(buffers[4]);
  auto *scale_inv = reinterpret_cast<float *>(buffers[5]);
  auto *output = buffers[6];
  auto *mu = buffers[7];
  auto *rsigma = buffers[8];
  auto *amax_out = buffers[9];
  auto *workspace = buffers[10];
  auto *barrier = buffers[11];
  NVTE_CHECK(amax_out == amax,
             "amax not bound to amax_out in TE/JAX LayerNormForwardFP8 primitive");

  const auto &desc = *UnpackOpaque<CustomCallNormDescriptor>(opaque, opaque_len);
  auto batch_size = desc.batch_size;
  auto hidden_size = desc.hidden_size;
  auto wkspace_size = desc.wkspace_size;
  auto barrier_size = desc.barrier_size;
  auto in_dtype = desc.x_dtype;
  auto w_dtype = desc.w_dtype;
  auto wkspace_dtype = desc.wkspace_dtype;
  auto barrier_dtype = desc.barrier_dtype;
  auto eps = desc.eps;
  auto zero_centered_gamma = desc.zero_centered_gamma;
  auto sm_margin = desc.sm_margin;

  auto out_dtype = DType::kFloat8E4M3;

  LayerNormForwardImpl(batch_size, hidden_size, wkspace_size, barrier_size, zero_centered_gamma,
                       eps, input, in_dtype, weight, w_dtype, bias, output, out_dtype, workspace,
                       wkspace_dtype, barrier, barrier_dtype, mu, rsigma, amax, scale, scale_inv,
                       sm_margin, stream);
}

void LayerNormForward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  auto *input = buffers[0];
  auto *weight = buffers[1];
  auto *bias = buffers[2];
  auto *output = buffers[3];
  auto *mu = buffers[4];
  auto *rsigma = buffers[5];
  auto *workspace = buffers[6];
  auto *barrier = buffers[7];

  float *amax = nullptr;
  float *scale = nullptr;
  float *scale_inv = nullptr;

  const auto &desc = *UnpackOpaque<CustomCallNormDescriptor>(opaque, opaque_len);
  auto batch_size = desc.batch_size;
  auto hidden_size = desc.hidden_size;
  auto wkspace_size = desc.wkspace_size;
  auto barrier_size = desc.barrier_size;
  auto in_dtype = desc.x_dtype;
  auto w_dtype = desc.w_dtype;
  auto wkspace_dtype = desc.wkspace_dtype;
  auto barrier_dtype = desc.barrier_dtype;
  auto eps = desc.eps;
  auto out_dtype = in_dtype;
  auto zero_centered_gamma = desc.zero_centered_gamma;
  auto sm_margin = desc.sm_margin;

  LayerNormForwardImpl(batch_size, hidden_size, wkspace_size, barrier_size, zero_centered_gamma,
                       eps, input, in_dtype, weight, w_dtype, bias, output, out_dtype, workspace,
                       wkspace_dtype, barrier, barrier_dtype, mu, rsigma, amax, scale, scale_inv,
                       sm_margin, stream);
}

void LayerNormBackward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  const auto &desc = *UnpackOpaque<CustomCallNormDescriptor>(opaque, opaque_len);

  auto batch_size = desc.batch_size;
  auto hidden_size = desc.hidden_size;
  auto wkspace_size = desc.wkspace_size;
  auto barrier_size = desc.barrier_size;
  auto dgamma_part_shape = desc.dgamma_part_shape;
  auto dbeta_part_shape = desc.dbeta_part_shape;
  auto in_dtype = desc.x_dtype;
  auto w_dtype = desc.w_dtype;
  auto wkspace_dtype = desc.wkspace_dtype;
  auto barrier_dtype = desc.barrier_dtype;
  auto dgamma_part_dtype = desc.dgamma_part_dtype;
  auto dbeta_part_dtype = desc.dbeta_part_dtype;
  auto eps = desc.eps;
  auto zero_centered_gamma = desc.zero_centered_gamma;
  auto sm_margin = desc.sm_margin;

  auto *ograd = buffers[0];
  auto *mu = buffers[1];
  auto *rsigma = buffers[2];
  auto *input = buffers[3];
  auto *weight = buffers[4];
  auto *xgrad = buffers[5];
  auto *wgrad = buffers[6];
  auto *dbeta = buffers[7];
  auto *workspace = buffers[8];
  auto *barrier = buffers[9];
  auto *dgamma_part = buffers[10];
  auto *dbeta_part = buffers[11];

  LayerNormBackwardImpl(batch_size, hidden_size, wkspace_size, barrier_size, dgamma_part_shape,
                        dbeta_part_shape, zero_centered_gamma, eps, input, in_dtype, weight,
                        w_dtype, ograd, workspace, wkspace_dtype, barrier, barrier_dtype, mu,
                        rsigma, xgrad, wgrad, dbeta, dgamma_part, dgamma_part_dtype, dbeta_part,
                        dbeta_part_dtype, sm_margin, stream);
}

void RMSNormForwardFP8(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  auto *input = buffers[0];
  auto *weight = buffers[1];
  auto *amax = reinterpret_cast<float *>(buffers[2]);
  auto *scale = reinterpret_cast<float *>(buffers[3]);
  auto *scale_inv = reinterpret_cast<float *>(buffers[4]);
  auto *output = buffers[5];
  auto *rsigma = buffers[6];
  auto *amax_out = buffers[7];
  auto *workspace = buffers[8];
  auto *barrier = buffers[9];
  NVTE_CHECK(amax_out == amax, "amax not bound to amax_out in TE/JAX RSMNormForwardFP8 primitive.");

  void *bias = nullptr;
  void *mu = nullptr;

  const auto &desc = *UnpackOpaque<CustomCallNormDescriptor>(opaque, opaque_len);
  auto batch_size = desc.batch_size;
  auto hidden_size = desc.hidden_size;
  auto wkspace_size = desc.wkspace_size;
  auto barrier_size = desc.barrier_size;
  auto in_dtype = desc.x_dtype;
  auto w_dtype = desc.w_dtype;
  auto wkspace_dtype = desc.wkspace_dtype;
  auto barrier_dtype = desc.barrier_dtype;
  auto eps = desc.eps;
  auto zero_centered_gamma = desc.zero_centered_gamma;
  auto sm_margin = desc.sm_margin;
  auto out_dtype = DType::kFloat8E4M3;

  LayerNormForwardImpl(batch_size, hidden_size, wkspace_size, barrier_size, zero_centered_gamma,
                       eps, input, in_dtype, weight, w_dtype, bias, output, out_dtype, workspace,
                       wkspace_dtype, barrier, barrier_dtype, mu, rsigma, amax, scale, scale_inv,
                       sm_margin, stream);
}

void RMSNormForward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  auto *input = buffers[0];
  auto *weight = buffers[1];
  auto *output = buffers[2];
  auto *rsigma = buffers[3];
  auto *workspace = buffers[4];
  auto *barrier = buffers[5];

  void *bias = nullptr;
  void *mu = nullptr;
  float *amax = nullptr;
  float *scale = nullptr;
  float *scale_inv = nullptr;

  const auto &desc = *UnpackOpaque<CustomCallNormDescriptor>(opaque, opaque_len);
  auto batch_size = desc.batch_size;
  auto hidden_size = desc.hidden_size;
  auto wkspace_size = desc.wkspace_size;
  auto barrier_size = desc.barrier_size;
  auto in_dtype = desc.x_dtype;
  auto w_dtype = desc.w_dtype;
  auto wkspace_dtype = desc.wkspace_dtype;
  auto barrier_dtype = desc.barrier_dtype;
  auto eps = desc.eps;
  auto zero_centered_gamma = desc.zero_centered_gamma;
  auto sm_margin = desc.sm_margin;
  auto out_dtype = in_dtype;

  LayerNormForwardImpl(batch_size, hidden_size, wkspace_size, barrier_size, zero_centered_gamma,
                       eps, input, in_dtype, weight, w_dtype, bias, output, out_dtype, workspace,
                       wkspace_dtype, barrier, barrier_dtype, mu, rsigma, amax, scale, scale_inv,
                       sm_margin, stream);
}

void RMSNormBackward(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  auto *ograd = buffers[0];
  auto *rsigma = buffers[1];
  auto *input = buffers[2];
  auto *weight = buffers[3];
  auto *xgrad = buffers[4];
  auto *wgrad = buffers[5];
  auto *workspace = buffers[6];
  auto *barrier = buffers[7];
  auto *dgamma_part = buffers[8];

  void *mu = nullptr;
  void *dbeta = nullptr;
  void *dbeta_part = nullptr;

  const auto &desc = *UnpackOpaque<CustomCallNormDescriptor>(opaque, opaque_len);
  auto batch_size = desc.batch_size;
  auto hidden_size = desc.hidden_size;
  auto wkspace_size = desc.wkspace_size;
  auto barrier_size = desc.barrier_size;
  auto dgamma_part_shape = desc.dgamma_part_shape;
  Shape dbeta_part_shape;
  dbeta_part_shape.from_vector({0, 0});
  auto in_dtype = desc.x_dtype;
  auto w_dtype = desc.w_dtype;
  auto wkspace_dtype = desc.wkspace_dtype;
  auto barrier_dtype = desc.barrier_dtype;
  auto dgamma_part_dtype = desc.dgamma_part_dtype;
  auto dbeta_part_dtype = DType::kByte;
  auto eps = desc.eps;
  auto zero_centered_gamma = desc.zero_centered_gamma;
  auto sm_margin = desc.sm_margin;

  LayerNormBackwardImpl(batch_size, hidden_size, wkspace_size, barrier_size, dgamma_part_shape,
                        dbeta_part_shape, zero_centered_gamma, eps, input, in_dtype, weight,
                        w_dtype, ograd, workspace, wkspace_dtype, barrier, barrier_dtype, mu,
                        rsigma, xgrad, wgrad, dbeta, dgamma_part, dgamma_part_dtype, dbeta_part,
                        dbeta_part_dtype, sm_margin, stream);
}

}  // namespace jax
}  // namespace transformer_engine

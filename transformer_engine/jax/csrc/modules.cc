/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "jax/csrc/modules.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "transformer_engine/activation.h"
#include "transformer_engine/gemm.h"
#include "transformer_engine/layer_norm.h"
#include "transformer_engine/rmsnorm.h"
#include "transformer_engine/transformer_engine.h"
#include "transformer_engine/transpose.h"

using TEDType = ::transformer_engine::DType;

inline bool use_fp8(TEDType type) {
  return type == TEDType::kFloat8E4M3 || type == TEDType::kFloat8E5M2;
}

constexpr size_t kCublasLtForwardWorkspaceSize = 32 * 1024 * 1024;
constexpr size_t kCublasLtBackwardWorkspaceSize = 32 * 1024 * 1024;

template <template <typename> typename Container, typename T>
size_t get_numel(const Container<T> &shape) {
  return std::accumulate(shape.begin(), shape.end(), T(1),
                         std::multiplies<T>());
}

namespace transformer_engine {
size_t typeToSize(const DType type);

namespace jax {

pybind11::bytes PackTEMatDescriptor(std::uint64_t m, std::uint64_t n,
                                    std::uint64_t fwd_dtype,
                                    std::uint64_t bwd_dtype) {
  return PackDescriptor(TEMatDescriptor{m, n, fwd_dtype, bwd_dtype});
}

pybind11::bytes PackTEGemmDescriptor(std::uint64_t m, std::uint64_t n,
                                     std::uint64_t k, std::uint64_t A_ctype,
                                     std::uint64_t B_ctype,
                                     std::uint64_t D_ctype, bool transa,
                                     bool transb, bool use_split_accumulator) {
  return PackDescriptor(TEGemmDescriptor{m, n, k, A_ctype, B_ctype, D_ctype,
                                         transa, transb,
                                         use_split_accumulator});
}

pybind11::bytes PackLayerNormDescriptor(std::uint64_t n, std::uint64_t hidden,
                                        std::uint64_t x_dtype,
                                        std::uint64_t w_dtype, float eps) {
  return PackDescriptor(LayerNormDescriptor{n, hidden, x_dtype, w_dtype, eps});
}

void TransposeImpl(void *input, std::uint64_t rows, std::uint64_t cols,
                   DType dtype, cudaStream_t stream, void *output) {
  auto input_shape = std::vector<size_t>{rows, cols};
  auto output_shape = std::vector<size_t>{cols, rows};

  auto input_tensor = TensorWrapper(input, input_shape, dtype);
  auto transposed_tensor = TensorWrapper(output, output_shape, dtype);

  nvte_transpose(input_tensor.data(), transposed_tensor.data(), stream);
}

void TETranspose(cudaStream_t stream, void **buffers, const char *opaque,
                 std::size_t opaque_len) {
  const TEMatDescriptor &descriptor =
      *UnpackDescriptor<TEMatDescriptor>(opaque, opaque_len);

  // input
  void *input = buffers[0];

  // output
  void *output = buffers[1];

  auto rows = descriptor.m;
  auto cols = descriptor.n;
  assert(descriptor.in_ctype == descriptor.out_ctype);
  auto dtype = static_cast<DType>(descriptor.out_ctype);

  TransposeImpl(input, rows, cols, dtype, stream, output);
}

void TECastTranspose(cudaStream_t stream, void **buffers, const char *opaque,
                     std::size_t opaque_len) {
  const TEMatDescriptor &d =
      *UnpackDescriptor<TEMatDescriptor>(opaque, opaque_len);

  const void *input = reinterpret_cast<const void *>(buffers[0]);
  const void *amax = reinterpret_cast<const void *>(buffers[1]);
  float *scale = reinterpret_cast<float *>(buffers[2]);
  float *scale_inv = reinterpret_cast<float *>(buffers[3]);

  void *input_cast = buffers[4];
  void *input_cast_trans = buffers[5];
  float *amax_out = reinterpret_cast<float *>(buffers[6]);

  assert(amax == amax_out);
      if (!use_fp8(static_cast<DType>(d.out_ctype))) {
      scale = nullptr;
      scale_inv = nullptr;
      amax_out = nullptr;
  }
  auto input_shape = std::vector<size_t>{d.m, d.n};
  auto input_trans_shape = std::vector<size_t>{d.n, d.m};

  auto input_tensor = TensorWrapper(const_cast<void *>(input), input_shape,
                                    static_cast<DType>(d.in_ctype));
  auto input_cast_tensor =
      TensorWrapper(input_cast, input_shape, static_cast<DType>(d.out_ctype),
                    amax_out, scale, scale_inv);
  auto input_cast_trans_tensor = TensorWrapper(
      input_cast_trans, input_trans_shape, static_cast<DType>(d.out_ctype),
      amax_out, scale, scale_inv);

  nvte_cast_transpose(input_tensor.data(),
                      input_cast_tensor.data(),
                      input_cast_trans_tensor.data(),
                      stream);
}

void GatedGeluImpl(void *input, std::uint64_t m, std::uint64_t n,
                   DType in_dtype, DType out_dtype, float *scale,
                   cudaStream_t stream, float *scale_inverse, float *amax,
                   void *output) {
  auto input_shape = std::vector<size_t>{m, n * 2};
  auto output_shape = std::vector<size_t>{m, n};

  auto input_tensor =
      TensorWrapper(input, input_shape, static_cast<DType>(in_dtype));

  auto output_tensor =
      TensorWrapper(output, output_shape, static_cast<DType>(out_dtype),
                    amax, scale, scale_inverse);

  nvte_geglu(input_tensor.data(), output_tensor.data(), stream);
}

void TEGatedGelu(cudaStream_t stream, void **buffers, const char *opaque,
                 std::size_t opaque_len) {
  const TEMatDescriptor &d =
      *UnpackDescriptor<TEMatDescriptor>(opaque, opaque_len);

  void *input = reinterpret_cast<void *>(buffers[0]);
  void *output = buffers[1];

  GatedGeluImpl(input, d.m, d.n, static_cast<DType>(d.in_ctype),
                static_cast<DType>(d.out_ctype), nullptr, stream, nullptr,
                nullptr, output);
}

void TEGatedGeluFP8(cudaStream_t stream, void **buffers, const char *opaque,
                    std::size_t opaque_len) {
  const TEMatDescriptor &d =
      *UnpackDescriptor<TEMatDescriptor>(opaque, opaque_len);

  void *input = reinterpret_cast<void *>(buffers[0]);
  void *amax = reinterpret_cast<void *>(buffers[1]);
  float *scale = reinterpret_cast<float *>(buffers[2]);
  float *scale_inv = reinterpret_cast<float *>(buffers[3]);

  void *output = buffers[4];
  float *amax_out = reinterpret_cast<float *>(buffers[5]);

  assert(amax == amax_out);

  GatedGeluImpl(input, d.m, d.n, static_cast<DType>(d.in_ctype),
                static_cast<DType>(d.out_ctype), scale, stream, scale_inv,
                amax_out, output);
}

void TEDGatedGelu(cudaStream_t stream, void **buffers, const char *opaque,
                  std::size_t opaque_len) {
  const TEMatDescriptor &d =
      *UnpackDescriptor<TEMatDescriptor>(opaque, opaque_len);

  const void *input = reinterpret_cast<const void *>(buffers[0]);
  const void *gelu_input = reinterpret_cast<const void *>(buffers[1]);
  void *output = buffers[2];

  auto input_shape = std::vector<size_t>{d.m, d.n};
  auto gelu_input_shape = std::vector<size_t>{d.m, d.n * 2};
  auto output_shape = std::vector<size_t>{d.m, d.n * 2};

  auto input_tensor = TensorWrapper(const_cast<void *>(input), input_shape,
                                    static_cast<DType>(d.in_ctype));
  auto gelu_input_tensor =
      TensorWrapper(const_cast<void *>(gelu_input), gelu_input_shape,
                    static_cast<DType>(d.in_ctype));
  auto output_tensor =
      TensorWrapper(output, output_shape, static_cast<DType>(d.out_ctype));

  nvte_dgeglu(input_tensor.data(), gelu_input_tensor.data(),
                   output_tensor.data(), stream);
}

void TEDGatedGeluCastTranspose(cudaStream_t stream, void **buffers,
                               const char *opaque, std::size_t opaque_len) {
  const TEMatDescriptor &d =
      *UnpackDescriptor<TEMatDescriptor>(opaque, opaque_len);

  const void *input = reinterpret_cast<const void *>(buffers[0]);
  const void *gelu_input = reinterpret_cast<const void *>(buffers[1]);
  const void *amax = reinterpret_cast<const void *>(buffers[2]);
  float *scale = reinterpret_cast<float *>(buffers[3]);
  float *scale_inv = reinterpret_cast<float *>(buffers[4]);

  void *output = buffers[5];
  void *output_trans = buffers[6];
  float *amax_out = reinterpret_cast<float *>(buffers[7]);

  assert(amax == amax_out);

  auto input_shape = std::vector<size_t>{d.m, d.n};
  auto gelu_input_shape = std::vector<size_t>{d.m, d.n * 2};
  auto output_shape = std::vector<size_t>{d.m, d.n * 2};
  auto output_trans_shape = std::vector<size_t>{d.n * 2, d.m};

  auto input_tensor = TensorWrapper(const_cast<void *>(input), input_shape,
                                    static_cast<DType>(d.in_ctype));
  auto gelu_input_tensor =
      TensorWrapper(const_cast<void *>(gelu_input), gelu_input_shape,
                    static_cast<DType>(d.in_ctype));
  auto output_tensor =
      TensorWrapper(output, output_shape, static_cast<DType>(d.out_ctype),
                    amax_out, scale, scale_inv);
  auto output_trans_tensor = TensorWrapper(output_trans, output_trans_shape,
                                           static_cast<DType>(d.out_ctype),
                                           amax_out, scale, scale_inv);


  nvte_dgeglu_cast_transpose(
      input_tensor.data(), gelu_input_tensor.data(),
      output_tensor.data(), output_trans_tensor.data(), stream);
}

void TEGemm(cudaStream_t stream, void **buffers, const char *opaque,
            std::size_t opaque_len) {
  const TEGemmDescriptor &d =
      *UnpackDescriptor<TEGemmDescriptor>(opaque, opaque_len);

  const void *A = reinterpret_cast<const void *>(buffers[0]);
  const void *B = reinterpret_cast<const void *>(buffers[1]);
  float *A_scale_inverse = reinterpret_cast<float *>(buffers[2]);
  float *B_scale_inverse = reinterpret_cast<float *>(buffers[3]);

  void *D = buffers[4];

  // We transposes shape of A, B and D here to correctly invoke
  // cuBlasLt GEMM (col-major) for row-major data.
  auto A_shape = std::vector<size_t>{d.k, d.m};
  auto A_tensor = TensorWrapper(const_cast<void *>(A), A_shape,
                                static_cast<DType>(d.A_ctype),
                                nullptr, nullptr, A_scale_inverse);

  auto B_shape = std::vector<size_t>{d.n, d.k};
  auto B_tensor = TensorWrapper(const_cast<void *>(B), B_shape,
                                static_cast<DType>(d.B_ctype),
                                nullptr, nullptr, B_scale_inverse);

  auto D_shape = std::vector<size_t>{d.n, d.m};
  auto D_tensor = TensorWrapper(const_cast<void *>(D), D_shape,
                                static_cast<DType>(d.D_ctype));

  auto null_tensor =
      TensorWrapper(nullptr, std::vector<size_t>{0}, DType::kFloat32);

  size_t workspace_size = kCublasLtForwardWorkspaceSize;
  void *workspace =
      cublasLtMetaManager::Instance().GetWorkspace(workspace_size);
  auto wk_tensor = TensorWrapper(const_cast<void *>(workspace),
                                 std::vector<size_t>{workspace_size},
                                 static_cast<DType>(DType::kByte));

  nvte_cublas_gemm(A_tensor.data(), B_tensor.data(),  D_tensor.data(),
                   null_tensor.data(), null_tensor.data(),
                   (d.transa) ? CUBLAS_OP_T : CUBLAS_OP_N,
                   (d.transb) ? CUBLAS_OP_T : CUBLAS_OP_N, false,
                   wk_tensor.data(), false, d.use_split_accumulator, stream);
}

struct TensorAddr {
    void *ptr;
    DType dtype;
};

struct FP8Meta {
    float *amax{nullptr};
    float *scale{nullptr};
    float *scale_inv{nullptr};
    FP8Meta() {}
    FP8Meta(void *amax_, void *scale_,
        void *scale_inv_, void *out) {

        assert(amax_ == out);
        amax = reinterpret_cast<float*>(amax_);
        scale = reinterpret_cast<float*>(scale_);
        scale_inv = reinterpret_cast<float*>(scale_inv_);
    }
};

struct NormalFwdParameter {
    TensorAddr x;
    TensorAddr w;
    void *bias;
    uint64_t n;
    uint64_t hidden;
    float eps;
    FP8Meta meta;
    TensorAddr y;
    void *mu;
    void *rsigma;
};

void LayerNormForwardImpl(NormalFwdParameter& p, cudaStream_t stream) {

  auto n = p.n;
  auto hidden = p.hidden;
  auto input_shape = std::vector<size_t>{n, hidden};
  auto weight_shape = std::vector<size_t>{hidden};
  auto intermediates_shape = std::vector<size_t>{n};
  auto eps = p.eps;
  auto is_apex_norm = (p.bias) ? true:false;

  auto *input = p.x.ptr;
  auto in_dtype = p.x.dtype;
  auto input_tensor = TensorWrapper(input, input_shape, in_dtype);

  auto *weight = p.w.ptr;
  auto w_dtype = p.w.dtype;
  auto gamma_tensor = TensorWrapper(weight, weight_shape, w_dtype);

  // assume output dtype = input dtype
  // If we need mixed I/O precision in the future, we need an additional
  // parameter for output type
  auto *output = p.y.ptr;
  auto out_dtype = p.y.dtype;
  auto *amax = p.meta.amax;
  auto *scale = p.meta.scale;
  auto *scale_inv = p.meta.scale_inv;
  auto output_tensor = TensorWrapper(output, input_shape, out_dtype,
    amax, scale, scale_inv);

  auto *rsigma = p.rsigma;
  auto rsigma_tensor =
      TensorWrapper(rsigma, intermediates_shape, DType::kFloat32);

  // Create uninitialized workspace, barrier and init them on the first
  TensorWrapper dummy_workspace_tensor, dummy_barrier_tensor;
  auto num_sm = cudaDevicePropertiesManager::Instance().GetMultiProcessorCount();

  // The first call is to query the required workspace
  if (is_apex_norm) {
    auto *bias = p.bias;
    auto beta_tensor = TensorWrapper(bias, weight_shape, w_dtype);
    auto *mu = p.mu;
    auto mu_tensor = TensorWrapper(mu, intermediates_shape, DType::kFloat32);

    nvte_layernorm_fwd(
        input_tensor.data(), gamma_tensor.data(), beta_tensor.data(),
        eps, output_tensor.data(), mu_tensor.data(),
        rsigma_tensor.data(), stream, num_sm,
        dummy_workspace_tensor.data(), dummy_barrier_tensor.data());
  }
  else {
    nvte_rmsnorm_fwd(
        input_tensor.data(), gamma_tensor.data(), eps,
        output_tensor.data(), rsigma_tensor.data(), stream, num_sm,
        dummy_workspace_tensor.data(), dummy_barrier_tensor.data());
  }


  size_t workspace_size = dummy_workspace_tensor.shape().data[0] *
    typeToSize(dummy_workspace_tensor.dtype()) +
    dummy_barrier_tensor.shape().data[0] *
    typeToSize(dummy_barrier_tensor.dtype());

  void *workspace =
      cublasLtMetaManager::Instance().GetWorkspace(workspace_size);

  auto workspace_tensor =
      TensorWrapper(workspace, dummy_workspace_tensor.shape(),
      dummy_workspace_tensor.dtype());

  auto barrier_tensor =
      TensorWrapper(reinterpret_cast<char *>(workspace) +
      dummy_workspace_tensor.shape().data[0],
      dummy_barrier_tensor.shape(), dummy_barrier_tensor.dtype());

  if (is_apex_norm) {
      auto *bias = p.bias;
      auto beta_tensor = TensorWrapper(bias, weight_shape, w_dtype);
      auto *mu = p.mu;
      auto mu_tensor = TensorWrapper(mu, intermediates_shape, DType::kFloat32);

      nvte_layernorm_fwd(
          input_tensor.data(), gamma_tensor.data(), beta_tensor.data(),
          eps, output_tensor.data(), mu_tensor.data(),
          rsigma_tensor.data(), stream, num_sm,
          workspace_tensor.data(), barrier_tensor.data());
  }
  else {
      nvte_rmsnorm_fwd(
          input_tensor.data(), gamma_tensor.data(), eps,
          output_tensor.data(), rsigma_tensor.data(), stream, num_sm,
          workspace_tensor.data(), barrier_tensor.data());
  }
}

void LayerNormBackwardImpl(void *grad_output, void *mu, void *rsigma, void *x,
                           void *weight, std::uint64_t n, std::uint64_t hidden,
                           float eps, DType x_dtype, DType w_dtype,
                           cudaStream_t stream, void *xgrad, void *dgamma,
                           void *dbeta) {
  auto input_shape = std::vector<size_t>{n, hidden};
  auto weight_shape = std::vector<size_t>{hidden};
  auto intermediates_shape = std::vector<size_t>{n};
  auto intermediates_dtype = DType::kFloat32;

  // assume input type = output type
  auto dz_tensor = TensorWrapper(grad_output, input_shape, x_dtype);

  auto mu_tensor = TensorWrapper(mu, intermediates_shape, intermediates_dtype);

  auto rsigma_tensor =
      TensorWrapper(rsigma, intermediates_shape, intermediates_dtype);

  auto x_tensor = TensorWrapper(x, input_shape, x_dtype);

  auto gamma_tensor = TensorWrapper(weight, weight_shape, w_dtype);

  auto xgrad_tensor = TensorWrapper(xgrad, input_shape, x_dtype);

  auto dgamma_tensor = TensorWrapper(dgamma, weight_shape, w_dtype);

  auto dbeta_tensor = TensorWrapper(dbeta, weight_shape, w_dtype);

  TensorWrapper dummy_workspace_tensor, dummy_barrier_tensor;
  TensorWrapper dummy_dgamma_part_tensor, dummy_dbeta_part_tensor;

  // The first call is to query the workspace
  nvte_layernorm_bwd(
      dz_tensor.data(), x_tensor.data(), mu_tensor.data(), rsigma_tensor.data(),
      gamma_tensor.data(), xgrad_tensor.data(), dgamma_tensor.data(),
      dbeta_tensor.data(), dummy_dgamma_part_tensor.data(),
      dummy_dbeta_part_tensor.data(), stream,
      cudaDevicePropertiesManager::Instance().GetMultiProcessorCount(),
      dummy_workspace_tensor.data(), dummy_barrier_tensor.data());

  size_t workspace_size = dummy_workspace_tensor.shape().data[0] *
                          typeToSize(dummy_workspace_tensor.dtype());
  size_t barrier_size = dummy_barrier_tensor.shape().data[0] *
                        typeToSize(dummy_barrier_tensor.dtype());
  size_t dgamma_part_size = dummy_dgamma_part_tensor.shape().data[0] *
                            dummy_dgamma_part_tensor.shape().data[1] *
                            typeToSize(dummy_dgamma_part_tensor.dtype());
  size_t dbeta_part_size = dummy_dbeta_part_tensor.shape().data[0] *
                           dummy_dbeta_part_tensor.shape().data[1] *
                           typeToSize(dummy_dbeta_part_tensor.dtype());

  size_t total_workspace_size =
      workspace_size + barrier_size + dgamma_part_size + dbeta_part_size;

  void *workspace =
      cublasLtMetaManager::Instance().GetWorkspace(total_workspace_size);
  void *dgamma_part = static_cast<char *>(workspace) + workspace_size;
  void *dbeta_part = static_cast<char *>(dgamma_part) + dgamma_part_size;
  void *barrier = static_cast<char *>(dbeta_part) + dbeta_part_size;

  auto workspace_tensor =
      TensorWrapper(workspace, dummy_workspace_tensor.shape(),
                    dummy_workspace_tensor.dtype());

  auto barrier_tensor = TensorWrapper(barrier, dummy_barrier_tensor.shape(),
                                      dummy_barrier_tensor.dtype());

  auto dgamma_part_tensor =
      TensorWrapper(dgamma_part, dummy_dgamma_part_tensor.shape(),
                    dummy_dgamma_part_tensor.dtype());

  auto dbeta_part_tensor =
      TensorWrapper(dbeta_part, dummy_dbeta_part_tensor.shape(),
                    dummy_dbeta_part_tensor.dtype());

  nvte_layernorm_bwd(
      dz_tensor.data(), x_tensor.data(), mu_tensor.data(), rsigma_tensor.data(),
      gamma_tensor.data(), xgrad_tensor.data(), dgamma_tensor.data(),
      dbeta_tensor.data(), dgamma_part_tensor.data(), dbeta_part_tensor.data(),
      stream, cudaDevicePropertiesManager::Instance().GetMultiProcessorCount(),
      workspace_tensor.data(), barrier_tensor.data());
}

void TELayerNormForwardFP8(cudaStream_t stream, void **buffers,
                           const char *opaque, std::size_t opaque_len) {
  const LayerNormDescriptor &descriptor =
      *UnpackDescriptor<LayerNormDescriptor>(opaque, opaque_len);

  NormalFwdParameter p{};
  // input
  p.x = {buffers[0], static_cast<DType>(descriptor.x_dtype)};
  p.w = {buffers[1], static_cast<DType>(descriptor.w_dtype)};
  p.bias = buffers[2];

  // attributes
  p.n = descriptor.n;
  p.hidden = descriptor.hidden;
  p.eps = descriptor.eps;

  // fp8 meta
  p.meta = FP8Meta{buffers[3], buffers[4], buffers[5], buffers[9]};

  // output
  p.y = {buffers[6], DType::kFloat8E4M3};
  p.mu = buffers[7];
  p.rsigma = buffers[8];

  LayerNormForwardImpl(p, stream);
}

void TELayerNormForward(cudaStream_t stream, void **buffers, const char *opaque,
                        std::size_t opaque_len) {
  const LayerNormDescriptor &descriptor =
      *UnpackDescriptor<LayerNormDescriptor>(opaque, opaque_len);

  NormalFwdParameter p{};
  // input
  p.x = {buffers[0], static_cast<DType>(descriptor.x_dtype)};
  p.w = {buffers[1], static_cast<DType>(descriptor.w_dtype)};
  p.bias = buffers[2];

  // attributes
  p.n = descriptor.n;
  p.hidden = descriptor.hidden;
  p.eps = descriptor.eps;

  // output
  p.y = {buffers[3], p.x.dtype};
  p.mu = buffers[4];
  p.rsigma = buffers[5];

  LayerNormForwardImpl(p, stream);
}

void TELayerNormBackward(cudaStream_t stream, void **buffers,
                         const char *opaque, std::size_t opaque_len) {
  const LayerNormDescriptor &descriptor =
      *UnpackDescriptor<LayerNormDescriptor>(opaque, opaque_len);

  // input
  void *grad_output = buffers[0];
  void *mu = buffers[1];
  void *rsigma = buffers[2];
  void *x = buffers[3];
  void *weight = buffers[4];

  // output
  void *xgrad = buffers[5];
  void *dgamma = buffers[6];
  void *dbeta = buffers[7];

  // attributes
  auto n = descriptor.n;
  auto hidden = descriptor.hidden;
  auto eps = descriptor.eps;
  auto x_dtype = descriptor.x_dtype;
  auto w_dtype = descriptor.w_dtype;

  LayerNormBackwardImpl(grad_output, mu, rsigma, x, weight, n, hidden, eps,
                        static_cast<DType>(x_dtype),
                        static_cast<DType>(w_dtype), stream, xgrad, dgamma,
                        dbeta);
}


void RMSNormBackwardImpl(void *grad_output, void *rsigma, void *x, void *weight,
                         std::uint64_t n, std::uint64_t hidden, float eps,
                         DType x_dtype, DType w_dtype, cudaStream_t stream,
                         void *xgrad, void *wgrad) {
  auto input_shape = std::vector<size_t>{n, hidden};
  auto weight_shape = std::vector<size_t>{hidden};
  auto intermediates_shape = std::vector<size_t>{n};
  auto intermediates_dtype = DType::kFloat32;

  // assume input type = output type
  auto dz_tensor = TensorWrapper(grad_output, input_shape, x_dtype);

  auto rsigma_tensor =
      TensorWrapper(rsigma, intermediates_shape, intermediates_dtype);

  auto x_tensor = TensorWrapper(x, input_shape, x_dtype);

  auto gamma_tensor = TensorWrapper(weight, weight_shape, w_dtype);

  auto xgrad_tensor = TensorWrapper(xgrad, input_shape, x_dtype);

  auto wgrad_tensor = TensorWrapper(wgrad, weight_shape, w_dtype);

  TensorWrapper dummy_workspace_tensor, dummy_barrier_tensor;
  TensorWrapper dummy_dgamma_part_tensor;

  // The first call is to query the workspace
  nvte_rmsnorm_bwd(
      dz_tensor.data(), x_tensor.data(), rsigma_tensor.data(),
      gamma_tensor.data(), xgrad_tensor.data(), wgrad_tensor.data(),
      dummy_dgamma_part_tensor.data(), stream,
      cudaDevicePropertiesManager::Instance().GetMultiProcessorCount(),
      dummy_workspace_tensor.data(), dummy_barrier_tensor.data());

  size_t workspace_size = dummy_workspace_tensor.shape().data[0] *
                          typeToSize(dummy_workspace_tensor.dtype());
  size_t barrier_size = dummy_barrier_tensor.shape().data[0] *
                        typeToSize(dummy_barrier_tensor.dtype());
  size_t dgamma_part_size = dummy_dgamma_part_tensor.shape().data[0] *
                            dummy_dgamma_part_tensor.shape().data[1] *
                            typeToSize(dummy_dgamma_part_tensor.dtype());

  size_t total_workspace_size =
      workspace_size + barrier_size + dgamma_part_size;

  void *workspace =
      cublasLtMetaManager::Instance().GetWorkspace(total_workspace_size);
  void *dgamma_part = reinterpret_cast<char *>(workspace) + workspace_size;
  void *barrier = reinterpret_cast<char *>(dgamma_part) + dgamma_part_size;

  auto workspace_tensor =
      TensorWrapper(workspace, dummy_workspace_tensor.shape(),
                    dummy_workspace_tensor.dtype());

  auto barrier_tensor = TensorWrapper(barrier, dummy_barrier_tensor.shape(),
                                      dummy_barrier_tensor.dtype());

  auto dgamma_part_tensor =
      TensorWrapper(dgamma_part, dummy_dgamma_part_tensor.shape(),
                    dummy_dgamma_part_tensor.dtype());

  nvte_rmsnorm_bwd(
      dz_tensor.data(), x_tensor.data(), rsigma_tensor.data(),
      gamma_tensor.data(), xgrad_tensor.data(), wgrad_tensor.data(),
      dgamma_part_tensor.data(), stream,
      cudaDevicePropertiesManager::Instance().GetMultiProcessorCount(),
      workspace_tensor.data(), barrier_tensor.data());
}

void TERMSNormForwardFP8(cudaStream_t stream, void **buffers,
                         const char *opaque, std::size_t opaque_len) {
  const LayerNormDescriptor &descriptor =
      *UnpackDescriptor<LayerNormDescriptor>(opaque, opaque_len);

  NormalFwdParameter p{};
  // input
  p.x = {buffers[0], static_cast<DType>(descriptor.x_dtype)};
  p.w = {buffers[1], static_cast<DType>(descriptor.w_dtype)};

  // attributes
  p.n = descriptor.n;
  p.hidden = descriptor.hidden;
  p.eps = descriptor.eps;

  // fp8 meta
  p.meta = FP8Meta{buffers[2], buffers[3], buffers[4], buffers[7]};

  // output
  p.y = {buffers[5], DType::kFloat8E4M3};
  p.rsigma = buffers[6];

  LayerNormForwardImpl(p, stream);
}

void TERMSNormForward(cudaStream_t stream, void **buffers, const char *opaque,
                      std::size_t opaque_len) {
  const LayerNormDescriptor &descriptor =
      *UnpackDescriptor<LayerNormDescriptor>(opaque, opaque_len);

  NormalFwdParameter p{};
  // input
  p.x = {buffers[0], static_cast<DType>(descriptor.x_dtype)};
  p.w = {buffers[1], static_cast<DType>(descriptor.w_dtype)};

  // attributes
  p.n = descriptor.n;
  p.hidden = descriptor.hidden;
  p.eps = descriptor.eps;

  // output
  p.y = {buffers[2], p.x.dtype};
  p.rsigma = buffers[3];

  LayerNormForwardImpl(p, stream);
}

void TERMSNormBackward(cudaStream_t stream, void **buffers, const char *opaque,
                       std::size_t opaque_len) {
  const LayerNormDescriptor &descriptor =
      *UnpackDescriptor<LayerNormDescriptor>(opaque, opaque_len);

  // input
  void *grad_output = buffers[0];
  void *rsigma = buffers[1];
  void *x = buffers[2];
  void *weight = buffers[3];

  // output
  void *xgrad = buffers[4];
  void *wgrad = buffers[5];

  // attributes
  auto n = descriptor.n;
  auto hidden = descriptor.hidden;
  auto eps = descriptor.eps;
  auto x_dtype = descriptor.x_dtype;
  auto w_dtype = descriptor.w_dtype;

  RMSNormBackwardImpl(grad_output, rsigma, x, weight, n, hidden, eps,
                      static_cast<DType>(x_dtype), static_cast<DType>(w_dtype),
                      stream, xgrad, wgrad);
}

}  // namespace jax
}  // namespace transformer_engine

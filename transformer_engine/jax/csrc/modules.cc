/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
                                    std::uint64_t fwd_ctype,
                                    std::uint64_t bwd_ctype) {
  return PackDescriptor(TEMatDescriptor{m, n, fwd_ctype, bwd_ctype});
}

pybind11::bytes PackTEGemmDescriptor(std::uint64_t m, std::uint64_t n,
                                     std::uint64_t k, std::uint64_t A_ctype,
                                     std::uint64_t B_ctype,
                                     std::uint64_t D_ctype, bool transa,
                                     bool transb) {
  return PackDescriptor(
      TEGemmDescriptor{m, n, k, A_ctype, B_ctype, D_ctype, transa, transb});
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
  const float *scale = reinterpret_cast<const float *>(buffers[2]);
  const float *scale_inv = reinterpret_cast<const float *>(buffers[3]);

  void *input_cast = buffers[4];
  void *input_cast_trans = buffers[5];
  float *amax_out = reinterpret_cast<float *>(buffers[6]);

  auto input_shape = std::vector<size_t>{d.m, d.n};
  auto input_trans_shape = std::vector<size_t>{d.n, d.m};

  auto input_tensor = TensorWrapper(const_cast<void *>(input), input_shape,
                                    static_cast<DType>(d.in_ctype));
  auto input_scale_tensor =
      TensorWrapper(const_cast<void *>((const void *)scale),
                    std::vector<size_t>{1}, DType::kFloat32);
  auto input_scale_inverse_tensor =
      TensorWrapper(const_cast<void *>((const void *)scale_inv),
                    std::vector<size_t>{1}, DType::kFloat32);
  auto input_cast_tensor =
      TensorWrapper(input_cast, input_shape, static_cast<DType>(d.out_ctype));
  auto input_cast_trans_tensor = TensorWrapper(
      input_cast_trans, input_trans_shape, static_cast<DType>(d.out_ctype));
  auto input_amax_tensor =
      TensorWrapper(reinterpret_cast<void *>(amax_out), std::vector<size_t>{1},
                    DType::kFloat32);

  NVTE_CHECK_CUDA(cudaMemcpyAsync(amax_out, amax, sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream));

  nvte_cast_transpose(input_tensor.data(), input_scale_tensor.data(),
                      input_cast_tensor.data(), input_cast_trans_tensor.data(),
                      input_amax_tensor.data(),
                      input_scale_inverse_tensor.data(), stream);
}

void TEGatedGelu(cudaStream_t stream, void **buffers, const char *opaque,
                 std::size_t opaque_len) {
  const TEMatDescriptor &d =
      *UnpackDescriptor<TEMatDescriptor>(opaque, opaque_len);

  const void *input = reinterpret_cast<const void *>(buffers[0]);
  const void *amax = reinterpret_cast<const void *>(buffers[1]);
  const float *scale = reinterpret_cast<const float *>(buffers[2]);
  const float *scale_inv = reinterpret_cast<const float *>(buffers[3]);

  void *output = buffers[4];
  float *amax_out = reinterpret_cast<float *>(buffers[5]);

  auto input_shape = std::vector<size_t>{d.m, d.n * 2};
  auto output_shape = std::vector<size_t>{d.m, d.n};

  auto input_tensor = TensorWrapper(const_cast<void *>(input), input_shape,
                                    static_cast<DType>(d.in_ctype));
  auto input_scale_tensor =
      TensorWrapper(const_cast<void *>((const void *)scale),
                    std::vector<size_t>{1}, DType::kFloat32);
  auto input_scale_inverse_tensor =
      TensorWrapper(const_cast<void *>((const void *)scale_inv),
                    std::vector<size_t>{1}, DType::kFloat32);
  auto output_tensor =
      TensorWrapper(output, output_shape, static_cast<DType>(d.out_ctype));
  auto input_amax_tensor =
      TensorWrapper(reinterpret_cast<void *>(amax_out), std::vector<size_t>{1},
                    DType::kFloat32);

  NVTE_CHECK_CUDA(cudaMemcpyAsync(amax_out, amax, sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream));

  nvte_gated_gelu(input_tensor.data(), output_tensor.data(),
                  input_scale_tensor.data(), input_amax_tensor.data(),
                  input_scale_inverse_tensor.data(), stream);
}

void TECastTransposeDGatedGelu(cudaStream_t stream, void **buffers,
                               const char *opaque, std::size_t opaque_len) {
  const TEMatDescriptor &d =
      *UnpackDescriptor<TEMatDescriptor>(opaque, opaque_len);

  const void *input = reinterpret_cast<const void *>(buffers[0]);
  const void *gelu_input = reinterpret_cast<const void *>(buffers[1]);
  const void *amax = reinterpret_cast<const void *>(buffers[2]);
  const float *scale = reinterpret_cast<const float *>(buffers[3]);
  const float *scale_inv = reinterpret_cast<const float *>(buffers[4]);

  void *output = buffers[5];
  void *output_trans = buffers[6];
  float *amax_out = reinterpret_cast<float *>(buffers[7]);

  auto input_shape = std::vector<size_t>{d.m, d.n};
  auto gelu_input_shape = std::vector<size_t>{d.m, d.n * 2};
  auto output_shape = std::vector<size_t>{d.m, d.n * 2};
  auto output_trans_shape = std::vector<size_t>{d.n * 2, d.m};

  auto input_tensor = TensorWrapper(const_cast<void *>(input), input_shape,
                                    static_cast<DType>(d.in_ctype));
  auto gelu_input_tensor =
      TensorWrapper(const_cast<void *>(gelu_input), gelu_input_shape,
                    static_cast<DType>(d.in_ctype));
  auto input_scale_tensor =
      TensorWrapper(const_cast<void *>((const void *)scale),
                    std::vector<size_t>{1}, DType::kFloat32);
  auto input_scale_inverse_tensor =
      TensorWrapper(const_cast<void *>((const void *)scale_inv),
                    std::vector<size_t>{1}, DType::kFloat32);
  auto output_tensor =
      TensorWrapper(output, output_shape, static_cast<DType>(d.out_ctype));
  auto output_trans_tensor = TensorWrapper(output_trans, output_trans_shape,
                                           static_cast<DType>(d.out_ctype));
  auto input_amax_tensor =
      TensorWrapper(reinterpret_cast<void *>(amax_out), std::vector<size_t>{1},
                    DType::kFloat32);

  NVTE_CHECK_CUDA(cudaMemcpyAsync(amax_out, amax, sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream));

  nvte_cast_transpose_dgated_gelu(
      input_tensor.data(), gelu_input_tensor.data(), input_scale_tensor.data(),
      output_tensor.data(), output_trans_tensor.data(),
      input_amax_tensor.data(), input_scale_inverse_tensor.data(), stream);
}

void TEGemm(cudaStream_t stream, void **buffers, const char *opaque,
            std::size_t opaque_len) {
  const TEGemmDescriptor &d =
      *UnpackDescriptor<TEGemmDescriptor>(opaque, opaque_len);

  const void *A = reinterpret_cast<const void *>(buffers[0]);
  const void *B = reinterpret_cast<const void *>(buffers[1]);
  const float *A_scale_inverse = reinterpret_cast<const float *>(buffers[2]);
  const float *B_scale_inverse = reinterpret_cast<const float *>(buffers[3]);

  void *D = buffers[4];

  // We transposes shape of A, B and D here to correctly invoke
  // cuBlasLt GEMM (col-major) for row-major data.
  auto A_shape = std::vector<size_t>{d.k, d.m};
  auto A_tensor = TensorWrapper(const_cast<void *>(A), A_shape,
                                static_cast<DType>(d.A_ctype));
  auto A_scale_inverse_tensor =
      TensorWrapper(const_cast<void *>((const void *)A_scale_inverse),
                    std::vector<size_t>{1}, DType::kFloat32);

  auto B_shape = std::vector<size_t>{d.n, d.k};
  auto B_tensor = TensorWrapper(const_cast<void *>(B), B_shape,
                                static_cast<DType>(d.B_ctype));
  auto B_scale_inverse_tensor =
      TensorWrapper(const_cast<void *>((const void *)B_scale_inverse),
                    std::vector<size_t>{1}, DType::kFloat32);

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

  nvte_cublas_gemm(A_tensor.data(), A_scale_inverse_tensor.data(),
                   B_tensor.data(), B_scale_inverse_tensor.data(),
                   D_tensor.data(), null_tensor.data(), null_tensor.data(),
                   (d.transa) ? CUBLAS_OP_T : CUBLAS_OP_N,
                   (d.transb) ? CUBLAS_OP_T : CUBLAS_OP_N, false,
                   wk_tensor.data(), false, false, stream);
}

void RMSNormForwardImpl(void *input, void *weight, std::uint64_t n,
                        std::uint64_t hidden, float eps, DType x_dtype,
                        DType w_dtype, float *scale, float *scale_inverse,
                        float *amax, cudaStream_t stream, void *output,
                        void *rsigma, bool fp8_out) {
  auto input_shape = std::vector<size_t>{n, hidden};
  auto weight_shape = std::vector<size_t>{hidden};
  auto intermediates_shape = std::vector<size_t>{n};

  auto input_tensor = TensorWrapper(input, input_shape, x_dtype);

  auto gamma_tensor = TensorWrapper(weight, weight_shape, w_dtype);

  auto output_dtype = fp8_out ? DType::kFloat8E4M3 : x_dtype;
  // assume output dtype = input dtype
  // If we need mixed I/O precision in the future, we need an additional
  // parameter for output type
  auto output_tensor = TensorWrapper(output, input_shape, output_dtype);

  auto rsigma_tensor =
      TensorWrapper(rsigma, intermediates_shape, DType::kFloat32);

  auto scale_tensor =
      TensorWrapper(scale, std::vector<size_t>{1}, DType::kFloat32);

  auto scale_inverse_tensor =
      TensorWrapper(scale_inverse, std::vector<size_t>{1}, DType::kFloat32);

  auto amax_tensor =
      TensorWrapper(amax, std::vector<size_t>{1}, DType::kFloat32);

  // Create uninitialized workspace, barrier and init them on the first
  // nvte_rmsnorm_fwd
  TensorWrapper dummy_workspace_tensor, dummy_barrier_tensor;

  // The first call is to query the required workspace
  nvte_rmsnorm_fwd(
      input_tensor.data(), gamma_tensor.data(), scale_tensor.data(), eps,
      output_tensor.data(), rsigma_tensor.data(), stream,
      cudaDevicePropertiesManager::Instance().GetMultiProcessorCount(),
      dummy_workspace_tensor.data(), dummy_barrier_tensor.data(),
      amax_tensor.data(), scale_inverse_tensor.data(), fp8_out);

  // TODO(rewang): normalize
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

  nvte_rmsnorm_fwd(
      input_tensor.data(), gamma_tensor.data(), scale_tensor.data(), eps,
      output_tensor.data(), rsigma_tensor.data(), stream,
      cudaDevicePropertiesManager::Instance().GetMultiProcessorCount(),
      workspace_tensor.data(), barrier_tensor.data(), amax_tensor.data(),
      scale_inverse_tensor.data(), fp8_out);
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

  // dummy tensor, TODO(rewang): remove this in the production code
  auto mu_tensor =
      TensorWrapper(nullptr, intermediates_shape, intermediates_dtype);

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
  void *barrier = reinterpret_cast<char *>(workspace) + workspace_size;
  void *dgamma_part = reinterpret_cast<char *>(barrier) + barrier_size;

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

pybind11::bytes PackRMSNormDescriptor(std::uint64_t n, std::uint64_t hidden,
                                      std::uint64_t x_dtype,
                                      std::uint64_t w_dtype, float eps) {
  return PackDescriptor(RMSNormDescriptor{n, hidden, x_dtype, w_dtype, eps});
}

void TERMSNormForwardFP8(cudaStream_t stream, void **buffers,
                         const char *opaque, std::size_t opaque_len) {
  const RMSNormDescriptor &descriptor =
      *UnpackDescriptor<RMSNormDescriptor>(opaque, opaque_len);

  // input
  void *input = buffers[0];
  void *weight = buffers[1];
  void *amax = buffers[2];
  float *scale = static_cast<float *>(buffers[3]);
  float *scale_inverse = static_cast<float *>(buffers[4]);

  // output
  void *output = buffers[5];
  void *rsigma = buffers[6];
  float *amax_out = static_cast<float *>(
      buffers[7]);  // TODO: remove this if jax support inplaced I/O

  // attributes
  auto n = descriptor.n;
  auto hidden = descriptor.hidden;
  auto eps = descriptor.eps;
  auto x_dtype = descriptor.x_dtype;
  auto w_dtype = descriptor.w_dtype;

  // TODO: remove this if jax support inplaced I/O
  NVTE_CHECK_CUDA(cudaMemcpyAsync(amax_out, amax, sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream));

  RMSNormForwardImpl(input, weight, n, hidden, eps, static_cast<DType>(x_dtype),
                     static_cast<DType>(w_dtype), scale, scale_inverse,
                     amax_out, stream, output, rsigma, true);
}

void TERMSNormForward(cudaStream_t stream, void **buffers, const char *opaque,
                      std::size_t opaque_len) {
  const RMSNormDescriptor &descriptor =
      *UnpackDescriptor<RMSNormDescriptor>(opaque, opaque_len);

  // input
  void *input = buffers[0];
  void *weight = buffers[1];

  // output
  void *output = buffers[2];
  void *rsigma = buffers[3];

  // attributes
  auto n = descriptor.n;
  auto hidden = descriptor.hidden;
  auto eps = descriptor.eps;
  auto x_dtype = descriptor.x_dtype;
  auto w_dtype = descriptor.w_dtype;

  RMSNormForwardImpl(input, weight, n, hidden, eps, static_cast<DType>(x_dtype),
                     static_cast<DType>(w_dtype), nullptr, nullptr, nullptr,
                     stream, output, rsigma, false);
}

void TERMSNormBackward(cudaStream_t stream, void **buffers, const char *opaque,
                       std::size_t opaque_len) {
  const RMSNormDescriptor &descriptor =
      *UnpackDescriptor<RMSNormDescriptor>(opaque, opaque_len);

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

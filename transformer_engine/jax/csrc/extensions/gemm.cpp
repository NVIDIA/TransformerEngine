/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "transformer_engine/gemm.h"

#include <memory>

#include "common/util/cuda_runtime.h"
#include "common/util/system.h"
#include "extensions.h"
#include "xla/ffi/api/c_api.h"

namespace transformer_engine {
namespace jax {

constexpr static size_t MXFP8_BLOCK_SIZE = 32;

// Note: we only support TN-GEMM for now (TN in cuBLASLt == NT in JAX)
Error_Type GroupedGemmImpl(uint8_t *lhs_ptr, const DType &lhs_dtype, uint8_t *lhs_sinv_ptr,
                           const DType &lhs_sinv_dtype, uint8_t *rhs_ptr, const DType &rhs_dtype,
                           uint8_t *rhs_sinv_ptr, const DType &rhs_sinv_dtype, uint8_t *bias_ptr,
                           const DType &bias_dtype, uint8_t *out_ptr, const DType &out_dtype,
                           uint8_t *workspace_ptr, const size_t workspace_size, size_t num_gemms,
                           int32_t *dim_list_ptr, const JAXX_Scaling_Mode scaling_mode,
                           cudaStream_t stream) {
  size_t lhs_dtype_bytes = te_dtype_bytes(lhs_dtype);
  size_t rhs_dtype_bytes = te_dtype_bytes(rhs_dtype);
  size_t lhs_sinv_dtype_bytes = te_dtype_bytes(lhs_sinv_dtype);
  size_t rhs_sinv_dtype_bytes = te_dtype_bytes(rhs_sinv_dtype);
  size_t bias_dtype_bytes = te_dtype_bytes(bias_dtype);
  size_t out_dtype_bytes = te_dtype_bytes(out_dtype);
  NVTE_CHECK(lhs_dtype_bytes == rhs_dtype_bytes, "sizeof(lhs_dtype) != sizeof(rhs_dtype)");
  NVTE_CHECK(lhs_sinv_dtype_bytes == rhs_sinv_dtype_bytes,
             "sizeof(lhs_sinv_dtype) != sizeof(rhs_sinv_dtype)");

  size_t dim_list_bytes = sizeof(int32_t) * 3 * num_gemms;
  std::unique_ptr<int32_t[]> dim_list_host = std::make_unique<int32_t[]>(3 * num_gemms);

  cudaMemcpyAsync(dim_list_host.get(), dim_list_ptr, dim_list_bytes, cudaMemcpyDeviceToHost,
                  stream);
  // Note: This may break cudaGraph.
  cudaStreamSynchronize(stream);

  // Notes on matrix layouts and transpose:
  // Jax uses row-major data_layout, on entering this function, each input matrix pair:
  //   A: row-major with size [m, k],
  //   B: row-major with size [n, k], needs transpose,
  // on exiting this function, JAX expect:
  //   C: row-major with size [m, n].
  // cuBLAS uses column-major data_layout, in this view, each input matrix pair:
  //   A: column-major with size [k, m], needs transpose,
  //   B: column-major with size [k, n].
  // If we call cuBLAS GEMM for A * B, the output will be:
  //   C: column-major with size [m, n] --> row-major with size [n, m].
  // To make the output compatible with JAX, we need to swap A and B in cuBLAS GEMM call.

  bool trans_lhs = true;
  bool trans_rhs = false;
  auto num_math_sm = cuda::sm_count() - getenv<int>("NVTE_EXT_MARGIN_SM", 0);
  bool grad = false;
  bool accumulate = false;
  bool use_split_accumulator = false;

  // These lists are to keep the TensorWrapper objects alive
  std::vector<TensorWrapper> lhs_wrapper_list;
  std::vector<TensorWrapper> rhs_wrapper_list;
  std::vector<TensorWrapper> bias_wrapper_list;
  std::vector<TensorWrapper> pre_gelu_wrapper_list;
  std::vector<TensorWrapper> out_wrapper_list;
  std::vector<TensorWrapper> workspace_wrapper_list;

  // These lists are the actual NVTETensor (void *) lists for multi-stream GEMM
  std::vector<NVTETensor> lhs_list;
  std::vector<NVTETensor> rhs_list;
  std::vector<NVTETensor> bias_list;
  std::vector<NVTETensor> pre_gelu_list;
  std::vector<NVTETensor> out_list;
  std::vector<NVTETensor> workspace_list;

  for (int i = 0; i < num_gemms; i++) {
    size_t m = dim_list_host[i * 3];
    size_t n = dim_list_host[i * 3 + 1];
    size_t k = dim_list_host[i * 3 + 2];

    auto lhs_shape = std::vector<size_t>{m, k};
    auto rhs_shape = std::vector<size_t>{n, k};
    auto out_shape = std::vector<size_t>{n, m};
    auto lhs_sinv_shape = std::vector<size_t>{1, 1};
    auto rhs_sinv_shape = std::vector<size_t>{1, 1};

    auto lhs_i = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
    auto rhs_i = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
    lhs_i.set_rowwise_data(static_cast<void *>(lhs_ptr), lhs_dtype, lhs_shape);
    rhs_i.set_rowwise_data(static_cast<void *>(rhs_ptr), rhs_dtype, rhs_shape);

    if (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING) {
      lhs_i.set_rowwise_scale_inv(static_cast<void *>(lhs_sinv_ptr), DType::kFloat32,
                                  std::vector<size_t>{1});
      rhs_i.set_rowwise_scale_inv(static_cast<void *>(rhs_sinv_ptr), DType::kFloat32,
                                  std::vector<size_t>{1});
    } else if (scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING) {
      NVTE_CHECK(k % MXFP8_BLOCK_SIZE == 0, "MXFP8 K-dim being divisble by %d (got %d)",
                 MXFP8_BLOCK_SIZE, k);
      size_t sinv_k = k / MXFP8_BLOCK_SIZE;
      lhs_sinv_shape[0] = m;
      lhs_sinv_shape[1] = sinv_k;
      rhs_sinv_shape[0] = n;
      rhs_sinv_shape[1] = sinv_k;

      // Note: the scale_inv array should have been swizzled in Python before lowering
      lhs_i.set_rowwise_scale_inv(static_cast<void *>(lhs_sinv_ptr), DType::kFloat8E8M0,
                                  lhs_sinv_shape);
      rhs_i.set_rowwise_scale_inv(static_cast<void *>(rhs_sinv_ptr), DType::kFloat8E8M0,
                                  rhs_sinv_shape);
    } else {
      NVTE_ERROR("Unsupported scaling mode: ", static_cast<int>(scaling_mode));
    }
    lhs_wrapper_list.push_back(std::move(lhs_i));
    rhs_wrapper_list.push_back(std::move(rhs_i));

    auto out_i = TensorWrapper(static_cast<void *>(out_ptr), out_shape, out_dtype);
    lhs_ptr += m * k * lhs_dtype_bytes;
    rhs_ptr += n * k * rhs_dtype_bytes;
    out_ptr += m * n * out_dtype_bytes;
    lhs_sinv_ptr += lhs_sinv_shape[0] * lhs_sinv_shape[1] * lhs_sinv_dtype_bytes;
    rhs_sinv_ptr += rhs_sinv_shape[0] * rhs_sinv_shape[1] * rhs_sinv_dtype_bytes;

    void *pre_gelu_ptr = nullptr;
    auto bias_shape = std::vector<size_t>{0};
    auto pre_gelu_shape = std::vector<size_t>{0};
    if (bias_ptr != nullptr) bias_shape[0] = n;
    auto bias_i = TensorWrapper(bias_ptr, bias_shape, bias_dtype);
    if (bias_ptr != nullptr) bias_ptr += n * bias_dtype_bytes;
    auto pre_gelu_i = TensorWrapper(pre_gelu_ptr, pre_gelu_shape, out_dtype);

    out_wrapper_list.push_back(std::move(out_i));
    bias_wrapper_list.push_back(std::move(bias_i));
    pre_gelu_wrapper_list.push_back(std::move(pre_gelu_i));

    lhs_list.push_back(lhs_wrapper_list.back().data());
    rhs_list.push_back(rhs_wrapper_list.back().data());
    bias_list.push_back(bias_wrapper_list.back().data());
    pre_gelu_list.push_back(pre_gelu_wrapper_list.back().data());
    out_list.push_back(out_wrapper_list.back().data());
  }

  auto workspace_shape = std::vector<size_t>{workspace_size};
  for (int i = 0; i < num_streams; i++) {
    auto workspace_i =
        TensorWrapper(static_cast<void *>(workspace_ptr), workspace_shape, DType::kByte);
    workspace_wrapper_list.push_back(std::move(workspace_i));
    workspace_list.push_back(workspace_wrapper_list.back().data());
    workspace_ptr += workspace_size;
  }

  nvte_multi_stream_cublas_gemm(rhs_list.data(), lhs_list.data(), out_list.data(), bias_list.data(),
                                pre_gelu_list.data(), num_gemms, trans_lhs, trans_rhs, grad,
                                workspace_list.data(), accumulate, use_split_accumulator,
                                num_math_sm, stream);

  return ffi_with_cuda_error_check();
}

Error_Type GroupedGemmFFI(cudaStream_t stream, Buffer_Type lhs_flatten,
                          Buffer_Type lhs_sinv_flatten, Buffer_Type rhs_flatten,
                          Buffer_Type rhs_sinv_flatten, Buffer_Type bias_flatten,
                          Buffer_Type dim_list, Result_Type out_flatten,
                          Result_Type workspace_flatten, int64_t num_gemms,
                          JAXX_Scaling_Mode scaling_mode) {
  // Inputs
  auto lhs_ptr = reinterpret_cast<uint8_t *>(lhs_flatten.untyped_data());
  auto rhs_ptr = reinterpret_cast<uint8_t *>(rhs_flatten.untyped_data());
  auto lhs_sinv_ptr = reinterpret_cast<uint8_t *>(lhs_sinv_flatten.untyped_data());
  auto rhs_sinv_ptr = reinterpret_cast<uint8_t *>(rhs_sinv_flatten.untyped_data());
  auto bias_ptr = reinterpret_cast<uint8_t *>(bias_flatten.untyped_data());
  auto dim_list_ptr = reinterpret_cast<int32_t *>(dim_list.untyped_data());
  auto lhs_dtype = convert_ffi_datatype_to_te_dtype(lhs_flatten.element_type());
  auto rhs_dtype = convert_ffi_datatype_to_te_dtype(rhs_flatten.element_type());
  auto lhs_sinv_dtype = convert_ffi_datatype_to_te_dtype(lhs_sinv_flatten.element_type());
  auto rhs_sinv_dtype = convert_ffi_datatype_to_te_dtype(rhs_sinv_flatten.element_type());
  auto bias_dtype = convert_ffi_datatype_to_te_dtype(bias_flatten.element_type());

  // Outputs
  auto out_ptr = reinterpret_cast<uint8_t *>(out_flatten->untyped_data());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(out_flatten->element_type());
  auto workspace_ptr = reinterpret_cast<uint8_t *>(workspace_flatten->untyped_data());
  auto workspace_size = workspace_flatten->dimensions().back() / num_streams;

  return GroupedGemmImpl(lhs_ptr, lhs_dtype, lhs_sinv_ptr, lhs_sinv_dtype, rhs_ptr, rhs_dtype,
                         rhs_sinv_ptr, rhs_sinv_dtype, bias_ptr, bias_dtype, out_ptr, out_dtype,
                         workspace_ptr, workspace_size, num_gemms, dim_list_ptr, scaling_mode,
                         stream);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmHandler, GroupedGemmFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // lhs_flatten
                                  .Arg<Buffer_Type>()      // lhs_sinv_flatten
                                  .Arg<Buffer_Type>()      // rhs_flatten
                                  .Arg<Buffer_Type>()      // rhs_sinv_flatten
                                  .Arg<Buffer_Type>()      // bias_flatten
                                  .Arg<Buffer_Type>()      // dim_list
                                  .Ret<Buffer_Type>()      // out_flatten
                                  .Ret<Buffer_Type>()      // workspace_flatten
                                  .Attr<int64_t>("num_gemms")
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode"),
                              FFI_CudaGraph_Traits);


Error_Type GroupedGemmNewFFI(cudaStream_t stream, Variadic_Buffer_Type input_list,
                             Variadic_Result_Type output_list, int64_t num_gemms,
                             int64_t scaling_mode, int64_t has_bias, int64_t workspace_size) {
  // Notes on matrix layouts and transpose:
  // Jax uses row-major data_layout, on entering this function, each input matrix pair:
  //   A: row-major with size [m, k],
  //   B: row-major with size [n, k], needs transpose,
  // on exiting this function, JAX expect:
  //   C: row-major with size [m, n].
  // cuBLAS uses column-major data_layout, in this view, each input matrix pair:
  //   A: column-major with size [k, m], needs transpose,
  //   B: column-major with size [k, n].
  // If we call cuBLAS GEMM for A * B, the output will be:
  //   C: column-major with size [m, n] --> row-major with size [n, m].
  // To make the output compatible with JAX, we need to swap A and B in cuBLAS GEMM call.

  bool trans_lhs = true;
  bool trans_rhs = false;
  auto num_math_sm = cuda::sm_count() - getenv<int>("NVTE_EXT_MARGIN_SM", 0);
  bool grad = false;
  bool accumulate = false;
  bool use_split_accumulator = false;

  // These lists are to keep the TensorWrapper objects alive
  std::vector<TensorWrapper> lhs_wrapper_list;
  std::vector<TensorWrapper> rhs_wrapper_list;
  std::vector<TensorWrapper> bias_wrapper_list;
  std::vector<TensorWrapper> pre_gelu_wrapper_list;
  std::vector<TensorWrapper> out_wrapper_list;
  std::vector<TensorWrapper> workspace_wrapper_list;

  // These lists are the actual NVTETensor (void *) lists for multi-stream GEMM
  std::vector<NVTETensor> lhs_list;
  std::vector<NVTETensor> rhs_list;
  std::vector<NVTETensor> bias_list;
  std::vector<NVTETensor> pre_gelu_list;
  std::vector<NVTETensor> out_list;
  std::vector<NVTETensor> workspace_list;

  //printf("[DEBUG] num_gemms, scaling_mode, has_bias, workspace_size: %ld %ld %ld %ld\n",
  //       num_gemms, scaling_mode, has_bias, workspace_size);
  //fflush(stdout);

  int lhs_list_offset      = 0;
  int rhs_list_offset      = num_gemms;
  int lhs_sinv_list_offset = 2 * num_gemms;
  int rhs_sinv_list_offset = 3 * num_gemms;
  int bias_list_offset     = 4 * num_gemms;
  int out_list_offset      = 0;
  for (int i = 0; i < num_gemms; i++) {
    Buffer_Type lhs_i      = input_list.get<Buffer_Type>(lhs_list_offset + i).value();
    Buffer_Type rhs_i      = input_list.get<Buffer_Type>(rhs_list_offset + i).value();
    Buffer_Type lhs_sinv_i = input_list.get<Buffer_Type>(lhs_sinv_list_offset + i).value();
    Buffer_Type rhs_sinv_i = input_list.get<Buffer_Type>(rhs_sinv_list_offset + i).value();
    Result_Type out_i      = output_list.get<Buffer_Type>(out_list_offset + i).value();

    DType lhs_dtype = convert_ffi_datatype_to_te_dtype(lhs_i.element_type());
    DType rhs_dtype = convert_ffi_datatype_to_te_dtype(rhs_i.element_type());
    DType out_dtype = convert_ffi_datatype_to_te_dtype(out_i->element_type());

    void *lhs_ptr      = lhs_i.untyped_data();
    void *rhs_ptr      = rhs_i.untyped_data();
    void *lhs_sinv_ptr = lhs_sinv_i.untyped_data();
    void *rhs_sinv_ptr = rhs_sinv_i.untyped_data();
    void *out_ptr      = out_i->untyped_data();

    // Placeholder for bias since it can be empty
    DType bias_dtype = DType::kFloat32;
    void *bias_ptr   = nullptr;

    auto lhs_shape_ = lhs_i.dimensions();
    auto rhs_shape_ = rhs_i.dimensions();

    // lhs and rhs has shape [1, m, k] and [1, n, k]
    size_t m = lhs_shape_[1];
    size_t n = rhs_shape_[1];
    size_t k = lhs_shape_[2];

    auto lhs_shape = std::vector<size_t>{m, k};
    auto rhs_shape = std::vector<size_t>{n, k};
    auto out_shape = std::vector<size_t>{n, m};
    auto lhs_sinv_shape = std::vector<size_t>{1, 1};
    auto rhs_sinv_shape = std::vector<size_t>{1, 1};

    if (scaling_mode == NVTE_DELAYED_TENSOR_SCALING) {
      auto lhs_i_ = TensorWrapper(lhs_ptr, lhs_shape, lhs_dtype, nullptr, nullptr,
                                  reinterpret_cast<float *>(lhs_sinv_ptr));
      auto rhs_i_ = TensorWrapper(rhs_ptr, rhs_shape, rhs_dtype, nullptr, nullptr,
                                  reinterpret_cast<float *>(rhs_sinv_ptr));
      lhs_wrapper_list.push_back(std::move(lhs_i_));
      rhs_wrapper_list.push_back(std::move(rhs_i_));
    } else if (scaling_mode == NVTE_MXFP8_1D_SCALING) {
      NVTE_CHECK(k % MXFP8_BLOCK_SIZE == 0, "MXFP8 K-dim being divisble by %d (got %d)",
                 MXFP8_BLOCK_SIZE, k);
      size_t sinv_k = k / MXFP8_BLOCK_SIZE;
      lhs_sinv_shape[0] = m;
      lhs_sinv_shape[1] = sinv_k;
      rhs_sinv_shape[0] = n;
      rhs_sinv_shape[1] = sinv_k;

      // Note: the scale_inv array should have been swizzled in Python before lowering
      TensorWrapper lhs_i_(NVTE_MXFP8_1D_SCALING);
      TensorWrapper rhs_i_(NVTE_MXFP8_1D_SCALING);
      lhs_i_.set_rowwise_data(lhs_ptr, lhs_dtype, lhs_shape);
      rhs_i_.set_rowwise_data(rhs_ptr, rhs_dtype, rhs_shape);
      lhs_i_.set_rowwise_scale_inv(lhs_sinv_ptr, DType::kFloat8E8M0, lhs_sinv_shape);
      rhs_i_.set_rowwise_scale_inv(rhs_sinv_ptr, DType::kFloat8E8M0, rhs_sinv_shape);

      lhs_wrapper_list.push_back(std::move(lhs_i_));
      rhs_wrapper_list.push_back(std::move(rhs_i_));
    } else {
      NVTE_ERROR("Unsupported scaling mode: ", scaling_mode);
    }

    /*
    printf("[DEBUG] i: %d, lhs_dtype, rhs_dtype, out_dtype = %d, %d, %d\n", i, lhs_dtype, rhs_dtype, out_dtype);
    printf("[DEBUG] lhs_shape: %d %d, ptr = %p\n", lhs_shape_[1], lhs_shape_[2], lhs_ptr);
    printf("[DEBUG] rhs_shape: %d %d, ptr = %p\n", rhs_shape_[1], rhs_shape_[2], rhs_ptr);
    printf("[DEBUG] out_shape: %d %d, ptr = %p\n", out_i->dimensions()[1], out_i->dimensions()[2], out_ptr);
    printf("[DEBUG] lhs_sinv_shape: %d %d, ptr = %p\n", lhs_sinv_shape[0], lhs_sinv_shape[1], lhs_sinv_ptr);
    printf("[DEBUG] rhs_sinv_shape: %d %d, ptr = %p\n", rhs_sinv_shape[0], rhs_sinv_shape[1], rhs_sinv_ptr);
    fflush(stdout);
    */

    auto out_i_ = TensorWrapper(out_ptr, out_shape, out_dtype);
    void *pre_gelu_ptr = nullptr;
    auto bias_shape = std::vector<size_t>{0};
    auto pre_gelu_shape = std::vector<size_t>{0};
    if (has_bias)
    {
      auto bias_i_get = input_list.get<Buffer_Type>(bias_list_offset + i);
      Buffer_Type bias_i = bias_i_get.value();
      bias_ptr = bias_i.untyped_data();
      bias_dtype = convert_ffi_datatype_to_te_dtype(bias_i.element_type());
      bias_shape[0] = n;
    }
    auto bias_i = TensorWrapper(bias_ptr, bias_shape, bias_dtype);
    auto pre_gelu_i = TensorWrapper(pre_gelu_ptr, pre_gelu_shape, out_dtype);

    out_wrapper_list.push_back(std::move(out_i_));
    bias_wrapper_list.push_back(std::move(bias_i));
    pre_gelu_wrapper_list.push_back(std::move(pre_gelu_i));

    lhs_list.push_back(lhs_wrapper_list.back().data());
    rhs_list.push_back(rhs_wrapper_list.back().data());
    bias_list.push_back(bias_wrapper_list.back().data());
    pre_gelu_list.push_back(pre_gelu_wrapper_list.back().data());
    out_list.push_back(out_wrapper_list.back().data());
  }

  auto workspace_get = output_list.get<Buffer_Type>(num_gemms);
  Result_Type workspace = workspace_get.value();
  uint8_t *workspace_ptr = reinterpret_cast<uint8_t *>(workspace->untyped_data());
  auto workspace_shape = std::vector<size_t>{workspace_size};
  for (int i = 0; i < num_streams; i++) {
    auto workspace_i =
        TensorWrapper(static_cast<void *>(workspace_ptr), workspace_shape, DType::kByte);
    workspace_wrapper_list.push_back(std::move(workspace_i));
    workspace_list.push_back(workspace_wrapper_list.back().data());
    workspace_ptr += workspace_size;
  }
  //printf("[DEBUG] workspace packing done\n");
  //fflush(stdout);

  nvte_multi_stream_cublas_gemm(rhs_list.data(), lhs_list.data(), out_list.data(), bias_list.data(),
                                pre_gelu_list.data(), num_gemms, trans_lhs, trans_rhs, grad,
                                workspace_list.data(), accumulate, use_split_accumulator,
                                num_math_sm, stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmNewHandler, GroupedGemmNewFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .RemainingArgs()         // input list
                                  .RemainingRets()         // output list
                                  .Attr<int64_t>("num_gemms")
                                  .Attr<int64_t>("scaling_mode")
                                  .Attr<int64_t>("has_bias")
                                  .Attr<int64_t>("workspace_size"),
                              FFI_CudaGraph_Traits);

Error_Type GroupedAddFFI(cudaStream_t stream, Variadic_Buffer_Type input_list,
                         Variadic_Result_Type output_list, int64_t num_pairs) {
  for (size_t i = 0; i < static_cast<size_t>(num_pairs); i++) {
    auto A_i_get = input_list.get<Buffer_Type>(i);
    auto B_i_get = input_list.get<Buffer_Type>(num_pairs + i);
    auto C_i_get = output_list.get<Buffer_Type>(i);
    Buffer_Type A_i = A_i_get.value();
    Buffer_Type B_i = B_i_get.value();
    Result_Type C_i = C_i_get.value();
    auto A_ptr = reinterpret_cast<float *>(A_i.untyped_data());
    auto B_ptr = reinterpret_cast<float *>(B_i.untyped_data());
    auto C_ptr = reinterpret_cast<float *>(C_i->untyped_data());
    auto A_shape = A_i.dimensions();
    auto B_shape = B_i.dimensions();
    auto C_shape = C_i->dimensions();
    printf("Pair %ld: A shape ", i);
    for (size_t j = 0; j < A_shape.size(); j++) printf("%ld ", A_shape[j]);
    printf("; B shape ");
    for (size_t j = 0; j < B_shape.size(); j++) printf("%ld ", B_shape[j]);
    printf("; C shape ");
    for (size_t j = 0; j < C_shape.size(); j++) printf("%ld ", C_shape[j]);
    printf("\n");
    size_t A_size = product(A_shape);
    size_t B_size = product(B_shape);
    size_t C_size = product(C_shape);
    float *A_ptr_host = (float *) malloc(A_size * sizeof(float));
    float *B_ptr_host = (float *) malloc(B_size * sizeof(float));
    float *C_ptr_host = (float *) malloc(C_size * sizeof(float));
    cudaMemcpyAsync(A_ptr_host, A_ptr, A_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(B_ptr_host, B_ptr, B_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(C_ptr_host, C_ptr, C_size * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    for (size_t j = 0; j < A_size; j++)
      C_ptr_host[j] = A_ptr_host[j] + B_ptr_host[j];
    cudaMemcpyAsync(C_ptr, C_ptr_host, C_size * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    free(A_ptr_host);
    free(B_ptr_host);
    free(C_ptr_host);
  }
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedAddHandler, GroupedAddFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .RemainingArgs()         // input list
                                  .RemainingRets()         // output list
                                  .Attr<int64_t>("num_pairs"),
                              FFI_CudaGraph_Traits);


}  // namespace jax
}  // namespace transformer_engine

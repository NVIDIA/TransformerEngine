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

    if (scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING ||
        scaling_mode == JAXX_Scaling_Mode::CURRENT_TENSOR_SCALING) {
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

}  // namespace jax
}  // namespace transformer_engine

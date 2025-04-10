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
                           int32_t *dim_list_ptr, const int64_t &scaling_mode,
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
  // Jax uses row-major layout, on entering this function, each input matrix pair:
  //   A: row-major with size [m, k],
  //   B: row-major with size [n, k], needs transpose,
  // on exiting this function, JAX expect:
  //   C: row-major with size [m, n].
  // cuBLAS uses column-major layout, in this view, each input matrix pair:
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

    if (scaling_mode == NVTE_NO_SCALING || scaling_mode == NVTE_DELAYED_TENSOR_SCALING) {
      auto lhs_i = TensorWrapper(static_cast<void *>(lhs_ptr), lhs_shape, lhs_dtype, nullptr,
                                 nullptr, reinterpret_cast<float *>(lhs_sinv_ptr));
      auto rhs_i = TensorWrapper(static_cast<void *>(rhs_ptr), rhs_shape, rhs_dtype, nullptr,
                                 nullptr, reinterpret_cast<float *>(rhs_sinv_ptr));
      lhs_wrapper_list.push_back(std::move(lhs_i));
      rhs_wrapper_list.push_back(std::move(rhs_i));
    } else if (scaling_mode == NVTE_MXFP8_1D_SCALING) {
      NVTE_CHECK(k % MXFP8_BLOCK_SIZE == 0, "MXFP8 K-dim being divisble by %d (got %d)",
                 MXFP8_BLOCK_SIZE, k);
      size_t sinv_k = k / MXFP8_BLOCK_SIZE;
      lhs_sinv_shape[0] = m;
      lhs_sinv_shape[1] = sinv_k;
      rhs_sinv_shape[0] = n;
      rhs_sinv_shape[1] = sinv_k;

      // Note: the scale_inv array should have been swizzled in Python before lowering
      TensorWrapper lhs_i(NVTE_MXFP8_1D_SCALING);
      TensorWrapper rhs_i(NVTE_MXFP8_1D_SCALING);
      lhs_i.set_rowwise_data(static_cast<void *>(lhs_ptr), lhs_dtype, lhs_shape);
      rhs_i.set_rowwise_data(static_cast<void *>(rhs_ptr), rhs_dtype, rhs_shape);
      lhs_i.set_rowwise_scale_inv(static_cast<void *>(lhs_sinv_ptr), DType::kFloat8E8M0,
                                  lhs_sinv_shape);
      rhs_i.set_rowwise_scale_inv(static_cast<void *>(rhs_sinv_ptr), DType::kFloat8E8M0,
                                  rhs_sinv_shape);

      lhs_wrapper_list.push_back(std::move(lhs_i));
      rhs_wrapper_list.push_back(std::move(rhs_i));
    } else {
      NVTE_ERROR("Unsupported scaling mode: ", scaling_mode);
    }

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
                          Result_Type workspace_flatten, int64_t num_gemms, int64_t scaling_mode) {
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
                                  .Attr<int64_t>("scaling_mode"),
                              FFI_CudaGraph_Traits);

void GemmImpl(cudaStream_t stream, void *lhs, const std::vector<size_t> &lhs_shape,
              float *lhs_scale_inv, bool lhs_trans, void *rhs, const std::vector<size_t> &rhs_shape,
              float *rhs_scale_inv, bool rhs_trans, DType operand_dtype, void *bias,
              DType bias_dtype, void *out, float *out_amax, float *out_scale, DType out_dtype,
              void *pre_gelu_out, void *workspace, size_t workspace_size, bool fuse_gelu,
              bool fuse_bias, bool grad, bool accumulate, bool use_split_accumulator) {
  auto lhs_ = TensorWrapper(lhs, lhs_shape, operand_dtype, nullptr, nullptr, lhs_scale_inv);
  auto rhs_ = TensorWrapper(rhs, rhs_shape, operand_dtype, nullptr, nullptr, rhs_scale_inv);

  std::vector<size_t> out_shape(2, 0);
  out_shape[0] = (lhs_trans) ? lhs_shape[1] : lhs_shape[0];
  out_shape[1] = (rhs_trans) ? rhs_shape[0] : rhs_shape[1];
  auto out_ = TensorWrapper(out, out_shape, out_dtype, out_amax, out_scale, nullptr);

  void *bias_ptr = (fuse_bias) ? bias : nullptr;
  std::vector<size_t> bias_shape =
      (fuse_bias) ? std::vector<size_t>{out_shape[1]} : std::vector<size_t>{0};
  auto bias_ = TensorWrapper(bias_ptr, bias_shape, bias_dtype);

  void *pre_gelu_ptr = (fuse_gelu) ? pre_gelu_out : nullptr;
  std::vector<size_t> pre_gelu_shape = (fuse_gelu) ? out_shape : std::vector<size_t>{0};
  auto pre_gelu_out_ = TensorWrapper(pre_gelu_ptr, pre_gelu_shape, bias_dtype);
  auto workspace_ = TensorWrapper(workspace, std::vector<size_t>{workspace_size}, DType::kByte);

  // cuBLAS is column-major, so we swap LHS and RHS in the arguments
  auto num_math_sm = cuda::sm_count() - getenv<int>("NVTE_EXT_MARGIN_SM", 0);
  nvte_cublas_gemm(rhs_.data(), lhs_.data(), out_.data(), bias_.data(), pre_gelu_out_.data(),
                   (rhs_trans) ? CUBLAS_OP_T : CUBLAS_OP_N, (lhs_trans) ? CUBLAS_OP_T : CUBLAS_OP_N,
                   grad, workspace_.data(), accumulate, use_split_accumulator, num_math_sm, stream);
}

Error_Type GemmFFI(cudaStream_t stream, Buffer_Type lhs, Buffer_Type lhs_scale_inv, Buffer_Type rhs,
                   Buffer_Type rhs_scale_inv, Buffer_Type bias, Buffer_Type gelu_input,
                   Buffer_Type out_amax, Buffer_Type out_scale, Result_Type out,
                   Result_Type out_amax_updated, Result_Type out_scale_updated,
                   Result_Type pre_gelu_out, Result_Type bias_grad, Result_Type workspace,
                   bool lhs_trans, bool rhs_trans, bool fuse_gelu, bool fuse_bias, bool grad,
                   bool accumulate, bool use_split_accumulator) {
  // Inputs
  auto lhs_ptr = lhs.untyped_data();
  auto lhs_scale_inv_ptr = reinterpret_cast<float *>(lhs_scale_inv.untyped_data());
  auto rhs_ptr = rhs.untyped_data();
  auto rhs_scale_inv_ptr = reinterpret_cast<float *>(rhs_scale_inv.untyped_data());
  auto operand_dtype = convert_ffi_datatype_to_te_dtype(lhs.element_type());
  auto bias_ptr = bias.untyped_data();
  auto bias_dtype = convert_ffi_datatype_to_te_dtype(bias.element_type());
  auto gelu_input_ptr = gelu_input.untyped_data();
  auto out_amax_ptr = reinterpret_cast<float *>(out_amax.untyped_data());
  auto out_scale_ptr = reinterpret_cast<float *>(out_scale.untyped_data());

  // Outputs
  auto out_ptr = out->untyped_data();
  auto out_amax_updated_ptr = reinterpret_cast<float *>(out_amax_updated->untyped_data());
  auto out_scale_updated_ptr = reinterpret_cast<float *>(out_scale_updated->untyped_data());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(out->element_type());
  auto pre_gelu_out_ptr = pre_gelu_out->untyped_data();
  auto bias_grad_ptr = bias_grad->untyped_data();
  auto workspace_ptr = workspace->untyped_data();
  auto workspace_size = workspace->dimensions().back();

  // Operand aliasing
  NVTE_CHECK(bias_ptr == bias_grad_ptr, "bias not bound to bias_grad in TE/JAX GEMM");
  NVTE_CHECK(gelu_input_ptr == pre_gelu_out_ptr,
             "gelu_input not bound to pre_gelu_out in TE/JAX GEMM");
  NVTE_CHECK(out_amax_ptr == out_amax_updated_ptr,
             "out_amax not bound to out_amax_updated in TE/JAX GEMM");
  NVTE_CHECK(out_scale_ptr == out_scale_updated_ptr,
             "out_scale not bound to out_scale_updated in TE/JAX GEMM");

  // GEMM sizing
  std::vector<size_t> lhs_shape(lhs.dimensions().begin(), lhs.dimensions().end());
  std::vector<size_t> rhs_shape(rhs.dimensions().begin(), rhs.dimensions().end());

  // Swap A and B argument locations to match what the TE/common kernel expects
  GemmImpl(stream, lhs_ptr, lhs_shape, lhs_scale_inv_ptr, lhs_trans, rhs_ptr, rhs_shape,
           rhs_scale_inv_ptr, rhs_trans, operand_dtype, bias_ptr, bias_dtype, out_ptr, out_amax_ptr,
           out_scale_ptr, out_dtype, pre_gelu_out_ptr, workspace_ptr, workspace_size, fuse_gelu,
           fuse_bias, grad, accumulate, use_split_accumulator);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GemmHandler, GemmFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // lhs
                                  .Arg<Buffer_Type>()      // lhs_scale_inv
                                  .Arg<Buffer_Type>()      // rhs
                                  .Arg<Buffer_Type>()      // rhs_scale_inv
                                  .Arg<Buffer_Type>()      // bias
                                  .Arg<Buffer_Type>()      // gelu_input
                                  .Arg<Buffer_Type>()      // out_amax
                                  .Arg<Buffer_Type>()      // out_scale
                                  .Ret<Buffer_Type>()      // out
                                  .Ret<Buffer_Type>()      // out_amax_updated
                                  .Ret<Buffer_Type>()      // out_scale_updated
                                  .Ret<Buffer_Type>()      // pre_gelu_out
                                  .Ret<Buffer_Type>()      // bias_grad
                                  .Ret<Buffer_Type>()      // workspace
                                  .Attr<bool>("lhs_trans")
                                  .Attr<bool>("rhs_trans")
                                  .Attr<bool>("fuse_gelu")
                                  .Attr<bool>("fuse_bias")
                                  .Attr<bool>("grad")
                                  .Attr<bool>("accumulate")
                                  .Attr<bool>("use_split_accumulator"),
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine

/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/gemm.h"

#include "common/util/cuda_runtime.h"
#include "common/util/system.h"
#include "extensions.h"

namespace transformer_engine {

namespace jax {

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

void Gemm(cudaStream_t stream, void **buffers, const char *opaque, size_t opaque_len) {
  // Inputs
  auto *lhs = buffers[0];
  auto *lhs_scale_inv = reinterpret_cast<float *>(buffers[1]);
  auto *rhs = buffers[2];
  auto *rhs_scale_inv = reinterpret_cast<float *>(buffers[3]);
  auto *bias = buffers[4];
  auto *gelu_input = buffers[5];
  auto *out_amax = reinterpret_cast<float *>(buffers[6]);
  auto *out_scale = reinterpret_cast<float *>(buffers[7]);

  // Outputs
  auto *out = buffers[8];
  auto *out_amax_updated = reinterpret_cast<float *>(buffers[9]);
  auto *out_scale_updated = reinterpret_cast<float *>(buffers[10]);
  auto *pre_gelu_out = buffers[11];
  auto *bias_grad = buffers[12];
  // buffers[13] is the extra output for comm+GEMM overlap, not used here
  auto *workspace = buffers[14];

  // Operand aliasing
  NVTE_CHECK(bias == bias_grad, "bias not bound to bias_grad in TE/JAX GEMM");
  NVTE_CHECK(gelu_input == pre_gelu_out, "gelu_input not bound to pre_gelu_out in TE/JAX GEMM");
  NVTE_CHECK(out_amax == out_amax_updated, "out_amax not bound to out_amax_updated in TE/JAX GEMM");
  NVTE_CHECK(out_scale == out_scale_updated,
             "out_scale not bound to out_scale_updated in TE/JAX GEMM");

  // GEMM sizing
  const auto &desc = *UnpackOpaque<CustomCallGemmDescriptor>(opaque, opaque_len);
  std::vector<size_t> lhs_shape = {(desc.lhs_trans) ? desc.k : desc.m,
                                   (desc.lhs_trans) ? desc.m : desc.k};
  std::vector<size_t> rhs_shape = {(desc.rhs_trans) ? desc.n : desc.k,
                                   (desc.rhs_trans) ? desc.k : desc.n};

  GemmImpl(stream, lhs, lhs_shape, lhs_scale_inv, desc.lhs_trans, rhs, rhs_shape, rhs_scale_inv,
           desc.rhs_trans, desc.operand_dtype, bias, desc.bias_dtype, out, out_amax, out_scale,
           desc.out_dtype, pre_gelu_out, workspace, desc.workspace_size, desc.fuse_gelu,
           desc.fuse_bias, desc.grad, desc.accumulate, desc.use_split_accumulator);
}

Error_Type GemmFFI(cudaStream_t stream, Buffer_Type lhs, Buffer_Type lhs_scale_inv, Buffer_Type rhs,
                   Buffer_Type rhs_scale_inv, Buffer_Type bias, Buffer_Type gelu_input,
                   Buffer_Type out_amax, Buffer_Type out_scale, Result_Type out,
                   Result_Type out_amax_updated, Result_Type out_scale_updated,
                   Result_Type pre_gelu_out, Result_Type bias_grad, Result_Type dummy_out,
                   Result_Type workspace, bool lhs_trans, bool rhs_trans, bool fuse_gelu,
                   bool fuse_bias, bool grad, bool accumulate, bool use_split_accumulator) {
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
  // dummy_out is the extra output for comm+GEMM overlap, not used here
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
                                  .Ret<Buffer_Type>()      // dummy_out
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

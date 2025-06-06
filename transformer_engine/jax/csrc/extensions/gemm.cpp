/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "transformer_engine/gemm.h"
#include "transformer_engine/swizzle.h"

#include <memory>
#include <string_view>
#include <tuple>

#include "../extensions.h"
#include "common/util/cuda_runtime.h"
#include "common/util/system.h"
#include "common/util/string.h"
#include "extensions.h"
#include "transformer_engine/multi_stream.h"
#include "xla/ffi/api/c_api.h"

namespace transformer_engine {
namespace jax {

std::tuple<TensorWrapper, std::vector<size_t>> xla_buffer_to_nvte_gemm_operand(
    cudaStream_t stream, Buffer_Type buffer, Buffer_Type scale_inv, Result_Type swizzled_scale_inv,
    JAXX_Scaling_Mode scaling_mode, size_t axis_boundary, bool rowwise) {
  // Set tensor data with collapsed 2D shape
  auto buffer_dims = buffer.dimensions();
  std::vector<size_t> input_shape = {product(buffer_dims, 0, axis_boundary),
                                     product(buffer_dims, axis_boundary, buffer_dims.size())};
  DType input_dtype = convert_ffi_datatype_to_te_dtype(buffer.element_type());
  TensorWrapper input(get_nvte_scaling_mode(scaling_mode));

  // Set scaling factor for quantized tensors
  if (scaling_mode != JAXX_Scaling_Mode::NO_SCALING) {
    NVTE_CHECK(input.element_size() == 1, "Quantized GEMM requires 8-bit operands.");
    NVTE_CHECK(scale_inv.element_count() > 0, "Missing inverse scaling factor for quantized GEMM.");

    std::vector<size_t> scale_shape(scale_inv.dimensions().begin(), scale_inv.dimensions().end());
    DType scale_dtype = convert_ffi_datatype_to_te_dtype(scale_inv.element_type());
    if (rowwise) {
      input.set_rowwise_data(buffer.untyped_data(), input_dtype, input_shape);
      input.set_rowwise_scale_inv(scale_inv.untyped_data(), scale_dtype, scale_shape);
    } else {
      input.set_columnwise_data(buffer.untyped_data(), input_dtype, input_shape);
      input.set_columnwise_scale_inv(scale_inv.untyped_data(), scale_dtype, scale_shape);
    }

    // Swizzle scaling factors for MXFP8
    if (scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING) {
      // Get the swizzle buffer
      NVTE_CHECK(swizzled_scale_inv->element_count() > 0,
                 "Missing swizzled inverse scale buffer in the JAX primitive.");
      NVTE_CHECK(scale_inv.element_size() == 1 && swizzled_scale_inv->element_size() == 1,
                 "Inverse scale factors need to have an 8-bit data type.");

      // Create tensor to hold swizzled scale factor
      TensorWrapper output(NVTE_MXFP8_1D_SCALING);
      if (rowwise) {
        output.set_rowwise_data(buffer.untyped_data(), input_dtype, input_shape);
        output.set_rowwise_scale_inv(swizzled_scale_inv->untyped_data(), DType::kFloat8E8M0,
                                     scale_shape);
      } else {
        output.set_columnwise_data(buffer.untyped_data(), input_dtype, input_shape);
        output.set_columnwise_scale_inv(swizzled_scale_inv->untyped_data(), DType::kFloat8E8M0,
                                        scale_shape);
      }

      // Launch swizzle kernel
      nvte_swizzle_scaling_factors(input.data(), output.data(), stream);

      // Set swizzled scales into the input tensor
      if (rowwise) {
        input.set_rowwise_scale_inv(swizzled_scale_inv->untyped_data(), DType::kFloat8E8M0,
                                    scale_shape);
      } else {
        input.set_columnwise_scale_inv(swizzled_scale_inv->untyped_data(), DType::kFloat8E8M0,
                                       scale_shape);
      }
    }
  } else{
    // Non-quantized tensors always get set into the rowwise data
    input.set_rowwise_data(buffer.untyped_data(), input_dtype, input_shape);
  }

  return std::make_tuple(std::move(input), input_shape);
}

Error_Type GemmFFI(
    cudaStream_t stream, Buffer_Type lhs, Buffer_Type lhs_scale_inv, Buffer_Type rhs,
    Buffer_Type rhs_scale_inv, Buffer_Type bias, Buffer_Type gelu_input, Result_Type output,
    Result_Type bias_grad, Result_Type pre_gelu_out, Result_Type lhs_swizzle,
    Result_Type rhs_swizzle,Result_Type workspace, int64_t lhs_axis_boundary,
    int64_t rhs_axis_boundary, JAXX_Scaling_Mode scaling_mode, bool lhs_scaled_colwise,
    bool rhs_scaled_colwise, bool lhs_transposed, bool rhs_transposed, bool fuse_bias,
    bool fuse_gelu, bool grad, bool accumulate, bool use_split_accumulator) {
  // Operands (includes swizzling MXFP8 scaling factors if needed)
  auto [lhs_, lhs_shape] = xla_buffer_to_nvte_gemm_operand(
      stream, lhs, lhs_scale_inv, lhs_swizzle, scaling_mode, lhs_axis_boundary,
      !lhs_scaled_colwise);
  auto [rhs_, rhs_shape] = xla_buffer_to_nvte_gemm_operand(
      stream, rhs, rhs_scale_inv, lhs_swizzle, scaling_mode, rhs_axis_boundary,
      !rhs_scaled_colwise);

  // Output tensor
  std::vector<size_t> out_shape = {(lhs_transposed) ? lhs_shape[1] : lhs_shape[0],
                                   (rhs_transposed) ? rhs_shape[0] : rhs_shape[1]};
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output->element_type());
  auto out_ = TensorWrapper(output->untyped_data(), out_shape, out_dtype);
  NVTE_CHECK(out_.numel() == output->element_count(), "cuBLAS GEMM output buffer size is incorrect, "
             "expected ", out_.numel(), " elements ", to_string_like(out_shape), " but got ",
             output->element_count(), " elements ",
             to_string_like(std::vector<size_t>(output->dimensions().begin(),
                                                output->dimensions().end())));

  // Bias input to forward pass or bias gradient output from backward pass
  void* bias_ptr = nullptr;
  std::vector<size_t> bias_shape = {0};
  DType bias_dtype = out_dtype;
  if (fuse_bias) {
    if (!grad) {
      NVTE_CHECK(bias_grad->untyped_data() == bias.untyped_data(),
                 "Missing operand-output aliasing in GemmPrimitive: bias <-> bias_grad");
    }
    bias_ptr = bias_grad->untyped_data();
    bias_shape.at(0) = bias_grad->dimensions().front();
    bias_dtype = convert_ffi_datatype_to_te_dtype(bias_grad->element_type());
  }
  auto bias_ = TensorWrapper(bias_ptr, bias_shape, bias_dtype);

  // Pre-GeLU output from forward pass or input to backward pass
  void* pre_gelu_ptr = nullptr;
  std::vector<size_t> pre_gelu_shape = {0};
  DType pre_gelu_dtype = out_dtype;
  if (gelu_input.element_count() > 0) {
    if (grad) {
      NVTE_CHECK(pre_gelu_out->untyped_data() == gelu_input.untyped_data(),
                 "Missing operand-output aliasing in GemmPrimitive: gelu_input <-> pre_gelu_out");
    }
    pre_gelu_ptr = pre_gelu_out->untyped_data();
    pre_gelu_shape = {product(pre_gelu_out->dimensions(), 0, pre_gelu_out->dimensions().size() - 1),
                      static_cast<size_t>(pre_gelu_out->dimensions().back())};
    pre_gelu_dtype = convert_ffi_datatype_to_te_dtype(pre_gelu_out->element_type());
  }
  auto pre_gelu_ = TensorWrapper(pre_gelu_ptr, pre_gelu_shape, pre_gelu_dtype);

  // cuBLAS workspace
  std::vector<size_t> workspace_shape = {static_cast<size_t>(workspace->element_count())};
  auto workspace_ = TensorWrapper(workspace->untyped_data(), workspace_shape, DType::kByte);

  // Launch TE/common kernel with swapped LHS/RHS for cuBLAS column-major order
  auto num_math_sm = cuda::sm_count() - getenv<int>("NVTE_EXT_MARGIN_SM", 0);
  nvte_cublas_gemm(rhs_.data(), lhs_.data(), out_.data(), bias_.data(), pre_gelu_.data(),
                   rhs_transposed, lhs_transposed, grad, workspace_.data(), accumulate,
                   use_split_accumulator, num_math_sm, stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GemmHandler, GemmFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()  // lhs
                                  .Arg<Buffer_Type>()  // lhs_scale_inv
                                  .Arg<Buffer_Type>()  // rhs
                                  .Arg<Buffer_Type>()  // rhs_scale_inv
                                  .Arg<Buffer_Type>()  // bias
                                  .Arg<Buffer_Type>()  // gelu_input
                                  .Ret<Buffer_Type>()  // output
                                  .Ret<Buffer_Type>()  // bias_grad
                                  .Ret<Buffer_Type>()  // pre_gelu_out
                                  .Ret<Buffer_Type>()  // lhs_swizzled
                                  .Ret<Buffer_Type>()  // rhs_swizzled
                                  .Ret<Buffer_Type>()  // workspace
                                  .Attr<int64_t>("lhs_axis_boundary")
                                  .Attr<int64_t>("rhs_axis_boundary")
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<bool>("lhs_scaled_colwise")
                                  .Attr<bool>("rhs_scaled_colwise")
                                  .Attr<bool>("lhs_transposed")
                                  .Attr<bool>("rhs_transposed")
                                  .Attr<bool>("fuse_bias")
                                  .Attr<bool>("fuse_gelu")
                                  .Attr<bool>("grad")
                                  .Attr<bool>("accumulate")
                                  .Attr<bool>("use_split_accumulator"),
                              FFI_CudaGraph_Traits);

Error_Type GroupedGemmFFI(cudaStream_t stream, Variadic_Buffer_Type input_list,
                          Variadic_Result_Type output_list, int64_t num_gemms,
                          JAXX_Scaling_Mode scaling_mode, int64_t has_bias) {
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

  if (num_gemms <= 0) {
    return ffi_with_cuda_error_check();
  }
  size_t expected_input_size = has_bias ? 5 * num_gemms : 4 * num_gemms;
  size_t expected_output_size = num_gemms + 1;
  size_t actual_input_size = input_list.size();
  size_t actual_output_size = output_list.size();
  NVTE_CHECK(actual_input_size == expected_input_size, "Expected %zu input tensors, got %zu",
             expected_input_size, actual_input_size);
  NVTE_CHECK(actual_output_size == expected_output_size, "Expected %zu output tensors, got %zu",
             expected_output_size, actual_output_size);

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

  int lhs_list_offset = 0;
  int rhs_list_offset = num_gemms;
  int lhs_sinv_list_offset = 2 * num_gemms;
  int rhs_sinv_list_offset = 3 * num_gemms;
  int bias_list_offset = 4 * num_gemms;
  int out_list_offset = 0;
  for (int i = 0; i < num_gemms; i++) {
    Buffer_Type lhs_i = input_list.get<Buffer_Type>(lhs_list_offset + i).value();
    Buffer_Type rhs_i = input_list.get<Buffer_Type>(rhs_list_offset + i).value();
    Buffer_Type lhs_sinv_i = input_list.get<Buffer_Type>(lhs_sinv_list_offset + i).value();
    Buffer_Type rhs_sinv_i = input_list.get<Buffer_Type>(rhs_sinv_list_offset + i).value();
    Result_Type out_i = output_list.get<Buffer_Type>(out_list_offset + i).value();

    DType lhs_dtype = convert_ffi_datatype_to_te_dtype(lhs_i.element_type());
    DType rhs_dtype = convert_ffi_datatype_to_te_dtype(rhs_i.element_type());
    DType out_dtype = convert_ffi_datatype_to_te_dtype(out_i->element_type());

    void *lhs_ptr = lhs_i.untyped_data();
    void *rhs_ptr = rhs_i.untyped_data();
    void *lhs_sinv_ptr = lhs_sinv_i.untyped_data();
    void *rhs_sinv_ptr = rhs_sinv_i.untyped_data();
    void *out_ptr = out_i->untyped_data();

    // Placeholder for bias since it can be empty
    DType bias_dtype = DType::kFloat32;
    void *bias_ptr = nullptr;

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

    if (scaling_mode == JAXX_Scaling_Mode::NO_SCALING ||
        scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING ||
        scaling_mode == JAXX_Scaling_Mode::CURRENT_TENSOR_SCALING) {
      float *amax_dptr = nullptr;
      float *scale_dptr = nullptr;
      auto lhs_i_ = TensorWrapper(lhs_ptr, lhs_shape, lhs_dtype, amax_dptr, scale_dptr,
                                  reinterpret_cast<float *>(lhs_sinv_ptr));
      auto rhs_i_ = TensorWrapper(rhs_ptr, rhs_shape, rhs_dtype, amax_dptr, scale_dptr,
                                  reinterpret_cast<float *>(rhs_sinv_ptr));
      lhs_wrapper_list.push_back(std::move(lhs_i_));
      rhs_wrapper_list.push_back(std::move(rhs_i_));
    } else if (scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING) {
      // Note: the scale_inv array should have been swizzled in Python before lowering
      auto lhs_sinv_shape_ = lhs_sinv_i.dimensions();
      auto rhs_sinv_shape_ = rhs_sinv_i.dimensions();
      for (int i = 0; i < 2; i++) {
        lhs_sinv_shape[i] = lhs_sinv_shape_[i];
        rhs_sinv_shape[i] = rhs_sinv_shape_[i];
      }

      NVTEScalingMode nvte_scaling_mode = get_nvte_scaling_mode(scaling_mode);
      TensorWrapper lhs_i_(nvte_scaling_mode);
      TensorWrapper rhs_i_(nvte_scaling_mode);
      lhs_i_.set_rowwise_data(lhs_ptr, lhs_dtype, lhs_shape);
      rhs_i_.set_rowwise_data(rhs_ptr, rhs_dtype, rhs_shape);
      lhs_i_.set_rowwise_scale_inv(lhs_sinv_ptr, DType::kFloat8E8M0, lhs_sinv_shape);
      rhs_i_.set_rowwise_scale_inv(rhs_sinv_ptr, DType::kFloat8E8M0, rhs_sinv_shape);

      lhs_wrapper_list.push_back(std::move(lhs_i_));
      rhs_wrapper_list.push_back(std::move(rhs_i_));
    } else {
      NVTE_ERROR("Unsupported scaling mode: ", static_cast<int>(scaling_mode));
    }

    auto out_i_ = TensorWrapper(out_ptr, out_shape, out_dtype);
    void *pre_gelu_ptr = nullptr;
    auto bias_shape = std::vector<size_t>{0};
    auto pre_gelu_shape = std::vector<size_t>{0};
    if (has_bias) {
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
  auto num_streams = nvte_get_num_compute_streams();
  size_t workspace_size = workspace->dimensions()[0] / num_streams;
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

XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmHandler, GroupedGemmFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .RemainingArgs()         // input list
                                  .RemainingRets()         // output list
                                  .Attr<int64_t>("num_gemms")
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<int64_t>("has_bias"),
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine

/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "transformer_engine/gemm.h"

#include <memory>
#include <string_view>
#include <tuple>

#include "../extensions.h"
#include "common/util/cuda_runtime.h"
#include "common/util/string.h"
#include "common/util/system.h"
#include "transformer_engine/swizzle.h"
#include "xla/ffi/api/c_api.h"

#define MXFP8_BLOCK_SIZE 32

namespace transformer_engine {
namespace jax {

std::tuple<TensorWrapper, std::vector<size_t>> xla_buffer_to_nvte_gemm_operand(
    cudaStream_t stream, Buffer_Type buffer, Buffer_Type scale_inv, Result_Type swizzled_scale_inv,
    JAXX_Scaling_Mode scaling_mode, size_t axis_boundary, bool rowwise) {
  // Set tensor data with collapsed 2D shape
  auto buffer_dims = buffer.dimensions();
  std::vector<size_t> input_shape = {product(buffer_dims, 0, axis_boundary),
                                     product(buffer_dims, axis_boundary, buffer_dims.size())};
  auto input_dtype = convert_ffi_datatype_to_te_dtype(buffer.element_type());
  TensorWrapper input(get_nvte_scaling_mode(scaling_mode));

  if (rowwise) {
    input.set_rowwise_data(buffer.untyped_data(), input_dtype, input_shape);
  } else {
    input.set_columnwise_data(buffer.untyped_data(), input_dtype, input_shape);
  }

  // Set scaling factor for quantized tensors
  if (scaling_mode != JAXX_Scaling_Mode::NO_SCALING) {
    NVTE_CHECK(typeToSize(input_dtype) == 1, "Quantized GEMM requires 8-bit operands.");
    NVTE_CHECK(scale_inv.element_count() > 0, "Missing inverse scaling factor for quantized GEMM.");

    std::vector<size_t> scale_shape(scale_inv.dimensions().begin(), scale_inv.dimensions().end());
    auto scale_dtype = (scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING)
                           ? DType::kFloat8E8M0
                           : convert_ffi_datatype_to_te_dtype(scale_inv.element_type());
    if (rowwise) {
      input.set_rowwise_scale_inv(scale_inv.untyped_data(), scale_dtype, scale_shape);
    } else {
      input.set_columnwise_scale_inv(scale_inv.untyped_data(), scale_dtype, scale_shape);
    }

    // Swizzle scaling factors for MXFP8
    if (is_block_scaling(scaling_mode)) {
      // Get the swizzle buffer
      NVTE_CHECK(swizzled_scale_inv->element_count() > 0,
                 "Missing swizzled inverse scale buffer in the JAX primitive.");
      auto scale_inv_dtype = convert_ffi_datatype_to_te_dtype(scale_inv.element_type());
      auto swizzled_scale_inv_dtype =
          convert_ffi_datatype_to_te_dtype(swizzled_scale_inv->element_type());
      NVTE_CHECK(typeToSize(scale_inv_dtype) == 1 && typeToSize(swizzled_scale_inv_dtype) == 1,
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
  }

  return std::make_tuple(std::move(input), input_shape);
}

Error_Type GemmFFI(cudaStream_t stream, Buffer_Type lhs, Buffer_Type lhs_scale_inv, Buffer_Type rhs,
                   Buffer_Type rhs_scale_inv, Buffer_Type bias, Buffer_Type gelu_input,
                   Result_Type output, Result_Type bias_grad, Result_Type pre_gelu_out,
                   Result_Type lhs_swizzle, Result_Type rhs_swizzle, Result_Type workspace,
                   JAXX_Scaling_Mode scaling_mode, int64_t lhs_axis_boundary,
                   int64_t rhs_axis_boundary, bool lhs_transposed, bool rhs_transposed,
                   bool fuse_bias, bool fuse_gelu, bool grad) {
  // Operands (this includes swizzling MXFP8 scaling factors)
  // NOTE: TensorWrapper operands are always rowwise for full-precision GEMM, or FP8 GEMM when
  //       device supports non-TN layouts (compute capability >= 10.0, excluding 12.x)
  bool always_rowwise = (scaling_mode == JAXX_Scaling_Mode::NO_SCALING ||
                         (is_tensor_scaling(scaling_mode) && nvte_is_non_tn_fp8_gemm_supported()));
  bool make_lhs_rowwise = (always_rowwise) ? true : !lhs_transposed;
  bool make_rhs_rowwise = (always_rowwise) ? true : rhs_transposed;
  auto [lhs_, lhs_shape] = xla_buffer_to_nvte_gemm_operand(
      stream, lhs, lhs_scale_inv, lhs_swizzle, scaling_mode, lhs_axis_boundary, make_lhs_rowwise);
  auto [rhs_, rhs_shape] = xla_buffer_to_nvte_gemm_operand(
      stream, rhs, rhs_scale_inv, lhs_swizzle, scaling_mode, rhs_axis_boundary, make_rhs_rowwise);

  // Output tensor
  std::vector<size_t> out_shape = {(lhs_transposed) ? lhs_shape[1] : lhs_shape[0],
                                   (rhs_transposed) ? rhs_shape[0] : rhs_shape[1]};
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output->element_type());
  auto out_ = TensorWrapper(output->untyped_data(), out_shape, out_dtype);
  NVTE_CHECK(out_.numel() == output->element_count(),
             "cuBLAS GEMM output buffer size is incorrect, "
             "expected ",
             out_.numel(), " elements ", to_string_like(out_shape), " but got ",
             output->element_count(), " elements ", to_string_like(output->dimensions()));

  // Bias input to forward pass or bias gradient output from backward pass
  void *bias_ptr = nullptr;
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
  void *pre_gelu_ptr = nullptr;
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
                   rhs_transposed, lhs_transposed, grad, workspace_.data(), false, false,
                   num_math_sm, stream);

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
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // bias_grad
                                  .Ret<Buffer_Type>()      // pre_gelu_out
                                  .Ret<Buffer_Type>()      // lhs_swizzled
                                  .Ret<Buffer_Type>()      // rhs_swizzled
                                  .Ret<Buffer_Type>()      // workspace
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<int64_t>("lhs_axis_boundary")
                                  .Attr<int64_t>("rhs_axis_boundary")
                                  .Attr<bool>("lhs_transposed")
                                  .Attr<bool>("rhs_transposed")
                                  .Attr<bool>("fuse_bias")
                                  .Attr<bool>("fuse_gelu")
                                  .Attr<bool>("grad"),
                              FFI_CudaGraph_Traits);

Error_Type GroupedGemmFFI(cudaStream_t stream, Buffer_Type lhs_data, Buffer_Type lhs_sinv,
                          Buffer_Type rhs_data, Buffer_Type rhs_sinv, Buffer_Type bias,
                          Buffer_Type group_sizes, Buffer_Type group_offset, Result_Type output,
                          Result_Type workspace, size_t m, size_t n, size_t k, bool lhs_is_trans,
                          bool rhs_is_trans, JAXX_Scaling_Mode scaling_mode, bool has_bias,
                          bool is_grouped_dense_wgrad) {
  // Notes on matrix layouts and transpose:
  // Jax uses row-major data_layout, on entering this function, each input matrix pair:
  //   A: row-major [m, k] for N - [k, m] for T
  //   B: row-major [k, n] for N - [n, k] for T
  // on exiting this function, JAX expect:
  //   C: row-major with size [m, n].
  // cuBLAS uses column-major data_layout, in this view, each input matrix pair:
  //   A: column-major with size [k, m] for T - [m, k] for N
  //   B: column-major with size [n, k] for T - [k, n] for N
  //
  // If we call cuBLAS GEMM for A * B, the output will be:
  //   C: column-major with size [m, n] --> row-major with size [n, m].
  // To make the output compatible with JAX, we need to swap A and B in cuBLAS GEMM call.

  int num_streams = nvte_get_num_compute_streams();

  // Inputs
  auto lhs_ptr = reinterpret_cast<uint8_t *>(lhs_data.untyped_data());
  auto rhs_ptr = reinterpret_cast<uint8_t *>(rhs_data.untyped_data());
  auto lhs_sinv_ptr = reinterpret_cast<uint8_t *>(lhs_sinv.untyped_data());
  auto rhs_sinv_ptr = reinterpret_cast<uint8_t *>(rhs_sinv.untyped_data());
  auto lhs_dtype = convert_ffi_datatype_to_te_dtype(lhs_data.element_type());
  auto rhs_dtype = convert_ffi_datatype_to_te_dtype(rhs_data.element_type());
  auto lhs_sinv_dtype = convert_ffi_datatype_to_te_dtype(lhs_sinv.element_type());
  auto rhs_sinv_dtype = convert_ffi_datatype_to_te_dtype(rhs_sinv.element_type());
  auto bias_ptr = has_bias ? reinterpret_cast<uint8_t *>(bias.untyped_data()) : nullptr;
  auto bias_dtype = convert_ffi_datatype_to_te_dtype(bias.element_type());

  NVTE_CHECK(group_sizes.dimensions().size() == 1);
  size_t num_gemms = group_sizes.dimensions()[0];

  // Outputs
  auto out_ptr = reinterpret_cast<uint8_t *>(output->untyped_data());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output->element_type());
  // Here we clear the lower 8 bits of the buffer address to ensure the buffer is 256-aligned
  auto workspace_ptr =
      reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(workspace->untyped_data()) + 255) &
                                  ~static_cast<uintptr_t>(255));
  auto workspace_total_size = product(workspace->dimensions()) - 255;
  auto workspace_size = workspace_total_size / num_streams;

  size_t lhs_dtype_bytes = te_dtype_bytes(lhs_dtype);
  size_t rhs_dtype_bytes = te_dtype_bytes(rhs_dtype);
  size_t lhs_sinv_dtype_bytes = te_dtype_bytes(lhs_sinv_dtype);
  size_t rhs_sinv_dtype_bytes = te_dtype_bytes(rhs_sinv_dtype);
  size_t bias_dtype_bytes = te_dtype_bytes(bias_dtype);
  size_t out_dtype_bytes = te_dtype_bytes(out_dtype);

  NVTE_CHECK(lhs_dtype_bytes == rhs_dtype_bytes, "sizeof(lhs_dtype) != sizeof(rhs_dtype)");
  NVTE_CHECK(lhs_sinv_dtype_bytes == rhs_sinv_dtype_bytes,
             "sizeof(lhs_sinv_dtype) != sizeof(rhs_sinv_dtype)");

  size_t expected_lhs_size = m * k;
  size_t expected_rhs_size = is_grouped_dense_wgrad ? (k * n) : (num_gemms * k * n);
  size_t expected_out_size = is_grouped_dense_wgrad ? (num_gemms * m * n) : (m * n);
  size_t actual_lhs_size = product(lhs_data.dimensions());
  size_t actual_rhs_size = product(rhs_data.dimensions());
  size_t actual_out_size = product(output->dimensions());
  NVTE_CHECK(expected_lhs_size == actual_lhs_size, "Unexpected lhs size! Expect ",
             expected_lhs_size, ", got ", actual_lhs_size);
  if (!is_grouped_dense_wgrad) {
    NVTE_CHECK(expected_rhs_size == actual_rhs_size,
               "Unexpected rhs size! Expect num_gemms * n * k = ", num_gemms, " * ", n, " * ", k,
               " = ", expected_rhs_size, ", got ", actual_rhs_size);
    NVTE_CHECK(expected_out_size == actual_out_size, "Unexpected output size! Expect m * n = ", m,
               " * ", n, " = ", expected_out_size, ", got ", actual_out_size);
  } else {
    NVTE_CHECK(expected_rhs_size == actual_rhs_size, "Unexpected rhs size! Expect k * n = ", k,
               " * ", n, " = ", expected_rhs_size, ", got ", actual_rhs_size);
    NVTE_CHECK(expected_out_size == actual_out_size,
               "Unexpected output size! Expect num_gemms * m * n = ", num_gemms, " * ", m, " * ", n,
               " = ", expected_out_size, ", got ", actual_out_size);
  }

  size_t dim_list_bytes = sizeof(int32_t) * num_gemms;
  std::vector<int32_t> dim_list_host(num_gemms);
  auto dim_list_ptr = reinterpret_cast<int32_t *>(group_sizes.untyped_data());
  cudaMemcpyAsync(dim_list_host.data(), dim_list_ptr, dim_list_bytes, cudaMemcpyDeviceToHost,
                  stream);
  // Note: This may break cudaGraph.
  cudaStreamSynchronize(stream);
  size_t sum_group_sizes = std::accumulate(dim_list_host.begin(), dim_list_host.end(), 0);
  if (!is_grouped_dense_wgrad) {
    NVTE_CHECK(m == sum_group_sizes, "Unexpected group_sizes! M = ", m,
               ", got sum(group_sizes)=", sum_group_sizes);
  } else {
    NVTE_CHECK(k == sum_group_sizes, "Unexpected group_sizes! K = ", k,
               ", got sum(group_sizes)=", sum_group_sizes);
  }

  auto num_math_sm = cuda::sm_count() - getenv<int>("NVTE_EXT_MARGIN_SM", 0);
  bool grad = false;
  bool accumulate = false;
  bool use_split_accumulator = false;
  auto bias_shape = std::vector<size_t>{has_bias ? n : 0};
  const int arch = cuda::sm_arch();

  // It is weird that TE/Common GEMM only use colwise for MXFP8
  const bool is_fp8_gemm = is_fp8_dtype(lhs_dtype);
  const bool is_mxfp8_scaling = scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING;
  const bool rhs_use_colwise = is_mxfp8_scaling && !rhs_is_trans;
  const bool lhs_use_colwise = is_mxfp8_scaling && lhs_is_trans;

  if (arch < 100 && is_fp8_gemm) {
    NVTE_CHECK(!lhs_is_trans && rhs_is_trans,
               "For SM90 or older archs and FP8 input, only NT (row-major) GEMM is supported, ",
               "got lhs_is_trans=", lhs_is_trans, ", rhs_is_trans=", rhs_is_trans);
  }

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

  for (size_t i = 0; i < num_gemms; i++) {
    // Matrix data shapes
    size_t m_i = dim_list_host[i];
    auto lhs_shape = std::vector<size_t>{m_i, k};
    auto rhs_shape = std::vector<size_t>{rhs_is_trans ? n : k, rhs_is_trans ? k : n};
    auto out_shape = std::vector<size_t>{m_i, n};
    if (is_grouped_dense_wgrad) {
      size_t k_i = dim_list_host[i];
      lhs_shape[0] = lhs_is_trans ? k_i : m;
      lhs_shape[1] = lhs_is_trans ? m : k_i;
      rhs_shape[0] = rhs_is_trans ? n : k_i;
      rhs_shape[1] = rhs_is_trans ? k_i : n;
      out_shape[0] = m;
      out_shape[1] = n;
    }

    // Set matrix data pointers
    auto lhs_i = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
    auto rhs_i = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
    auto out_i = TensorWrapper(static_cast<void *>(out_ptr), out_shape, out_dtype);
    void *lhs_vptr = static_cast<void *>(lhs_ptr);
    void *rhs_vptr = static_cast<void *>(rhs_ptr);
    if (rhs_use_colwise)  // MatA to enter cuBLAS
      rhs_i.set_columnwise_data(rhs_vptr, rhs_dtype, rhs_shape);
    else
      rhs_i.set_rowwise_data(rhs_vptr, rhs_dtype, rhs_shape);
    if (lhs_use_colwise)  // MatB to enter cuBLAS
      lhs_i.set_columnwise_data(lhs_vptr, lhs_dtype, lhs_shape);
    else
      lhs_i.set_rowwise_data(lhs_vptr, lhs_dtype, lhs_shape);

    // Scale_inv shapes
    auto lhs_sinv_size = std::vector<size_t>{1};
    auto rhs_sinv_size = std::vector<size_t>{1};
    if (is_mxfp8_scaling) {
      NVTE_CHECK(k % MXFP8_BLOCK_SIZE == 0, "MXFP8 K-dim being divisble by %d (got %d)",
                 MXFP8_BLOCK_SIZE, k);
      size_t scale_k = k / MXFP8_BLOCK_SIZE;
      lhs_sinv_size[0] = m_i * scale_k;
      rhs_sinv_size[0] = n * scale_k;
      // Need to add swizzle here
    }

    // Set scale_inv pointers
    void *rhs_sinv_vptr = static_cast<void *>(rhs_sinv_ptr);
    void *lhs_sinv_vptr = static_cast<void *>(lhs_sinv_ptr);
    if (is_fp8_gemm) {
      if (rhs_use_colwise)  // MatA to enter cuBLAS
        rhs_i.set_columnwise_scale_inv(rhs_sinv_vptr, rhs_sinv_dtype, rhs_sinv_size);
      else
        rhs_i.set_rowwise_scale_inv(rhs_sinv_vptr, rhs_sinv_dtype, rhs_sinv_size);
      if (lhs_use_colwise)  // MatB to enter cuBLAS
        lhs_i.set_columnwise_scale_inv(lhs_sinv_vptr, lhs_sinv_dtype, lhs_sinv_size);
      else
        lhs_i.set_rowwise_scale_inv(lhs_sinv_vptr, lhs_sinv_dtype, lhs_sinv_size);
    } else {
      NVTE_CHECK(scaling_mode == JAXX_Scaling_Mode::NO_SCALING,
                 "Unsupported scaling mode: ", static_cast<int>(scaling_mode));
    }

    auto bias_i = TensorWrapper(bias_ptr, bias_shape, bias_dtype);
    auto pre_gelu_i = TensorWrapper(nullptr, std::vector<size_t>{0}, out_dtype);

    // Update pointer for the next GEMM pair
    lhs_ptr += lhs_shape[0] * lhs_shape[1] * lhs_dtype_bytes;
    rhs_ptr += rhs_shape[0] * rhs_shape[1] * rhs_dtype_bytes;
    out_ptr += out_shape[0] * out_shape[1] * out_dtype_bytes;
    if (is_fp8_gemm) {
      lhs_sinv_ptr += lhs_sinv_size[0] * lhs_sinv_dtype_bytes;
      rhs_sinv_ptr += rhs_sinv_size[0] * rhs_sinv_dtype_bytes;
    }
    if (has_bias) bias_ptr += n * bias_dtype_bytes;

    // Move objects to the lists to keep them alive
    lhs_wrapper_list.push_back(std::move(lhs_i));
    rhs_wrapper_list.push_back(std::move(rhs_i));
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
                                pre_gelu_list.data(), num_gemms, rhs_is_trans, lhs_is_trans, grad,
                                workspace_list.data(), accumulate, use_split_accumulator,
                                num_math_sm, stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmHandler, GroupedGemmFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // lhs_data
                                  .Arg<Buffer_Type>()      // lhs_sinv
                                  .Arg<Buffer_Type>()      // rhs_data
                                  .Arg<Buffer_Type>()      // rhs_sinv
                                  .Arg<Buffer_Type>()      // bias
                                  .Arg<Buffer_Type>()      // group_sizes
                                  .Arg<Buffer_Type>()      // group_offset
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // workspace
                                  .Attr<int64_t>("M")
                                  .Attr<int64_t>("N")
                                  .Attr<int64_t>("K")
                                  .Attr<bool>("lhs_is_trans")
                                  .Attr<bool>("rhs_is_trans")
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<bool>("has_bias")
                                  .Attr<bool>("is_grouped_dense_wgrad"),
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine

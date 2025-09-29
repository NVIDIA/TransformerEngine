/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/
#include "transformer_engine/gemm.h"

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <tuple>

#include "../extensions.h"
#include "cgemm_helper.h"
#include "common.h"
#include "common/util/cuda_runtime.h"
#include "common/util/string.h"
#include "common/util/system.h"
#include "cuda_runtime.h"
#include "nccl.h"
#include "transformer_engine/swizzle.h"
#include "xla/ffi/api/c_api.h"

#define MXFP8_BLOCK_SIZE 32

namespace transformer_engine {
namespace jax {

static uint8_t *move_ptr_to_next_256B_aligned(uint8_t *ptr) {
  // Move the pointer to the next 256B aligned address
  return reinterpret_cast<uint8_t *>((reinterpret_cast<uintptr_t>(ptr) + 255) &
                                     ~static_cast<uintptr_t>(255));
}

std::tuple<TensorWrapper, std::vector<size_t>> xla_buffer_to_nvte_gemm_operand(
    cudaStream_t stream, Buffer_Type buffer, Buffer_Type scale_inv, JAXX_Scaling_Mode scaling_mode,
    size_t axis_boundary, bool rowwise) {
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

    std::vector<size_t> scale_shape = {1};
    if (scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING) {
      // Block scaling also needs to be collapsed to match 2D data
      scale_shape = {product(scale_inv.dimensions(), 0, axis_boundary),
                     product(scale_inv.dimensions(), axis_boundary, scale_inv.dimensions().size())};
    }

    auto scale_dtype = convert_ffi_datatype_to_te_dtype(scale_inv.element_type());
    if (rowwise) {
      input.set_rowwise_scale_inv(scale_inv.untyped_data(), scale_dtype, scale_shape);
    } else {
      input.set_columnwise_scale_inv(scale_inv.untyped_data(), scale_dtype, scale_shape);
    }
  }

  return std::make_tuple(std::move(input), input_shape);
}

Error_Type CollectiveGemmInitFFI(Buffer_Type lhs, Buffer_Type lhs_scale_inv, Buffer_Type rhs,
                                 Buffer_Type rhs_scale_inv, Buffer_Type bias,
                                 Buffer_Type gelu_input, Result_Type output, Result_Type bias_grad,
                                 Result_Type pre_gelu_out, Result_Type workspace,
                                 JAXX_Scaling_Mode scaling_mode, int64_t lhs_axis_boundary,
                                 int64_t rhs_axis_boundary, bool lhs_transposed,
                                 bool rhs_transposed, bool fuse_bias, bool fuse_gelu, bool grad,
                                 bool use_split_accumulator, JAXX_Collective_Op collective_op) {
  nvte_cublas_handle_init();

  // Init UB buffer
  if (collective_op != JAXX_Collective_Op::NONE) {
    auto &comm_handler = CommunicatorHandler::get();
    std::vector<size_t> lhs_shape = {
        product(lhs.dimensions(), 0, lhs_axis_boundary),
        product(lhs.dimensions(), lhs_axis_boundary, lhs.dimensions().size())};
    std::vector<size_t> rhs_shape = {
        product(rhs.dimensions(), 0, rhs_axis_boundary),
        product(rhs.dimensions(), rhs_axis_boundary, rhs.dimensions().size())};

    std::vector<size_t> out_shape = {(lhs_transposed) ? lhs_shape[1] : lhs_shape[0],
                                     (rhs_transposed) ? rhs_shape[0] : rhs_shape[1]};

    std::vector<size_t> buffer_shape{0, 0};
    DType buffer_dtype = convert_ffi_datatype_to_te_dtype(output->element_type());
    if (collective_op == JAXX_Collective_Op::ALL_GATHER) {
      buffer_shape[0] = lhs_shape[0] * comm_handler.tp_size;
      buffer_shape[1] = lhs_shape[1];
      buffer_dtype = convert_ffi_datatype_to_te_dtype(lhs.element_type());
    } else if (collective_op == JAXX_Collective_Op::REDUCE_SCATTER) {
      buffer_shape[0] = out_shape[0];
      buffer_shape[1] = out_shape[1];
    }
    auto _ = CollectiveGemmPlanRegistry::getInstance().get_executor(buffer_shape, buffer_dtype,
                                                                    collective_op);
  }
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CollectiveGemmInitHandler, CollectiveGemmInitFFI,
                              FFI::Bind<FFI_Prepare>()
                                  .Arg<Buffer_Type>()  // lhs
                                  .Arg<Buffer_Type>()  // lhs_scale_inv
                                  .Arg<Buffer_Type>()  // rhs
                                  .Arg<Buffer_Type>()  // rhs_scale_inv
                                  .Arg<Buffer_Type>()  // bias
                                  .Arg<Buffer_Type>()  // gelu_input
                                  .Ret<Buffer_Type>()  // output
                                  .Ret<Buffer_Type>()  // bias_grad
                                  .Ret<Buffer_Type>()  // pre_gelu_out
                                  .Ret<Buffer_Type>()  // workspace
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<int64_t>("lhs_axis_boundary")
                                  .Attr<int64_t>("rhs_axis_boundary")
                                  .Attr<bool>("lhs_transposed")
                                  .Attr<bool>("rhs_transposed")
                                  .Attr<bool>("fuse_bias")
                                  .Attr<bool>("fuse_gelu")
                                  .Attr<bool>("grad")
                                  .Attr<bool>("use_split_accumulator")
                                  .Attr<JAXX_Collective_Op>("collective_op"));

Error_Type GemmFFI(cudaStream_t stream, Buffer_Type lhs, Buffer_Type lhs_scale_inv, Buffer_Type rhs,
                   Buffer_Type rhs_scale_inv, Buffer_Type bias, Buffer_Type gelu_input,
                   Result_Type output, Result_Type bias_grad, Result_Type pre_gelu_out,
                   Result_Type workspace, JAXX_Scaling_Mode scaling_mode, int64_t lhs_axis_boundary,
                   int64_t rhs_axis_boundary, bool lhs_transposed, bool rhs_transposed,
                   bool fuse_bias, bool fuse_gelu, bool grad, bool use_split_accumulator,
                   JAXX_Collective_Op collective_op) {
  // NOTE: TensorWrapper operands are always rowwise for full-precision GEMM, or FP8 GEMM when
  //       device supports non-TN layouts (compute capability >= 10.0, excluding 12.x)
  bool always_rowwise = (scaling_mode == JAXX_Scaling_Mode::NO_SCALING ||
                         (is_tensor_scaling(scaling_mode) && nvte_is_non_tn_fp8_gemm_supported()));
  bool make_lhs_rowwise = (always_rowwise) ? true : !lhs_transposed;
  bool make_rhs_rowwise = (always_rowwise) ? true : rhs_transposed;
  auto [lhs_, lhs_shape] = xla_buffer_to_nvte_gemm_operand(stream, lhs, lhs_scale_inv, scaling_mode,
                                                           lhs_axis_boundary, make_lhs_rowwise);
  auto [rhs_, rhs_shape] = xla_buffer_to_nvte_gemm_operand(stream, rhs, rhs_scale_inv, scaling_mode,
                                                           rhs_axis_boundary, make_rhs_rowwise);

  std::vector<size_t> out_shape = {(lhs_transposed) ? lhs_shape[1] : lhs_shape[0],
                                   (rhs_transposed) ? rhs_shape[0] : rhs_shape[1]};
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output->element_type());

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

  // cuBLAS workspace + 256 alignment enforcement
  auto workspace_ptr = reinterpret_cast<uint8_t *>(workspace->untyped_data());
  workspace_ptr = move_ptr_to_next_256B_aligned(workspace_ptr);
  std::vector<size_t> workspace_shape = {static_cast<size_t>(workspace->element_count()) - 256};
  auto workspace_ = TensorWrapper(workspace_ptr, workspace_shape, DType::kByte);

  // Launch TE/common kernel with swapped LHS/RHS for cuBLAS column-major order
  auto num_math_sm = cuda::sm_count() - getenv<int>("NVTE_EXT_MARGIN_SM", 0);

  if (collective_op == JAXX_Collective_Op::NONE) {
    auto out_ = TensorWrapper(output->untyped_data(), out_shape, out_dtype);
    NVTE_CHECK(out_.numel() == output->element_count(),
               "cuBLAS GEMM output buffer size is incorrect, expected ", out_.numel(), " elements ",
               to_string_like(out_shape), " but got ", output->element_count(), " elements ",
               to_string_like(output->dimensions()));

    nvte_cublas_gemm(rhs_.data(), lhs_.data(), out_.data(), bias_.data(), pre_gelu_.data(),
                     rhs_transposed, lhs_transposed, grad, workspace_.data(), false,
                     use_split_accumulator, num_math_sm, stream);
  } else {
    std::vector<size_t> buffer_shape{0, 0};
    DType buffer_dtype = out_dtype;
    auto &comm_handler = CommunicatorHandler::get();
    if (collective_op == JAXX_Collective_Op::ALL_GATHER) {
      buffer_shape[0] = lhs_shape[0] * comm_handler.tp_size;
      buffer_shape[1] = lhs_shape[1];
      out_shape[0] = out_shape[0] * comm_handler.tp_size;
      buffer_dtype = convert_ffi_datatype_to_te_dtype(lhs.element_type());
    } else if (collective_op == JAXX_Collective_Op::REDUCE_SCATTER) {
      buffer_shape[0] = out_shape[0];
      buffer_shape[1] = out_shape[1];
      out_shape[0] = out_shape[0] / comm_handler.tp_size;
    }
    auto executor = CollectiveGemmPlanRegistry::getInstance().get_executor(
        buffer_shape, buffer_dtype, collective_op);
    if (collective_op == JAXX_Collective_Op::REDUCE_SCATTER) {
      auto ubuf_out_ = TensorWrapper(executor->get_ubuf_dptr(), buffer_shape, out_dtype);
      // Prepare the auxiliary buffer for the reduce-scattered GEMM output
      auto out_ = TensorWrapper(output->untyped_data(), out_shape, out_dtype);
      NVTE_CHECK(out_.numel() == output->element_count(),
                 "cuBLAS GEMM output buffer size is incorrect, expected ", out_.numel(),
                 " elements ", to_string_like(out_shape), " but got ", output->element_count(),
                 " elements ", to_string_like(output->dimensions()));

      // Launch GEMM+RS
      executor->split_overlap_rs(rhs_, rhs_transposed, lhs_, lhs_transposed, ubuf_out_, bias_,
                                 pre_gelu_, workspace_, grad, false, use_split_accumulator, out_,
                                 stream);

    } else if (collective_op == JAXX_Collective_Op::ALL_GATHER) {
      auto aux_out_ = TensorWrapper(nullptr, std::vector<size_t>{0}, out_dtype);  // Empty

      auto out_ = TensorWrapper(output->untyped_data(), out_shape, out_dtype);
      NVTE_CHECK(out_.numel() == output->element_count(),
                 "cuBLAS GEMM output buffer size is incorrect, expected ", out_.numel(),
                 " elements ", to_string_like(out_shape), " but got ", output->element_count(),
                 " elements ", to_string_like(output->dimensions()));
      // Copy the distributed LHS operand into the local chunk of the communication buffer
      executor->copy_into_buffer(stream, lhs_, true, make_lhs_rowwise);
      // Launch AG+GEMM
      executor->split_overlap_ag(rhs_, rhs_transposed, lhs_, lhs_transposed, out_, bias_, pre_gelu_,
                                 workspace_, grad, false, use_split_accumulator, aux_out_, stream);
    }
  }

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
                                  .Ret<Buffer_Type>()      // workspace
                                  .Attr<JAXX_Scaling_Mode>("scaling_mode")
                                  .Attr<int64_t>("lhs_axis_boundary")
                                  .Attr<int64_t>("rhs_axis_boundary")
                                  .Attr<bool>("lhs_transposed")
                                  .Attr<bool>("rhs_transposed")
                                  .Attr<bool>("fuse_bias")
                                  .Attr<bool>("fuse_gelu")
                                  .Attr<bool>("grad")
                                  .Attr<bool>("use_split_accumulator")
                                  .Attr<JAXX_Collective_Op>("collective_op"),
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

  // It is weird that TE/Common GEMM only use colwise for MXFP8
  const bool is_fp8_gemm = is_fp8_dtype(lhs_dtype);
  const bool is_tensor_scaling = scaling_mode == JAXX_Scaling_Mode::DELAYED_TENSOR_SCALING ||
                                 scaling_mode == JAXX_Scaling_Mode::CURRENT_TENSOR_SCALING;
  const bool is_mxfp8_scaling = scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING;
  const bool rhs_use_colwise = is_mxfp8_scaling && !rhs_is_trans;
  const bool lhs_use_colwise = is_mxfp8_scaling && lhs_is_trans;

  // Outputs
  auto out_ptr = reinterpret_cast<uint8_t *>(output->untyped_data());
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output->element_type());
  // Here we clear the lower 8 bits of the buffer address to ensure the buffer is 256-aligned
  auto workspace_ptr = reinterpret_cast<uint8_t *>(workspace->untyped_data());
  workspace_ptr = move_ptr_to_next_256B_aligned(workspace_ptr);
  auto workspace_total_size = product(workspace->dimensions());

  auto lhs_sinv_size = product(lhs_sinv.dimensions());
  auto rhs_sinv_size = product(rhs_sinv.dimensions());
  const size_t workspace_alignment_padding = 256;
  const size_t tensor_scaling_sinv_aligment = 16;
  const size_t mxfp8_scaling_sinv_alignment_padding = 256;
  auto workspace_size = workspace_total_size - workspace_alignment_padding;
  if (is_mxfp8_scaling) {
    // For MXFP8 swizzled scale_inv buffers, only the first pointer needs to be with 256B alignment padding. Later pointers are guaranteed to be 256-aligned as the scale_inv shapes are padded by 128x4.
    workspace_size -= (lhs_sinv_size + rhs_sinv_size + 2 * mxfp8_scaling_sinv_alignment_padding);
  } else if (is_tensor_scaling) {
    // For tensor scaling, each matrix has a single scale value, and all scales need to be aligned
    // by 16 bytes to meet the requirement of CUDA 12.9.1 and later.
    workspace_size -= tensor_scaling_sinv_aligment * (lhs_sinv_size + rhs_sinv_size);
  }
  workspace_size = workspace_size / num_streams;
  auto swizzled_lhs_sinv_ptr = workspace_ptr + workspace_size * num_streams;
  swizzled_lhs_sinv_ptr = move_ptr_to_next_256B_aligned(swizzled_lhs_sinv_ptr);
  auto swizzled_rhs_sinv_ptr = swizzled_lhs_sinv_ptr + lhs_sinv_size;
  swizzled_rhs_sinv_ptr = move_ptr_to_next_256B_aligned(swizzled_rhs_sinv_ptr);
  auto lhs_scatter_aligned_ptr = swizzled_lhs_sinv_ptr;  // Already 256B aligned
  auto rhs_scatter_aligned_ptr = lhs_scatter_aligned_ptr + num_gemms * tensor_scaling_sinv_aligment;

  size_t lhs_dtype_bytes = te_dtype_bytes(lhs_dtype);
  size_t rhs_dtype_bytes = te_dtype_bytes(rhs_dtype);
  size_t lhs_sinv_dtype_bytes = te_dtype_bytes(lhs_sinv_dtype);
  size_t rhs_sinv_dtype_bytes = te_dtype_bytes(rhs_sinv_dtype);
  size_t bias_dtype_bytes = te_dtype_bytes(bias_dtype);
  size_t out_dtype_bytes = te_dtype_bytes(out_dtype);

  if (is_tensor_scaling) {
    size_t dpitch = tensor_scaling_sinv_aligment;
    size_t spitch = lhs_sinv_dtype_bytes;
    size_t width = lhs_sinv_dtype_bytes;
    size_t height = lhs_sinv_size;
    cudaMemcpy2DAsync(lhs_scatter_aligned_ptr, dpitch, lhs_sinv_ptr, spitch, width, height,
                      cudaMemcpyDeviceToDevice, stream);
    spitch = rhs_sinv_dtype_bytes;
    width = rhs_sinv_dtype_bytes;
    height = rhs_sinv_size;
    cudaMemcpy2DAsync(rhs_scatter_aligned_ptr, dpitch, rhs_sinv_ptr, spitch, width, height,
                      cudaMemcpyDeviceToDevice, stream);
    lhs_sinv_ptr = lhs_scatter_aligned_ptr;
    rhs_sinv_ptr = rhs_scatter_aligned_ptr;
  }

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

  if (arch < 100 && is_fp8_gemm) {
    NVTE_CHECK(!lhs_is_trans && rhs_is_trans,
               "For SM90 or older archs and FP8 input, only NT (row-major) GEMM is supported, ",
               "got lhs_is_trans=", lhs_is_trans, ", rhs_is_trans=", rhs_is_trans);
  }

  // These lists are to keep the TensorWrapper objects alive
  std::vector<TensorWrapper> lhs_wrapper_list;
  std::vector<TensorWrapper> rhs_wrapper_list;
  std::vector<TensorWrapper> lhs_swizzle_wrapper_list;  // For MXFP8 scale_inv swizzling
  std::vector<TensorWrapper> rhs_swizzle_wrapper_list;
  std::vector<TensorWrapper> bias_wrapper_list;
  std::vector<TensorWrapper> pre_gelu_wrapper_list;
  std::vector<TensorWrapper> out_wrapper_list;
  std::vector<TensorWrapper> workspace_wrapper_list;

  // These lists are the actual NVTETensor (void *) lists for multi-stream GEMM
  std::vector<NVTETensor> lhs_list;
  std::vector<NVTETensor> rhs_list;
  std::vector<NVTETensor> lhs_swizzle_list;
  std::vector<NVTETensor> rhs_swizzle_list;
  std::vector<NVTETensor> bias_list;
  std::vector<NVTETensor> pre_gelu_list;
  std::vector<NVTETensor> out_list;
  std::vector<NVTETensor> workspace_list;

  size_t lhs_sinv_total_size = 0;
  size_t rhs_sinv_total_size = 0;

  std::vector<void *> zero_out_dptr_list;
  std::vector<size_t> zero_out_size_list;

  for (size_t i = 0; i < num_gemms; i++) {
    // Matrix data shapes
    size_t m_i = dim_list_host[i];
    auto lhs_shape_i = std::vector<size_t>{m_i, k};
    auto rhs_shape_i = std::vector<size_t>{rhs_is_trans ? n : k, rhs_is_trans ? k : n};
    auto out_shape_i = std::vector<size_t>{m_i, n};
    if (is_grouped_dense_wgrad) {
      size_t k_i = dim_list_host[i];
      lhs_shape_i[0] = lhs_is_trans ? k_i : m;
      lhs_shape_i[1] = lhs_is_trans ? m : k_i;
      rhs_shape_i[0] = rhs_is_trans ? n : k_i;
      rhs_shape_i[1] = rhs_is_trans ? k_i : n;
      out_shape_i[0] = m;
      out_shape_i[1] = n;
    }

    size_t lhs_size = lhs_shape_i[0] * lhs_shape_i[1];
    size_t rhs_size = rhs_shape_i[0] * rhs_shape_i[1];
    size_t out_size = out_shape_i[0] * out_shape_i[1];
    bool is_empty_gemm = lhs_size == 0 || rhs_size == 0;
    if (is_empty_gemm && out_size > 0) {
      zero_out_dptr_list.push_back(out_ptr);
      zero_out_size_list.push_back(out_size * out_dtype_bytes);
    }

    // Set matrix data pointers
    auto lhs_i = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
    auto rhs_i = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
    auto out_i = TensorWrapper(static_cast<void *>(out_ptr), out_shape_i, out_dtype);
    void *lhs_vptr = static_cast<void *>(lhs_ptr);
    void *rhs_vptr = static_cast<void *>(rhs_ptr);
    if (rhs_use_colwise)  // MatA to enter cuBLAS
      rhs_i.set_columnwise_data(rhs_vptr, rhs_dtype, rhs_shape_i);
    else
      rhs_i.set_rowwise_data(rhs_vptr, rhs_dtype, rhs_shape_i);
    if (lhs_use_colwise)  // MatB to enter cuBLAS
      lhs_i.set_columnwise_data(lhs_vptr, lhs_dtype, lhs_shape_i);
    else
      lhs_i.set_rowwise_data(lhs_vptr, lhs_dtype, lhs_shape_i);

    // Set scale_inv shapes and pointers
    void *rhs_sinv_vptr = static_cast<void *>(rhs_sinv_ptr);
    void *lhs_sinv_vptr = static_cast<void *>(lhs_sinv_ptr);
    size_t lhs_sinv_size_i = 0;
    size_t rhs_sinv_size_i = 0;
    if (is_tensor_scaling) {
      auto tensor_scaling_sinv_shape = std::vector<size_t>{1};
      // If is_empty_gemm, scale_inv does not have the corresponding value, do not move the pointers
      if (!is_empty_gemm) {
        lhs_sinv_size_i = tensor_scaling_sinv_aligment / lhs_sinv_dtype_bytes;
        rhs_sinv_size_i = tensor_scaling_sinv_aligment / rhs_sinv_dtype_bytes;
      }
      if (rhs_use_colwise)  // MatA to enter cuBLAS
        rhs_i.set_columnwise_scale_inv(rhs_sinv_vptr, rhs_sinv_dtype, tensor_scaling_sinv_shape);
      else
        rhs_i.set_rowwise_scale_inv(rhs_sinv_vptr, rhs_sinv_dtype, tensor_scaling_sinv_shape);
      if (lhs_use_colwise)  // MatB to enter cuBLAS
        lhs_i.set_columnwise_scale_inv(lhs_sinv_vptr, lhs_sinv_dtype, tensor_scaling_sinv_shape);
      else
        lhs_i.set_rowwise_scale_inv(lhs_sinv_vptr, lhs_sinv_dtype, tensor_scaling_sinv_shape);
    } else if (is_mxfp8_scaling) {
      auto lhs_swizzle_i = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
      auto rhs_swizzle_i = TensorWrapper(get_nvte_scaling_mode(scaling_mode));
      void *swizzled_lhs_sinv_vptr = static_cast<void *>(swizzled_lhs_sinv_ptr);
      void *swizzled_rhs_sinv_vptr = static_cast<void *>(swizzled_rhs_sinv_ptr);

      // {lhs, rhs}_swizzle_i point to unswizzled scale_inv data as input, while {lhs, rhs}_i
      // point to swizzled scale_inv data (store on workspace, only used for GEMM).
      // Note: even if is_empty_gemm is true, sinv are still non-empty, need to move the pointers
      auto lhs_sinv_shape_i =
          get_mxfp8_scale_shape(lhs_shape_i[0], lhs_shape_i[1], lhs_use_colwise);
      auto rhs_sinv_shape_i =
          get_mxfp8_scale_shape(rhs_shape_i[0], rhs_shape_i[1], rhs_use_colwise);
      lhs_sinv_size_i = lhs_sinv_shape_i[0] * lhs_sinv_shape_i[1];
      rhs_sinv_size_i = rhs_sinv_shape_i[0] * rhs_sinv_shape_i[1];
      if (lhs_use_colwise) {
        lhs_swizzle_i.set_columnwise_data(lhs_vptr, lhs_dtype, lhs_shape_i);
        lhs_swizzle_i.set_columnwise_scale_inv(lhs_sinv_vptr, lhs_sinv_dtype, lhs_sinv_shape_i);
        lhs_i.set_columnwise_scale_inv(swizzled_lhs_sinv_vptr, lhs_sinv_dtype, lhs_sinv_shape_i);
      } else {
        lhs_swizzle_i.set_rowwise_data(lhs_vptr, lhs_dtype, lhs_shape_i);
        lhs_swizzle_i.set_rowwise_scale_inv(lhs_sinv_vptr, lhs_sinv_dtype, lhs_sinv_shape_i);
        lhs_i.set_rowwise_scale_inv(swizzled_lhs_sinv_vptr, lhs_sinv_dtype, lhs_sinv_shape_i);
      }
      if (rhs_use_colwise) {
        rhs_swizzle_i.set_columnwise_data(rhs_vptr, rhs_dtype, rhs_shape_i);
        rhs_swizzle_i.set_columnwise_scale_inv(rhs_sinv_vptr, rhs_sinv_dtype, rhs_sinv_shape_i);
        rhs_i.set_columnwise_scale_inv(swizzled_rhs_sinv_vptr, rhs_sinv_dtype, rhs_sinv_shape_i);
      } else {
        rhs_swizzle_i.set_rowwise_data(rhs_vptr, rhs_dtype, rhs_shape_i);
        rhs_swizzle_i.set_rowwise_scale_inv(rhs_sinv_vptr, rhs_sinv_dtype, rhs_sinv_shape_i);
        rhs_i.set_rowwise_scale_inv(swizzled_rhs_sinv_vptr, rhs_sinv_dtype, rhs_sinv_shape_i);
      }

      if (!is_empty_gemm) {
        lhs_swizzle_wrapper_list.push_back(std::move(lhs_swizzle_i));
        rhs_swizzle_wrapper_list.push_back(std::move(rhs_swizzle_i));
        lhs_swizzle_list.push_back(lhs_swizzle_wrapper_list.back().data());
        rhs_swizzle_list.push_back(rhs_swizzle_wrapper_list.back().data());
      }
    } else {
      NVTE_CHECK(scaling_mode == JAXX_Scaling_Mode::NO_SCALING,
                 "Unsupported scaling mode: ", static_cast<int>(scaling_mode));
    }

    auto bias_i = TensorWrapper(bias_ptr, bias_shape, bias_dtype);
    auto pre_gelu_i = TensorWrapper(nullptr, std::vector<size_t>{0}, out_dtype);

    // Update pointer for the next GEMM pair
    lhs_ptr += lhs_size * lhs_dtype_bytes;
    rhs_ptr += rhs_size * rhs_dtype_bytes;
    out_ptr += out_size * out_dtype_bytes;
    if (is_fp8_gemm) {
      lhs_sinv_ptr += lhs_sinv_size_i * lhs_sinv_dtype_bytes;
      rhs_sinv_ptr += rhs_sinv_size_i * rhs_sinv_dtype_bytes;
      lhs_sinv_total_size += lhs_sinv_size_i;
      rhs_sinv_total_size += rhs_sinv_size_i;
      if (is_mxfp8_scaling) {
        swizzled_lhs_sinv_ptr += lhs_sinv_size_i * lhs_sinv_dtype_bytes;
        swizzled_rhs_sinv_ptr += rhs_sinv_size_i * rhs_sinv_dtype_bytes;
      }
    }
    if (has_bias) bias_ptr += n * bias_dtype_bytes;

    // Move objects to the lists to keep them alive
    if (is_empty_gemm) continue;
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

  if (is_fp8_gemm) {
    if (is_tensor_scaling) {
      lhs_sinv_size *= tensor_scaling_sinv_aligment;
      rhs_sinv_size *= tensor_scaling_sinv_aligment;
    }
    NVTE_CHECK(lhs_sinv_total_size <= lhs_sinv_size, "Actual total lhs_sinv size ",
               lhs_sinv_total_size, " exceeds estimated upper bound ", lhs_sinv_size);
    NVTE_CHECK(rhs_sinv_total_size <= rhs_sinv_size, "Actual total rhs_sinv size ",
               rhs_sinv_total_size, " exceeds estimated upper bound ", rhs_sinv_size);
  }

  size_t num_non_empty_gemms = lhs_list.size();

  if (is_mxfp8_scaling) {
    for (int i = 0; i < num_non_empty_gemms; i++) {
      // The i-th GEMM will use the (i % num_streams)-th stream to compute,
      // use the same stream to swizzle the scaling factors to make sure that
      // the swizzling is done before the GEMM computation starts.
      int stream_id = i % num_streams;
      cudaStream_t stream_i = nvte_get_compute_stream(stream_id);
      nvte_swizzle_scaling_factors(lhs_swizzle_list[i], lhs_list[i], stream_i);
      nvte_swizzle_scaling_factors(rhs_swizzle_list[i], rhs_list[i], stream_i);
    }
  }

  // Launch zero-out kernels before the GEMM calls to use the sync in the multi-stream GEMM
  size_t num_zero_outs = zero_out_dptr_list.size();
  for (int i = 0; i < num_zero_outs; i++) {
    int stream_id = i % num_streams;
    cudaStream_t stream_i = nvte_get_compute_stream(stream_id);
    void *dptr = zero_out_dptr_list[i];
    size_t count = zero_out_size_list[i];
    NVTE_CHECK_CUDA(cudaMemsetAsync(dptr, 0, count, stream_i));
  }

  nvte_multi_tensor_gemm(rhs_list.data(), lhs_list.data(), out_list.data(), bias_list.data(),
                         pre_gelu_list.data(), num_non_empty_gemms, rhs_is_trans, lhs_is_trans,
                         grad, workspace_list.data(), accumulate, use_split_accumulator,
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
                                  .Attr<bool>("is_grouped_dense_wgrad"));

}  // namespace jax
}  // namespace transformer_engine

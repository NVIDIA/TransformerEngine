/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    cudaStream_t stream, Buffer_Type buffer, Buffer_Type scale_inv, uint8_t *swizzle_scale_ptr,
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
    NVTE_CHECK(is_nvfp4_scaling(scaling_mode) || typeToSize(input_dtype) == 1,
               "Quantized GEMM requires 4-bit or 8-bit operands.");
    NVTE_CHECK(scale_inv.element_count() > 0, "Missing inverse scaling factor for quantized GEMM.");

    std::vector<size_t> scale_shape = {1};
    auto is_nvfp4 = is_nvfp4_scaling(scaling_mode);
    auto scale_dtype = convert_ffi_datatype_to_te_dtype(scale_inv.element_type());

    if (scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING || is_nvfp4) {
      // Block scaling also needs to be collapsed to match 2D data
      scale_shape = {product(scale_inv.dimensions(), 0, axis_boundary),
                     product(scale_inv.dimensions(), axis_boundary, scale_inv.dimensions().size())};
      NVTE_CHECK(typeToSize(scale_dtype) == 1,
                 "Inverse scale factors need to have an 8-bit data type.");
    }
    if (scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING) {
      // Assume MXFP8 scales are already swizzled
      if (rowwise) {
        input.set_rowwise_scale_inv(scale_inv.untyped_data(), scale_dtype, scale_shape);
      } else {
        input.set_columnwise_scale_inv(scale_inv.untyped_data(), scale_dtype, scale_shape);
      }
      input.set_with_gemm_swizzled_scales(true);
    } else if (is_nvfp4) {  // Swizzle for NVFP4
      NVTE_CHECK(rowwise, "NVFP4 GEMM expects rowwise for both LHS and RHS");
      input.set_rowwise_scale_inv(scale_inv.untyped_data(), scale_dtype, scale_shape);
      // Create tensor to hold swizzled scale factor
      TensorWrapper output(get_nvte_scaling_mode(scaling_mode));
      output.set_rowwise_data(buffer.untyped_data(), input_dtype, input_shape);
      output.set_rowwise_scale_inv(swizzle_scale_ptr, scale_dtype, scale_shape);
      output.set_with_gemm_swizzled_scales(true);
      // Launch swizzle kernel
      nvte_swizzle_scaling_factors(input.data(), output.data(), stream);
      // Set swizzled scales into the input tensor
      input.set_rowwise_scale_inv(swizzle_scale_ptr, scale_dtype, scale_shape);
      input.set_with_gemm_swizzled_scales(true);
    } else {  // Tensor scaling
      if (rowwise) {
        input.set_rowwise_scale_inv(scale_inv.untyped_data(), scale_dtype, scale_shape);
      } else {
        input.set_columnwise_scale_inv(scale_inv.untyped_data(), scale_dtype, scale_shape);
      }
    }
  }

  return std::make_tuple(std::move(input), input_shape);
}

Error_Type GemmInitV2FFI(Buffer_Type lhs, Buffer_Type lhs_scale_inv, Buffer_Type rhs,
                         Buffer_Type rhs_scale_inv, Buffer_Type bias, Buffer_Type alpha,
                         Buffer_Type beta, Result_Type output, Result_Type workspace,
                         GemmConfig config) {
  nvte_cublas_handle_init();

  // Init UB buffer
  if (config.collective_op != JAXX_Collective_Op::NONE) {
    auto &comm_handler = CommunicatorHandler::get();
    std::vector<size_t> lhs_shape = {
        product(lhs.dimensions(), 0, config.lhs_axis_boundary),
        product(lhs.dimensions(), config.lhs_axis_boundary, lhs.dimensions().size())};
    std::vector<size_t> rhs_shape = {
        product(rhs.dimensions(), 0, config.rhs_axis_boundary),
        product(rhs.dimensions(), config.rhs_axis_boundary, rhs.dimensions().size())};

    std::vector<size_t> out_shape = {(config.lhs_transposed) ? lhs_shape[1] : lhs_shape[0],
                                     (config.rhs_transposed) ? rhs_shape[0] : rhs_shape[1]};

    std::vector<size_t> buffer_shape{0, 0};
    DType buffer_dtype = convert_ffi_datatype_to_te_dtype(output->element_type());
    if (config.collective_op == JAXX_Collective_Op::ALL_GATHER) {
      buffer_shape[0] = lhs_shape[0] * comm_handler.tp_size;
      buffer_shape[1] = lhs_shape[1];
      buffer_dtype = convert_ffi_datatype_to_te_dtype(lhs.element_type());
    } else if (config.collective_op == JAXX_Collective_Op::REDUCE_SCATTER) {
      buffer_shape[0] = out_shape[0];
      buffer_shape[1] = out_shape[1];
    }
    [[maybe_unused]] auto _ = CollectiveGemmPlanRegistry::getInstance().get_executor(
        buffer_shape, buffer_dtype, config.collective_op);
  }
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GemmInitV2Handler, GemmInitV2FFI,
                              FFI::Bind<FFI_Prepare>()
                                  .Arg<Buffer_Type>()  // lhs
                                  .Arg<Buffer_Type>()  // lhs_scale_inv
                                  .Arg<Buffer_Type>()  // rhs
                                  .Arg<Buffer_Type>()  // rhs_scale_inv
                                  .Arg<Buffer_Type>()  // bias
                                  .Arg<Buffer_Type>()  // alpha
                                  .Arg<Buffer_Type>()  // beta
                                  .Ret<Buffer_Type>()  // output
                                  .Ret<Buffer_Type>()  // workspace
                                  .Attr<GemmConfig>("config"),
                              FFI_CudaGraph_Traits);

Error_Type CollectiveGemmInitFFI(Buffer_Type lhs, Buffer_Type lhs_scale_inv, Buffer_Type rhs,
                                 Buffer_Type rhs_scale_inv, Buffer_Type bias,
                                 Buffer_Type gelu_input, Buffer_Type alpha, Buffer_Type beta,
                                 Result_Type output, Result_Type bias_grad,
                                 Result_Type pre_gelu_out, Result_Type workspace,
                                 JAXX_Scaling_Mode scaling_mode, int64_t lhs_axis_boundary,
                                 int64_t rhs_axis_boundary, bool lhs_transposed,
                                 bool rhs_transposed, bool fuse_bias, bool fuse_gelu, bool grad,
                                 bool use_split_accumulator, JAXX_Collective_Op collective_op) {
  static std::once_flag gemm_init_warned;
  std::call_once(gemm_init_warned, []() {
    std::cerr << "[CollectiveGemmInitFFI] Deprecation: This API is deprecated and will be removed "
                 "in September 2026. Use GemmInitV2FFI instead."
              << std::endl;
  });
  return GemmInitV2FFI(lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, alpha, beta, output, workspace,
                       GemmConfig{scaling_mode, collective_op, lhs_axis_boundary, rhs_axis_boundary,
                                  lhs_transposed, rhs_transposed, use_split_accumulator});
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CollectiveGemmInitHandler, CollectiveGemmInitFFI,
                              FFI::Bind<FFI_Prepare>()
                                  .Arg<Buffer_Type>()  // lhs
                                  .Arg<Buffer_Type>()  // lhs_scale_inv
                                  .Arg<Buffer_Type>()  // rhs
                                  .Arg<Buffer_Type>()  // rhs_scale_inv
                                  .Arg<Buffer_Type>()  // bias
                                  .Arg<Buffer_Type>()  // gelu_input
                                  .Arg<Buffer_Type>()  // alpha
                                  .Arg<Buffer_Type>()  // beta
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
                                  .Attr<JAXX_Collective_Op>("collective_op"),
                              FFI_CudaGraph_Traits);

Error_Type GemmV2FFI(cudaStream_t stream, Buffer_Type lhs, Buffer_Type lhs_scale_inv,
                     Buffer_Type rhs, Buffer_Type rhs_scale_inv, Buffer_Type bias,
                     Buffer_Type alpha, Buffer_Type beta, Result_Type output, Result_Type workspace,
                     GemmConfig config) {
  // cuBLAS workspace + 256 alignment enforcement (+ swizzle scales)
  uint8_t *lhs_swizzle_scale_ptr = nullptr, *rhs_swizzle_scale_ptr = nullptr;
  auto workspace_ptr = reinterpret_cast<uint8_t *>(workspace->untyped_data());
  workspace_ptr = move_ptr_to_next_256B_aligned(workspace_ptr);
  size_t workspace_size = static_cast<size_t>(workspace->element_count()) - 256;

  if (is_nvfp4_scaling(config.scaling_mode)) {
    auto lhs_scale_size = product(lhs_scale_inv.dimensions());
    auto rhs_scale_size = product(rhs_scale_inv.dimensions());
    workspace_size = workspace_size - lhs_scale_size - rhs_scale_size;
    lhs_swizzle_scale_ptr = workspace_ptr;
    rhs_swizzle_scale_ptr = workspace_ptr + lhs_scale_size;
    workspace_ptr = rhs_swizzle_scale_ptr + rhs_scale_size;
  }
  auto workspace_ = TensorWrapper(workspace_ptr, std::vector<size_t>{workspace_size}, DType::kByte);

  // NOTE: TensorWrapper operands are always rowwise for full-precision GEMM, or FP8 GEMM when
  //       device supports non-TN layouts (compute capability >= 10.0, excluding 12.x)
  bool always_rowwise =
      (config.scaling_mode == JAXX_Scaling_Mode::NO_SCALING ||
       (is_tensor_scaling(config.scaling_mode) && nvte_is_non_tn_fp8_gemm_supported()));
  bool make_lhs_rowwise = (always_rowwise) ? true : !config.lhs_transposed;
  bool make_rhs_rowwise = (always_rowwise) ? true : config.rhs_transposed;

  auto [lhs_, lhs_shape] = xla_buffer_to_nvte_gemm_operand(
      stream, lhs, lhs_scale_inv, lhs_swizzle_scale_ptr, config.scaling_mode,
      config.lhs_axis_boundary, make_lhs_rowwise);
  auto [rhs_, rhs_shape] = xla_buffer_to_nvte_gemm_operand(
      stream, rhs, rhs_scale_inv, rhs_swizzle_scale_ptr, config.scaling_mode,
      config.rhs_axis_boundary, make_rhs_rowwise);

  std::vector<size_t> out_shape = {(config.lhs_transposed) ? lhs_shape[1] : lhs_shape[0],
                                   (config.rhs_transposed) ? rhs_shape[0] : rhs_shape[1]};
  auto out_dtype = convert_ffi_datatype_to_te_dtype(output->element_type());

  // Bias input to forward pass or bias gradient output from backward pass
  void *bias_ptr = nullptr;
  size_t bias_size = 0;
  DType bias_dtype = out_dtype;
  auto fuse_bias = bias.element_count() > 0;
  if (fuse_bias) {
    bias_ptr = bias.untyped_data();
    bias_size = product(bias.dimensions());
    bias_dtype = convert_ffi_datatype_to_te_dtype(bias.element_type());
  }
  auto bias_ = TensorWrapper(bias_ptr, std::vector<size_t>{bias_size}, bias_dtype);

  auto num_math_sm = cuda::sm_count() - getenv<int>("NVTE_EXT_MARGIN_SM", 0);

  float one = 1.;
  float zero = 0.;
  // alpha, beta
  float *alpha_ptr = &one, *beta_ptr = &zero;
  if (is_nvfp4_scaling(config.scaling_mode)) {
    NVTE_CHECK(alpha.element_count() == 1 &&
               convert_ffi_datatype_to_te_dtype(alpha.element_type()) == DType::kFloat32);
    alpha_ptr = reinterpret_cast<float *>(alpha.untyped_data());
    NVTE_CHECK(beta.element_count() == 1 &&
               convert_ffi_datatype_to_te_dtype(beta.element_type()) == DType::kFloat32);
    beta_ptr = reinterpret_cast<float *>(beta.untyped_data());
  }

  // Construct GEMM config
  transformer_engine::MatmulConfigWrapper matmul_config;
  matmul_config.set_use_split_accumulator(config.use_split_accumulator);
  matmul_config.set_sm_count(num_math_sm);
  if (fuse_bias) matmul_config.set_bias_tensor(bias_.data());

  if (config.collective_op == JAXX_Collective_Op::NONE) {
    auto out_ = TensorWrapper(output->untyped_data(), out_shape, out_dtype);
    NVTE_CHECK(out_.numel() == output->element_count(),
               "cuBLAS GEMM output buffer size is incorrect, expected ", out_.numel(), " elements ",
               to_string_like(out_shape), " but got ", output->element_count(), " elements ",
               to_string_like(output->dimensions()));
    NVTE_CHECK(!fuse_bias || bias_size == out_shape[1], "bias_size=", bias_size,
               ", out_shape[1]=", out_shape[1]);

    // Launch TE/common kernel with swapped LHS/RHS for cuBLAS column-major order
    nvte_cublas_gemm_v2(config.rhs_transposed /*transa*/, config.lhs_transposed /*transb*/,
                        alpha_ptr, rhs_.data() /*A*/, lhs_.data() /*B*/, beta_ptr,
                        out_.data() /*C*/, out_.data() /*D*/, workspace_.data(), matmul_config,
                        stream);
  } else {
    std::vector<size_t> buffer_shape{0, 0};
    DType buffer_dtype = out_dtype;
    auto &comm_handler = CommunicatorHandler::get();
    if (config.collective_op == JAXX_Collective_Op::ALL_GATHER) {
      buffer_shape[0] = lhs_shape[0] * comm_handler.tp_size;
      buffer_shape[1] = lhs_shape[1];
      out_shape[0] = out_shape[0] * comm_handler.tp_size;
      buffer_dtype = convert_ffi_datatype_to_te_dtype(lhs.element_type());
    } else if (config.collective_op == JAXX_Collective_Op::REDUCE_SCATTER) {
      buffer_shape[0] = out_shape[0];
      buffer_shape[1] = out_shape[1];
      out_shape[0] = out_shape[0] / comm_handler.tp_size;
    }
    NVTE_CHECK(!fuse_bias || bias_size == out_shape[1], "bias_size=", bias_size,
               ", out_shape[1]=", out_shape[1]);
    auto executor = CollectiveGemmPlanRegistry::getInstance().get_executor(
        buffer_shape, buffer_dtype, config.collective_op);
    auto pre_gelu_ = TensorWrapper(nullptr, std::vector<size_t>{0}, DType::kByte);
    if (config.collective_op == JAXX_Collective_Op::REDUCE_SCATTER) {
      auto ubuf_out_ = TensorWrapper(executor->get_ubuf_dptr(), buffer_shape, out_dtype);
      // Prepare the auxiliary buffer for the reduce-scattered GEMM output
      auto out_ = TensorWrapper(output->untyped_data(), out_shape, out_dtype);
      NVTE_CHECK(out_.numel() == output->element_count(),
                 "cuBLAS GEMM output buffer size is incorrect, expected ", out_.numel(),
                 " elements ", to_string_like(out_shape), " but got ", output->element_count(),
                 " elements ", to_string_like(output->dimensions()));

      // Launch GEMM+RS
      executor->split_overlap_rs(rhs_, config.rhs_transposed, lhs_, config.lhs_transposed,
                                 ubuf_out_, bias_, pre_gelu_, workspace_, false /*grad*/,
                                 false /*accumulate*/, config.use_split_accumulator, out_, stream);

    } else if (config.collective_op == JAXX_Collective_Op::ALL_GATHER) {
      auto aux_out_ = TensorWrapper(nullptr, std::vector<size_t>{0}, out_dtype);  // Empty

      auto out_ = TensorWrapper(output->untyped_data(), out_shape, out_dtype);
      NVTE_CHECK(out_.numel() == output->element_count(),
                 "cuBLAS GEMM output buffer size is incorrect, expected ", out_.numel(),
                 " elements ", to_string_like(out_shape), " but got ", output->element_count(),
                 " elements ", to_string_like(output->dimensions()));
      // Copy the distributed LHS operand into the local chunk of the communication buffer
      executor->copy_into_buffer(stream, lhs_, true, make_lhs_rowwise);
      // Launch AG+GEMM
      executor->split_overlap_ag(rhs_, config.rhs_transposed, lhs_, config.lhs_transposed, out_,
                                 bias_, pre_gelu_, workspace_, false /*grad*/, false /*accumulate*/,
                                 config.use_split_accumulator, aux_out_, stream);
    }
  }

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GemmV2Handler, GemmV2FFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // lhs
                                  .Arg<Buffer_Type>()      // lhs_scale_inv
                                  .Arg<Buffer_Type>()      // rhs
                                  .Arg<Buffer_Type>()      // rhs_scale_inv
                                  .Arg<Buffer_Type>()      // bias
                                  .Arg<Buffer_Type>()      // alpha
                                  .Arg<Buffer_Type>()      // beta
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // workspace
                                  .Attr<GemmConfig>("config"),
                              FFI_CudaGraph_Traits);

Error_Type GemmFFI(cudaStream_t stream, Buffer_Type lhs, Buffer_Type lhs_scale_inv, Buffer_Type rhs,
                   Buffer_Type rhs_scale_inv, Buffer_Type bias, Buffer_Type gelu_input,
                   Buffer_Type alpha, Buffer_Type beta, Result_Type output, Result_Type bias_grad,
                   Result_Type pre_gelu_out, Result_Type workspace, JAXX_Scaling_Mode scaling_mode,
                   int64_t lhs_axis_boundary, int64_t rhs_axis_boundary, bool lhs_transposed,
                   bool rhs_transposed, bool fuse_bias, bool fuse_gelu, bool grad,
                   bool use_split_accumulator, JAXX_Collective_Op collective_op) {
  static std::once_flag once_fuse_bias;
  static std::once_flag once_fuse_gelu_grad;
  static std::once_flag once_api;
  if (fuse_bias) {
    std::call_once(once_fuse_bias, [] {
      std::cerr << "[GemmFFI] Deprecation: fuse_bias is deprecated; bias fusion is inferred from "
                   "non-empty bias. This parameter will be removed in future release."
                << std::endl;
    });
  }
  if (fuse_gelu || grad) {
    std::call_once(once_fuse_gelu_grad, [] {
      std::cerr << "[GemmFFI] Deprecation: fuse_gelu and grad are deprecated. These options are "
                   "ignored as there is no support for them in the current implementation. "
                << std::endl;
    });
  }
  std::call_once(once_api, [] {
    std::cerr << "[GemmFFI] Deprecation: This API is deprecated in Sep 2026. Use GemmV2FFI instead."
              << std::endl;
  });

  return GemmV2FFI(stream, lhs, lhs_scale_inv, rhs, rhs_scale_inv, bias, alpha, beta, output,
                   workspace,
                   GemmConfig{scaling_mode, collective_op, lhs_axis_boundary, rhs_axis_boundary,
                              lhs_transposed, rhs_transposed, use_split_accumulator});
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
                                  .Arg<Buffer_Type>()      // alpha
                                  .Arg<Buffer_Type>()      // beta
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

size_t GroupedGemmGetGroupSizes(cudaStream_t stream, size_t num_gemms, int32_t *dev_group_sizes,
                                int32_t *host_group_sizes) {
  static std::once_flag init_flag;
  static cudaEvent_t d2h_event;
  static size_t host_num_gemms;
  static const size_t max_num_gemms = 1024;
  //static int32_t host_group_sizes_internal[max_num_gemms];
  static int32_t *host_group_sizes_internal = nullptr;
  auto init = [&]() {
    NVTE_CHECK_CUDA(cudaEventCreate(&d2h_event));
    NVTE_CHECK_CUDA(cudaMallocHost(&host_group_sizes_internal, sizeof(int32_t) * max_num_gemms));
  };
  std::call_once(init_flag, init);

  NVTE_CHECK(dev_group_sizes == nullptr || host_group_sizes == nullptr,
             "Only one of dev_group_sizes and host_group_sizes can be non-nullptr.");

  if (dev_group_sizes != nullptr) {
    NVTE_CHECK(num_gemms <= max_num_gemms, "num_gemms ", num_gemms, " exceeds the maximum ",
               "supported number ", max_num_gemms, " to be downloaded in advance.");
    host_num_gemms = num_gemms;
    // Wait for current compute stream to finish
    cudaStream_t compute_stream_0 = nvte_get_compute_stream(0);
    NVTE_CHECK_CUDA(cudaEventRecord(d2h_event, stream));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(compute_stream_0, d2h_event));
    // Async copy group_sizes from device to host
    size_t copy_bytes = sizeof(int32_t) * num_gemms;
    NVTE_CHECK_CUDA(cudaMemcpyAsync(host_group_sizes_internal, dev_group_sizes, copy_bytes,
                                    cudaMemcpyDeviceToHost, compute_stream_0));
    NVTE_CHECK_CUDA(cudaEventRecord(d2h_event, compute_stream_0));
    return num_gemms;
  }

  if (host_group_sizes != nullptr) {
    if (host_num_gemms == 0) return 0;
    NVTE_CHECK(host_num_gemms == num_gemms, "num_gemms ", num_gemms,
               " does not match the previous value ", host_num_gemms, ".");
    // Wait for the async copy to finish, then copy group_sizes to user buffer
    // Note: This may break cudaGraph.
    NVTE_CHECK_CUDA(cudaEventSynchronize(d2h_event));
    memcpy(host_group_sizes, host_group_sizes_internal, sizeof(int32_t) * host_num_gemms);
    return host_num_gemms;
  }
}

Error_Type GroupedGemmD2HGroupSizesFFI(cudaStream_t stream, Buffer_Type group_sizes,
                                       Result_Type dummy_output, size_t num_gemms) {
  int32_t *dev_group_sizes = reinterpret_cast<int32_t *>(group_sizes.untyped_data());
  GroupedGemmGetGroupSizes(stream, num_gemms, dev_group_sizes, nullptr);
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmD2HGroupSizesHandler, GroupedGemmD2HGroupSizesFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // group_sizes
                                  .Ret<Buffer_Type>()      // dummy_output
                                  .Attr<int64_t>("num_gemms"));

class JAXX_GroupedTensorWrapper {
 public:
  JAXX_GroupedTensorWrapper() = delete;
  JAXX_GroupedTensorWrapper(JAXX_Scaling_Mode scaling_mode, size_t num_tensors,
                            NVTEShape const &dataShape);
  JAXX_GroupedTensorWrapper(JAXX_GroupedTensorWrapper const &) = delete;
  JAXX_GroupedTensorWrapper &operator=(JAXX_GroupedTensorWrapper const &) = delete;
  JAXX_GroupedTensorWrapper(JAXX_GroupedTensorWrapper &&other) noexcept
      : m_data_shape(other.m_data_shape),
        m_grouped_tensor(other.m_grouped_tensor),
        m_data_tensor(other.m_data_tensor),
        m_scale_inv_tensor(other.m_scale_inv_tensor),
        m_colwise_data_tensor(other.m_colwise_data_tensor),
        m_colwise_scale_inv_tensor(other.m_colwise_scale_inv_tensor),
        m_sizes_tensor(other.m_sizes_tensor),
        m_offsets_tensor(other.m_offsets_tensor) {
    other.m_grouped_tensor = nullptr;
  }
  JAXX_GroupedTensorWrapper &operator=(JAXX_GroupedTensorWrapper &&) = delete;
  ~JAXX_GroupedTensorWrapper();

  void set_rowwise(Buffer_Type const &data, std::optional<Buffer_Type> const &scale_inv);
  void set_columnwise(Buffer_Type const &data, std::optional<Buffer_Type> const &scale_inv);
  void set_with_gemm_swizzled_scales(bool val);
  void replace_scale_inv(bool use_colwise, uint8_t *sinv_ptr, NVTEDType sinv_dtype,
                         NVTEShape sinv_shape);
  void set_group_info(Buffer_Type const &group_sizes, Buffer_Type const &group_offsets,
                      NVTEGroupedTensorParam group_sizes_param_name);
  // Set only group sizes (no offsets); the setup kernel will compute offsets from sizes.
  void set_group_sizes_only(const int64_t *sizes_ptr, size_t num_tensors,
                            NVTEGroupedTensorParam group_sizes_param_name);

  operator NVTEGroupedTensor() const { return m_grouped_tensor; }
  NVTEGroupedTensor const &get_grouped_tensor() const;

 private:
  NVTEShape m_data_shape{};
  NVTEGroupedTensor m_grouped_tensor{};

  // Internal tensors. These need to be kept alive as long as the grouped tensor is alive.
  NVTEBasicTensor m_data_tensor{};
  NVTEBasicTensor m_scale_inv_tensor{};
  NVTEBasicTensor m_colwise_data_tensor{};
  NVTEBasicTensor m_colwise_scale_inv_tensor{};

  NVTEBasicTensor m_sizes_tensor{};
  NVTEBasicTensor m_offsets_tensor{};
};

JAXX_GroupedTensorWrapper::JAXX_GroupedTensorWrapper(JAXX_Scaling_Mode scaling_mode,
                                                     size_t num_tensors,
                                                     NVTEShape const &dataShape) {
  m_data_shape = dataShape;
  m_grouped_tensor =
      nvte_create_grouped_tensor(get_nvte_scaling_mode(scaling_mode), num_tensors, dataShape);
}

JAXX_GroupedTensorWrapper::~JAXX_GroupedTensorWrapper() {
  if (m_grouped_tensor != nullptr) {
    nvte_destroy_grouped_tensor(m_grouped_tensor);
  }
}

void JAXX_GroupedTensorWrapper::set_rowwise(Buffer_Type const &data,
                                            std::optional<Buffer_Type> const &scale_inv) {
  NVTEDType data_dtype =
      static_cast<NVTEDType>(convert_ffi_datatype_to_te_dtype(data.element_type()));
  m_data_tensor =
      NVTEBasicTensor{reinterpret_cast<uint8_t *>(data.untyped_data()), data_dtype, m_data_shape};

  nvte_set_grouped_tensor_param(m_grouped_tensor, kNVTEGroupedRowwiseData, &m_data_tensor,
                                sizeof(m_data_tensor));

  if (scale_inv.has_value()) {
    NVTEDType scale_inv_dtype =
        static_cast<NVTEDType>(convert_ffi_datatype_to_te_dtype(scale_inv->element_type()));
    NVTEShape logical_scale_shape{};
    if (scale_inv->dimensions().size() == 1) {
      logical_scale_shape.ndim = 1;
      logical_scale_shape.data[0] = scale_inv->dimensions()[0];
    } else if (scale_inv->dimensions().size() == 2) {
      logical_scale_shape.ndim = 2;
      logical_scale_shape.data[0] = scale_inv->dimensions()[0];
      logical_scale_shape.data[1] = scale_inv->dimensions()[1];
    } else {
      NVTE_CHECK(false, "Expected 1D or 2D tensor for GEMM scale_inv but received ndim=",
                 scale_inv->dimensions().size());
    }
    m_scale_inv_tensor = NVTEBasicTensor{reinterpret_cast<uint8_t *>(scale_inv->untyped_data()),
                                         scale_inv_dtype, logical_scale_shape};
    nvte_set_grouped_tensor_param(m_grouped_tensor, kNVTEGroupedRowwiseScaleInv,
                                  &m_scale_inv_tensor, sizeof(m_scale_inv_tensor));
  }
}

void JAXX_GroupedTensorWrapper::set_columnwise(Buffer_Type const &data,
                                               std::optional<Buffer_Type> const &scale_inv) {
  NVTEDType data_dtype =
      static_cast<NVTEDType>(convert_ffi_datatype_to_te_dtype(data.element_type()));
  m_colwise_data_tensor =
      NVTEBasicTensor{reinterpret_cast<uint8_t *>(data.untyped_data()), data_dtype, m_data_shape};

  nvte_set_grouped_tensor_param(m_grouped_tensor, kNVTEGroupedColumnwiseData,
                                &m_colwise_data_tensor, sizeof(m_colwise_data_tensor));

  if (scale_inv.has_value()) {
    NVTEDType scale_inv_dtype =
        static_cast<NVTEDType>(convert_ffi_datatype_to_te_dtype(scale_inv->element_type()));
    NVTEShape logical_scale_shape{};
    if (scale_inv->dimensions().size() == 1) {
      logical_scale_shape.ndim = 1;
      logical_scale_shape.data[0] = scale_inv->dimensions()[0];
    } else if (scale_inv->dimensions().size() == 2) {
      logical_scale_shape.ndim = 2;
      logical_scale_shape.data[0] = scale_inv->dimensions()[0];
      logical_scale_shape.data[1] = scale_inv->dimensions()[1];
    } else {
      NVTE_CHECK(false, "Expected 1D or 2D tensor for GEMM columnwise scale_inv but received ndim=",
                 scale_inv->dimensions().size());
    }
    m_colwise_scale_inv_tensor =
        NVTEBasicTensor{reinterpret_cast<uint8_t *>(scale_inv->untyped_data()), scale_inv_dtype,
                        logical_scale_shape};
    nvte_set_grouped_tensor_param(m_grouped_tensor, kNVTEGroupedColumnwiseScaleInv,
                                  &m_colwise_scale_inv_tensor, sizeof(m_colwise_scale_inv_tensor));
  }
}

void JAXX_GroupedTensorWrapper::set_with_gemm_swizzled_scales(bool val) {
  auto v = static_cast<uint8_t>(val);
  nvte_set_grouped_tensor_param(m_grouped_tensor, kNVTEGroupedWithGEMMSwizzledScales, &v,
                                sizeof(v));
}

void JAXX_GroupedTensorWrapper::replace_scale_inv(bool use_colwise, uint8_t *sinv_ptr,
                                                  NVTEDType sinv_dtype, NVTEShape sinv_shape) {
  if (use_colwise) {
    m_colwise_scale_inv_tensor = NVTEBasicTensor{sinv_ptr, sinv_dtype, sinv_shape};
    nvte_set_grouped_tensor_param(m_grouped_tensor, kNVTEGroupedColumnwiseScaleInv,
                                  &m_colwise_scale_inv_tensor, sizeof(m_colwise_scale_inv_tensor));
  } else {
    m_scale_inv_tensor = NVTEBasicTensor{sinv_ptr, sinv_dtype, sinv_shape};
    nvte_set_grouped_tensor_param(m_grouped_tensor, kNVTEGroupedRowwiseScaleInv,
                                  &m_scale_inv_tensor, sizeof(m_scale_inv_tensor));
  }
}

void JAXX_GroupedTensorWrapper::set_group_info(Buffer_Type const &group_sizes,
                                               Buffer_Type const &group_offsets,
                                               NVTEGroupedTensorParam group_sizes_param_name) {
  NVTEDType sizes_dtype =
      static_cast<NVTEDType>(convert_ffi_datatype_to_te_dtype(group_sizes.element_type()));
  NVTEDType offsets_dtype =
      static_cast<NVTEDType>(convert_ffi_datatype_to_te_dtype(group_offsets.element_type()));

  NVTE_CHECK(sizes_dtype == NVTEDType::kNVTEInt64, "group_sizes must be of type int64.");
  NVTE_CHECK(offsets_dtype == NVTEDType::kNVTEInt64, "group_offsets must be of type int64.");

  size_t num_tensors = group_sizes.dimensions()[0];
  NVTE_CHECK(group_sizes.dimensions().size() == 1,
             "group_sizes must be a 1D tensor with length equal to the number of tensors.");
  NVTE_CHECK(group_offsets.dimensions().size() == 1,
             "group_offsets must be a 1D tensor with length equal to the number of tensors.");
  NVTE_CHECK(group_offsets.dimensions()[0] == num_tensors,
             "group_sizes and group_offsets must have the same number of elements.");

  NVTEShape shape{};
  shape.ndim = 1;
  shape.data[0] = num_tensors;

  m_sizes_tensor = NVTEBasicTensor{reinterpret_cast<uint8_t *>(group_sizes.untyped_data()),
                                   NVTEDType::kNVTEInt64, shape};
  m_offsets_tensor = NVTEBasicTensor{reinterpret_cast<uint8_t *>(group_offsets.untyped_data()),
                                     NVTEDType::kNVTEInt64, shape};

  nvte_set_grouped_tensor_param(m_grouped_tensor, group_sizes_param_name, &m_sizes_tensor,
                                sizeof(m_sizes_tensor));
  nvte_set_grouped_tensor_param(m_grouped_tensor, kNVTEGroupedTensorOffsets, &m_offsets_tensor,
                                sizeof(m_offsets_tensor));
}

void JAXX_GroupedTensorWrapper::set_group_sizes_only(
    const int64_t *sizes_ptr, size_t num_tensors, NVTEGroupedTensorParam group_sizes_param_name) {
  NVTEShape shape{};
  shape.ndim = 1;
  shape.data[0] = num_tensors;
  m_sizes_tensor = NVTEBasicTensor{reinterpret_cast<uint8_t *>(const_cast<int64_t *>(sizes_ptr)),
                                   NVTEDType::kNVTEInt64, shape};
  nvte_set_grouped_tensor_param(m_grouped_tensor, group_sizes_param_name, &m_sizes_tensor,
                                sizeof(m_sizes_tensor));
  // Intentionally no offset tensor: offsets will be computed by the setup kernel.
}

NVTEGroupedTensor const &JAXX_GroupedTensorWrapper::get_grouped_tensor() const {
  return m_grouped_tensor;
}

JAXX_GroupedTensorWrapper make_grouped_tensor(Buffer_Type const &data,
                                              std::optional<Buffer_Type> scale_inv,
                                              JAXX_Scaling_Mode scaling_mode, size_t num_tensors,
                                              NVTEShape const &dataShape) {
  JAXX_GroupedTensorWrapper grouped_tensor_wrapper(scaling_mode, num_tensors, dataShape);
  if (scaling_mode == JAXX_Scaling_Mode::NO_SCALING) {
    scale_inv = std::nullopt;
  }
  grouped_tensor_wrapper.set_rowwise(data, scale_inv);

  return std::move(grouped_tensor_wrapper);
}

// V2 variant (NO_SCALING): derives data shape from the XLA buffer directly, converts group_sizes
// int32→int64 per-tensor into a dedicated slot of int64_workspace, and wires first_dims/last_dims.
// int64_offset (in int64 elements) is updated on return to the next available slot so callers can
// thread it through successive make_grouped_tensor calls without aliasing.  Bounds are checked
// before each slot is used.  Only NO_SCALING is supported by this overload.
JAXX_GroupedTensorWrapper make_grouped_tensor(
    Buffer_Type const &data, Buffer_Type const &first_dims, Buffer_Type const &last_dims,
    int64_t *int64_workspace_base, size_t int64_workspace_capacity, size_t &int64_offset,
    size_t num_gemms, cudaStream_t stream, size_t left_size, size_t right_size) {
  auto dims = data.dimensions();
  NVTE_CHECK(product(dims) == left_size * right_size,
             "grouped GEMM data buffer element count does not match the provided 2D shape.");
  NVTEShape dataShape{.data = {left_size, right_size}, .ndim = 2};
  JAXX_GroupedTensorWrapper wrapper(JAXX_Scaling_Mode::NO_SCALING, num_gemms, dataShape);
  wrapper.set_rowwise(data, std::nullopt);
  if (first_dims.element_count() > 0) {
    NVTE_CHECK(first_dims.element_type() == xla::ffi::DataType::S32, "group_sizes must be int32.");
    NVTE_CHECK(int64_offset + num_gemms <= int64_workspace_capacity,
               "int64_workspace overflow: not enough space for first_dims conversion.");
    auto *slot = int64_workspace_base + int64_offset;
    nvte_convert_int32_to_int64(reinterpret_cast<const int32_t *>(first_dims.untyped_data()), slot,
                                num_gemms, stream);
    wrapper.set_group_sizes_only(slot, num_gemms, kNVTEGroupedFirstDims);
    int64_offset += num_gemms;
  }
  if (last_dims.element_count() > 0) {
    NVTE_CHECK(last_dims.element_type() == xla::ffi::DataType::S32, "group_sizes must be int32.");
    NVTE_CHECK(int64_offset + num_gemms <= int64_workspace_capacity,
               "int64_workspace overflow: not enough space for last_dims conversion.");
    auto *slot = int64_workspace_base + int64_offset;
    nvte_convert_int32_to_int64(reinterpret_cast<const int32_t *>(last_dims.untyped_data()), slot,
                                num_gemms, stream);
    wrapper.set_group_sizes_only(slot, num_gemms, kNVTEGroupedLastDims);
    int64_offset += num_gemms;
  }
  return wrapper;
}

// V2 variant with scaling support (MXFP8 or NO_SCALING).  Accepts scale_inv buffer and
// use_colwise flag to wire rowwise or columnwise data+scales for the grouped tensor.
// Pre-swizzled scales are indicated via set_with_gemm_swizzled_scales(true).
JAXX_GroupedTensorWrapper make_grouped_tensor(
    Buffer_Type const &data, Buffer_Type const &scale_inv, JAXX_Scaling_Mode scaling_mode,
    bool use_colwise, Buffer_Type const &first_dims, Buffer_Type const &last_dims,
    int64_t *int64_workspace_base, size_t int64_workspace_capacity, size_t &int64_offset,
    size_t num_gemms, cudaStream_t stream, size_t left_size, size_t right_size) {
  auto dims = data.dimensions();
  NVTE_CHECK(product(dims) == left_size * right_size,
             "grouped GEMM data buffer element count does not match the provided 2D shape.");
  NVTEShape dataShape{.data = {left_size, right_size}, .ndim = 2};
  JAXX_GroupedTensorWrapper wrapper(scaling_mode, num_gemms, dataShape);

  const bool is_mxfp8 = scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING;
  if (is_mxfp8 && use_colwise) {
    wrapper.set_columnwise(data, scale_inv);
  } else if (is_mxfp8) {
    wrapper.set_rowwise(data, scale_inv);
  } else {
    // NO_SCALING: no scale_inv needed
    wrapper.set_rowwise(data, std::nullopt);
  }
  if (is_mxfp8) {
    wrapper.set_with_gemm_swizzled_scales(true);
  }

  if (first_dims.element_count() > 0) {
    NVTE_CHECK(first_dims.element_type() == xla::ffi::DataType::S32, "group_sizes must be int32.");
    NVTE_CHECK(int64_offset + num_gemms <= int64_workspace_capacity,
               "int64_workspace overflow: not enough space for first_dims conversion.");
    auto *slot = int64_workspace_base + int64_offset;
    nvte_convert_int32_to_int64(reinterpret_cast<const int32_t *>(first_dims.untyped_data()), slot,
                                num_gemms, stream);
    wrapper.set_group_sizes_only(slot, num_gemms, kNVTEGroupedFirstDims);
    int64_offset += num_gemms;
  }
  if (last_dims.element_count() > 0) {
    NVTE_CHECK(last_dims.element_type() == xla::ffi::DataType::S32, "group_sizes must be int32.");
    NVTE_CHECK(int64_offset + num_gemms <= int64_workspace_capacity,
               "int64_workspace overflow: not enough space for last_dims conversion.");
    auto *slot = int64_workspace_base + int64_offset;
    nvte_convert_int32_to_int64(reinterpret_cast<const int32_t *>(last_dims.untyped_data()), slot,
                                num_gemms, stream);
    wrapper.set_group_sizes_only(slot, num_gemms, kNVTEGroupedLastDims);
    int64_offset += num_gemms;
  }
  return wrapper;
}

// Returns num_gemms from the first non-empty per-tensor group_sizes buffer,
// falling back to the element count of alpha for the uniform-batch case.
size_t grouped_gemm_num_gemms(Buffer_Type const &lhs_first_dims, Buffer_Type const &lhs_last_dims,
                              Buffer_Type const &rhs_first_dims, Buffer_Type const &rhs_last_dims,
                              Buffer_Type const &out_first_dims, Buffer_Type const &out_last_dims,
                              Buffer_Type const &alpha) {
  if (lhs_first_dims.element_count() > 0) {
    return lhs_first_dims.element_count();
  } else if (lhs_last_dims.element_count() > 0) {
    return lhs_last_dims.element_count();
  } else if (rhs_first_dims.element_count() > 0) {
    return rhs_first_dims.element_count();
  } else if (rhs_last_dims.element_count() > 0) {
    return rhs_last_dims.element_count();
  } else if (out_first_dims.element_count() > 0) {
    return out_first_dims.element_count();
  } else if (out_last_dims.element_count() > 0) {
    return out_last_dims.element_count();
  } else {
    return alpha.element_count();  // uniform batch: no ragged tensor
  }
}

/* EXPERIMENTAL FEATURE AND SUBJECT TO CHANGE. */
/*! \brief Compute estimates for average dimensions of a grouped tensor.
 *
 * Returns a pair of {non_contracting_avg, contracting_avg} dimensions for the given grouped tensor, to estimate per-group GEMM sizes. When a dimension is ragged, we estimate the average size by dividing the dim size by G ("num_gemms"). When a dimension has no ragged dims, we assume it is of shape (G*K, N) or (G*N, K) so we divide the first dim by G to get the average per-group size.
 *
 * Examples:
 *  - fwd lhs: shape_2d=[ragged M, K], first_dims=[M,...] (ragged M) → avg_m = (G*M)/G = M, avg_k = K
 *  - fwd rhs: shape_2d=[G*K, N], last_dims=None (static K) → avg_k = (G*K)/G = K, avg_n = N
 *  - wgrad lhs: shape_2d=[M, ragged K], last_dims=[K,...] (ragged K) → avg_k = (G*K)/G = K, avg_m = M
 *  - wgrad rhs: shape_2d=[N, ragged K], last_dims=[K,...] (ragged K) → avg_k = (G*K)/G = K, avg_n = N
 *
 *  \param[in]  first_dims           XLA buffer of on-device first dimensions. Shape (G,) if ragged, empty otherwise.
 *  \param[in]  last_dims            XLA buffer of on-device last dimensions. Shape (G,) if ragged, empty otherwise.
 *  \param[in]  shape_2d             Pair of total 2D dimensions (rows, cols) for the operand.
 *  \param[in]  num_gemms            Number of GEMMs (G) in the grouped operation.
 *  \param[in]  is_trans             Whether the operand is transposed.
 *  \return Pair of {non_contracting_avg, contracting_avg}, i.e. {avg_m, avg_k} for lhs or
 *         {avg_n, avg_k} for rhs.
 */
std::pair<int64_t, int64_t> grouped_gemm_avg_dims(Buffer_Type const &first_dims,
                                                  Buffer_Type const &last_dims,
                                                  std::pair<size_t, size_t> const &shape_2d,
                                                  size_t num_gemms, bool is_trans) {
  bool first_ragged = first_dims.element_count() > 0;
  bool last_ragged = last_dims.element_count() > 0;
  bool any_ragged = first_ragged || last_ragged;

  std::pair<size_t, size_t> per_group_shape_2d{};
  if (first_ragged) {
    per_group_shape_2d = {
        static_cast<size_t>(std::round(static_cast<double>(shape_2d.first) / num_gemms)),
        shape_2d.second};
  } else if (!any_ragged) {
    per_group_shape_2d = {
        static_cast<size_t>(std::round(static_cast<double>(shape_2d.first) / num_gemms)),
        shape_2d.second};
  } else if (last_ragged && !first_ragged) {
    per_group_shape_2d = {
        shape_2d.first,
        static_cast<size_t>(std::round(static_cast<double>(shape_2d.second) / num_gemms))};
  } else {
    NVTE_CHECK(false, "Grouped GEMM with both first_dims and last_dims ragged is not supported.");
  }

  int64_t non_contract =
      static_cast<int64_t>(is_trans ? per_group_shape_2d.second : per_group_shape_2d.first);
  int64_t contract =
      static_cast<int64_t>(is_trans ? per_group_shape_2d.first : per_group_shape_2d.second);
  return {non_contract, contract};
}

}  // namespace jax
}  // namespace transformer_engine

namespace transformer_engine {
namespace jax {

// This FFI is EXPERIMENTAL and subject to change without deprecation, intended for use in JAX's internal implementation of grouped GEMM.
Error_Type GroupedGemmV2FFI(cudaStream_t stream, Buffer_Type lhs_data, Buffer_Type lhs_sinv,
                            Buffer_Type rhs_data, Buffer_Type rhs_sinv, Buffer_Type bias,
                            Buffer_Type lhs_first_dims, Buffer_Type lhs_last_dims,
                            Buffer_Type rhs_first_dims, Buffer_Type rhs_last_dims,
                            Buffer_Type out_first_dims, Buffer_Type out_last_dims,
                            Buffer_Type alpha, Buffer_Type beta, Result_Type output,
                            Result_Type cublas_workspace, Result_Type setup_workspace,
                            Result_Type int64_workspace, GroupedGemmV2Config config) {
  auto [lhs_is_trans, rhs_is_trans, scaling_mode, lhs_axis_boundary, rhs_axis_boundary,
        lhs_left_size, lhs_right_size, rhs_left_size, rhs_right_size] = config;

  NVTE_CHECK(scaling_mode == JAXX_Scaling_Mode::NO_SCALING ||
                 scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING,
             "Only NO_SCALING and MXFP8_1D_SCALING are supported in the V2 grouped GEMM.");

  const bool is_mxfp8 = scaling_mode == JAXX_Scaling_Mode::MXFP8_1D_SCALING;

  size_t num_gemms = grouped_gemm_num_gemms(lhs_first_dims, lhs_last_dims, rhs_first_dims,
                                            rhs_last_dims, out_first_dims, out_last_dims, alpha);

  // Workspaces.
  // V2 GEMM receives scale_inv already swizzled by nvte_group_quantize (V2 grouped quantize
  // fuses the swizzle).  No extra sinv reservation is needed; the full cublas_workspace is
  // available for cuBLAS.
  auto setup_workspace_ptr = reinterpret_cast<uint8_t *>(setup_workspace->untyped_data());
  auto cublas_workspace_ptr = reinterpret_cast<uint8_t *>(cublas_workspace->untyped_data());
  cublas_workspace_ptr = move_ptr_to_next_256B_aligned(cublas_workspace_ptr);
  auto workspace_size = product(cublas_workspace->dimensions()) - 256;
  TensorWrapper workspace_setup(setup_workspace_ptr,
                                std::vector<size_t>{product(setup_workspace->dimensions())},
                                DType::kByte);
  TensorWrapper workspace_cublas(cublas_workspace_ptr, std::vector<size_t>{workspace_size},
                                 DType::kByte);

  TensorWrapper alpha_tensor(static_cast<void *>(alpha.untyped_data()),
                             std::vector<size_t>{num_gemms},
                             convert_ffi_datatype_to_te_dtype(alpha.element_type()));
  TensorWrapper beta_tensor(static_cast<void *>(beta.untyped_data()),
                            std::vector<size_t>{num_gemms},
                            convert_ffi_datatype_to_te_dtype(beta.element_type()));

  // Build grouped tensors from XLA buffer shapes and group_sizes — no m/n/k derivation needed.
  // int64_workspace is partitioned into per-ragged-buffer slots of num_gemms int64 elements each.
  // int64_offset is threaded through the three make_grouped_tensor calls so each non-empty *_dims
  // buffer gets its own non-aliasing slot; bounds are checked inside make_grouped_tensor.
  auto *int64_base = reinterpret_cast<int64_t *>(int64_workspace->untyped_data());
  size_t int64_capacity = int64_workspace->element_count() / sizeof(int64_t);
  size_t int64_offset = 0;

  // For MXFP8: in JAX, rhs=cuBLAS_A, lhs=cuBLAS_B (swapped).
  // Colwise is needed when the operand's contracting dim is NOT the last dim in its layout.
  const bool rhs_use_colwise = is_mxfp8 && !rhs_is_trans;
  const bool lhs_use_colwise = is_mxfp8 && lhs_is_trans;

  // For MXFP8: scale_inv is already swizzled (pre-swizzled by V2 grouped quantize via
  // nvte_group_quantize).  Pass the buffers directly to make_grouped_tensor which sets
  // with_gemm_swizzled_scales(true) for MXFP8 automatically.  No re-swizzling needed.
  auto rhs_tensor =
      is_mxfp8
          ? make_grouped_tensor(rhs_data, rhs_sinv, scaling_mode, rhs_use_colwise, rhs_first_dims,
                                rhs_last_dims, int64_base, int64_capacity, int64_offset, num_gemms,
                                stream, rhs_left_size, rhs_right_size)
          : make_grouped_tensor(rhs_data, rhs_first_dims, rhs_last_dims, int64_base, int64_capacity,
                                int64_offset, num_gemms, stream, rhs_left_size, rhs_right_size);
  auto lhs_tensor =
      is_mxfp8
          ? make_grouped_tensor(lhs_data, lhs_sinv, scaling_mode, lhs_use_colwise, lhs_first_dims,
                                lhs_last_dims, int64_base, int64_capacity, int64_offset, num_gemms,
                                stream, lhs_left_size, lhs_right_size)
          : make_grouped_tensor(lhs_data, lhs_first_dims, lhs_last_dims, int64_base, int64_capacity,
                                int64_offset, num_gemms, stream, lhs_left_size, lhs_right_size);

  // Output stays NO_SCALING. Derive 2D shape from the output buffer's own dims using
  // last-dim-as-columns convention (equivalent to axis_boundary=-1 in the old API).
  auto out_dims = output->dimensions();
  NVTE_CHECK(out_dims.size() > 0, "output buffer must have at least 1 dimension");
  size_t out_left_size = product(out_dims, 0, out_dims.size() - 1);
  size_t out_right_size = static_cast<size_t>(out_dims[out_dims.size() - 1]);
  auto out_tensor =
      make_grouped_tensor(*output, out_first_dims, out_last_dims, int64_base, int64_capacity,
                          int64_offset, num_gemms, stream, out_left_size, out_right_size);

  auto [avg_m, avg_k_lhs] = grouped_gemm_avg_dims(
      lhs_first_dims, lhs_last_dims, {lhs_left_size, lhs_right_size}, num_gemms, lhs_is_trans);
  auto [avg_n, avg_k_rhs] = grouped_gemm_avg_dims(
      rhs_first_dims, rhs_last_dims, {rhs_left_size, rhs_right_size}, num_gemms, !rhs_is_trans);
  // Use k from lhs (both sides should agree for well-formed inputs).
  NVTE_CHECK(avg_k_lhs == avg_k_rhs, "Contracting dimension mismatch: lhs avg_k=", avg_k_lhs,
             " vs rhs avg_k=", avg_k_rhs);

  GroupedMatmulConfigWrapper gemmConfig{};
  gemmConfig.set_avg_m(avg_m);
  gemmConfig.set_avg_n(avg_n);
  gemmConfig.set_avg_k(avg_k_lhs);

  nvte_grouped_gemm(rhs_tensor, rhs_is_trans, lhs_tensor, lhs_is_trans, nullptr, out_tensor,
                    alpha_tensor.data(), beta_tensor.data(), workspace_setup.data(),
                    workspace_cublas.data(), gemmConfig, stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GroupedGemmV2Handler, GroupedGemmV2FFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // lhs_data
                                  .Arg<Buffer_Type>()      // lhs_sinv
                                  .Arg<Buffer_Type>()      // rhs_data
                                  .Arg<Buffer_Type>()      // rhs_sinv
                                  .Arg<Buffer_Type>()      // bias
                                  .Arg<Buffer_Type>()      // lhs_first_dims (G,) or empty (0,)
                                  .Arg<Buffer_Type>()      // lhs_last_dims (G,) or empty (0,)
                                  .Arg<Buffer_Type>()      // rhs_first_dims (G,) or empty (0,)
                                  .Arg<Buffer_Type>()      // rhs_last_dims (G,) or empty (0,)
                                  .Arg<Buffer_Type>()      // out_first_dims (G,) or empty (0,)
                                  .Arg<Buffer_Type>()      // out_last_dims (G,) or empty (0,)
                                  .Arg<Buffer_Type>()      // alpha
                                  .Arg<Buffer_Type>()      // beta
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // cublas_workspace
                                  .Ret<Buffer_Type>()      // setup_workspace
                                  .Ret<Buffer_Type>()      // int64_workspace
                                  .Attrs<GroupedGemmV2Config>(),
                              FFI_CudaGraph_Traits);

Error_Type GroupedGemmFFI(cudaStream_t stream, Buffer_Type lhs_data, Buffer_Type lhs_sinv,
                          Buffer_Type rhs_data, Buffer_Type rhs_sinv, Buffer_Type bias,
                          Buffer_Type lhs_first_dims, Buffer_Type lhs_last_dims,
                          Buffer_Type rhs_first_dims, Buffer_Type rhs_last_dims,
                          Buffer_Type out_first_dims, Buffer_Type out_last_dims,
                          Buffer_Type group_offset, Result_Type output, Result_Type workspace,
                          GroupedGemmConfig config) {
  auto [lhs_is_trans, rhs_is_trans, scaling_mode, has_bias, use_async_d2h_group_sizes,
        lhs_axis_boundary, rhs_axis_boundary, lhs_left_size, lhs_right_size, rhs_left_size,
        rhs_right_size] = config;
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

  // Determine which group_sizes buffers are active (non-empty = ragged dimension).
  bool is_lhs_first_ragged = lhs_first_dims.element_count() > 0;
  bool is_lhs_last_ragged = lhs_last_dims.element_count() > 0;
  bool is_rhs_first_ragged = rhs_first_dims.element_count() > 0;
  bool is_rhs_last_ragged = rhs_last_dims.element_count() > 0;
  bool is_lhs_ragged = is_lhs_first_ragged || is_lhs_last_ragged;
  bool is_rhs_ragged = is_rhs_first_ragged || is_rhs_last_ragged;
  bool any_ragged = is_lhs_ragged || is_rhs_ragged;

  size_t num_gemms;
  if (is_lhs_first_ragged)
    num_gemms = lhs_first_dims.dimensions()[0];
  else if (is_lhs_last_ragged)
    num_gemms = lhs_last_dims.dimensions()[0];
  else if (is_rhs_first_ragged)
    num_gemms = rhs_first_dims.dimensions()[0];
  else if (is_rhs_last_ragged)
    num_gemms = rhs_last_dims.dimensions()[0];
  else
    NVTE_CHECK(false,
               "GroupedGemmFFI (v1): At least one of the group size buffers must be non-empty to "
               "determine num_gemms.");

  const Buffer_Type *active_gs_ptr = nullptr;
  if (is_lhs_first_ragged)
    active_gs_ptr = &lhs_first_dims;
  else if (is_lhs_last_ragged)
    active_gs_ptr = &lhs_last_dims;
  else if (is_rhs_first_ragged)
    active_gs_ptr = &rhs_first_dims;
  else if (is_rhs_last_ragged)
    active_gs_ptr = &rhs_last_dims;

  // Derive m, n, k from pre-computed original shape sizes (passed from Python).
  // lhs_left_size = product of original lhs dims before axis_boundary
  // lhs_right_size = product of original lhs dims after axis_boundary
  // Same pattern for rhs.
  size_t k = lhs_is_trans ? lhs_left_size : lhs_right_size;
  size_t m, n;
  if (is_rhs_ragged) {
    // wgrad: non-contracting lhs dims form M; non-contracting rhs dims form N
    m = lhs_is_trans ? lhs_right_size : lhs_left_size;
    n = rhs_is_trans ? rhs_left_size : rhs_right_size;
  } else {
    m = lhs_is_trans ? lhs_right_size : lhs_left_size;  // total M (sum of group sizes)
    n = rhs_is_trans ? rhs_left_size / num_gemms : rhs_right_size;
  }

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
  if (is_tensor_scaling) {
    // For tensor scaling, each matrix has a single scale value, and all scales need to be aligned
    // by 16 bytes to meet the requirement of CUDA 12.9.1 and later.
    workspace_size -= tensor_scaling_sinv_aligment * (lhs_sinv_size + rhs_sinv_size);
  }
  workspace_size = workspace_size / num_streams;
  auto lhs_scatter_aligned_ptr = workspace_ptr + workspace_size * num_streams;
  lhs_scatter_aligned_ptr = move_ptr_to_next_256B_aligned(lhs_scatter_aligned_ptr);
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
  size_t expected_rhs_size = is_rhs_ragged ? (k * n) : (num_gemms * k * n);
  size_t expected_out_size = is_rhs_ragged ? (num_gemms * m * n) : (m * n);
  size_t actual_lhs_size = product(lhs_data.dimensions());
  size_t actual_rhs_size = product(rhs_data.dimensions());
  size_t actual_out_size = product(output->dimensions());
  NVTE_CHECK(expected_lhs_size == actual_lhs_size, "Unexpected lhs size! Expect ",
             expected_lhs_size, ", got ", actual_lhs_size);
  if (!is_rhs_ragged) {
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
  if (any_ragged) {
    size_t host_num_gemms = 0;
    if (use_async_d2h_group_sizes) {
      host_num_gemms = GroupedGemmGetGroupSizes(stream, num_gemms, nullptr, dim_list_host.data());
      NVTE_CHECK(host_num_gemms == num_gemms, "num_gemms ", num_gemms,
                 " does not match the return of GroupedGemmGetGroupSizes ", host_num_gemms, ".");
    } else {
      NVTE_CHECK(active_gs_ptr != nullptr, "active_gs_ptr is null but any_ragged is true.");
      auto gs_data_ptr = reinterpret_cast<const int32_t *>(active_gs_ptr->untyped_data());
      cudaMemcpyAsync(dim_list_host.data(), gs_data_ptr, dim_list_bytes, cudaMemcpyDeviceToHost,
                      stream);
      // Note: This may break cudaGraph.
      cudaStreamSynchronize(stream);
    }
    size_t sum_group_sizes = std::accumulate(dim_list_host.begin(), dim_list_host.end(), 0);
    if (!is_rhs_ragged) {
      NVTE_CHECK(m == sum_group_sizes, "Unexpected group_sizes! M = ", m,
                 ", got sum(group_sizes)=", sum_group_sizes);
    } else {
      NVTE_CHECK(k == sum_group_sizes, "Unexpected group_sizes! K = ", k,
                 ", got sum(group_sizes)=", sum_group_sizes);
    }
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
    if (is_rhs_ragged) {
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
      // MXFP8 scales are pre-swizzled by the quantize kernel (both V1 and V2),
      // so we pass them directly to the GEMM without a separate swizzle pass.
      // Note: even if is_empty_gemm is true, sinv are still non-empty, need to move the pointers
      auto lhs_sinv_shape_i =
          get_block_scale_shape(scaling_mode, lhs_shape_i[0], lhs_shape_i[1], lhs_use_colwise);
      auto rhs_sinv_shape_i =
          get_block_scale_shape(scaling_mode, rhs_shape_i[0], rhs_shape_i[1], rhs_use_colwise);
      lhs_sinv_size_i = lhs_sinv_shape_i[0] * lhs_sinv_shape_i[1];
      rhs_sinv_size_i = rhs_sinv_shape_i[0] * rhs_sinv_shape_i[1];
      if (lhs_use_colwise) {
        lhs_i.set_columnwise_scale_inv(lhs_sinv_vptr, lhs_sinv_dtype, lhs_sinv_shape_i);
      } else {
        lhs_i.set_rowwise_scale_inv(lhs_sinv_vptr, lhs_sinv_dtype, lhs_sinv_shape_i);
      }
      lhs_i.set_with_gemm_swizzled_scales(true);
      if (rhs_use_colwise) {
        rhs_i.set_columnwise_scale_inv(rhs_sinv_vptr, rhs_sinv_dtype, rhs_sinv_shape_i);
      } else {
        rhs_i.set_rowwise_scale_inv(rhs_sinv_vptr, rhs_sinv_dtype, rhs_sinv_shape_i);
      }
      rhs_i.set_with_gemm_swizzled_scales(true);
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
                                  .Arg<Buffer_Type>()      // lhs_first_dims (G,) or empty (0,)
                                  .Arg<Buffer_Type>()      // lhs_last_dims (G,) or empty (0,)
                                  .Arg<Buffer_Type>()      // rhs_first_dims (G,) or empty (0,)
                                  .Arg<Buffer_Type>()      // rhs_last_dims (G,) or empty (0,)
                                  .Arg<Buffer_Type>()      // out_first_dims (G,) or empty (0,)
                                  .Arg<Buffer_Type>()      // out_last_dims (G,) or empty (0,)
                                  .Arg<Buffer_Type>()      // group_offset
                                  .Ret<Buffer_Type>()      // output
                                  .Ret<Buffer_Type>()      // workspace
                                  .Attrs<GroupedGemmConfig>());

}  // namespace jax
}  // namespace transformer_engine

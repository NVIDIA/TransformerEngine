/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/fused_router.h>

#include <cstdint>
#include <vector>

#include "../extensions.h"

namespace transformer_engine {
namespace jax {

// Score function mapping: sigmoid=0, softmax=1
constexpr int kScoreFunctionSigmoid = 0;
constexpr int kScoreFunctionSoftmax = 1;

Error_Type FusedTopkWithScoreFunctionForwardFFI(
    cudaStream_t stream, Buffer_Type logits_buf, Buffer_Type expert_bias_buf,
    Result_Type probs_buf, Result_Type routing_map_buf, Result_Type intermediate_output_buf,
    int64_t num_tokens, int64_t num_experts, int64_t topk, bool use_pre_softmax,
    int64_t num_groups, int64_t group_topk, double scaling_factor, int64_t score_function) {

  auto logits_dtype = convert_ffi_datatype_to_te_dtype(logits_buf.element_type());
  auto logits_shape = std::vector<size_t>{static_cast<size_t>(num_tokens),
                                          static_cast<size_t>(num_experts)};

  auto *logits = logits_buf.untyped_data();
  auto logits_tensor = TensorWrapper(logits, logits_shape, logits_dtype);

  auto expert_bias_tensor = TensorWrapper();
  bool has_expert_bias = expert_bias_buf.element_count() > 0;
  if (has_expert_bias) {
    auto expert_bias_dtype = convert_ffi_datatype_to_te_dtype(expert_bias_buf.element_type());
    auto *expert_bias = expert_bias_buf.untyped_data();
    auto expert_bias_shape = std::vector<size_t>{static_cast<size_t>(num_experts)};
    expert_bias_tensor = TensorWrapper(expert_bias, expert_bias_shape, expert_bias_dtype);
  }

  auto *probs = probs_buf->untyped_data();
  auto probs_tensor = TensorWrapper(probs, logits_shape, logits_dtype);

  auto *routing_map = routing_map_buf->untyped_data();
  auto routing_map_tensor = TensorWrapper(routing_map, logits_shape, DType::kByte);

  auto *intermediate_output = intermediate_output_buf->untyped_data();
  auto intermediate_output_tensor =
      TensorWrapper(intermediate_output, logits_shape, logits_dtype);

  nvte_fused_topk_with_score_function_forward(
      logits_tensor.data(), static_cast<int>(num_tokens), static_cast<int>(num_experts),
      static_cast<int>(topk), static_cast<int>(use_pre_softmax), static_cast<int>(num_groups),
      static_cast<int>(group_topk), static_cast<float>(scaling_factor),
      static_cast<int>(score_function),
      expert_bias_tensor.data(), probs_tensor.data(),
      routing_map_tensor.data(), intermediate_output_tensor.data(), stream);

  return ffi_with_cuda_error_check();
}

Error_Type FusedTopkWithScoreFunctionBackwardFFI(
    cudaStream_t stream, Buffer_Type routing_map_buf, Buffer_Type intermediate_output_buf,
    Buffer_Type grad_probs_buf, Result_Type grad_logits_buf, int64_t num_tokens,
    int64_t num_experts, int64_t topk, bool use_pre_softmax, double scaling_factor,
    int64_t score_function) {
  auto grad_probs_dtype = convert_ffi_datatype_to_te_dtype(grad_probs_buf.element_type());
  auto tensor_shape = std::vector<size_t>{static_cast<size_t>(num_tokens),
                                          static_cast<size_t>(num_experts)};

  auto *routing_map = routing_map_buf.untyped_data();
  auto routing_map_tensor = TensorWrapper(routing_map, tensor_shape, DType::kByte);

  auto *intermediate_output = intermediate_output_buf.untyped_data();
  auto intermediate_output_tensor =
      TensorWrapper(intermediate_output, tensor_shape, grad_probs_dtype);

  auto *grad_probs = grad_probs_buf.untyped_data();
  auto grad_probs_tensor = TensorWrapper(grad_probs, tensor_shape, grad_probs_dtype);

  auto *grad_logits = grad_logits_buf->untyped_data();
  auto grad_logits_tensor = TensorWrapper(grad_logits, tensor_shape, grad_probs_dtype);

  nvte_fused_topk_with_score_function_backward(
      routing_map_tensor.data(), intermediate_output_tensor.data(), grad_probs_tensor.data(),
      static_cast<int>(num_tokens), static_cast<int>(num_experts), static_cast<int>(topk),
      static_cast<int>(use_pre_softmax), static_cast<float>(scaling_factor),
      static_cast<int>(score_function), grad_logits_tensor.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FusedTopkWithScoreFunctionForwardHandler,
                              FusedTopkWithScoreFunctionForwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // logits
                                  .Arg<Buffer_Type>()      // expert_bias (optional)
                                  .Ret<Buffer_Type>()      // probs
                                  .Ret<Buffer_Type>()      // routing_map
                                  .Ret<Buffer_Type>()      // intermediate_output
                                  .Attr<int64_t>("num_tokens")
                                  .Attr<int64_t>("num_experts")
                                  .Attr<int64_t>("topk")
                                  .Attr<bool>("use_pre_softmax")
                                  .Attr<int64_t>("num_groups")
                                  .Attr<int64_t>("group_topk")
                                  .Attr<double>("scaling_factor")
                                  .Attr<int64_t>("score_function"),
                              FFI_CudaGraph_Traits);

XLA_FFI_DEFINE_HANDLER_SYMBOL(FusedTopkWithScoreFunctionBackwardHandler,
                              FusedTopkWithScoreFunctionBackwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // routing_map
                                  .Arg<Buffer_Type>()      // intermediate_output
                                  .Arg<Buffer_Type>()      // grad_probs
                                  .Ret<Buffer_Type>()      // grad_logits
                                  .Attr<int64_t>("num_tokens")
                                  .Attr<int64_t>("num_experts")
                                  .Attr<int64_t>("topk")
                                  .Attr<bool>("use_pre_softmax")
                                  .Attr<double>("scaling_factor")
                                  .Attr<int64_t>("score_function"),
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine

